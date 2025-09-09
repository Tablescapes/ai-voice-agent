#!/usr/bin/env python3
"""
Self-Sufficient AI Voice Agent System – Final Production Version
- Async-safe (aiosqlite + threadpools for blocking ops)
- Truthful audio MIME (gTTS -> MP3)
- Buffered WS audio with size caps and timeouts
- Circuit breakers + retries for OpenAI calls
- Monotonic, thread-safe rate limiting + global connection cap
- Atomic DB updates and correct UTC date-range stats
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import os
import logging
import asyncio
import base64
import tempfile
import subprocess
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from uuid import uuid4
from pathlib import Path
from collections import defaultdict

import aiosqlite
from starlette.concurrency import run_in_threadpool

# Retries (sync callables)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Cloud LLM/STT/TTS (sync clients)
from openai import OpenAI
from gtts import gTTS


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s"
)
logger = logging.getLogger("voice_agent")


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    
    # App
    BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
    DATABASE_PATH = os.getenv("DATABASE_PATH", "voice_agent.db")

    # Audio
    SAMPLE_RATE = 16000
    STT_TARGET_RATE = 16000
    AUDIO_TTS_MIME = "audio/mpeg"  # gTTS -> MP3
    MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE", "5242880"))       # 5 MB per frame
    MAX_COMBINED_AUDIO = int(os.getenv("MAX_COMBINED_AUDIO", "15728640"))  # 15 MB per batch
    AUDIO_BUFFER_SIZE = int(os.getenv("AUDIO_BUFFER_SIZE", "3"))       # frames per STT batch

    # KB / matching
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

    # Call control
    MAX_CALL_DURATION = int(os.getenv("MAX_CALL_DURATION", "600"))     # seconds
    WEBSOCKET_TIMEOUT = int(os.getenv("WEBSOCKET_TIMEOUT", "10"))      # seconds

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # Rate limiting / connections
    MAX_CONCURRENT_CONNECTIONS = int(os.getenv("MAX_CONCURRENT_CONNECTIONS", "10"))  # global cap
    RATE_LIMIT_CONNECTIONS = int(os.getenv("RATE_LIMIT_CONNECTIONS", "3"))           # per-IP/window
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "300"))                   # seconds

    # Circuit breaker
    CB_FAIL_THRESHOLD = int(os.getenv("CB_FAIL_THRESHOLD", "3"))
    CB_RESET_TIMEOUT = int(os.getenv("CB_RESET_TIMEOUT", "30"))
    CB_HALF_OPEN_LIMIT = int(os.getenv("CB_HALF_OPEN_LIMIT", "2"))

    # Misc
    MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "4000"))


# ------------------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------------------
@dataclass
class Call:
    call_id: str
    caller_info: str
    start_time: datetime
    end_time: Optional[datetime] = None
    transcript: str = ""
    outcome: str = ""     # answered, transferred, failed
    confidence_score: float = 0.0
    audio_file: Optional[str] = None


@dataclass
class KnowledgeItem:
    id: str
    question: str
    answer: str
    source: str
    embedding: Optional[List[float]] = None
    last_updated: Optional[datetime] = None
    usage_count: int = 0


@dataclass
class UnknownQuestion:
    id: str
    question: str
    call_id: str
    timestamp: datetime
    resolved: bool = False
    suggested_answer: Optional[str] = None


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def today_utc_range() -> Tuple[str, str]:
    start = datetime.now(timezone.utc).date()
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def limit_len(s: str, maxlen: int) -> str:
    return s if len(s) <= maxlen else s[:maxlen]


def ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------
# Connection tracking + rate limiting (thread-safe)
# ------------------------------------------------------------------------------
class ConnectionTracker:
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.active = set()
        self.lock = threading.Lock()

    def add(self, conn_id: str) -> bool:
        with self.lock:
            if len(self.active) >= self.max_connections:
                return False
            self.active.add(conn_id)
            return True

    def remove(self, conn_id: str):
        with self.lock:
            self.active.discard(conn_id)

    def count(self) -> int:
        with self.lock:
            return len(self.active)


connection_tracker = ConnectionTracker(Config.MAX_CONCURRENT_CONNECTIONS)


class RateLimiter:
    """Per-client token-like limiter: max 'connections' within window."""
    def __init__(self, max_connections: int = 3, window_seconds: int = 300, max_clients: int = 1000):
        self.max_connections = max_connections
        self.window = window_seconds
        self.max_clients = max_clients
        self.connections = defaultdict(list)   # client_id -> [timestamps]
        self.lock = threading.Lock()
        self.last_cleanup = time.monotonic()
        self.cleanup_interval = 300

    def is_allowed(self, client_id: str) -> bool:
        now = time.monotonic()
        with self.lock:
            if now - self.last_cleanup > self.cleanup_interval:
                self._cleanup(now)
                self.last_cleanup = now

            win_start = now - self.window
            lst = self.connections[client_id]
            self.connections[client_id] = [t for t in lst if t > win_start]
            lst = self.connections[client_id]

            if len(lst) < self.max_connections:
                lst.append(now)
                return True
            return False

    def _cleanup(self, now: float):
        cutoff = now - self.window * 2
        to_del = [cid for cid, ts in self.connections.items() if not ts or ts[-1] < cutoff]
        for cid in to_del:
            del self.connections[cid]
        if len(self.connections) > self.max_clients:
            # drop oldest
            items = sorted(self.connections.items(), key=lambda kv: kv[1][-1] if kv[1] else 0)
            for cid, _ in items[:len(self.connections) - self.max_clients]:
                del self.connections[cid]

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "active_clients": len(self.connections),
                "total_marks": sum(len(v) for v in self.connections.values()),
                "window_seconds": self.window,
                "max_connections_per_window": self.max_connections
            }


rate_limiter = RateLimiter(Config.RATE_LIMIT_CONNECTIONS, Config.RATE_LIMIT_WINDOW)


# ------------------------------------------------------------------------------
# Circuit Breaker
# ------------------------------------------------------------------------------
class CircuitBreaker:
    def __init__(self, fail_threshold: int = 3, reset_timeout: int = 30, half_open_limit: int = 2):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.half_open_limit = half_open_limit

        self.state = "closed"  # closed, open, half-open
        self.failure_count = 0
        self.success_count = 0
        self.opened_at: Optional[float] = None
        self.half_open_calls = 0
        self.lock = threading.Lock()

    def can_call(self) -> bool:
        with self.lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if self.opened_at is None:
                    return False
                if time.monotonic() - self.opened_at >= self.reset_timeout:
                    self.state = "half-open"
                    self.half_open_calls = 0
                    self.success_count = 0
                    logger.info("Circuit -> half-open")
                    return True
                return False
            if self.state == "half-open":
                if self.half_open_calls < self.half_open_limit:
                    self.half_open_calls += 1
                    return True
                return False
            return False

    def record_success(self):
        with self.lock:
            if self.state == "half-open":
                self.success_count += 1
                if self.success_count >= 2:
                    self.state = "closed"
                    self.failure_count = 0
                    self.opened_at = None
                    self.half_open_calls = 0
                    self.success_count = 0
                    logger.info("Circuit -> closed (recovered)")
            elif self.state == "closed":
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        with self.lock:
            if self.state == "half-open":
                self.state = "open"
                self.opened_at = time.monotonic()
                self.half_open_calls = 0
                self.success_count = 0
                logger.warning("Circuit half-open call failed -> open")
            else:
                self.failure_count += 1
                if self.failure_count >= self.fail_threshold:
                    self.state = "open"
                    self.opened_at = time.monotonic()
                    logger.warning("Circuit opened")

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "half_open_calls": self.half_open_calls
            }


# ------------------------------------------------------------------------------
# AI Engine
# ------------------------------------------------------------------------------
class LightweightAIEngine:
    def __init__(self):
        self.openai_client: Optional[OpenAI] = None
        if Config.OPENAI_API_KEY:
            self.openai_client = OpenAI(
                api_key=Config.OPENAI_API_KEY,
                timeout=30.0,
                max_retries=0  # tenacity handles retries
            )
            logger.info("OpenAI client initialized.")
        else:
            logger.info("No OPENAI_API_KEY set. Falling back to rule-based responses.")

        self.cb_stt = CircuitBreaker(Config.CB_FAIL_THRESHOLD, Config.CB_RESET_TIMEOUT, Config.CB_HALF_OPEN_LIMIT)
        self.cb_chat = CircuitBreaker(Config.CB_FAIL_THRESHOLD, Config.CB_RESET_TIMEOUT, Config.CB_HALF_OPEN_LIMIT)

    # Retried sync callables
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10),
           retry=retry_if_exception_type(Exception), reraise=True)
    def _sync_openai_stt(self, file_obj, model: str):
        return self.openai_client.audio.transcriptions.create(model=model, file=file_obj)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10),
           retry=retry_if_exception_type(Exception), reraise=True)
    def _sync_openai_chat(self, messages, model, temperature, max_tokens):
        return self.openai_client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    async def convert_audio_to(self, data: bytes, target_sample_rate: int = 16000) -> Tuple[bytes, str]:
        """Convert to 16kHz mono WAV using ffmpeg. Do not truncate container bytes."""
        if len(data) > Config.MAX_COMBINED_AUDIO:
            raise ValueError("audio_combination_too_large")

        async def _run():
            with tempfile.TemporaryDirectory() as td:
                in_path = Path(td) / "in.webm"
                out_path = Path(td) / "out.wav"
                try:
                    in_path.write_bytes(data)
                    subprocess.run(
                        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                         "-i", str(in_path), "-ar", str(target_sample_rate), "-ac", "1",
                         "-f", "wav", str(out_path), "-y"],
                        check=True, timeout=30
                    )
                    return out_path.read_bytes(), "wav"
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    logger.warning(f"Audio conversion failed: {e}")
                    return data, "webm"
        return await run_in_threadpool(_run)

    async def speech_to_text(self, audio_bytes: bytes) -> str:
        """Batch STT with circuit breaker and retries."""
        if not self.openai_client:
            return ""
        if not self.cb_stt.can_call():
            logger.warning("STT circuit open; skipping call.")
            return ""
        if len(audio_bytes) > Config.MAX_COMBINED_AUDIO:
            logger.warning("Audio too large for STT batch.")
            return ""

        try:
            wav_bytes, fmt = await self.convert_audio_to(audio_bytes, Config.STT_TARGET_RATE)
        except ValueError:
            return ""
        suffix = ".wav" if fmt == "wav" else ".webm"

        def _transcribe_sync(path: Path, model: str) -> str:
            with path.open("rb") as f:
                resp = self._sync_openai_stt(f, model)
                return (getattr(resp, "text", "") or "").strip()

        async def _transcribe():
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / f"audio{suffix}"
                p.write_bytes(wav_bytes)
                try:
                    text = await run_in_threadpool(_transcribe_sync, p, Config.OPENAI_STT_MODEL)
                    self.cb_stt.record_success()
                    return text
                except Exception as e:
                    self.cb_stt.record_failure()
                    logger.error(f"STT error after retries: {e}")
                    return ""
        return await _transcribe()

    def _tts_sync(self, text: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "tts.mp3"
            gTTS(text=text, lang="en", slow=False).save(str(out))
            return out.read_bytes()

    async def text_to_speech(self, text: str) -> bytes:
        """TTS via gTTS (sync) executed in a threadpool, with timeout."""
        text = limit_len(text, Config.MAX_INPUT_LEN)
        try:
            return await asyncio.wait_for(
                run_in_threadpool(self._tts_sync, text),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.error("TTS timeout")
            return b""
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""

    async def generate_response(self, user_input: str, context: str = "") -> str:
        """LLM response or rule-based fallback, guarded by circuit breaker."""
        user_input = limit_len(user_input, Config.MAX_INPUT_LEN)

        if self.openai_client and self.cb_chat.can_call():
            try:
                resp = await run_in_threadpool(
                    self._sync_openai_chat,
                    [
                        {"role": "system",
                         "content": f"You are a concise customer service agent for a tableware and event-rental business. Keep answers accurate and succinct. Context: {context}"},
                        {"role": "user", "content": user_input}
                    ],
                    Config.OPENAI_CHAT_MODEL,
                    0.2,
                    220
                )
                text = (resp.choices[0].message.content or "").strip()
                self.cb_chat.record_success()
                if text:
                    return text
            except Exception as e:
                self.cb_chat.record_failure()
                logger.error(f"Chat error after retries: {e}")

        return self._rule_based_response(user_input)

    def _rule_based_response(self, user_input: str) -> str:
        u = (user_input or "").lower()
        if any(w in u for w in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return "Hello! How can I help you today?"
        if any(w in u for w in ["help", "support", "question"]):
            return "Sure—what do you need help with?"
        if any(w in u for w in ["price", "cost", "pricing", "how much"]):
            return "Which item do you want pricing for?"
        if any(w in u for w in ["hours", "open", "closed"]):
            return "Tell me which location and I’ll check hours."
        if any(w in u for w in ["shipping", "delivery", "ship"]):
            return "What would you like to know about delivery options?"
        if any(w in u for w in ["return", "exchange", "refund"]):
            return "Which item/order are you asking about?"
        return "I can help with that. Could you clarify the item or topic?"

    def get_embedding(self, text: str) -> List[float]:
        words = text.lower().split()
        features = [
            float(len(words)),
            float(sum(len(w) for w in words) / max(1, len(words))),
            float(sum(1 for w in words if any(c.isdigit() for c in w))),
            float(sum(1 for w in words if ('?' in w or '!' in w))),
        ]
        while len(features) < 10:
            features.append(0.0)
        return features[:10]

    def breakers_status(self) -> Dict[str, Any]:
        return {"stt": self.cb_stt.status(), "chat": self.cb_chat.status()}


# ------------------------------------------------------------------------------
# Database (aiosqlite)
# ------------------------------------------------------------------------------
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_database(self):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("PRAGMA busy_timeout=5000;")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    caller_info TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    transcript TEXT,
                    outcome TEXT,
                    confidence_score REAL,
                    audio_file TEXT
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id TEXT PRIMARY KEY,
                    question TEXT,
                    answer TEXT,
                    source TEXT,
                    embedding BLOB,
                    last_updated TEXT,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS unknown_questions (
                    id TEXT PRIMARY KEY,
                    question TEXT,
                    call_id TEXT,
                    timestamp TEXT,
                    resolved INTEGER DEFAULT 0,
                    suggested_answer TEXT
                )
            """)
            await conn.commit()

    async def save_call(self, call: Call):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO calls
                (call_id, caller_info, start_time, end_time, transcript, outcome, confidence_score, audio_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call.call_id, call.caller_info,
                call.start_time.isoformat() if call.start_time else None,
                call.end_time.isoformat() if call.end_time else None,
                call.transcript, call.outcome, call.confidence_score, call.audio_file
            ))
            await conn.commit()

    async def save_knowledge_item(self, item: KnowledgeItem):
        import pickle
        blob = pickle.dumps(item.embedding) if item.embedding else None
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO knowledge_base
                (id, question, answer, source, embedding, last_updated, usage_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id, item.question, item.answer, item.source,
                blob,
                item.last_updated.isoformat() if item.last_updated else None,
                item.usage_count
            ))
            await conn.commit()

    async def increment_usage_count(self, item_id: str):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("UPDATE knowledge_base SET usage_count = usage_count + 1 WHERE id = ?", (item_id,))
            await conn.commit()

    async def save_unknown_question(self, q: UnknownQuestion):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO unknown_questions
                (id, question, call_id, timestamp, resolved, suggested_answer)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                q.id, q.question, q.call_id, q.timestamp.isoformat(), int(q.resolved), q.suggested_answer
            ))
            await conn.commit()

    async def get_knowledge_base(self) -> List[KnowledgeItem]:
        import pickle
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM knowledge_base") as cur:
                rows = await cur.fetchall()
        items: List[KnowledgeItem] = []
        for r in rows:
            emb = pickle.loads(r["embedding"]) if r["embedding"] else None
            items.append(
                KnowledgeItem(
                    id=r["id"], question=r["question"], answer=r["answer"], source=r["source"],
                    embedding=emb,
                    last_updated=datetime.fromisoformat(r["last_updated"]) if r["last_updated"] else None,
                    usage_count=r["usage_count"] or 0
                )
            )
        return items

    async def get_unknown_questions(self, resolved: bool = False) -> List[UnknownQuestion]:
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(
                "SELECT * FROM unknown_questions WHERE resolved = ? ORDER BY timestamp DESC", (int(resolved),)
            ) as cur:
                rows = await cur.fetchall()
        out: List[UnknownQuestion] = []
        for r in rows:
            out.append(
                UnknownQuestion(
                    id=r["id"], question=r["question"], call_id=r["call_id"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    resolved=bool(r["resolved"]),
                    suggested_answer=r["suggested_answer"]
                )
            )
        return out

    async def get_call_statistics(self) -> Dict[str, Any]:
        start_iso, end_iso = today_utc_range()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("""
                SELECT COUNT(*) AS total_calls,
                       AVG(confidence_score) AS avg_confidence,
                       SUM(CASE WHEN outcome='answered' THEN 1 ELSE 0 END) AS answered_calls,
                       SUM(CASE WHEN outcome='transferred' THEN 1 ELSE 0 END) AS transferred_calls
                FROM calls
                WHERE start_time >= ? AND start_time < ?
            """, (start_iso, end_iso)) as cur:
                row = await cur.fetchone()

            async with conn.execute("""
                SELECT COUNT(*) AS c FROM unknown_questions
                WHERE timestamp >= ? AND timestamp < ? AND resolved = 0
            """, (start_iso, end_iso)) as cur2:
                row2 = await cur2.fetchone()

        return {
            "total_calls": row["total_calls"] or 0,
            "avg_confidence": row["avg_confidence"] or 0.0,
            "answered_calls": row["answered_calls"] or 0,
            "transferred_calls": row["transferred_calls"] or 0,
            "unknown_questions": row2["c"] or 0
        }


# ------------------------------------------------------------------------------
# Knowledge Manager
# ------------------------------------------------------------------------------
class KnowledgeBaseManager:
    def __init__(self, ai_engine: LightweightAIEngine, db_manager: DatabaseManager):
        self.ai = ai_engine
        self.db = db_manager
        self.knowledge_items: List[KnowledgeItem] = []

    async def load(self):
        self.knowledge_items = await self.db.get_knowledge_base()
        logger.info(f"Loaded {len(self.knowledge_items)} knowledge items.")

    async def add_knowledge_item(self, question: str, answer: str, source: str = "manual"):
        k_id = uuid4().hex
        emb = self.ai.get_embedding(question)
        item = KnowledgeItem(
            id=k_id,
            question=limit_len(question, Config.MAX_INPUT_LEN),
            answer=limit_len(answer, Config.MAX_INPUT_LEN),
            source=source,
            embedding=emb,
            last_updated=now_utc()
        )
        self.knowledge_items.append(item)
        await self.db.save_knowledge_item(item)

    async def find_answer(self, question: str) -> Tuple[str, float, Optional[str]]:
        if not self.knowledge_items:
            return "", 0.0, None
        q_words = set(question.lower().split())
        best: Tuple[str, float, Optional[str]] = ("", 0.0, None)

        for it in self.knowledge_items:
            it_words = set((it.question or "").lower().split())
            if not it_words:
                continue
            inter = q_words.intersection(it_words)
            union = q_words.union(it_words)
            sim = len(inter) / len(union) if union else 0.0
            if sim > best[1] and sim >= Config.SIMILARITY_THRESHOLD:
                best = (it.answer, sim, it.id)

        if best[2]:
            await self.db.increment_usage_count(best[2])
            # Database is source of truth - no local update needed
        
                return best


# ------------------------------------------------------------------------------
# App wiring
# ------------------------------------------------------------------------------
db_manager = DatabaseManager(Config.DATABASE_PATH)
ai_engine = LightweightAIEngine()
knowledge_manager = KnowledgeBaseManager(ai_engine, db_manager)

app = FastAPI(title="AI Voice Agent System - Final Production", version="1.5.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    await db_manager.init_database()
    await knowledge_manager.load()
    if not ffmpeg_available():
        logger.warning("ffmpeg not found; STT will fall back to WEBM input for transcription.")
    logger.info("Startup complete.")


# ------------------------------------------------------------------------------
# WebSocket voice endpoint
# ------------------------------------------------------------------------------
@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    call_id = f"call_{uuid4().hex}"

    # Global capacity cap
    if not connection_tracker.add(call_id):
        await websocket.send_json({"type": "error", "message": "Server at capacity. Try again later."})
        await websocket.close(code=1013)
        return

    call_created = False
    try:
        call = Call(call_id=call_id, caller_info="websocket", start_time=now_utc())
        call_created = True
        logger.info(f"Call started: {call_id} (active={connection_tracker.count()})")

        start_ts = time.monotonic()

        # Per-IP limiter
        client_ip = websocket.client.host if websocket.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            await websocket.send_json({"type": "error", "message": "Rate limit exceeded for your IP."})
            await websocket.close(code=1008)
            return

        audio_buffer: List[bytes] = []
        buffer_lock = asyncio.Lock()
        last_process_time = time.monotonic()

        while True:
            # Duration limit
            if time.monotonic() - start_ts > Config.MAX_CALL_DURATION:
                await websocket.send_json({"type": "closing", "reason": "max_duration"})
                await websocket.close(code=1000)
                call.outcome = call.outcome or "answered"
                break

            # Receive with timeout
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=Config.WEBSOCKET_TIMEOUT)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
                continue
            except Exception as e:
                logger.warning(f"Receive error ({call_id}): {e}")
                break

            if not data:
                await websocket.send_json({"type": "noop"})
                continue

            if len(data) > Config.MAX_AUDIO_SIZE:
                await websocket.send_json({"type": "error", "message": "audio_frame_too_large"})
                continue

            # Buffer frames safely
            async with buffer_lock:
                audio_buffer.append(data)
                should_process = (
                    len(audio_buffer) >= Config.AUDIO_BUFFER_SIZE or
                    (time.monotonic() - last_process_time) > 2.0
                )
                if should_process:
                    batch = b"".join(audio_buffer)
                    audio_buffer.clear()
                    last_process_time = time.monotonic()
                else:
                    batch = None

            if not batch:
                continue

            # Combined size check (prevents truncation in converter)
            if len(batch) > Config.MAX_COMBINED_AUDIO:
                await websocket.send_json({"type": "error", "message": "audio_combination_too_large"})
                continue

            # STT
            text = await ai_engine.speech_to_text(batch)
            if not text:
                await websocket.send_json({"type": "listening"})
                continue

            call.transcript += f"User: {text}\n"

            # KB lookup
            answer, confidence, item_id = await knowledge_manager.find_answer(text)

            if confidence >= Config.CONFIDENCE_THRESHOLD:
                response_text = answer
                call.outcome = call.outcome or "answered"
            else:
                # human escalation?
                if any(k in text.lower() for k in ["human", "person", "agent", "representative", "manager", "supervisor", "help me", "speak to someone"]):
                    await websocket.send_json({"type": "transfer", "message": "Transferring you to a human agent."})
                    await websocket.send_json({"type": "closing", "reason": "transfer"})
                    await websocket.close(code=1000)
                    call.outcome = "transferred"
                    break

                # LLM response
                response_text = await ai_engine.generate_response(text)

                # Record unknown for later curation
                uq = UnknownQuestion(
                    id=f"unknown_{uuid4().hex}",
                    question=text,
                    call_id=call_id,
                    timestamp=now_utc(),
                    suggested_answer=response_text
                )
                await db_manager.save_unknown_question(uq)

            call.transcript += f"Agent: {response_text}\n"
            call.confidence_score = confidence

            # TTS
            audio_bytes = await ai_engine.text_to_speech(response_text)

            await websocket.send_json({
                "type": "response",
                "text": response_text,
                "audio_b64": base64.b64encode(audio_bytes).decode() if audio_bytes else "",
                "audio_mime": Config.AUDIO_TTS_MIME,
                "confidence": confidence
            })

    except WebSocketDisconnect:
        logger.info(f"Call disconnected: {call_id}")
    except Exception as e:
        logger.error(f"WebSocket error ({call_id}): {e}")
        try:
            await websocket.send_json({"type": "error", "message": "internal_error"})
        except Exception:
            pass
    finally:
        connection_tracker.remove(call_id)
        if call_created:
            try:
                call.end_time = now_utc()
                await db_manager.save_call(call)
            except Exception as e:
                logger.error(f"Failed to persist call {call_id}: {e}")
        logger.info(f"Call ended: {call_id} (active={connection_tracker.count()})")


# ------------------------------------------------------------------------------
# Minimal health/stats endpoint
# ------------------------------------------------------------------------------
@app.get("/health")
async def health():
    stats = await db_manager.get_call_statistics()
    return {
        "status": "healthy",
        "time": now_utc().isoformat(),
        "active_connections": connection_tracker.count(),
        "rate_limiter": rate_limiter.stats(),
        "circuit_breakers": ai_engine.breakers_status(),
        "stats_today": stats
    }


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting AI Voice Agent System (Final Production Version)")
    logger.info(f"Dashboard: {Config.BASE_URL}")
    logger.info(f"WS:       {Config.BASE_URL.replace('http', 'ws')}/ws/voice")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
