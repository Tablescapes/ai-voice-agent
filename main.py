#!/usr/bin/env python3
"""
AI Voice Agent – Final Production + Full UI (known-good)
- Async-safe (aiosqlite + threadpools for blocking ops)
- Truthful audio MIME (gTTS -> MP3)
- Buffered WS audio with size caps and timeouts
- Circuit breakers + retries for OpenAI calls
- Monotonic, thread-safe rate limiting + global connection cap
- Atomic DB updates and correct UTC date-range stats
- Two UIs:
  - "/"      : Voice UI (with Browser STT toggle; default ON)
  - "/chat"  : Plain chat UI (text only)
  - "/admin" : Admin console (KB add/scrape/unknowns/stats)
"""

from __future__ import annotations

import os
import logging
import asyncio
import base64
import tempfile
import subprocess
import time
import threading
import json
import uvicorn
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from uuid import uuid4
from pathlib import Path
from collections import defaultdict
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

# Retries (sync callables)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Cloud LLM/STT/TTS (sync clients)
from openai import OpenAI
from gtts import gTTS
from bs4 import BeautifulSoup
import httpx

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s"
)
logger = logging.getLogger("voice_agent")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Tablescapes Assistant")


AI_UNAVAILABLE_MSG = (
    "Sorry—our AI is temporarily unavailable. I can transfer you to a person or try again in a moment."
)

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # CORS
    CORS_ORIGINS = [o.strip() for o in os.getenv(
        "CORS_ORIGINS",
        "https://ai-voice-agent-uz9g.onrender.com,http://localhost:3000,http://localhost:8080"
    ).split(",") if o.strip()]

    # App
    BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
    DATABASE_PATH = os.getenv("DATABASE_PATH", "voice_agent.db")

    # Audio
    SAMPLE_RATE = 16000
    STT_TARGET_RATE = 16000
    AUDIO_TTS_MIME = "audio/mpeg"  # gTTS -> MP3
    MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE", "5242880"))            # 5 MB/frame
    MAX_COMBINED_AUDIO = int(os.getenv("MAX_COMBINED_AUDIO", "15728640"))   # 15 MB/batch
    AUDIO_BUFFER_SIZE = int(os.getenv("AUDIO_BUFFER_SIZE", "4"))            # frames per STT batch

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

    # Scraper
    ALLOWED_SCRAPE_HOSTS = {h.strip().lower() for h in os.getenv(
        "ALLOWED_SCRAPE_HOSTS", "tablescapes.com,www.tablescapes.com"
    ).split(",") if h.strip()}
    SCRAPE_MAX_PAGES = int(os.getenv("SCRAPE_MAX_PAGES", "40"))
    SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT", "10"))  # seconds

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
        """Convert to 16kHz mono WAV using ffmpeg. Avoid partial-chunk issues by batching before calling."""
        if len(data) > Config.MAX_COMBINED_AUDIO:
            raise ValueError("audio_combination_too_large")

        def _run():
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
                    # Fall back to raw WEBM (Whisper API accepts webm if complete enough)
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

        text = await _transcribe()
        logger.info(f"STT text='{text}'")
        return text

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
                         "content": (
                             "You are a concise customer service agent for a tableware and "
                             "event-rental business. Keep answers accurate and succinct. "
                             f"Context: {context}"
                         )},
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
                import traceback
                self.cb_chat.record_failure()
                logger.error("Chat error after retries: %s\n%s", repr(e), traceback.format_exc())
                # If it's quota/rate-limit, do NOT return the outage message; fall through to rules/KB.
                # This keeps chat usable even when the LLM is throttled or out of quota.
                # (We deliberately avoid returning AI_UNAVAILABLE_MSG here.)


        # Fallback (rule-based)
        u = (user_input or "").strip().lower()

        if not u:
            return "Could you repeat that?"

        # Small talk / presence
        if any(p in u for p in ["are you there", "you there", "can you hear me", "hello?", "hi?"]):
            return "I'm here and listening."

        # Greeting
        if any(p in u for p in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
            return f"Hi—I'm {ASSISTANT_NAME}. How can I help?"

        # Identity
        if any(p in u for p in ["what's your name", "whats your name", "what is your name", "who are you", "your name"]):
            return f"I'm {ASSISTANT_NAME}."

        # Capabilities
        if any(p in u for p in ["what can you do", "what do you do", "capabilities"]):
            return ("I can answer questions about inventory, pricing, delivery, policies, and general event-rental info. "
                    "Tell me what item or topic you have in mind.")

        # “Are you educated?”
        if any(p in u for p in ["are you educated", "are you smart", "education", "degree"]):
            return ("I’m a virtual assistant trained on our knowledge base and common patterns. "
                    "I don’t hold degrees, but I can help with accurate info and next steps.")

        # Existing task intents
        if any(p in u for p in ["price", "cost", "pricing", "how much"]):
            return "Which item do you want pricing for?"
        if any(p in u for p in ["hours", "open", "closed"]):
            return "Tell me which location and I’ll check hours."
        if any(p in u for p in ["shipping", "delivery", "ship"]):
            return "What would you like to know about delivery options?"
        if any(p in u for p in ["return", "exchange", "refund"]):
            return "Which item/order are you asking about?"

        # Politeness / closing
        if any(p in u for p in ["thanks", "thank you", "thx", "appreciate it"]):
            return "You’re welcome!"
        if any(p in u for p in ["bye", "goodbye", "see ya", "see you"]):
            return "Thanks for visiting—talk soon."

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
        return best

# ------------------------------------------------------------------------------
# Scraper (guarded, async) with Q&A + fallback summaries
# ------------------------------------------------------------------------------
class WebScraper:
    def __init__(self, allowed_hosts: set[str], max_pages: int = 40, timeout: int = 10):
        self.allowed_hosts = allowed_hosts
        self.max_pages = max_pages
        self.timeout = timeout

    def _allowed(self, url: str) -> bool:
        try:
            from urllib.parse import urlparse
            host = urlparse(url).netloc.split(":")[0].lower()
            return any(host == h or host.endswith(f".{h}") for h in self.allowed_hosts)
        except Exception:
            return False

    async def scrape(self, base_url: str) -> List[Dict[str, str]]:
        if not self._allowed(base_url):
            raise ValueError("Host not allowed by ALLOWED_SCRAPE_HOSTS")
        results: List[Dict[str, str]] = []
        seen = set()
        queue = [base_url]

        logger.info(f"Scrape requested: {base_url}")

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            while queue and len(seen) < self.max_pages:
                url = queue.pop(0)
                if url in seen:
                    continue
                seen.add(url)
                try:
                    r = await client.get(url)
                    if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                        continue
                    soup = BeautifulSoup(r.text, "html.parser")

                    page_title = (soup.title.string.strip() if soup.title and soup.title.string else "").strip()
                    text = self._clean_text(soup.get_text(" "))

                    # Q&A first
                    qa = self._extract_qa(text, url)
                    if qa:
                        results += qa
                    else:
                        # Fallback summary item so you always get something
                        summary = self._summary_from_text(text, 600)
                        if summary:
                            q = f"{page_title or 'Page'} – summary"
                            results.append({"question": q[:200], "answer": summary, "source": url})

                    # enqueue same-site links
                    for a in soup.find_all("a", href=True):
                        from urllib.parse import urljoin
                        next_url = urljoin(url, a["href"])
                        if self._allowed(next_url) and next_url not in seen:
                            queue.append(next_url)
                except Exception as e:
                    logger.warning(f"Scrape error at {url}: {e}")

        logger.info(f"Scrape finished: {base_url} -> items={len(results)}, pages_scanned={len(seen)}")
        return results

    def _clean_text(self, t: str) -> str:
        lines = [ln.strip() for ln in t.splitlines()]
        return " ".join([ln for ln in lines if ln])

    def _extract_qa(self, text: str, source: str) -> List[Dict[str, str]]:
        """Very simple heuristic: split on '? ' and treat trailing sentence as answer."""
        out: List[Dict[str, str]] = []
        if "?" not in text:
            return out
        parts = [p.strip() for p in text.split("? ") if p.strip()]
        for p in parts:
            if len(p) < 15:
                continue
            if " " not in p:
                continue
            q = p.split(" ")[0] + "?"
            a = p[len(q):].strip()
            if len(a) > 15 and len(a) < 500:
                out.append({"question": q, "answer": a, "source": source})
        return out

    def _summary_from_text(self, text: str, max_len: int = 600) -> str:
        """Grab the first meaningful chunk of text as a summary."""
        if not text:
            return ""
        sentences = [s.strip() for s in text.split(". ") if s.strip()]
        buf = []
        total = 0
        for s in sentences:
            if len(s) < 5:
                continue
            if total + len(s) + 2 > max_len:
                break
            buf.append(s)
            total += len(s) + 2
            if total > max_len * 0.6:
                break
        out = ". ".join(buf).strip()
        return out[:max_len] if out else text[:max_len]

# ------------------------------------------------------------------------------
# App wiring
# ------------------------------------------------------------------------------
db_manager = DatabaseManager(Config.DATABASE_PATH)
ai_engine = LightweightAIEngine()
knowledge_manager = KnowledgeBaseManager(ai_engine, db_manager)
scraper = WebScraper(Config.ALLOWED_SCRAPE_HOSTS, Config.SCRAPE_MAX_PAGES, Config.SCRAPE_TIMEOUT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_manager.init_database()
    await knowledge_manager.load()
    if not ffmpeg_available():
        logger.warning("ffmpeg not found; STT will attempt WEBM directly; results may be worse.")

    eff_port = int(os.getenv("PORT", str(Config.PORT)))
    external = os.getenv("RENDER_EXTERNAL_URL")  # e.g., https://ai-voice-agent-uz9g.onrender.com

    if external:
        base_url = external.rstrip("/")
        ws_url = base_url.replace("https://", "wss://") + "/ws/voice"
    else:
        base_url = f"http://localhost:{eff_port}"
        ws_url = f"ws://localhost:{eff_port}/ws/voice"

    try:
        Config.BASE_URL = base_url
    except Exception:
        pass

    logger.info("Starting AI Voice Agent System (Final Production + Full UI)")
    logger.info(f"Dashboard (effective): {base_url}")
    logger.info(f"WS (effective):        {ws_url}")
    logger.info(f"Configured BASE_URL:   {Config.BASE_URL}")
    logger.info("Startup complete.")
    yield

app = FastAPI(
    title="AI Voice Agent System - Final Production",
    version="1.8.0",
    lifespan=lifespan
)

@app.head("/", include_in_schema=False)
def root_head():
    return Response(status_code=200)

# Move JSON health to /healthz
@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS if Config.CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Voice UI (/) with Browser STT toggle (default ON)
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse("""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>AI Voice Agent</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root { --bg:#0f172a; --fg:#e2e8f0; --muted:#94a3b8; --ok:#22c55e; --warn:#f59e0b; --err:#ef4444; --btn:#1e40af; --btn2:#0891b2; }
* { box-sizing:border-box }
body { background:var(--bg); color:var(--fg); font:16px/1.4 Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin:0; padding:24px }
h1 { margin:0 0 12px 0 }
.card { background:#0b1220; border:1px solid #263147; border-radius:14px; padding:16px; margin:16px 0; box-shadow:0 0 0 1px rgba(255,255,255,.02) inset }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:12px 0 }
button { padding:12px 16px; border-radius:12px; border:none; background:var(--btn); color:white; cursor:pointer; font-weight:600 }
button.secondary { background:var(--btn2) }
button.danger { background:var(--err) }
button:disabled { opacity:.5; cursor:not-allowed }
.toggle { display:flex; align-items:center; gap:8px }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; background:#172036; color:#a6b0c6; font-size:12px; }
#log { white-space:pre-wrap; background:#0a0f1f; color:#cfe8ff; padding:12px; border-radius:12px; min-height:140px; max-height:300px; overflow:auto }
label { color:#9fb2d4; font-size:14px }
a.link { color:#7dd3fc; text-decoration:none }
</style>
<h1>AI Voice Agent</h1>

<div class="card">
  <div class="row"><span>Status:</span> <span id="status" class="badge">Idle</span></div>
  <div class="row">
    <div class="toggle">
      <input type="checkbox" id="useBrowserSTT" checked>
      <label for="useBrowserSTT">Use Browser STT (Web Speech API)</label>
    </div>
  </div>
  <div class="row">
    <button id="connectBtn">Connect</button>
    <button id="startBtn" disabled>Start</button>
    <button id="stopBtn" disabled>Stop</button>
    <button id="disconnectBtn" class="danger" disabled>Disconnect</button>
  </div>
  <div class="row">
    <audio id="player" controls></audio>
  </div>
</div>

<div class="card">
  <h3 style="margin:0 0 8px 0;">Log</h3>
  <div id="log"></div>
  <div class="row"><a class="link" href="/admin" target="_blank">Open Admin Console →</a> <a class="link" href="/chat" target="_blank" style="margin-left:8px">Open Chat →</a></div>
</div>

<script>
(() => {
  const statusEl = document.getElementById('status');
  const logEl = document.getElementById('log');
  const player = document.getElementById('player');
  const useBrowserSTT = document.getElementById('useBrowserSTT');

  const connectBtn = document.getElementById('connectBtn');
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const disconnectBtn = document.getElementById('disconnectBtn');

  let ws = null, stream = null, recorder = null, speech = null;

  function log(msg, obj){
    const s = (new Date()).toLocaleTimeString() + " — " + msg + (obj? " " + JSON.stringify(obj): "");
    logEl.textContent += s + "\\n";
    logEl.scrollTop = logEl.scrollHeight;
  }
  function setStatus(s){ statusEl.textContent = s; }

  function wsUrl(){
    const proto = (location.protocol === 'https:') ? 'wss://' : 'ws://';
    return proto + location.host + '/ws/voice';
  }

  function keepAlive(){
    if(ws && ws.readyState === WebSocket.OPEN){
      ws.send(JSON.stringify({type:'ping'}));
    }
  }
  setInterval(keepAlive, 20000);

  function enableControls(connected){
    connectBtn.disabled = connected;
    startBtn.disabled = !connected;
    stopBtn.disabled = true;
    disconnectBtn.disabled = !connected;
  }

  connectBtn.onclick = () => {
    if(ws && ws.readyState === WebSocket.OPEN) return;
    ws = new WebSocket(wsUrl());
    ws.binaryType = 'arraybuffer';
    setStatus('Connecting');
    ws.onopen = () => { setStatus('Connected'); log('WS open'); enableControls(true); };
    ws.onclose = ev => {
      setStatus('Disconnected'); log('WS close', {code: ev.code, reason: ev.reason});
      if(recorder && recorder.state !== 'inactive') recorder.stop();
      if(stream) { stream.getTracks().forEach(t=>t.stop()); stream = null; }
      if(speech) try{ speech.stop(); }catch(e){}
      enableControls(false);
    };
    ws.onerror = () => { log('WS error'); };
    ws.onmessage = ev => {
      try {
        const msg = JSON.parse(ev.data);
        if(msg.type === 'heartbeat') setStatus('Connected (heartbeat)');
        else if(msg.type === 'listening') setStatus('Listening…');
        else if(msg.type === 'response'){
          setStatus('Responded');
          log('Agent: ' + msg.text + ' (conf=' + (msg.confidence ?? 0).toFixed(2) + ')');
          if(msg.audio_b64 && msg.audio_mime){
            player.src = 'data:' + msg.audio_mime + ';base64,' + msg.audio_b64;
            player.play().catch(()=>{});
          }
        } else if(msg.type === 'transfer'){ setStatus('Transfer'); log('Transfer: ' + msg.message); }
        else if(msg.type === 'error'){ setStatus('Error'); log('Server error: ' + msg.message); }
        else if(msg.type === 'closing'){ setStatus('Closing: ' + (msg.reason||'')); }
        else { log('Recv', msg); }
      } catch(e) { log('Non-JSON message'); }
    };
  };

  startBtn.onclick = async () => {
    if(!ws || ws.readyState !== WebSocket.OPEN) { log('WS not open'); return; }
    const useBrowser = useBrowserSTT.checked;

    if(useBrowser){
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if(!SR) {
        alert('Web Speech API not supported. Disable "Use Browser STT" or use Chrome.');
        return;
      }
      speech = new SR();
      speech.lang = 'en-US';
      speech.interimResults = true;
      speech.continuous = true;
      speech.onresult = (e) => {
        let finalText = '';
        for(let i=e.resultIndex; i<e.results.length; i++) {
          if(e.results[i].isFinal) finalText += e.results[i][0].transcript;
        }
        if(finalText.trim() && ws && ws.readyState === WebSocket.OPEN){
          ws.send(JSON.stringify({type:'text', text: finalText}));
        }
      };
      speech.onerror = (e)=> log('Speech error: ' + e.error);
      speech.onend = ()=> log('Speech end');
      try { speech.start(); setStatus('Browser STT Running'); }
      catch(err){ log('Speech start failed: ' + err); }
      startBtn.disabled = true; stopBtn.disabled = false; disconnectBtn.disabled = false;
      return;
    }

    if(!navigator.mediaDevices || !window.MediaRecorder){
      alert('MediaRecorder not supported in this browser.');
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio: true});
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' :
                   (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');
      recorder = new MediaRecorder(stream, mime ? {mimeType: mime, audioBitsPerSecond: 32000} : undefined);
      recorder.ondataavailable = async (e) => {
        if(e.data && e.data.size > 0 && ws && ws.readyState === WebSocket.OPEN){
          const buf = await e.data.arrayBuffer();
          ws.send(buf);
        }
      };
      recorder.start(1000);
      setStatus('Recording'); startBtn.disabled = true; stopBtn.disabled = false; disconnectBtn.disabled = false;
      log('Recorder started with mime=' + (mime||'default'));
    } catch(err){
      log('getUserMedia error: ' + err);
    }
  };

  stopBtn.onclick = () => {
    if(recorder && recorder.state !== 'inactive') recorder.stop();
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream = null; }
    if(speech) try{ speech.stop(); }catch(e){}
    setStatus('Connected'); startBtn.disabled = false; stopBtn.disabled = true; log('Stopped');
  };

  disconnectBtn.onclick = () => {
    if(recorder && recorder.state !== 'inactive') recorder.stop();
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream = null; }
    if(speech) try{ speech.stop(); }catch(e){}
    if(ws){ ws.close(1000, 'client_close'); }
  };
})();
</script>
</html>
""")

# ------------------------------------------------------------------------------
# Chat UI (text only)
# ------------------------------------------------------------------------------
@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    return HTMLResponse("""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>AI Voice Agent – Chat</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root { --bg:#0f172a; --fg:#e2e8f0; --muted:#94a3b8; --card:#0b1220; --border:#263147; --btn:#1e40af; }
*{box-sizing:border-box} body{background:var(--bg);color:var(--fg);font:16px/1.4 Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;padding:24px}
h1{margin:0 0 12px 0} .card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px;margin:16px 0}
#msgs{max-height:60vh;overflow:auto;padding:12px;background:#0a0f1f;border-radius:12px}
.msg{margin:8px 0;padding:10px;border-radius:10px;border:1px solid var(--border)}
.msg.user{background:#0f1a33} .msg.assistant{background:#101b2e}
small{color:var(--muted)} input,button{font:16px}
input{width:100%;padding:12px;border-radius:12px;border:1px solid var(--border);background:#0a0f1f;color:#cfe8ff}
button{padding:12px 16px;border-radius:12px;border:none;background:var(--btn);color:#fff;font-weight:600;cursor:pointer}
.row{display:flex;gap:8px;margin-top:8px}
</style>
<h1>Chat</h1>
<div class="card">
  <div id="msgs" aria-live="polite"></div>
  <div class="row">
    <input id="inp" placeholder="Type a message and press Enter">
    <button id="send">Send</button>
  </div>
  <div class="row"><small id="meta"></small></div>
</div>
<script>
(() => {
  const msgs = document.getElementById('msgs');
  const inp  = document.getElementById('inp');
  const btn  = document.getElementById('send');
  const meta = document.getElementById('meta');
  const history = [];

  function add(role, text){
    const d = document.createElement('div');
    d.className = 'msg ' + role;
    d.innerHTML = '<strong>' + (role==='user'?'You':'Agent') + ':</strong> ' + text;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
  }

  async function send(){
    const text = inp.value.trim();
    if(!text) return;
    add('user', text);
    history.push({role:'user', text});
    inp.value=''; meta.textContent='';

    try{
      const r = await fetch('/api/chat', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({message: text, history})
      });
      const j = await r.json();
      if(!j.ok){ add('assistant', 'Error: ' + (j.error||'unknown')); return; }
      add('assistant', j.reply);
      history.push({role:'assistant', text: j.reply});
      meta.textContent = 'source=' + j.source + (typeof j.confidence==='number' ? ' • kb_conf=' + j.confidence.toFixed(2) : '');
    }catch(e){
      add('assistant', 'Network error.');
    }
  }

  btn.onclick = send;
  inp.addEventListener('keydown', e => { if(e.key==='Enter') send(); });
})();
</script>
</html>
""")

# ------------------------------------------------------------------------------
# Admin UI (/admin) – dev-only, no auth (your call)
# ------------------------------------------------------------------------------
@app.get("/admin", response_class=HTMLResponse)
async def admin():
    return HTMLResponse("""
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Admin Console – AI Voice Agent (Dev)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial;background:#0f172a;color:#e5e7eb;margin:0;padding:24px}
  h1{margin:0 0 8px 0}
  .card{background:#0b1220;border:1px solid #263147;border-radius:14px;padding:16px;margin:16px 0}
  input,textarea{width:100%;padding:10px;border-radius:10px;border:1px solid #263147;background:#0a0f1f;color:#dbeafe}
  button{padding:10px 14px;border-radius:10px;border:none;background:#2563eb;color:white;cursor:pointer;font-weight:600}
  table{width:100%;border-collapse:collapse}
  th,td{border-bottom:1px solid #263147;padding:8px;text-align:left}
  .muted{color:#93a0b8;font-size:12px}
  .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
</style>
<h1>Admin Console <span class="muted">(dev only)</span></h1>

<div class="card">
  <h3>Add KB Item</h3>
  <div><input id="q" placeholder="Question"></div>
  <div style="margin-top:8px"><textarea id="a" rows="4" placeholder="Answer"></textarea></div>
  <div style="margin-top:8px"><button onclick="addKB()">Add</button></div>
</div>

<div class="card">
  <h3>Scrape Website (allowed hosts only)</h3>
  <div class="row"><input id="url" placeholder="https://www.example.com"> <button onclick="scrape()">Scrape</button></div>
  <div id="scrapeResult" class="muted"></div>
</div>

<div class="card">
  <h3>Unknown Questions</h3>
  <div id="unknowns"></div>
</div>

<div class="card">
  <h3>Stats</h3>
  <pre id="stats"></pre>
</div>

<script>
async function addKB(){
  const q = document.getElementById('q').value.trim();
  const a = document.getElementById('a').value.trim();
  if(!q || !a){ alert('Enter both question and answer'); return; }
  const r = await fetch('/api/knowledge/add', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q,answer:a})});
  const j = await r.json();
  alert(j.message || 'OK');
  loadUnknowns(); loadStats();
}

async function scrape(){
  const urlEl = document.getElementById('url');
  const outEl = document.getElementById('scrapeResult');
  const btn = document.querySelector('button[onclick="scrape()"]');

  const url = urlEl.value.trim();
  if(!url){ alert('Enter URL'); return; }

  btn.disabled = true;
  const t0 = Date.now();
  outEl.textContent = `Starting scrape… ${url}`;

  try {
    const r = await fetch('/api/knowledge/scrape', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({url})
    });
    const text = await r.text();
    const msg = `Finished in ${((Date.now()-t0)/1000).toFixed(1)}s\\n` +
                (r.ok ? text : `HTTP ${r.status}\\n${text}`);
    outEl.textContent = msg;
  } catch (e) {
    outEl.textContent = `Network error: ${String(e)}`;
  } finally {
    btn.disabled = false;
    loadStats();
  }
}
async function resolveUnknown(id){
  const ans = prompt('Provide the answer for this question:');
  if(!ans) return;
  const r = await fetch('/api/knowledge/resolve', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question_id:id, answer:ans})});
  const j = await r.json();
  alert(j.message || 'Resolved');
  loadUnknowns(); loadStats();
}

async function loadUnknowns(){
  const r = await fetch('/api/knowledge/unknown');
  const j = await r.json();
  const div = document.getElementById('unknowns');
  if(!j.questions || !j.questions.length){ div.innerHTML = '<div class="muted">None</div>'; return; }
  div.innerHTML = '<table><tr><th>Time</th><th>Question</th><th>Suggested</th><th></th></tr>' +
    j.questions.map(q => '<tr><td>'+ (q.timestamp||'') +'</td><td>'+ (q.question||'') +'</td><td>'+ (q.suggested_answer||'') +'</td><td><button onclick="resolveUnknown(\\''+q.id+'\\')">Resolve</button></td></tr>').join('') +
    '</table>';
}

async function loadStats(){
  const r = await fetch('/api/stats');
  const j = await r.json();
  document.getElementById('stats').textContent = JSON.stringify(j, null, 2);
}

loadUnknowns(); loadStats();
</script>
</html>
""")

# ------------------------------------------------------------------------------
# Chat API – reuses KB + engine
# ------------------------------------------------------------------------------
@app.post("/api/chat")
async def api_chat(payload: Dict[str, Any]):
    message = (payload.get("message") or "").strip()
    history = payload.get("history") or []  # [{role:"user"/"assistant", text:"..."}]

    if not message:
        return JSONResponse({"ok": False, "error": "empty_message"}, status_code=400)

    # Build lightweight context from recent turns
    try:
        lines = []
        for h in history[-8:]:
            role = (h.get("role") or "").strip().lower()
            txt = limit_len((h.get("text") or "").strip(), 500)
            if not role or not txt:
                continue
            lines.append(f"{role.capitalize()}: {txt}")
        context = "\n".join(lines)
    except Exception:
        context = ""

    # First try KB
    answer, confidence, _ = await knowledge_manager.find_answer(message)
    if confidence >= Config.CONFIDENCE_THRESHOLD:
        return {"ok": True, "reply": answer, "confidence": confidence, "source": "kb"}

    # Fall back to engine (may be LLM or rule-based or outage line)
    reply = await ai_engine.generate_response(message, context=context)
    source = "rules"
    if reply == AI_UNAVAILABLE_MSG:
        source = "llm_unavailable"
    elif ai_engine.openai_client:
        source = "llm"

    # Save unknown for curation only if we produced a real AI answer
    if reply != AI_UNAVAILABLE_MSG:
        uq = UnknownQuestion(
            id=f"unknown_{uuid4().hex}",
            question=message,
            call_id="chat",  # not a phone call; mark as chat
            timestamp=now_utc(),
            suggested_answer=reply
        )
        await db_manager.save_unknown_question(uq)

    return {"ok": True, "reply": reply, "confidence": confidence, "source": source}

# ------------------------------------------------------------------------------
# KB + Scraper + Stats APIs
# ------------------------------------------------------------------------------
@app.post("/api/knowledge/add")
async def api_add_kb(payload: Dict[str, Any]):
    q = (payload.get("question") or "").strip()
    a = (payload.get("answer") or "").strip()
    if not q or not a:
        return JSONResponse({"status":"error","message":"question and answer required"}, status_code=400)
    await knowledge_manager.add_knowledge_item(q, a, source="manual")
    return {"status":"success","message":"Knowledge item added"}

@app.get("/debug/openai", include_in_schema=False)
async def debug_openai():
    try:
        if not ai_engine.openai_client:
            raise RuntimeError("no_openai_client (missing OPENAI_API_KEY?)")
        resp = await run_in_threadpool(
            ai_engine._sync_openai_chat,
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "ping"}
            ],
            Config.OPENAI_CHAT_MODEL,
            0.2,
            10
        )
        txt = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "model": Config.OPENAI_CHAT_MODEL, "reply": txt[:100]}
    except Exception as e:
        return {"ok": False, "error": repr(e)}

@app.post("/api/knowledge/scrape")
async def api_scrape(payload: Dict[str, Any]):
    url = (payload.get("url") or "").strip()
    if not url:
        return JSONResponse({"status":"error","message":"url required"}, status_code=400)
    try:
        qa = await scraper.scrape(url)
        added = 0
        for item in qa:
            await knowledge_manager.add_knowledge_item(
                item["question"], item["answer"],
                source=f"scraped:{item.get('source','')}"
            )
            added += 1
        pages = len({x.get("source") for x in qa}) or 0
        if added == 0:
            return {"status":"success","message":"Scrape completed, but no Q&A patterns were found","pages_scanned": pages}
        return {"status":"success","message":f"Scraped {len(qa)} items; added {added}","pages_scanned": pages}
    except ValueError as ve:
        return JSONResponse({"status":"error","message":str(ve)}, status_code=400)
    except Exception:
        logger.exception("Scrape failed")
        return JSONResponse({"status":"error","message":"scrape_failed"}, status_code=500)

@app.get("/api/knowledge/unknown")
async def api_unknown():
    qs = await db_manager.get_unknown_questions(resolved=False)
    out = []
    for q in qs:
        out.append({
            "id": q.id,
            "question": q.question,
            "call_id": q.call_id,
            "timestamp": q.timestamp.isoformat(),
            "resolved": q.resolved,
            "suggested_answer": q.suggested_answer
        })
    return {"questions": out}

@app.post("/api/knowledge/resolve")
async def api_resolve(payload: Dict[str, Any]):
    qid = (payload.get("question_id") or "").strip()
    ans = (payload.get("answer") or "").strip()
    if not qid or not ans:
        return JSONResponse({"status":"error","message":"question_id and answer required"}, status_code=400)
    unknowns = await db_manager.get_unknown_questions(resolved=False)
    target = next((u for u in unknowns if u.id == qid), None)
    if not target:
        return JSONResponse({"status":"error","message":"unknown_question_not_found"}, status_code=404)

    await knowledge_manager.add_knowledge_item(target.question, ans, source="resolved")

    async with aiosqlite.connect(Config.DATABASE_PATH) as conn:
        await conn.execute("UPDATE unknown_questions SET resolved = 1 WHERE id = ?", (qid,))
        await conn.commit()
    return {"status":"success","message":"Question resolved and added to KB"}

@app.get("/api/stats")
async def api_stats():
    stats = await db_manager.get_call_statistics()
    return stats

# ------------------------------------------------------------------------------
# WebSocket voice endpoint – accepts binary frames and JSON text
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

            # Receive a frame (text OR bytes) with timeout
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=Config.WEBSOCKET_TIMEOUT)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
                continue
            except Exception as e:
                logger.warning(f"Receive error ({call_id}): {e}")
                break

            # Handle client close
            if msg.get("type") == "websocket.disconnect":
                break

            # ---- TEXT FRAME PATH (browser STT / control messages) ----
            if "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    payload = {"type": "text", "text": msg["text"]}

                if payload.get("type") == "ping":
                    await websocket.send_json({"type": "heartbeat"})
                    continue

                if payload.get("type") == "text":
                    text = (payload.get("text") or "").strip()
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
                        if any(k in text.lower() for k in
                               ["human", "person", "agent", "representative", "manager", "supervisor", "help me", "speak to someone"]):
                            await websocket.send_json({"type": "transfer", "message": "Transferring you to a human agent."})
                            await websocket.send_json({"type": "closing", "reason": "transfer"})
                            await websocket.close(code=1000)
                            call.outcome = "transferred"
                            break

                        # LLM/rule response
                        response_text = await ai_engine.generate_response(text)

                        # Record unknown for later curation if we produced a real answer
                        if response_text != AI_UNAVAILABLE_MSG:
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

                    # TTS (always generated on server so the user hears something)
                    audio_bytes = await ai_engine.text_to_speech(response_text)

                    await websocket.send_json({
                        "type": "response",
                        "text": response_text,
                        "audio_b64": base64.b64encode(audio_bytes).decode() if audio_bytes else "",
                        "audio_mime": Config.AUDIO_TTS_MIME,
                        "confidence": confidence
                    })
                    continue  # done with text frame

                # Unknown control message
                await websocket.send_json({"type": "error", "message": "unrecognized_control_message"})
                continue

            # ---- BINARY FRAME PATH (mic audio) ----
            data: Optional[bytes] = msg.get("bytes")
            if not data:
                await websocket.send_json({"type": "noop"})
                continue

            if len(data) > Config.MAX_AUDIO_SIZE:
                await websocket.send_json({"type": "error", "message": "audio_frame_too_large"})
                continue

            # Buffer frames safely (batching)
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

            if len(batch) > Config.MAX_COMBINED_AUDIO:
                await websocket.send_json({"type": "error", "message": "audio_combination_too_large"})
                continue

            logger.info(f"STT batch bytes={len(batch)}")

            # STT
            text = await ai_engine.speech_to_text(batch)
            logger.info(f"STT text='{text}'")
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
                if any(k in text.lower() for k in
                       ["human", "person", "agent", "representative", "manager", "supervisor", "help me", "speak to someone"]):
                    await websocket.send_json({"type": "transfer", "message": "Transferring you to a human agent."})
                    await websocket.send_json({"type": "closing", "reason": "transfer"})
                    await websocket.close(code=1000)
                    call.outcome = "transferred"
                    break

                # LLM/rule response
                response_text = await ai_engine.generate_response(text)

                # Log unknown only if we produced a real AI answer
                if response_text != AI_UNAVAILABLE_MSG:
                    uq = UnknownQuestion(
                        id=f"unknown_{uuid4().hex}",
                        question=text,
                        call_id=call_id,
                        timestamp=now_utc(),
                        suggested_answer=response_text
                    )
                    await db_manager.save_unknown_question(uq)

            # ALWAYS send a response (even the outage message)
            call.transcript += f"Agent: {response_text}\n"
            call.confidence_score = confidence

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
# Health
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
    eff_port = int(os.getenv("PORT", str(Config.PORT)))
    uvicorn.run(app, host=Config.HOST, port=eff_port, log_level="info")
