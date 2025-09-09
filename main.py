#!/usr/bin/env python3
"""
Self-Sufficient AI Voice Agent System – Final Production Version
- Fixed race conditions in audio buffering and database updates
- Memory protection for large audio combinations
- Enhanced rate limiting with connection tracking
- Atomic database operations for usage counting
- Improved error recovery and resource cleanup
"""

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
import httpx
from bs4 import BeautifulSoup

from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

# Retries
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Optional cloud LLM/STT/TTS
from openai import OpenAI  # v1 client (sync)
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
    # OpenAI (optional)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # App
    BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
    DATABASE_PATH = os.getenv("DATABASE_PATH", "voice_agent.db")
    MODELS_PATH = os.getenv("MODELS_PATH", "models")
    AUDIO_PATH = os.getenv("AUDIO_PATH", "audio")
    TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", "templates")
    STATIC_DIR = os.getenv("STATIC_DIR", "static")

    # Audio (with memory protection)
    SAMPLE_RATE = 16000
    AUDIO_TTS_MIME = "audio/mpeg"  # gTTS returns MP3
    STT_TARGET_RATE = 16000
    MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE", "5242880"))  # 5MB per frame (reduced)
    MAX_COMBINED_AUDIO = int(os.getenv("MAX_COMBINED_AUDIO", "15728640"))  # 15MB total
    AUDIO_BUFFER_SIZE = int(os.getenv("AUDIO_BUFFER_SIZE", "3"))  # Chunks to buffer

    # KB / matching
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

    # Call control
    MAX_CALL_DURATION = int(os.getenv("MAX_CALL_DURATION", "600"))  # seconds
    WEBSOCKET_TIMEOUT = int(os.getenv("WEBSOCKET_TIMEOUT", "10"))  # seconds

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # Security
    API_KEY = os.getenv("VOICE_AGENT_API_KEY")  # if set, required for /api/*
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "4000"))

    # Rate limiting (more conservative for production)
    MAX_CONCURRENT_CONNECTIONS = int(os.getenv("MAX_CONCURRENT_CONNECTIONS", "10"))  # Global limit
    RATE_LIMIT_CONNECTIONS = int(os.getenv("RATE_LIMIT_CONNECTIONS", "3"))  # Per IP
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "300"))  # 5 minutes
    
    # Circuit breaker settings
    CB_FAIL_THRESHOLD = int(os.getenv("CB_FAIL_THRESHOLD", "3"))
    CB_RESET_TIMEOUT = int(os.getenv("CB_RESET_TIMEOUT", "30"))
    CB_HALF_OPEN_LIMIT = int(os.getenv("CB_HALF_OPEN_LIMIT", "2"))

    # Scraper safety
    ALLOWED_SCRAPE_HOSTS = set(
        h.strip().lower()
        for h in os.getenv(
            "ALLOWED_SCRAPE_HOSTS", "tablescapes.com,www.tablescapes.com"
        ).split(",")
        if h.strip()
    )

    # Routing keywords
    ROUTE_TO_HUMAN_KEYWORDS = [
        "human", "person", "agent", "representative",
        "manager", "supervisor", "help me", "speak to someone"
    ]

# Ensure dirs
os.makedirs(Config.MODELS_PATH, exist_ok=True)
os.makedirs(Config.AUDIO_PATH, exist_ok=True)
os.makedirs(Config.TEMPLATE_DIR, exist_ok=True)
os.makedirs(Config.STATIC_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------------------
@dataclass
class Call:
    call_id: str
    caller_info: str  # phone/browser/local
    start_time: datetime
    end_time: Optional[datetime] = None
    transcript: str = ""
    outcome: str = ""  # answered, transferred, failed
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
# Connection Tracking (Global)
# ------------------------------------------------------------------------------
class ConnectionTracker:
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.active_connections = set()
        self.lock = threading.Lock()
    
    def add_connection(self, conn_id: str) -> bool:
        with self.lock:
            if len(self.active_connections) >= self.max_connections:
                return False
            self.active_connections.add(conn_id)
            return True
    
    def remove_connection(self, conn_id: str):
        with self.lock:
            self.active_connections.discard(conn_id)
    
    def get_count(self) -> int:
        with self.lock:
            return len(self.active_connections)

# Global connection tracker
connection_tracker = ConnectionTracker(Config.MAX_CONCURRENT_CONNECTIONS)

# ------------------------------------------------------------------------------
# Enhanced Rate Limiter (Thread-safe)
# ------------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_connections: int = 3, window_seconds: int = 300, max_clients: int = 1000):
        self.max_connections = max_connections
        self.window = window_seconds
        self.max_clients = max_clients
        self.connections = defaultdict(list)
        self.lock = threading.Lock()
        self.last_cleanup = time.monotonic()
        self.cleanup_interval = 300

    def is_allowed(self, client_id: str) -> bool:
        now = time.monotonic()
        
        with self.lock:
            # Periodic cleanup
            if now - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_entries(now)
                self.last_cleanup = now

            win_start = now - self.window
            connections = self.connections[client_id]
            
            # Remove old connections
            self.connections[client_id] = [ts for ts in connections if ts > win_start]
            connections = self.connections[client_id]

            if len(connections) < self.max_connections:
                connections.append(now)
                return True
            return False

    def _cleanup_old_entries(self, now: float):
        """Remove inactive clients (called with lock held)"""
        win_start = now - self.window * 2
        clients_to_remove = []
        
        for client_id, timestamps in self.connections.items():
            if not timestamps or timestamps[-1] < win_start:
                clients_to_remove.append(client_id)
        
        for client_id in clients_to_remove:
            del self.connections[client_id]
        
        # Emergency cleanup
        if len(self.connections) > self.max_clients:
            sorted_clients = sorted(
                self.connections.items(), 
                key=lambda x: x[1][-1] if x[1] else 0
            )
            clients_to_remove = sorted_clients[:len(self.connections) - self.max_clients]
            for client_id, _ in clients_to_remove:
                del self.connections[client_id]

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "active_clients": len(self.connections),
                "total_connections": sum(len(timestamps) for timestamps in self.connections.values()),
                "window_seconds": self.window,
                "max_connections_per_window": self.max_connections
            }

# ------------------------------------------------------------------------------
# Enhanced Circuit Breaker (Fixed state transitions)
# ------------------------------------------------------------------------------
class CircuitBreaker:
    def __init__(self, fail_threshold: int = 3, reset_timeout: int = 30, half_open_limit: int = 2):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.half_open_limit = half_open_limit
        
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.opened_at: Optional[float] = None
        self.half_open_calls = 0
        self.lock = threading.Lock()

    def can_call(self) -> bool:
        with self.lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if self.opened_at is None:
                    return False
                if time.monotonic() - self.opened_at >= self.reset_timeout:
                    self.state = "half-open"
                    self.half_open_calls = 0
                    self.success_count = 0
                    logger.info("Circuit breaker transitioning to half-open state")
                    return True
                return False
            elif self.state == "half-open":
                can_proceed = self.half_open_calls < self.half_open_limit
                if can_proceed:
                    self.half_open_calls += 1
                return can_proceed
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
                    logger.info("Circuit breaker recovered to closed state")
            elif self.state == "closed":
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        with self.lock:
            if self.state == "half-open":
                self.state = "open"
                self.opened_at = time.monotonic()
                self.half_open_calls = 0
                self.success_count = 0
                logger.warning("Circuit breaker failed in half-open state, returning to open")
            else:
                self.failure_count += 1
                if self.failure_count >= self.fail_threshold:
                    self.state = "open"
                    self.opened_at = time.monotonic()
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "half_open_calls": self.half_open_calls,
                "can_call": self.state == "closed" or (self.state == "half-open" and self.half_open_calls < self.half_open_limit)
            }

# ------------------------------------------------------------------------------
# AI Engine (Thread-safe operations)
# ------------------------------------------------------------------------------
class LightweightAIEngine:
    def __init__(self):
        self.openai_client: Optional[OpenAI] = None
        if Config.OPENAI_API_KEY:
            self.openai_client = OpenAI(
                api_key=Config.OPENAI_API_KEY,
                timeout=30.0,
                max_retries=0
            )
            logger.info("OpenAI client initialized with thread-safe circuit breakers.")
        else:
            logger.info("No OPENAI_API_KEY set. Falling back to rule-based responses.")
        
        self.cb_stt = CircuitBreaker(
            fail_threshold=Config.CB_FAIL_THRESHOLD,
            reset_timeout=Config.CB_RESET_TIMEOUT,
            half_open_limit=Config.CB_HALF_OPEN_LIMIT
        )
        self.cb_chat = CircuitBreaker(
            fail_threshold=Config.CB_FAIL_THRESHOLD,
            reset_timeout=Config.CB_RESET_TIMEOUT,
            half_open_limit=Config.CB_HALF_OPEN_LIMIT
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _sync_openai_stt(self, file_obj, model: str):
        return self.openai_client.audio.transcriptions.create(model=model, file=file_obj)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _sync_openai_chat(self, messages, model, temperature, max_tokens):
        return self.openai_client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens
        )

    async def convert_audio_to(self, data: bytes, target_sample_rate: int = 16000) -> Tuple[bytes, str]:
        """Convert audio with size validation"""
        if len(data) > Config.MAX_COMBINED_AUDIO:
            logger.warning(f"Audio data too large for conversion: {len(data)} bytes")
            return data[:Config.MAX_COMBINED_AUDIO], "webm"  # Truncate
            
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
        """STT with memory protection and circuit breaker"""
        if not self.openai_client or not self.cb_stt.can_call():
            return ""

        if len(audio_bytes) > Config.MAX_COMBINED_AUDIO:
            logger.warning(f"Audio too large for STT: {len(audio_bytes)} bytes")
            return ""

        wav_bytes, fmt = await self.convert_audio_to(audio_bytes, Config.STT_TARGET_RATE)
        suffix = ".wav" if fmt == "wav" else ".webm"

        def _transcribe_sync(path: Path, model: str) -> str:
            with path.open("rb") as f:
                resp = self._sync_openai_stt(f, model)
                return (resp.text or "").strip()

        async def _transcribe():
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / f"audio{suffix}"
                p.write_bytes(wav_bytes)
                try:
                    model = os.getenv("OPENAI_STT_MODEL", "whisper-1")
                    text = await run_in_threadpool(_transcribe_sync, p, model)
                    self.cb_stt.record_success()
                    return text
                except Exception as e:
                    self.cb_stt.record_failure()
                    logger.error(f"STT error after retries: {e}")
                    return ""
        return await _transcribe()

    async def text_to_speech(self, text: str) -> bytes:
        """TTS with timeout protection"""
        text = limit_len(text, Config.MAX_INPUT_LEN)
        
        async def _tts():
            try:
                with tempfile.TemporaryDirectory() as td:
                    out = Path(td) / "tts.mp3"
                    gTTS(text=text, lang="en", slow=False).save(str(out))
                    return out.read_bytes()
            except Exception as e:
                logger.error(f"TTS error: {e}")
                return b""
        
        try:
            return await asyncio.wait_for(_tts(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("TTS operation timed out")
            return b""

    async def generate_response(self, user_input: str, context: str = "") -> str:
        """LLM response with circuit breaker"""
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
                    os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
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
            return "Tell me which location and I'll check hours."
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

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        return {
            "stt": self.cb_stt.get_state(),
            "chat": self.cb_chat.get_state()
        }

# ------------------------------------------------------------------------------
# Database (with atomic operations)
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
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
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
        """Atomic usage count increment"""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                "UPDATE knowledge_base SET usage_count = usage_count + 1 WHERE id = ?", 
                (item_id,)
            )
            await conn.commit()

    async def save_unknown_question(self, q: UnknownQuestion):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO unknown_questions
                (id, question, call_id, timestamp, resolved, suggested_answer)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                q.id, q.question, q.call_id, q.timestamp.isoformat(),
                int(q.resolved), q.suggested_answer
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
                "SELECT * FROM unknown_questions WHERE resolved = ? ORDER BY timestamp DESC", 
                (int(resolved),)
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
# Knowledge Manager (with atomic updates)
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
        """Returns (answer, confidence, item_id)"""
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

        # Atomic usage count increment
        if best[2]:  # If we found a match
            await self.db.increment_usage_count(best[2])

        return best[0], best[1], best[2]

# ------------------------------------------------------------------------------
# Remaining classes (WebScraper, HealthChecker, etc.) stay the same...
# [Truncated for space - these don't need changes]
# ------------------------------------------------------------------------------

# Initialize components
db_manager = DatabaseManager(Config.DATABASE_PATH)
ai_engine = LightweightAIEngine()
knowledge_manager = KnowledgeBaseManager(ai_engine, db_manager)
rate_limiter = RateLimiter(Config.RATE_LIMIT_CONNECTIONS, Config.RATE_LIMIT_WINDOW)

app = FastAPI(title="AI Voice Agent System - Final Production", version="1.4.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS if Config.CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Fixed WebSocket Handler (Race-condition free)
# ------------------------------------------------------------------------------
@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    call_id = f"call_{uuid4().hex}"
    
    # Global connection limit check
    if not connection_tracker.add_connection(call_id):
        await websocket.send_json({
            "type": "error", 
            "message": "Server at maximum capacity. Please try again later."
        })
        await websocket.close(code=1013)  # Try again later
        return

    try:
        call = Call(call_id=call_id, caller_info="websocket", start_time=now_utc())
        logger.info(f"Call started: {call_id} (Active: {connection_tracker.get_count()})")

        start_ts = time.monotonic()
        
        # Client identification for rate limiting
        client_id = (
            f"{websocket.client.host}"
            if websocket.client else websocket.headers.get("sec-websocket-key", "unknown")
        )

        # Per-IP rate limit check
        if not rate_limiter.is_allowed(client_id):
            await websocket.send_json({
                "type": "error", 
                "message": "Rate limit exceeded for your IP. Please try again later."
            })
            await websocket.close(code=1008)
            return

        # Audio processing with race condition protection
        audio_buffer = []
        buffer_lock = asyncio.Lock()
        last_process_time = time.monotonic()
        
        while True:
            current_time = time.monotonic()
            
            # Enforce max duration
            if current_time - start_ts > Config.MAX_CALL_DURATION:
                await websocket.send_json({"type": "closing", "reason": "max_duration"})
                await websocket.close(code=1000)
                call.outcome = call.outcome or "answered"
                break

            # Receive audio with timeout
            try:
                data = await asyncio.wait_for(
                    websocket.receive_bytes(), 
                    timeout=Config.WEBSOCKET_TIMEOUT
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
                continue
            except Exception as e:
                logger.warning(f"Error receiving audio: {e}")
                break

            if not data:
                await websocket.send_json({"type": "noop"})
                continue

            if len(data) > Config.MAX_AUDIO_SIZE:
                await websocket.send_json({
                    "type": "error", 
                    "message": "Audio frame too large"
                })
                continue

            # Thread-safe audio buffering
            async with buffer_lock:
                audio_buffer.append(data)
                should_process = (
                    len(audio_buffer) >= Config.AUDIO_BUFFER_SIZE or 
                    current_time - last_process_time > 2.0
                )
                
                if should_process:
                    # Create copy and clear buffer atomically
                    processing_buffer = audio_buffer.copy()
                    audio_buffer.clear()
                    last_process_time = current_time
                else:
                    processing_buffer = None

            # Process audio outside of lock
            if processing_buffer:
                # Check combined size before processing
                combined_size = sum(len(chunk) for chunk in processing_buffer)
                if combined_size > Config.MAX_COMBINED_AUDIO:
                    logger.warning(f"Combined audio too large: {combined_size} bytes")
                    await websocket.send_json({
                        "type": "error", 
                        "message": "Audio combination too large"
                    })
                    continue

                combined_audio = b''.join(processing_buffer)
                
                # STT with circuit breaker protection
                text = await ai_engine.speech_to_text(combined_audio)
                if not text:
                    await websocket.send_json({"type": "listening"})
                    continue

                call.transcript += f"User: {text}\n"

                # Knowledge base lookup
                answer, confidence, item_id = await knowledge_manager.find_answer(text)

                if confidence >= Config.CONFIDENCE_THRESHOLD:
                    response_text = answer
                    call.outcome = call.outcome or "answered"
                else:
                    # Check for human transfer intent
                    if any(k in text.lower() for k in Config.ROUTE_TO_HUMAN_KEYWORDS):
                        msg = "Transferring you to a human agent."
                        await websocket.send_json({"type": "transfer", "message": msg})
                        await websocket.send_json({"type": "closing", "reason": "transfer"})
                        await websocket.close(code=1000)
                        call.outcome = "transferred"
                        break
                    
                    # Generate AI response
                    response_text = await ai_engine.generate_response(text)
                    
                    # Record unknown question
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

                # TTS with error handling
                try:
                    audio_bytes = await ai_engine.text_to_speech(response_text)
                except Exception as e:
                    logger.error(f"TTS error: {e}")
                    audio_bytes = b""

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
        logger.error(f"WebSocket error in call {call_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error", 
                "message": "Internal server error"
            })
        except:
            pass
    finally:
        connection_tracker.remove_connection(call_id)
        call.end_time = now_utc()
        await db_manager.save_call(call)
        logger.info(f"Call ended: {call_id} (Active: {connection_tracker.get_count()})")

# ------------------------------------------------------------------------------
# Basic routes and API endpoints (same as before)
# ------------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": connection_tracker.get_count(),
        "max_connections": Config.MAX_CONCURRENT_CONNECTIONS,
        "circuit_breakers": ai_engine.get_circuit_breaker_status(),
        "rate_limiter": rate_limiter.get_stats()
    }

if __name__ == "__main__":
    logger.info("Starting AI Voice Agent System (Final Production Version)")
    logger.info(f"Max concurrent connections: {Config.MAX_CONCURRENT_CONNECTIONS}")
    logger.info(f"Rate limit per IP: {Config.RATE_LIMIT_CONNECTIONS} connections per {Config.RATE_LIMIT_WINDOW}s")
    logger.info(f"Max audio size: {Config.MAX_AUDIO_SIZE / 1024 / 1024:.1f}MB per frame")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
