#!/usr/bin/env python3  
"""
Complete AI Voice Agent System - Production Ready & Fully Functional
Single file implementation with all components working out of the box.

Run: python main.py
Then open: http://localhost:8000

Requirements:
pip install fastapi uvicorn websockets aiosqlite pydantic httpx gtts openai sentence-transformers numpy
"""

import asyncio
import logging
import os
import json
import time
import base64
import tempfile
import hashlib
import re
import sqlite3
import io
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, AsyncGenerator
from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager
from enum import Enum
from dataclasses import dataclass
import threading
from collections import defaultdict, deque
import traceback
import csv
from io import StringIO

# Core dependencies
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import httpx
import aiosqlite

# Optional AI dependencies with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

class Settings:
    # Application
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "voice-agent-secret-key-change-in-production-32chars")
    CORS_ORIGINS = ["*"] if os.getenv("CORS_ORIGINS", "*") == "*" else os.getenv("CORS_ORIGINS", "").split(",")
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")  # if set, required for admin + KB mutating endpoints

    # Database
    DATABASE_PATH = os.getenv("DATABASE_PATH", "./voice_agent.db")

    # AI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_WHISPER_MODEL = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")

    # Audio
    MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE", "10485760"))  # 10MB
    AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

    # WebSocket
    MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "50"))
    WEBSOCKET_TIMEOUT = int(os.getenv("WEBSOCKET_TIMEOUT", "300"))

    # Rate limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))

    # Retrieval
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.78"))  # one global threshold

    # Cache
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

settings = Settings()

# ==============================================================================
# Data Models
# ==============================================================================

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class CallStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TRANSFERRED = "transferred"

class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=4000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None

    @validator('message')
    def validate_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Message cannot be empty')
        # Basic XSS prevention
        suspicious = ['<script', 'javascript:', 'data:', 'vbscript:', 'onload=']
        if any(pattern in v.lower() for pattern in suspicious):
            raise ValueError('Invalid characters in message')
        return v

class ChatResponse(BaseModel):
    message: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class KnowledgeItem(BaseModel):
    id: Optional[str] = None
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=5000)
    category: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[str] = None  # NEW
    usage_count: int = 0
    confidence_threshold: float = Field(settings.SIMILARITY_THRESHOLD, ge=0.0, le=1.0)
    active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class UnknownQuestion(BaseModel):
    id: Optional[str] = None
    call_id: str
    question: str
    suggested_answer: Optional[str] = None
    resolved: bool = False
    created_at: Optional[datetime] = None

class BaseResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

# ==============================================================================
# Utilities
# ==============================================================================

class TTLCache:
    """Simple TTL cache implementation."""

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None

            value, expire_time = self._cache[key]
            if time.time() > expire_time:
                del self._cache[key]
                return None

            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._cache) >= self.maxsize:
                self._cleanup()

            expire_time = time.time() + self.ttl
            self._cache[key] = (value, expire_time)

    def _cleanup(self):
        current_time = time.time()
        expired = [k for k, (_, exp) in self._cache.items() if current_time > exp]
        for k in expired:
            del self._cache[k]

        # Remove oldest if still too many
        if len(self._cache) >= self.maxsize:
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
            for k, _ in sorted_items[:len(sorted_items)//4]:
                del self._cache[k]

class CircuitBreaker:
    """Circuit breaker for handling failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True

            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half_open"
                    return True
                return False

            return True  # half_open

    def record_success(self):
        with self._lock:
            self.failure_count = 0
            self.state = "closed"

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self.active_connections: Dict[str, WebSocket] = {}
        self._lock = threading.Lock()

    def can_connect(self) -> bool:
        with self._lock:
            return len(self.active_connections) < self.max_connections

    def connect(self, session_id: str, websocket: WebSocket) -> bool:
        with self._lock:
            if not self.can_connect():
                return False
            self.active_connections[session_id] = websocket
            return True

    def disconnect(self, session_id: str):
        with self._lock:
            self.active_connections.pop(session_id, None)

    def get_connection_count(self) -> int:
        with self._lock:
            return len(self.active_connections)

class RateLimiter:
    """Simple rate limiter."""

    def __init__(self, max_requests: int = 100, window: int = 3600):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        with self._lock:
            now = time.time()
            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window
            ]

            if len(self.requests[client_id]) >= self.max_requests:
                return False

            self.requests[client_id].append(now)
            return True

# ==============================================================================
# Database Manager
# ==============================================================================

class DatabaseManager:
    """Async database manager with SQLite."""

    def __init__(self, db_path: str = "./voice_agent.db"):
        self.db_path = db_path
        self._initialized = False

    async def initialize(self):
        """Initialize database and create tables."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            # Create tables
            await db.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    id TEXT PRIMARY KEY,
                    session_id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    transcript TEXT DEFAULT '',
                    confidence_score REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    category TEXT,
                    source TEXT,
                    tags TEXT,                         -- NEW
                    usage_count INTEGER DEFAULT 0,
                    confidence_threshold REAL DEFAULT 0.75,
                    active BOOLEAN DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    embedding BLOB                      -- NEW
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS unknown_questions (
                    id TEXT PRIMARY KEY,
                    call_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    suggested_answer TEXT,
                    resolved BOOLEAN DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (call_id) REFERENCES calls (id)
                )
            """)

            # Add missing columns (migrations)
            try:
                await db.execute("ALTER TABLE knowledge_items ADD COLUMN tags TEXT")
            except Exception:
                pass
            try:
                await db.execute("ALTER TABLE knowledge_items ADD COLUMN embedding BLOB")
            except Exception:
                pass

            # Indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_calls_session ON calls(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_active ON knowledge_items(active)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_unknown_resolved ON unknown_questions(resolved)")
            await db.commit()

        self._initialized = True
        logger.info("Database initialized")

    async def create_call(self, session_id: str) -> str:
        call_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO calls (id, session_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (call_id, session_id, CallStatus.ACTIVE.value, now, now))
            await db.commit()

        return call_id

    async def update_call(self, call_id: str, **updates) -> bool:
        if not updates:
            return False

        set_clauses = []
        values = []
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)

        set_clauses.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(call_id)

        query = f"UPDATE calls SET {', '.join(set_clauses)} WHERE id = ?"

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(query, values)
            await db.commit()

        return True

    async def add_knowledge_item_raw(self, item: KnowledgeItem, embedding_bytes: Optional[bytes]) -> str:
        item_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO knowledge_items
                    (id, question, answer, category, source, tags, usage_count,
                     confidence_threshold, active, created_at, updated_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, 1, ?, ?, ?)
            """, (item_id, item.question, item.answer, item.category, item.source,
                  item.tags, item.confidence_threshold, now, now, embedding_bytes))
            await db.commit()

        return item_id

    async def get_knowledge_items(self, active_only: bool = True) -> List[KnowledgeItem]:
        query = "SELECT * FROM knowledge_items"
        params = []
        if active_only:
            query += " WHERE active = ?"
            params.append(1)
        query += " ORDER BY usage_count DESC, created_at DESC"

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        items = []
        for row in rows:
            items.append(KnowledgeItem(
                id=row['id'],
                question=row['question'],
                answer=row['answer'],
                category=row['category'],
                source=row['source'],
                tags=row['tags'],
                usage_count=row['usage_count'],
                confidence_threshold=row['confidence_threshold'],
                active=bool(row['active']),
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            ))
            # attach raw embedding for in-memory use (not part of pydantic model)
            setattr(items[-1], "_embedding_raw", row["embedding"])
        return items

    async def set_item_embedding(self, item_id: str, embedding_bytes: bytes):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE knowledge_items SET embedding = ?, updated_at = ? WHERE id = ?",
                (embedding_bytes, datetime.utcnow().isoformat(), item_id),
            )
            await db.commit()

    async def increment_usage(self, item_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE knowledge_items SET usage_count = usage_count + 1, updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), item_id))
            await db.commit()

    async def add_unknown_question(self, question: UnknownQuestion) -> str:
        question_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO unknown_questions (id, call_id, question, suggested_answer, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (question_id, question.call_id, question.question,
                  question.suggested_answer, now))
            await db.commit()

        return question_id

    async def get_unknown_questions(self, resolved: bool = False, limit: int = 100) -> List[UnknownQuestion]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM unknown_questions WHERE resolved = ?
                ORDER BY created_at DESC LIMIT ?
            """, (int(resolved), limit)) as cursor:
                rows = await cursor.fetchall()

        questions = []
        for row in rows:
            questions.append(UnknownQuestion(
                id=row['id'],
                call_id=row['call_id'],
                question=row['question'],
                suggested_answer=row['suggested_answer'],
                resolved=bool(row['resolved']),
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            ))

        return questions

    async def get_stats(self) -> Dict[str, Any]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM calls") as cursor:
                total_calls = (await cursor.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM calls WHERE status = ?",
                                  (CallStatus.ACTIVE.value,)) as cursor:
                active_calls = (await cursor.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM knowledge_items WHERE active = 1") as cursor:
                knowledge_items = (await cursor.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM unknown_questions WHERE resolved = 0") as cursor:
                unknown_questions = (await cursor.fetchone())[0]

        return {
            "total_calls": total_calls,
            "active_calls": active_calls,
            "knowledge_items": knowledge_items,
            "unknown_questions": unknown_questions,
            "uptime": time.time() - start_time
        }

# ==============================================================================
# AI Engine
# ==============================================================================

class AIEngine:
    """AI engine with multiple providers and fallbacks."""

    def __init__(self):
        self.openai_client = None
        self.embedding_model = None
        self.cache = TTLCache(maxsize=1000, ttl=settings.CACHE_TTL)
        self.circuit_breaker = CircuitBreaker()

        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            self.openai_client = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=60.0
            )
            logger.info("OpenAI client initialized")

        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")

    async def speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """Convert audio to text."""
        if not self.openai_client or not self.circuit_breaker.can_execute():
            return None

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)
                temp_file.flush()

            with open(temp_path, 'rb') as audio_file:
                response = await self.openai_client.audio.transcriptions.create(
                    model=settings.OPENAI_WHISPER_MODEL,
                    file=audio_file,
                    response_format="text"
                )

            self.circuit_breaker.record_success()
            return response.strip() if response else None

        except Exception as e:
            logger.error(f"STT error: {e}")
            self.circuit_breaker.record_failure()
            return None
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    async def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech."""
        if not GTTS_AVAILABLE:
            return None

        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(temp_file.name)
                with open(temp_file.name, 'rb') as audio_file:
                    audio_data = audio_file.read()
            Path(temp_file.name).unlink(missing_ok=True)
            return audio_data

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    async def generate_response(self, message: str, context: List[ChatMessage] = None) -> str:
        """Generate conversational response."""
        cache_key = f"response:{hashlib.md5(message.encode()).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        if self.openai_client and self.circuit_breaker.can_execute():
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant. Provide concise, accurate responses."}
                ]
                if context:
                    for msg in context[-5:]:
                        messages.append({"role": msg.role.value, "content": msg.content})
                messages.append({"role": "user", "content": message})

                response = await self.openai_client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200
                )

                result = response.choices[0].message.content.strip()
                self.circuit_breaker.record_success()
                self.cache.set(cache_key, result)
                return result

            except Exception as e:
                logger.error(f"OpenAI chat error: {e}")
                self.circuit_breaker.record_failure()

        # Fallback
        return self._fallback_response(message)

    def _fallback_response(self, message: str) -> str:
        msg_lower = message.lower()
        if any(word in msg_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "Hello! How can I help you today?"
        if any(word in msg_lower for word in ['thank', 'thanks']):
            return "You're welcome! Is there anything else I can help you with?"
        if any(word in msg_lower for word in ['bye', 'goodbye', 'see you']):
            return "Goodbye! Have a great day!"
        if any(word in msg_lower for word in ['help', 'support']):
            return "I'm here to help! What can I assist you with?"
        return "I understand. Let me connect you with someone who can help with that specific question."

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate text embedding."""
        if not EMBEDDINGS_AVAILABLE or not self.embedding_model:
            # Simple fallback embedding
            words = text.lower().split()
            embedding = [0.0] * 384
            for i, word in enumerate(words[:10]):
                h = hash(word) % 384
                embedding[h] += 1.0
            total = sum(embedding) or 1.0
            return [val / total for val in embedding]

        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Cosine similarity."""
        if not embedding1 or not embedding2:
            return 0.0
        dot = sum(a * b for a, b in zip(embedding1, embedding2))
        n1 = sum(a * a for a in embedding1) ** 0.5
        n2 = sum(b * b for b in embedding2) ** 0.5
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

# ==============================================================================
# Knowledge Manager
# ==============================================================================

def _pack_embedding(vec: List[float]) -> bytes:
    if not EMBEDDINGS_AVAILABLE:
        # store as json bytes when numpy isn't available
        return json.dumps(vec).encode("utf-8")
    return np.asarray(vec, dtype=np.float32).tobytes()

def _unpack_embedding(raw: Optional[bytes]) -> Optional[List[float]]:
    if raw is None:
        return None
    if not EMBEDDINGS_AVAILABLE:
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None
    try:
        return np.frombuffer(raw, dtype=np.float32).tolist()
    except Exception:
        return None

class KnowledgeManager:
    """Knowledge base manager with similarity search."""

    def __init__(self, db_manager: DatabaseManager, ai_engine: AIEngine):
        self.db_manager = db_manager
        self.ai_engine = ai_engine
        self.items_cache = TTLCache(maxsize=1000, ttl=1800)  # 30 minutes
        self.backfill_lock = asyncio.Lock()

    def _rich_text(self, item: KnowledgeItem) -> str:
        parts = [f"QUESTION: {item.question}", f"ANSWER: {item.answer}"]
        if item.category:
            parts.append(f"CATEGORY: {item.category}")
        if item.tags:
            parts.append(f"TAGS: {item.tags}")
        return "\n".join(parts)

    async def _ensure_embeddings_backfilled(self, items: List[KnowledgeItem]):
        """Backfill embeddings async if any items are missing them."""
        async with self.backfill_lock:
            tasks = []
            for it in items:
                if getattr(it, "_embedding_raw", None) is None:
                    tasks.append(self._compute_and_store_embedding(it))
            if tasks:
                await asyncio.gather(*tasks)

    async def _compute_and_store_embedding(self, item: KnowledgeItem):
        rich = self._rich_text(item)
        vec = await self.ai_engine.generate_embedding(rich) or []
        raw = _pack_embedding(vec)
        await self.db_manager.set_item_embedding(item.id, raw)
        setattr(item, "_embedding_raw", raw)

    async def find_answer(self, question: str) -> Tuple[str, float, Optional[str]]:
        """Find best answer."""
        items = self.items_cache.get("all_items")
        if items is None:
            items = await self.db_manager.get_knowledge_items()
            self.items_cache.set("all_items", items)
            # fire-and-forget backfill
            asyncio.create_task(self._ensure_embeddings_backfilled(items))

        if not items:
            return "", 0.0, None

        q_vec = await self.ai_engine.generate_embedding(question) or []

        best_item = None
        best_sim = 0.0

        q_lower = question.lower()

        for item in items:
            if not item.active:
                continue

            raw = getattr(item, "_embedding_raw", None)
            if raw is None:
                # compute in-line (and persist later via background)
                rich = self._rich_text(item)
                vec = await self.ai_engine.generate_embedding(rich) or []
                sim = self.ai_engine.calculate_similarity(q_vec, vec)
            else:
                vec = _unpack_embedding(raw) or []
                sim = self.ai_engine.calculate_similarity(q_vec, vec)

            # small tag keyword boost (non-destructive)
            boost = 0.0
            if item.tags:
                for t in [x.strip() for x in item.tags.split(",")]:
                    if t and t in q_lower:
                        boost += 0.02
            if item.category and item.category.lower() in q_lower:
                boost += 0.02
            sim += boost

            if sim > best_sim:
                best_sim = sim
                best_item = item

        if best_item and best_sim >= settings.SIMILARITY_THRESHOLD:
            await self.db_manager.increment_usage(best_item.id)
            return best_item.answer, best_sim, best_item.id

        return "", 0.0, None

    async def add_item(self, item: KnowledgeItem) -> str:
        """Add new knowledge item with embedding."""
        rich = self._rich_text(item)
        vec = await self.ai_engine.generate_embedding(rich) or []
        raw = _pack_embedding(vec)
        item_id = await self.db_manager.add_knowledge_item_raw(item, raw)
        self.items_cache.set("all_items", None)
        return item_id

# ==============================================================================
# Global Services
# ==============================================================================

start_time = time.time()
db_manager = DatabaseManager(settings.DATABASE_PATH)
ai_engine = AIEngine()
knowledge_manager = KnowledgeManager(db_manager, ai_engine)
connection_manager = ConnectionManager(settings.MAX_CONNECTIONS)
rate_limiter = RateLimiter(settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW)

# ==============================================================================
# FastAPI Application
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Voice Agent System...")
    await db_manager.initialize()

    # Seed defaults if empty
    items = await db_manager.get_knowledge_items()
    if not items:
        default_items = [
            KnowledgeItem(
                question="What services do you offer?",
                answer="We offer AI-powered customer service with voice and chat capabilities. Our system can handle inquiries, provide information, and transfer to human agents when needed.",
                category="general",
                source="default",
                tags="services,voice,chat"
            ),
            KnowledgeItem(
                question="How does the voice system work?",
                answer="Our voice system uses advanced AI to convert your speech to text, understand your question, and provide spoken responses. Just click the microphone button and start talking!",
                category="technical",
                source="default",
                tags="voice,stt,tts"
            ),
            KnowledgeItem(
                question="Can I speak to a human?",
                answer="Of course! While our AI handles many questions effectively, you can always request to speak with a human agent. Just say 'I want to speak to a person' or similar.",
                category="support",
                source="default",
                tags="human,transfer"
            )
        ]
        for item in default_items:
            await knowledge_manager.add_item(item)
        logger.info("Added default knowledge items")

    logger.info("System startup complete")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="AI Voice Agent System",
    description="Production-ready AI voice agent with comprehensive features",
    version="2.0.1",
    lifespan=lifespan
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Auth dependency for admin/KB mutation (optional)
# ==============================================================================

def require_admin(x_admin_token: Optional[str] = Header(None)):
    if settings.ADMIN_TOKEN:
        if not x_admin_token or x_admin_token != settings.ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# ==============================================================================
# WebSocket Endpoint
# ==============================================================================

@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """Voice WebSocket endpoint."""
    await websocket.accept()

    session_id = str(uuid4())

    if not connection_manager.can_connect():
        await websocket.send_json({
            "type": "error",
            "message": "Server at maximum capacity. Please try again later."
        })
        await websocket.close(code=1013)
        return

    connection_manager.connect(session_id, websocket)

    call_id = None
    transcript: List[str] = []

    try:
        call_id = await db_manager.create_call(session_id)
        logger.info(f"Voice session started: {session_id}")

        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to voice agent"
        })

        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=settings.WEBSOCKET_TIMEOUT)

                # Disconnection
                if msg.get("type") == "websocket.disconnect":
                    break

                # Bytes (audio)
                if "bytes" in msg and msg["bytes"] is not None:
                    audio_data = msg["bytes"]

                    if len(audio_data) > settings.MAX_AUDIO_SIZE:
                        await websocket.send_json({"type": "error", "message": "Audio data too large"})
                        continue

                    await websocket.send_json({"type": "processing", "message": "Processing speech..."})
                    user_text = await ai_engine.speech_to_text(audio_data)

                    if not user_text:
                        await websocket.send_json({"type": "error", "message": "Could not process speech. Please try again."})
                        continue

                    transcript.append(f"User: {user_text}")
                    # Answer
                    answer, confidence, item_id = await knowledge_manager.find_answer(user_text)
                    if confidence >= settings.SIMILARITY_THRESHOLD:
                        response_text = answer
                        source = "knowledge_base"
                    else:
                        response_text = await ai_engine.generate_response(user_text, [ChatMessage(role=MessageRole.USER, content=user_text)])
                        source = "ai_assistant"
                        # Record unknown for training
                        try:
                            await db_manager.add_unknown_question(UnknownQuestion(call_id=call_id, question=user_text, suggested_answer=response_text))
                        except Exception:
                            pass

                    transcript.append(f"Assistant: {response_text}")

                    audio_response = await ai_engine.text_to_speech(response_text)
                    payload = {
                        "type": "response",
                        "text": response_text,
                        "user_text": user_text,
                        "confidence": float(confidence),
                        "source": source,
                        "session_id": session_id
                    }
                    if audio_response:
                        payload["audio_b64"] = base64.b64encode(audio_response).decode()
                        payload["audio_mime"] = "audio/mpeg"

                    await websocket.send_json(payload)

                # Text (JSON control or manual text)
                elif "text" in msg and msg["text"] is not None:
                    raw = msg["text"].strip()
                    user_message = None
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict) and data.get("type") == "text" and data.get("text"):
                            user_message = str(data["text"]).strip()
                        elif isinstance(data, dict) and data.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                            continue
                        else:
                            # unknown JSON -> ignore
                            continue
                    except json.JSONDecodeError:
                        # treat as plain text
                        user_message = raw

                    if not user_message:
                        continue

                    transcript.append(f"User: {user_message}")

                    if any(phrase in user_message.lower() for phrase in
                           ['human', 'person', 'agent', 'representative', 'speak to someone']):
                        await websocket.send_json({"type": "transfer", "message": "I'll connect you with a human agent right away."})
                        await db_manager.update_call(call_id, status=CallStatus.TRANSFERRED.value, transcript="\n".join(transcript))
                        break

                    answer, confidence, item_id = await knowledge_manager.find_answer(user_message)
                    if confidence >= settings.SIMILARITY_THRESHOLD:
                        response_text = answer
                        source = "knowledge_base"
                    else:
                        response_text = await ai_engine.generate_response(user_message, [ChatMessage(role=MessageRole.USER, content=user_message)])
                        source = "ai_assistant"
                        try:
                            await db_manager.add_unknown_question(UnknownQuestion(call_id=call_id, question=user_message, suggested_answer=response_text))
                        except Exception:
                            pass

                    transcript.append(f"Assistant: {response_text}")

                    audio_data = await ai_engine.text_to_speech(response_text)
                    payload = {
                        "type": "response",
                        "text": response_text,
                        "confidence": float(confidence),
                        "source": source,
                        "session_id": session_id
                    }
                    if audio_data:
                        payload["audio_b64"] = base64.b64encode(audio_data).decode()
                        payload["audio_mime"] = "audio/mpeg"
                    await websocket.send_json(payload)

            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()})
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                await websocket.send_json({"type": "error", "message": "An error occurred processing your request"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())
    finally:
        connection_manager.disconnect(session_id)
        try:
            if call_id:
                await db_manager.update_call(call_id, status=CallStatus.COMPLETED.value,
                                             transcript="\n".join(transcript) if transcript else "")
        except Exception as e:
            logger.error(f"Error updating call record: {e}")
        logger.info(f"Voice session ended: {session_id}")

# ==============================================================================
# HTTP API Endpoints
# ==============================================================================

@app.get("/")
async def get_index():
    """Serve the main application interface."""
    # (UNCHANGED UI CONTENT)
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Voice Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; display: flex; align-items: center; justify-content: center; padding: 20px; }
        .container { max-width: 800px; width: 100%; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { color: white; font-size: 3rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { color: rgba(255,255,255,0.9); font-size: 1.2rem; }
        .main-card { background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2); }
        .voice-interface { text-align: center; }
        .microphone-button { width: 150px; height: 150px; border-radius: 50%; border: none; background: linear-gradient(45deg, #ff6b6b, #ee5a52); color: white; font-size: 3rem; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 10px 30px rgba(238, 90, 82, 0.3); margin: 20px auto; display: flex; align-items: center; justify-content: center; }
        .microphone-button:hover { transform: translateY(-3px); box-shadow: 0 15px 40px rgba(238, 90, 82, 0.4); }
        .microphone-button.active { background: linear-gradient(45deg, #51cf66, #40c057); animation: pulse 2s infinite; }
        .microphone-button.disabled { background: #ccc; cursor: not-allowed; transform: none; box-shadow: none; }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
        .status { margin: 20px 0; padding: 15px; border-radius: 10px; font-weight: 500; min-height: 50px; display: flex; align-items: center; justify-content: center; }
        .status.disconnected { background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
        .status.connected { background: #e3f2fd; color: #1976d2; border: 1px solid #bbdefb; }
        .status.listening { background: #f3e5f5; color: #7b1fa2; border: 1px solid #e1bee7; }
        .status.processing { background: #fff3e0; color: #f57c00; border: 1px solid #ffcc02; }
        .status.error { background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
        .transcript { background: #f8f9fa; border-radius: 15px; padding: 20px; margin: 20px 0; min-height: 300px; max-height: 400px; overflow-y: auto; border: 1px solid #e9ecef; text-align: left; }
        .message { margin: 15px 0; padding: 12px 16px; border-radius: 12px; max-width: 85%; word-wrap: break-word; }
        .message.user { background: #e3f2fd; margin-left: auto; text-align: right; }
        .message.assistant { background: #e8f5e8; margin-right: auto; }
        .message .label { font-weight: bold; margin-bottom: 5px; }
        .message .confidence { font-size: 0.8rem; opacity: 0.7; margin-top: 5px; }
        .controls { display: flex; gap: 15px; justify-content: center; margin-top: 30px; flex-wrap: wrap; }
        .btn { padding: 12px 24px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.3s ease; font-size: 1rem; }
        .btn-primary { background: #667eea; color: white; }
        .btn-secondary { background: #6c757d; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; box-shadow: none; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; text-align: center; }
        .stat { background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(102, 126, 234, 0.2); }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #667eea; }
        .stat-label { font-size: 0.9rem; color: #666; margin-top: 5px; }
        .admin-link { position: fixed; top: 20px; right: 20px; background: rgba(255,255,255,0.2); color: white; padding: 10px 20px; border-radius: 20px; text-decoration: none; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.3); transition: all 0.3s ease; }
        .admin-link:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
        @media (max-width: 768px) {
            .header h1 { font-size: 2rem; }
            .microphone-button { width: 120px; height: 120px; font-size: 2rem; }
            .main-card { padding: 20px; }
        }
        .hidden { display: none; }
    </style>
</head>
<body>
    <a href="/admin" class="admin-link">Admin Panel</a>
    <div class="container">
        <div class="header">
            <h1>🎤 AI Voice Agent</h1>
            <p>Advanced AI-powered customer service system</p>
        </div>
        <div class="main-card">
            <div class="voice-interface">
                <div id="status" class="status disconnected">Click Connect to start</div>
                <button id="micButton" class="microphone-button disabled" disabled>🎤</button>
                <div class="transcript" id="transcript">
                    <div style="text-align: center; color: #666; margin-top: 100px;">Your conversation will appear here...</div>
                </div>
                <div class="controls">
                    <button id="connectBtn" class="btn btn-primary">Connect</button>
                    <button id="disconnectBtn" class="btn btn-danger" disabled>Disconnect</button>
                    <button id="clearBtn" class="btn btn-secondary">Clear Chat</button>
                </div>
                <div class="stats" id="stats">
                    <div class="stat">
                        <div class="stat-value" id="activeConnections">-</div>
                        <div class="stat-label">Active Connections</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="totalCalls">-</div>
                        <div class="stat-label">Total Calls</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="knowledgeItems">-</div>
                        <div class="stat-label">Knowledge Items</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        class VoiceAgent {
            constructor() {
                this.websocket = null;
                this.isConnected = false;
                this.isListening = false;
                this.mediaRecorder = null;
                this.audioChunks = [];
                this.sessionId = null;

                this.initializeElements();
                this.bindEvents();
                this.loadStats();
                setInterval(() => this.loadStats(), 30000);
            }
            initializeElements() {
                this.elements = {
                    status: document.getElementById('status'),
                    micButton: document.getElementById('micButton'),
                    connectBtn: document.getElementById('connectBtn'),
                    disconnectBtn: document.getElementById('disconnectBtn'),
                    clearBtn: document.getElementById('clearBtn'),
                    transcript: document.getElementById('transcript'),
                    activeConnections: document.getElementById('activeConnections'),
                    totalCalls: document.getElementById('totalCalls'),
                    knowledgeItems: document.getElementById('knowledgeItems')
                };
            }
            bindEvents() {
                this.elements.connectBtn.addEventListener('click', () => this.connect());
                this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
                this.elements.micButton.addEventListener('click', () => this.toggleRecording());
                this.elements.clearBtn.addEventListener('click', () => this.clearTranscript());
            }
            async connect() {
                if (this.isConnected) return;
                try {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws/voice`;
                    this.websocket = new WebSocket(wsUrl);

                    this.websocket.onopen = () => {
                        this.isConnected = true;
                        this.updateStatus('Connected', 'connected');
                        this.updateButtons();
                    };
                    this.websocket.onmessage = (event) => {
                        this.handleMessage(JSON.parse(event.data));
                    };
                    this.websocket.onclose = () => {
                        this.isConnected = false;
                        this.isListening = false;
                        this.updateStatus('Disconnected', 'disconnected');
                        this.updateButtons();
                        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                            this.mediaRecorder.stop();
                        }
                    };
                    this.websocket.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        this.updateStatus('Connection error', 'error');
                    };

                } catch (error) {
                    console.error('Connection error:', error);
                    this.updateStatus('Failed to connect', 'error');
                }
            }
            disconnect() {
                if (this.websocket) this.websocket.close();
                if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') this.mediaRecorder.stop();
            }
            async toggleRecording() {
                if (!this.isConnected) { alert('Please connect first'); return; }
                if (this.isListening) { this.stopRecording(); } else { await this.startRecording(); }
            }
            async startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
                    });
                    this.mediaRecorder = new MediaRecorder(stream, {
                        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm'
                    });
                    this.audioChunks = [];
                    this.mediaRecorder.ondataavailable = (event) => { if (event.data.size > 0) this.audioChunks.push(event.data); };
                    this.mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                        this.sendAudioData(audioBlob);
                        stream.getTracks().forEach(track => track.stop());
                    };
                    this.mediaRecorder.start();
                    this.isListening = true;
                    this.updateStatus('Listening... (Click to stop)', 'listening');
                    this.updateButtons();
                    setTimeout(() => { if (this.isListening) this.stopRecording(); }, 10000);
                } catch (error) {
                    console.error('Error starting recording:', error);
                    this.updateStatus('Microphone access denied', 'error');
                    alert('Please allow microphone access and try again.');
                }
            }
            stopRecording() {
                if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                    this.mediaRecorder.stop();
                    this.isListening = false;
                    this.updateStatus('Processing...', 'processing');
                    this.updateButtons();
                }
            }
            async sendAudioData(audioBlob) {
                if (!this.isConnected || !this.websocket) return;
                try {
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    this.websocket.send(arrayBuffer);
                } catch (error) {
                    console.error('Error sending audio:', error);
                    this.updateStatus('Error sending audio', 'error');
                }
            }
            handleMessage(data) {
                switch (data.type) {
                    case 'connected':
                        this.sessionId = data.session_id;
                        this.updateStatus('Connected - Ready to chat!', 'connected');
                        break;
                    case 'response':
                        this.addMessage('assistant', data.text, data.confidence, data.source);
                        if (data.user_text) this.addMessage('user', data.user_text);
                        this.updateStatus('Connected - Ready to chat!', 'connected');
                        if (data.audio_b64 && data.audio_mime) this.playAudio(data.audio_b64, data.audio_mime);
                        break;
                    case 'processing':
                        this.updateStatus(data.message || 'Processing...', 'processing');
                        break;
                    case 'transfer':
                        this.updateStatus('Transferring to human agent...', 'processing');
                        this.addMessage('system', data.message);
                        setTimeout(() => { this.disconnect(); }, 2000);
                        break;
                    case 'error':
                        this.updateStatus(data.message || 'Error occurred', 'error');
                        break;
                    case 'heartbeat':
                        break;
                    default:
                        console.log('Unknown message type:', data.type);
                }
            }
            addMessage(sender, text, confidence = null, source = null) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                let content = `<div class="label">${sender === 'user' ? 'You' : sender === 'assistant' ? 'Assistant' : 'System'}:</div>`;
                content += `<div>${text}</div>`;
                if (confidence !== null && sender === 'assistant') {
                    content += `<div class="confidence">Confidence: ${(confidence * 100).toFixed(1)}% | Source: ${source || 'unknown'}</div>`;
                }
                messageDiv.innerHTML = content;
                this.elements.transcript.appendChild(messageDiv);
                this.elements.transcript.scrollTop = this.elements.transcript.scrollHeight;
            }
            playAudio(audioB64, mimeType) {
                try { new Audio(`data:${mimeType};base64,${audioB64}`).play().catch(e => console.error('Error playing audio:', e)); }
                catch (error) { console.error('Error creating audio:', error); }
            }
            updateStatus(message, className) {
                this.elements.status.textContent = message;
                this.elements.status.className = `status ${className}`;
            }
            updateButtons() {
                this.elements.connectBtn.disabled = this.isConnected;
                this.elements.connectBtn.textContent = this.isConnected ? 'Connected' : 'Connect';
                this.elements.disconnectBtn.disabled = !this.isConnected;
                this.elements.micButton.disabled = !this.isConnected;
                this.elements.micButton.className = `microphone-button ${!this.isConnected ? 'disabled' : this.isListening ? 'active' : ''}`;
                this.elements.micButton.textContent = this.isListening ? '⏹️' : '🎤';
            }
            clearTranscript() {
                this.elements.transcript.innerHTML = `<div style="text-align: center; color: #666; margin-top: 100px;">Your conversation will appear here...</div>`;
            }
            async loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    this.elements.activeConnections.textContent = stats.active_calls || 0;
                    this.elements.totalCalls.textContent = stats.total_calls || 0;
                    this.elements.knowledgeItems.textContent = stats.knowledge_items || 0;
                } catch (error) { console.error('Error loading stats:', error); }
            }
        }
        document.addEventListener('DOMContentLoaded', () => { new VoiceAgent(); });
    </script>
</body>
</html>""")

@app.get("/admin")
async def get_admin():
    """Admin panel interface."""
    # (UNCHANGED UI CONTENT)
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Panel - AI Voice Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }
        .header { background: #667eea; color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 1.5rem; }
        .back-link { color: white; text-decoration: none; padding: 0.5rem 1rem; border: 1px solid white; border-radius: 4px; transition: background 0.3s; }
        .back-link:hover { background: rgba(255,255,255,0.1); }
        .container { max-width: 1200px; margin: 2rem auto; padding: 0 2rem; }
        .card { background: white; border-radius: 8px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .card h2 { margin-bottom: 1rem; color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem; }
        .form-group { margin-bottom: 1rem; }
        label { display: block; margin-bottom: 0.5rem; font-weight: 600; }
        input, textarea, select { width: 100%; padding: 0.75rem; border: 1px solid #ddd; border-radius: 4px; font-size: 1rem; }
        textarea { resize: vertical; min-height: 100px; }
        .btn { background: #667eea; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 4px; cursor: pointer; font-size: 1rem; transition: background 0.3s; }
        .btn:hover { background: #5a6fd8; }
        .btn-secondary { background: #6c757d; } .btn-secondary:hover { background: #5a6268; }
        .btn-danger { background: #dc3545; } .btn-danger:hover { background: #c82333; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .stat-card { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 1.5rem; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem; }
        .stat-label { font-size: 0.9rem; opacity: 0.9; }
        .table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        .table th, .table td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #ddd; }
        .table th { background: #f8f9fa; font-weight: 600; }
        .table tr:hover { background: #f8f9fa; }
        .message { padding: 1rem; margin: 1rem 0; border-radius: 4px; }
        .message.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .loading { text-align: center; color: #666; font-style: italic; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        @media (max-width: 768px) { .container { padding: 0 1rem; } .grid-2 { grid-template-columns: 1fr; } .header { padding: 1rem; } .header h1 { font-size: 1.2rem; } }
    </style>
</head>
<body>
    <div class="header"><h1>Admin Panel</h1><a href="/" class="back-link">← Back to Voice Agent</a></div>
    <div class="container">
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card"><div class="stat-value" id="totalCalls">-</div><div class="stat-label">Total Calls</div></div>
            <div class="stat-card"><div class="stat-value" id="activeCalls">-</div><div class="stat-label">Active Calls</div></div>
            <div class="stat-card"><div class="stat-value" id="knowledgeItems">-</div><div class="stat-label">Knowledge Items</div></div>
            <div class="stat-card"><div class="stat-value" id="unknownQuestions">-</div><div class="stat-label">Unknown Questions</div></div>
        </div>
        <div class="grid-2">
            <div class="card">
                <h2>Add Knowledge Item</h2>
                <form id="addKnowledgeForm">
                    <div class="form-group"><label for="question">Question:</label><input type="text" id="question" name="question" required maxlength="2000" placeholder="Enter the question customers might ask..."></div>
                    <div class="form-group"><label for="answer">Answer:</label><textarea id="answer" name="answer" required maxlength="5000" placeholder="Enter the response the AI should provide..."></textarea></div>
                    <div class="form-group"><label for="category">Category (optional):</label><input type="text" id="category" name="category" maxlength="100" placeholder="e.g., technical, billing, general"></div>
                    <div class="form-group"><label for="source">Source (optional):</label><input type="text" id="source" name="source" maxlength="255" placeholder="e.g., manual, imported, training"></div>
                    <div class="form-group"><label for="tags">Tags (comma-separated, optional):</label><input type="text" id="tags" name="tags" maxlength="255" placeholder="e.g., chairs, blue, chargers"></div>
                    <div class="form-group"><label for="confidence">Confidence Threshold:</label><input type="number" id="confidence" name="confidence" min="0" max="1" step="0.01" value="0.78"></div>
                    <button type="submit" class="btn">Add Knowledge Item</button>
                </form>
                <div id="addMessage"></div>
            </div>
            <div class="card">
                <h2>Bulk Import (CSV)</h2>
                <p style="margin-bottom: 1rem; color: #666; font-size: 0.9rem;">
                    Upload a CSV file with columns: question, answer, category, source, tags<br>First row should contain headers.
                </p>
                <div class="form-group"><label for="csvFile">Select CSV File:</label><input type="file" id="csvFile" accept=".csv" /></div>
                <button id="importBtn" class="btn">Import CSV</button>
                <div id="importMessage"></div>
            </div>
        </div>
        <div class="card">
            <h2>Unknown Questions</h2>
            <p style="margin-bottom: 1rem; color: #666;">Questions that couldn't be answered from the knowledge base.</p>
            <div id="unknownQuestionsContainer"><div class="loading">Loading unknown questions...</div></div>
        </div>
        <div class="card">
            <h2>Knowledge Base Items</h2>
            <p style="margin-bottom: 1rem; color: #666;">Current knowledge base items ordered by usage count.</p>
            <div id="knowledgeItemsContainer"><div class="loading">Loading knowledge items...</div></div>
        </div>
    </div>
    <script>
        class AdminPanel {
            constructor() {
                this.bindEvents();
                this.loadStats();
                this.loadUnknownQuestions();
                this.loadKnowledgeItems();
                setInterval(() => { this.loadStats(); this.loadUnknownQuestions(); }, 30000);
            }
            bindEvents() {
                document.getElementById('addKnowledgeForm').addEventListener('submit', (e) => { e.preventDefault(); this.addKnowledgeItem(); });
                document.getElementById('importBtn').addEventListener('click', () => { this.importCSV(); });
            }
            async headers() {
                const h = { 'Content-Type': 'application/json' };
                const token = %s ? %s : null;
                if (token) h['X-Admin-Token'] = token;
                return h;
            }
            async loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    document.getElementById('totalCalls').textContent = stats.total_calls || 0;
                    document.getElementById('activeCalls').textContent = stats.active_calls || 0;
                    document.getElementById('knowledgeItems').textContent = stats.knowledge_items || 0;
                    document.getElementById('unknownQuestions').textContent = stats.unknown_questions || 0;
                } catch (error) { console.error('Error loading stats:', error); }
            }
            async addKnowledgeItem() {
                const formData = new FormData(document.getElementById('addKnowledgeForm'));
                const data = {
                    question: (formData.get('question') || '').trim(),
                    answer: (formData.get('answer') || '').trim(),
                    category: (formData.get('category') || '').trim() || null,
                    source: (formData.get('source') || '').trim() || null,
                    tags: (formData.get('tags') || '').trim() || null,
                    confidence_threshold: parseFloat(formData.get('confidence') || '0.78')
                };
                if (!data.question || !data.answer) { this.showMessage('addMessage', 'Question and answer are required', 'error'); return; }
                try {
                    const response = await fetch('/api/knowledge/add', { method: 'POST', headers: await this.headers(), body: JSON.stringify(data) });
                    const result = await response.json();
                    if (response.ok) {
                        this.showMessage('addMessage', 'Knowledge item added successfully!', 'success');
                        document.getElementById('addKnowledgeForm').reset();
                        document.getElementById('confidence').value = '0.78';
                        this.loadStats(); this.loadKnowledgeItems();
                    } else {
                        this.showMessage('addMessage', result.detail || 'Error adding knowledge item', 'error');
                    }
                } catch (error) { console.error('Error adding knowledge item:', error); this.showMessage('addMessage', 'Network error occurred', 'error'); }
            }
            async importCSV() {
                const fileInput = document.getElementById('csvFile');
                const file = fileInput.files[0];
                if (!file) { this.showMessage('importMessage', 'Please select a CSV file', 'error'); return; }
                const text = await file.text();
                try {
                    const response = await fetch('/api/knowledge/bulk', { method: 'POST', headers: await this.headers(), body: JSON.stringify({ csv_data: text }) });
                    const result = await response.json();
                    if (response.ok) {
                        this.showMessage('importMessage', `Successfully imported ${result.data?.added || 0} items`, 'success');
                        fileInput.value = ''; this.loadStats(); this.loadKnowledgeItems();
                    } else {
                        this.showMessage('importMessage', result.detail || 'Import failed', 'error');
                    }
                } catch (error) { console.error('Error importing CSV:', error); this.showMessage('importMessage', 'Network error occurred', 'error'); }
            }
            async loadUnknownQuestions() {
                try {
                    const response = await fetch('/api/knowledge/unknown');
                    const questions = await response.json();
                    const container = document.getElementById('unknownQuestionsContainer');
                    if (!questions.length) { container.innerHTML = '<p>No unknown questions at this time.</p>'; return; }
                    let html = '<table class="table"><thead><tr><th>Question</th><th>Suggested Answer</th><th>Created</th><th>Actions</th></tr></thead><tbody>';
                    questions.forEach(q => {
                        const createdDate = q.created_at ? new Date(q.created_at).toLocaleString() : 'Unknown';
                        html += `<tr>
                            <td style="max-width: 300px; word-wrap: break-word;">${this.escapeHtml(q.question)}</td>
                            <td style="max-width: 300px; word-wrap: break-word;">${this.escapeHtml(q.suggested_answer || 'No suggestion')}</td>
                            <td>${createdDate}</td>
                            <td><button class="btn btn-secondary" onclick="adminPanel.resolveQuestion('${q.id}', '${this.escapeHtml(q.question)}')">Resolve</button></td>
                        </tr>`;
                    });
                    html += '</tbody></table>';
                    container.innerHTML = html;
                } catch (error) { console.error('Error loading unknown questions:', error); document.getElementById('unknownQuestionsContainer').innerHTML = '<p class="error">Error loading unknown questions</p>'; }
            }
            async loadKnowledgeItems() {
                try {
                    const response = await fetch('/api/knowledge/items');
                    const items = await response.json();
                    const container = document.getElementById('knowledgeItemsContainer');
                    if (!items.length) { container.innerHTML = '<p>No knowledge items found.</p>'; return; }
                    let html = '<table class="table"><thead><tr><th>Question</th><th>Answer</th><th>Category</th><th>Tags</th><th>Usage</th><th>Actions</th></tr></thead><tbody>';
                    items.slice(0, 50).forEach(item => {
                        html += `<tr>
                            <td style="max-width: 250px; word-wrap: break-word;">${this.escapeHtml(item.question)}</td>
                            <td style="max-width: 300px; word-wrap: break-word;">${this.escapeHtml(item.answer)}</td>
                            <td>${this.escapeHtml(item.category || 'General')}</td>
                            <td>${this.escapeHtml(item.tags || '')}</td>
                            <td>${item.usage_count}</td>
                            <td><button class="btn btn-danger btn-sm" onclick="adminPanel.deleteItem('${item.id}')">Delete</button></td>
                        </tr>`;
                    });
                    html += '</tbody></table>';
                    if (items.length > 50) html += `<p style="margin-top: 1rem; color: #666;">Showing first 50 of ${items.length} items</p>`;
                    container.innerHTML = html;
                } catch (error) { console.error('Error loading knowledge items:', error); document.getElementById('knowledgeItemsContainer').innerHTML = '<p class="error">Error loading knowledge items</p>'; }
            }
            async resolveQuestion(questionId, questionText) {
                const answer = prompt(`Please provide an answer for:\n\n"${questionText}"`);
                if (!answer) return;
                try {
                    const response = await fetch('/api/knowledge/resolve', { method: 'POST', headers: await this.headers(), body: JSON.stringify({ question_id: questionId, answer: answer }) });
                    const result = await response.json();
                    if (response.ok) { alert('Question resolved and added to knowledge base!'); this.loadStats(); this.loadUnknownQuestions(); this.loadKnowledgeItems(); }
                    else { alert(result.detail || 'Error resolving question'); }
                } catch (error) { console.error('Error resolving question:', error); alert('Network error occurred'); }
            }
            async deleteItem(itemId) {
                if (!confirm('Are you sure you want to delete this knowledge item?')) return;
                try {
                    const response = await fetch(`/api/knowledge/items/${itemId}`, { method: 'DELETE', headers: await this.headers() });
                    if (response.ok) { alert('Knowledge item deleted!'); this.loadStats(); this.loadKnowledgeItems(); }
                    else { alert('Error deleting item'); }
                } catch (error) { console.error('Error deleting item:', error); alert('Network error occurred'); }
            }
            showMessage(containerId, message, type) { const c = document.getElementById(containerId); c.innerHTML = `<div class="message ${type}">${message}</div>`; setTimeout(() => { c.innerHTML = ''; }, 5000); }
            escapeHtml(text) { const div = document.createElement('div'); div.textContent = text || ''; return div.innerHTML; }
        }
        // Inject ADMIN_TOKEN value at render-time (not exposed unless set)
        const ADMIN_TOKEN = %r ? '%s' : null;
        const adminPanel = new AdminPanel();
        // Patch headers method with token
        AdminPanel.prototype.headers = async function() {
            const h = { 'Content-Type': 'application/json' };
            if (ADMIN_TOKEN) h['X-Admin-Token'] = ADMIN_TOKEN;
            return h;
        }
    </script>
</body>
</html>""" % (
        "false" if not settings.ADMIN_TOKEN else "true",
        "null" if not settings.ADMIN_TOKEN else "ADMIN_TOKEN",
        bool(settings.ADMIN_TOKEN),
        settings.ADMIN_TOKEN or ""
    ))

@app.post("/api/chat")
async def api_chat(request: Request, payload: ChatRequest):
    """Chat API endpoint."""
    try:
        client_ip = request.client.host if request.client else "default"
        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        context: List[ChatMessage] = []
        answer, confidence, item_id = await knowledge_manager.find_answer(payload.message)

        if confidence >= settings.SIMILARITY_THRESHOLD:
            response_text = answer
            source = "knowledge_base"
        else:
            response_text = await ai_engine.generate_response(payload.message, context)
            source = "ai_assistant"

        return ChatResponse(
            message=response_text,
            confidence=float(confidence),
            source=source,
            session_id=payload.session_id or str(uuid4())
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/knowledge/add", dependencies=[Depends(require_admin)])
async def api_add_knowledge(item: KnowledgeItem):
    """Add knowledge base item."""
    try:
        item_id = await knowledge_manager.add_item(item)
        return BaseResponse(success=True, message="Knowledge item added successfully", data={"id": item_id})
    except Exception as e:
        logger.error(f"Add knowledge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/bulk", dependencies=[Depends(require_admin)])
async def api_bulk_import(data: Dict[str, str]):
    """Bulk import knowledge items from CSV (robust)."""
    try:
        csv_data = data.get("csv_data", "")
        if not csv_data:
            raise HTTPException(status_code=400, detail="No CSV data provided")

        reader = csv.DictReader(StringIO(csv_data))
        if not reader.fieldnames:
            raise HTTPException(status_code=400, detail="CSV must include a header row")

        headers = [h.strip().lower() for h in reader.fieldnames]
        if "question" not in headers or "answer" not in headers:
            raise HTTPException(status_code=400, detail="CSV must include headers: question, answer (optional: category, source, tags)")

        added = 0
        for row in reader:
            q = (row.get("question") or row.get("Question") or "").strip()
            a = (row.get("answer") or row.get("Answer") or "").strip()
            if not q or not a:
                continue
            item = KnowledgeItem(
                question=q,
                answer=a,
                category=(row.get("category") or row.get("Category") or "").strip() or None,
                source=(row.get("source") or row.get("Source") or "").strip() or "bulk_import",
                tags=(row.get("tags") or row.get("Tags") or "").strip() or None,
                confidence_threshold=settings.SIMILARITY_THRESHOLD
            )
            await knowledge_manager.add_item(item)
            added += 1

        return BaseResponse(success=True, message=f"Successfully imported {added} items", data={"added": added})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk import error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge/items")
async def api_get_knowledge_items():
    try:
        items = await db_manager.get_knowledge_items()
        out = []
        for it in items:
            d = it.dict()
            out.append(d)
        return out
    except Exception as e:
        logger.error(f"Get knowledge items error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/knowledge/items/{item_id}", dependencies=[Depends(require_admin)])
async def api_delete_knowledge_item(item_id: str):
    try:
        async with aiosqlite.connect(settings.DATABASE_PATH) as db:
            await db.execute("UPDATE knowledge_items SET active = 0 WHERE id = ?", (item_id,))
            await db.commit()
        return BaseResponse(success=True, message="Item deleted successfully")
    except Exception as e:
        logger.error(f"Delete knowledge item error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge/unknown")
async def api_get_unknown_questions():
    try:
        questions = await db_manager.get_unknown_questions(resolved=False, limit=50)
        return [q.dict() for q in questions]
    except Exception as e:
        logger.error(f"Get unknown questions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/resolve", dependencies=[Depends(require_admin)])
async def api_resolve_question(data: Dict[str, str]):
    try:
        question_id = data.get("question_id")
        answer = (data.get("answer") or "").strip()
        if not question_id or not answer:
            raise HTTPException(status_code=400, detail="Question ID and answer required")

        async with aiosqlite.connect(settings.DATABASE_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM unknown_questions WHERE id = ?", (question_id,)) as cursor:
                row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Question not found")

        item = KnowledgeItem(
            question=row['question'],
            answer=answer,
            source="resolved",
            tags=None
        )
        await knowledge_manager.add_item(item)

        async with aiosqlite.connect(settings.DATABASE_PATH) as db:
            await db.execute("UPDATE unknown_questions SET resolved = 1 WHERE id = ?", (question_id,))
            await db.commit()

        return BaseResponse(success=True, message="Question resolved and added to knowledge base")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resolve question error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def api_get_stats():
    try:
        stats = await db_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Get stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def api_health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "connections": connection_manager.get_connection_count(),
        "uptime": time.time() - start_time
    }

# ==============================================================================
# Application Entry Point
# ==============================================================================

if __name__ == "__main__":
    logger.info("Starting AI Voice Agent System...")
    logger.info(f"OpenAI Available: {OPENAI_AVAILABLE}")
    logger.info(f"gTTS Available: {GTTS_AVAILABLE}")
    logger.info(f"Embeddings Available: {EMBEDDINGS_AVAILABLE}")
    if settings.ADMIN_TOKEN:
        logger.info("Admin auth enabled (X-Admin-Token)")
    else:
        logger.info("Admin auth disabled (set ADMIN_TOKEN to enable)")

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info" if not settings.DEBUG else "debug",
        access_log=settings.DEBUG
    )
