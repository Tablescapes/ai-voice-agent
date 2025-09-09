#!/usr/bin/env python3
"""
Self-Sufficient AI Voice Agent System - Lightweight Version (refactored)
- Async-safe (aiosqlite/httpx)
- Truthful audio formats & MIME
- Safer scraping (allowlist)
- Correct date stats
- Basic API key auth and input limits
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
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from uuid import uuid4
from pathlib import Path

import aiosqlite
import httpx
from bs4 import BeautifulSoup

from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

# Optional cloud LLM/STT/TTS
from openai import OpenAI  # v1 client
from gtts import gTTS

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
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

    # Audio
    SAMPLE_RATE = 16000
    AUDIO_TTS_MIME = "audio/mpeg"  # gTTS returns MP3
    STT_TARGET_RATE = 16000

    # KB / matching
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

    # Call control
    MAX_CALL_DURATION = int(os.getenv("MAX_CALL_DURATION", "600"))  # seconds

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # Security
    API_KEY = os.getenv("VOICE_AGENT_API_KEY")  # if set, required for /api/*
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "4000"))

    # Scraper safety
    ALLOWED_SCRAPE_HOSTS = set(
        h.strip().lower()
        for h in os.getenv("ALLOWED_SCRAPE_HOSTS", "tablescapes.com,www.tablescapes.com").split(",")
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
    # ISO date boundaries in UTC
    start = datetime.now(timezone.utc).date()
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()

def limit_len(s: str, maxlen: int) -> str:
    return s if len(s) <= maxlen else s[:maxlen]

# ------------------------------------------------------------------------------
# AI Engine
# ------------------------------------------------------------------------------
class LightweightAIEngine:
    def __init__(self):
        self.openai_client: Optional[OpenAI] = None
        if Config.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.info("OpenAI client initialized.")
        else:
            logger.info("No OPENAI_API_KEY set. Falling back to rule-based responses.")
        # nothing else to init

    async def convert_audio_to(self, data: bytes, target_sample_rate: int = 16000) -> Tuple[bytes, str]:
        """
        Try converting arbitrary audio (e.g., WEBM/Opus) to WAV 16k mono using ffmpeg.
        On failure, return original bytes and best-effort format hint 'webm'.
        """
        async def _run():
            with tempfile.TemporaryDirectory() as td:
                in_path = Path(td) / "in.webm"
                out_path = Path(td) / "out.wav"
                in_path.write_bytes(data)
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                            "-i", str(in_path), "-ar", str(target_sample_rate), "-ac", "1",
                            "-f", "wav", str(out_path), "-y"
                        ],
                        check=True
                    )
                    return out_path.read_bytes(), "wav"
                except subprocess.CalledProcessError:
                    # return original; tell caller it's webm
                    return in_path.read_bytes(), "webm"
        return await run_in_threadpool(_run)

    async def speech_to_text(self, audio_bytes: bytes) -> str:
        """
        Batch STT using OpenAI Whisper (or successor). Returns empty string on failure.
        """
        if not self.openai_client:
            return ""

        wav_bytes, fmt = await self.convert_audio_to(audio_bytes, Config.STT_TARGET_RATE)
        suffix = ".wav" if fmt == "wav" else ".webm"

        async def _transcribe():
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / f"audio{suffix}"
                p.write_bytes(wav_bytes)
                with p.open("rb") as f:
                    try:
                        # Prefer current small STT-capable models if available; fall back to whisper-1
                        # Some orgs have 'gpt-4o-mini-transcribe'. If not, keep whisper-1.
                        try_model = os.getenv("OPENAI_STT_MODEL", "whisper-1")
                        resp = self.openai_client.audio.transcriptions.create(
                            model=try_model, file=f
                        )
                        text = (resp.text or "").strip()
                        return text
                    except Exception as e:
                        logger.error(f"STT error: {e}")
                        return ""
        return await run_in_threadpool(_transcribe)

    async def text_to_speech(self, text: str) -> bytes:
        """
        TTS via gTTS (MP3). Wrapped in threadpool. Returns bytes or b"" on failure.
        """
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
        return await run_in_threadpool(_tts)

    async def generate_response(self, user_input: str, context: str = "") -> str:
        """
        LLM response or rule-based fallback.
        """
        user_input = limit_len(user_input, Config.MAX_INPUT_LEN)
        if self.openai_client:
            async def _chat():
                try:
                    resp = self.openai_client.chat.completions.create(
                        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                        messages=[
                            {"role": "system",
                             "content": f"You are a concise customer service agent for a tableware and event-rental business. Keep answers accurate and succinct. Context: {context}"},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=0.2,
                        max_tokens=220,
                    )
                    return (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    return ""
            text = await run_in_threadpool(_chat)
            if text:
                return text

        # fallback
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
        """
        Very simple hand-made 'embedding' (kept for zero-cost fallback).
        """
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

# ------------------------------------------------------------------------------
# Database (aiosqlite)
# ------------------------------------------------------------------------------
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_database(self):
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("PRAGMA busy_timeout=3000;")
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
                "SELECT * FROM unknown_questions WHERE resolved = ?", (int(resolved),)
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

    async def find_answer(self, question: str) -> Tuple[str, float]:
        if not self.knowledge_items:
            return "", 0.0

        q_words = set(question.lower().split())
        best: Tuple[str, float] = ("", 0.0)

        for it in self.knowledge_items:
            it_words = set((it.question or "").lower().split())
            if not it_words:
                continue
            inter = q_words.intersection(it_words)
            union = q_words.union(it_words)
            sim = len(inter) / len(union) if union else 0.0
            if sim > best[1] and sim >= Config.SIMILARITY_THRESHOLD:
                best = (it.answer, sim)
                it.usage_count += 1
                await self.db.save_knowledge_item(it)

        return best

# ------------------------------------------------------------------------------
# Scraper (async, newline-preserving, allowlist)
# ------------------------------------------------------------------------------
class WebScraper:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0, headers={
            "User-Agent": "Mozilla/5.0 (compatible; AI-Voice-Agent/1.1)"
        })

    async def close(self):
        await self.client.aclose()

    async def scrape_website(self, base_url: str, max_pages: int = 50) -> List[Dict[str, str]]:
        logger.info(f"Scraping {base_url}")
        from urllib.parse import urlparse, urljoin
        host = (urlparse(base_url).hostname or "").lower()
        if host not in Config.ALLOWED_SCRAPE_HOSTS:
            raise HTTPException(status_code=400, detail="Target host not allowed.")

        visited = set()
        queue = [base_url]
        out: List[Dict[str, str]] = []

        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            try:
                resp = await self.client.get(url)
                if resp.status_code == 200:
                    visited.add(url)
                    soup = BeautifulSoup(resp.content, "html.parser")
                    text = self._extract_text_content(soup)
                    out.extend(self._extract_qa_pairs(text, url))
                    # enqueue same-host links
                    for a in soup.find_all("a", href=True):
                        nu = urljoin(base_url, a["href"])
                        if (urlparse(nu).hostname or "").lower() == host and nu not in visited:
                            queue.append(nu)
                    logger.info(f"Scraped {url} -> {len(out)} total QAs")
            except Exception as e:
                logger.error(f"Scrape error {url}: {e}")

        return out

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        # Preserve line breaks for heuristic scanning
        return soup.get_text("\n", strip=True)

    def _extract_qa_pairs(self, text: str, source_url: str) -> List[Dict[str, str]]:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        qa: List[Dict[str, str]] = []
        for i, line in enumerate(lines):
            low = line.lower()
            if len(line) > 10 and (
                line.endswith("?") or
                low.startswith(("what", "how", "why", "when", "where")) or
                (line.isupper() and len(line) < 120)
            ):
                # take next 1-3 lines as the 'answer'
                ans = " ".join(lines[i+1:i+4]).strip()
                if 20 <= len(ans) <= 600:
                    qa.append({"question": line, "answer": ans, "source": source_url})
        return qa

# ------------------------------------------------------------------------------
# Security helpers
# ------------------------------------------------------------------------------
def require_api_key(request: Request):
    if Config.API_KEY:
        provided = request.headers.get("X-API-Key")
        if not provided or provided != Config.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# ------------------------------------------------------------------------------
# FastAPI app + lifespan
# ------------------------------------------------------------------------------
db_manager = DatabaseManager(Config.DATABASE_PATH)
ai_engine = LightweightAIEngine()
knowledge_manager = KnowledgeBaseManager(ai_engine, db_manager)
scraper = WebScraper()

app = FastAPI(title="AI Voice Agent System", version="1.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS if Config.CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounts
app.mount("/static", StaticFiles(directory=Config.STATIC_DIR), name="static")
app.mount("/audio", StaticFiles(directory=Config.AUDIO_PATH), name="audio")
templates = Jinja2Templates(directory=Config.TEMPLATE_DIR)

@app.on_event("startup")
async def _startup():
    await db_manager.init_database()
    await knowledge_manager.load()
    logger.info("Startup complete.")

@app.on_event("shutdown")
async def _shutdown():
    await scraper.close()
    logger.info("Shutdown complete.")

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    stats = await db_manager.get_call_statistics()
    unknown = await db_manager.get_unknown_questions(resolved=False)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "unknown_questions": [asdict(q) for q in unknown[:5]],
    })

@app.get("/voice-interface", response_class=HTMLResponse)
async def voice_interface(request: Request):
    return templates.TemplateResponse("voice_interface.html", {"request": request})

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    call_id = f"call_{uuid4().hex}"
    call = Call(call_id=call_id, caller_info="websocket", start_time=now_utc())
    logger.info(f"Call started: {call_id}")
    start_ts = asyncio.get_event_loop().time()

    try:
        while True:
            # Enforce max duration
            if asyncio.get_event_loop().time() - start_ts > Config.MAX_CALL_DURATION:
                await websocket.send_json({"type": "closing", "reason": "max_duration"})
                await websocket.close(code=1000)
                call.outcome = call.outcome or "answered"
                break

            # Receive next audio frame (bytes)
            data = await websocket.receive_bytes()

            # STT
            text = await ai_engine.speech_to_text(data)
            if not text:
                # send minimal ack to avoid client "stuck" states
                await websocket.send_json({"type": "noop"})
                continue

            call.transcript += f"User: {text}\n"

            # Check KB first
            answer, confidence = await knowledge_manager.find_answer(text)

            if confidence >= Config.CONFIDENCE_THRESHOLD:
                response_text = answer
                call.outcome = call.outcome or "answered"
            else:
                # human transfer intent?
                if any(k in text.lower() for k in Config.ROUTE_TO_HUMAN_KEYWORDS):
                    msg = "Transferring you to a human agent."
                    await websocket.send_json({"type": "transfer", "message": msg})
                    # Graceful close so client escalates
                    await websocket.send_json({"type": "closing", "reason": "transfer"})
                    await websocket.close(code=1000)
                    call.outcome = "transferred"
                    break
                # LLM
                response_text = await ai_engine.generate_response(text)
                # record unknown for later curation
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

            # TTS (MP3)
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
        logger.error(f"WebSocket error: {e}")
    finally:
        call.end_time = now_utc()
        await db_manager.save_call(call)

# ---- API (protected if API key configured) -----------------------------------
@app.post("/api/knowledge/add")
async def add_knowledge(request: Request, _: None = Depends(require_api_key)):
    data = await request.json()
    q = (data.get("question") or "").strip()
    a = (data.get("answer") or "").strip()
    if not q or not a:
        raise HTTPException(status_code=400, detail="Missing question/answer.")
    if len(q) > Config.MAX_INPUT_LEN or len(a) > Config.MAX_INPUT_LEN:
        raise HTTPException(status_code=413, detail="Input too long.")
    await knowledge_manager.add_knowledge_item(q, a, source="manual")
    return {"status": "success", "message": "Knowledge item added."}

@app.post("/api/knowledge/scrape")
async def scrape_website(request: Request, _: None = Depends(require_api_key)):
    data = await request.json()
    url = (data.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing url.")
    qa_pairs = await scraper.scrape_website(url)
    for qa in qa_pairs:
        await knowledge_manager.add_knowledge_item(
            qa["question"], qa["answer"], source=f"scraped:{qa['source']}"
        )
    return {"status": "success", "count": len(qa_pairs)}

@app.get("/api/knowledge/unknown")
async def get_unknown_questions(_: None = Depends(require_api_key)):
    qs = await db_manager.get_unknown_questions(resolved=False)
    return {"questions": [asdict(q) for q in qs]}

@app.post("/api/knowledge/resolve")
async def resolve_unknown_question(request: Request, _: None = Depends(require_api_key)):
    data = await request.json()
    qid = (data.get("question_id") or "").strip()
    q = (data.get("question") or "").strip()
    a = (data.get("answer") or "").strip()
    if not (qid and q and a):
        raise HTTPException(status_code=400, detail="Missing fields.")

    await knowledge_manager.add_knowledge_item(q, a, source="resolved")

    async with aiosqlite.connect(Config.DATABASE_PATH) as conn:
        await conn.execute("UPDATE unknown_questions SET resolved = 1 WHERE id = ?", (qid,))
        await conn.commit()

    return {"status": "success", "message": "Question resolved."}

@app.get("/api/stats")
async def get_statistics(_: None = Depends(require_api_key)):
    stats = await db_manager.get_call_statistics()
    stats["knowledge_items"] = len(knowledge_manager.knowledge_items)
    return stats

@app.post("/api/sms/send")
async def send_sms(request: Request, _: None = Depends(require_api_key)):
    # Placeholder; integrate Twilio/other here
    data = await request.json()
    phone = (data.get("phone") or "").strip()
    msg = (data.get("message") or "").strip()
    if not phone or not msg:
        raise HTTPException(status_code=400, detail="Missing phone/message.")
    logger.info(f"[SIMULATED SMS] to {phone}: {msg}")
    return {"status": "success", "message": "SMS sent (simulated)."}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Voice Agent"}

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting AI Voice Agent System (Lightweight Version, refactored)")
    logger.info(f"Dashboard: http://localhost:{Config.PORT}")
    logger.info(f"Voice UI:  http://localhost:{Config.PORT}/voice-interface")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
