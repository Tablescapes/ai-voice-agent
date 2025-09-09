#!/usr/bin/env python3
"""
Self-Sufficient AI Voice Agent System - Lightweight Version
Uses cloud APIs for speech processing to reduce deployment size
"""

from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import asyncio
import aiohttp
from urllib.parse import urljoin
import sqlite3
from contextlib import asynccontextmanager
import hashlib
import re
from dataclasses import dataclass, asdict
import requests
from bs4 import BeautifulSoup
import numpy as np
import pickle
import threading
import time
import base64
import wave
import io
import subprocess
import tempfile
import websockets
import ssl
from pathlib import Path
import queue

# Lightweight AI imports
import speech_recognition as sr
from gtts import gTTS
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Cloud AI Services (lighter than local models)
    SPEECH_RECOGNITION_SERVICE = "google"  # Uses Google's free API
    TTS_SERVICE = "gtts"  # Google Text-to-Speech
    
    # OpenAI for conversations (optional but recommended)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    
    # Application settings
    BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
    DATABASE_PATH = "voice_agent.db"
    KNOWLEDGE_BASE_PATH = "knowledge_base.pkl"
    MODELS_PATH = "models"
    AUDIO_PATH = "audio"
    
    # Voice settings
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    AUDIO_FORMAT = "wav"
    
    # Learning settings
    SIMILARITY_THRESHOLD = 0.7
    MAX_CALL_DURATION = 600  # 10 minutes
    CONFIDENCE_THRESHOLD = 0.6
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))
    
    # WebRTC settings
    WEBRTC_PORT = 8001
    SIP_PORT = 5060
    
    # Routing keywords
    ROUTE_TO_HUMAN_KEYWORDS = [
        "human", "person", "agent", "representative", 
        "manager", "supervisor", "help me", "speak to someone"
    ]

# Ensure directories exist
os.makedirs(Config.MODELS_PATH, exist_ok=True)
os.makedirs(Config.AUDIO_PATH, exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Data models
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
    last_updated: datetime = None
    usage_count: int = 0

@dataclass
class UnknownQuestion:
    id: str
    question: str
    call_id: str
    timestamp: datetime
    resolved: bool = False
    suggested_answer: Optional[str] = None

class LightweightAIEngine:
    """Lightweight AI engine using cloud services"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.openai_client = None
        if Config.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize speech recognition and other services"""
        logger.info("Initializing lightweight AI services...")
        
        try:
            # Configure speech recognition
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            
            if Config.OPENAI_API_KEY:
                logger.info("OpenAI API key found - will use GPT for conversations")
            else:
                logger.info("No OpenAI API key - will use rule-based responses")
            
            logger.info("Lightweight AI services initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing AI services: {e}")
            raise
    def convert_webm_to_wav(self, webm_data: bytes) -> bytes:
        """Convert WebM audio to WAV format"""
        try:
            # Save WebM data to temp file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
                webm_file.write(webm_data)
                webm_path = webm_file.name
            
            # Convert to WAV using ffmpeg (if available) or return original
            wav_path = webm_path.replace('.webm', '.wav')
            
            try:
                subprocess.run([
                    'ffmpeg', '-i', webm_path, '-ar', '16000', '-ac', '1', 
                    '-f', 'wav', wav_path, '-y'
                ], check=True, capture_output=True)
                
                with open(wav_path, 'rb') as f:
                    wav_data = f.read()
                
                os.unlink(webm_path)
                os.unlink(wav_path)
                return wav_data
                
            except:
                # If ffmpeg fails, return original data
                os.unlink(webm_path)
                return webm_data
                
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return webm_data



    
    def speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech to text using OpenAI Whisper"""
        try:
            if not self.openai_client:
                return ""
                
            # Convert and save audio data to temporary file
            converted_audio = self.convert_webm_to_wav(audio_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(converted_audio)
                temp_path = temp_file.name
            
            # Use OpenAI Whisper API
            with open(temp_path, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                
            # Clean up temp file
            os.unlink(temp_path)
            
            text = transcript.text.strip()
            logger.info(f"Transcribed: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Speech to text error: {e}")
            return ""
    
    def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech using Google Text-to-Speech"""
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                tts.save(temp_file.name)
                temp_path = temp_file.name
            
            # Read audio data
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Text to speech error: {e}")
            return b""
    
    def generate_response(self, user_input: str, context: str = "") -> str:
        """Generate conversational response"""
        try:
            if self.openai_client:
                # Use OpenAI for better responses
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are a helpful customer service agent for a tableware and home decor company. Keep responses concise and helpful. Context: {context}"},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            else:
                # Fallback to rule-based responses
                return self._rule_based_response(user_input)
                
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I apologize, but I'm having trouble processing your request right now. How else can I help you?"
    
    def _rule_based_response(self, user_input: str) -> str:
        """Enhanced rule-based responses for tablescapes business"""
        user_input = user_input.lower()
        
        if any(word in user_input for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return "Hello! Welcome to Tablescapes. How can I help you find the perfect items for your home today?"
        elif any(word in user_input for word in ["help", "support", "question"]):
            return "I'm here to help! Are you looking for tableware, home decor, or do you have questions about our products?"
        elif any(word in user_input for word in ["price", "cost", "pricing", "how much"]):
            return "I'd be happy to help with pricing information. What specific items are you interested in?"
        elif any(word in user_input for word in ["hours", "open", "closed", "when"]):
            return "Let me help you with our hours and availability information."
        elif any(word in user_input for word in ["blue", "color", "colors"]):
            return "We have beautiful blue items! Would you like me to help you find specific blue tableware or decor pieces?"
        elif any(word in user_input for word in ["shipping", "delivery", "ship"]):
            return "I can help you with shipping information. What would you like to know about delivery options?"
        elif any(word in user_input for word in ["return", "exchange", "refund"]):
            return "I can assist you with returns and exchanges. What item would you like to return or exchange?"
        else:
            return "I understand you're asking about that. Let me connect you with someone who can provide more detailed information about our products and services."
    
    def get_embedding(self, text: str) -> List[float]:
        """Simple embedding using basic text features (fallback)"""
        try:
            # Simple word-based embedding (very basic)
            words = text.lower().split()
            # Create a simple vector based on word length and character features
            features = [
                len(words),
                sum(len(word) for word in words) / max(len(words), 1),
                sum(1 for word in words if any(char.isdigit() for char in word)),
                sum(1 for word in words if '?' in word or '!' in word)
            ]
            # Pad to make it 10-dimensional
            while len(features) < 10:
                features.append(0.0)
            
            return features[:10]  # Return first 10 features
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return [0.0] * 10  # Return zero vector

class KnowledgeBaseManager:
    """Manages the learning knowledge base"""
    
    def __init__(self, ai_engine: LightweightAIEngine, db_manager):
        self.ai_engine = ai_engine
        self.db_manager = db_manager
        self.knowledge_items: List[KnowledgeItem] = []
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load knowledge base from database"""
        self.knowledge_items = self.db_manager.get_knowledge_base()
        logger.info(f"Loaded {len(self.knowledge_items)} knowledge items")
    
    def add_knowledge_item(self, question: str, answer: str, source: str = "manual"):
        """Add new knowledge item"""
        item_id = hashlib.md5(question.encode()).hexdigest()
        embedding = self.ai_engine.get_embedding(question)
        
        item = KnowledgeItem(
            id=item_id,
            question=question,
            answer=answer,
            source=source,
            embedding=embedding,
            last_updated=datetime.now()
        )
        
        self.knowledge_items.append(item)
        self.db_manager.save_knowledge_item(item)
        logger.info(f"Added knowledge item: {question[:50]}...")
    
    def find_answer(self, question: str) -> tuple[str, float]:
        """Find best answer for question using simple text matching"""
        if not self.knowledge_items:
            return "", 0.0
        
        question_lower = question.lower()
        best_answer = ""
        best_score = 0.0
        
        for item in self.knowledge_items:
            # Simple keyword matching
            question_words = set(question_lower.split())
            item_words = set(item.question.lower().split())
            
            # Calculate word overlap score
            common_words = question_words.intersection(item_words)
            if len(question_words) > 0:
                similarity = len(common_words) / len(question_words.union(item_words))
            else:
                similarity = 0.0
            
            if similarity > best_score and similarity > Config.SIMILARITY_THRESHOLD:
                best_answer = item.answer
                best_score = similarity
                
                # Update usage count
                item.usage_count += 1
                self.db_manager.save_knowledge_item(item)
        
        return best_answer, best_score
    
    def scrape_website_knowledge(self, website_url: str):
        """Scrape website and add to knowledge base"""
        scraper = WebScraper()
        qa_pairs = scraper.scrape_website(website_url)
        
        for qa_pair in qa_pairs:
            self.add_knowledge_item(
                question=qa_pair["question"],
                answer=qa_pair["answer"],
                source=f"scraped:{website_url}"
            )
        
        logger.info(f"Added {len(qa_pairs)} items from website scraping")

class WebScraper:
    """Website scraping for knowledge base building"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; AI-Voice-Agent/1.0)'
        })
    
    def scrape_website(self, base_url: str, max_pages: int = 50) -> List[Dict[str, str]]:
        """Scrape website content and extract Q&A pairs"""
        logger.info(f"Starting website scrape of {base_url}")
        
        scraped_content = []
        visited_urls = set()
        urls_to_visit = [base_url]
        
        while urls_to_visit and len(visited_urls) < max_pages:
            url = urls_to_visit.pop(0)
            if url in visited_urls:
                continue
                
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    visited_urls.add(url)
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract text content
                    text_content = self._extract_text_content(soup)
                    
                    # Find new links
                    for link in soup.find_all('a', href=True):
                        new_url = urljoin(base_url, link['href'])
                        if new_url.startswith(base_url) and new_url not in visited_urls:
                            urls_to_visit.append(new_url)
                    
                    # Extract Q&A pairs from content
                    qa_pairs = self._extract_qa_pairs(text_content, url)
                    scraped_content.extend(qa_pairs)
                    
                    logger.info(f"Scraped {url} - found {len(qa_pairs)} Q&A pairs")
                    
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
        
        logger.info(f"Website scraping complete. Total Q&A pairs: {len(scraped_content)}")
        return scraped_content
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_qa_pairs(self, text: str, source_url: str) -> List[Dict[str, str]]:
        """Extract question-answer pairs from text"""
        qa_pairs = []
        
        # Look for FAQ patterns and headings
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for headings or questions
            if len(line) > 10 and (
                line.endswith('?') or 
                any(line.lower().startswith(word) for word in ['what', 'how', 'why', 'when', 'where']) or
                line.isupper() or
                (len(line.split()) < 10 and ':' not in line)
            ):
                # Find the next few lines as potential answer
                answer_lines = []
                for j in range(i + 1, min(i + 4, len(lines))):
                    if j < len(lines) and lines[j].strip():
                        answer_lines.append(lines[j].strip())
                
                if answer_lines:
                    answer = ' '.join(answer_lines)
                    if len(answer) > 20 and len(answer) < 500:
                        qa_pairs.append({
                            "question": line,
                            "answer": answer,
                            "source": source_url
                        })
        
        return qa_pairs

class DatabaseManager:
    """Database management for persistence"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calls table
        cursor.execute("""
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
        
        # Knowledge base table
        cursor.execute("""
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
        
        # Unknown questions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS unknown_questions (
                id TEXT PRIMARY KEY,
                question TEXT,
                call_id TEXT,
                timestamp TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                suggested_answer TEXT
            )
        """)
        
        # Settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_call(self, call: Call):
        """Save call record to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO calls 
            (call_id, caller_info, start_time, end_time, transcript, outcome, confidence_score, audio_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            call.call_id, call.caller_info,
            call.start_time.isoformat() if call.start_time else None,
            call.end_time.isoformat() if call.end_time else None,
            call.transcript, call.outcome, call.confidence_score, call.audio_file
        ))
        
        conn.commit()
        conn.close()
    
    def save_knowledge_item(self, item: KnowledgeItem):
        """Save knowledge item to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = pickle.dumps(item.embedding) if item.embedding else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO knowledge_base 
            (id, question, answer, source, embedding, last_updated, usage_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            item.id, item.question, item.answer, item.source,
            embedding_blob, item.last_updated.isoformat() if item.last_updated else None,
            item.usage_count
        ))
        
        conn.commit()
        conn.close()
    
    def save_unknown_question(self, question: UnknownQuestion):
        """Save unknown question to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO unknown_questions 
            (id, question, call_id, timestamp, resolved, suggested_answer)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            question.id, question.question, question.call_id,
            question.timestamp.isoformat(), question.resolved, question.suggested_answer
        ))
        
        conn.commit()
        conn.close()
    
    def get_knowledge_base(self) -> List[KnowledgeItem]:
        """Get all knowledge base items"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM knowledge_base")
        rows = cursor.fetchall()
        
        items = []
        for row in rows:
            embedding = pickle.loads(row[4]) if row[4] else None
            item = KnowledgeItem(
                id=row[0], question=row[1], answer=row[2], source=row[3],
                embedding=embedding,
                last_updated=datetime.fromisoformat(row[5]) if row[5] else None,
                usage_count=row[6] if len(row) > 6 else 0
            )
            items.append(item)
        
        conn.close()
        return items
    
    def get_unknown_questions(self, resolved: bool = False) -> List[UnknownQuestion]:
        """Get unknown questions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM unknown_questions WHERE resolved = ?", (resolved,))
        rows = cursor.fetchall()
        
        questions = []
        for row in rows:
            question = UnknownQuestion(
                id=row[0], question=row[1], call_id=row[2],
                timestamp=datetime.fromisoformat(row[3]), resolved=row[4],
                suggested_answer=row[5] if len(row) > 5 else None
            )
            questions.append(question)
        
        conn.close()
        return questions
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Get call statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get today's stats
        today = datetime.now().date().isoformat()
        
        cursor.execute("""
            SELECT COUNT(*) as total_calls,
                   AVG(confidence_score) as avg_confidence,
                   SUM(CASE WHEN outcome = 'answered' THEN 1 ELSE 0 END) as answered_calls,
                   SUM(CASE WHEN outcome = 'transferred' THEN 1 ELSE 0 END) as transferred_calls
            FROM calls 
            WHERE DATE(start_time) = ?
        """, (today,))
        
        stats = cursor.fetchone()
        
        # Get unknown questions count
        cursor.execute("""
            SELECT COUNT(*) FROM unknown_questions 
            WHERE DATE(timestamp) = ? AND resolved = FALSE
        """, (today,))
        
        unknown_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_calls": stats[0] or 0,
            "avg_confidence": stats[1] or 0.0,
            "answered_calls": stats[2] or 0,
            "transferred_calls": stats[3] or 0,
            "unknown_questions": unknown_count
        }

# Initialize global components
db_manager = DatabaseManager(Config.DATABASE_PATH)
ai_engine = LightweightAIEngine()
knowledge_manager = KnowledgeBaseManager(ai_engine, db_manager)

# FastAPI application
app = FastAPI(title="AI Voice Agent System", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio", StaticFiles(directory=Config.AUDIO_PATH), name="audio")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard"""
    stats = db_manager.get_call_statistics()
    unknown_questions = db_manager.get_unknown_questions(resolved=False)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "unknown_questions": unknown_questions[:5]  # Show latest 5
    })

@app.get("/voice-interface", response_class=HTMLResponse)
async def voice_interface(request: Request):
    """Voice interface page"""
    return templates.TemplateResponse("voice_interface.html", {
        "request": request
    })

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time voice communication"""
    await websocket.accept()
    
    call_id = f"call_{int(time.time())}"
    call = Call(
        call_id=call_id,
        caller_info="websocket",
        start_time=datetime.now()
    )
    
    logger.info(f"Voice call started: {call_id}")
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process speech to text
            text = ai_engine.speech_to_text(data)
            
            if text:
                call.transcript += f"User: {text}\n"
                
                # Find answer in knowledge base
                answer, confidence = knowledge_manager.find_answer(text)
                
                if confidence > Config.CONFIDENCE_THRESHOLD:
                    response_text = answer
                    call.outcome = "answered"
                else:
                    # Check if should transfer to human
                    if any(keyword in text.lower() for keyword in Config.ROUTE_TO_HUMAN_KEYWORDS):
                        response_text = "Let me transfer you to a human agent who can better assist you."
                        call.outcome = "transferred"
                        
                        # Send transfer signal
                        await websocket.send_json({
                            "type": "transfer",
                            "message": response_text
                        })
                        break
                    else:
                        # Generate AI response
                        response_text = ai_engine.generate_response(text)
                        
                        # Log as unknown question if confidence is low
                        if confidence < Config.CONFIDENCE_THRESHOLD:
                            unknown_q = UnknownQuestion(
                                id=f"unknown_{int(time.time())}",
                                question=text,
                                call_id=call_id,
                                timestamp=datetime.now(),
                                suggested_answer=response_text
                            )
                            db_manager.save_unknown_question(unknown_q)
                
                call.transcript += f"Agent: {response_text}\n"
                call.confidence_score = confidence
                
                # Convert response to speech
                audio_response = ai_engine.text_to_speech(response_text)
                
                # Send response
                await websocket.send_json({
                    "type": "response",
                    "text": response_text,
                    "audio": base64.b64encode(audio_response).decode() if audio_response else "",
                    "confidence": confidence
                })
                
    except WebSocketDisconnect:
        logger.info(f"Voice call ended: {call_id}")
    except Exception as e:
        logger.error(f"Voice call error: {e}")
    finally:
        call.end_time = datetime.now()
        db_manager.save_call(call)

@app.post("/api/knowledge/add")
async def add_knowledge(request: Request):
    """Add knowledge item manually"""
    data = await request.json()
    
    knowledge_manager.add_knowledge_item(
        question=data["question"],
        answer=data["answer"],
        source="manual"
    )
    
    return {"status": "success", "message": "Knowledge item added"}

@app.post("/api/knowledge/scrape")
async def scrape_website(request: Request):
    """Scrape website for knowledge"""
    data = await request.json()
    website_url = data["url"]
    
    try:
        knowledge_manager.scrape_website_knowledge(website_url)
        return {"status": "success", "message": f"Website {website_url} scraped successfully"}
    except Exception as e:
        logger.error(f"Website scraping error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/knowledge/unknown")
async def get_unknown_questions():
    """Get unresolved questions"""
    questions = db_manager.get_unknown_questions(resolved=False)
    return {"questions": [asdict(q) for q in questions]}

@app.post("/api/knowledge/resolve")
async def resolve_unknown_question(request: Request):
    """Resolve unknown question by adding to knowledge base"""
    data = await request.json()
    
    # Add to knowledge base
    knowledge_manager.add_knowledge_item(
        question=data["question"],
        answer=data["answer"],
        source="resolved"
    )
    
    # Mark as resolved
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE unknown_questions SET resolved = TRUE WHERE id = ?",
        (data["question_id"],)
    )
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "Question resolved"}

@app.get("/api/stats")
async def get_statistics():
    """Get system statistics"""
    stats = db_manager.get_call_statistics()
    knowledge_count = len(knowledge_manager.knowledge_items)
    
    stats["knowledge_items"] = knowledge_count
    return stats

@app.post("/api/sms/send")
async def send_sms(request: Request):
    """Send SMS (mock implementation for local system)"""
    data = await request.json()
    
    # In a real implementation, this would integrate with Twilio
    logger.info(f"SMS would be sent to {data['phone']}: {data['message']}")
    
    return {"status": "success", "message": "SMS sent (simulated)"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Voice Agent"}

if __name__ == "__main__":
    logger.info("Starting AI Voice Agent System (Lightweight Version)")
    logger.info(f"Dashboard will be available at http://localhost:{Config.PORT}")
    logger.info(f"Voice interface at http://localhost:{Config.PORT}/voice-interface")
    
    uvicorn.run(
        app, 
        host=Config.HOST, 
        port=Config.PORT,
        log_level="info"
    )
