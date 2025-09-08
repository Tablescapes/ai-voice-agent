#!/usr/bin/env python3
"""
Self-Sufficient AI Voice Agent System
Complete voice agent system with NO subscription dependencies
Uses: Local Whisper, Local TTS, WebRTC, Open-source components only
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
from sentence_transformers import SentenceTransformer
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

# Local AI imports
import whisper
import torch
from transformers import pipeline
import pyttsx3
import speech_recognition as sr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Local AI Models
    WHISPER_MODEL = "base"  # tiny, base, small, medium, large
    TTS_ENGINE = "pyttsx3"  # Local TTS engine
    LLM_MODEL = "microsoft/DialoGPT-medium"  # Local conversational AI
    
    # Fallback to OpenAI if API key provided (optional)
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
    PORT = 8000
    
    # WebRTC settings
    WEBRTC_PORT = 8001
    SIP_PORT = 5060
    
    # Routing keywords - FIXED: Added missing configuration
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

class LocalAIEngine:
    """Local AI processing engine - no external APIs required"""
    
    def __init__(self):
        self.whisper_model = None
        self.tts_engine = None
        self.llm_pipeline = None
        self.sentence_transformer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all local AI models"""
        logger.info("Initializing local AI models...")
        
        try:
            # Initialize Whisper for speech recognition
            logger.info("Loading Whisper model...")
            self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
            
            # Initialize local TTS
            logger.info("Initializing TTS engine...")
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Speed
            self.tts_engine.setProperty('volume', 0.9)  # Volume
            
            # Initialize sentence transformer for embeddings
            logger.info("Loading sentence transformer...")
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize conversational AI
            if Config.OPENAI_API_KEY:
                logger.info("OpenAI API key found - will use GPT for conversations")
            else:
                logger.info("Loading local conversational AI...")
                try:
                    self.llm_pipeline = pipeline(
                        "conversational",
                        model=Config.LLM_MODEL,
                        device=0 if torch.cuda.is_available() else -1
                    )
                except Exception as e:
                    logger.warning(f"Could not load local LLM: {e}")
                    logger.info("Will use rule-based responses")
            
            logger.info("All AI models initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
            raise
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech to text using local Whisper"""
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_path)
            text = result["text"].strip()
            
            # Clean up temp file
            os.unlink(temp_path)
            
            logger.info(f"Transcribed: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Speech to text error: {e}")
            return ""
    
    def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech using local TTS - FIXED: Threading issues"""
        try:
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Use subprocess to run TTS to avoid threading issues
            try:
                # Try using espeak as a more reliable alternative
                subprocess.run([
                    'espeak', '-w', temp_path, '-s', '150', '-v', 'en', text
                ], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to pyttsx3 with proper threading
                result_queue = queue.Queue()
                
                def tts_worker():
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        engine.setProperty('volume', 0.9)
                        engine.save_to_file(text, temp_path)
                        engine.runAndWait()
                        result_queue.put(True)
                    except Exception as e:
                        result_queue.put(e)
                
                thread = threading.Thread(target=tts_worker)
                thread.start()
                thread.join(timeout=10)  # 10 second timeout
                
                result = result_queue.get_nowait() if not result_queue.empty() else False
                if isinstance(result, Exception):
                    raise result
                elif not result:
                    raise Exception("TTS timeout")
            
            # Read audio data
            if os.path.exists(temp_path):
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up
                os.unlink(temp_path)
                
                return audio_data
            else:
                raise Exception("TTS output file not created")
                
        except Exception as e:
            logger.error(f"Text to speech error: {e}")
            return b""
    
    def generate_response(self, user_input: str, context: str = "") -> str:
        """Generate conversational response - FIXED: Updated OpenAI API"""
        try:
            if Config.OPENAI_API_KEY:
                # Use OpenAI if available (updated API format)
                from openai import OpenAI
                client = OpenAI(api_key=Config.OPENAI_API_KEY)
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are a helpful customer service agent. Context: {context}"},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
                
            elif self.llm_pipeline:
                # Use local LLM
                from transformers import Conversation
                conversation = Conversation(user_input)
                response = self.llm_pipeline(conversation)
                return response.generated_responses[-1]
                
            else:
                # Fallback to rule-based responses
                return self._rule_based_response(user_input)
                
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I apologize, but I'm having trouble processing your request right now."
    
    def _rule_based_response(self, user_input: str) -> str:
        """Simple rule-based responses as fallback"""
        user_input = user_input.lower()
        
        if any(word in user_input for word in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"
        elif any(word in user_input for word in ["help", "support"]):
            return "I'm here to help! What specific information are you looking for?"
        elif any(word in user_input for word in ["price", "cost", "pricing"]):
            return "Let me help you with pricing information. What specific service are you interested in?"
        elif any(word in user_input for word in ["hours", "open", "closed"]):
            return "Let me check our current hours for you."
        else:
            return "I understand you're asking about that. Let me find the best person to help you with this specific question."
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            embedding = self.sentence_transformer.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return []

class KnowledgeBaseManager:
    """Manages the learning knowledge base"""
    
    def __init__(self, ai_engine: LocalAIEngine, db_manager):
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
        """Find best answer for question"""
        if not self.knowledge_items:
            return "", 0.0
        
        question_embedding = self.ai_engine.get_embedding(question)
        if not question_embedding:
            return "", 0.0
        
        best_answer = ""
        best_score = 0.0
        
        for item in self.knowledge_items:
            if not item.embedding:
                continue
                
            # Calculate similarity
            similarity = self._calculate_similarity(question_embedding, item.embedding)
            
            if similarity > best_score and similarity > Config.SIMILARITY_THRESHOLD:
                best_answer = item.answer
                best_score = similarity
                
                # Update usage count
                item.usage_count += 1
                self.db_manager.save_knowledge_item(item)
        
        return best_answer, best_score
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0
    
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
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Look for FAQ patterns
        faq_patterns = [
            r'(?i)(what|how|why|when|where|who|can|is|are|do|does|will|would)\s+.*\?',
            r'(?i)q:\s*(.*?)\s*a:\s*(.*?)(?=q:|$)',
            r'(?i)question:\s*(.*?)\s*answer:\s*(.*?)(?=question:|$)'
        ]
        
        for pattern in faq_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    question = match.group(1).strip()
                    answer = match.group(2).strip()
                else:
                    question = match.group(0).strip()
                    answer = self._find_answer_nearby(text, match.end())
                
                if len(question) > 10 and len(answer) > 10:
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "source": source_url
                    })
        
        # Extract headings as potential Q&A
        headings = re.findall(r'<h[1-6][^>]*>(.*?)</h[1-6]>', text, re.IGNORECASE)
        for heading in headings:
            if len(heading) > 5:
                # Find content after heading
                heading_pos = text.find(heading)
                if heading_pos != -1:
                    content_after = text[heading_pos + len(heading):heading_pos + len(heading) + 500]
                    first_sentence = re.split(r'[.!?]+', content_after)[0].strip()
                    
                    if len(first_sentence) > 20:
                        qa_pairs.append({
                            "question": f"Tell me about {heading}",
                            "answer": first_sentence,
                            "source": source_url
                        })
        
        return qa_pairs
    
    def _find_answer_nearby(self, text: str, start_pos: int, max_length: int = 300) -> str:
        """Find answer text near a question"""
        end_pos = min(start_pos + max_length, len(text))
        nearby_text = text[start_pos:end_pos]
        
        # Find first complete sentence
        sentences = re.split(r'[.!?]+', nearby_text)
        if sentences and len(sentences[0]) > 10:
            return sentences[0].strip()
        
        return nearby_text.strip()

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
        
        # Analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                date TEXT PRIMARY KEY,
                total_calls INTEGER,
                answered_calls INTEGER,
                transferred_calls INTEGER,
                avg_confidence REAL,
                unknown_questions INTEGER
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
ai_engine = LocalAIEngine()
knowledge_manager = KnowledgeBaseManager(ai_engine, db_manager)

# FastAPI application
app = FastAPI(title="Self-Sufficient AI Voice Agent", version="1.0.0")

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
                    "audio": base64.b64encode(audio_response).decode(),
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
    
    # In a real implementation, this would integrate with local SMS gateway
    # For now, we'll log the SMS
    logger.info(f"SMS would be sent to {data['phone']}: {data['message']}")
    
    return {"status": "success", "message": "SMS sent (simulated)"}

if __name__ == "__main__":
    logger.info("Starting Self-Sufficient AI Voice Agent System")
    logger.info(f"Dashboard will be available at http://localhost:{Config.PORT}")
    logger.info(f"Voice interface at http://localhost:{Config.PORT}/voice-interface")
    
    uvicorn.run(
        app, 
        host=Config.HOST, 
        port=Config.PORT,
        log_level="info"
    )
