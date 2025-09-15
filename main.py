import os
import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, AsyncGenerator, List, Dict, Any
import logging
from dotenv import load_dotenv
import re
import aiohttp
from functools import lru_cache
import time
from collections import defaultdict
from threading import Lock
import concurrent.futures  # Added to fix undefined variable

# Load environment variables
load_dotenv()

# Core imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Database
from motor.motor_asyncio import AsyncIOMotorClient

# AI and Tools
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

# External APIs
import speech_recognition as sr
from pydub import AudioSegment
import io
import base64

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key")
SERPAPI_KEY = os.getenv("SERP_API_KEY", "your-serpapi-key")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "your-gnews-key")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
PORT = int(os.getenv("PORT", 8000))

# Database client with optimized settings
client = AsyncIOMotorClient(
    MONGODB_URL,
    maxPoolSize=10,
    minPoolSize=2,
    maxIdleTimeMS=30000,
    waitQueueTimeoutMS=5000,
    serverSelectionTimeoutMS=5000
)
db = client.chatbot_db
sessions_collection = db.sessions
conversations_collection = db.conversations

# Optimized in-memory caches with size limits
memory_cache = {}
tool_cache = defaultdict(dict)
cache_locks = defaultdict(Lock)
CACHE_TTL = 3600

# Pre-computed responses for ultra-fast replies
INSTANT_RESPONSES = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! What can I assist you with?", 
    "hey": "Hey! What's on your mind?",
    "thanks": "You're welcome! Anything else I can help with?",
    "thank you": "Happy to help! Is there anything else you need?",
    "bye": "Goodbye! Have a great day!",
    "goodbye": "Take care! Feel free to come back anytime.",
    "how are you": "I'm doing great, thanks for asking! How about you?",
    "what's the weather": "Could you specify a location for the weather forecast?",
    "how's it going": "All good here! What's on your mind?"
}

# Thread pool for blocking operations
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    voice_data: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    message: str

# Optimized cache functions
@lru_cache(maxsize=1000)
def get_cached_result(cache_key: str, cache_type: str = 'general', ttl: int = CACHE_TTL):
    """Thread-safe cache retrieval with LRU caching"""
    cache_dict = tool_cache[cache_type]
    with cache_locks[cache_type]:
        if cache_key in cache_dict:
            result, timestamp = cache_dict[cache_key]
            if time.time() - timestamp < ttl:
                return result
    return None

def set_cache_result(cache_key: str, result: Any, cache_type: str = 'general', ttl: int = CACHE_TTL):
    """Thread-safe cache setting"""
    cache_dict = tool_cache[cache_type]
    with cache_locks[cache_type]:
        cache_dict[cache_key] = (result, time.time())

# Global aiohttp session
global_session = None

# Optimized Tools with better async handling
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information, breaking news, recent events, or specific facts I don't know. Use for: current events, recent news, today's weather, live sports scores, stock prices, recent updates about people/companies, or when you need verified current information."
    
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, WebSearchTool) and self.name == other.name

    async def _arun(self, query: str) -> str:
        cache_key = f"web_{hash(query)}"
        cached = get_cached_result(cache_key, 'web', 1800)
        if cached:
            return cached

        try:
            if not SERPAPI_KEY or SERPAPI_KEY == "your-serpapi-key":
                logger.error("Invalid or missing SERPAPI_KEY")
                return json.dumps([])

            url = "https://serpapi.com/search"
            params = {
                "engine": "google",
                "q": query,
                "api_key": SERPAPI_KEY,
                "num": 5,
                "ijn": 0
            }
            
            async with global_session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Web search API returned status: {response.status}")
                    return json.dumps([])
                
                data = await response.json()
                results = []
                for item in (data.get("organic_results") or [])[:2]:
                    title = item.get("title", "")[:100]
                    snippet = item.get("snippet", "")[:200]
                    link = item.get("link", "")
                    if title and snippet:
                        results.append({"title": title, "snippet": snippet, "url": link})
                
                result = json.dumps(results)
                set_cache_result(cache_key, result, 'web', 1800)
                return result
                
        except asyncio.TimeoutError:
            logger.error(f"Web search timeout for query: {query}")
            return json.dumps([])
        except Exception as e:
            logger.error(f"Web search error for query '{query}': {str(e)}")
            return json.dumps([])

    def _run(self, query: str) -> str:
        """Synchronous version using thread pool"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = thread_pool.submit(asyncio.run, self._arun(query))
                return future.result(timeout=8)
            else:
                return loop.run_until_complete(self._arun(query))
        except Exception as e:
            logger.error(f"Web search sync error: {e}")
            return json.dumps([])

class ImageSearchTool(BaseTool):
    name: str = "image_search"
    description: str = "Search for images when user specifically asks for images, pictures, photos, or visual content. Returns image URLs and thumbnails."
    
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, ImageSearchTool) and self.name == other.name

    async def _arun(self, query: str) -> str:
        cache_key = f"img_{hash(query)}"
        cached = get_cached_result(cache_key, 'image', 7200)
        if cached:
            return cached

        try:
            if not SERPAPI_KEY or SERPAPI_KEY == "your-serpapi-key":
                return json.dumps({"error": "Invalid or missing SerpAPI key"})

            url = "https://serpapi.com/search"
            params = {
                "engine": "google",
                "q": query,
                "tbm": "isch",
                "api_key": SERPAPI_KEY,
                "num": 6,
                "ijn": 0
            }
            
            async with global_session.get(url, params=params) as response:
                if response.status != 200:
                    error_msg = f"Image search API error: HTTP {response.status}"
                    return json.dumps({"error": error_msg})
                
                data = await response.json()
                if "error" in data:
                    return json.dumps({"error": f"SerpAPI error: {data['error']}"})
                
                images = []
                for img in (data.get("images_results") or [])[:2]:
                    title = img.get("title", "Untitled Image")[:80]
                    original = img.get("original", "")
                    thumbnail = img.get("thumbnail", original)
                    if original:
                        images.append({"title": title, "url": original, "thumbnail": thumbnail})
                
                result = json.dumps(images)
                set_cache_result(cache_key, result, 'image', 7200)
                return result
                
        except asyncio.TimeoutError:
            return json.dumps({"error": "Image search timeout"})
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return json.dumps({"error": f"Image search failed: {str(e)}"})

    def _run(self, query: str) -> str:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = thread_pool.submit(asyncio.run, self._arun(query))
                return future.result(timeout=8)
            else:
                return loop.run_until_complete(self._arun(query))
        except Exception as e:
            return json.dumps({"error": f"Image search failed: {str(e)}"})

class NewsSearchTool(BaseTool):
    name: str = "news_search"
    description: str = "Search for latest news and current events. Use when user asks for news, breaking news, current events, or recent happenings about specific topics."

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, NewsSearchTool) and self.name == other.name

    async def _arun(self, query: str) -> str:
        cache_key = f"news_{hash(query)}"
        cached = get_cached_result(cache_key, 'news', 900)
        if cached:
            return cached

        try:
            if not GNEWS_API_KEY or GNEWS_API_KEY == "your-gnews-key":
                return "News search unavailable - API key not configured"

            url = "https://gnews.io/api/v4/search"
            params = {"q": query, "token": GNEWS_API_KEY, "max": 3, "lang": "en"}
            
            async with global_session.get(url, params=params) as response:
                data = await response.json()
                articles = data.get("articles") or []
                if not articles:
                    return "No recent news found for this topic"

                results = []
                for article in articles[:2]:
                    title = article.get("title", "No title")[:100]
                    url_link = article.get("url", "")
                    image_url = article.get("image", "https://example.com/placeholder.jpg")
                    published = article.get("publishedAt", "Unknown date")

                    card = f"""**{title}**

![News Image]({image_url})

Published: {published}
Source: [{url_link}]({url_link})"""
                    results.append(card.strip())
                
                result = "\n\n---\n\n".join(results) if results else "No news found"
                set_cache_result(cache_key, result, 'news', 900)
                return result
                
        except Exception as e:
            logger.error(f"News search error: {e}")
            return f"News search temporarily unavailable: {str(e)}"

    def _run(self, query: str) -> str:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = thread_pool.submit(asyncio.run, self._arun(query))
                return future.result(timeout=8)
            else:
                return loop.run_until_complete(self._arun(query))
        except Exception as e:
            return f"News search failed: {str(e)}"

class DateTimeTool(BaseTool):
    name: str = "datetime"
    description: str = "Get current date and time when user asks about the current time, date, or when answering questions that need current timestamp."

    def _run(self, query: str = "") -> str:
        now = datetime.now()
        return f"**Current date and time:** {now.strftime('%Y-%m-%d %H:%M:%S')}"

# Optimized voice processing
async def process_voice_input(voice_data: str) -> str:
    """Process voice input with better error handling"""
    cache_key = f"voice_{hash(voice_data)}"
    cached = get_cached_result(cache_key, 'voice', 3600)
    if cached:
        return cached

    def sync_process_voice():
        try:
            audio_bytes = base64.b64decode(voice_data)
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_frame_rate(16000)
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.8
            
            with sr.AudioFile(wav_buffer) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language='en-US')
                return text.strip()
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return "Could not process voice input"

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(thread_pool, sync_process_voice)
    if result != "Could not process voice input":
        set_cache_result(cache_key, result, 'voice', 3600)
    return result

# FastAPI app
app = FastAPI(
    title="Cassy AI Assistant", 
    description="Fast conversational AI with smart function calling",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-initialize tools
tools = [WebSearchTool(), ImageSearchTool(), NewsSearchTool(), DateTimeTool()]

# Improved prompt template - let LLM decide when to use tools
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"""You are cassy, a helpful AI assistant. You have access to these tools when needed:

🔍 **web_search**: For current events, breaking news, recent information, live data, financial data or facts you're unsure about  
📸 **image_search**: When users specifically request images, pictures, or visual content  
📰 **news_search**: For latest news and current events  
📅 **datetime**: For current date/time

**When to use tools:**
- Use tools ONLY when you need current/live information
- For general knowledge, explanations, definitions, historical facts → answer directly without tools
- For "who is [person]" questions → perform an image search of the person. Show exactly 2 images side by side (horizontal format) of the person like this :

"Images:  
{{  
  "images": [  
    {{ "url": "IMAGE_URL_1", "width": 200 }},  
    {{ "url": "IMAGE_URL_2", "width": 200 }}  
  ]  
}}"

- For image requests → use image_search and return the raw JSON exactly as received
- For news requests → use news_search
- For time/date questions → use datetime
- For any query that mentions a stock, company, or financial index (e.g., "Bank Nifty", "Nifty Fifty", "Nasdaq", "Wall Street", "Tokyo Stock Exchange", or any publicly traded company):
  1. Always perform a web_search to fetch the latest data.
  2. Include latest price, day change, % change, and key events if relevant.
  3. Cite the source or note "according to the latest available data."

**Formatting rules:**
- Always add blank lines before/after tables and lists
- Use **bold** for important terms
- Use proper Markdown syntax
- For image search results: return the raw JSON array exactly as received from the tool

Current time: {current_time}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Simple prompt for non-tool responses
simple_prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"""You are cassy, a helpful and concise AI assistant.

**Formatting rules:**
- Always add blank lines before/after tables and lists
- Use **bold** for important terms  
- Use proper Markdown syntax
- Keep responses focused and helpful

Current time: {current_time}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Ultra-optimized session memory with aggressive caching
async def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Lightning-fast session memory with multi-layer caching"""
    
    memory_data = memory_cache.get(session_id)
    if memory_data:
        value, timestamp = memory_data
        if time.time() - timestamp < 180:
            memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=6)
            for msg_type, content in value:
                if msg_type == "human":
                    memory.chat_memory.add_message(HumanMessage(content=content))
                else:
                    memory.chat_memory.add_message(AIMessage(content=content))
            return memory

    try:
        session_doc = await sessions_collection.find_one(
            {"session_id": session_id}, 
            {"messages": {"$slice": -12}, "_id": 0}
        )
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=6)
        cached_memory = []
        
        if session_doc and "messages" in session_doc:
            for msg in session_doc["messages"][-12:]:
                content = msg["content"][:300]
                if msg["type"] == "human":
                    memory.chat_memory.add_message(HumanMessage(content=content))
                    cached_memory.append(("human", content))
                else:
                    memory.chat_memory.add_message(AIMessage(content=content))
                    cached_memory.append(("ai", content))
        
        memory_cache[session_id] = (cached_memory, time.time())
        return memory
        
    except Exception as e:
        logger.error(f"Memory error: {e}")
        return ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=6)

# Ultra-fast save with write-behind pattern
async def save_conversation(session_id: str, human_message: str, ai_message: str):
    """Fire-and-forget save with immediate cache update"""
    try:
        memory_data = memory_cache.get(session_id)
        if memory_data:
            value, _ = memory_data
            value.append(("human", human_message[:300]))
            value.append(("ai", ai_message[:300]))
            value = value[-12:]
            memory_cache[session_id] = (value, time.time())
    except:
        pass
    
    asyncio.create_task(_batch_save_to_db(session_id, human_message, ai_message))

# Batch database writes for better performance
_pending_writes = {}
_write_lock = asyncio.Lock()

async def _batch_save_to_db(session_id: str, human_message: str, ai_message: str):
    """Batch database writes to reduce I/O"""
    async with _write_lock:
        if session_id not in _pending_writes:
            _pending_writes[session_id] = []
        
        timestamp = datetime.now()
        _pending_writes[session_id].extend([
            {"type": "human", "content": human_message[:800], "timestamp": timestamp},
            {"type": "ai", "content": ai_message[:1500], "timestamp": timestamp}
        ])
        
        if len(_pending_writes[session_id]) >= 8:
            messages = _pending_writes.pop(session_id)
            asyncio.create_task(_write_batch(session_id, messages))

async def _write_batch(session_id: str, messages: list):
    """Write batch to database"""
    try:
        await sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {"$each": messages, "$slice": -30}
                },
                "$set": {"updated_at": datetime.now()}
            },
            upsert=True
        )
    except Exception as e:
        logger.error(f"Batch write error: {e}")

# Pre-initialize LLM with optimized settings
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    streaming=True,
    openai_api_key=OPENAI_API_KEY,
    timeout=12,
    max_retries=2
)

# Optimized response generation
async def generate_streaming_response(
    message: str,
    session_id: str,
    memory: ConversationBufferWindowMemory
) -> AsyncGenerator[str, None]:
    """Generate streaming response with LLM deciding tool usage"""
    try:
        lower_msg = message.lower().strip()
        
        if lower_msg in INSTANT_RESPONSES:
            response_text = INSTANT_RESPONSES[lower_msg]
            yield f"data: {json.dumps({'token': response_text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            asyncio.create_task(save_conversation(session_id, message, response_text))
            return

        vague_queries = {
            "news": "Could you specify a topic for the news (e.g., 'news about technology' or 'latest sports news')?",
            "images": "What specific images are you looking for? (e.g., 'images of mountains' or 'pictures of cats')",
            "pictures": "What specific pictures would you like to see?",
            "photos": "What specific photos are you interested in?"
        }
        
        if lower_msg in vague_queries:
            response_text = vague_queries[lower_msg]
            await save_conversation(session_id, message, response_text)
            yield f"data: {json.dumps({'token': response_text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        tool_indicators = ["current", "latest", "today", "news", "recent", "now", "2024", "2025", 
                         "images", "pictures", "photos", "what time", "who is"]
        might_need_tools = any(indicator in lower_msg for indicator in tool_indicators)

        if not might_need_tools:
            async for chunk in llm.astream(simple_prompt_template.format_messages(
                chat_history=memory.load_memory_variables({})["chat_history"],
                input=message
            )):
                response_text = chunk.content
                yield f"data: {json.dumps({'token': response_text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            asyncio.create_task(save_conversation(session_id, message, response_text))
            return

        if might_need_tools:
            yield f"data: {json.dumps({'token': '🔎 Processing...\n\n'})}\n\n"

        agent = create_openai_functions_agent(llm, tools, prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,
            max_execution_time=15,
            max_iterations=2,
            early_stopping_method="generate"
        )

        try:
            result = await agent_executor.ainvoke({"input": message})
            response_text = result.get("output", "")

            if "image" in lower_msg and response_text:
                try:
                    json.loads(response_text)
                    images = json.loads(response_text)
                    if isinstance(images, dict) and "error" in images:
                        response_text = f"Failed to fetch images: {images['error']}"
                    elif isinstance(images, list):
                        cards = []
                        for img in images:
                            title = img.get("title", "Untitled Image")
                            thumbnail = img.get("thumbnail", img.get("url", ""))
                            url = img.get("url", "")
                            if url:
                                card = f"""**{title}**

![Image]({thumbnail})

Source: [View Original]({url})"""
                                cards.append(card.strip())
                        response_text = "\n\n---\n\n".join(cards) if cards else "No images found for your query."
                except (json.JSONDecodeError, TypeError):
                    pass

            response_text = response_text.replace("\r\n", "\n").strip()
            
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            if "image" in lower_msg:
                response_text = "Unable to fetch images at the moment. Please try again."
            elif "news" in lower_msg:
                response_text = "Unable to fetch news at the moment. Please try again or be more specific."
            else:
                response_text = f"I encountered an error: {str(e)}. Please try rephrasing your question."

        if response_text:
            chunk_size = 80
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i+chunk_size]
                yield f"data: {json.dumps({'token': chunk})}\n\n"
                await asyncio.sleep(0.01)

        yield f"data: {json.dumps({'done': True})}\n\n"
        
        asyncio.create_task(save_conversation(session_id, message, response_text))

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_message = f"❌ Error: {str(e)}"
        yield f"data: {json.dumps({'token': error_message})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cassy – Chat UI</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.5/dist/purify.min.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css" />
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js"></script>
  <style>
    :root {
      --bg: #0b0f0d;
      --text: #e6f2eb;
      --muted: #98a39b;
      --emerald: #10b981;
      --border: #1f2a22;
    }
    * { box-sizing: border-box; }
    html, body { 
      height: 100%; 
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, sans-serif;
    }
    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 0;
    }
    .sidebar {
      background: #111516;
      padding: 16px;
      border-right: 1px solid var(--border);
    }
    .sb-title {
      display: flex; 
      align-items: center; 
      gap: 8px; 
      font-weight: 700;
      margin-bottom: 12px;
    }
    .chat {
      display: flex; 
      flex-direction: column; 
      height: 100vh;
    }
    .header {
      display: flex; 
      align-items: center; 
      justify-content: space-between;
      padding: 14px 18px; 
      border-bottom: 1px solid var(--border); 
      background: #0e1411;
    }
    .brand { 
      display: flex; 
      align-items: center; 
      gap: 10px; 
      font-weight: 800; 
    }
    .brand .dot { 
      width: 10px; 
      height: 10px; 
      border-radius: 50%; 
      background: var(--emerald); 
      box-shadow: 0 0 18px var(--emerald); 
    }
    .messages { 
      flex: 1; 
      overflow: auto; 
      padding: 24px; 
    }
    .row { 
      display: flex; 
      gap: 10px; 
      margin: 10px 0; 
    }
    .row.user { 
      justify-content: flex-end; 
    }
    .bubble {
      max-width: 70%; 
      padding: 12px 14px; 
      border-radius: 14px; 
      border: 1px solid var(--border); 
      background: #0e1411;
      word-wrap: break-word;
      overflow-wrap: break-word;
      white-space: normal;
      line-height: 1.5;
    }
    .row.user .bubble { 
      background: #1a2a1e; 
      border-color: #1c3b28; 
    }
    .avatar { 
      width: 36px; 
      height: 36px; 
      border-radius: 50%; 
      display: flex;
      align-items: center;
      justify-content: center;
      background: #0f1812; 
      border: 1px solid var(--border); 
      flex-shrink: 0;
    }
    .composer {
      display: flex; 
      gap: 8px; 
      align-items: center; 
      background: #0f1812; 
      border: 1px solid var(--border);
      border-radius: 12px; 
      padding: 8px;
      margin: 12px;
    }
    .composer input {
      flex: 1; 
      background: transparent; 
      border: 0; 
      color: var(--text); 
      font-size: 15px; 
      outline: none;
      padding: 8px;
    }
    .btn, .mic {
      width: 40px; 
      height: 40px; 
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 10px; 
      border: 1px solid var(--border);
      background: #0e1411; 
      color: var(--text); 
      cursor: pointer;
      font-size: 16px;
      flex-shrink: 0;
    }
    .btn.primary { 
      background: linear-gradient(180deg, #16a34a, #10b981); 
      border: none; 
    }
    .btn:disabled { 
      opacity: 0.6; 
      cursor: not-allowed; 
    }
    .mic.record { 
      background: linear-gradient(180deg, #dc2626, #ef4444); 
      border: none; 
    }
    .welcome {
      display: flex; 
      flex-direction: column; 
      align-items: center; 
      justify-content: center;
      height: 100%; 
      text-align: center; 
      padding: 40px;
    }
    .welcome-title { 
      font-size: 28px; 
      font-weight: 700; 
      margin-bottom: 12px; 
    }
    .welcome-subtitle { 
      color: var(--muted); 
      font-size: 16px; 
      margin-bottom: 24px; 
    }
    .typing { 
      display: flex; 
      gap: 6px; 
      align-items: center; 
    }
    .dot { 
      width: 8px; 
      height: 8px; 
      border-radius: 50%; 
      background: #2a3b31; 
      animation: blink 1.2s infinite ease-in-out; 
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink { 
      0%, 80%, 100% { opacity: 0.2; } 
      40% { opacity: 1; } 
    }
    .bubble.markdown {
      white-space: normal;
    }
    .bubble h1, .bubble h2, .bubble h3, .bubble h4, .bubble h5, .bubble h6 { 
      margin: 16px 0 8px 0; 
      color: var(--emerald); 
      font-weight: 600;
      line-height: 1.3;
    }
    .bubble h1 { font-size: 1.5em; }
    .bubble h2 { font-size: 1.3em; }
    .bubble h3 { font-size: 1.1em; }
    .bubble p { 
      margin: 8px 0; 
      line-height: 1.5;
    }
    .bubble p:first-child { margin-top: 0; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble code { 
      background: #0b1510; 
      padding: 3px 6px; 
      border-radius: 4px; 
      color: #86efac; 
      font-family: 'Courier New', monospace;
      font-size: 0.9em;
    }
    .bubble pre { 
      background: #0b1510; 
      padding: 12px; 
      border-radius: 8px; 
      margin: 12px 0; 
      overflow-x: auto;
      border: 1px solid var(--border);
    }
    .bubble pre code {
      background: transparent;
      padding: 0;
      color: #86efac;
    }
    .bubble ul, .bubble ol { 
      margin: 12px 0; 
      padding-left: 24px; 
      color: var(--text);
      width: 100%;
      max-width: 100%;
    }
    .bubble ul {
      list-style-type: disc;
    }
    .bubble ol {
      list-style-type: decimal;
    }
    .bubble li {
      margin: 6px 0;
      line-height: 1.5;
      color: var(--text);
    }
    .bubble li::marker {
      color: var(--emerald);
    }
    .bubble ul ul, .bubble ol ol, .bubble ul ol, .bubble ol ul {
      margin: 4px 0;
    }
    .bubble strong { 
      color: #86efac; 
      font-weight: 600;
    }
    .bubble em {
      color: #a7f3d0;
      font-style: italic;
    }
    .bubble blockquote {
      border-left: 3px solid var(--emerald);
      margin: 12px 0;
      padding-left: 12px;
      color: var(--muted);
      font-style: italic;
    }
    .bubble a {
      color: var(--emerald);
      text-decoration: underline;
    }
    .bubble a:hover {
      color: #86efac;
    }
    .bubble table {
      border-collapse: collapse;
      width: 100%;
      max-width: 100%;
      margin: 12px 0;
      background: #0b1510;
      border-radius: 6px;
      overflow: hidden;
      table-layout: auto;
    }
    .bubble th, .bubble td {
      border: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }
    .bubble th {
      background: #1a2a1e;
      color: var(--emerald);
      font-weight: 600;
    }
    .bubble td {
      color: var(--text);
    }
    .bubble tr:nth-child(even) td {
      background: rgba(16, 185, 129, 0.05);
    }
    .bubble img.md-img {
      max-width: 300px;
      height: 200px;
      object-fit: cover;
      border-radius: 8px;
      margin: 5px 0;
      display: block;
    }
    .bubble hr {
      border: none;
      height: 1px;
      background: var(--border);
      margin: 16px 0;
    }
  </style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="sb-title">🟢 <span>cassy</span></div>
    <div style="background: #0e1411; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin-top: 10px; color: var(--muted);">Optimized AI Assistant</div>
    <div style="background: #0e1411; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin-top: 10px; color: var(--muted);">Now faster with smart function calling!</div>
  </aside>
  <main class="chat">
    <div class="header">
      <div class="brand">
        <div class="dot"></div> 
        <div>cassy v2.0</div>
      </div>
      <div style="color:#86efac">online</div>
    </div>
    <div id="messages" class="messages">
      <div class="welcome" id="welcome">
        <div class="welcome-title">👋 Hey there!</div>
        <div class="welcome-subtitle">I'm cassy v2.0 - faster and smarter! Ask me anything!</div>
      </div>
    </div>
    <div class="composer">
      <input type="text" id="messageInput" placeholder="Ask cassy anything..." />
      <button type="button" id="micButton" class="mic">🎤</button>
      <button type="button" id="sendButton" class="btn primary">➤</button>
    </div>
  </main>
</div>
<script>
function debounce(func, wait) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}
let sessionId = null;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
const BASE_URL = window.location.origin;
const messagesDiv = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const micButton = document.getElementById('micButton');
const welcomeDiv = document.getElementById('welcome');

const debouncedRenderMarkdown = debounce((bubble, text) => {
  if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
    bubble.textContent = text;
    return;
  }
  try {
    bubble.classList.add('markdown');
    let processed = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function(_, alt, src) {
      return `<img class="md-img" src="${src}" alt="${alt}" />`;
    });
    marked.setOptions({ 
      gfm: true, 
      breaks: true, 
      tables: true,
      headerIds: false,
      mangle: false
    });
    const rawHtml = marked.parse(processed);
    const safeHtml = DOMPurify.sanitize(rawHtml, {
      USE_PROFILES: { html: true },
      ALLOWED_TAGS: [
        'p', 'br', 'strong', 'b', 'em', 'i', 'code', 'pre',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'dl', 'dt', 'dd',
        'blockquote', 'a', 'hr',
        'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
        'colgroup', 'col', 'img', 'span', 'div'
      ],
      ALLOWED_ATTR: [
        'href', 'target', 'rel', 'colspan', 'rowspan', 'align', 
        'src', 'alt', 'title', 'class', 'id', 'style'
      ],
      ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|cid|xmpp|data):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i
    });
    bubble.innerHTML = safeHtml;
    if (typeof Prism !== 'undefined') {
      try {
        Prism.highlightAllUnder(bubble);
      } catch (e) {}
    }
  } catch (e) {
    bubble.classList.remove('markdown');
    bubble.textContent = text;
  }
}, 50);

function hideWelcome() {
  if (welcomeDiv) welcomeDiv.style.display = 'none';
}

function addMessage(text, isUser = false) {
  hideWelcome();
  const row = document.createElement('div');
  row.className = 'row' + (isUser ? ' user' : '');
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (isUser) {
    bubble.textContent = text;
  } else {
    debouncedRenderMarkdown(bubble, text);
  }
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.innerHTML = isUser ? '👤' : '🤖';
  if (isUser) {
    row.appendChild(bubble);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(bubble);
  }
  messagesDiv.appendChild(row);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  return bubble;
}

function showTyping() {
  const row = document.createElement('div');
  row.className = 'row';
  row.id = 'typing-indicator';
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.innerHTML = '🤖';
  const bubble = document.createElement('div');
  bubble.className = 'bubble typing';
  bubble.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesDiv.appendChild(row);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function hideTyping() {
  const typing = document.getElementById('typing-indicator');
  if (typing) typing.remove();
}

async function sendMessage() {
  const message = messageInput.value.trim();
  if (!message) return;
  
  addMessage(message, true);
  messageInput.value = '';
  sendButton.disabled = true;
  showTyping();
  
  try {
    const response = await fetch(`${BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    hideTyping();
    let botBubble = null;
    let fullResponse = '';
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        if (botBubble && fullResponse.trim()) debouncedRenderMarkdown(botBubble, fullResponse);
        break;
      }
      
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.session_id && !sessionId) sessionId = data.session_id;
            if (data.token) {
              if (!botBubble) {
                botBubble = addMessage('', false);
              }
              fullResponse += data.token;
              debouncedRenderMarkdown(botBubble, fullResponse);
              messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
          } catch (e) {}
        }
      }
    }
  } catch (error) {
    hideTyping();
    addMessage(`Sorry, there was an error: ${error.message}. Please check if the server is running.`, false);
  } finally {
    sendButton.disabled = false;
  }
}

async function startRecording() {
  if (!navigator.mediaDevices) { 
    alert('Microphone not available'); 
    return; 
  }
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    
    mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const arrayBuffer = await audioBlob.arrayBuffer();
      const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      
      showTyping();
      
      try {
        const response = await fetch(`${BASE_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: '', voice_data: base64, session_id: sessionId })
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        let userMessageAdded = false;
        let botBubble = null;
        let fullResponse = '';
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            if (botBubble && fullResponse.trim()) debouncedRenderMarkdown(botBubble, fullResponse);
            break;
          }
          
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.session_id && !sessionId) sessionId = data.session_id;
                if (data.kind === 'transcript' && data.token && !userMessageAdded) {
                  hideTyping();
                  addMessage(data.token, true);
                  userMessageAdded = true;
                  showTyping();
                } else if (data.token && !data.kind) {
                  if (!botBubble) {
                    hideTyping();
                    botBubble = addMessage('', false);
                  }
                  fullResponse += data.token;
                  debouncedRenderMarkdown(botBubble, fullResponse);
                  messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }
              } catch (e) {}
            }
          }
        }
      } catch (error) {
        hideTyping();
        addMessage(`Sorry, there was an error processing your voice: ${error.message}.`, false);
      }
      
      stream.getTracks().forEach(track => track.stop());
    };
    
    mediaRecorder.start();
    isRecording = true;
    micButton.classList.add('record');
    micButton.innerHTML = '⏹';
    setTimeout(stopRecording, 10000);
  } catch (error) {
    alert('Could not access microphone: ' + error.message);
  }
}

function stopRecording() {
  if (isRecording && mediaRecorder) {
    mediaRecorder.stop();
    isRecording = false;
    micButton.classList.remove('record');
    micButton.innerHTML = '🎤';
  }
}

function attachEventListeners() {
  if (sendButton) sendButton.addEventListener('click', function(e) { 
    e.preventDefault(); 
    sendMessage(); 
  });
  
  if (messageInput) messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) { 
      e.preventDefault(); 
      sendMessage(); 
    }
  });
  
  if (micButton) micButton.addEventListener('click', function(e) {
    e.preventDefault(); 
    if (isRecording) stopRecording(); 
    else startRecording();
  });
}

document.addEventListener('DOMContentLoaded', function() {
  attachEventListeners();
  if (messageInput) messageInput.focus();
});

if (document.readyState !== 'loading') {
  attachEventListeners();
  if (messageInput) messageInput.focus();
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html_content.replace('\n', ' ').strip())

@app.post("/chat/stream")
async def chat_stream(chat_message: ChatMessage):
    try:
        session_id = chat_message.session_id or str(uuid.uuid4())
        message = chat_message.message or ""
        transcript = None
        
        if chat_message.voice_data:
            voice_text = await process_voice_input(chat_message.voice_data)
            if voice_text and voice_text != "Could not process voice input":
                transcript = voice_text.strip()
                message = transcript

        memory = await get_session_memory(session_id)

        async def stream_wrapper():
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            if transcript:
                yield f"data: {json.dumps({'token': transcript, 'kind': 'transcript'})}\n\n"
            async for chunk in generate_streaming_response(message, session_id, memory):
                yield chunk

        return StreamingResponse(
            stream_wrapper(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache", 
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )

    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_non_stream(chat_message: ChatMessage):
    """Non-streaming endpoint for direct API access"""
    try:
        session_id = chat_message.session_id or str(uuid.uuid4())
        message = chat_message.message or ""
        
        if chat_message.voice_data:
            voice_text = await process_voice_input(chat_message.voice_data)
            if voice_text and voice_text != "Could not process voice input":
                message = voice_text.strip()

        memory = await get_session_memory(session_id)
        
        agent = create_openai_functions_agent(llm, tools, prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,
            max_execution_time=15,
            max_iterations=2,
            early_stopping_method="generate"
        )

        try:
            result = await agent_executor.ainvoke({"input": message})
            response_text = result.get("output", "").replace("\r\n", "\n").strip()
            
            lower_msg = message.lower()
            if "image" in lower_msg and response_text:
                try:
                    images = json.loads(response_text)
                    if isinstance(images, dict) and "error" in images:
                        response_text = f"Failed to fetch images: {images['error']}"
                    elif isinstance(images, list):
                        cards = []
                        for img in images:
                            title = img.get("title", "Untitled Image")
                            thumbnail = img.get("thumbnail", img.get("url", ""))
                            url = img.get("url", "")
                            if url:
                                card = f"""**{title}**

![Image]({thumbnail})

Source: [View Original]({url})"""
                                cards.append(card.strip())
                        response_text = "\n\n---\n\n".join(cards) if cards else "No images found for your query."
                except (json.JSONDecodeError, TypeError):
                    pass

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            response_text = f"I encountered an error: {str(e)}. Please try rephrasing your question."

        await save_conversation(session_id, message, response_text)
        return SessionResponse(session_id=session_id, message=response_text)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health and utility endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "version": "2.0",
        "timestamp": datetime.now(),
        "cache_stats": {
            "memory_cache_size": len(memory_cache),
            "tool_cache_sizes": {k: len(v) for k, v in tool_cache.items()}
        }
    }

@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches"""
    try:
        tool_cache.clear()
        memory_cache.clear()
        get_cached_result.cache_clear()
        logger.info("All caches cleared")
        return {"status": "success", "message": "All caches cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "version": "2.0",
        "cache_stats": {
            "memory_cache_size": len(memory_cache),
            "tool_cache_sizes": {k: len(v) for k, v in tool_cache.items()},
            "lru_cache_info": get_cached_result.cache_info()._asdict()
        },
        "tools_available": [tool.name for tool in tools],
        "timestamp": datetime.now()
    }

# Background tasks
async def cleanup_caches():
    while True:
        try:
            for cache_type, cache_dict in list(tool_cache.items()):
                with cache_locks[cache_type]:
                    for key, (value, timestamp) in list(cache_dict.items()):
                        if time.time() - timestamp > CACHE_TTL:
                            del cache_dict[key]
            await asyncio.sleep(300)
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

async def flush_pending_writes():
    while True:
        try:
            async with _write_lock:
                for session_id, messages in list(_pending_writes.items()):
                    if messages:
                        await _write_batch(session_id, messages)
                        _pending_writes[session_id] = []
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Pending writes flush error: {e}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database and start background tasks"""
    global global_session
    try:
        global_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5),
            timeout=aiohttp.ClientTimeout(total=6, connect=2)
        )
        await sessions_collection.create_index("session_id", unique=True)
        await sessions_collection.create_index("updated_at")
        await sessions_collection.create_index([("messages.timestamp", -1)])
        asyncio.create_task(cleanup_caches())
        asyncio.create_task(flush_pending_writes())
        logger.info("🚀 Cassy v2.1 ULTRA-FAST started successfully")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    try:
        if global_session:
            await global_session.close()
        if _pending_writes:
            for session_id, messages in _pending_writes.items():
                if messages:
                    await _write_batch(session_id, messages)
        thread_pool.shutdown(wait=True)
        await client.close()
        logger.info("✅ Cassy v2.1 shutdown complete")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=PORT, 
        log_level="warning",
        access_log=False,
        reload=False,
        workers=1,
        loop="auto"
    )



