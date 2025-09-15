import os
import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, AsyncGenerator
import logging
from dotenv import load_dotenv
import re
import aiohttp
from functools import lru_cache, wraps
import time

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key")
SERPAPI_KEY = os.getenv("SERP_API_KEY", "your-serpapi-key")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "your-gnews-key")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
PORT = int(os.getenv("PORT", 8000))

# Database client
client = AsyncIOMotorClient(MONGODB_URL)
db = client.chatbot_db
sessions_collection = db.sessions
conversations_collection = db.conversations

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    voice_data: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    message: str

# Search gating
RECENCY_TERMS = {
    "today", "this week", "this month", "latest", "current", "recent",
    "breaking", "news", "update", "updated", "now", "live", "real-time",
    "2024", "2025"
}
PERSON_TRIGGERS = [
    r"\bwho is\b", r"\bbiography\b", r"\bage\b", r"\bborn\b", r"\bdied\b",
    r"\bceo\b", r"\bprime minister\b", r"\bpresident\b"
]
IMAGE_TRIGGERS = {
    "image", "images", "picture", "pictures", "photo", "photos", "pic", "pics", "img"
}

def _is_current_query(t: str) -> bool:
    tl = t.lower()
    return any(term in tl for term in RECENCY_TERMS)

def _mentions_person(t: str) -> bool:
    return any(re.search(p, t, re.IGNORECASE) for p in PERSON_TRIGGERS)

def is_image_query(t: str) -> bool:
    tl = t.lower()
    return any(term in tl for term in IMAGE_TRIGGERS)

def should_use_web_search(message: str) -> bool:
    t = (message or "").strip().lower()
    if is_image_query(t):
        return True  # Allow image searches only for explicit image queries
    if _mentions_person(t):
        return True  # Allow web search for bio details but not images
    if _is_current_query(t):
        return True  # Allow web search for recent queries
    return False

# Custom cache for time-sensitive data
def timed_lru_cache(seconds: int):
    def decorator(func):
        @lru_cache(maxsize=100)
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs)
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = (args, tuple(sorted(kwargs.items())))
            if cache_key in wrapper.cache:
                result, timestamp = wrapper.cache[cache_key]
                if time.time() - timestamp < seconds:
                    return result
            result = await func(*args, **kwargs)
            wrapper.cache[cache_key] = (result, time.time())
            return result
        wrapper.cache = {}
        wrapper.clear_cache = lambda: [wrapper.cache.clear(), cached_func.cache_clear()]
        return wrapper
    return decorator

# Tools
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information using SerpAPI. Keep outputs concise."
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, WebSearchTool) and self.name == other.name

    @timed_lru_cache(seconds=3600)
    async def _arun(self, query: str) -> str:
        try:
            if not SERPAPI_KEY or SERPAPI_KEY == "your-serpapi-key":
                logger.error("Invalid or missing SERPAPI_KEY")
                return json.dumps([])

            async with aiohttp.ClientSession() as session:
                url = "https://serpapi.com/search"
                params = {
                    "engine": "google",
                    "q": query,
                    "api_key": SERPAPI_KEY,
                    "num": 6,
                    "ijn": 0
                }
                logger.info(f"Sending web search request for query: {query}")
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"Web search API returned status: {response.status} - {await response.text()}")
                        return json.dumps([])
                    data = await response.json()
                    logger.debug(f"SerpAPI response: {json.dumps(data)[:500]}...")
                    results = []
                    for item in (data.get("organic_results") or [])[:3]:
                        title = item.get("title", "Untitled")
                        snippet = item.get("snippet", "")
                        link = item.get("link", "")
                        if link:
                            results.append({"title": title, "snippet": snippet, "url": link})
                    if not results:
                        logger.warning(f"No valid results found for query: {query}")
                    return json.dumps(results)
        except Exception as e:
            logger.error(f"Web search error for query '{query}': {str(e)}")
            return json.dumps([])

    def _run(self, query: str) -> str:
        return asyncio.run(self._arun(query))

class ImageSearchTool(BaseTool):
    name: str = "image_search"
    description: str = "Search for images using SerpAPI. Returns a JSON array of image results with title, url, and thumbnail, or an error object."
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, ImageSearchTool) and self.name == other.name

    @timed_lru_cache(seconds=3600)
    async def _arun(self, query: str) -> str:
        try:
            if not SERPAPI_KEY or SERPAPI_KEY == "your-serpapi-key":
                logger.error("Invalid or missing SERPAPI_KEY")
                return json.dumps({"error": "Invalid or missing SerpAPI key. Please configure a valid key."})

            async with aiohttp.ClientSession() as session:
                url = "https://serpapi.com/search"
                params = {
                    "engine": "google",
                    "q": query,
                    "tbm": "isch",
                    "api_key": SERPAPI_KEY,
                    "num": 6,
                    "ijn": 0
                }
                logger.info(f"Sending image search request for query: {query}")
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Image search API returned status: {response.status} - {error_text}")
                        if response.status == 401:
                            return json.dumps({"error": "Invalid SerpAPI key. Please check your API key."})
                        elif response.status == 429:
                            return json.dumps({"error": "SerpAPI rate limit exceeded. Please try again later."})
                        return json.dumps({"error": f"Image search API error: HTTP {response.status}"})
                    data = await response.json()
                    logger.debug(f"SerpAPI response: {json.dumps(data)[:500]}...")
                    if "error" in data:
                        logger.error(f"SerpAPI returned error: {data['error']}")
                        return json.dumps({"error": f"SerpAPI error: {data['error']}"})
                    images = []
                    for img in (data.get("images_results") or [])[:3]:
                        title = img.get("title", "Untitled Image")
                        original = img.get("original", "")
                        thumbnail = img.get("thumbnail", original)
                        if original:
                            images.append({"title": title, "url": original, "thumbnail": thumbnail})
                    return json.dumps(images)
        except Exception as e:
            logger.error(f"Image search error for query '{query}': {str(e)}")
            return json.dumps({"error": f"Image search failed: {str(e)}"})

    def _run(self, query: str) -> str:
        return asyncio.run(self._arun(query))

class NewsSearchTool(BaseTool):
    name: str = "news_search"
    description: str = "Search for latest news using GNews API. Returns formatted news cards with images and headlines."

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, NewsSearchTool):
            return False
        return self.name == other.name

    async def _arun(self, query: str) -> str:
        logger.info(f"Running NewsSearchTool for query: {query}")
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://gnews.io/api/v4/search"
                params = {"q": query, "token": GNEWS_API_KEY, "max": 3, "lang": "en"}
                async with session.get(url, params=params, timeout=10) as response:
                    data = await response.json()
                    logger.info(f"GNews API response: {json.dumps(data)[:100]}...")
                    articles = data.get("articles") or []
                    if not articles:
                        return "No news found"

                    results = []
                    for article in articles[:3]:
                        title = article.get("title", "No title")
                        url = article.get("url", "")
                        image_url = article.get("image", "https://example.com/placeholder.jpg")
                        published = article.get("publishedAt", "Unknown date")

                        # Format as news card
                        card = f"""
**{title}**

![News Image]({image_url})

Published: {published}
Source: [{url}]({url})
"""
                        results.append(card.strip())
                    response_text = "\n\n---\n\n".join(results) if results else "No news found"
                    logger.info(f"Generated news response: {response_text[:100]}...")
                    return response_text
        except Exception as e:
            logger.error(f"News search async error: {e}")
            return f"News search error: {str(e)}"

    def _run(self, query: str) -> str:
        return asyncio.run(self._arun(query))

class DateTimeTool(BaseTool):
    name: str = "datetime"
    description: str = "Get current date and time."

    def _run(self, query: str = "") -> str:
        now = datetime.now()
        return f"**Current date and time:** {now.strftime('%Y-%m-%d %H:%M:%S')}"

# Voice processing
def process_voice_input(voice_data: str) -> str:
    try:
        audio_bytes = base64.b64decode(voice_data)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        return "Could not process voice input"

# FastAPI app
app = FastAPI(title="cassy", description="Conversational AI assistant with voice and smart search")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tools list
tools = [WebSearchTool(), ImageSearchTool(), NewsSearchTool(), DateTimeTool()]

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are cassy, a concise, friendly, expert AI assistant.

Search policy:
- Only use external tools for current events, breaking news, live updates, or clearly time-sensitive queries.
- For queries like "who is [person]", use the web_search tool to gather biographical details and structure them in a Markdown card with fields: **Name**, **Occupation**, **Born**, **Nationality**. Do NOT use the image_search tool for these queries.
- For queries requesting images, pictures, or photos, use the image_search tool. **YOU MUST RETURN THE RAW JSON OUTPUT FROM image_search (array or error object) EXACTLY AS RECEIVED, WITHOUT ANY MODIFICATION, REFORMATTING, OR CONVERSION TO MARKDOWN.** Do not add text like "Here are some images" or format results as a list.
- Do not use external tools for general explanations, definitions, or evergreen topics.

MANDATORY FORMATTING - YOU MUST FOLLOW THESE RULES:
1. ALWAYS add blank lines before and after tables.
2. ALWAYS add blank lines before and after lists.
3. Use **bold** for important terms.
4. Use proper Markdown syntax for tables, lists, and images.
5. For tables, ensure headers are separated by a row of `|---|` and columns are aligned.
6. For lists, use `-` for each item, one per line, with no inline lists.
7. For news, format as cards with image, headline, publication date, and source link.
8. For image_search, **return only the raw JSON array or error object as-is, with no additional text or formatting.**

Current datetime: 2025-09-09 15:02:00 IST"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

simple_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are cassy, a concise, friendly, expert AI assistant.

Search policy:
- Only use external tools for current events, breaking news, live updates, or clearly time-sensitive queries.
- Do not use external tools for general explanations, definitions, or evergreen topics.

MANDATORY FORMATTING - YOU MUST FOLLOW THESE RULES:
1. ALWAYS add blank lines before and after tables.
2. ALWAYS add blank lines before and after lists.
3. Use **bold** for important terms.
4. Use proper Markdown syntax for tables and lists.
5. For tables, ensure headers are separated by a row of `|---|` and columns are aligned.
6. For lists, use `-` for each item, one per line, with no inline lists.
7. For image_search, **do not format the output; return the raw JSON array or error object as-is.**

Example of CORRECT formatting:

Here is some text.

**Fruit List:**

- Apple
- Banana
- Orange

**Comparison Table:**

| Fruit  | Color  | Taste       |
|--------|--------|-------------|
| Apple  | Red    | Sweet/Tart  |
| Orange | Orange | Citrusy     |

Current datetime: 2025-09-09 15:02:00 IST"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Session memory
async def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    try:
        session_doc = await sessions_collection.find_one({"session_id": session_id})
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)
        if session_doc and "messages" in session_doc:
            for msg in session_doc["messages"]:
                if msg["type"] == "human":
                    memory.chat_memory.add_message(HumanMessage(content=msg["content"]))
                else:
                    memory.chat_memory.add_message(AIMessage(content=msg["content"]))
        return memory
    except Exception as e:
        logger.error(f"Error retrieving session memory: {e}")
        return ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)

async def save_conversation(session_id: str, human_message: str, ai_message: str):
    try:
        await sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {"type": "human", "content": human_message, "timestamp": datetime.now()},
                            {"type": "ai", "content": ai_message, "timestamp": datetime.now()}
                        ]
                    }
                },
                "$set": {"updated_at": datetime.now()}
            },
            upsert=True
        )
        await conversations_collection.insert_one({
            "session_id": session_id,
            "human_message": human_message,
            "ai_message": ai_message,
            "timestamp": datetime.now(),
            "used_web_search": should_use_web_search(human_message)
        })
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")

# Markdown normalization
def normalize_markdown(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n")
    fence = "`" * 3
    if s.count(fence) % 2 == 1:
        s += "\n" + fence
    return s

def fix_markdown_line(line: str, prev_line: str, next_line: str = '') -> str:
    stripped = line.strip()
    result = line
    if stripped.startswith('|') or stripped.endswith('|'):
        if prev_line.strip() and not prev_line.strip().startswith(('- ', '|')):
            result = '\n' + line
    elif prev_line.strip().startswith(('- ', '|')) and not stripped.startswith(('- ', '|')):
        result = '\n' + line
    if stripped.startswith('* '):
        result = result.replace('* ', '- ')
    return result

# Streaming generator
async def generate_streaming_response(
    message: str,
    session_id: str,
    llm: ChatOpenAI,
    memory: ConversationBufferWindowMemory
) -> AsyncGenerator[str, None]:
    try:
        needs_search = should_use_web_search(message)
        logger.info(f"Message: {message[:80]} | Needs search: {needs_search}")

        if not needs_search:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            formatted_prompt = simple_prompt_template.partial(datetime=current_time)
            prompt_value = formatted_prompt.format_prompt(
                chat_history=memory.chat_memory.messages,
                input=message
            )
            resp = await llm.agenerate([prompt_value.to_messages()])
            response_text = normalize_markdown(resp.generations[0][0].text)
            memory.chat_memory.add_message(HumanMessage(content=message))
            memory.chat_memory.add_message(AIMessage(content=response_text))
            await save_conversation(session_id, message, response_text)

            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                prev_line = lines[i-1] if i > 0 else ''
                next_line = lines[i+1] if i < len(lines)-1 else ''
                fixed_line = fix_markdown_line(line, prev_line, next_line)
                if fixed_line.strip():
                    yield f"data: {json.dumps({'token': fixed_line + '\n'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        if message.lower().strip() in ["give some news", "news", "tell news"]:
            response_text = "Could you specify a topic or region for the news (e.g., 'news on Korea politics' or 'latest sports news')?"
            await save_conversation(session_id, message, response_text)
            yield f"data: {json.dumps({'token': response_text + '\n'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        if message.lower().strip() in ["images", "pictures", "photos", "give some images", "show images", "tell images"]:
            response_text = "Could you specify what images you want (e.g., 'images of the Eiffel Tower' or 'pictures of cats')?"
            await save_conversation(session_id, message, response_text)
            yield f"data: {json.dumps({'token': response_text + '\n'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_prompt = prompt_template.partial(datetime=current_time)
        financial_query = re.search(r"\b(stock|price|sensex|nifty|market)\b", message.lower())
        news_search_tool = next((tool for tool in tools if tool.name == "news_search"), None)
        image_search_tool = next((tool for tool in tools if tool.name == "image_search"), None)
        image_query = is_image_query(message)

        active_tools = tools
        if image_query and image_search_tool:
            active_tools = [image_search_tool]
        elif "news" in message.lower() and not financial_query and news_search_tool:
            active_tools = [news_search_tool]
        elif _mentions_person(message) and not image_query:
            active_tools = [tool for tool in tools if tool.name != "image_search"]

        agent = create_openai_functions_agent(llm, active_tools, formatted_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=active_tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_execution_time=20,
            max_iterations=5
        )

        yield f"data: {json.dumps({'token': '🔎 Searching...\n\n'})}\n\n"

        try:
            logger.info(f"Invoking agent with input: {message}, active tools: {[tool.name for tool in active_tools]}")
            result = await agent_executor.ainvoke({"input": message})
            response_text = result.get("output", "")

            # Handle image search results
            if image_query and response_text:
                try:
                    images = json.loads(response_text)
                    if isinstance(images, dict) and "error" in images:
                        logger.error(f"Image search tool returned error: {images['error']}")
                        response_text = f"Failed to fetch images for '{message}': {images['error']}. Please try again later or check your API configuration."
                    elif isinstance(images, list):
                        cards = []
                        for img in images:
                            title = img.get("title", "Untitled Image")
                            thumbnail = img.get("thumbnail", img.get("url", ""))
                            url = img.get("url", "")
                            if url:
                                card = f"""
**{title}**

![Image]({thumbnail})

Source: [View Original]({url})
"""
                                cards.append(card.strip())
                        response_text = "\n\n---\n\n".join(cards) if cards else f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
                    else:
                        logger.warning(f"Unexpected image search result format: {response_text}")
                        response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse image_search JSON: {response_text}, error: {str(e)}")
                    response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
            response_text = normalize_markdown(response_text)
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            if image_query:
                response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
            elif "news" in message.lower():
                response_text = """
**Recent News Fallback:**

- **Korea News**: Unable to fetch live news due to technical issues. Try specifying a topic like 'Korea politics news' or check sources like The Korea Herald (https://www.koreaherald.com) for updates.
"""
            else:
                response_text = f"Error fetching results: {str(e)}. Please try again or rephrase your query."
            response_text = normalize_markdown(response_text)
            yield f"data: {json.dumps({'token': response_text + '\n'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            await save_conversation(session_id, message, response_text)
            return

        lines = response_text.split('\n')
        for i, line in enumerate(lines):
            prev_line = lines[i-1] if i > 0 else ''
            next_line = lines[i+1] if i < len(lines)-1 else ''
            fixed_line = fix_markdown_line(line, prev_line, next_line)
            if fixed_line.strip():
                yield f"data: {json.dumps({'token': fixed_line + '\n'})}\n\n"
                await asyncio.sleep(0.005)
        yield f"data: {json.dumps({'done': True})}\n\n"

        await save_conversation(session_id, message, response_text)

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_message = f"❌ Error: {str(e)}"
        yield f"data: {json.dumps({'token': error_message})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

# Cache clearing endpoint
@app.post("/clear-cache")
async def clear_cache():
    try:
        # Clear caches for WebSearchTool and ImageSearchTool
        WebSearchTool._arun.clear_cache()
        ImageSearchTool._arun.clear_cache()
        logger.info("Caches cleared for WebSearchTool and ImageSearchTool")
        return {"status": "success", "message": "Caches cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

# UI (unchanged, included for completeness)
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
    <div style="background: #0e1411; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin-top: 10px; color: var(--muted);">Welcome to cassy</div>
    <div style="background: #0e1411; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin-top: 10px; color: var(--muted);">Try asking anything!</div>
  </aside>
  <main class="chat">
    <div class="header">
      <div class="brand">
        <div class="dot"></div> 
        <div>cassy</div>
      </div>
      <div style="color:#86efac">online</div>
    </div>
    <div id="messages" class="messages">
      <div class="welcome" id="welcome">
        <div class="welcome-title">👋 Hey there!</div>
        <div class="welcome-subtitle">I'm cassy, your AI assistant. Ask me anything!</div>
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
console.log('Page loaded, elements found:', {
  messagesDiv: !!messagesDiv,
  messageInput: !!messageInput,
  sendButton: !!sendButton,
  micButton: !!micButton
});
const debouncedRenderMarkdown = debounce((bubble, text) => {
  if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
    console.warn('marked or DOMPurify not available');
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
      } catch (e) {
        console.warn('Prism highlight failed:', e);
      }
    }
  } catch (e) {
    console.warn('Markdown rendering failed, falling back to plain text:', e);
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
  console.log('Sending message:', message, 'to:', `${BASE_URL}/chat/stream`);
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
      console.error('Fetch error:', response.status, response.statusText);
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
          } catch (e) {
            console.warn('Parse error:', e);
          }
        }
      }
    }
  } catch (error) {
    console.error('Error:', error);
    hideTyping();
    addMessage(`Sorry, there was an error: ${error.message}. Please check if the server is running at ${BASE_URL}.`, false);
  } finally {
    sendButton.disabled = false;
  }
}
async function startRecording() {
  if (!navigator.mediaDevices) { alert('Microphone not available'); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const arrayBuffer = await audioBlob.arrayBuffer();
      const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      console.log('Sending voice data to:', `${BASE_URL}/chat/stream`);
      showTyping();
      try {
        const response = await fetch(`${BASE_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: '', voice_data: base64, session_id: sessionId })
        });
        if (!response.ok) {
          console.error('Fetch error:', response.status, response.statusText);
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
              } catch (e) {
                console.warn('Parse error:', e);
              }
            }
          }
        }
      } catch (error) {
        console.error('Voice error:', error);
        hideTyping();
        addMessage(`Sorry, there was an error processing your voice: ${error.message}. Please check if the server is running at ${BASE_URL}.`, false);
      }
      stream.getTracks().forEach(track => track.stop());
    };
    mediaRecorder.start();
    isRecording = true;
    micButton.classList.add('record');
    micButton.innerHTML = '⏹';
    setTimeout(stopRecording, 10000);
  } catch (error) {
    console.error('Microphone error:', error);
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
  if (sendButton) sendButton.addEventListener('click', function(e) { e.preventDefault(); sendMessage(); });
  if (messageInput) messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  if (micButton) micButton.addEventListener('click', function(e) {
    e.preventDefault(); if (isRecording) stopRecording(); else startRecording();
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
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat/stream")
async def chat_stream(chat_message: ChatMessage):
    try:
        session_id = chat_message.session_id or str(uuid.uuid4())
        message = chat_message.message or ""
        transcript = None
        if chat_message.voice_data:
            voice_text = process_voice_input(chat_message.voice_data)
            if voice_text and voice_text != "Could not process voice input":
                transcript = voice_text.strip()
                message = transcript

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            streaming=True,
            openai_api_key=OPENAI_API_KEY,
            timeout=10
        )

        memory = await get_session_memory(session_id)

        async def stream_wrapper():
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            if transcript:
                yield f"data: {json.dumps({'token': transcript, 'kind': 'transcript'})}\n\n"
            async for chunk in generate_streaming_response(message, session_id, llm, memory):
                yield chunk

        return StreamingResponse(
            stream_wrapper(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_non_stream(chat_message: ChatMessage):
    try:
        session_id = chat_message.session_id or str(uuid.uuid4())
        message = chat_message.message or ""
        if chat_message.voice_data:
            voice_text = process_voice_input(chat_message.voice_data)
            if voice_text and voice_text != "Could not process voice input":
                message = voice_text.strip()

        needs_search = should_use_web_search(message)
        logger.info(f"Message: {message[:80]} | Needs search: {needs_search}")

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=OPENAI_API_KEY,
            timeout=10
        )

        memory = await get_session_memory(session_id)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_prompt = prompt_template.partial(datetime=current_time)

        if not needs_search:
            formatted_prompt = simple_prompt_template.partial(datetime=current_time)
            prompt_value = formatted_prompt.format_prompt(
                chat_history=memory.chat_memory.messages,
                input=message
            )
            resp = await llm.agenerate([prompt_value.to_messages()])
            response_text = normalize_markdown(resp.generations[0][0].text)
            await save_conversation(session_id, message, response_text)
            return SessionResponse(session_id=session_id, message=response_text)

        news_search_tool = next((tool for tool in tools if tool.name == "news_search"), None)
        image_search_tool = next((tool for tool in tools if tool.name == "image_search"), None)
        image_query = is_image_query(message)
        financial_query = re.search(r"\b(stock|price|sensex|nifty|market)\b", message.lower())

        active_tools = tools
        if image_query and image_search_tool:
            active_tools = [image_search_tool]
        elif "news" in message.lower() and not financial_query and news_search_tool:
            active_tools = [news_search_tool]
        elif _mentions_person(message) and not image_query:
            active_tools = [tool for tool in tools if tool.name != "image_search"]

        agent = create_openai_functions_agent(llm, active_tools, formatted_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=active_tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_execution_time=20,
            max_iterations=5
        )

        try:
            result = await agent_executor.ainvoke({"input": message})
            response_text = result.get("output", "")

            # Handle image search results
            if image_query and response_text:
                try:
                    images = json.loads(response_text)
                    if isinstance(images, dict) and "error" in images:
                        logger.error(f"Image search tool returned error: {images['error']}")
                        response_text = f"Failed to fetch images for '{message}': {images['error']}. Please try again later or check your API configuration."
                    elif isinstance(images, list):
                        cards = []
                        for img in images:
                            title = img.get("title", "Untitled Image")
                            thumbnail = img.get("thumbnail", img.get("url", ""))
                            url = img.get("url", "")
                            if url:
                                card = f"""
**{title}**

![Image]({thumbnail})

Source: [View Original]({url})
"""
                                cards.append(card.strip())
                        response_text = "\n\n---\n\n".join(cards) if cards else f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
                    else:
                        logger.warning(f"Unexpected image search result format: {response_text}")
                        response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse image_search JSON: {response_text}, error: {str(e)}")
                    response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
            response_text = normalize_markdown(response_text)
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            if image_query:
                response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
            elif "news" in message.lower():
                response_text = """
**Recent News Fallback:**

- **Korea News**: Unable to fetch live news due to technical issues. Try specifying a topic like 'Korea politics news' or check sources like The Korea Herald (https://www.koreaherald.com) for updates.
"""
            else:
                response_text = f"Error fetching results: {str(e)}. Please try again or rephrase your query."
            response_text = normalize_markdown(response_text)

        await save_conversation(session_id, message, response_text)
        return SessionResponse(session_id=session_id, message=response_text)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.on_event("startup")
async def startup_event():
    try:
        await sessions_collection.create_index("session_id", unique=True)
        await conversations_collection.create_index("session_id")
        await conversations_collection.create_index("timestamp")
        logger.info("Database indexes created")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info", access_log=True)






# import os
# import asyncio
# import json
# import uuid
# from datetime import datetime
# from typing import Optional, AsyncGenerator
# import logging
# from dotenv import load_dotenv
# import re
# import aiohttp
# from functools import lru_cache, wraps
# import time

# # Load environment variables
# load_dotenv()

# # Core imports
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import StreamingResponse, HTMLResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import uvicorn

# # Database
# from motor.motor_asyncio import AsyncIOMotorClient

# # AI and Tools
# from langchain.agents import create_openai_functions_agent, AgentExecutor
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.schema import HumanMessage, AIMessage
# from langchain_openai import ChatOpenAI
# from langchain.tools import BaseTool

# # External APIs
# import speech_recognition as sr
# from pydub import AudioSegment
# import io
# import base64

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key")
# SERPAPI_KEY = os.getenv("SERP_API_KEY", "your-serpapi-key")
# GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "your-gnews-key")
# MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
# PORT = int(os.getenv("PORT", 8000))

# # Database client
# client = AsyncIOMotorClient(MONGODB_URL)
# db = client.chatbot_db
# sessions_collection = db.sessions
# conversations_collection = db.conversations

# # Pydantic models
# class ChatMessage(BaseModel):
#     message: str
#     session_id: Optional[str] = None
#     voice_data: Optional[str] = None

# class SessionResponse(BaseModel):
#     session_id: str
#     message: str

# # Search gating
# RECENCY_TERMS = {
#     "today", "this week", "this month", "latest", "current", "recent",
#     "breaking", "news", "update", "updated", "now", "live", "real-time",
#     "2024", "2025"
# }
# PERSON_TRIGGERS = [
#     r"\bwho is\b", r"\bbiography\b", r"\bage\b", r"\bborn\b", r"\bdied\b",
#     r"\bceo\b", r"\bprime minister\b", r"\bpresident\b"
# ]

# def _is_current_query(t: str) -> bool:
#     tl = t.lower()
#     return any(term in tl for term in RECENCY_TERMS)

# def _mentions_person(t: str) -> bool:
#     return any(re.search(p, t, re.IGNORECASE) for p in PERSON_TRIGGERS)

# IMAGE_TRIGGERS = {
#     "image", "images", "picture", "pictures", "photo", "photos", "pic", "pics", "img"
# }

# def is_image_query(t: str) -> bool:
#     tl = t.lower()
#     return any(term in tl for term in IMAGE_TRIGGERS)

# def should_use_web_search(message: str) -> bool:
#      t = (message or "").strip().lower()
#      return True

# # Custom cache for time-sensitive data
# def timed_lru_cache(seconds: int):
#     def decorator(func):
#         @lru_cache(maxsize=100)
#         def cached_func(*args, **kwargs):
#             return func(*args, **kwargs)
#         @wraps(func)
#         async def wrapper(*args, **kwargs):
#             cache_key = (args, tuple(sorted(kwargs.items())))
#             if cache_key in wrapper.cache:
#                 result, timestamp = wrapper.cache[cache_key]
#                 if time.time() - timestamp < seconds:
#                     return result
#             result = await func(*args, **kwargs)
#             wrapper.cache[cache_key] = (result, time.time())
#             return result
#         wrapper.cache = {}
#         return wrapper
#     return decorator

# # Tools
# class WebSearchTool(BaseTool):
#     name: str = "web_search"
#     description: str = "Search the web for current information using SerpAPI. Keep outputs concise."
#     def __hash__(self):
#         return hash(self.name)

#     def __eq__(self, other):
#         return isinstance(other, WebSearchTool) and self.name == other.name

#     @timed_lru_cache(seconds=3600)
#     async def _arun(self, query: str) -> str:
#         try:
#             if not SERPAPI_KEY or SERPAPI_KEY == "your-serpapi-key":
#                 logger.error("Invalid or missing SERPAPI_KEY")
#                 return json.dumps([])

#             async with aiohttp.ClientSession() as session:
#                 url = "https://serpapi.com/search"
#                 params = {
#                     "engine": "google",
#                     "q": query,
#                     "tbm": "isch",
#                     "api_key": SERPAPI_KEY,
#                     "num": 6,
#                     "ijn": 0  # Ensure first page of results
#                 }
#                 logger.info(f"Sending image search request for query: {query}")
#                 async with session.get(url, params=params, timeout=10) as response:
#                     if response.status != 200:
#                         logger.error(f"Image search API returned status: {response.status} - {await response.text()}")
#                         return json.dumps([])
#                     data = await response.json()
#                     logger.debug(f"SerpAPI response: {json.dumps(data)[:500]}...")
#                     images = []
#                     for img in (data.get("images_results") or [])[:3]:
#                         title = img.get("title", "Untitled Image")
#                         original = img.get("original", "")
#                         thumbnail = img.get("thumbnail", original)  # Fallback to original if no thumbnail
#                         if original:
#                             images.append({"title": title, "url": original, "thumbnail": thumbnail})
#                     if not images:
#                         logger.warning(f"No valid images found for query: {query}")
#                     return json.dumps(images)
#         except Exception as e:
#             logger.error(f"Image search error for query '{query}': {str(e)}")
#             return json.dumps([])

#     def _run(self, query: str) -> str:
#         return asyncio.run(self._arun(query))

# class ImageSearchTool(BaseTool):
#     name: str = "image_search"
#     description: str = "Search for images using SerpAPI. Returns a JSON array of image results with title, url, and thumbnail, or an error object."
#     def __hash__(self):
#         return hash(self.name)

#     def __eq__(self, other):
#         return isinstance(other, ImageSearchTool) and self.name == other.name

#     @timed_lru_cache(seconds=3600)
#     async def _arun(self, query: str) -> str:
#         try:
#             if not SERPAPI_KEY or SERPAPI_KEY == "your-serpapi-key":
#                 logger.error("Invalid or missing SERPAPI_KEY")
#                 return json.dumps({"error": "Invalid or missing SerpAPI key. Please configure a valid key."})

#             async with aiohttp.ClientSession() as session:
#                 url = "https://serpapi.com/search"
#                 params = {
#                     "engine": "google",
#                     "q": query,
#                     "tbm": "isch",
#                     "api_key": SERPAPI_KEY,
#                     "num": 6,
#                     "ijn": 0
#                 }
#                 logger.info(f"Sending image search request for query: {query}")
#                 async with session.get(url, params=params, timeout=10) as response:
#                     if response.status != 200:
#                         error_text = await response.text()
#                         logger.error(f"Image search API returned status: {response.status} - {error_text}")
#                         if response.status == 401:
#                             return json.dumps({"error": "Invalid SerpAPI key. Please check your API key."})
#                         elif response.status == 429:
#                             return json.dumps({"error": "SerpAPI rate limit exceeded. Please try again later."})
#                         return json.dumps({"error": f"Image search API error: HTTP {response.status}"})
#                     data = await response.json()
#                     logger.debug(f"SerpAPI response: {json.dumps(data)[:500]}...")
#                     if "error" in data:
#                         logger.error(f"SerpAPI returned error: {data['error']}")
#                         return json.dumps({"error": f"SerpAPI error: {data['error']}"})
#                     images = []
#                     for img in (data.get("images_results") or [])[:3]:
#                         title = img.get("title", "Untitled Image")
#                         original = img.get("original", "")
#                         thumbnail = img.get("thumbnail", original)
#                         if original:
#                             images.append({"title": title, "url": original, "thumbnail": thumbnail})
#                     return json.dumps(images)
#         except Exception as e:
#             logger.error(f"Image search error for query '{query}': {str(e)}")
#             return json.dumps({"error": f"Image search failed: {str(e)}"})

#     def _run(self, query: str) -> str:
#         return asyncio.run(self._arun(query))

# class NewsSearchTool(BaseTool):
#     name: str = "news_search"
#     description: str = "Search for latest news using GNews API. Returns formatted news cards with images and headlines."

#     def __hash__(self):
#         return hash(self.name)

#     def __eq__(self, other):
#         if not isinstance(other, NewsSearchTool):
#             return False
#         return self.name == other.name

#     async def _arun(self, query: str) -> str:
#         logger.info(f"Running NewsSearchTool for query: {query}")
#         try:
#             async with aiohttp.ClientSession() as session:
#                 url = "https://gnews.io/api/v4/search"
#                 params = {"q": query, "token": GNEWS_API_KEY, "max": 3, "lang": "en"}
#                 async with session.get(url, params=params, timeout=10) as response:
#                     data = await response.json()
#                     logger.info(f"GNews API response: {json.dumps(data)[:100]}...")
#                     articles = data.get("articles") or []
#                     if not articles:
#                         return "No news found"

#                     results = []
#                     for article in articles[:3]:
#                         title = article.get("title", "No title")
#                         url = article.get("url", "")
#                         image_url = article.get("image", "https://example.com/placeholder.jpg")
#                         published = article.get("publishedAt", "Unknown date")

#                         # Format as news card
#                         card = f"""
# **{title}**

# ![News Image]({image_url})

# Published: {published}
# Source: [{url}]({url})
# """
#                         results.append(card.strip())
#                     response_text = "\n\n---\n\n".join(results) if results else "No news found"
#                     logger.info(f"Generated news response: {response_text[:100]}...")
#                     return response_text
#         except Exception as e:
#             logger.error(f"News search async error: {e}")
#             return f"News search error: {str(e)}"

#     def _run(self, query: str) -> str:
#         return asyncio.run(self._arun(query))

# class DateTimeTool(BaseTool):
#     name: str = "datetime"
#     description: str = "Get current date and time."

#     def _run(self, query: str = "") -> str:
#         now = datetime.now()
#         return f"**Current date and time:** {now.strftime('%Y-%m-%d %H:%M:%S')}"

# # Voice processing
# def process_voice_input(voice_data: str) -> str:
#     try:
#         audio_bytes = base64.b64decode(voice_data)
#         audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
#         wav_buffer = io.BytesIO()
#         audio_segment.export(wav_buffer, format="wav")
#         wav_buffer.seek(0)
#         recognizer = sr.Recognizer()
#         with sr.AudioFile(wav_buffer) as source:
#             audio = recognizer.record(source)
#             text = recognizer.recognize_google(audio)
#             return text
#     except Exception as e:
#         logger.error(f"Voice processing error: {e}")
#         return "Could not process voice input"

# # FastAPI app
# app = FastAPI(title="cassy", description="Conversational AI assistant with voice and smart search")

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Tools list
# tools = [WebSearchTool(), ImageSearchTool(), NewsSearchTool(), DateTimeTool()]

# # Prompt template
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", """You are cassy, a concise, friendly, expert AI assistant.

# Search policy:
# - Only use external tools for current events, breaking news, live updates, or clearly time-sensitive queries.
# - For queries like "who is [person]", use the web_search tool to gather biographical details and structure them in a Markdown card with fields: **Name**, **Occupation**, **Born**, **Nationality**. Then, use the image_search tool to fetch exactly two images of the person, returning the raw JSON array of image results (with title, url, thumbnail) without modification.
# - For queries requesting images, pictures, or photos, use the image_search tool. **YOU MUST RETURN THE RAW JSON OUTPUT FROM image_search (array or error object) EXACTLY AS RECEIVED, WITHOUT ANY MODIFICATION, REFORMATTING, OR CONVERSION TO MARKDOWN.** Do not add text like "Here are some images" or format results as a list.
# - Do not use external tools for general explanations, definitions, or evergreen topics.

# MANDATORY FORMATTING - YOU MUST FOLLOW THESE RULES:
# 1. ALWAYS add blank lines before and after tables.
# 2. ALWAYS add blank lines before and after lists.
# 3. Use **bold** for important terms.
# 4. Use proper Markdown syntax for tables, lists, and images.
# 5. For tables, ensure headers are separated by a row of `|---|` and columns are aligned.
# 6. For lists, use `-` for each item, one per line, with no inline lists.
# 7. For news, format as cards with image, headline, publication date, and source link.
# 8. For image_search, **return only the raw JSON array or error object as-is, with no additional text or formatting.**

# Current datetime: 2025-09-09 12:56:00 IST"""),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])

# simple_prompt_template = ChatPromptTemplate.from_messages([
#     ("system", """You are cassy, a concise, friendly, expert AI assistant.

# Search policy:
# - Only use external tools for current events, breaking news, live updates, or clearly time-sensitive queries.
# - Do not use external tools for general explanations, definitions, or evergreen topics.

# MANDATORY FORMATTING - YOU MUST FOLLOW THESE RULES:
# 1. ALWAYS add blank lines before and after tables.
# 2. ALWAYS add blank lines before and after lists.
# 3. Use **bold** for important terms.
# 4. Use proper Markdown syntax for tables and lists.
# 5. For tables, ensure headers are separated by a row of `|---|` and columns are aligned.
# 6. For lists, use `-` for each item, one per line, with no inline lists.
# 7. For image_search, **do not format the output; return the raw JSON array or error object as-is.**

# Example of CORRECT formatting:

# Here is some text.

# **Fruit List:**

# - Apple
# - Banana
# - Orange

# **Comparison Table:**

# | Fruit  | Color  | Taste       |
# |--------|--------|-------------|
# | Apple  | Red    | Sweet/Tart  |
# | Orange | Orange | Citrusy     |

# Current datetime: 2025-09-09 12:56:00 IST"""),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}")
# ])

# # Session memory
# async def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
#     try:
#         session_doc = await sessions_collection.find_one({"session_id": session_id})
#         memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)
#         if session_doc and "messages" in session_doc:
#             for msg in session_doc["messages"]:
#                 if msg["type"] == "human":
#                     memory.chat_memory.add_message(HumanMessage(content=msg["content"]))
#                 else:
#                     memory.chat_memory.add_message(AIMessage(content=msg["content"]))
#         return memory
#     except Exception as e:
#         logger.error(f"Error retrieving session memory: {e}")
#         return ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)

# async def save_conversation(session_id: str, human_message: str, ai_message: str):
#     try:
#         await sessions_collection.update_one(
#             {"session_id": session_id},
#             {
#                 "$push": {
#                     "messages": {
#                         "$each": [
#                             {"type": "human", "content": human_message, "timestamp": datetime.now()},
#                             {"type": "ai", "content": ai_message, "timestamp": datetime.now()}
#                         ]
#                     }
#                 },
#                 "$set": {"updated_at": datetime.now()}
#             },
#             upsert=True
#         )
#         await conversations_collection.insert_one({
#             "session_id": session_id,
#             "human_message": human_message,
#             "ai_message": ai_message,
#             "timestamp": datetime.now(),
#             "used_web_search": should_use_web_search(human_message)
#         })
#     except Exception as e:
#         logger.error(f"Error saving conversation: {e}")

# # Markdown normalization
# def normalize_markdown(s: str) -> str:
#     if not s:
#         return ""
#     s = s.replace("\r\n", "\n")
#     fence = "`" * 3
#     if s.count(fence) % 2 == 1:
#         s += "\n" + fence
#     return s

# def fix_markdown_line(line: str, prev_line: str, next_line: str = '') -> str:
#     stripped = line.strip()
#     result = line
#     if stripped.startswith('|') or stripped.startswith('- '):
#         if prev_line.strip() and not prev_line.strip().startswith(('- ', '|')):
#             result = '\n' + line
#     elif prev_line.strip().startswith(('- ', '|')) and not stripped.startswith(('- ', '|')):
#         result = '\n' + line
#     if stripped.startswith('* '):
#         result = result.replace('* ', '- ')
#     return result

# # Streaming generator
# async def generate_streaming_response(
#     message: str,
#     session_id: str,
#     llm: ChatOpenAI,
#     memory: ConversationBufferWindowMemory
# ) -> AsyncGenerator[str, None]:
#     try:
#         needs_search = should_use_web_search(message)
#         logger.info(f"Message: {message[:80]} | Needs search: {needs_search}")

#         if not needs_search:
#             current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             formatted_prompt = simple_prompt_template.partial(datetime=current_time)
#             prompt_value = formatted_prompt.format_prompt(
#                 chat_history=memory.chat_memory.messages,
#                 input=message
#             )
#             resp = await llm.agenerate([prompt_value.to_messages()])
#             response_text = normalize_markdown(resp.generations[0][0].text)
#             memory.chat_memory.add_message(HumanMessage(content=message))
#             memory.chat_memory.add_message(AIMessage(content=response_text))
#             await save_conversation(session_id, message, response_text)

#             lines = response_text.split('\n')
#             for i, line in enumerate(lines):
#                 prev_line = lines[i-1] if i > 0 else ''
#                 next_line = lines[i+1] if i < len(lines)-1 else ''
#                 fixed_line = fix_markdown_line(line, prev_line, next_line)
#                 if fixed_line.strip():
#                     yield f"data: {json.dumps({'token': fixed_line + '\n'})}\n\n"
#             yield f"data: {json.dumps({'done': True})}\n\n"
#             return

#         if message.lower().strip() in ["give some news", "news", "tell news"]:
#             response_text = "Could you specify a topic or region for the news (e.g., 'news on Korea politics' or 'latest sports news')?"
#             await save_conversation(session_id, message, response_text)
#             yield f"data: {json.dumps({'token': response_text + '\n'})}\n\n"
#             yield f"data: {json.dumps({'done': True})}\n\n"
#             return

#         if message.lower().strip() in ["images", "pictures", "photos", "give some images", "show images", "tell images"]:
#             response_text = "Could you specify what images you want (e.g., 'images of the Eiffel Tower' or 'pictures of cats')?"
#             await save_conversation(session_id, message, response_text)
#             yield f"data: {json.dumps({'token': response_text + '\n'})}\n\n"
#             yield f"data: {json.dumps({'done': True})}\n\n"
#             return

#         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         formatted_prompt = prompt_template.partial(datetime=current_time)
#         financial_query = re.search(r"\b(stock|price|sensex|nifty|market)\b", message.lower())
#         news_search_tool = next((tool for tool in tools if tool.name == "news_search"), None)
#         image_search_tool = next((tool for tool in tools if tool.name == "image_search"), None)
#         image_query = is_image_query(message)

#         active_tools = tools
#         if image_query and image_search_tool:
#             active_tools = [image_search_tool]
#         elif "news" in message.lower() and not financial_query and news_search_tool:
#             active_tools = [news_search_tool]

#         agent = create_openai_functions_agent(llm, active_tools, formatted_prompt)
#         agent_executor = AgentExecutor(
#             agent=agent,
#             tools=active_tools,
#             memory=memory,
#             verbose=True,
#             handle_parsing_errors=True,
#             max_execution_time=20,
#             max_iterations=5
#         )

#         yield f"data: {json.dumps({'token': '🔎 Searching...\n\n'})}\n\n"

#         try:
#             logger.info(f"Invoking agent with input: {message}, active tools: {[tool.name for tool in active_tools]}")
#             result = await agent_executor.ainvoke({"input": message})
#             response_text = result.get("output", "")

#             # Handle image search results
#             if image_query and response_text:
#                 try:
#                     images = json.loads(response_text)
#                     if isinstance(images, dict) and "error" in images:
#                         logger.error(f"Image search tool returned error: {images['error']}")
#                         response_text = f"Failed to fetch images for '{message}': {images['error']}. Please try again later or check your API configuration."
#                     elif isinstance(images, list):
#                         cards = []
#                         for img in images:
#                             title = img.get("title", "Untitled Image")
#                             thumbnail = img.get("thumbnail", img.get("url", ""))
#                             url = img.get("url", "")
#                             if url:
#                                 card = f"""
# **{title}**

# ![Image]({thumbnail})

# Source: [View Original]({url})
# """
#                                 cards.append(card.strip())
#                         response_text = "\n\n---\n\n".join(cards) if cards else f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
#                     else:
#                         logger.warning(f"Unexpected image search result format: {response_text}")
#                         response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
#                 except json.JSONDecodeError as e:
#                     logger.error(f"Failed to parse image_search JSON: {response_text}, error: {str(e)}")
#                     response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
#             response_text = normalize_markdown(response_text)
#         except Exception as e:
#             logger.error(f"Agent execution error: {e}")
#             if image_query:
#                 response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
#             elif "news" in message.lower():
#                 response_text = """
# **Recent News Fallback:**

# - **Korea News**: Unable to fetch live news due to technical issues. Try specifying a topic like 'Korea politics news' or check sources like The Korea Herald (https://www.koreaherald.com) for updates.
# """
#             else:
#                 response_text = f"Error fetching results: {str(e)}. Please try again or rephrase your query."
#             response_text = normalize_markdown(response_text)
#             yield f"data: {json.dumps({'token': response_text + '\n'})}\n\n"
#             yield f"data: {json.dumps({'done': True})}\n\n"
#             await save_conversation(session_id, message, response_text)
#             return

#         lines = response_text.split('\n')
#         for i, line in enumerate(lines):
#             prev_line = lines[i-1] if i > 0 else ''
#             next_line = lines[i+1] if i < len(lines)-1 else ''
#             fixed_line = fix_markdown_line(line, prev_line, next_line)
#             if fixed_line.strip():
#                 yield f"data: {json.dumps({'token': fixed_line + '\n'})}\n\n"
#                 await asyncio.sleep(0.005)
#         yield f"data: {json.dumps({'done': True})}\n\n"

#         await save_conversation(session_id, message, response_text)

#     except Exception as e:
#         logger.error(f"Streaming error: {e}")
#         error_message = f"❌ Error: {str(e)}"
#         yield f"data: {json.dumps({'token': error_message})}\n\n"
#         yield f"data: {json.dumps({'done': True})}\n\n"

# # UI
# @app.get("/", response_class=HTMLResponse)
# async def root():
#     html_content = r"""
# <!DOCTYPE html>
# <html lang="en">
# <head>
#   <meta charset="utf-8" />
#   <meta name="viewport" content="width=device-width, initial-scale=1" />
#   <title>Cassy – Chat UI</title>

#   <!-- Libraries -->
#   <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
#   <script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.5/dist/purify.min.js"></script>
#   <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css" />
#   <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js"></script>

#   <style>
#     :root {
#       --bg: #0b0f0d;
#       --text: #e6f2eb;
#       --muted: #98a39b;
#       --emerald: #10b981;
#       --border: #1f2a22;
#     }
#     * { box-sizing: border-box; }
#     html, body { 
#       height: 100%; 
#       margin: 0;
#       background: var(--bg);
#       color: var(--text);
#       font-family: system-ui, -apple-system, sans-serif;
#     }
#     .app {
#       height: 100vh;
#       display: grid;
#       grid-template-columns: 320px 1fr;
#       gap: 0;
#     }
#     .sidebar {
#       background: #111516;
#       padding: 16px;
#       border-right: 1px solid var(--border);
#     }
#     .sb-title {
#       display: flex; 
#       align-items: center; 
#       gap: 8px; 
#       font-weight: 700;
#       margin-bottom: 12px;
#     }
#     .chat {
#       display: flex; 
#       flex-direction: column; 
#       height: 100vh;
#     }
#     .header {
#       display: flex; 
#       align-items: center; 
#       justify-content: space-between;
#       padding: 14px 18px; 
#       border-bottom: 1px solid var(--border); 
#       background: #0e1411;
#     }
#     .brand { 
#       display: flex; 
#       align-items: center; 
#       gap: 10px; 
#       font-weight: 800; 
#     }
#     .brand .dot { 
#       width: 10px; 
#       height: 10px; 
#       border-radius: 50%; 
#       background: var(--emerald); 
#       box-shadow: 0 0 18px var(--emerald); 
#     }
#     .messages { 
#       flex: 1; 
#       overflow: auto; 
#       padding: 24px; 
#     }
#     .row { 
#       display: flex; 
#       gap: 10px; 
#       margin: 10px 0; 
#     }
#     .row.user { 
#       justify-content: flex-end; 
#     }
#     .bubble {
#       max-width: 70%; 
#       padding: 12px 14px; 
#       border-radius: 14px; 
#       border: 1px solid var(--border); 
#       background: #0e1411;
#       word-wrap: break-word;
#       overflow-wrap: break-word;
#       white-space: normal;
#       line-height: 1.5;
#     }
#     .row.user .bubble { 
#       background: #1a2a1e; 
#       border-color: #1c3b28; 
#     }
#     .avatar { 
#       width: 36px; 
#       height: 36px; 
#       border-radius: 50%; 
#       display: flex;
#       align-items: center;
#       justify-content: center;
#       background: #0f1812; 
#       border: 1px solid var(--border); 
#       flex-shrink: 0;
#     }
#     .composer {
#       display: flex; 
#       gap: 8px; 
#       align-items: center; 
#       background: #0f1812; 
#       border: 1px solid var(--border);
#       border-radius: 12px; 
#       padding: 8px;
#       margin: 12px;
#     }
#     .composer input {
#       flex: 1; 
#       background: transparent; 
#       border: 0; 
#       color: var(--text); 
#       font-size: 15px; 
#       outline: none;
#       padding: 8px;
#     }
#     .btn, .mic {
#       width: 40px; 
#       height: 40px; 
#       display: flex;
#       align-items: center;
#       justify-content: center;
#       border-radius: 10px; 
#       border: 1px solid var(--border);
#       background: #0e1411; 
#       color: var(--text); 
#       cursor: pointer;
#       font-size: 16px;
#       flex-shrink: 0;
#     }
#     .btn.primary { 
#       background: linear-gradient(180deg, #16a34a, #10b981); 
#       border: none; 
#     }
#     .btn:disabled { 
#       opacity: 0.6; 
#       cursor: not-allowed; 
#     }
#     .mic.record { 
#       background: linear-gradient(180deg, #dc2626, #ef4444); 
#       border: none; 
#     }
#     .welcome {
#       display: flex; 
#       flex-direction: column; 
#       align-items: center; 
#       justify-content: center;
#       height: 100%; 
#       text-align: center; 
#       padding: 40px;
#     }
#     .welcome-title { 
#       font-size: 28px; 
#       font-weight: 700; 
#       margin-bottom: 12px; 
#     }
#     .welcome-subtitle { 
#       color: var(--muted); 
#       font-size: 16px; 
#       margin-bottom: 24px; 
#     }
#     .typing { 
#       display: flex; 
#       gap: 6px; 
#       align-items: center; 
#     }
#     .dot { 
#       width: 8px; 
#       height: 8px; 
#       border-radius: 50%; 
#       background: #2a3b31; 
#       animation: blink 1.2s infinite ease-in-out; 
#     }
#     .dot:nth-child(2) { animation-delay: 0.2s; }
#     .dot:nth-child(3) { animation-delay: 0.4s; }
#     @keyframes blink { 
#       0%, 80%, 100% { opacity: 0.2; } 
#       40% { opacity: 1; } 
#     }
#     .bubble.markdown {
#       white-space: normal;
#     }
#     .bubble h1, .bubble h2, .bubble h3, .bubble h4, .bubble h5, .bubble h6 { 
#       margin: 16px 0 8px 0; 
#       color: var(--emerald); 
#       font-weight: 600;
#       line-height: 1.3;
#     }
#     .bubble h1 { font-size: 1.5em; }
#     .bubble h2 { font-size: 1.3em; }
#     .bubble h3 { font-size: 1.1em; }
#     .bubble p { 
#       margin: 8px 0; 
#       line-height: 1.5;
#     }
#     .bubble p:first-child { margin-top: 0; }
#     .bubble p:last-child { margin-bottom: 0; }
#     .bubble code { 
#       background: #0b1510; 
#       padding: 3px 6px; 
#       border-radius: 4px; 
#       color: #86efac; 
#       font-family: 'Courier New', monospace;
#       font-size: 0.9em;
#     }
#     .bubble pre { 
#       background: #0b1510; 
#       padding: 12px; 
#       border-radius: 8px; 
#       margin: 12px 0; 
#       overflow-x: auto;
#       border: 1px solid var(--border);
#     }
#     .bubble pre code {
#       background: transparent;
#       padding: 0;
#       color: #86efac;
#     }
#     .bubble ul, .bubble ol { 
#       margin: 12px 0; 
#       padding-left: 24px; 
#       color: var(--text);
#       width: 100%;
#       max-width: 100%;
#     }
#     .bubble ul {
#       list-style-type: disc;
#     }
#     .bubble ol {
#       list-style-type: decimal;
#     }
#     .bubble li {
#       margin: 6px 0;
#       line-height: 1.5;
#       color: var(--text);
#     }
#     .bubble li::marker {
#       color: var(--emerald);
#     }
#     .bubble ul ul, .bubble ol ol, .bubble ul ol, .bubble ol ul {
#       margin: 4px 0;
#     }
#     .bubble strong { 
#       color: #86efac; 
#       font-weight: 600;
#     }
#     .bubble em {
#       color: #a7f3d0;
#       font-style: italic;
#     }
#     .bubble blockquote {
#       border-left: 3px solid var(--emerald);
#       margin: 12px 0;
#       padding-left: 12px;
#       color: var(--muted);
#       font-style: italic;
#     }
#     .bubble a {
#       color: var(--emerald);
#       text-decoration: underline;
#     }
#     .bubble a:hover {
#       color: #86efac;
#     }
#     .bubble table {
#       border-collapse: collapse;
#       width: 100%;
#       max-width: 100%;
#       margin: 12px 0;
#       background: #0b1510;
#       border-radius: 6px;
#       overflow: hidden;
#       table-layout: auto;
#     }
#     .bubble th, .bubble td {
#       border: 1px solid var(--border);
#       padding: 10px 12px;
#       text-align: left;
#       vertical-align: top;
#     }
#     .bubble th {
#       background: #1a2a1e;
#       color: var(--emerald);
#       font-weight: 600;
#     }
#     .bubble td {
#       color: var(--text);
#     }
#     .bubble tr:nth-child(even) td {
#       background: rgba(16, 185, 129, 0.05);
#     }
#     .bubble img.md-img {
#       max-width: 300px;
#       height: 200px;
#       object-fit: cover;
#       border-radius: 8px;
#       margin: 5px 0;
#       display: block;
#     }
#     .bubble hr {
#       border: none;
#       height: 1px;
#       background: var(--border);
#       margin: 16px 0;
#     }
#   </style>
# </head>
# <body>
# <div class="app">
#   <aside class="sidebar">
#     <div class="sb-title">🟢 <span>cassy</span></div>
#     <div style="background: #0e1411; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin-top: 10px; color: var(--muted);">Welcome to cassy</div>
#     <div style="background: #0e1411; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin-top: 10px; color: var(--muted);">Try asking anything!</div>
#   </aside>
#   <main class="chat">
#     <div class="header">
#       <div class="brand">
#         <div class="dot"></div> 
#         <div>cassy</div>
#       </div>
#       <div style="color:#86efac">online</div>
#     </div>
#     <div id="messages" class="messages">
#       <div class="welcome" id="welcome">
#         <div class="welcome-title">👋 Hey there!</div>
#         <div class="welcome-subtitle">I'm cassy, your AI assistant. Ask me anything!</div>
#       </div>
#     </div>
#     <div class="composer">
#       <input type="text" id="messageInput" placeholder="Ask cassy anything..." />
#       <button type="button" id="micButton" class="mic">🎤</button>
#       <button type="button" id="sendButton" class="btn primary">➤</button>
#     </div>
#   </main>
# </div>

# <script>
# function debounce(func, wait) {
#   let timeout;
#   return function (...args) {
#     clearTimeout(timeout);
#     timeout = setTimeout(() => func.apply(this, args), wait);
#   };
# }

# let sessionId = null;
# let isRecording = false;
# let mediaRecorder = null;
# let audioChunks = [];
# const BASE_URL = window.location.origin;

# const messagesDiv = document.getElementById('messages');
# const messageInput = document.getElementById('messageInput');
# const sendButton = document.getElementById('sendButton');
# const micButton = document.getElementById('micButton');
# const welcomeDiv = document.getElementById('welcome');

# console.log('Page loaded, elements found:', {
#   messagesDiv: !!messagesDiv,
#   messageInput: !!messageInput,
#   sendButton: !!sendButton,
#   micButton: !!micButton
# });

# const debouncedRenderMarkdown = debounce((bubble, text) => {
#   if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
#     console.warn('marked or DOMPurify not available');
#     bubble.textContent = text;
#     return;
#   }

#   try {
#     bubble.classList.add('markdown');
#     let processed = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function(_, alt, src) {
#       return `<img class="md-img" src="${src}" alt="${alt}" />`;
#     });

#     marked.setOptions({ 
#       gfm: true, 
#       breaks: true, 
#       tables: true,
#       headerIds: false,
#       mangle: false
#     });

#     const rawHtml = marked.parse(processed);
#     const safeHtml = DOMPurify.sanitize(rawHtml, {
#       USE_PROFILES: { html: true },
#       ALLOWED_TAGS: [
#         'p', 'br', 'strong', 'b', 'em', 'i', 'code', 'pre',
#         'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
#         'ul', 'ol', 'li', 'dl', 'dt', 'dd',
#         'blockquote', 'a', 'hr',
#         'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
#         'colgroup', 'col', 'img', 'span', 'div'
#       ],
#       ALLOWED_ATTR: [
#         'href', 'target', 'rel', 'colspan', 'rowspan', 'align', 
#         'src', 'alt', 'title', 'class', 'id', 'style'
#       ],
#       ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|cid|xmpp|data):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i
#     });

#     bubble.innerHTML = safeHtml;

#     if (typeof Prism !== 'undefined') {
#       try {
#         Prism.highlightAllUnder(bubble);
#       } catch (e) {
#         console.warn('Prism highlight failed:', e);
#       }
#     }

#   } catch (e) {
#     console.warn('Markdown rendering failed, falling back to plain text:', e);
#     bubble.classList.remove('markdown');
#     bubble.textContent = text;
#   }
# }, 50);

# function hideWelcome() {
#   if (welcomeDiv) welcomeDiv.style.display = 'none';
# }

# function addMessage(text, isUser = false) {
#   hideWelcome();
#   const row = document.createElement('div');
#   row.className = 'row' + (isUser ? ' user' : '');
#   const bubble = document.createElement('div');
#   bubble.className = 'bubble';
#   if (isUser) {
#     bubble.textContent = text;
#   } else {
#     debouncedRenderMarkdown(bubble, text);
#   }
#   const avatar = document.createElement('div');
#   avatar.className = 'avatar';
#   avatar.innerHTML = isUser ? '👤' : '🤖';
#   if (isUser) {
#     row.appendChild(bubble);
#     row.appendChild(avatar);
#   } else {
#     row.appendChild(avatar);
#     row.appendChild(bubble);
#   }
#   messagesDiv.appendChild(row);
#   messagesDiv.scrollTop = messagesDiv.scrollHeight;
#   return bubble;
# }

# function showTyping() {
#   const row = document.createElement('div');
#   row.className = 'row';
#   row.id = 'typing-indicator';
#   const avatar = document.createElement('div');
#   avatar.className = 'avatar';
#   avatar.innerHTML = '🤖';
#   const bubble = document.createElement('div');
#   bubble.className = 'bubble typing';
#   bubble.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
#   row.appendChild(avatar);
#   row.appendChild(bubble);
#   messagesDiv.appendChild(row);
#   messagesDiv.scrollTop = messagesDiv.scrollHeight;
# }

# function hideTyping() {
#   const typing = document.getElementById('typing-indicator');
#   if (typing) typing.remove();
# }

# async function sendMessage() {
#   const message = messageInput.value.trim();
#   if (!message) return;
#   console.log('Sending message:', message, 'to:', `${BASE_URL}/chat/stream`);
#   addMessage(message, true);
#   messageInput.value = '';
#   sendButton.disabled = true;
#   showTyping();
#   try {
#     const response = await fetch(`${BASE_URL}/chat/stream`, {
#       method: 'POST',
#       headers: { 'Content-Type': 'application/json' },
#       body: JSON.stringify({ message, session_id: sessionId })
#     });
#     if (!response.ok) {
#       console.error('Fetch error:', response.status, response.statusText);
#       throw new Error(`HTTP ${response.status}: ${response.statusText}`);
#     }
#     hideTyping();
#     let botBubble = null;
#     let fullResponse = '';
#     const reader = response.body.getReader();
#     const decoder = new TextDecoder();
#     while (true) {
#       const { value, done } = await reader.read();
#       if (done) {
#         if (botBubble && fullResponse.trim()) debouncedRenderMarkdown(botBubble, fullResponse);
#         break;
#       }
#       const chunk = decoder.decode(value, { stream: true });
#       const lines = chunk.split('\n');
#       for (const line of lines) {
#         if (line.startsWith('data: ')) {
#           try {
#             const data = JSON.parse(line.slice(6));
#             if (data.session_id && !sessionId) sessionId = data.session_id;
#             if (data.token) {
#               if (!botBubble) {
#                 botBubble = addMessage('', false);
#               }
#               fullResponse += data.token;
#               debouncedRenderMarkdown(botBubble, fullResponse);
#               messagesDiv.scrollTop = messagesDiv.scrollHeight;
#             }
#           } catch (e) {
#             console.warn('Parse error:', e);
#           }
#         }
#       }
#     }
#   } catch (error) {
#     console.error('Error:', error);
#     hideTyping();
#     addMessage(`Sorry, there was an error: ${error.message}. Please check if the server is running at ${BASE_URL}.`, false);
#   } finally {
#     sendButton.disabled = false;
#   }
# }

# async function startRecording() {
#   if (!navigator.mediaDevices) { alert('Microphone not available'); return; }
#   try {
#     const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
#     mediaRecorder = new MediaRecorder(stream);
#     audioChunks = [];
#     mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
#     mediaRecorder.onstop = async () => {
#       const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
#       const arrayBuffer = await audioBlob.arrayBuffer();
#       const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
#       console.log('Sending voice data to:', `${BASE_URL}/chat/stream`);
#       showTyping();
#       try {
#         const response = await fetch(`${BASE_URL}/chat/stream`, {
#           method: 'POST',
#           headers: { 'Content-Type': 'application/json' },
#           body: JSON.stringify({ message: '', voice_data: base64, session_id: sessionId })
#         });
#         if (!response.ok) {
#           console.error('Fetch error:', response.status, response.statusText);
#           throw new Error(`HTTP ${response.status}: ${response.statusText}`);
#         }
#         let userMessageAdded = false;
#         let botBubble = null;
#         let fullResponse = '';
#         const reader = response.body.getReader();
#         const decoder = new TextDecoder();
#         while (true) {
#           const { value, done } = await reader.read();
#           if (done) {
#             if (botBubble && fullResponse.trim()) debouncedRenderMarkdown(botBubble, fullResponse);
#             break;
#           }
#           const chunk = decoder.decode(value, { stream: true });
#           const lines = chunk.split('\n');
#           for (const line of lines) {
#             if (line.startsWith('data: ')) {
#               try {
#                 const data = JSON.parse(line.slice(6));
#                 if (data.session_id && !sessionId) sessionId = data.session_id;
#                 if (data.kind === 'transcript' && data.token && !userMessageAdded) {
#                   hideTyping();
#                   addMessage(data.token, true);
#                   userMessageAdded = true;
#                   showTyping();
#                 } else if (data.token && !data.kind) {
#                   if (!botBubble) {
#                     hideTyping();
#                     botBubble = addMessage('', false);
#                   }
#                   fullResponse += data.token;
#                   debouncedRenderMarkdown(botBubble, fullResponse);
#                   messagesDiv.scrollTop = messagesDiv.scrollHeight;
#                 }
#               } catch (e) {
#                 console.warn('Parse error:', e);
#               }
#             }
#           }
#         }
#       } catch (error) {
#         console.error('Voice error:', error);
#         hideTyping();
#         addMessage(`Sorry, there was an error processing your voice: ${error.message}. Please check if the server is running at ${BASE_URL}.`, false);
#       }
#       stream.getTracks().forEach(track => track.stop());
#     };
#     mediaRecorder.start();
#     isRecording = true;
#     micButton.classList.add('record');
#     micButton.innerHTML = '⏹';
#     setTimeout(stopRecording, 10000);
#   } catch (error) {
#     console.error('Microphone error:', error);
#     alert('Could not access microphone: ' + error.message);
#   }
# }

# function stopRecording() {
#   if (isRecording && mediaRecorder) {
#     mediaRecorder.stop();
#     isRecording = false;
#     micButton.classList.remove('record');
#     micButton.innerHTML = '🎤';
#   }
# }

# function attachEventListeners() {
#   if (sendButton) sendButton.addEventListener('click', function(e) { e.preventDefault(); sendMessage(); });
#   if (messageInput) messageInput.addEventListener('keypress', function(e) {
#     if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
#   });
#   if (micButton) micButton.addEventListener('click', function(e) {
#     e.preventDefault(); if (isRecording) stopRecording(); else startRecording();
#   });
# }

# document.addEventListener('DOMContentLoaded', function() {
#   attachEventListeners();
#   if (messageInput) messageInput.focus();
# });

# if (document.readyState !== 'loading') {
#   attachEventListeners();
#   if (messageInput) messageInput.focus();
# }
# </script>
# </body>
# </html>
#     """
#     return HTMLResponse(content=html_content)

# @app.post("/chat/stream")
# async def chat_stream(chat_message: ChatMessage):
#     try:
#         session_id = chat_message.session_id or str(uuid.uuid4())
#         message = chat_message.message or ""
#         transcript = None
#         if chat_message.voice_data:
#             voice_text = process_voice_input(chat_message.voice_data)
#             if voice_text and voice_text != "Could not process voice input":
#                 transcript = voice_text.strip()
#                 message = transcript

#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0.2,
#             streaming=True,
#             openai_api_key=OPENAI_API_KEY,
#             timeout=10
#         )

#         memory = await get_session_memory(session_id)

#         async def stream_wrapper():
#             yield f"data: {json.dumps({'session_id': session_id})}\n\n"
#             if transcript:
#                 yield f"data: {json.dumps({'token': transcript, 'kind': 'transcript'})}\n\n"
#             async for chunk in generate_streaming_response(message, session_id, llm, memory):
#                 yield chunk

#         return StreamingResponse(
#             stream_wrapper(),
#             media_type="text/event-stream",
#             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
#         )

#     except Exception as e:
#         logger.error(f"Chat stream error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat")
# async def chat_non_stream(chat_message: ChatMessage):
#     try:
#         session_id = chat_message.session_id or str(uuid.uuid4())
#         message = chat_message.message or ""
#         if chat_message.voice_data:
#             voice_text = process_voice_input(chat_message.voice_data)
#             if voice_text and voice_text != "Could not process voice input":
#                 message = voice_text.strip()

#         needs_search = should_use_web_search(message)
#         logger.info(f"Message: {message[:80]} | Needs search: {needs_search}")

#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0.2,
#             openai_api_key=OPENAI_API_KEY,
#             timeout=10
#         )

#         memory = await get_session_memory(session_id)
#         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         formatted_prompt = prompt_template.partial(datetime=current_time)

#         if not needs_search:
#             formatted_prompt = simple_prompt_template.partial(datetime=current_time)
#             prompt_value = formatted_prompt.format_prompt(
#                 chat_history=memory.chat_memory.messages,
#                 input=message
#             )
#             resp = await llm.agenerate([prompt_value.to_messages()])
#             response_text = normalize_markdown(resp.generations[0][0].text)
#             await save_conversation(session_id, message, response_text)
#             return SessionResponse(session_id=session_id, message=response_text)

#         news_search_tool = next((tool for tool in tools if tool.name == "news_search"), None)
#         image_search_tool = next((tool for tool in tools if tool.name == "image_search"), None)
#         image_query = is_image_query(message)
#         financial_query = re.search(r"\b(stock|price|sensex|nifty|market)\b", message.lower())

#         active_tools = tools
#         if image_query and image_search_tool:
#             active_tools = [image_search_tool]
#         elif "news" in message.lower() and not financial_query and news_search_tool:
#             active_tools = [news_search_tool]

#         agent = create_openai_functions_agent(llm, active_tools, formatted_prompt)
#         agent_executor = AgentExecutor(
#             agent=agent,
#             tools=active_tools,
#             memory=memory,
#             verbose=True,
#             handle_parsing_errors=True,
#             max_execution_time=20,
#             max_iterations=5
#         )

#         try:
#             result = await agent_executor.ainvoke({"input": message})
#             response_text = result.get("output", "")

#             # Handle image search results
#             if image_query and response_text:
#                 try:
#                     images = json.loads(response_text)
#                     if isinstance(images, dict) and "error" in images:
#                         logger.error(f"Image search tool returned error: {images['error']}")
#                         response_text = f"Failed to fetch images for '{message}': {images['error']}. Please try again later or check your API configuration."
#                     elif isinstance(images, list):
#                         cards = []
#                         for img in images:
#                             title = img.get("title", "Untitled Image")
#                             thumbnail = img.get("thumbnail", img.get("url", ""))
#                             url = img.get("url", "")
#                             if url:
#                                 card = f"""
# **{title}**

# ![Image]({thumbnail})

# Source: [View Original]({url})
# """
#                                 cards.append(card.strip())
#                         response_text = "\n\n---\n\n".join(cards) if cards else f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
#                     else:
#                         logger.warning(f"Unexpected image search result format: {response_text}")
#                         response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
#                 except json.JSONDecodeError as e:
#                     logger.error(f"Failed to parse image_search JSON: {response_text}, error: {str(e)}")
#                     response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
#             response_text = normalize_markdown(response_text)
#         except Exception as e:
#             logger.error(f"Agent execution error: {e}")
#             if image_query:
#                 response_text = f"No images found for '{message}'. Try a more specific query (e.g., 'images of {message.replace('images of ', '').replace('photos of ', '')} at an event') or check sites like Getty Images."
#             elif "news" in message.lower():
#                 response_text = """
# **Recent News Fallback:**

# - **Korea News**: Unable to fetch live news due to technical issues. Try specifying a topic like 'Korea politics news' or check sources like The Korea Herald (https://www.koreaherald.com) for updates.
# """
#             else:
#                 response_text = f"Error fetching results: {str(e)}. Please try again or rephrase your query."
#             response_text = normalize_markdown(response_text)

#         await save_conversation(session_id, message, response_text)
#         return SessionResponse(session_id=session_id, message=response_text)

#     except Exception as e:
#         logger.error(f"Chat error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "timestamp": datetime.now()}

# @app.on_event("startup")
# async def startup_event():
#     try:
#         await sessions_collection.create_index("session_id", unique=True)
#         await conversations_collection.create_index("session_id")
#         await conversations_collection.create_index("timestamp")
#         logger.info("Database indexes created")
#     except Exception as e:
#         logger.error(f"Error creating indexes: {e}")

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info", access_log=True)
