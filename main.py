"""
YouTube RAG Q&A — FastAPI Backend
----------------------------------
Endpoints:
  POST /index  — accept raw transcript text from browser, build FAISS index
  POST /ask    — answer a question using the loaded index
  GET  /health — health check for Render

NOTE: Transcript is fetched by the browser (user's IP), not the server.
      This avoids YouTube IP bans on cloud providers like Render.
"""

import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import yt_dlp
import re as _re
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
HF_TOKEN      = os.getenv("HF_TOKEN")
GROQ_MODEL    = "llama-3.1-8b-instant"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 6

# In-memory store: video_id → retriever
video_store: dict = {}

# Lazy-loaded embedding model
_embedding_fn = None

def get_embedding():
    """Lazy-load embedding on first use to keep startup RAM low."""
    global _embedding_fn
    if _embedding_fn is None:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set on server.")
        _embedding_fn = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=HF_TOKEN,
        )
    return _embedding_fn


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(title="ChaTube API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from a YouTube URL or return raw ID as-is."""
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id.strip()):
        return url_or_id.strip()
    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def build_retriever_from_text(transcript: str):
    """Chunk transcript text, embed it, return FAISS retriever."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, get_embedding())
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )


def build_chain(retriever):
    """Build the RAG chain with Groq LLM."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set on server.")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.3,
    )

    prompt = PromptTemplate(
        template="""You are a helpful assistant answering questions about a YouTube video using its transcript.

Use the transcript context below to answer the question as fully and helpfully as possible.
- If the answer is clearly stated, give it directly.
- If the answer is spread across the transcript, synthesize and summarize it.
- Only say "I don't know" if the topic is genuinely not mentioned at all.
- Do NOT start your answer with phrases like "Based on the transcript," or "According to the transcript," — just answer directly.

Transcript context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel({
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class IndexRequest(BaseModel):
    url: str          # YouTube URL

class AskRequest(BaseModel):
    video_id: str
    question: str

class IndexResponse(BaseModel):
    video_id: str
    message: str

class AskResponse(BaseModel):
    answer: str


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


def fetch_transcript_ytdlp(video_id: str) -> str:
    """Fetch transcript using yt-dlp — better YouTube bypass than youtube-transcript-api."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitleslangs": ["en", "hi"],
        "subtitlesformat": "json3",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Try manual subtitles first, then auto-generated
            subs = info.get("subtitles", {})
            auto = info.get("automatic_captions", {})

            # Priority: en manual → hi manual → en auto → hi auto → first available
            track_data = None
            for lang in ["en", "hi"]:
                if lang in subs:
                    track_data = subs[lang]
                    break
            if not track_data:
                for lang in ["en", "hi"]:
                    if lang in auto:
                        track_data = auto[lang]
                        break
            if not track_data:
                all_tracks = list(subs.values()) or list(auto.values())
                if all_tracks:
                    track_data = all_tracks[0]

            if not track_data:
                raise HTTPException(status_code=422, detail="No transcript available for this video.")

            # Find json3 format or fallback to first available
            json3_entry = next((t for t in track_data if t.get("ext") == "json3"), track_data[0])
            
            import urllib.request
            with urllib.request.urlopen(json3_entry["url"]) as resp:
                import json
                data = json.loads(resp.read())

            # Extract text from json3 format
            texts = []
            for event in data.get("events", []):
                for seg in event.get("segs", []):
                    txt = seg.get("utf8", "").strip()
                    if txt and txt != "\n":
                        texts.append(txt)

            transcript = " ".join(texts)
            if not transcript.strip():
                raise HTTPException(status_code=422, detail="Transcript is empty.")
            return transcript

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch transcript: {str(e)}")


@app.post("/index", response_model=IndexResponse)
def index_transcript(req: IndexRequest):
    """
    Fetches transcript via yt-dlp and builds FAISS index.
    """
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        transcript = fetch_transcript_ytdlp(video_id)
        retriever = build_retriever_from_text(transcript)
        video_store[video_id] = retriever
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index transcript: {str(e)}")

    return IndexResponse(
        video_id=video_id,
        message="Transcript indexed successfully."
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    Accepts a video_id and question, runs the RAG chain, returns the answer.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retriever = video_store.get(req.video_id)
    if not retriever:
        raise HTTPException(
            status_code=404,
            detail="Video not indexed. Load the video first."
        )

    chain = build_chain(retriever)
    try:
        answer = chain.invoke(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return AskResponse(answer=answer)