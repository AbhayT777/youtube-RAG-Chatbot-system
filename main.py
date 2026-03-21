"""
YouTube RAG Q&A — FastAPI Backend
----------------------------------
Endpoints:
  POST /load   — fetch transcript, build FAISS index for a video
  POST /ask    — answer a question using the loaded index
  GET  /health — health check for Render
"""

import os
import re
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = "llama-3.1-8b-instant"          # Free, fast, great quality
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K_RESULTS   = 6

# In-memory store: video_id → retriever
# (For production scale, replace with Redis or a persistent store)
video_store: dict = {}

# Shared embedding model (loaded once at startup)
embedding_model = None


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Embedding model ready.")
    yield
    print("Shutting down.")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(title="YouTube RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Tighten this to your Netlify URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from a YouTube URL or return as-is if already an ID."""
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
    # If no pattern matched, assume it's already a raw ID
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id.strip()):
        return url_or_id.strip()
    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def build_retriever(video_id: str):
    """Fetch transcript, chunk it, embed it, return a FAISS retriever."""
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        raise HTTPException(status_code=422, detail="Captions are disabled for this video.")
    except NoTranscriptFound:
        raise HTTPException(status_code=422, detail="No transcript found for this video.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch transcript: {str(e)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(chunks, embedding_model)
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

class LoadRequest(BaseModel):
    url: str                   # Full YouTube URL or raw video ID

class AskRequest(BaseModel):
    video_id: str
    question: str

class LoadResponse(BaseModel):
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


@app.post("/load", response_model=LoadResponse)
def load_video(req: LoadRequest):
    """
    Accepts a YouTube URL, extracts the video ID, fetches the transcript,
    builds a FAISS index, and stores the retriever in memory.
    """
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if video_id not in video_store:
        retriever = build_retriever(video_id)
        video_store[video_id] = retriever

    return LoadResponse(
        video_id=video_id,
        message="Transcript loaded and indexed successfully."
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    Accepts a video_id and a question, runs the RAG chain, returns the answer.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retriever = video_store.get(req.video_id)
    if not retriever:
        raise HTTPException(
            status_code=404,
            detail="Video not loaded. Call /load first."
        )

    chain = build_chain(retriever)
    try:
        answer = chain.invoke(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return AskResponse(answer=answer)