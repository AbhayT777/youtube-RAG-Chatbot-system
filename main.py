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
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
WEBSHARE_PROXY_URL = os.getenv("WEBSHARE_PROXY_URL")
GROQ_MODEL     = "llama-3.1-8b-instant"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
TOP_K_RESULTS  = 6

# In-memory store: video_id → retriever
video_store: dict = {}

# Shared embedding function (lazy-loaded on first /load call)
_embedding_fn = None

def get_embedding():
    """Lazy-load embedding on first use to keep startup RAM low."""
    global _embedding_fn
    if _embedding_fn is None:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set. Add it to your .env file.")
        _embedding_fn = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=hf_token,
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


def build_retriever(video_id: str):
    """Fetch transcript (English → Hindi → any available), chunk, embed, return FAISS retriever."""
    try:
        # Use proxy if available (required on cloud servers like Render)
        if WEBSHARE_PROXY_URL:
            from youtube_transcript_api.proxies import GenericProxyConfig
            proxy_config = GenericProxyConfig(
                http_url=WEBSHARE_PROXY_URL,
                https_url=WEBSHARE_PROXY_URL,
            )
            api = YouTubeTranscriptApi(proxy_config=proxy_config)
        else:
            api = YouTubeTranscriptApi()

        # List all available transcripts for this video
        transcript_list = api.list(video_id)
        available = {t.language_code: t for t in transcript_list}

        # Priority: English first, then Hindi, then first available language
        if "en" in available:
            fetched = available["en"].fetch()
        elif "hi" in available:
            fetched = available["hi"].fetch()
        else:
            first = next(iter(available.values()))
            fetched = first.fetch()

        transcript = " ".join(chunk.text for chunk in fetched)

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

class LoadRequest(BaseModel):
    url: str

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