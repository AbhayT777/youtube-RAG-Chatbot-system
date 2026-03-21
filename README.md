# 🎬 ChaTube — YouTube RAG Q&A System

> Ask anything about any YouTube video using AI.

Built by **Abhay Kumar Tiwari**

---

## 📌 What is ChaTube?

ChaTube is a full-stack AI-powered web application that lets users ask natural language questions about any YouTube video. It uses **Retrieval-Augmented Generation (RAG)** — it fetches the video transcript, splits it into chunks, embeds them into a vector database, and retrieves the most relevant chunks to answer each question using a large language model.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | HTML / CSS / JS | Single-page UI hosted on Netlify |
| Backend | FastAPI (Python) | REST API hosted on Render |
| LLM | Groq (llama-3.1-8b-instant) | Free cloud inference, no GPU needed |
| Embeddings | HuggingFace MiniLM | Local sentence embeddings |
| Vector Store | FAISS | In-memory similarity search |
| Transcripts | youtube-transcript-api | Fetches YouTube captions |
| Orchestration | LangChain | RAG pipeline & chain management |

---

## 📁 Project Structure

```
youtube-RAG-Chatbot-system/
├── main.py              ← FastAPI backend (RAG pipeline)
├── index.html           ← Frontend (single HTML file)
├── requirements.txt     ← Python dependencies for Render
├── render.yaml          ← Render deployment configuration
├── .env                 ← Secret keys (never commit this)
├── .env.example         ← Safe template to commit
└── .gitignore
```

---

## ⚙️ How It Works (RAG Pipeline)

1. User pastes a YouTube URL and clicks **Load**
2. Frontend sends `POST /load` to the FastAPI backend
3. Backend extracts the video ID from the URL
4. `youtube-transcript-api` fetches the full transcript
5. Transcript is split into 1000-character chunks (200-char overlap)
6. HuggingFace MiniLM embeds all chunks into vectors
7. FAISS stores the vectors in memory, indexed by video ID
8. User types a question and clicks **Ask**
9. Frontend sends `POST /ask` with `video_id` and `question`
10. FAISS retrieves the top 6 most semantically similar chunks
11. A prompt is built with the retrieved context + question
12. Groq LLM generates the answer
13. Answer is returned to the frontend and displayed in the chat

---

## 🚀 Local Development Setup

### Prerequisites
- Python 3.10+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Step 1 — Install dependencies

```bash
pip install fastapi uvicorn[standard] langchain langchain-core langchain-community
pip install langchain-huggingface langchain-groq faiss-cpu youtube-transcript-api
pip install sentence-transformers python-dotenv requests
```

### Step 2 — Set up your `.env` file

```
GROQ_API_KEY=gsk_your_key_here
```

### Step 3 — Start the backend

```bash
uvicorn main:app --reload
```

Backend runs at: `http://localhost:8000`

### Step 4 — Open the frontend

Double-click `index.html` in your file explorer. Make sure `API_BASE` in `index.html` is set to `http://localhost:8000`.

---

## 📡 API Reference

### `GET /health`
Health check endpoint.
```json
{ "status": "ok" }
```

---

### `POST /load`
Fetch transcript and build FAISS index for a video.

**Request:**
```json
{ "url": "https://www.youtube.com/watch?v=VIDEO_ID" }
```

**Response:**
```json
{ "video_id": "VIDEO_ID", "message": "Transcript loaded and indexed successfully." }
```

**Errors:**
- `400` — Could not extract video ID
- `422` — Captions disabled or transcript not found
- `500` — Server error

---

### `POST /ask`
Answer a question using the loaded video index.

**Request:**
```json
{ "video_id": "VIDEO_ID", "question": "What is this video about?" }
```

**Response:**
```json
{ "answer": "This video is about..." }
```

**Errors:**
- `400` — Empty question
- `404` — Video not loaded (call `/load` first)
- `500` — LLM error

---

## ☁️ Deployment

### Backend — Render (Free)

1. Push your project to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repository
4. Use these settings:

| Field | Value |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | Free |

5. Add environment variable: `GROQ_API_KEY` = your key
6. Deploy — you'll get a URL like `https://chatube-api.onrender.com`

---

### Frontend — Netlify (Free)

1. In `index.html`, update:
```js
const API_BASE = "https://chatube-api.onrender.com";
```
2. Push to GitHub
3. Go to [netlify.com](https://netlify.com) → **Add new site → Import from GitHub**
4. Set publish directory to `/` (root) and deploy
5. You'll get a free URL like `https://chatube.netlify.app`

---

## 🔧 Configuration Reference

All settings are at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model for answers |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `1000` | Characters per transcript chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `6` | Chunks retrieved per question |

---

## ⚠️ Known Limitations

- **In-memory store** — video indexes are lost when Render restarts (free tier spins down after inactivity). User just needs to click Load again.
- **No captions = no answer** — videos with disabled or missing transcripts cannot be processed.
- **Render cold start** — first request after inactivity may take 30–60 seconds to wake up.
- **No authentication** — API is open. Add API key auth before any serious public deployment.

---

## 🔮 Future Improvements

- Persistent vector store (Redis / Pinecone) so indexes survive restarts
- Streaming responses — answer appears word by word
- Multilingual support via multilingual embedding models
- Video metadata display (title, thumbnail, channel name)
- Chat history persistence
- Rate limiting and authentication

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Made with ❤ by Abhay Kumar Tiwari*