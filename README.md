## Study Assistant (LangChain + Groq + Hugging Face)

This is a terminal study assistant chatbot built with LangChain. It uses:

- Groq LLM for fast, high-quality generation
- Hugging Face sentence-transformer embeddings for semantic memory
- Chroma as a local vector store to retain notes across sessions
- Conversational message history and a prompt template
- Structured output parsed into a Pydantic model

### 1) Prerequisites
- Python 3.10+
- A Groq API key
- Optional: a Hugging Face token (only needed for some gated models)

### 2) Setup

1. Create a `.env` file (copy from `.env.example`) and set your key(s):
```
GROQ_API_KEY=your_groq_key_here
# Optional if needed
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

2. Install dependencies in the existing virtual environment:
```
./venv/Scripts/python -m pip install -U pip
./venv/Scripts/pip install -r requirements.txt
```

### 3) Run

Interactive chat:
```
./venv/Scripts/python main.py
```

Single-turn prompt:
```
./venv/Scripts/python main.py --once "Explain the Doppler effect like I'm a beginner"
```

Advanced options:
```
./venv/Scripts/python main.py \
  --session-id my_session \
  --model llama3-70b-8192 \
  --embed-model sentence-transformers/all-MiniLM-L6-v2
```

### Notes
- The app persists semantic memory in `./storage/memory_db` using Chroma. You can delete that folder to reset long-term notes.
- The assistant produces structured responses with fields like `answer`, `key_points`, `suggested_questions`, and `references`.
- If you see model download warnings, the first run may take longer as the embedding model is fetched.

### 4) API Server (FastAPI)

Run the HTTP API:
```
./venv/Scripts/python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

- Endpoint: POST /chat
- Body: { "message": string, "session_id": string }
- Response: { answer, key_points, suggested_questions, references }

### 5) Web Frontend (React + Vite + Tailwind)

Install and run:
```
cd "study assistant/web"
npm install
npm run dev
```

Configure API URL in a `.env` file inside `web/` (optional):
```
VITE_API_URL=http://localhost:8000/chat
```

Open the app at http://localhost:5173
