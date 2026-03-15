# SmartDoc AI 🗂️

> Upload any document. Ask anything. Get precise answers — instantly.

SmartDoc AI is an AI-powered document question-answering system built from scratch. Instead of manually scrolling through pages trying to find what you need, you simply upload your document and have a conversation with it. It understands what you're asking, finds the most relevant parts of your document, and gives you a direct, accurate answer.

No fluff. No hallucinations. Just answers.

---

## What it does

You upload a PDF, Word doc, CSV, or any text-based file. The system breaks it into chunks, converts those chunks into vector embeddings, and stores them. When you ask a question, it searches for the most semantically similar chunks — not just keyword matches — and feeds them to a large language model to generate a grounded response.

That's Retrieval Augmented Generation (RAG) in plain English.

---

## Built with

| Layer | Technology |
|---|---|
| Backend | Flask (Python) |
| AI Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` |
| LLM | Groq API — LLaMA 3.3 70B |
| Frontend | Vanilla HTML, CSS, JavaScript |

---

## Features

- Upload documents in PDF, DOCX, TXT, CSV, JSON, or Markdown format
- Ask questions in plain natural language
- Semantic search — finds meaning, not just keywords
- MMR retrieval — picks diverse, relevant chunks to avoid repetitive answers
- Delete uploaded documents and clear the vector store
- Export your entire chat session as a text file
- Clean, dark UI with quick query suggestions

---

## Getting started

### Prerequisites

- Python 3.11
- A Groq API key — get one free at [console.groq.com](https://console.groq.com)

### Installation

```bash
# Clone the repo
git clone https://github.com/roshan-kamath/SmartDocAI.git
cd SmartDocAI

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Set your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Run the app

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000` and you're good to go.

---

## How to use it

1. Click **Process Document** and upload any supported file
2. Wait a few seconds while the system chunks and embeds your document
3. Type your question in the chat box and hit Enter
4. Get a direct, accurate answer sourced from your document

That's it. No setup beyond the first run.

---

## Project structure

```
SmartDocAI/
├── app.py              # Flask backend — handles upload, ask, delete routes
├── rag_pipeline.py     # The entire RAG pipeline — loading, chunking, embedding, retrieval
├── requirements.txt    # Python dependencies
├── Procfile            # For deployment on Render
├── .gitignore
└── templates/
    └── index.html      # Frontend — single page chat interface
```

---

## How the RAG pipeline works

```
Document uploaded
      ↓
Text extracted → Split into 1000-token chunks with 200-token overlap
      ↓
Each chunk converted to a 384-dim vector using sentence-transformers
      ↓
Vectors stored in ChromaDB (local persistent store)
      ↓
User asks a question
      ↓
Question embedded using same model
      ↓
MMR search retrieves 6 most relevant chunks
      ↓
Chunks + question sent to LLaMA 3.3 70B via Groq
      ↓
Grounded answer returned to user
```

---

## Deployment

This app is configured for deployment on [Render](https://render.com).

1. Push your code to GitHub
2. Create a new Web Service on Render and connect your repo
3. Add `GROQ_API_KEY` as an environment variable
4. Render will detect the `Procfile` and deploy automatically

The `Procfile` uses `gunicorn` as the production server:
```
web: gunicorn app:app
```

---

## Known limitations

- Very large files (50MB+) may take longer to process
- Scanned PDFs (image-based) are not supported — text must be selectable
- The vector database is stored locally — on Render's free tier, it resets on restart
- Responses are strictly limited to the content of the uploaded document

---

## Team

Built by [Roshan Kamath](https://github.com/roshan-kamath) and [Manvith](https://github.com/manvithh06) as part of a hands-on AI project exploring real-world RAG architecture.

---

## License

MIT — use it, build on it, break it, learn from it.