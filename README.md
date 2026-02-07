# RAG-Based AI Assistant

A Retrieval-Augmented Generation (RAG) AI assistant that answers questions by searching through a document corpus and generating contextual responses using LLMs. Built with LangChain, ChromaDB, and Sentence Transformers.

## Features

- **Document ingestion** — loads `.txt` files from the `data/` directory and chunks them for retrieval
- **Vector search** — embeds documents using Sentence Transformers and stores them in ChromaDB for similarity search
- **Multi-provider LLM support** — works with OpenAI, Groq, or Google Gemini
- **Conversational interface** — interactive CLI with chat history tracking
- **Structured prompting** — chain-of-thought prompt template with role, instructions, and output constraints

## Architecture

```
User Question
    |
    v
Vector DB Search (ChromaDB + Sentence Transformers)
    |
    v
Format Retrieved Context (top 5 chunks with source metadata)
    |
    v
Build Prompt (context + chat history + question)
    |
    v
LLM Response (OpenAI / Groq / Google Gemini)
```

## Project Structure

```
rt-aaidc-project1-template/
├── src/
│   ├── app.py             # Main RAG application and CLI
│   └── vectordb.py        # ChromaDB vector database wrapper
├── data/                  # Document corpus (.txt files)
│   ├── artificial_intelligence.txt
│   ├── biotechnology.txt
│   ├── climate_science.txt
│   ├── quantum_computing.txt
│   ├── space_exploration.txt
│   ├── sustainable_energy.txt
│   └── sample_documents.txt
├── chroma_db/             # Persistent vector database storage
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
├── .env.example           # Environment variable template
└── .gitignore
```

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| Vector Database | ChromaDB (persistent local storage) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` (500 char chunks, 200 char overlap) |
| LLM Providers | OpenAI, Groq, Google Gemini |

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An API key from **one** of these providers:
  - [OpenAI](https://platform.openai.com/api-keys)
  - [Groq](https://console.groq.com/keys) (free tier available)
  - [Google AI](https://aistudio.google.com/app/apikey)

### Installation

1. **Clone the repository:**

   ```bash
   git clone [your-repo-url]
   cd rt-aaidc-project1-template
   ```

2. **Install dependencies:**

   Using uv (recommended):
   ```bash
   uv sync
   ```

   Using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**

   ```bash
   cp .env.example .env    # Linux/Mac
   copy .env.example .env  # Windows
   ```

   Edit `.env` and add your API key for at least one provider:

   ```
   OPENAI_API_KEY=your_key_here
   # OR
   GROQ_API_KEY=your_key_here
   # OR
   GOOGLE_API_KEY=your_key_here
   ```

   See [.env.example](.env.example) for all available configuration options (model selection, embedding model, collection name).

## Usage

Run the assistant:

```bash
uv run python src/app.py
```

Or with pip:
```bash
python src/app.py
```

The assistant will:
1. Load all `.txt` documents from the `data/` directory
2. Chunk and embed them into ChromaDB
3. Start an interactive prompt where you can ask questions

Type `quit` to exit.

### Example

```
$ uv run python src/app.py
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Vector database initialized with collection: rag_documents
Using Groq model: llama-3.1-8b-instant
RAG Assistant initialized successfully

Loading documents...
Successfully loaded: artificial_intelligence.txt
Successfully loaded: quantum_computing.txt
Successfully loaded: climate_science.txt
...
Loaded 7 sample documents

Enter a question or 'quit' to exit: What is quantum computing?

Answer:
 **Reasoning:**
 Based on the retrieved context from quantum_computing.txt, quantum computing
 is described as a field that leverages quantum mechanical phenomena to process
 information in fundamentally different ways than classical computers.

 **Answer:**
 Quantum computing is a type of computation that uses quantum bits (qubits)
 instead of classical bits. Unlike classical bits which can only be 0 or 1,
 qubits can exist in superposition — representing both states simultaneously.
 This allows quantum computers to solve certain problems exponentially faster
 than classical computers, particularly in areas like cryptography,
 optimization, and molecular simulation.
```

## Configuration

All configuration is done through environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `GOOGLE_API_KEY` | — | Google AI API key |
| `GOOGLE_MODEL` | `gemini-2.0-flash` | Google model to use |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |

The assistant auto-detects which provider to use based on which API key is set (checks OpenAI first, then Groq, then Google).

## Adding Your Own Documents

Place `.txt` files in the `data/` directory. The assistant loads all files from this directory on startup. Each file is automatically chunked and embedded into the vector database.
