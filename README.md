# Agentic RAG Chatbot

An intelligent multi-tool agentic chatbot built with **LangGraph** and **Streamlit**, powered by **Groq LLaMA 3.3 70B**. The chatbot autonomously selects tools to answer questions — from searching the web, performing calculations, fetching live stock prices, to answering questions from uploaded PDF documents using RAG.

---

## Features

- **RAG (Retrieval Augmented Generation)** — Upload any PDF and chat with it. Each conversation thread has its own isolated PDF context using per-thread ChromaDB vector stores.
- **Web Search** — Real-time web search powered by DuckDuckGo
- **Wikipedia Search** — Factual queries answered using Wikipedia
- **Calculator** — Performs basic arithmetic operations
- **Stock Price** — Fetches live stock prices via Alpha Vantage API
- **Persistent Memory** — Full chat history stored in SQLite, survives server restarts
- **Multi-thread Conversations** — Each chat session is an independent thread with its own history
- **Resume Chat** — Click any past conversation from the sidebar to resume it
- **Streaming Responses** — Token-by-token streaming for real-time output
- **Autonomous Tool Selection** — LLM decides which tool to call based on the user query

---


## Architecture

```
User Input
    |
Streamlit Frontend (streamlit_rag_frontend.py)
    |
LangGraph ReAct Agent (langgraph_rag_backend.py)
    |
LLM decides which tool to use
    |
    |---------|---------|---------|---------|
    |         |         |         |         |
DuckDuckGo  Calculator  Stock  Wikipedia  RAG Tool
                        API                  |
                                        ChromaDB
                                     (per thread PDF)
    |
SQLite Checkpointer (persistent memory)
    |
Streaming Response to Streamlit UI
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Agent Framework | LangGraph 0.2.70 |
| Frontend | Streamlit |
| Embeddings | HuggingFace — BAAI/bge-large-en-v1.5 |
| Vector Store | ChromaDB (per thread, persistent on disk) |
| Memory | SQLite (persistent checkpointing) |
| Web Search | DuckDuckGo |
| Wiki Search | Wikipedia |
| Stock Data | Alpha Vantage API |

---



## Project Structure

```
agentic_rag_chatbot/
├── langgraph_rag_backend.py    # LangGraph agent, tools, RAG setup
├── streamlit_rag_frontend.py   # Streamlit UI, thread management, streaming
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```


## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/jinendrajain765/Agentic_rag_chatbot.git
cd Agentic_rag_chatbot
```


### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
# 4
Add your API key to `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the application
```bash
streamlit run streamlit_rag_frontend.py
```

---



## How It Works

The chatbot uses a **ReAct (Reason + Act)** loop:

1. **Think** — LLM reads the user query and reasons about which tool to use
2. **Act** — LLM calls the appropriate tool with the right parameters
3. **Observe** — LLM reads the tool output
4. **Repeat** — If needed, LLM calls another tool
5. **Answer** — LLM synthesizes a clean final response

This makes the agent autonomous — the developer does not hardcode which tool to use, the LLM decides at runtime based on the query.

---

## Key Design Decisions

**Per-thread PDF isolation** — Each conversation thread has its own ChromaDB vector store persisted to disk under `./chroma_db/{thread_id}`. This means different users can upload different PDFs in different threads without any interference, and PDFs persist across server restarts.

**ChromaDB over FAISS** — ChromaDB persists vector data to disk so uploaded PDFs survive server restarts. FAISS stores everything in RAM which means PDFs are lost on restart.

**BAAI/bge-large-en-v1.5 embeddings** — Upgraded from all-MiniLM-L6-v2 to BGE large for significantly better retrieval quality. BGE large has 1024 dimensions vs 384 for MiniLM.

**SQLite over InMemorySaver** — Persistent SQLite checkpointing ensures all chat history survives server restarts. With InMemorySaver, history is lost when the server stops.

**Streaming** — Token-by-token streaming via LangGraph's stream mode improves user experience by displaying responses in real time instead of waiting for the full output.

---
### Example queries

| Query | Tool Used |
|---|---|
| Who is the CEO according to the document? | RAG Tool |
| What is the stock price of AAPL? | Stock Price Tool |
| What is 25 multiplied by 48? | Calculator Tool |
| What are the latest developments in AI? | DuckDuckGo Search |
| Who is Lionel Messi? | Wikipedia |

---


## Future Improvements

- Expose backend via FastAPI for production deployment
- Add support for multiple PDFs per thread
- Deploy on Streamlit Cloud
- Add authentication for multi-user support

---



