# Agentic RAG Chatbot

An intelligent **Agentic Retrieval-Augmented Generation (RAG)** chatbot built using **LangGraph** and **Streamlit**, powered by **OpenAI GPT-OSS 120B (via Groq)**. The chatbot autonomously selects the appropriate tool to answer user queries, including document retrieval, web search, Wikipedia search, mathematical calculations, and live stock price retrieval.

The document retrieval pipeline uses **Hybrid Search (ChromaDB + BM25)** to combine semantic similarity and keyword-based retrieval, improving retrieval quality over similarity search alone.

---

## Highlights

- Hybrid Retrieval using **ChromaDB + BM25 (EnsembleRetriever)**
- OpenAI **GPT-OSS 120B** served through **Groq**
- LangGraph ReAct Agent with autonomous tool selection
- Per-thread persistent PDF knowledge base
- SQLite-based persistent conversational memory
- Streaming responses in Streamlit
- Evaluated using **RAGAS**

---

# Features

- **Hybrid RAG** — Upload a PDF and chat with it using a hybrid retrieval pipeline combining semantic search (ChromaDB) and keyword search (BM25).
- **Autonomous Tool Selection** — The agent automatically decides which tool to invoke based on the user's query.
- **Web Search** — Real-time web search powered by DuckDuckGo.
- **Wikipedia Search** — Retrieve factual information from Wikipedia.
- **Calculator Tool** — Perform arithmetic operations.
- **Live Stock Price Tool** — Retrieve real-time stock prices using Alpha Vantage.
- **Persistent Chat Memory** — SQLite checkpointing preserves conversation history across application restarts.
- **Per-thread PDF Isolation** — Each conversation maintains an independent document index and memory.
- **Resume Previous Conversations** — Continue any previous chat from the sidebar.
- **Streaming Responses** — Token-by-token response generation for improved user experience.

---

# Architecture

```
User Input
      │
      ▼
Streamlit Frontend
      │
      ▼
LangGraph ReAct Agent
      │
      ▼
LLM decides which tool to invoke
      │
 ┌────┼──────────┬──────────┬──────────┬────────────┐
 │    │          │          │          │            │
 ▼    ▼          ▼          ▼          ▼
DuckDuckGo   Calculator   Stocks   Wikipedia   Hybrid RAG
                                              │
                             ┌────────────────┴──────────────┐
                             │                               │
                             ▼                               ▼
                     ChromaDB Retriever             BM25 Retriever
                             │                               │
                             └──────── EnsembleRetriever ────┘
                                              │
                                              ▼
                                    Retrieved Context
                                              │
                                              ▼
                                  GPT-OSS 120B (Groq)
                                              │
                                              ▼
                                   Streaming Response
```

---

# Tech Stack

| Component | Technology |
|------------|------------|
| LLM | OpenAI GPT-OSS 120B (via Groq) |
| Agent Framework | LangGraph |
| Frontend | Streamlit |
| Embedding Model | HuggingFace BAAI/bge-large-en-v1.5 |
| Vector Database | ChromaDB |
| Keyword Retriever | BM25 |
| Hybrid Retriever | LangChain EnsembleRetriever |
| Persistent Memory | SQLite Checkpointer |
| Web Search | DuckDuckGo |
| Knowledge Search | Wikipedia |
| Stock API | Alpha Vantage |
| Evaluation | RAGAS |

---

# Project Structure

```
agentic_rag_chatbot/
├── langgraph_rag_backend.py
├── streamlit_rag_frontend.py
├── evaluate_rag.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Setup

### Clone Repository

```bash
git clone https://github.com/jinendrajain765/Agentic_rag_chatbot.git
cd Agentic_rag_chatbot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file and add:

```
GROQ_API_KEY=your_api_key
```

### Run

```bash
streamlit run streamlit_rag_frontend.py
```

---

# How It Works

The chatbot follows the **ReAct (Reason + Act)** workflow.

1. The user submits a query.
2. GPT-OSS 120B determines whether a tool is required.
3. If necessary, LangGraph invokes the selected tool.
4. Tool outputs are returned to the model.
5. The model synthesizes the final response and streams it back to the user.

This allows the chatbot to dynamically decide which tool to use without hardcoded routing logic.

---

# Key Design Decisions

### Hybrid Retrieval

The document retrieval pipeline combines **semantic search (ChromaDB)** with **keyword search (BM25)** using LangChain's `EnsembleRetriever`. This approach improves retrieval robustness by leveraging both semantic similarity and exact keyword matching.

### Per-thread Document Isolation

Each conversation maintains an independent ChromaDB vector store (`./chroma_db/{thread_id}`), ensuring uploaded documents remain isolated between conversations.

### Persistent Memory

SQLite checkpointing preserves chat history across application restarts, enabling long-running conversations.

### High-quality Embeddings

The chatbot uses **BAAI/bge-large-en-v1.5** embeddings for semantic document retrieval.

### Streaming Responses

Responses are streamed token-by-token to provide a more responsive user experience.

---

# RAG Evaluation

The retrieval pipeline was evaluated using **RAGAS** on a benchmark set of document question-answer pairs.

## Evaluation Results

| Metric | Score |
|---------|------:|
| Faithfulness | **1.0000** |
| Answer Relevancy | **0.7617** |
| Context Precision | **0.6250** |
| Context Recall | **1.0000** |

## Improvement After Hybrid Retrieval

| Metric | Similarity Search | Hybrid Retrieval |
|---------|------------------:|-----------------:|
| Faithfulness | **1.0000** | **1.0000** |
| Answer Relevancy | **0.7000** | **0.7617** |
| Context Precision | **0.6100** | **0.6250** |
| Context Recall | N/A | **1.0000** |

The hybrid retrieval pipeline improved answer relevancy and retrieval quality while maintaining perfect faithfulness.

---

# Example Queries

| Query | Tool |
|--------|------|
| Who prepared these digital notes? | Hybrid RAG |
| What is the stock price of AAPL? | Stock Tool |
| What is 25 × 48? | Calculator |
| Latest AI news | DuckDuckGo |
| Who is Lionel Messi? | Wikipedia |

---

# Future Improvements

- FastAPI backend for production deployment
- Multi-PDF retrieval per conversation
- Authentication and multi-user support
- Docker deployment
- Cloud deployment

---
