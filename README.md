# Agentic RAG Chatbot

An agentic Retrieval-Augmented Generation chatbot that autonomously routes between document retrieval, web search, Wikipedia, live stock prices, and calculation tools — built on **LangGraph**, served through **Groq (GPT-OSS 120B)**, and rigorously evaluated with **RAGAS**.

Adding a cross-encoder reranker on top of hybrid retrieval took **Context Precision from 0.625 to 1.0** and **Context Recall to a perfect 1.0**, with retrieval quality independently verified across two full RAGAS evaluation passes on unseen, non-parametric content the model could not have memorized during training.

---

## Highlights

- Hybrid Retrieval — **ChromaDB (dense) + BM25 (sparse)** via LangChain's `EnsembleRetriever`
- **Cross-encoder reranking** (`BAAI/bge-reranker-base`) on top of hybrid retrieval to filter retrieval noise
- LangGraph ReAct agent with autonomous, prompt-governed tool selection
- Per-thread persistent PDF knowledge base with isolated vector stores
- SQLite-based persistent conversational memory
- Streaming responses in Streamlit
- Evaluated end-to-end using **RAGAS**, with a documented before/after comparison across three retrieval configurations

---

# Features

- **Hybrid + Reranked RAG** — Upload a PDF and chat with it using dense + sparse retrieval, refined by a cross-encoder reranker before context reaches the LLM.
- **Autonomous Tool Selection** — The agent decides which tool to invoke based on the query, governed by an explicit tool-use policy in the system prompt (see *Key Design Decisions* below).
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
                             │                                │
                             ▼                                ▼
                     ChromaDB Retriever              BM25 Retriever
                             │                                │
                             └──────── EnsembleRetriever ─────┘
                                              │
                                              ▼
                                  Cross-Encoder Reranker
                                     (bge-reranker-base)
                                              │
                                              ▼
                                    Top-k Reranked Context
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
| Reranker | HuggingFace BAAI/bge-reranker-base (cross-encoder) |
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
├── ragas_results.csv                   # baseline (pre-reranker) evaluation
├── ragas_results_with_reranker.csv     # post-reranker evaluation
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

### Hybrid Retrieval + Reranking

The retrieval pipeline combines **semantic search (ChromaDB)** with **keyword search (BM25)** via LangChain's `EnsembleRetriever`, then re-scores the merged candidate set with a **cross-encoder reranker** (`bge-reranker-base`) before the top results reach the LLM. Hybrid search alone widened recall but left retrieval noise in the final context; the reranker was added specifically to address that gap — measured directly through RAGAS Context Precision (see *RAG Evaluation* below).

### Tool-Use Policy

The agent's system prompt encodes an explicit tool-use policy: once `rag_tool` returns content for an uploaded document, that result is treated as ground truth and is not re-verified against external tools, even for unfamiliar or fictional-sounding subject matter. This was a deliberate fix after evaluation surfaced cases where the agent second-guessed correct retrieval results by redundantly invoking web search, which cost accuracy and latency for no benefit.

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

The retrieval pipeline was evaluated using **RAGAS** across three configurations: similarity search alone, hybrid retrieval, and hybrid retrieval + cross-encoder reranking. The final evaluation pass used a purpose-built, entirely fictional benchmark document — invented company, product, and figures — specifically so that a correct answer could only come from retrieval, not from the LLM's own training knowledge, isolating retrieval quality from parametric recall.

## Full Comparison: Similarity Search → Hybrid Retrieval → Hybrid + Reranking

| Metric | Similarity Search | Hybrid Retrieval | Hybrid + Reranking |
|---------|------------------:|------------------:|--------------------:|
| Faithfulness | 1.0000 | 1.0000 | 0.7917 |
| Answer Relevancy | 0.7000 | 0.7617 | 0.7961 |
| Context Precision | 0.6100 | 0.6250 | **1.0000** |
| Context Recall | N/A | 1.0000 | **1.0000** |

**Context Precision improved from 0.625 to a perfect 1.0** after adding the reranker, confirming it successfully filters retrieval noise that hybrid search alone left behind. Context Recall held at a perfect 1.0, meaning no relevant information was lost in the process. Answer Relevancy also improved.

Faithfulness dropped slightly in the reranked run (0.79 vs. 1.00) on a small subset of questions where the agent occasionally declined to answer or introduced an unsupported detail despite the correct context being retrieved — a generation-layer issue, not a retrieval failure, since Context Precision and Recall were both perfect on those same rows. This was traced to agent tool-routing behavior and addressed via the tool-use policy described above.

Full per-question results are available in [`ragas_results.csv`](./ragas_results.csv) (baseline) and [`ragas_results_with_reranker.csv`](./ragas_results_with_reranker.csv) (post-reranker).

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
