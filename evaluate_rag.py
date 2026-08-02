import sys
sys.path.append(".")  # so it can import your backend

from langgraph_rag_backend import (
    chatbot1, ingest_pdf, _get_retriever,
    _THREAD_RETRIEVERS, _THREAD_METADATA,
)
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from ragas.embeddings import LangchainEmbeddingsWrapper
from langgraph_rag_backend import embeddings
from groq import APIStatusError, BadRequestError
from langgraph.errors import GraphRecursionError
import time
from ragas.run_config import RunConfig
answer_relevancy.strictness=1



# Ingest test PDF into a base thread
THREAD_ID = "eval_thread_10"
PDF_PATH = "Nimbustack_Technical_Overview.pdf"

test_questions = [
    "Who founded NimbusStack Technologies and in what year?",
    "What are the four core modules of FerroGrid?",
    "What algorithm does the Quillion Scheduler use and what improvement did it provide?",
    "What port does the Glyph Transport Protocol use by default?",
    "How many synchronous and asynchronous replicas does Snowline Replication maintain by default?",
    "What is the maximum storage capacity per cluster supported by the Tundra Storage Layer?",
    "What z-score threshold does SparkWatch use to flag anomalies?",
    "What was codenamed 'Marrow' and what did it introduce?",
    "What is the price and node limit of the Kestrel pricing tier?",
    "How often must clusters rotate their root signing key according to the security team?",
    "What caused the Thornfield Outage and how long did it last?",
    "What is the support response time SLA for Sparrow tier customers?",
]

ground_truths = [
    "NimbusStack Technologies was founded in 2019 by Devraj Malhotra and Priya Ashwin",
    "Quillion Scheduler, Basalt Network Mesh, Tundra Storage Layer, and Ember Observability Suite",
    "It uses Weighted Fracture Balancing (WFB), which reduced node hotspotting by 42 percent compared to the earlier round-robin scheduler",
    "Port 6743",
    "Three synchronous replicas and two asynchronous replicas",
    "18 petabytes per cluster",
    "A rolling z-score threshold of 3.2",
    "FerroGrid version 3.0 was codenamed 'Marrow' and it introduced GTP-7 as a replacement for the Basalt Legacy Protocol",
    "890 fictional credits per month, supporting up to 75 nodes",
    "Every 45 days",
    "It was caused by a misconfigured Snowline Replication policy leading to a cascading failure in the Tundra Storage Layer, and it lasted 3 hours and 12 minutes",
    "24 hours",
]

# Ingest once (does the actual chunking/embedding/reranker setup)
with open(PDF_PATH, "rb") as f:
    ingest_pdf(file_bytes=f.read(), thread_id=THREAD_ID, filename=PDF_PATH)

# problem came during evaluating
# Each chat call below uses a different thread_id (THREAD_ID_0, THREAD_ID_1, ...)
# so conversation history doesn't accumulate across questions (this avoids the
# TPM crash from growing history). But rag_tool looks up the retriever by
# THAT thread_id, and it was only ever registered under the base THREAD_ID --
# so every rag_tool call during eval was failing silently, causing the model
# to fall back to web search and loop. Fix: register the same retriever/
# metadata objects under every per-question sub-thread too. This is just a
# dict assignment (no re-embedding), so it's instant.
base_retriever = _THREAD_RETRIEVERS[str(THREAD_ID)]
base_metadata = _THREAD_METADATA[str(THREAD_ID)]
for i in range(len(test_questions)):
    _THREAD_RETRIEVERS[f"{THREAD_ID}_{i}"] = base_retriever
    _THREAD_METADATA[f"{THREAD_ID}_{i}"] = base_metadata


def invoke_with_retry(chatbot, messages, config, max_retries=5):
    """Runs chatbot.invoke normally with no delay. Only backs off if a rate
    limit error is actually hit. Also retries on malformed tool-call
    generations (output_parse_failed / tool_use_failed), which are usually
    transient."""
    for attempt in range(max_retries):
        try:
            return chatbot.invoke(messages, config=config)
        except APIStatusError as e:
            if "rate_limit_exceeded" in str(e) or "413" in str(e) or "429" in str(e):
                wait_time = 10 * (attempt + 1)
                print(f"Rate limit hit on attempt {attempt + 1}, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise
        except BadRequestError as e:
            if "output_parse_failed" in str(e) or "tool_use_failed" in str(e):
                print(f"Malformed generation on attempt {attempt + 1}, retrying immediately...")
                continue
            else:
                raise
    raise RuntimeError("Max retries exceeded.")


answers = []
contexts = []

for i, q in enumerate(test_questions):
    eval_config = {"configurable": {"thread_id": f"{THREAD_ID}_{i}"}}
    try:
        result = invoke_with_retry(
            chatbot1,
            {"messages": [{"role": "user", "content": q}]},
            eval_config,
        )
        final_answer = result["messages"][-1].content
    except GraphRecursionError:
        print(f"Question {i+1} hit recursion limit. Recording as failed and moving on.")
        final_answer = "No answer generated -- agent exceeded tool-call recursion limit."

    answers.append(final_answer)

    # retrieve context using the SAME sub-thread id the chat call used, for consistency
    retriever = _get_retriever(f"{THREAD_ID}_{i}")
    docs = retriever.invoke(q)
    contexts.append([d.page_content for d in docs])

    if i < len(test_questions) - 1:
        time.sleep(20)

# RAGAS
#judge_llm = LangchainLLMWrapper(ChatGroq(model="openai/gpt-oss-120b"))
judge_llm = LangchainLLMWrapper(ChatGroq(model="openai/gpt-oss-20b"))

eval_data = Dataset.from_dict({
    "question": test_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

result = evaluate(
    eval_data,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=LangchainEmbeddingsWrapper(embeddings),
    run_config=RunConfig(max_workers=1,timeout=120)
)

print(result)
result.to_pandas().to_csv("ragas_results_with_reranker.csv", index=False)