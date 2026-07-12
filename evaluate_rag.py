import sys
sys.path.append(".")  # so it can import your backend

from langgraph_rag_backend import chatbot1, ingest_pdf, _get_retriever
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from ragas.embeddings import LangchainEmbeddingsWrapper
from langgraph_rag_backend import embeddings 
#Ingesttest PDF into a fresh thread 
THREAD_ID = "eval_thread_1"
PDF_PATH = "MACHINE LEARNING(R20A0518).pdf"

with open(PDF_PATH, "rb") as f:
    ingest_pdf(file_bytes=f.read(), thread_id=THREAD_ID, filename=PDF_PATH)

test_questions = [
    "Who prepared these digital notes on Machine Learning?",
    "What is the course code for this Machine Learning subject?",
    "What are the three types of Machine Learning mentioned?",
    "What formula is used for min-max normalization?",
    "Who introduced the Principal Component Analysis method?",
    "What are the two components of dimensionality reduction?",
    "What does MAE stand for in performance metrics?",
    "What is the name of the method used in Random Forest that combines multiple decision trees?",
]
ground_truths = [
   "Padmaja",
    "R20A0518",
    "Supervised learning, Unsupervised learning, and Reinforcement learning",
    "Xn = (X - Xminimum) / (Xmaximum - Xminimum)",
    "Karl Pearson",
    "Feature selection and feature extraction",
    "Mean Absolute Error",
    "Bagging (Bootstrap Aggregation)",
]

# Run each question through  chatbot 
answers = []
contexts = []

config = {"configurable": {"thread_id": THREAD_ID}}

for q in test_questions:
    result = chatbot1.invoke({"messages": [{"role": "user", "content": q}]}, config=config)
    final_answer = result["messages"][-1].content
    answers.append(final_answer)

    retriever = _get_retriever(THREAD_ID)
    docs = retriever.invoke(q)
    contexts.append([d.page_content for d in docs])

# RAGAS 
judge_llm = LangchainLLMWrapper(ChatGroq(model="openai/gpt-oss-120b"))

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
)
print(result)
result.to_pandas().to_csv("ragas_results.csv", index=False)