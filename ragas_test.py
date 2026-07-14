import os
import json

from dotenv import load_dotenv
from chatbot.rag_engine import ask_question
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

# Load environment variables
load_dotenv()

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

with open("evaluation_data.json", "r") as f:
    eval_data = json.load(f)

evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

evaluator_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

questions = []
answers = []
contexts = []
ground_truths = []

for item in eval_data:
    result = ask_question(
        item["question"],
        history=[],
        selected_bike="scrambler"
    )

    questions.append(item["question"])
    answers.append(result["answer"])
    contexts.append([
        chunk["content"]
        for chunk in result["retrieved_chunks"]
    ])
    ground_truths.append(item["ground_truth"])

dataset = Dataset.from_dict({
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
})

result = evaluate(
    dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

print(result)

df = result.to_pandas()
df.to_csv("ragas_report.csv", index=False)