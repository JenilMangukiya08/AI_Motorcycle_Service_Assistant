import json
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import evaluate
from chatbot.rag_engine import ask_question
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM
import os 
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ---------------- Groq LLM ----------------
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

class GroqEvalModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model
    def load_model(self):
        return self.model
    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "groq-llama"


evaluator_llm = GroqEvalModel(groq_llm)

# ---------------- Metrics ----------------
answer_metric = AnswerRelevancyMetric(
    model=evaluator_llm,
    threshold=0.7
)

faithfulness_metric = FaithfulnessMetric(
    model=evaluator_llm,
    threshold=0.7
)

# ---------------- Load data ----------------
with open("evaluation_data.json", "r") as f:
    eval_data = json.load(f)

test_cases = []

# ---------------- Run evaluation ----------------
for item in eval_data:

    result = ask_question(
        item["question"],
        history=[],
        selected_bike="scrambler"
    )

    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["ground_truth"],
        retrieval_context=[
            chunk["content"] for chunk in result["retrieved_chunks"]
        ]
    )

    test_cases.append(test_case)

    print("\nQuestion:", item["question"])

    answer_metric.measure(test_case)
    print("Answer Relevancy:", answer_metric.score)

    faithfulness_metric.measure(test_case)
    print("Faithfulness:", faithfulness_metric.score)

# ---------------- Final report ----------------
evaluate(test_cases)