from chatbot.rag_engine import ask_question
import json

queries = [
    "What is mentioned in bsamanual?",
    "What is in new.pdf?",
    "What is in new2.pdf?"
]

for q in queries:
    print(f"\nQUERY: {q}")
    result = ask_question(q)
    print(f"ANSWER: {result['answer']}")
    print("-" * 40)
