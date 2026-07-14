import os
import django

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "RAGCHATBOT.settings"
)

django.setup()

from chatbot.rag_engine import ask_question

benchmark_questions = [
    {"question": "How to replace engine oil filter?","expected_page": 221},
    {"question": "part number of engine mounting fixture?","expected_page": 34},
    {"question": "what are the points about fuel lines?","expected_page": 46},
    {"question": "how to remove and install spark plug?","expected_page": 49},
    {"question": "what is the process for valve clearance?","expected_page": 50},
    {"question": "LH Side and RH Side Panels Removal and Installation Procedure?","expected_page": 63},
    {"question": "REMOVAL AND ASSEMBLY OF BATTERY BOX PROCESS?","expected_page": 74},
    {"question": "How to remove lower fender?","expected_page": 84},
    {"question": "Maintenance check points for Front fork?","expected_page": 105},
    {"question": "DISASSEMBLY OF HANDLEBAR & MOUNTINGS?","expected_page": 110},
    {"question": "DISASSEMBLY AND ASSEMBLY OF SUB-FRAME?","expected_page": 128},
    {"question": "Safety Measures during Assembly process of swing arm?","expected_page": 149},
    {"question": "ASSEMBLY OF THROTTLE BODY?","expected_page": 162},
    {"question": "ENGINE COMPRESSOR PRESSURE CHECKING?","expected_page": 203},
    {"question": "ECU REMOVAL PROCEDURE?","expected_page":333 },
    {"question": "how to remove steering stem ?","expected_page":116 }
]

top1_correct = 0
top5_correct = 0

for test in benchmark_questions:


    result = ask_question(
        test["question"],
        selected_bike="scrambler"
    )

    chunks = result["retrieved_chunks"]

    if not chunks:
        continue
    
    # Top-1
    if chunks[0]["page"] == test["expected_page"]:
        top1_correct += 1

    # Top-5
    retrieved_pages = [
        chunk["page"]
        for chunk in chunks[:5]
    ]

    if test["expected_page"] in retrieved_pages:
        top5_correct += 1

    print(
        test["question"],
        "Expected:",
        test["expected_page"],
        "Retrieved:",
        retrieved_pages
    )
    

    
top1_accuracy = (
    top1_correct / len(benchmark_questions)
) * 100

top5_accuracy = (
    top5_correct / len(benchmark_questions)
) * 100

print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
