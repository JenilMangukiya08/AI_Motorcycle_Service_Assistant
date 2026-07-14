import pickle
import os

BASE_DIR = os.path.join(os.getcwd(), "chatbot")
chunk_metadata_path = os.path.join(BASE_DIR, "chunk_metadata.pkl")

if os.path.exists(chunk_metadata_path):
    with open(chunk_metadata_path, "rb") as f:
        meta = pickle.load(f)
    
    print(f"Total chunks: {len(meta)}")
    
    pdf_counts = {}
    for m in meta:
        pdf = m.get("pdf", "Unknown")
        pdf_counts[pdf] = pdf_counts.get(pdf, 0) + 1
    
    print("PDF counts in metadata:")
    for pdf, count in pdf_counts.items():
        print(f"  {pdf}: {count}")
    
    print("\nFirst 5 entries:")
    for i in range(min(5, len(meta))):
        print(meta[i])
else:
    print(f"File not found: {chunk_metadata_path}")
