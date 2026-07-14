import pickle
import chromadb
from sentence_transformers import SentenceTransformer

# Load data
with open("chatbot/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("chatbot/chunk_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="manuals"
)
import time

start = time.time()
BATCH_SIZE = 5000

for i in range(0, len(chunks), BATCH_SIZE):
    collection.add(
        documents=chunks[i:i+BATCH_SIZE],
        metadatas=metadata[i:i+BATCH_SIZE],
        embeddings=embeddings[i:i+BATCH_SIZE],
        ids=[str(j) for j in range(i, min(i+BATCH_SIZE, len(chunks)))]
    )
end = time.time()

print(
    f"Chroma Ingestion: {(end-start):.4f} sec"
)
query = "How to remove oil filter?"

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)
start = time.time()
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5
)
end = time.time()

print(
    f"Chroma Query: {(end-start)*1000:.2f} ms"
)
print(results)