import pickle
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)
import time

start_time = time.time()
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

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="manuals",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

points = []

for i, emb in enumerate(embeddings):

    points.append(
        PointStruct(
            id=i,
            vector=emb.tolist(),
            payload=metadata[i]
        )
    )
import time

start = time.time()

client.upsert(
    collection_name="manuals",
    points=points
)

end = time.time()

print(
    f"Qdrant Ingestion: {(end-start):.4f} sec"
)
print(client.get_collections())
query = "How to remove oil filter?"

query_embedding = model.encode(
    query,
    normalize_embeddings=True
)
start = time.time()
results = client.query_points(
    collection_name="manuals",
    query=query_embedding,
    limit=5
).points
end = time.time()

print(
    f"Qdrant Query: {(end-start)*1000:.2f} ms"
)
for hit in results:

    print(
        hit.payload["pdf"],
        hit.payload["page"]
    )