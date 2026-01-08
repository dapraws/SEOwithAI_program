import json
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "./data/apps.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Load model
model = SentenceTransformer(MODEL_NAME)

# Load database
with open(DB_PATH, "r", encoding="utf-8") as f:
    db = json.load(f)

apps = db["apps"]

# Pre-load embeddings
app_embeddings = []
valid_apps = []

for app in apps:
    if app.get("embedding"):
        app_embeddings.append(app["embedding"])
        valid_apps.append(app)

app_embeddings = np.array(app_embeddings)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_apps(query: str, top_k: int = TOP_K):
    query_embedding = model.encode(query)

    scores = []

    for idx, app_embedding in enumerate(app_embeddings):
        score = cosine_similarity(query_embedding, app_embedding)
        scores.append((score, valid_apps[idx]))

    scores.sort(key=lambda x: x[0], reverse=True)

    return scores[:top_k]


# ===== CLI TEST =====
if __name__ == "__main__":
    while True:
        query = input("\n🔎 Search (ketik 'exit' untuk keluar): ")
        if query.lower() == "exit":
            break

        results = search_apps(query)

        print("\n📌 Hasil Pencarian:")
        for rank, (score, app) in enumerate(results, start=1):
            print(
                f"{rank}. {app['name']} "
                f"(category: {app['category']}, score: {score:.3f})"
            )
