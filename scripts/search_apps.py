import json
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "./data/apps.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

model = SentenceTransformer(MODEL_NAME)

with open(DB_PATH, "r", encoding="utf-8") as f:
    db = json.load(f)

apps = db["apps"]

app_embeddings = []
valid_apps = []

for app in apps:
    if app.get("embedding"):
        app_embeddings.append(app["embedding"])
        valid_apps.append(app)

app_embeddings = np.array(app_embeddings)
app_embeddings = app_embeddings / np.linalg.norm(
    app_embeddings, axis=1, keepdims=True
)


def keyword_score(query: str, app: dict) -> float:
    query = query.lower()

    text = " ".join(
        [
            app.get("name", ""),
            " ".join(app.get("aliases", [])),
            " ".join(app.get("features", [])),
        ]
    ).lower()

    score = 0.0
    for token in query.split():
        if token in text:
            score += 1

    return min(score / 5, 1.0) 


def category_boost(query: str, app: dict) -> float:
    if app.get("category", "").lower() in query.lower():
        return 1.0
    return 0.0


def search_apps(query: str, top_k: int = TOP_K):
    query_embedding = model.encode(query)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    semantic_scores = np.dot(app_embeddings, query_embedding)

    results = []

    for idx, app in enumerate(valid_apps):
        sem_score = float(semantic_scores[idx])
        key_score = keyword_score(query, app)
        cat_boost = category_boost(query, app)

        final_score = (
            0.7 * sem_score
            + 0.2 * key_score
            + 0.1 * cat_boost
        )

        results.append(
            {
                "app": app,
                "semantic_score": sem_score,
                "keyword_score": key_score,
                "category_boost": cat_boost,
                "final_score": final_score,
            }
        )

    results.sort(key=lambda x: x["final_score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    while True:
        query = input("\n🔎 Search (ketik 'exit' untuk keluar): ")
        if query.lower() == "exit":
            break

        results = search_apps(query)

        print("\n📌 Hasil Pencarian:")
        for i, r in enumerate(results, 1):
            app = r["app"]
            print(
                f"{i}. {app['name']} "
                f"(final: {r['final_score']:.3f}, "
                f"semantic: {r['semantic_score']:.3f}, "
                f"keyword: {r['keyword_score']:.2f})"
            )
