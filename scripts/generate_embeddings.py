import json
import hashlib
from datetime import datetime
from sentence_transformers import SentenceTransformer

DB_PATH = "./data/apps.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model (sekali saja)
model = SentenceTransformer(MODEL_NAME)


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Load database
with open(DB_PATH, "r", encoding="utf-8") as f:
    db = json.load(f)

updated_count = 0

for app in db["apps"]:
    search_text = app.get("search_text")

    if not search_text:
        print(f"⚠️  {app['id']} tidak punya search_text, skip")
        continue

    new_hash = hash_content(search_text)

    # Skip jika tidak berubah
    if app.get("content_hash") == new_hash and app.get("embedding"):
        continue

    print(f"🔄 Generating embedding for {app['id']} ({app['name']})")

    embedding = model.encode(search_text).tolist()

    app["embedding"] = embedding
    app["content_hash"] = new_hash
    app["embedding_model"] = MODEL_NAME

    updated_count += 1

# Update meta
db["meta"]["last_updated"] = datetime.utcnow().isoformat()

# Save back to file
with open(DB_PATH, "w", encoding="utf-8") as f:
    json.dump(db, f, indent=2, ensure_ascii=False)

print(f"✅ Embedding updated: {updated_count} app(s)")
