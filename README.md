# 🚀 SEO with AI (program version)

**AI-powered search engine** menggunakan **Sentence Transformers** untuk meningkatkan relevansi pencarian aplikasi berbasis **semantic search + keyword hybrid scoring**.

---

## ✨ Features

- ✅ Semantic Search menggunakan embeddings
- ✅ Hybrid Search
  - 50% Semantic similarity
  - 20% Keyword matching
  - 20% Category boost

---

## 🗂 Project Structure

```

SEOwithAI_program/
├── data/
│   └── apps.json          # Database aplikasi + embedding
├── generate_embeddings.py # Script generate & update embedding
├── search_apps.py         # Hybrid semantic search engine
└── README.md

```

---

## 📄 Data Schema (`apps.json`)

```json
{
  "meta": {
    "version": "1.0",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "last_updated": "2026-01-08"
  },
  "apps": [
    {
      "id": "app_001",
      "name": "HR-X9",
      "aliases": ["absensi HR", "attendance app"],
      "category": "Human Resource",
      "description": "Aplikasi absensi karyawan berbasis mobile.",
      "features": ["GPS tracking", "face recognition"],
      "search_text": "...",
      "embedding": [],
      "content_hash": "",
      "status": "active"
    }
  ]
}
```

### 🔑 Important Fields

| Field          | Description                     |
| -------------- | ------------------------------- |
| `search_text`  | Text utama untuk embedding      |
| `embedding`    | Vector hasil encoding           |
| `content_hash` | Untuk mendeteksi perubahan data |
| `aliases`      | Sinonim / variasi nama          |
| `features`     | Keyword pendukung               |
| `category`     | Untuk boosting relevansi        |

---

## 🧠 Embedding Model

Menggunakan model open-source:

```
sentence-transformers/all-MiniLM-L6-v2
```

**Kenapa model ini?**

- ⚡ Cepat & ringan
- 📦 Gratis (tanpa API key)
- 🎯 Akurasi bagus untuk semantic search

---

## ⚙️ Installation

### Install dependencies

```bash
pip install sentence-transformers numpy
```

---

## 🧩 Generate Embeddings

- Generate embedding baru
- Skip data yang tidak berubah (hash-based)
- Update `last_updated`

```bash
python generate_embeddings.py
```

Output example:

```
🔄 Generating embedding for app_001 (HR-X9)
✅ Embedding updated: 1 app(s)
```

---

## 🔎 Run Search Engine

```bash
python search_apps.py
```

Contoh input:

```
aplikasi absensi karyawan HR
```

Contoh output:

```
1. HR-X9 (final: 0.812, semantic: 0.79, keyword: 0.80)
```

---

## 🧪 Scoring Formula

```python
final_score =
  0.5 * semantic_similarity
+ 0.2 * keyword_score
+ 0.2 * category_boost
```

### 🧩 Explanation

- **Semantic similarity** → makna konteks
- **Keyword score** → kecocokan literal
- **Category boost** → niat pencarian spesifik
