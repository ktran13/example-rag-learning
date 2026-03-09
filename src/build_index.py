from pathlib import Path
import json
from rag.embeddings import embed_texts
from rag.vectordb import FaissStore, Meta
import os
from dotenv import load_dotenv

load_dotenv()

CHUNKS = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("data/index")

def main():
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

    texts = []
    metas = []
    with open(CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            metas.append(Meta(doc_id=r["doc_id"], chunk_id=r["chunk_id"], text=r["text"], source=r["source"]))

    vectors = []
    batch = 64
    for i in range(0, len(texts), batch):
        vectors.extend(embed_texts(texts[i:i+batch], model=model))
        print(f"Embedded {min(i+batch, len(texts))}/{len(texts)}")

    dim = len(vectors[0])
    store = FaissStore(dim)
    store.add(vectors, metas)
    store.save(INDEX_DIR)
    print(f"Saved index to {INDEX_DIR}")

if __name__ == "__main__":
    main()