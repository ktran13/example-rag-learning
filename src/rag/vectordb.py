from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import faiss

@dataclass
class Meta:
    doc_id: str
    chunk_id: int
    text: str
    source: str  # filename

class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine if we normalize vectors
        self.metas = []

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(v)
        return v

    def add(self, vectors, metas):
        v = np.array(vectors, dtype="float32")
        self._normalize(v)
        self.index.add(v)
        self.metas.extend(metas)

    def search(self, query_vec, k=5):
        q = np.array([query_vec], dtype="float32")
        self._normalize(q)
        scores, idxs = self.index.search(q, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            results.append((float(score), self.metas[i]))
        return results

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "faiss.index"))
        with open(folder / "metas.jsonl", "w", encoding="utf-8") as f:
            for m in self.metas:
                f.write(json.dumps(m.__dict__, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, folder: Path):
        index = faiss.read_index(str(folder / "faiss.index"))
        metas = []
        with open(folder / "metas.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                metas.append(Meta(**json.loads(line)))
        store = cls(index.d)
        store.index = index
        store.metas = metas
        return store