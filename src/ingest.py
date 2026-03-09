from pathlib import Path
import json
from rag.io import list_pdfs, read_pdf_text
from rag.chunking import chunk_text

RAW = Path("data/raw")
OUT = Path("data/processed/chunks.jsonl")

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for pdf in list_pdfs(RAW):
            text = read_pdf_text(pdf)
            doc_id = pdf.stem
            chunks = chunk_text(doc_id, text)
            for c in chunks:
                rec = {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text, "source": pdf.name}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Ingested {pdf.name}: {len(chunks)} chunks")

if __name__ == "__main__":
    main()