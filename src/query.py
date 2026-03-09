from pathlib import Path
import os, time
from openai import OpenAI
from rag.embeddings import embed_texts
from rag.vectordb import FaissStore
from rag.prompts import SYSTEM_PROMPT
from rag.guardrails import looks_like_prompt_injection
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
INDEX_DIR = Path("data/index")

def main():
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

    store = FaissStore.load(INDEX_DIR)

    while True:
        q = input("\nAsk> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        if looks_like_prompt_injection(q):
            print("⚠️ Query looks like prompt injection. Refusing; please rephrase.")
            continue

        t0 = time.time()
        qvec = embed_texts([q], model=embed_model)[0]
        hits = store.search(qvec, k=6)
        
        # If retrieval similarity is weak, refuse to answer
        if not hits or hits[0][0] < 0.20:
            print("I don't know based on the provided documents (retrieval confidence low).")
            continue

        context_blocks = []
        citations = []
        for score, meta in hits:
            context_blocks.append(f"[{meta.source}:{meta.chunk_id}] {meta.text}")
            citations.append(f"[{meta.source}:{meta.chunk_id}]")

        context = "\n\n".join(context_blocks)

        user_prompt = f"""Question: {q}
Sources:
{context}

Instructions:
- Answer only using the sources above.
- Every sentence must include one or more citations.
- Use citations in exactly this format: [source:chunk_id]
- Example citation: [VenueContract.pdf:12]
- If you cannot answer from the sources, say exactly: "I don't know based on the provided documents."
"""

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": user_prompt},
            ],
        )

        dt = time.time() - t0
        answer = resp.choices[0].message.content
        print(f"\nAnswer (latency {dt:.2f}s):\n{answer}")

if __name__ == "__main__":
    main()