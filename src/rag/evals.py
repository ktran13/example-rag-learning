from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from openai import OpenAI
from dotenv import load_dotenv

from .embeddings import embed_texts
from .vectordb import FaissStore, Meta
from .prompts import SYSTEM_PROMPT

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class EvalItem:
    """One evaluation case."""
    id: str
    question: str

    # Retrieval ground truth: which chunks should be retrieved (one or more).
    # Use (source, chunk_id) tuples. You can start with 1 target chunk per question.
    gold_chunks: List[Tuple[str, int]]

    # Optional short “expected answer” for human review (not used in metrics)
    notes: Optional[str] = None


@dataclass
class EvalResult:
    id: str
    question: str
    retrieved: List[Tuple[float, str, int]]  # score, source, chunk_id
    recall_at_k: Dict[int, float]
    mrr: float
    answer: str
    cited_chunks: List[Tuple[str, int]]
    grounded: Optional[bool]
    latency_s: float


# -----------------------------
# Helpers
# -----------------------------

CITE_RE = re.compile(r"\[([^\[\]:]+):(\d+)\]")  # matches [file.pdf:12]

def extract_citations(text: str) -> List[Tuple[str, int]]:
    cites = []
    for m in CITE_RE.finditer(text):
        cites.append((m.group(1), int(m.group(2))))
    # de-dupe preserve order
    seen = set()
    out = []
    for c in cites:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def compute_recall_at_k(retrieved, gold, ks=(1,3,5,10)):
    gold_set = set(tuple(x) for x in gold)
    retrieved_pairs = [(m.source, m.chunk_id) for _, m in retrieved]
    out = {}
    for k in ks:
        topk = set(retrieved_pairs[:k])
        out[k] = 1.0 if len(topk.intersection(gold_set)) > 0 else 0.0
    return out

def compute_mrr(retrieved, gold):
    gold_set = set(tuple(x) for x in gold)
    for rank, (_, m) in enumerate(retrieved, start=1):
        if (m.source, m.chunk_id) in gold_set:
            return 1.0 / rank
    return 0.0


def judge_groundedness(question: str, answer: str, sources: str, model: str = "gpt-4o-mini") -> bool:
    """
    LLM-as-judge: returns True if answer is supported by sources and does not introduce unsupported facts.
    Keep this strict. In gov settings, prefer false negatives over false positives.
    """
    rubric = """You are grading whether an answer is fully supported by the provided sources.
Return ONLY JSON with keys:
- grounded: true/false
- rationale: short (<=2 sentences)

Rules:
- grounded=true only if every factual claim in the answer is supported by the sources.
- If answer says "I don't know" and sources truly lack the answer, grounded=true.
- If the answer contains any unsupported claim, grounded=false.
"""
    prompt = f"""Question:
{question}

Answer:
{answer}

Sources:
{sources}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    txt = resp.choices[0].message.content.strip()
    try:
        data = json.loads(txt)
        return bool(data.get("grounded", False))
    except Exception:
        # If judge returns non-JSON, be conservative
        return False


# -----------------------------
# Core eval runner
# -----------------------------

def run_rag_once(
    store: FaissStore,
    question: str,
    embed_model: str,
    chat_model: str,
    k: int = 6,
    max_context_chars: int = 9000,
) -> Tuple[List[Tuple[float, Meta]], str, float, str]:
    """
    Returns (retrieved, answer, latency, context_sources_text)
    """
    t0 = time.time()

    qvec = embed_texts([question], model=embed_model)[0]
    hits = store.search(qvec, k=k)  # returns List[(score, Meta)]
    # build sources text
    blocks = []
    for score, meta in hits:
        blocks.append(f"[{meta.source}:{meta.chunk_id}] {meta.text}")
    sources = "\n\n".join(blocks)[:max_context_chars]

    user_prompt = f"""Question: {question}

Sources:
{sources}

Answer with citations in-line like [source:chunk_id]. If sources don't contain the answer, say "I don't know based on the provided documents."""

    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    answer = resp.choices[0].message.content
    latency = time.time() - t0
    return hits, answer, latency, sources


def run_evals(
    eval_path: Path = Path("data/evals/eval_set.jsonl"),
    out_path: Path = Path("data/evals/results.jsonl"),
    index_dir: Path = Path("data/index"),
    embed_model: str | None = None,
    chat_model: str | None = None,
    k: int = 6,
    do_groundedness_judge: bool = True,
) -> Dict[str, float]:
    embed_model = embed_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    chat_model = chat_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    store = FaissStore.load(index_dir)

    items: List[EvalItem] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["gold_chunks"] = [tuple(x) for x in r["gold_chunks"]]
            items.append(EvalItem(**r))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg_recall = {1:0.0, 3:0.0, 5:0.0, 10:0.0}
    agg_mrr = 0.0
    grounded_true = 0
    grounded_total = 0

    with out_path.open("w", encoding="utf-8") as out:
        for it in items:
            hits, answer, latency, sources = run_rag_once(
                store=store,
                question=it.question,
                embed_model=embed_model,
                chat_model=chat_model,
                k=k,
            )

            recall = compute_recall_at_k(hits, it.gold_chunks, ks=(1,3,5,10))
            mrr = compute_mrr(hits, it.gold_chunks)

            cited = extract_citations(answer)

            grounded = None
            if do_groundedness_judge:
                grounded = judge_groundedness(it.question, answer, sources, model=chat_model)
                grounded_total += 1
                grounded_true += 1 if grounded else 0

            # write result
            res = EvalResult(
                id=it.id,
                question=it.question,
                retrieved=[(s, m.source, m.chunk_id) for s, m in hits],
                recall_at_k=recall,
                mrr=mrr,
                answer=answer,
                cited_chunks=cited,
                grounded=grounded,
                latency_s=latency,
            )
            out.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")

            for kk, v in recall.items():
                agg_recall[kk] += v
            agg_mrr += mrr

    n = max(1, len(items))
    summary = {
        "n": float(n),
        "recall@1": agg_recall[1]/n,
        "recall@3": agg_recall[3]/n,
        "recall@5": agg_recall[5]/n,
        "recall@10": agg_recall[10]/n,
        "mrr": agg_mrr/n,
        "grounded_rate": (grounded_true/grounded_total) if grounded_total else None,
    }
    return summary


# -----------------------------
# Utilities to help build eval set
# -----------------------------

def make_template_eval_set(out_path: Path = Path("data/evals/eval_set.jsonl")):
    """
    Creates a starter eval set template for you to fill in.
    You will replace gold_chunks after you discover the right ones.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = [
        EvalItem(id="q1", question="What is XXX?", gold_chunks=[("YOURFILE.pdf", 0)], notes="Find the XX clause."),
        EvalItem(id="q2", question="What is XXX?", gold_chunks=[("YOURFILE.pdf", 0)], notes="Find XXX milestones."),
        EvalItem(id="q3", question="Other question XXX?", gold_chunks=[("YOURFILE.pdf", 0)], notes="Find XXX language."),
    ]
    with out_path.open("w", encoding="utf-8") as f:
        for it in template:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    print(f"Wrote template eval set to {out_path}")