from dataclasses import dataclass
from typing import List
import tiktoken

@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str

def chunk_text(doc_id: str, text: str, chunk_tokens=450, overlap_tokens=80, encoding_name="o200k_base") -> List[Chunk]:
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    cid = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_tokens_slice = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens_slice)
        chunks.append(Chunk(doc_id=doc_id, chunk_id=cid, text=chunk_text_str))
        cid += 1
        start = end - overlap_tokens if end - overlap_tokens > start else end
    return chunks