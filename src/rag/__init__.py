# src/rag/__init__.py

from .chunking import chunk_text
from .io import read_pdf_text, list_pdfs
from .embeddings import embed_texts
from .vectordb import FaissStore, Meta
from dotenv import load_dotenv

load_dotenv()