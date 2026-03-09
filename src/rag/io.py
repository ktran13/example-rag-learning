from pathlib import Path
from pypdf import PdfReader

def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            texts.append(t)
    return "\n".join(texts)

def list_pdfs(folder: Path):
    return sorted(folder.glob("*.pdf"))