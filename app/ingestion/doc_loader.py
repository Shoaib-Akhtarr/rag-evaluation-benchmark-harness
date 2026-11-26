import io
import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader


def chunk_text(text: str, chunk_words: int = 95, overlap: int = 15) -> List[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)] if words else []
    chunks = []
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        part = " ".join(words[start : start + chunk_words])
        if part:
            chunks.append(part)
        if start + chunk_words >= len(words):
            break
    return chunks


def extract_numeric_invariants(text: str) -> List[str]:
    return re.findall(r"(?:\$?\d[\d,]*(?:\.\d+)?%?)", text)


def parse_document_bytes(filename: str, raw_bytes: bytes) -> List[Dict[str, Any]]:
    ext = os.path.splitext(filename)[1].lower()
    pages_text = []
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw_bytes))
        for page_num, page in enumerate(reader.pages, 1):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages_text.append((page_num, txt))
    else:
        txt = raw_bytes.decode("utf-8", errors="ignore").strip()
        if txt:
            pages_text.append((1, txt))

    doc_id = os.path.splitext(filename)[0]
    chunks = []
    idx = 0
    for page_num, p_text in pages_text:
        for seg in chunk_text(p_text):
            idx += 1
            chunks.append({
                "chunk_id": f"{doc_id}::p{page_num}::c{idx}",
                "doc_id": doc_id,
                "filename": filename,
                "page": page_num,
                "content": seg,
                "numeric_invariants": extract_numeric_invariants(seg),
            })
    return chunks


def load_directory(docs_dir: str) -> List[Dict[str, Any]]:
    chunks = []
    if not os.path.exists(docs_dir):
        return chunks
    for fname in sorted(os.listdir(docs_dir)):
        if fname.lower().endswith((".pdf", ".md", ".txt")):
            with open(os.path.join(docs_dir, fname), "rb") as f:
                chunks.extend(parse_document_bytes(fname, f.read()))
    return chunks
