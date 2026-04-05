"""
utils/rag.py — File text extraction for optional RAG context injection.
Supports: PDF, DOCX, plain TXT.
"""
from __future__ import annotations
import io
import logging

logger = logging.getLogger(__name__)

def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    try:
        if name.endswith(".pdf"):
            return _extract_pdf(raw_bytes)
        elif name.endswith(".docx"):
            return _extract_docx(raw_bytes)
        else:
            return raw_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.error(f"Text extraction failed for {name}: {exc}")
        return ""

def _extract_pdf(raw_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [page.extract_text().strip() for page in reader.pages if page.extract_text()]
    return "\n\n".join(pages)

def _extract_docx(raw_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(raw_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
