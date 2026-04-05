"""
utils/rag.py
------------
Extracts plain text from Streamlit UploadedFile objects.
Supports: PDF, DOCX, TXT
"""

from __future__ import annotations

import io
from typing import Optional

import streamlit as st


def extract_text(uploaded_file) -> Optional[str]:
    """
    Extract plain text from a Streamlit UploadedFile.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The file object returned by st.file_uploader.

    Returns
    -------
    str | None
        Extracted text, or None if the file type is unsupported or extraction fails.
    """
    if uploaded_file is None:
        return None

    filename: str = uploaded_file.name.lower()
    raw_bytes: bytes = uploaded_file.read()

    try:
        if filename.endswith(".pdf"):
            return _extract_pdf(raw_bytes)
        elif filename.endswith(".docx"):
            return _extract_docx(raw_bytes)
        elif filename.endswith(".txt"):
            return _extract_txt(raw_bytes)
        else:
            st.warning(
                f"Unsupported file type: '{uploaded_file.name}'. "
                "Please upload a PDF, DOCX, or TXT file."
            )
            return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to extract text from '{uploaded_file.name}': {exc}")
        return None
    finally:
        # Always reset read pointer so the file can be re-read if needed.
        uploaded_file.seek(0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_pdf(raw_bytes: bytes) -> str:
    """Extract text from a PDF byte string using pypdf."""
    from pypdf import PdfReader  # lazy import

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)


def _extract_docx(raw_bytes: bytes) -> str:
    """Extract text from a DOCX byte string using python-docx."""
    from docx import Document  # lazy import

    doc = Document(io.BytesIO(raw_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def _extract_txt(raw_bytes: bytes) -> str:
    """Decode a plain-text byte string, falling back to latin-1 on errors."""
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1")
