"""
Document reader utility — supports PDF, DOCX, and TXT files.
"""

from pathlib import Path

import pdfplumber
from docx import Document


def read_document(file_path: str) -> str:
    """
    Read text content from a PDF, DOCX, or TXT file.

    Args:
        file_path: Path to the document file.

    Returns:
        Extracted text as a string.

    Raises:
        ValueError: If the file format is unsupported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _read_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return _read_docx(file_path)
    elif ext == ".txt":
        return _read_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .docx, .txt")


def read_document_from_bytes(content: bytes, filename: str) -> str:
    """
    Read text content from in-memory file bytes (for Streamlit uploads).

    Args:
        content: File bytes.
        filename: Original filename (used to detect format).

    Returns:
        Extracted text as a string.
    """
    import tempfile
    import os

    ext = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return read_document(tmp_path)
    finally:
        os.unlink(tmp_path)


def _read_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def _read_docx(file_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs).strip()


def _read_txt(file_path: str) -> str:
    """Read plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
