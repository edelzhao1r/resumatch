"""Resume parsing utilities (PDF + DOCX)."""

from __future__ import annotations

import os
from typing import IO, Union

import pdfplumber
from docx import Document

# Streamlit's UploadedFile satisfies a file-like interface; we accept either
# a path or a file-like object so the same helper works for CLI and web.
ResumeSource = Union[str, IO[bytes]]


def parse_resume_pdf(pdf_source: ResumeSource) -> str:
    """Extract plain text from a PDF resume.

    Args:
        pdf_source: Path to a PDF file or a file-like object (e.g. Streamlit upload).

    Returns:
        The concatenated text content of the PDF, one page per chunk separated
        by blank lines.

    Raises:
        ValueError: If the PDF cannot be parsed or contains no extractable text
            (e.g. a scanned image-only PDF).
    """
    try:
        with pdfplumber.open(pdf_source) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text)
    except Exception as exc:
        raise ValueError(
            "We couldn't read your PDF. Please make sure it's a valid, "
            "non-corrupted PDF file and try again."
        ) from exc

    if not pages_text:
        raise ValueError(
            "No text could be extracted from this PDF. It may be a scanned "
            "image — please upload a text-based PDF (or run OCR first)."
        )

    return "\n\n".join(pages_text).strip()


def parse_resume_docx(file_path: ResumeSource) -> str:
    """Extract plain text from a .docx resume using python-docx.

    Args:
        file_path: Path to the .docx file or a file-like object.

    Returns:
        All paragraph text joined by newlines.

    Raises:
        ValueError: If the document is malformed or contains no text.
    """
    try:
        # Reset the read position if it's a file-like object (Streamlit uploads
        # may have been peeked at by the unified dispatcher).
        if hasattr(file_path, "seek"):
            try:
                file_path.seek(0)
            except Exception:
                pass
        document = Document(file_path)
        paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "We couldn't read your .docx file. Please make sure it's a valid, "
            "non-corrupted Word document and try again."
        ) from exc

    if not paragraphs:
        raise ValueError(
            "No text could be extracted from this .docx file. The document "
            "appears to be empty."
        )

    return "\n".join(paragraphs).strip()


def parse_resume(file_source: ResumeSource) -> str:
    """Detect file type by extension and dispatch to the right parser.

    Accepts a path (str) or any file-like object that exposes a ``.name``
    attribute (e.g. Streamlit's UploadedFile).
    """
    name = getattr(file_source, "name", None) or (
        file_source if isinstance(file_source, str) else ""
    )
    ext = os.path.splitext(str(name).lower())[1]

    if ext == ".pdf":
        return parse_resume_pdf(file_source)
    if ext == ".docx":
        return parse_resume_docx(file_source)

    raise ValueError(
        "Unsupported resume format. Please upload a .pdf or .docx file."
    )


__all__ = ["parse_resume", "parse_resume_pdf", "parse_resume_docx"]
