from __future__ import annotations

from pathlib import Path
from typing import List


def render_pdf_pages(
    pdf_path: str,
    out_dir: Path,
    dpi: int = 144,
    max_pages: int = 0,
) -> List[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError(
            "PyMuPDF is required for PDF rendering: pip install PyMuPDF"
        ) from e

    pdf = Path(pdf_path)
    if not pdf.is_file():
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf.stem

    doc = fitz.open(pdf)
    try:
        n = doc.page_count
        if max_pages > 0:
            n = min(n, max_pages)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        outs: List[str] = []
        for i in range(n):
            out_path = out_dir / f"{stem}_p{i+1:02d}.png"
            if not out_path.is_file():
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pix.save(str(out_path))
            outs.append(str(out_path))
        return outs
    finally:
        doc.close()


def extract_pdf_text(pdf_path: str, max_chars: int = 4000) -> str:
    try:
        import fitz
    except ImportError:
        return ""
    pdf = Path(pdf_path)
    if not pdf.is_file():
        return ""
    try:
        doc = fitz.open(pdf)
    except Exception:
        return ""
    try:
        chunks: List[str] = []
        total = 0
        for page in doc:
            t = page.get_text("text") or ""
            chunks.append(t)
            total += len(t)
            if max_chars > 0 and total >= max_chars:
                break
        text = "\n".join(chunks).strip()
        if max_chars > 0:
            text = text[:max_chars]
        return text
    finally:
        doc.close()
