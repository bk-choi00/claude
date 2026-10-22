"""extract stage: deterministic vector extraction of P&ID objects from readable PDFs.

Entry points: :func:`extract_pdf` (a file) and :func:`extract_page` (one PyMuPDF page).
"""

from .pipeline import ExtractOptions, extract_page, extract_pdf

__all__ = ["ExtractOptions", "extract_page", "extract_pdf"]
