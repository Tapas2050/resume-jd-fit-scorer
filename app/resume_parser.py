from pathlib import Path
import pymupdf

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB limit per Master §5.3
MIN_PDF_EXTRACTED_CHARS = 50  # Scanned/OCR-less threshold per Master §5.10
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
PDF_MAGIC_BYTES = b"%PDF-"


class ResumeParseError(Exception):
    """Exception raised when resume parsing or validation fails."""

    def __init__(self, error: str = "unparsable_resume", detail: str = "", status_code: int = 422):
        self.error = error
        self.detail = detail
        self.status_code = status_code
        super().__init__(f"{error}: {detail}")


def validate_file_metadata(filename: str, content: bytes) -> str:
    """
    Validate file size and extension.
    Returns normalized lowercase extension.
    """
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ResumeParseError(
            error="file_too_large",
            detail=f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB.",
            status_code=413,
        )

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ResumeParseError(
            error="unsupported_file_type",
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
            status_code=422,
        )

    return ext


def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract text from PDF content using PyMuPDF.
    Validates magic bytes and detects unparsable or OCR-less/scanned PDFs.
    """
    # Validate magic bytes first (Master §5.3)
    if not content.startswith(PDF_MAGIC_BYTES):
        raise ResumeParseError(
            error="unparsable_resume",
            detail="Invalid PDF file: Missing %PDF- signature.",
            status_code=422,
        )

    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ResumeParseError(
            error="unparsable_resume",
            detail=f"Corrupt or unreadable PDF: {exc}",
            status_code=422,
        )

    try:
        if doc.page_count == 0:
            raise ResumeParseError(
                error="unparsable_resume",
                detail="PDF contains no pages.",
                status_code=422,
            )

        extracted_pages: list[str] = []
        for page in doc:
            page_text = page.get_text()
            if page_text:
                extracted_pages.append(page_text)

        full_text = "\n".join(extracted_pages).strip()

        # Scanned / OCR-less check: threshold relative to document content (Master §2.4, §5.10)
        if len(full_text) < MIN_PDF_EXTRACTED_CHARS:
            raise ResumeParseError(
                error="unparsable_resume",
                detail="PDF text extraction returned empty or insufficient content (possible scanned/image-only PDF).",
                status_code=422,
            )

        return full_text
    finally:
        doc.close()


def extract_text_from_txt(content: bytes) -> str:
    """
    Extract text from TXT content safely.
    Handles encoding gracefully and rejects empty content.
    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception as exc:
            raise ResumeParseError(
                error="unparsable_resume",
                detail=f"Unable to decode text file: {exc}",
                status_code=422,
            )

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ResumeParseError(
            error="unparsable_resume",
            detail="Uploaded text file is empty.",
            status_code=422,
        )

    return cleaned_text


def parse_resume(filename: str, content: bytes) -> str:
    """
    Synchronous entry point for parsing resume files (.pdf and .txt).
    Enforces upload validation, magic byte verification, and text extraction.
    Designed as a synchronous function so FastAPI can dispatch it to a threadpool (Master §5.2).
    """
    ext = validate_file_metadata(filename=filename, content=content)

    if ext == ".pdf":
        return extract_text_from_pdf(content=content)
    elif ext == ".txt":
        return extract_text_from_txt(content=content)
    else:
        raise ResumeParseError(
            error="unsupported_file_type",
            detail=f"Unsupported file format '{ext}'.",
            status_code=422,
        )
