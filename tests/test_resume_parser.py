import pytest
import pymupdf

from app.resume_parser import (
    MAX_FILE_SIZE_BYTES,
    ResumeParseError,
    extract_text_from_pdf,
    extract_text_from_txt,
    parse_resume,
    validate_file_metadata,
)


def _create_sample_pdf(text: str = "") -> bytes:
    """Helper to generate in-memory PDF bytes using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_valid_txt_extraction():
    """Verify clean text extraction from valid TXT content."""
    sample_text = "Jane Doe\nSenior Python Engineer\n6 years experience in FastAPI and cloud systems."
    content = sample_text.encode("utf-8")

    result = parse_resume(filename="resume.txt", content=content)
    assert result == sample_text


def test_valid_pdf_extraction():
    """Verify PyMuPDF text extraction from a valid PDF."""
    sample_text = (
        "Jane Doe\n"
        "Staff Software Engineer with 8+ years developing scalable distributed systems.\n"
        "Expert in Python, AWS, Docker, Kubernetes, and REST API development."
    )
    pdf_bytes = _create_sample_pdf(sample_text)

    result = parse_resume(filename="jane_doe_resume.pdf", content=pdf_bytes)
    assert "Staff Software Engineer" in result
    assert "Expert in Python" in result


def test_empty_txt_rejected():
    """Verify empty or whitespace-only TXT files raise ResumeParseError."""
    with pytest.raises(ResumeParseError) as exc_info:
        parse_resume(filename="empty.txt", content=b"")
    assert exc_info.value.error == "unparsable_resume"
    assert exc_info.value.status_code == 422

    with pytest.raises(ResumeParseError) as exc_info:
        parse_resume(filename="whitespace.txt", content=b"   \n\t  \n")
    assert exc_info.value.error == "unparsable_resume"


def test_pdf_magic_byte_validation():
    """Verify PDF without %PDF- magic signature is rejected before parsing."""
    corrupt_content = b"This is plain text with a .pdf extension"
    with pytest.raises(ResumeParseError) as exc_info:
        parse_resume(filename="fake.pdf", content=corrupt_content)

    assert exc_info.value.error == "unparsable_resume"
    assert "Missing %PDF- signature" in exc_info.value.detail
    assert exc_info.value.status_code == 422


def test_corrupt_pdf_rejected():
    """Verify malformed PDF payload with valid header raises clean ResumeParseError."""
    corrupt_pdf = b"%PDF-1.5\nMalformed binary content that cannot be parsed by PyMuPDF \x00\xff\xfe"
    with pytest.raises(ResumeParseError) as exc_info:
        parse_resume(filename="corrupt.pdf", content=corrupt_pdf)

    assert exc_info.value.error == "unparsable_resume"
    assert "Corrupt or unreadable PDF" in exc_info.value.detail
    assert exc_info.value.status_code == 422


def test_empty_or_scanned_pdf_rejected():
    """Verify valid PDF with no extractable text (scanned/OCR-less) is rejected."""
    # Blank PDF page without text
    empty_pdf_bytes = _create_sample_pdf(text="")

    with pytest.raises(ResumeParseError) as exc_info:
        parse_resume(filename="scanned_resume.pdf", content=empty_pdf_bytes)

    assert exc_info.value.error == "unparsable_resume"
    assert "insufficient content" in exc_info.value.detail
    assert exc_info.value.status_code == 422


def test_unsupported_file_extension_rejected():
    """Verify files with unsupported extensions (.docx, .png, etc.) are rejected."""
    content = b"Some resume text"
    for invalid_filename in ["resume.docx", "resume.doc", "resume.png", "resume.json"]:
        with pytest.raises(ResumeParseError) as exc_info:
            parse_resume(filename=invalid_filename, content=content)

        assert exc_info.value.error == "unsupported_file_type"
        assert exc_info.value.status_code == 422


def test_file_larger_than_5mb_rejected():
    """Verify files exceeding 5MB limit are rejected with status 413."""
    oversized_content = b"A" * (MAX_FILE_SIZE_BYTES + 1024)

    with pytest.raises(ResumeParseError) as exc_info:
        parse_resume(filename="huge_resume.txt", content=oversized_content)

    assert exc_info.value.error == "file_too_large"
    assert exc_info.value.status_code == 413


def test_extracted_text_fidelity_no_fabrication():
    """Verify extracted text matches input exactly without fabrication or injection."""
    original_text = "Strictly testing that no LLM or synthetic scoring is invoked at parse time."
    extracted = extract_text_from_txt(original_text.encode("utf-8"))
    assert extracted == original_text
