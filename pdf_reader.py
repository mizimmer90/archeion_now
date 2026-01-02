"""PDF text extraction functionality."""
from pathlib import Path
from typing import Optional
import PyPDF2


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text or None if extraction failed
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_parts = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num} of {pdf_path}: {e}")
                    continue
            
            return "\n\n".join(text_parts) if text_parts else None
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

