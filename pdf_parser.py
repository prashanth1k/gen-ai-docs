"""
PDF parser module for gen-ai-docs CLI tool.
Handles extracting and processing text content from PDF documents.
Optimized for text-only PDFs with no image processing.
"""

import os

# Lazy import for unstructured - only loaded when using enhanced modes
_unstructured_partition_pdf = None


def _ensure_unstructured():
    """Lazy load unstructured.partition.pdf"""
    global _unstructured_partition_pdf
    if _unstructured_partition_pdf is None:
        from unstructured.partition.pdf import partition_pdf
        _unstructured_partition_pdf = partition_pdf
    return _unstructured_partition_pdf


class SimpleTextElement:
    """Simple text element wrapper for compatibility with unstructured format"""
    def __init__(self, text):
        self.text = text


def parse_pdf_lightweight(pdf_path: str) -> list:
    """
    Ultra-fast PDF parsing using PyPDF2 (text-only, no formatting).
    Use this if unstructured is still too slow.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of SimpleTextElement objects
    """
    try:
        import PyPDF2
        
        print(f"    ⚡ Using lightweight PyPDF2 extraction...")
        
        elements = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    # Split into paragraphs for better chunking
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    for paragraph in paragraphs:
                        if len(paragraph) > 20:  # Only meaningful text
                            elements.append(SimpleTextElement(paragraph))
        
        print(f"Successfully extracted {len(elements)} text blocks using PyPDF2")
        return elements
        
    except ImportError:
        raise Exception("PyPDF2 not installed. Install with: pip install PyPDF2")
    except Exception as e:
        raise Exception(f"PyPDF2 parsing failed: {str(e)}")


def parse_pdf(pdf_path: str, enhanced_mode: bool = False, hi_res: bool = False) -> list:
    """
    Parse a PDF file with ultra-fast text extraction as default.
    Use flags for enhanced processing with better formatting.
    
    Args:
        pdf_path (str): Path to the PDF file
        enhanced_mode (bool): If True, use unstructured library with fast strategy
        hi_res (bool): If True, use unstructured library with hi-res strategy (slowest)
        
    Returns:
        list: List of text elements
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If PDF parsing fails
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    
    # Default: Ultra-fast PyPDF2 extraction
    if not enhanced_mode and not hi_res:
        print(f"    ⚡ Using ultra-fast text extraction (file size: {file_size_mb:.1f}MB)")
        print(f"    📝 Pure text mode - maximum speed, no formatting")
        return parse_pdf_lightweight(pdf_path)
    
    # Enhanced mode: Use unstructured library
    try:
        # Lazy load unstructured only when needed
        partition_pdf = _ensure_unstructured()
        
        if hi_res:
            strategy = "hi_res"
            print(f"    🔍 Using high-resolution parsing (file size: {file_size_mb:.1f}MB)")
            print(f"    📝 Best quality but slowest processing...")
        else:
            strategy = "fast"
            print(f"    ⚡ Using enhanced fast parsing (file size: {file_size_mb:.1f}MB)")
            print(f"    📝 Better formatting but slower than default...")
        
        print(f"Reading PDF for file: {pdf_path} ...")
        
        # Parse the PDF with unstructured library
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            
            # Speed optimizations - still disable image processing
            extract_images_in_pdf=False,           # Never extract images
            infer_table_structure=(strategy == "hi_res"),  # Only in hi-res mode
            include_page_breaks=False,             # Skip page breaks
            
            # Disable OCR completely (major speedup!)
            ocr_languages=None,                    # No OCR languages
            skip_infer_table_types=[],            # Skip table type inference
            
            # Additional speed settings
            model_name=None,                       # Don't use ML models
            hi_res_model_name=None if strategy == "fast" else "yolox",  # Only use in hi-res
        )
        
        print(f"Successfully parsed PDF: {len(elements)} text elements extracted")
        
        # Filter to only text-containing elements
        text_elements = [elem for elem in elements if hasattr(elem, 'text') and elem.text and elem.text.strip()]
        print(f"Filtered to {len(text_elements)} elements with meaningful text content")
        
        return text_elements
        
    except Exception as e:
        print(f"    ⚠️  Enhanced parsing failed, falling back to ultra-fast mode...")
        print(f"    📝 Error: {str(e)}")
        return parse_pdf_lightweight(pdf_path)


def get_pdf_info(pdf_path: str) -> dict:
    """
    Get basic information about a PDF file for optimization decisions.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: Information about the PDF file
    """
    try:
        file_size_bytes = os.path.getsize(pdf_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        return {
            'file_size_bytes': file_size_bytes,
            'file_size_mb': round(file_size_mb, 2),
            'recommended_fast_mode': file_size_mb > 10,
            'file_path': pdf_path
        }
    except Exception as e:
        return {
            'error': str(e),
            'file_path': pdf_path
        } 