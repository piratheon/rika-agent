# skill: pdf
description: Read and extract text from PDF files, locally or via URL.
tools: run_python, run_shell_command, write_file

## When to use
- Summarising or analysing PDF documents
- Extracting structured data from scanned or text PDFs
- Converting PDF content to plain text for further processing

## Usage pattern
```python
# Extract text from a local PDF (requires pymupdf or pdfplumber)
run_python(code="""
import pathlib
try:
    import fitz  # pymupdf
    doc = fitz.open('document.pdf')
    text = '\n'.join(page.get_text() for page in doc)
except ImportError:
    import pdfplumber
    with pdfplumber.open('document.pdf') as pdf:
        text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
print(text[:5000])  # first 5000 chars for preview
""")

# Download a PDF from URL first
run_shell_command("curl -L -o document.pdf 'https://example.com/paper.pdf'")
```

## Notes
- pymupdf (fitz) is faster; pdfplumber handles complex layouts better
- Install: `pip install pymupdf pdfplumber`
- For scanned PDFs (images only), OCR is needed: `pip install pytesseract`
