"""
pdf_chunker.py

Splits PDF documents into readable text chunks for later annotation, summarization, or RAG ingestion.
"""

import fitz  # PyMuPDF
from pathlib import Path

# Parameters
input_pdf = Path("../raw_documents/sample_notice.pdf")
output_txt = Path("../instruction_data/pdf_chunks.txt")
chunk_size = 1000  # characters

# Read and extract text
doc = fitz.open(input_pdf)
text = "".join(page.get_text() for page in doc)

# Chunk and write
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

with open(output_txt, "w") as f:
    for idx, chunk in enumerate(chunks):
        f.write(f"--- Chunk {idx + 1} ---\n{chunk}\n\n")

print(f"Wrote {len(chunks)} chunks to {output_txt}")
