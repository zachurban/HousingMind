"""
HousingMind Document Processor

Processes housing policy documents (PDFs, text files) into chunks
suitable for RAG retrieval with rich metadata extraction.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class HousingDocProcessor:
    """
    Processes housing policy documents into chunks with metadata.

    Handles various document types including:
    - PIH Notices
    - HUD Handbooks
    - CFR Regulations
    - LIHTC Documents
    - HUD Forms
    """

    # Topic keywords for automatic classification
    TOPIC_KEYWORDS = {
        "rad": "RAD",
        "rental assistance demonstration": "RAD",
        "voucher": "HCV",
        "hcv": "HCV",
        "housing choice": "HCV",
        "section 8": "HCV",
        "public housing": "Public Housing",
        "pha": "Public Housing",
        "lihtc": "LIHTC",
        "low income housing tax credit": "LIHTC",
        "tax credit": "LIHTC",
        "inspection": "HQS/NSPIRE",
        "hqs": "HQS/NSPIRE",
        "nspire": "HQS/NSPIRE",
        "housing quality": "HQS/NSPIRE",
        "fair market rent": "FMR",
        "fmr": "FMR",
        "income limit": "Income Limits",
        "ami": "Income Limits",
        "area median income": "Income Limits",
        "mtw": "MTW",
        "moving to work": "MTW",
        "pbv": "PBV",
        "project-based": "PBV",
        "tenant protection": "Tenant Rights",
        "tenant rights": "Tenant Rights",
        "relocation": "Relocation",
        "portability": "Portability",
    }

    # Document type patterns
    DOC_TYPE_PATTERNS = {
        r"PIH[\s\-_]?\d{4}[\s\-_]?\d+": "PIH Notice",
        r"notice": "Notice",
        r"handbook": "Handbook",
        r"guidebook": "Guidebook",
        r"\d+\s*CFR": "Regulation",
        r"cfr": "Regulation",
        r"form[\s\-_]": "Form",
        r"hud[\s\-_]\d+": "Form",
        r"qap": "QAP",
        r"qualified allocation plan": "QAP",
        r"compliance": "Compliance Manual",
        r"lihtc": "LIHTC Document",
    }

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        )
        self.processed_count = 0
        self.error_count = 0
        self.errors = []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF while preserving structure.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        try:
            reader = pypdf.PdfReader(pdf_path)
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Add page marker for reference
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

            return "\n\n".join(text_parts)
        except Exception as e:
            self.errors.append(f"PDF extraction error for {pdf_path}: {str(e)}")
            return ""

    def extract_metadata(self, filename: str, content: str = "") -> Dict:
        """
        Extract metadata from filename and content.

        Args:
            filename: Name of the source file
            content: Text content of the document (optional)

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {
            "source": filename,
            "document_type": "Unknown",
            "year": None,
            "topic": None,
            "notice_number": None,
            "cfr_citation": None,
        }

        filename_lower = filename.lower()
        content_lower = content.lower()[:5000] if content else ""  # Check first 5000 chars
        search_text = f"{filename_lower} {content_lower}"

        # Extract document type
        for pattern, doc_type in self.DOC_TYPE_PATTERNS.items():
            if re.search(pattern, search_text, re.IGNORECASE):
                metadata["document_type"] = doc_type
                break

        # Extract year (look for 4-digit year between 1990-2030)
        year_match = re.search(r'(19|20)\d{2}', filename)
        if year_match:
            year = int(year_match.group())
            if 1990 <= year <= 2030:
                metadata["year"] = str(year)

        # Extract PIH Notice number
        pih_match = re.search(r'PIH[\s\-_]?(\d{4})[\s\-_]?(\d+)', search_text, re.IGNORECASE)
        if pih_match:
            metadata["notice_number"] = f"PIH {pih_match.group(1)}-{pih_match.group(2)}"
            metadata["document_type"] = "PIH Notice"

        # Extract CFR citation
        cfr_match = re.search(r'(\d+)\s*CFR\s*([\d\.]+)', search_text, re.IGNORECASE)
        if cfr_match:
            metadata["cfr_citation"] = f"{cfr_match.group(1)} CFR {cfr_match.group(2)}"

        # Extract topic
        for keyword, topic in self.TOPIC_KEYWORDS.items():
            if keyword in search_text:
                metadata["topic"] = topic
                break

        # Determine source directory for additional context
        if "lihtc" in filename_lower or "/lihtc/" in filename_lower:
            metadata["topic"] = metadata["topic"] or "LIHTC"
        elif "pih" in filename_lower:
            metadata["topic"] = metadata["topic"] or "HUD Policy"
        elif "form" in filename_lower:
            metadata["document_type"] = "Form"

        return metadata

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove common PDF artifacts
        text = re.sub(r'\x00', '', text)  # Null bytes
        text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f]', '', text)  # Control chars

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        return text.strip()

    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a single document into chunks with metadata.

        Args:
            file_path: Path to the document file

        Returns:
            List of Document objects with content and metadata
        """
        file_path = str(file_path)
        filename = os.path.basename(file_path)

        # Extract text based on file type
        if file_path.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.txt', '.md')):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except Exception as e:
                self.errors.append(f"Text file read error for {file_path}: {str(e)}")
                return []
        else:
            return []

        if not text or len(text.strip()) < 50:
            return []

        # Clean text
        text = self.clean_text(text)

        # Extract metadata
        metadata = self.extract_metadata(filename, text)
        metadata["file_path"] = file_path

        # Create chunks
        chunks = self.text_splitter.split_text(text)

        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20:  # Skip very small chunks
                continue

            doc_metadata = metadata.copy()
            doc_metadata["chunk_id"] = i
            doc_metadata["total_chunks"] = len(chunks)

            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))

        self.processed_count += 1
        return documents

    def process_directory(self, directory: str,
                         extensions: List[str] = None,
                         max_files: Optional[int] = None) -> List[Document]:
        """
        Process all documents in a directory recursively.

        Args:
            directory: Path to directory to process
            extensions: List of file extensions to process (default: pdf, txt, md)
            max_files: Maximum number of files to process (for testing)

        Returns:
            List of all Document objects from the directory
        """
        if extensions is None:
            extensions = ['.pdf', '.txt', '.md']

        all_documents = []
        files_processed = 0

        directory_path = Path(directory)
        if not directory_path.exists():
            print(f"Error: Directory not found: {directory}")
            return []

        # Collect all matching files
        all_files = []
        for ext in extensions:
            all_files.extend(directory_path.glob(f"**/*{ext}"))

        total_files = len(all_files)
        if max_files:
            all_files = all_files[:max_files]

        print(f"Found {total_files} documents to process")
        if max_files:
            print(f"Processing first {max_files} files")

        for file_path in all_files:
            try:
                docs = self.process_document(str(file_path))
                all_documents.extend(docs)
                files_processed += 1

                if files_processed % 50 == 0:
                    print(f"Processed {files_processed}/{len(all_files)} files, "
                          f"{len(all_documents)} chunks generated")

            except Exception as e:
                self.error_count += 1
                self.errors.append(f"Error processing {file_path}: {str(e)}")

        print(f"\nProcessing complete!")
        print(f"Files processed: {files_processed}")
        print(f"Total chunks generated: {len(all_documents)}")
        if self.errors:
            print(f"Errors encountered: {len(self.errors)}")

        return all_documents

    def get_processing_stats(self) -> Dict:
        """Get statistics about processing."""
        return {
            "files_processed": self.processed_count,
            "errors": self.error_count,
            "error_details": self.errors
        }


def main():
    """Example usage of the document processor."""
    import argparse

    parser = argparse.ArgumentParser(description="Process housing documents for RAG")
    parser.add_argument("--input-dir", "-i",
                       default="../raw_documents",
                       help="Directory containing documents to process")
    parser.add_argument("--chunk-size", "-c",
                       type=int, default=800,
                       help="Chunk size in characters")
    parser.add_argument("--overlap", "-o",
                       type=int, default=200,
                       help="Chunk overlap in characters")
    parser.add_argument("--max-files", "-m",
                       type=int, default=None,
                       help="Maximum files to process (for testing)")

    args = parser.parse_args()

    processor = HousingDocProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap
    )

    documents = processor.process_directory(
        args.input_dir,
        max_files=args.max_files
    )

    # Print sample
    if documents:
        print("\nSample document chunk:")
        print("-" * 50)
        sample = documents[0]
        print(f"Content: {sample.page_content[:200]}...")
        print(f"Metadata: {sample.metadata}")


if __name__ == "__main__":
    main()
