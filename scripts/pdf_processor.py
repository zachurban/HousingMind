"""
HousingMind PDF Processing System
=================================
Comprehensive document processing pipeline for affordable housing policy documents.
Designed for PIH notices, CFR regulations, handbooks, and PHA guidance.

Features:
- Intelligent chunking that preserves regulatory structure
- Metadata extraction (PIH numbers, CFR citations, effective dates)
- Hybrid retrieval (semantic + BM25)
- Citation tracking with page numbers
- Query reformulation for housing terminology
"""

import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

# PDF Processing
import fitz  # pymupdf

# Text Processing
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Vector/Embedding (assumes you have these installed)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Semantic search disabled.")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class DocumentType(Enum):
    PIH_NOTICE = "pih_notice"
    CFR_REGULATION = "cfr_regulation"
    HANDBOOK = "handbook"
    GUIDEBOOK = "guidebook"
    FAQ = "faq"
    ADMIN_PLAN = "admin_plan"
    STATE_GUIDANCE = "state_guidance"
    FORM = "form"
    UNKNOWN = "unknown"


class HousingProgram(Enum):
    HCV = "housing_choice_voucher"
    PUBLIC_HOUSING = "public_housing"
    RAD = "rental_assistance_demonstration"
    PBRA = "project_based_rental_assistance"
    HOPWA = "hopwa"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    """Comprehensive metadata for housing policy documents."""
    source_path: str
    document_id: str
    title: str = ""
    document_type: DocumentType = DocumentType.UNKNOWN
    program: HousingProgram = HousingProgram.UNKNOWN

    # Regulatory identifiers
    pih_notice_number: Optional[str] = None
    cfr_citations: List[str] = field(default_factory=list)

    # Temporal information
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    publication_date: Optional[str] = None
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None

    # Geographic scope
    region: Optional[str] = None  # e.g., "Region VIII", "Colorado"
    nationwide: bool = True

    # Processing metadata
    total_pages: int = 0
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_quality: float = 1.0  # 0-1 score for OCR quality

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['document_type'] = self.document_type.value
        d['program'] = self.program.value
        return d


@dataclass
class DocumentChunk:
    """A chunk of document content with full provenance."""
    chunk_id: str
    document_id: str
    content: str

    # Location in source
    page_numbers: List[int] = field(default_factory=list)
    section_header: Optional[str] = None
    subsection: Optional[str] = None

    # Chunk context
    chunk_index: int = 0
    total_chunks: int = 0

    # For retrieval
    embedding: Optional[List[float]] = None
    bm25_tokens: List[str] = field(default_factory=list)

    # Inherited metadata (denormalized for retrieval)
    metadata: Optional[DocumentMetadata] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['metadata'] = self.metadata.to_dict() if self.metadata else None
        # Don't serialize embeddings to JSON (too large)
        d.pop('embedding', None)
        return d


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval with full citation info."""
    chunk: DocumentChunk
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0

    def get_citation(self) -> str:
        """Generate a proper citation string."""
        meta = self.chunk.metadata
        parts = []

        if meta.pih_notice_number:
            parts.append(f"PIH {meta.pih_notice_number}")
        if meta.title:
            parts.append(meta.title)
        if self.chunk.page_numbers:
            pages = self.chunk.page_numbers
            if len(pages) == 1:
                parts.append(f"p. {pages[0]}")
            else:
                parts.append(f"pp. {min(pages)}-{max(pages)}")
        if meta.effective_date:
            parts.append(f"(effective {meta.effective_date})")

        return ", ".join(parts) if parts else meta.source_path


# =============================================================================
# PDF EXTRACTION
# =============================================================================

class HousingPDFExtractor:
    """
    Extracts text from housing policy PDFs with structure preservation.
    Uses PyMuPDF (fitz) for robust handling of government documents.
    """

    # Patterns for identifying document structure
    SECTION_PATTERNS = [
        r'^(\d+)\.\s+([A-Z][A-Za-z\s]+)',  # "1. Purpose"
        r'^([A-Z])\.\s+([A-Z][A-Za-z\s]+)',  # "A. Background"
        r'^Section\s+(\d+)[:\.]?\s*(.+)',  # "Section 5: Eligibility"
        r'^Chapter\s+(\d+)[:\.]?\s*(.+)',  # "Chapter 3: Income"
        r'^(\d+\.\d+)\s+([A-Z][A-Za-z\s]+)',  # "5.1 Income Calculation"
        r'^(?:PART|Part)\s+([IVX\d]+)[:\.]?\s*(.+)',  # "PART II: Procedures"
    ]

    # PIH Notice patterns
    PIH_PATTERNS = [
        r'PIH\s*[-–]?\s*(\d{4})\s*[-–]\s*(\d+)',  # PIH 2024-15, PIH-2024-15
        r'Notice\s+PIH\s*(\d{4})\s*[-–]\s*(\d+)',
        r'PIH\s*(\d{4})\s*[-–]\s*(\d+)\s*\([A-Z]+\)',  # PIH 2024-15 (HA)
    ]

    # CFR citation patterns
    CFR_PATTERNS = [
        r'24\s*C\.?F\.?R\.?\s*§?\s*(\d+)\.(\d+)',  # 24 CFR 982.201
        r'24\s*C\.?F\.?R\.?\s*[Pp]art\s*(\d+)',  # 24 CFR Part 982
        r'§\s*(\d+)\.(\d+)',  # §982.201 (when in CFR context)
    ]

    # Date patterns
    DATE_PATTERNS = [
        r'[Ee]ffective\s*[Dd]ate[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
        r'[Ee]ffective[:\s]+(\d{1,2}/\d{1,2}/\d{4})',
        r'[Ee]xpires?[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
    ]

    def __init__(self, ocr_fallback: bool = False):
        self.ocr_fallback = ocr_fallback

    def extract_document(self, pdf_path: Path) -> Tuple[str, DocumentMetadata, List[Dict]]:
        """
        Extract text and metadata from a PDF.

        Returns:
            Tuple of (full_text, metadata, page_data)
            where page_data is a list of {page_num, text, blocks} dicts
        """
        pdf_path = Path(pdf_path)

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            raise

        # Generate document ID
        doc_id = self._generate_doc_id(pdf_path)

        # Initialize metadata
        metadata = DocumentMetadata(
            source_path=str(pdf_path),
            document_id=doc_id,
            total_pages=len(doc)
        )

        full_text = ""
        page_data = []
        all_text_for_metadata = []

        for page_num, page in enumerate(doc, start=1):
            # Extract text with layout preservation
            text = page.get_text("text")

            # Get text blocks for structure analysis
            blocks = page.get_text("dict")["blocks"]

            # Check extraction quality (detect potential OCR issues)
            if not text.strip() and self.ocr_fallback:
                # Could integrate OCR here if needed
                logger.warning(f"Page {page_num} has no extractable text")
                metadata.extraction_quality *= 0.9

            page_data.append({
                'page_num': page_num,
                'text': text,
                'blocks': blocks
            })

            full_text += f"\n\n--- Page {page_num} ---\n\n{text}"
            all_text_for_metadata.append(text)

        doc.close()

        # Extract metadata from text
        combined_text = "\n".join(all_text_for_metadata[:5])  # First 5 pages for metadata
        self._extract_metadata(combined_text, metadata, pdf_path)

        return full_text, metadata, page_data

    def _generate_doc_id(self, pdf_path: Path) -> str:
        """Generate a unique document ID based on filename and content hash."""
        with open(pdf_path, 'rb') as f:
            content_hash = hashlib.md5(f.read()[:10000]).hexdigest()[:8]
        return f"{pdf_path.stem}_{content_hash}"

    def _extract_metadata(self, text: str, metadata: DocumentMetadata, pdf_path: Path):
        """Extract document metadata from text content."""

        # Extract PIH notice number
        for pattern in self.PIH_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                metadata.pih_notice_number = f"{groups[0]}-{groups[1]}"
                metadata.document_type = DocumentType.PIH_NOTICE
                break

        # Extract CFR citations
        for pattern in self.CFR_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    cfr = f"24 CFR {match[0]}.{match[1]}" if len(match) > 1 else f"24 CFR Part {match[0]}"
                else:
                    cfr = f"24 CFR {match}"
                if cfr not in metadata.cfr_citations:
                    metadata.cfr_citations.append(cfr)

        # Extract effective date
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                if 'effective' in pattern.lower():
                    metadata.effective_date = match.group(1)
                elif 'expire' in pattern.lower():
                    metadata.expiration_date = match.group(1)

        # Determine program type from content
        metadata.program = self._detect_program(text)

        # Determine document type if not already set
        if metadata.document_type == DocumentType.UNKNOWN:
            metadata.document_type = self._detect_document_type(text, pdf_path)

        # Extract title from first meaningful line
        metadata.title = self._extract_title(text, pdf_path)

    def _detect_program(self, text: str) -> HousingProgram:
        """Detect which housing program the document relates to."""
        text_lower = text.lower()

        programs_found = []

        if any(term in text_lower for term in ['housing choice voucher', 'section 8 voucher', 'hcv program', '24 cfr 982']):
            programs_found.append(HousingProgram.HCV)
        if any(term in text_lower for term in ['public housing', '24 cfr 960', '24 cfr 966', 'acc units']):
            programs_found.append(HousingProgram.PUBLIC_HOUSING)
        if any(term in text_lower for term in ['rental assistance demonstration', 'rad conversion', 'rad program']):
            programs_found.append(HousingProgram.RAD)
        if any(term in text_lower for term in ['project-based rental assistance', 'pbra', 'section 8 project']):
            programs_found.append(HousingProgram.PBRA)

        if len(programs_found) == 0:
            return HousingProgram.UNKNOWN
        elif len(programs_found) == 1:
            return programs_found[0]
        else:
            return HousingProgram.MIXED

    def _detect_document_type(self, text: str, pdf_path: Path) -> DocumentType:
        """Detect the type of document."""
        text_lower = text.lower()
        filename_lower = pdf_path.name.lower()

        if 'handbook' in filename_lower or 'handbook' in text_lower[:1000]:
            return DocumentType.HANDBOOK
        if 'guidebook' in filename_lower or 'guidebook' in text_lower[:1000]:
            return DocumentType.GUIDEBOOK
        if 'faq' in filename_lower or 'frequently asked' in text_lower[:1000]:
            return DocumentType.FAQ
        if 'administrative plan' in text_lower[:2000] or 'admin plan' in filename_lower:
            return DocumentType.ADMIN_PLAN
        if any(term in text_lower[:500] for term in ['24 cfr', 'code of federal regulations']):
            return DocumentType.CFR_REGULATION

        return DocumentType.UNKNOWN

    def _extract_title(self, text: str, pdf_path: Path) -> str:
        """Extract document title from text or filename."""
        lines = text.split('\n')

        # Look for title in first 20 non-empty lines
        for line in lines[:20]:
            line = line.strip()
            # Skip very short lines or lines that look like headers/footers
            if len(line) < 10 or len(line) > 200:
                continue
            if re.match(r'^(page\s*\d+|u\.s\.\s*department)', line, re.IGNORECASE):
                continue
            if re.match(r'^\d+$', line):  # Just a number
                continue
            # This might be the title
            return line

        # Fall back to filename
        return pdf_path.stem.replace('_', ' ').replace('-', ' ').title()


# =============================================================================
# INTELLIGENT CHUNKING
# =============================================================================

class HousingDocumentChunker:
    """
    Chunks housing policy documents while preserving regulatory structure.

    Key principles:
    - Never break numbered lists or regulatory subsections
    - Keep FAQ Q&As together
    - Preserve section headers in each chunk
    - Maintain paragraph boundaries where possible
    """

    def __init__(
        self,
        chunk_size: int = 1000,  # Target chunk size in tokens (approximate)
        chunk_overlap: int = 100,
        min_chunk_size: int = 200,
        max_chunk_size: int = 2000
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Patterns for detecting structure
        self.numbered_list_pattern = re.compile(
            r'^[\s]*(\d+\.|[a-z]\.|[ivx]+\.|•|[-–])\s+',
            re.MULTILINE
        )
        self.section_header_pattern = re.compile(
            r'^(?:(?:\d+\.)+\s*|[A-Z]\.\s*|Section\s+\d+[:\.]?\s*|Chapter\s+\d+[:\.]?\s*)([A-Z][A-Za-z\s]+)',
            re.MULTILINE
        )
        self.faq_pattern = re.compile(
            r'^(?:Q\d*[:.]?\s*|Question\s*\d*[:.]?\s*)(.+?)(?=\n(?:A\d*[:.]?\s*|Answer[:.]?\s*))',
            re.MULTILINE | re.DOTALL
        )

    def chunk_document(
        self,
        text: str,
        metadata: DocumentMetadata,
        page_data: List[Dict]
    ) -> List[DocumentChunk]:
        """
        Chunk a document while preserving structure.

        Returns list of DocumentChunk objects with full metadata.
        """
        # First, segment by major sections
        sections = self._segment_sections(text)

        chunks = []
        chunk_index = 0

        for section_header, section_content in sections:
            # Chunk within each section
            section_chunks = self._chunk_section(
                section_content,
                section_header,
                metadata,
                page_data
            )

            for chunk_text, page_nums in section_chunks:
                chunk_id = f"{metadata.document_id}_chunk_{chunk_index:04d}"

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=metadata.document_id,
                    content=chunk_text,
                    page_numbers=page_nums,
                    section_header=section_header,
                    chunk_index=chunk_index,
                    metadata=metadata,
                    bm25_tokens=self._tokenize_for_bm25(chunk_text)
                )

                chunks.append(chunk)
                chunk_index += 1

        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _segment_sections(self, text: str) -> List[Tuple[Optional[str], str]]:
        """Split document into major sections."""
        sections = []

        # Find all section headers
        header_matches = list(self.section_header_pattern.finditer(text))

        if not header_matches:
            # No clear sections, treat as one block
            return [(None, text)]

        # Extract sections
        for i, match in enumerate(header_matches):
            header = match.group(0).strip()
            start = match.end()
            end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(text)

            section_content = text[start:end].strip()
            if section_content:
                sections.append((header, section_content))

        # Don't forget content before first header
        if header_matches[0].start() > 100:
            preamble = text[:header_matches[0].start()].strip()
            if preamble:
                sections.insert(0, (None, preamble))

        return sections

    def _chunk_section(
        self,
        content: str,
        section_header: Optional[str],
        metadata: DocumentMetadata,
        page_data: List[Dict]
    ) -> List[Tuple[str, List[int]]]:
        """
        Chunk a section while preserving structural integrity.

        Returns list of (chunk_text, page_numbers) tuples.
        """
        # Check for special structures
        if self._is_numbered_list(content):
            return self._chunk_numbered_list(content, section_header, page_data)

        if metadata.document_type == DocumentType.FAQ or self._contains_faq(content):
            return self._chunk_faq(content, section_header, page_data)

        # Standard paragraph-aware chunking
        return self._chunk_paragraphs(content, section_header, page_data)

    def _is_numbered_list(self, content: str) -> bool:
        """Check if content is primarily a numbered list."""
        lines = content.split('\n')
        numbered_lines = sum(1 for line in lines if self.numbered_list_pattern.match(line))
        return numbered_lines > len(lines) * 0.3  # More than 30% numbered

    def _contains_faq(self, content: str) -> bool:
        """Check if content contains FAQ structure."""
        return bool(self.faq_pattern.search(content))

    def _chunk_numbered_list(
        self,
        content: str,
        section_header: Optional[str],
        page_data: List[Dict]
    ) -> List[Tuple[str, List[int]]]:
        """Chunk numbered list content, keeping list items together."""
        chunks = []
        current_chunk_parts = []
        current_size = 0

        # Split into list items
        items = self._split_list_items(content)

        for item in items:
            item_size = len(item.split())

            if current_size + item_size > self.chunk_size and current_chunk_parts:
                # Finish current chunk
                chunk_text = self._assemble_chunk(current_chunk_parts, section_header)
                page_nums = self._find_page_numbers(chunk_text, page_data)
                chunks.append((chunk_text, page_nums))

                # Start new chunk with overlap
                current_chunk_parts = current_chunk_parts[-1:] if current_chunk_parts else []
                current_size = sum(len(p.split()) for p in current_chunk_parts)

            current_chunk_parts.append(item)
            current_size += item_size

        # Don't forget the last chunk
        if current_chunk_parts:
            chunk_text = self._assemble_chunk(current_chunk_parts, section_header)
            page_nums = self._find_page_numbers(chunk_text, page_data)
            chunks.append((chunk_text, page_nums))

        return chunks

    def _split_list_items(self, content: str) -> List[str]:
        """Split content into individual list items."""
        items = []
        current_item = []

        for line in content.split('\n'):
            if self.numbered_list_pattern.match(line) and current_item:
                items.append('\n'.join(current_item))
                current_item = [line]
            else:
                current_item.append(line)

        if current_item:
            items.append('\n'.join(current_item))

        return items

    def _chunk_faq(
        self,
        content: str,
        section_header: Optional[str],
        page_data: List[Dict]
    ) -> List[Tuple[str, List[int]]]:
        """Chunk FAQ content, keeping Q&A pairs together."""
        chunks = []

        # Find Q&A pairs
        qa_pairs = self._extract_qa_pairs(content)

        current_chunk_parts = []
        current_size = 0

        for q, a in qa_pairs:
            qa_text = f"Q: {q}\nA: {a}"
            qa_size = len(qa_text.split())

            if current_size + qa_size > self.chunk_size and current_chunk_parts:
                chunk_text = self._assemble_chunk(current_chunk_parts, section_header)
                page_nums = self._find_page_numbers(chunk_text, page_data)
                chunks.append((chunk_text, page_nums))

                current_chunk_parts = []
                current_size = 0

            current_chunk_parts.append(qa_text)
            current_size += qa_size

        if current_chunk_parts:
            chunk_text = self._assemble_chunk(current_chunk_parts, section_header)
            page_nums = self._find_page_numbers(chunk_text, page_data)
            chunks.append((chunk_text, page_nums))

        return chunks

    def _extract_qa_pairs(self, content: str) -> List[Tuple[str, str]]:
        """Extract question-answer pairs from FAQ content."""
        pairs = []

        # Simple pattern matching for Q&A
        pattern = re.compile(
            r'(?:Q\d*[:.]?\s*|Question\s*\d*[:.]?\s*)(.+?)\n\s*(?:A\d*[:.]?\s*|Answer[:.]?\s*)(.+?)(?=(?:\n\s*Q\d*[:.]|\n\s*Question|$))',
            re.DOTALL | re.IGNORECASE
        )

        for match in pattern.finditer(content):
            q = match.group(1).strip()
            a = match.group(2).strip()
            pairs.append((q, a))

        # If no pairs found, fall back to treating as regular content
        if not pairs:
            return [(content, "")]

        return pairs

    def _chunk_paragraphs(
        self,
        content: str,
        section_header: Optional[str],
        page_data: List[Dict]
    ) -> List[Tuple[str, List[int]]]:
        """Standard paragraph-based chunking with sentence boundaries."""
        chunks = []

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)

        current_chunk_parts = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para.split())

            # If single paragraph exceeds max, split by sentences
            if para_size > self.max_chunk_size:
                if current_chunk_parts:
                    chunk_text = self._assemble_chunk(current_chunk_parts, section_header)
                    page_nums = self._find_page_numbers(chunk_text, page_data)
                    chunks.append((chunk_text, page_nums))
                    current_chunk_parts = []
                    current_size = 0

                # Split large paragraph
                sentence_chunks = self._chunk_by_sentences(para, section_header, page_data)
                chunks.extend(sentence_chunks)
                continue

            if current_size + para_size > self.chunk_size and current_chunk_parts:
                chunk_text = self._assemble_chunk(current_chunk_parts, section_header)
                page_nums = self._find_page_numbers(chunk_text, page_data)
                chunks.append((chunk_text, page_nums))

                # Overlap: keep last paragraph if small enough
                if len(current_chunk_parts[-1].split()) < self.chunk_overlap:
                    current_chunk_parts = [current_chunk_parts[-1]]
                    current_size = len(current_chunk_parts[0].split())
                else:
                    current_chunk_parts = []
                    current_size = 0

            current_chunk_parts.append(para)
            current_size += para_size

        if current_chunk_parts:
            chunk_text = self._assemble_chunk(current_chunk_parts, section_header)
            page_nums = self._find_page_numbers(chunk_text, page_data)
            chunks.append((chunk_text, page_nums))

        return chunks

    def _chunk_by_sentences(
        self,
        text: str,
        section_header: Optional[str],
        page_data: List[Dict]
    ) -> List[Tuple[str, List[int]]]:
        """Split large text block by sentence boundaries."""
        chunks = []
        sentences = sent_tokenize(text)

        current_sentences = []
        current_size = 0

        for sent in sentences:
            sent_size = len(sent.split())

            if current_size + sent_size > self.chunk_size and current_sentences:
                chunk_text = self._assemble_chunk([' '.join(current_sentences)], section_header)
                page_nums = self._find_page_numbers(chunk_text, page_data)
                chunks.append((chunk_text, page_nums))

                # Overlap
                overlap_sents = []
                overlap_size = 0
                for s in reversed(current_sentences):
                    if overlap_size + len(s.split()) <= self.chunk_overlap:
                        overlap_sents.insert(0, s)
                        overlap_size += len(s.split())
                    else:
                        break

                current_sentences = overlap_sents
                current_size = overlap_size

            current_sentences.append(sent)
            current_size += sent_size

        if current_sentences:
            chunk_text = self._assemble_chunk([' '.join(current_sentences)], section_header)
            page_nums = self._find_page_numbers(chunk_text, page_data)
            chunks.append((chunk_text, page_nums))

        return chunks

    def _assemble_chunk(self, parts: List[str], section_header: Optional[str]) -> str:
        """Assemble chunk parts with section header context."""
        content = '\n\n'.join(parts)

        if section_header:
            return f"[Section: {section_header}]\n\n{content}"
        return content

    def _find_page_numbers(self, chunk_text: str, page_data: List[Dict]) -> List[int]:
        """Find which pages contain this chunk's content."""
        pages = []

        # Use first 100 chars of chunk for matching
        search_text = chunk_text[:100].lower()

        for page in page_data:
            if search_text in page['text'].lower():
                pages.append(page['page_num'])

        # If no match found, try broader matching
        if not pages:
            words = search_text.split()[:5]
            for page in page_data:
                if all(w in page['text'].lower() for w in words):
                    pages.append(page['page_num'])

        return sorted(set(pages)) if pages else [1]

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        # Lowercase and tokenize
        tokens = word_tokenize(text.lower())
        # Remove very short tokens and punctuation
        tokens = [t for t in tokens if len(t) > 2 and t.isalnum()]
        return tokens


# =============================================================================
# QUERY REFORMULATION
# =============================================================================

class HousingQueryReformulator:
    """
    Reformulates queries using housing terminology synonyms and expansions.

    PHA staff often use operational language that differs from regulatory language.
    This bridges that gap.
    """

    # Housing terminology mappings
    SYNONYMS = {
        # Program terms
        'section 8': ['housing choice voucher', 'hcv', 'voucher program'],
        'hcv': ['housing choice voucher', 'section 8', 'voucher'],
        'voucher': ['housing choice voucher', 'hcv', 'section 8'],
        'public housing': ['low rent housing', 'conventional public housing', 'acc'],

        # Process terms
        'terminate': ['end participation', 'termination', 'discontinue assistance'],
        'termination': ['end of participation', 'terminate', 'discontinue'],
        'denial': ['deny', 'rejection', 'ineligible'],
        'port': ['portability', 'transfer', 'move'],
        'portability': ['port', 'transfer', 'billing arrangement'],

        # Income terms
        'income': ['annual income', 'gross income', 'adjusted income'],
        'rent': ['tenant rent', 'total tenant payment', 'ttp', 'family share'],
        'ttp': ['total tenant payment', 'tenant rent', 'family share'],
        'utility allowance': ['ua', 'utility reimbursement'],

        # Eligibility terms
        'criminal': ['criminal activity', 'criminal background', 'criminal history'],
        'drugs': ['drug-related', 'illegal drug', 'controlled substance'],
        'eviction': ['evict', 'lease termination', 'removal'],

        # HQS terms
        'inspection': ['hqs inspection', 'unit inspection', 'housing quality'],
        'hqs': ['housing quality standards', 'inspection standards'],
        'fail': ['failed inspection', 'deficiency', 'violation'],
        'abatement': ['rent abatement', 'hap abatement', 'payment suspension'],

        # VAWA terms
        'vawa': ['violence against women', 'domestic violence', 'dating violence'],
        'domestic violence': ['vawa', 'dv', 'family violence'],

        # Other common terms
        'briefing': ['voucher briefing', 'initial briefing', 'orientation'],
        'recertification': ['reexamination', 'annual review', 'recert'],
        'interim': ['interim reexamination', 'change in circumstances'],
        'repayment agreement': ['repayment', 'overpayment', 'recovery'],
    }

    # CFR section mappings for common topics
    TOPIC_TO_CFR = {
        'eligibility': ['24 CFR 982.201', '24 CFR 982.202', '24 CFR 982.204'],
        'income': ['24 CFR 5.609', '24 CFR 5.611', '24 CFR 982.516'],
        'termination': ['24 CFR 982.552', '24 CFR 982.553', '24 CFR 982.555'],
        'hqs': ['24 CFR 982.401', '24 CFR 982.404', '24 CFR 982.405'],
        'portability': ['24 CFR 982.353', '24 CFR 982.354', '24 CFR 982.355'],
        'rent': ['24 CFR 982.503', '24 CFR 982.505', '24 CFR 982.507'],
        'owner': ['24 CFR 982.306', '24 CFR 982.307', '24 CFR 982.308'],
        'vawa': ['24 CFR 5.2001', '24 CFR 5.2005', '24 CFR 5.2009'],
        'reasonable accommodation': ['24 CFR 8.33', '24 CFR 100.204'],
    }

    def reformulate(self, query: str) -> Dict[str, Any]:
        """
        Reformulate a query for better retrieval.

        Returns:
            Dict with 'original', 'expanded', 'synonyms_added', 'cfr_hints'
        """
        query_lower = query.lower()

        # Find synonyms to add
        synonyms_to_add = set()
        for term, synonyms in self.SYNONYMS.items():
            if term in query_lower:
                synonyms_to_add.update(synonyms)

        # Find relevant CFR sections
        cfr_hints = []
        for topic, cfrs in self.TOPIC_TO_CFR.items():
            if topic in query_lower:
                cfr_hints.extend(cfrs)

        # Build expanded query
        expanded_parts = [query]
        if synonyms_to_add:
            expanded_parts.extend(list(synonyms_to_add)[:5])  # Limit expansion

        expanded_query = ' '.join(expanded_parts)

        return {
            'original': query,
            'expanded': expanded_query,
            'synonyms_added': list(synonyms_to_add),
            'cfr_hints': cfr_hints
        }


# =============================================================================
# HYBRID RETRIEVAL
# =============================================================================

class HybridRetriever:
    """
    Combines semantic search with BM25 for optimal housing policy retrieval.

    Housing queries often need both:
    - Semantic understanding ("Can I deny someone for past eviction?")
    - Exact matches ("24 CFR 982.552(c)(1)(i)")
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = None

        # Will be populated when documents are indexed
        self.chunks: List[DocumentChunk] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.embeddings: Optional[np.ndarray] = None

        # Query reformulator
        self.reformulator = HousingQueryReformulator()

    def index_documents(self, chunks: List[DocumentChunk]):
        """Index chunks for hybrid retrieval."""
        self.chunks = chunks

        # Build BM25 index
        corpus = [chunk.bm25_tokens for chunk in chunks]
        self.bm25_index = BM25Okapi(corpus)

        # Build embedding index
        if self.embedding_model:
            texts = [chunk.content for chunk in chunks]
            self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

            # Store embeddings in chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = self.embeddings[i].tolist()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_reformulation: bool = True
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search with optional metadata filtering.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {'program': 'hcv', 'document_type': 'pih_notice'})
            use_reformulation: Whether to expand query with synonyms
        """
        if not self.chunks:
            raise ValueError("No documents indexed. Call index_documents first.")

        # Reformulate query
        if use_reformulation:
            reformulated = self.reformulator.reformulate(query)
            search_query = reformulated['expanded']
        else:
            search_query = query

        # Apply metadata filters first
        filtered_indices = self._apply_filters(filters) if filters else list(range(len(self.chunks)))

        if not filtered_indices:
            return []

        # Get BM25 scores
        query_tokens = word_tokenize(search_query.lower())
        query_tokens = [t for t in query_tokens if len(t) > 2 and t.isalnum()]

        bm25_scores = np.array(self.bm25_index.get_scores(query_tokens))

        # Get semantic scores
        if self.embedding_model and self.embeddings is not None:
            query_embedding = self.embedding_model.encode([search_query])[0]
            semantic_scores = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
            )
        else:
            semantic_scores = np.zeros(len(self.chunks))

        # Normalize scores
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        if semantic_scores.max() > 0:
            semantic_scores = semantic_scores / semantic_scores.max()

        # Combine scores (only for filtered indices)
        results = []
        for idx in filtered_indices:
            combined = (
                self.semantic_weight * semantic_scores[idx] +
                self.bm25_weight * bm25_scores[idx]
            )

            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                semantic_score=float(semantic_scores[idx]),
                bm25_score=float(bm25_scores[idx]),
                combined_score=float(combined)
            ))

        # Sort by combined score and return top_k
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]

    def _apply_filters(self, filters: Dict) -> List[int]:
        """Apply metadata filters to chunks."""
        valid_indices = []

        for i, chunk in enumerate(self.chunks):
            if not chunk.metadata:
                continue

            matches = True
            for key, value in filters.items():
                if key == 'program':
                    if chunk.metadata.program.value != value and chunk.metadata.program != HousingProgram.MIXED:
                        matches = False
                        break
                elif key == 'document_type':
                    if chunk.metadata.document_type.value != value:
                        matches = False
                        break
                elif key == 'region':
                    if chunk.metadata.region != value and chunk.metadata.nationwide:
                        matches = False
                        break
                elif key == 'current_only':
                    if value and chunk.metadata.superseded_by:
                        matches = False
                        break

            if matches:
                valid_indices.append(i)

        return valid_indices


# =============================================================================
# RESPONSE VALIDATION
# =============================================================================

class ResponseValidator:
    """
    Validates AI responses against retrieved chunks.
    Ensures citations are accurate and flags uncertainty.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def validate_response(
        self,
        response: str,
        retrieved_chunks: List[DocumentChunk],
        query: str
    ) -> Dict[str, Any]:
        """
        Validate an AI-generated response against source chunks.

        Returns:
            Dict with validation results and suggested citations
        """
        validation = {
            'is_supported': True,
            'confidence': 1.0,
            'citations': [],
            'warnings': [],
            'suggested_citations': []
        }

        # Check if response contains claims not in chunks
        response_sentences = sent_tokenize(response)
        chunk_texts = ' '.join([c.content.lower() for c in retrieved_chunks])

        unsupported_claims = []
        for sent in response_sentences:
            # Skip meta sentences
            if any(phrase in sent.lower() for phrase in [
                'based on', 'according to', 'the regulation states',
                'i recommend', 'you should consult', 'please note'
            ]):
                continue

            # Check if key terms appear in chunks
            key_terms = [w for w in word_tokenize(sent.lower()) if len(w) > 4]
            if key_terms:
                term_coverage = sum(1 for t in key_terms if t in chunk_texts) / len(key_terms)
                if term_coverage < 0.3:
                    unsupported_claims.append(sent)

        if unsupported_claims:
            validation['warnings'].append(
                f"Potentially unsupported claims detected: {len(unsupported_claims)} sentences"
            )
            validation['confidence'] *= 0.8

        # Generate suggested citations for each chunk used
        for chunk in retrieved_chunks:
            citation = self._generate_citation(chunk)
            validation['suggested_citations'].append(citation)

        validation['is_supported'] = validation['confidence'] >= self.confidence_threshold

        return validation

    def _generate_citation(self, chunk: DocumentChunk) -> str:
        """Generate a formatted citation for a chunk."""
        meta = chunk.metadata
        parts = []

        if meta.pih_notice_number:
            parts.append(f"PIH Notice {meta.pih_notice_number}")
        elif meta.title:
            parts.append(meta.title)

        if chunk.section_header:
            parts.append(f"Section: {chunk.section_header}")

        if chunk.page_numbers:
            if len(chunk.page_numbers) == 1:
                parts.append(f"p. {chunk.page_numbers[0]}")
            else:
                parts.append(f"pp. {min(chunk.page_numbers)}-{max(chunk.page_numbers)}")

        if meta.effective_date:
            parts.append(f"(effective {meta.effective_date})")

        return ", ".join(parts)


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

class HousingMindProcessor:
    """
    Main pipeline for processing housing policy documents.
    Combines extraction, chunking, and indexing.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        chunk_size: int = 1000
    ):
        self.extractor = HousingPDFExtractor()
        self.chunker = HousingDocumentChunker(chunk_size=chunk_size)
        self.retriever = HybridRetriever(embedding_model=embedding_model)
        self.validator = ResponseValidator()

        self.processed_documents: Dict[str, DocumentMetadata] = {}

    def process_pdf(self, pdf_path: Path) -> List[DocumentChunk]:
        """Process a single PDF and return chunks."""
        logger.info(f"Processing: {pdf_path}")

        # Extract
        text, metadata, page_data = self.extractor.extract_document(pdf_path)

        # Chunk
        chunks = self.chunker.chunk_document(text, metadata, page_data)

        # Store metadata
        self.processed_documents[metadata.document_id] = metadata

        logger.info(f"  -> {len(chunks)} chunks, type: {metadata.document_type.value}")

        return chunks

    def process_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> List[DocumentChunk]:
        """Process all PDFs in a directory."""
        directory = Path(directory)

        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files")

        all_chunks = []
        for pdf_path in pdf_files:
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue

        return all_chunks

    def build_index(self, chunks: List[DocumentChunk]):
        """Build the hybrid search index."""
        logger.info(f"Building index for {len(chunks)} chunks...")
        self.retriever.index_documents(chunks)
        logger.info("Index built successfully")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Search the indexed documents."""
        return self.retriever.search(query, top_k=top_k, filters=filters)

    def save_index(self, output_path: Path):
        """Save processed chunks and metadata to disk."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save chunks (without embeddings for JSON)
        chunks_data = [chunk.to_dict() for chunk in self.retriever.chunks]
        with open(output_path / "chunks.json", 'w') as f:
            json.dump(chunks_data, f, indent=2)

        # Save embeddings separately as numpy
        if self.retriever.embeddings is not None:
            np.save(output_path / "embeddings.npy", self.retriever.embeddings)

        # Save document metadata
        meta_data = {k: v.to_dict() for k, v in self.processed_documents.items()}
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(meta_data, f, indent=2)

        logger.info(f"Index saved to {output_path}")

    def load_index(self, index_path: Path):
        """Load a previously saved index."""
        index_path = Path(index_path)

        # Load chunks
        with open(index_path / "chunks.json", 'r') as f:
            chunks_data = json.load(f)

        # Reconstruct chunks
        chunks = []
        for cd in chunks_data:
            meta_dict = cd.pop('metadata', None)
            if meta_dict:
                meta_dict['document_type'] = DocumentType(meta_dict['document_type'])
                meta_dict['program'] = HousingProgram(meta_dict['program'])
                metadata = DocumentMetadata(**meta_dict)
            else:
                metadata = None

            chunk = DocumentChunk(**cd)
            chunk.metadata = metadata
            chunks.append(chunk)

        self.retriever.chunks = chunks

        # Load embeddings
        embeddings_path = index_path / "embeddings.npy"
        if embeddings_path.exists():
            self.retriever.embeddings = np.load(embeddings_path)

        # Rebuild BM25 index
        corpus = [chunk.bm25_tokens for chunk in chunks]
        self.retriever.bm25_index = BM25Okapi(corpus)

        # Load metadata
        with open(index_path / "metadata.json", 'r') as f:
            meta_data = json.load(f)

        for doc_id, meta_dict in meta_data.items():
            meta_dict['document_type'] = DocumentType(meta_dict['document_type'])
            meta_dict['program'] = HousingProgram(meta_dict['program'])
            self.processed_documents[doc_id] = DocumentMetadata(**meta_dict)

        logger.info(f"Loaded index with {len(chunks)} chunks")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """Example usage of the HousingMind processor."""

    # Initialize processor
    processor = HousingMindProcessor(
        embedding_model="BAAI/bge-large-en-v1.5",
        chunk_size=800  # Slightly smaller for better precision
    )

    # Example: Process a directory of PDFs
    # chunks = processor.process_directory(Path("./housing_docs"))

    # Example: Process a single PDF
    # chunks = processor.process_pdf(Path("./pih_2024-15.pdf"))

    # Build the index
    # processor.build_index(chunks)

    # Save for later use
    # processor.save_index(Path("./housingmind_index"))

    # Example search
    # results = processor.search(
    #     "Can we terminate for drug-related criminal activity?",
    #     top_k=5,
    #     filters={'program': 'housing_choice_voucher'}
    # )

    # for result in results:
    #     print(f"\n{'='*60}")
    #     print(f"Score: {result.combined_score:.3f} (Semantic: {result.semantic_score:.3f}, BM25: {result.bm25_score:.3f})")
    #     print(f"Citation: {result.get_citation()}")
    #     print(f"Content preview: {result.chunk.content[:300]}...")

    print("HousingMind PDF Processor initialized successfully.")
    print("\nExample usage:")
    print("  processor = HousingMindProcessor()")
    print("  chunks = processor.process_directory(Path('./housing_docs'))")
    print("  processor.build_index(chunks)")
    print("  results = processor.search('Can I terminate for criminal activity?')")


if __name__ == "__main__":
    main()
