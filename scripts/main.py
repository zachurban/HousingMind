#!/usr/bin/env python3
"""
HousingMind RAG System - Main Entry Point

This script provides the complete pipeline for setting up and running
the HousingMind RAG (Retrieval-Augmented Generation) system.

Uses:
- Llama 3.1 8B Instruct via Ollama for generation
- BAAI/bge-large-en-v1.5 for embeddings
- ChromaDB for vector storage

Usage:
    # Full setup (process documents + build database + run queries)
    python main.py --setup

    # Process documents only
    python main.py --process-docs

    # Build vector database from processed documents
    python main.py --build-db

    # Run interactive query mode
    python main.py --query

    # Run a single query
    python main.py --query "What are tenant rights under RAD?"
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from process_documents import HousingDocProcessor
from setup_vector_db import HousingMindVectorDB, DEFAULT_EMBEDDING_MODEL
from rag_engine import HousingMindRAG, DEFAULT_MODEL


# Default paths (relative to script location)
DEFAULT_RAW_DOCS = SCRIPT_DIR.parent / "raw_documents"
DEFAULT_VECTOR_DB = SCRIPT_DIR.parent / "vector_db"


def setup_housingmind(raw_docs_path: str = None,
                      vector_db_path: str = None,
                      chunk_size: int = 800,
                      chunk_overlap: int = 200,
                      max_files: int = None,
                      device: str = "cpu") -> HousingMindVectorDB:
    """
    Complete setup pipeline: process documents and build vector database.

    Args:
        raw_docs_path: Path to raw documents directory
        vector_db_path: Path for vector database storage
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        max_files: Maximum files to process (for testing)
        device: Device for embeddings ('cpu', 'cuda', 'mps')

    Returns:
        Initialized vector database
    """
    raw_docs_path = raw_docs_path or str(DEFAULT_RAW_DOCS)
    vector_db_path = vector_db_path or str(DEFAULT_VECTOR_DB)

    print("=" * 60)
    print("  HousingMind RAG System Setup")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Raw documents: {raw_docs_path}")
    print(f"  Vector DB path: {vector_db_path}")
    print(f"  Embedding model: {DEFAULT_EMBEDDING_MODEL}")
    print(f"  Device: {device}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Chunk overlap: {chunk_overlap}")
    if max_files:
        print(f"  Max files: {max_files}")
    print()

    # Step 1: Process documents
    print("Step 1: Processing documents...")
    print("-" * 40)

    processor = HousingDocProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    documents = processor.process_directory(
        raw_docs_path,
        max_files=max_files
    )

    if not documents:
        print("\nNo documents processed. Please check the raw_documents directory.")
        return None

    stats = processor.get_processing_stats()
    print(f"\nProcessing complete:")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Total chunks: {len(documents)}")
    if stats['errors']:
        print(f"  Errors: {stats['errors']}")

    # Step 2: Setup vector database
    print("\n" + "-" * 40)
    print("Step 2: Setting up vector database...")
    print("-" * 40)

    vector_db = HousingMindVectorDB(
        persist_directory=vector_db_path,
        collection_name="housing_docs",
        device=device
    )

    # Step 3: Add documents to database
    print("\n" + "-" * 40)
    print("Step 3: Adding documents to vector database...")
    print("-" * 40)

    vector_db.add_documents(documents)

    # Final stats
    db_stats = vector_db.get_collection_stats()
    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print(f"\nDatabase Statistics:")
    print(f"  Total documents: {db_stats['total_documents']}")
    print(f"  Collection: {db_stats['collection_name']}")
    print(f"  Location: {db_stats['persist_directory']}")
    print(f"  Embedding model: {db_stats['embedding_model']}")
    print(f"\nDocument types (sample):")
    for doc_type, count in db_stats.get('document_types_sample', {}).items():
        print(f"    {doc_type}: {count}")
    print(f"\nTopics (sample):")
    for topic, count in db_stats.get('topics_sample', {}).items():
        print(f"    {topic}: {count}")

    print("\n" + "=" * 60)
    print("  Next Steps:")
    print("=" * 60)
    print("  1. Make sure Ollama is running:")
    print("     ollama serve")
    print(f"  2. Pull the model (if not already):")
    print(f"     ollama pull {DEFAULT_MODEL}")
    print("  3. Run queries:")
    print("     python main.py --query")

    return vector_db


def process_documents_only(raw_docs_path: str = None,
                          chunk_size: int = 800,
                          chunk_overlap: int = 200,
                          max_files: int = None):
    """Process documents and display statistics without building database."""
    raw_docs_path = raw_docs_path or str(DEFAULT_RAW_DOCS)

    print("Processing documents...")
    processor = HousingDocProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    documents = processor.process_directory(
        raw_docs_path,
        max_files=max_files
    )

    if documents:
        print("\nSample chunks:")
        for i, doc in enumerate(documents[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"Type: {doc.metadata.get('document_type')}")
            print(f"Topic: {doc.metadata.get('topic')}")
            print(f"Content: {doc.page_content[:200]}...")

    return documents


def build_database_only(documents_path: str = None,
                        vector_db_path: str = None):
    """Build vector database from existing documents (requires prior processing)."""
    # This would need documents to be serialized first
    print("Note: For fresh setup, use --setup instead.")
    print("This command requires previously processed documents.")


def run_query_mode(vector_db_path: str = None,
                   model: str = DEFAULT_MODEL,
                   single_query: str = None,
                   device: str = "cpu"):
    """Run the RAG query system."""
    vector_db_path = vector_db_path or str(DEFAULT_VECTOR_DB)

    # Check if database exists
    if not Path(vector_db_path).exists():
        print(f"Vector database not found at: {vector_db_path}")
        print("Please run with --setup first to process documents and build the database.")
        return

    # Initialize RAG engine
    print(f"Loading model: {model}")
    print("(Make sure Ollama is running: ollama serve)\n")

    rag = HousingMindRAG(
        db_path=vector_db_path,
        model=model
    )

    if single_query:
        # Single query mode
        print(f"\nQuery: {single_query}")
        print("-" * 40)

        result = rag.query(single_query)

        print("\nAnswer:")
        print(result.answer)

        if result.citations_found:
            print("\nCitations:")
            for citation in result.citations_found:
                print(f"  - {citation}")

        print("\nSources:")
        for i, source in enumerate(result.sources, 1):
            print(f"  {i}. {source['source']} ({source['document_type']})")
    else:
        # Interactive mode
        rag.interactive_session()


def main():
    """Main entry point with argument parsing."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="HousingMind RAG System - Llama 3.1 + BGE Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                    # Full setup
  python main.py --setup --max-files 100    # Setup with limited files (testing)
  python main.py --setup --device cuda      # Use GPU for embeddings
  python main.py --query                    # Interactive query mode
  python main.py --query "What is RAD?"     # Single query
  python main.py --process-docs             # Process documents only

Prerequisites:
  1. Install Ollama: https://ollama.ai
  2. Pull the model: ollama pull llama3.1:8b-instruct-q8_0
  3. Start Ollama: ollama serve
        """
    )

    # Action arguments
    parser.add_argument("--setup", action="store_true",
                       help="Run full setup (process docs + build database)")
    parser.add_argument("--process-docs", action="store_true",
                       help="Process documents only (no database)")
    parser.add_argument("--build-db", action="store_true",
                       help="Build database from processed documents")
    parser.add_argument("--query", nargs="?", const=True, default=None,
                       help="Run query mode (interactive or single query)")

    # Path arguments
    parser.add_argument("--raw-docs", "-r",
                       default=None,
                       help=f"Path to raw documents (default: {DEFAULT_RAW_DOCS})")
    parser.add_argument("--db-path", "-d",
                       default=None,
                       help=f"Path to vector database (default: {DEFAULT_VECTOR_DB})")

    # Processing arguments
    parser.add_argument("--chunk-size", "-c",
                       type=int, default=800,
                       help="Chunk size in characters (default: 800)")
    parser.add_argument("--overlap", "-o",
                       type=int, default=200,
                       help="Chunk overlap in characters (default: 200)")
    parser.add_argument("--max-files", "-m",
                       type=int, default=None,
                       help="Maximum files to process (for testing)")

    # Model arguments
    parser.add_argument("--model",
                       default=DEFAULT_MODEL,
                       help=f"Ollama model for generation (default: {DEFAULT_MODEL})")
    parser.add_argument("--device",
                       default="cpu",
                       choices=["cpu", "cuda", "mps"],
                       help="Device for embeddings (default: cpu)")

    args = parser.parse_args()

    # Execute requested action
    if args.setup:
        setup_housingmind(
            raw_docs_path=args.raw_docs,
            vector_db_path=args.db_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            max_files=args.max_files,
            device=args.device
        )
    elif args.process_docs:
        process_documents_only(
            raw_docs_path=args.raw_docs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            max_files=args.max_files
        )
    elif args.build_db:
        build_database_only(
            vector_db_path=args.db_path
        )
    elif args.query is not None:
        single_query = args.query if isinstance(args.query, str) else None
        run_query_mode(
            vector_db_path=args.db_path,
            model=args.model,
            single_query=single_query,
            device=args.device
        )
    else:
        # Default: show help
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start:")
        print("=" * 60)
        print("  1. Install Ollama from https://ollama.ai")
        print(f"  2. Pull the model: ollama pull {DEFAULT_MODEL}")
        print("  3. Start Ollama: ollama serve")
        print("  4. Run setup to process documents:")
        print("     python main.py --setup")
        print("  5. Query the system:")
        print("     python main.py --query")
        print("=" * 60)


if __name__ == "__main__":
    main()
