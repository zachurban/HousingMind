#!/usr/bin/env python3
"""
HousingMind RAG System - Main Entry Point

This script provides the complete pipeline for setting up and running
the HousingMind RAG (Retrieval-Augmented Generation) system.

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
from setup_vector_db import HousingMindVectorDB
from rag_engine import HousingMindRAG


# Default paths (relative to script location)
DEFAULT_RAW_DOCS = SCRIPT_DIR.parent / "raw_documents"
DEFAULT_VECTOR_DB = SCRIPT_DIR.parent / "vector_db"


def setup_housingmind(raw_docs_path: str = None,
                      vector_db_path: str = None,
                      chunk_size: int = 800,
                      chunk_overlap: int = 200,
                      max_files: int = None) -> HousingMindVectorDB:
    """
    Complete setup pipeline: process documents and build vector database.

    Args:
        raw_docs_path: Path to raw documents directory
        vector_db_path: Path for vector database storage
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        max_files: Maximum files to process (for testing)

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
        collection_name="housing_docs"
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
    print(f"\nDocument types (sample):")
    for doc_type, count in db_stats.get('document_types_sample', {}).items():
        print(f"    {doc_type}: {count}")
    print(f"\nTopics (sample):")
    for topic, count in db_stats.get('topics_sample', {}).items():
        print(f"    {topic}: {count}")

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
                   model: str = "gpt-4-turbo-preview",
                   single_query: str = None):
    """Run the RAG query system."""
    vector_db_path = vector_db_path or str(DEFAULT_VECTOR_DB)

    # Check if database exists
    if not Path(vector_db_path).exists():
        print(f"Vector database not found at: {vector_db_path}")
        print("Please run with --setup first to process documents and build the database.")
        return

    # Initialize RAG engine
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
        description="HousingMind RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                    # Full setup
  python main.py --setup --max-files 100    # Setup with limited files (testing)
  python main.py --query                    # Interactive query mode
  python main.py --query "What is RAD?"     # Single query
  python main.py --process-docs             # Process documents only
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
                       default="gpt-4-turbo-preview",
                       help="OpenAI model for generation (default: gpt-4-turbo-preview)")

    args = parser.parse_args()

    # Check for OPENAI_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required.")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        print("Or create a .env file with: OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    # Execute requested action
    if args.setup:
        setup_housingmind(
            raw_docs_path=args.raw_docs,
            vector_db_path=args.db_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            max_files=args.max_files
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
            single_query=single_query
        )
    else:
        # Default: show help
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start:")
        print("  1. Set your OpenAI API key:")
        print("     export OPENAI_API_KEY='your-key'")
        print("  2. Run setup to process documents:")
        print("     python main.py --setup")
        print("  3. Query the system:")
        print("     python main.py --query")
        print("=" * 60)


if __name__ == "__main__":
    main()
