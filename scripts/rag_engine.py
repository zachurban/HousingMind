"""
HousingMind RAG Engine

The main query engine that combines vector search with LLM generation
to provide accurate, source-cited responses about housing policy.
"""

import os
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from openai import OpenAI
from setup_vector_db import HousingMindVectorDB


@dataclass
class RAGResponse:
    """Container for RAG query response."""
    answer: str
    sources: List[Dict]
    context_chunks: List[str]
    query: str
    citations_found: List[str]


class HousingMindRAG:
    """
    RAG (Retrieval-Augmented Generation) engine for HousingMind.

    Combines semantic search over housing policy documents with
    GPT-4 generation to provide accurate, cited answers.
    """

    SYSTEM_PROMPT = """You are HousingMind, an expert assistant on U.S. affordable housing policy, HUD regulations, and housing program administration.

You have access to a comprehensive database of HUD notices, handbooks, regulations, and housing policy documents including:
- PIH (Public and Indian Housing) Notices
- HUD Handbooks and Guidebooks
- Code of Federal Regulations (CFR) Title 24
- LIHTC (Low-Income Housing Tax Credit) guidance
- HCV (Housing Choice Voucher) program materials
- Public Housing regulations
- RAD (Rental Assistance Demonstration) documents

When answering questions:
1. Base your responses PRIMARILY on the provided context documents
2. Cite specific regulations, notices, or handbook sections when applicable (e.g., "According to PIH 2023-15..." or "24 CFR 982.201 states...")
3. If the context doesn't contain enough information to fully answer, acknowledge this clearly
4. For complex regulatory questions, explain both the rule and its practical application
5. Use clear, professional language appropriate for housing professionals, PHA staff, and program administrators
6. When discussing income limits, FMRs, or other location-specific data, note that values vary by geography
7. Distinguish between mandatory requirements and recommended practices

IMPORTANT: If the provided context does not contain relevant information for the question, say so explicitly. Do not make up regulations or policies."""

    QUERY_REWRITE_PROMPT = """Rewrite the following user question to be more specific and searchable for a housing policy document database.
The database contains HUD regulations, PIH notices, handbooks, LIHTC guidance, and housing program policies.

Original question: {query}

Rewrite the question to:
1. Use official HUD/housing terminology
2. Be specific about which program (HCV, Public Housing, LIHTC, RAD) if relevant
3. Include relevant regulatory references if obvious

Rewritten question (provide only the rewritten question, no explanation):"""

    def __init__(self,
                 vector_db: Optional[HousingMindVectorDB] = None,
                 db_path: str = "../vector_db",
                 model: str = "gpt-4-turbo-preview",
                 temperature: float = 0.3,
                 use_query_rewriting: bool = False):
        """
        Initialize the RAG engine.

        Args:
            vector_db: Pre-initialized vector database (optional)
            db_path: Path to vector database (if vector_db not provided)
            model: OpenAI model to use for generation
            temperature: Generation temperature (lower = more focused)
            use_query_rewriting: Whether to rewrite queries for better retrieval
        """
        self.model = model
        self.temperature = temperature
        self.use_query_rewriting = use_query_rewriting

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = OpenAI(api_key=api_key)

        # Initialize or use provided vector database
        if vector_db:
            self.vector_db = vector_db
        else:
            self.vector_db = HousingMindVectorDB(persist_directory=db_path)

    def _rewrite_query(self, query: str) -> str:
        """
        Optionally rewrite query for better retrieval.

        Args:
            query: Original user query

        Returns:
            Rewritten query optimized for search
        """
        if not self.use_query_rewriting:
            return query

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use faster model for rewriting
                messages=[
                    {"role": "user", "content": self.QUERY_REWRITE_PROMPT.format(query=query)}
                ],
                temperature=0.3,
                max_tokens=200
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else query
        except Exception:
            return query

    def _extract_citations(self, text: str) -> List[str]:
        """
        Extract regulatory citations from text.

        Args:
            text: Text to search for citations

        Returns:
            List of found citations
        """
        citations = []

        # CFR citations (e.g., "24 CFR 982.201")
        cfr_pattern = r'\b\d+\s+CFR\s+[\d\.]+\b'
        citations.extend(re.findall(cfr_pattern, text, re.IGNORECASE))

        # PIH Notice citations (e.g., "PIH 2023-15")
        pih_pattern = r'PIH[\s\-]?\d{4}[\s\-]?\d+'
        citations.extend(re.findall(pih_pattern, text, re.IGNORECASE))

        # Handbook sections
        handbook_pattern = r'Handbook\s+[\d\.\-]+'
        citations.extend(re.findall(handbook_pattern, text, re.IGNORECASE))

        return list(set(citations))

    def _format_context(self,
                       documents: List[str],
                       metadatas: List[Dict]) -> tuple[str, List[Dict]]:
        """
        Format retrieved documents into context for the LLM.

        Args:
            documents: List of document contents
            metadatas: List of document metadata

        Returns:
            Tuple of (formatted context string, source list)
        """
        context_parts = []
        sources = []

        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            source_name = metadata.get("source", "Unknown")
            doc_type = metadata.get("document_type", "Unknown")
            topic = metadata.get("topic", "")
            notice_num = metadata.get("notice_number", "")

            # Build source reference
            source_ref = f"[Source {i+1}: {source_name}"
            if doc_type != "Unknown":
                source_ref += f" ({doc_type})"
            if notice_num:
                source_ref += f" - {notice_num}"
            source_ref += "]"

            context_parts.append(f"{source_ref}\n{doc}")

            sources.append({
                "source": source_name,
                "document_type": doc_type,
                "topic": topic,
                "notice_number": notice_num,
                "chunk_id": metadata.get("chunk_id"),
            })

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def query(self,
              query: str,
              n_results: int = 5,
              filter_metadata: Optional[Dict] = None,
              include_context: bool = True) -> RAGResponse:
        """
        Process a query and generate a response.

        Args:
            query: User's question
            n_results: Number of documents to retrieve
            filter_metadata: Optional metadata filter
            include_context: Whether to include context chunks in response

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # Optionally rewrite query for better retrieval
        search_query = self._rewrite_query(query)

        # Retrieve relevant documents
        search_results = self.vector_db.search(
            query=search_query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )

        # Handle empty results
        if not search_results["documents"] or not search_results["documents"][0]:
            return RAGResponse(
                answer="I couldn't find relevant documents in the knowledge base to answer your question. Please try rephrasing or ask about a different topic related to affordable housing policy.",
                sources=[],
                context_chunks=[],
                query=query,
                citations_found=[]
            )

        # Format context
        context, sources = self._format_context(
            search_results["documents"][0],
            search_results["metadatas"][0]
        )

        # Build prompt
        user_prompt = f"""Based on the following housing policy documents, please answer the question.

CONTEXT FROM HOUSING POLICY DOCUMENTS:
{context}

---

QUESTION: {query}

Please provide a detailed, accurate answer based on the context above. Cite specific sources, regulations, or notice numbers when applicable."""

        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=2000
        )

        answer = response.choices[0].message.content

        # Extract citations from the answer
        citations = self._extract_citations(answer)

        return RAGResponse(
            answer=answer,
            sources=sources,
            context_chunks=search_results["documents"][0] if include_context else [],
            query=query,
            citations_found=citations
        )

    def query_with_topics(self,
                         query: str,
                         topics: List[str],
                         n_per_topic: int = 3) -> RAGResponse:
        """
        Query with specific topic filtering.

        Args:
            query: User's question
            topics: List of topics to search within
            n_per_topic: Number of results per topic

        Returns:
            RAGResponse with combined results
        """
        all_docs = []
        all_metadata = []

        for topic in topics:
            results = self.vector_db.search(
                query=query,
                n_results=n_per_topic,
                filter_metadata={"topic": topic}
            )

            if results["documents"] and results["documents"][0]:
                all_docs.extend(results["documents"][0])
                all_metadata.extend(results["metadatas"][0])

        if not all_docs:
            return RAGResponse(
                answer=f"No documents found for topics: {', '.join(topics)}",
                sources=[],
                context_chunks=[],
                query=query,
                citations_found=[]
            )

        # Format and generate
        context, sources = self._format_context(all_docs, all_metadata)

        user_prompt = f"""Based on the following housing policy documents about {', '.join(topics)}, please answer the question.

CONTEXT:
{context}

---

QUESTION: {query}

Please provide a detailed answer based on the context above."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=2000
        )

        answer = response.choices[0].message.content
        citations = self._extract_citations(answer)

        return RAGResponse(
            answer=answer,
            sources=sources,
            context_chunks=all_docs,
            query=query,
            citations_found=citations
        )

    def interactive_session(self):
        """Run an interactive query session."""
        print("\n" + "=" * 60)
        print("  HousingMind RAG System")
        print("  Ask questions about affordable housing policy")
        print("=" * 60)
        print("\nCommands:")
        print("  'quit' or 'exit' - End session")
        print("  'topics' - List available topics")
        print("  'stats' - Show database statistics")
        print("\n")

        while True:
            try:
                query = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                break

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if query.lower() == 'topics':
                topics = self.vector_db.list_unique_values("topic")
                print("\nAvailable topics:")
                for topic in topics:
                    print(f"  - {topic}")
                print()
                continue

            if query.lower() == 'stats':
                stats = self.vector_db.get_collection_stats()
                print("\nDatabase Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
                continue

            print("\nSearching knowledge base...\n")

            try:
                result = self.query(query)

                print("-" * 60)
                print("\nAnswer:")
                print(result.answer)

                if result.citations_found:
                    print("\nCitations referenced:")
                    for citation in result.citations_found:
                        print(f"  - {citation}")

                print("\nSources used:")
                for i, source in enumerate(result.sources, 1):
                    source_desc = source['source']
                    if source['notice_number']:
                        source_desc += f" ({source['notice_number']})"
                    elif source['document_type'] != 'Unknown':
                        source_desc += f" ({source['document_type']})"
                    print(f"  {i}. {source_desc}")

                print("\n" + "-" * 60 + "\n")

            except Exception as e:
                print(f"\nError: {str(e)}\n")


def main():
    """Main entry point for the RAG engine."""
    from dotenv import load_dotenv
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(description="HousingMind RAG Query Engine")
    parser.add_argument("--query", "-q",
                       default=None,
                       help="Single query to run (otherwise interactive mode)")
    parser.add_argument("--db-path", "-d",
                       default="../vector_db",
                       help="Path to vector database")
    parser.add_argument("--model", "-m",
                       default="gpt-4-turbo-preview",
                       help="OpenAI model to use")
    parser.add_argument("--results", "-n",
                       type=int, default=5,
                       help="Number of documents to retrieve")

    args = parser.parse_args()

    # Initialize RAG engine
    rag = HousingMindRAG(
        db_path=args.db_path,
        model=args.model
    )

    if args.query:
        # Single query mode
        result = rag.query(args.query, n_results=args.results)

        print("\nAnswer:")
        print(result.answer)

        print("\nSources:")
        for i, source in enumerate(result.sources, 1):
            print(f"  {i}. {source['source']} ({source['document_type']})")
    else:
        # Interactive mode
        rag.interactive_session()


if __name__ == "__main__":
    main()
