"""
HousingMind Vector Database Setup

Sets up and manages the ChromaDB vector database for storing
and retrieving housing policy document embeddings.
"""

import os
from typing import List, Dict, Optional, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_core.documents import Document


class HousingMindVectorDB:
    """
    Manages the vector database for HousingMind RAG system.

    Uses ChromaDB with OpenAI embeddings for semantic search
    across housing policy documents.
    """

    def __init__(self,
                 persist_directory: str = "./vector_db",
                 collection_name: str = "housing_docs",
                 embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the vector database.

        Args:
            persist_directory: Directory to store the database
            collection_name: Name of the document collection
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Set up OpenAI embeddings
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it with: export OPENAI_API_KEY='your-key'"
            )

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embedding_model
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "HousingMind affordable housing policy documents",
                "embedding_model": embedding_model
            }
        )

        print(f"Vector DB initialized at: {persist_directory}")
        print(f"Collection '{collection_name}' has {self.collection.count()} documents")

    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """
        Add documents to the vector database.

        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process per batch
        """
        if not documents:
            print("No documents to add")
            return

        # Prepare data for Chroma
        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            # Create unique ID from source and chunk_id
            source = doc.metadata.get('source', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 0)
            doc_id = f"{source}_{chunk_id}"

            # Sanitize ID (Chroma has restrictions)
            doc_id = doc_id.replace(" ", "_").replace("/", "_")[:512]

            ids.append(doc_id)
            texts.append(doc.page_content)

            # Convert metadata values to strings (Chroma requirement)
            clean_metadata = {}
            for key, value in doc.metadata.items():
                if value is not None:
                    clean_metadata[key] = str(value)
            metadatas.append(clean_metadata)

        # Add in batches
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch_num = i // batch_size + 1
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]

            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                print(f"Added batch {batch_num}/{total_batches} "
                      f"({len(batch_ids)} documents)")
            except Exception as e:
                print(f"Error adding batch {batch_num}: {str(e)}")
                # Try to add documents one by one to identify problematic ones
                for j, (doc_id, text, metadata) in enumerate(
                        zip(batch_ids, batch_texts, batch_metadatas)):
                    try:
                        self.collection.add(
                            ids=[doc_id],
                            documents=[text],
                            metadatas=[metadata]
                        )
                    except Exception as inner_e:
                        print(f"  Failed to add document {doc_id}: {str(inner_e)}")

        print(f"\nTotal documents in collection: {self.collection.count()}")

    def search(self,
               query: str,
               n_results: int = 5,
               filter_metadata: Optional[Dict] = None,
               include_distances: bool = True) -> Dict[str, Any]:
        """
        Search for relevant documents.

        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Metadata filter (e.g., {"document_type": "PIH Notice"})
            include_distances: Whether to include similarity distances

        Returns:
            Dictionary with documents, metadatas, and distances
        """
        include = ["documents", "metadatas"]
        if include_distances:
            include.append("distances")

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata,
            include=include
        )

        return results

    def search_with_topics(self,
                          query: str,
                          topics: List[str],
                          n_results: int = 5) -> Dict[str, Any]:
        """
        Search with topic filtering.

        Args:
            query: Search query text
            topics: List of topics to filter by (OR logic)
            n_results: Number of results per topic

        Returns:
            Combined search results from all topics
        """
        all_results = {
            "documents": [],
            "metadatas": [],
            "distances": []
        }

        for topic in topics:
            results = self.search(
                query=query,
                n_results=n_results,
                filter_metadata={"topic": topic}
            )

            if results["documents"] and results["documents"][0]:
                all_results["documents"].extend(results["documents"][0])
                all_results["metadatas"].extend(results["metadatas"][0])
                if "distances" in results:
                    all_results["distances"].extend(results["distances"][0])

        return all_results

    def delete_collection(self):
        """Delete the entire collection (use with caution)."""
        self.client.delete_collection(self.collection_name)
        print(f"Collection '{self.collection_name}' deleted")

        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        count = self.collection.count()

        # Sample some documents to get metadata distribution
        sample_size = min(count, 100)
        if sample_size > 0:
            sample = self.collection.peek(limit=sample_size)
            doc_types = {}
            topics = {}

            for metadata in sample.get("metadatas", []):
                doc_type = metadata.get("document_type", "Unknown")
                topic = metadata.get("topic", "Unknown")

                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                topics[topic] = topics.get(topic, 0) + 1
        else:
            doc_types = {}
            topics = {}

        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model,
            "document_types_sample": doc_types,
            "topics_sample": topics
        }

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a specific document by ID.

        Args:
            doc_id: The document ID

        Returns:
            Document data or None if not found
        """
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            if result["documents"]:
                return {
                    "content": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
        except Exception:
            pass
        return None

    def list_unique_values(self, metadata_field: str, limit: int = 100) -> List[str]:
        """
        List unique values for a metadata field.

        Args:
            metadata_field: The metadata field to query
            limit: Maximum number of documents to sample

        Returns:
            List of unique values
        """
        sample = self.collection.peek(limit=limit)
        values = set()

        for metadata in sample.get("metadatas", []):
            value = metadata.get(metadata_field)
            if value:
                values.add(value)

        return sorted(list(values))


def main():
    """Example usage of the vector database."""
    from dotenv import load_dotenv
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(description="HousingMind Vector DB Management")
    parser.add_argument("--action", "-a",
                       choices=["stats", "search", "clear"],
                       default="stats",
                       help="Action to perform")
    parser.add_argument("--query", "-q",
                       default=None,
                       help="Search query")
    parser.add_argument("--db-path", "-d",
                       default="../vector_db",
                       help="Path to vector database")

    args = parser.parse_args()

    # Initialize database
    db = HousingMindVectorDB(persist_directory=args.db_path)

    if args.action == "stats":
        stats = db.get_collection_stats()
        print("\nCollection Statistics:")
        print("-" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")

    elif args.action == "search" and args.query:
        print(f"\nSearching for: '{args.query}'")
        print("-" * 50)

        results = db.search(args.query, n_results=3)

        for i, (doc, metadata) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0])):
            print(f"\nResult {i+1}:")
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print(f"Type: {metadata.get('document_type', 'Unknown')}")
            print(f"Content: {doc[:300]}...")

    elif args.action == "clear":
        confirm = input("Are you sure you want to clear the database? (yes/no): ")
        if confirm.lower() == "yes":
            db.delete_collection()
            print("Database cleared")


if __name__ == "__main__":
    main()
