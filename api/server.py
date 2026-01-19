"""
HousingMind RAG API Server

FastAPI server providing REST API endpoints for the HousingMind RAG system.

Usage:
    uvicorn api.server:app --reload --port 8000

Endpoints:
    GET  /health           - Health check
    GET  /stats            - Database statistics
    POST /query            - Query the RAG system
    POST /query/topics     - Query with topic filtering
    GET  /topics           - List available topics
    GET  /document-types   - List document types
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from setup_vector_db import HousingMindVectorDB
from rag_engine import HousingMindRAG

# Load environment variables
load_dotenv()

# Default paths
DEFAULT_VECTOR_DB = Path(__file__).parent.parent / "vector_db"

# Global RAG engine instance
rag_engine: Optional[HousingMindRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG engine on startup."""
    global rag_engine

    db_path = os.environ.get("VECTOR_DB_PATH", str(DEFAULT_VECTOR_DB))
    model = os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is required")

    try:
        rag_engine = HousingMindRAG(db_path=db_path, model=model)
        print(f"RAG engine initialized with database at: {db_path}")
    except Exception as e:
        print(f"Warning: Could not initialize RAG engine: {e}")
        print("API will start but queries will fail until database is set up.")

    yield

    # Cleanup (if needed)
    rag_engine = None


# Create FastAPI app
app = FastAPI(
    title="HousingMind RAG API",
    description="API for querying affordable housing policy documents using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="The question to ask", min_length=3)
    n_results: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    filter_topic: Optional[str] = Field(None, description="Filter by topic")
    filter_document_type: Optional[str] = Field(None, description="Filter by document type")
    include_context: bool = Field(False, description="Include retrieved context chunks")


class TopicQueryRequest(BaseModel):
    """Request model for topic-filtered queries."""
    query: str = Field(..., description="The question to ask", min_length=3)
    topics: List[str] = Field(..., description="Topics to search within")
    n_per_topic: int = Field(3, description="Results per topic", ge=1, le=10)


class SourceInfo(BaseModel):
    """Information about a source document."""
    source: str
    document_type: str
    topic: Optional[str]
    notice_number: Optional[str]
    chunk_id: Optional[int]


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: List[SourceInfo]
    citations_found: List[str]
    query: str
    context_chunks: Optional[List[str]] = None


class StatsResponse(BaseModel):
    """Response model for database statistics."""
    total_documents: int
    collection_name: str
    persist_directory: str
    embedding_model: str
    document_types_sample: Dict[str, int]
    topics_sample: Dict[str, int]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    database_ready: bool
    document_count: int


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and database status."""
    if rag_engine is None:
        return HealthResponse(
            status="degraded",
            database_ready=False,
            document_count=0
        )

    try:
        stats = rag_engine.vector_db.get_collection_stats()
        return HealthResponse(
            status="healthy",
            database_ready=True,
            document_count=stats["total_documents"]
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            database_ready=False,
            document_count=0
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        stats = rag_engine.vector_db.get_collection_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a housing policy question.

    Returns an AI-generated answer based on retrieved documents,
    along with source citations.
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Run setup first."
        )

    # Build metadata filter
    filter_metadata = None
    if request.filter_topic or request.filter_document_type:
        filter_metadata = {}
        if request.filter_topic:
            filter_metadata["topic"] = request.filter_topic
        if request.filter_document_type:
            filter_metadata["document_type"] = request.filter_document_type

    try:
        result = rag_engine.query(
            query=request.query,
            n_results=request.n_results,
            filter_metadata=filter_metadata,
            include_context=request.include_context
        )

        return QueryResponse(
            answer=result.answer,
            sources=[SourceInfo(**s) for s in result.sources],
            citations_found=result.citations_found,
            query=result.query,
            context_chunks=result.context_chunks if request.include_context else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/topics", response_model=QueryResponse)
async def query_with_topics(request: TopicQueryRequest):
    """
    Query with specific topic filtering.

    Searches within specified topics and combines results.
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Run setup first."
        )

    try:
        result = rag_engine.query_with_topics(
            query=request.query,
            topics=request.topics,
            n_per_topic=request.n_per_topic
        )

        return QueryResponse(
            answer=result.answer,
            sources=[SourceInfo(**s) for s in result.sources],
            citations_found=result.citations_found,
            query=result.query,
            context_chunks=None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topics", response_model=List[str])
async def list_topics():
    """List all available topics in the database."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        topics = rag_engine.vector_db.list_unique_values("topic")
        return topics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document-types", response_model=List[str])
async def list_document_types():
    """List all document types in the database."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        doc_types = rag_engine.vector_db.list_unique_values("document_type")
        return doc_types
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API root - returns basic info."""
    return {
        "name": "HousingMind RAG API",
        "version": "1.0.0",
        "description": "Query affordable housing policy documents",
        "docs_url": "/docs",
        "health_url": "/health"
    }


# Run with: uvicorn api.server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
