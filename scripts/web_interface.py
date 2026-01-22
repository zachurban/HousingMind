"""
HousingMind Web Interface

Gradio-based web interface for the HousingMind RAG system with
built-in feedback collection for continuous improvement.
"""

import os
import sys
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from rag_engine import HousingMindRAG, DEFAULT_MODEL
from setup_vector_db import HousingMindVectorDB, DEFAULT_EMBEDDING_MODEL
from feedback_db import FeedbackDB, setup_feedback_db

# Load environment variables
load_dotenv()

# Default paths
DEFAULT_VECTOR_DB = SCRIPT_DIR.parent / "vector_db"
DEFAULT_FEEDBACK_DB = SCRIPT_DIR.parent / "data" / "housingmind_feedback.db"

# Global instances
rag_engine: Optional[HousingMindRAG] = None
feedback_db: Optional[FeedbackDB] = None


def initialize_system():
    """Initialize RAG engine and feedback database."""
    global rag_engine, feedback_db

    db_path = os.environ.get("VECTOR_DB_PATH", str(DEFAULT_VECTOR_DB))

    # Initialize RAG engine
    try:
        rag_engine = HousingMindRAG(db_path=db_path)
        print(f"RAG engine initialized")
        print(f"  Model: {rag_engine.model}")
    except Exception as e:
        print(f"Warning: Could not initialize RAG engine: {e}")
        print("Make sure Ollama is running and the vector database exists.")

    # Initialize feedback database
    feedback_db = setup_feedback_db(str(DEFAULT_FEEDBACK_DB))


def answer_query(query: str, session_state: Dict) -> Tuple[str, Dict, int]:
    """
    Process a query and return the response with sources.

    Args:
        query: User's question
        session_state: Session state dictionary

    Returns:
        Tuple of (response text, updated session state, feedback_id)
    """
    if rag_engine is None:
        return (
            "**Error:** RAG system not initialized. Please make sure:\n"
            "1. Ollama is running (`ollama serve`)\n"
            "2. Vector database exists (`python main.py --setup`)",
            session_state,
            None
        )

    if not query.strip():
        return "Please enter a question.", session_state, None

    # Initialize session if needed
    if session_state is None:
        session_state = {
            'session_id': str(uuid.uuid4()),
            'interactions': [],
            'last_feedback_id': None
        }

    # Generate response
    start_time = time.time()
    try:
        result = rag_engine.query(query)
        response_time_ms = int((time.time() - start_time) * 1000)
    except Exception as e:
        return f"**Error generating response:** {str(e)}", session_state, None

    # Format response
    response_text = result.answer

    # Add citations if found
    if result.citations_found:
        response_text += "\n\n**Citations:**\n"
        for citation in result.citations_found:
            response_text += f"- {citation}\n"

    # Add sources
    response_text += "\n\n**Sources:**\n"
    for i, source in enumerate(result.sources, 1):
        source_desc = source['source']
        if source.get('notice_number'):
            source_desc += f" ({source['notice_number']})"
        elif source.get('document_type') and source['document_type'] != 'Unknown':
            source_desc += f" ({source['document_type']})"
        response_text += f"{i}. {source_desc}\n"

    # Save interaction (without rating yet)
    retrieved_docs = [
        {
            'source': s['source'],
            'document_type': s.get('document_type', 'Unknown'),
            'topic': s.get('topic', ''),
            'notice_number': s.get('notice_number', '')
        }
        for s in result.sources
    ]

    feedback_id = feedback_db.save_feedback(
        query=query,
        response=result.answer,
        retrieved_docs=retrieved_docs,
        session_id=session_state['session_id'],
        response_time_ms=response_time_ms,
        model_used=rag_engine.model
    )

    # Update session state
    session_state['interactions'].append({
        'query': query,
        'response': result.answer,
        'feedback_id': feedback_id,
        'retrieved_docs': retrieved_docs
    })
    session_state['last_feedback_id'] = feedback_id

    return response_text, session_state, feedback_id


def submit_rating(rating: int, comment: str, session_state: Dict) -> str:
    """
    Submit a rating for the last response.

    Args:
        rating: 1 (helpful) or -1 (not helpful)
        comment: Optional feedback comment
        session_state: Session state dictionary

    Returns:
        Status message
    """
    if session_state is None or not session_state.get('last_feedback_id'):
        return "No response to rate yet. Ask a question first."

    feedback_id = session_state['last_feedback_id']

    try:
        feedback_db.update_rating(feedback_id, rating, comment if comment else None)
        emoji = "👍" if rating > 0 else "👎"
        return f"{emoji} Feedback saved! Thank you for helping improve HousingMind."
    except Exception as e:
        return f"Error saving feedback: {str(e)}"


def get_system_stats() -> str:
    """Get system statistics for display."""
    stats_text = "### System Status\n\n"

    # RAG engine status
    if rag_engine:
        db_stats = rag_engine.vector_db.get_collection_stats()
        stats_text += f"**Documents:** {db_stats['total_documents']:,} chunks\n"
        stats_text += f"**Model:** {rag_engine.model}\n"
        stats_text += f"**Embeddings:** {db_stats['embedding_model']}\n\n"
    else:
        stats_text += "**Status:** Not initialized\n\n"

    # Feedback stats
    if feedback_db:
        fb_stats = feedback_db.get_feedback_stats()
        stats_text += "### Feedback Statistics\n\n"
        stats_text += f"**Total Queries:** {fb_stats['total']}\n"
        stats_text += f"**Rated:** {fb_stats['rated']}\n"
        if fb_stats['rated'] > 0:
            stats_text += f"**Helpful:** {fb_stats['helpful']} ({fb_stats['helpful_pct']}%)\n"
        if fb_stats['avg_response_time_ms']:
            stats_text += f"**Avg Response Time:** {fb_stats['avg_response_time_ms']}ms\n"

    return stats_text


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""

    # Example queries for users to try
    example_queries = [
        "What are tenant rights during a RAD conversion?",
        "How do PHAs calculate annual income for HCV?",
        "What are the portability requirements for voucher holders?",
        "When can a PHA deny an applicant based on criminal history?",
        "What is the difference between HQS and NSPIRE inspections?",
    ]

    with gr.Blocks(
        title="HousingMind",
        theme=gr.themes.Soft(),
        css="""
        .feedback-btn { min-width: 100px; }
        .main-container { max-width: 1200px; margin: auto; }
        """
    ) as demo:
        # Session state
        session_state = gr.State(None)
        current_feedback_id = gr.State(None)

        gr.Markdown("""
        # HousingMind
        ### AI-Powered Housing Policy Assistant

        Ask questions about HUD regulations, PIH notices, Housing Choice Vouchers,
        Public Housing, LIHTC, and more. Responses are based on official HUD documentation.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Query input
                query_input = gr.Textbox(
                    label="Ask a housing policy question",
                    placeholder="Example: What are the income limits for the HCV program?",
                    lines=3,
                    max_lines=5
                )

                with gr.Row():
                    submit_btn = gr.Button("Ask HousingMind", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

                # Example queries
                gr.Markdown("**Try these examples:**")
                for example in example_queries:
                    gr.Button(
                        example[:50] + "..." if len(example) > 50 else example,
                        size="sm"
                    ).click(
                        fn=lambda x=example: x,
                        outputs=query_input
                    )

            with gr.Column(scale=1):
                stats_display = gr.Markdown(get_system_stats())
                refresh_stats_btn = gr.Button("Refresh Stats", size="sm")

        # Response area
        gr.Markdown("---")

        response_output = gr.Markdown(
            label="Response",
            value="*Ask a question to get started.*"
        )

        # Feedback section
        gr.Markdown("---")
        gr.Markdown("### Was this response helpful?")

        with gr.Row():
            thumbs_up = gr.Button("👍 Helpful", variant="primary", elem_classes="feedback-btn")
            thumbs_down = gr.Button("👎 Not Helpful", variant="secondary", elem_classes="feedback-btn")

        feedback_comment = gr.Textbox(
            label="Additional feedback (optional)",
            placeholder="What was good/bad about this response? What was missing?",
            lines=2,
            max_lines=4
        )

        feedback_status = gr.Markdown("")

        # Event handlers
        def on_submit(query, state):
            response, new_state, fb_id = answer_query(query, state)
            return response, new_state, fb_id, ""  # Clear feedback status

        submit_btn.click(
            fn=on_submit,
            inputs=[query_input, session_state],
            outputs=[response_output, session_state, current_feedback_id, feedback_status]
        )

        query_input.submit(
            fn=on_submit,
            inputs=[query_input, session_state],
            outputs=[response_output, session_state, current_feedback_id, feedback_status]
        )

        clear_btn.click(
            fn=lambda: ("", "*Ask a question to get started.*", ""),
            outputs=[query_input, response_output, feedback_status]
        )

        thumbs_up.click(
            fn=lambda c, s: submit_rating(1, c, s),
            inputs=[feedback_comment, session_state],
            outputs=feedback_status
        )

        thumbs_down.click(
            fn=lambda c, s: submit_rating(-1, c, s),
            inputs=[feedback_comment, session_state],
            outputs=feedback_status
        )

        refresh_stats_btn.click(
            fn=get_system_stats,
            outputs=stats_display
        )

        # Footer
        gr.Markdown("""
        ---
        *HousingMind uses Llama 3.1 8B and searches through 2,500+ HUD policy documents.
        Your feedback helps improve the system.*
        """)

    return demo


def main():
    """Launch the web interface."""
    import argparse

    parser = argparse.ArgumentParser(description="HousingMind Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    # Initialize system
    print("Initializing HousingMind...")
    initialize_system()

    # Create and launch interface
    demo = create_interface()

    print(f"\nLaunching HousingMind at http://{args.host}:{args.port}")
    if args.share:
        print("Creating public share link...")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
