"""
HousingMind Feedback Database

SQLite-based feedback collection system for tracking user interactions
and response quality. Enables continuous improvement through user ratings.
"""

import sqlite3
import datetime
import json
from pathlib import Path
from typing import List, Dict, Optional, Any


# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "housingmind_feedback.db"


class FeedbackDB:
    """
    Manages feedback collection and storage for HousingMind.

    Tracks:
    - User queries and responses
    - Thumbs up/down ratings
    - Source document usefulness
    - Session information for conversation tracking
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the feedback database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH

        # Create data directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = None
        self._connect()
        self._setup_tables()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access

    def _setup_tables(self):
        """Create database tables if they don't exist."""
        c = self.conn.cursor()

        # Main feedback table
        c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT DEFAULT 'anonymous',
                session_id TEXT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                retrieved_docs TEXT,
                rating INTEGER,
                comment TEXT,
                helpful BOOLEAN,
                response_time_ms INTEGER,
                model_used TEXT
            )
        ''')

        # Track which sources were most useful
        c.execute('''
            CREATE TABLE IF NOT EXISTS source_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id INTEGER NOT NULL,
                source_doc TEXT NOT NULL,
                document_type TEXT,
                was_helpful BOOLEAN,
                FOREIGN KEY (feedback_id) REFERENCES feedback(id)
            )
        ''')

        # Track query categories for analysis
        c.execute('''
            CREATE TABLE IF NOT EXISTS query_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                FOREIGN KEY (feedback_id) REFERENCES feedback(id)
            )
        ''')

        # Create indexes for common queries
        c.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_source_feedback_doc ON source_feedback(source_doc)')

        self.conn.commit()

    def save_feedback(self,
                     query: str,
                     response: str,
                     retrieved_docs: List[Dict] = None,
                     rating: int = None,
                     comment: str = "",
                     user_id: str = "anonymous",
                     session_id: str = None,
                     response_time_ms: int = None,
                     model_used: str = None) -> int:
        """
        Save user feedback to database.

        Args:
            query: User's original question
            response: Generated response
            retrieved_docs: List of retrieved document metadata
            rating: User rating (1 = helpful, -1 = not helpful, or 1-5 scale)
            comment: Optional user comment
            user_id: User identifier
            session_id: Session identifier for conversation tracking
            response_time_ms: Response generation time in milliseconds
            model_used: Name of the model used

        Returns:
            Feedback record ID
        """
        c = self.conn.cursor()

        helpful = None
        if rating is not None:
            helpful = rating > 0

        c.execute('''
            INSERT INTO feedback
            (timestamp, user_id, session_id, query, response, retrieved_docs,
             rating, comment, helpful, response_time_ms, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now().isoformat(),
            user_id,
            session_id,
            query,
            response,
            json.dumps(retrieved_docs) if retrieved_docs else None,
            rating,
            comment,
            helpful,
            response_time_ms,
            model_used
        ))

        feedback_id = c.lastrowid
        self.conn.commit()
        return feedback_id

    def update_rating(self, feedback_id: int, rating: int, comment: str = None):
        """
        Update the rating for an existing feedback record.

        Args:
            feedback_id: ID of the feedback record
            rating: New rating value
            comment: Optional updated comment
        """
        c = self.conn.cursor()

        helpful = rating > 0

        if comment is not None:
            c.execute('''
                UPDATE feedback
                SET rating = ?, helpful = ?, comment = ?
                WHERE id = ?
            ''', (rating, helpful, comment, feedback_id))
        else:
            c.execute('''
                UPDATE feedback
                SET rating = ?, helpful = ?
                WHERE id = ?
            ''', (rating, helpful, feedback_id))

        self.conn.commit()

    def save_source_feedback(self,
                            feedback_id: int,
                            source_doc: str,
                            was_helpful: bool,
                            document_type: str = None):
        """
        Track which specific sources were helpful.

        Args:
            feedback_id: ID of the parent feedback record
            source_doc: Source document identifier
            was_helpful: Whether this source was helpful
            document_type: Type of document (PIH Notice, Handbook, etc.)
        """
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO source_feedback (feedback_id, source_doc, document_type, was_helpful)
            VALUES (?, ?, ?, ?)
        ''', (feedback_id, source_doc, document_type, was_helpful))
        self.conn.commit()

    def add_query_category(self, feedback_id: int, category: str):
        """
        Add a category tag to a query for analysis.

        Args:
            feedback_id: ID of the feedback record
            category: Category name (e.g., "HCV", "Income", "Inspection")
        """
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO query_categories (feedback_id, category)
            VALUES (?, ?)
        ''', (feedback_id, category))
        self.conn.commit()

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get overall feedback statistics.

        Returns:
            Dictionary with feedback statistics
        """
        c = self.conn.cursor()

        stats = {}

        # Total feedback count
        stats['total'] = c.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]

        # Rated feedback
        stats['rated'] = c.execute(
            'SELECT COUNT(*) FROM feedback WHERE rating IS NOT NULL'
        ).fetchone()[0]

        # Helpful vs not helpful
        stats['helpful'] = c.execute(
            'SELECT COUNT(*) FROM feedback WHERE helpful = 1'
        ).fetchone()[0]

        stats['not_helpful'] = c.execute(
            'SELECT COUNT(*) FROM feedback WHERE helpful = 0'
        ).fetchone()[0]

        # Calculate percentages
        if stats['rated'] > 0:
            stats['helpful_pct'] = round(stats['helpful'] / stats['rated'] * 100, 1)
        else:
            stats['helpful_pct'] = 0

        # Average response time
        result = c.execute(
            'SELECT AVG(response_time_ms) FROM feedback WHERE response_time_ms IS NOT NULL'
        ).fetchone()[0]
        stats['avg_response_time_ms'] = round(result) if result else None

        # Feedback over time (last 7 days)
        stats['recent_feedback'] = c.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count,
                   SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful_count
            FROM feedback
            WHERE timestamp >= DATE('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''').fetchall()

        return stats

    def get_problem_queries(self, limit: int = 10) -> List[Dict]:
        """
        Get queries with consistently low ratings.

        Args:
            limit: Maximum number of results

        Returns:
            List of problematic query patterns
        """
        c = self.conn.cursor()

        results = c.execute('''
            SELECT query, response, AVG(rating) as avg_rating, COUNT(*) as count
            FROM feedback
            WHERE rating IS NOT NULL AND rating < 0
            GROUP BY query
            HAVING count >= 1
            ORDER BY count DESC, avg_rating ASC
            LIMIT ?
        ''', (limit,)).fetchall()

        return [dict(row) for row in results]

    def get_common_queries(self, limit: int = 10) -> List[Dict]:
        """
        Get most frequently asked queries.

        Args:
            limit: Maximum number of results

        Returns:
            List of common queries with ratings
        """
        c = self.conn.cursor()

        results = c.execute('''
            SELECT query, COUNT(*) as count,
                   AVG(CASE WHEN rating IS NOT NULL THEN rating ELSE NULL END) as avg_rating,
                   SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful_count
            FROM feedback
            GROUP BY query
            ORDER BY count DESC
            LIMIT ?
        ''', (limit,)).fetchall()

        return [dict(row) for row in results]

    def get_source_effectiveness(self, limit: int = 20) -> List[Dict]:
        """
        Get effectiveness ratings for different source documents.

        Args:
            limit: Maximum number of results

        Returns:
            List of sources with helpfulness stats
        """
        c = self.conn.cursor()

        results = c.execute('''
            SELECT source_doc, document_type,
                   COUNT(*) as times_used,
                   SUM(CASE WHEN was_helpful = 1 THEN 1 ELSE 0 END) as helpful_count,
                   ROUND(AVG(CASE WHEN was_helpful = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as helpful_pct
            FROM source_feedback
            GROUP BY source_doc
            ORDER BY times_used DESC
            LIMIT ?
        ''', (limit,)).fetchall()

        return [dict(row) for row in results]

    def export_training_data(self,
                            min_rating: int = 1,
                            output_path: str = None) -> int:
        """
        Export highly-rated examples for fine-tuning.

        Args:
            min_rating: Minimum rating to include
            output_path: Path for output JSONL file

        Returns:
            Number of examples exported
        """
        if output_path is None:
            output_path = self.db_path.parent / "housingmind_training.jsonl"

        c = self.conn.cursor()

        examples = c.execute('''
            SELECT query, response, retrieved_docs, rating
            FROM feedback
            WHERE rating IS NOT NULL AND rating >= ?
            ORDER BY rating DESC
        ''', (min_rating,)).fetchall()

        training_data = []
        for row in examples:
            docs = json.loads(row['retrieved_docs']) if row['retrieved_docs'] else []

            training_data.append({
                'instruction': row['query'],
                'context': [d.get('source', '') for d in docs],
                'response': row['response'],
                'rating': row['rating']
            })

        # Save as JSONL
        with open(output_path, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')

        return len(training_data)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def setup_feedback_db(db_path: str = None) -> FeedbackDB:
    """
    Initialize the feedback database.

    Args:
        db_path: Optional custom database path

    Returns:
        Initialized FeedbackDB instance
    """
    db = FeedbackDB(db_path)
    print(f"Feedback database initialized at: {db.db_path}")
    return db


# Convenience functions for quick access
_default_db: Optional[FeedbackDB] = None


def get_db() -> FeedbackDB:
    """Get or create the default feedback database."""
    global _default_db
    if _default_db is None:
        _default_db = FeedbackDB()
    return _default_db


def save_feedback(**kwargs) -> int:
    """Save feedback using the default database."""
    return get_db().save_feedback(**kwargs)


def get_stats() -> Dict:
    """Get feedback stats from the default database."""
    return get_db().get_feedback_stats()


if __name__ == "__main__":
    # Test the feedback system
    db = setup_feedback_db()

    # Example feedback
    feedback_id = db.save_feedback(
        query="What are the income limits for HCV?",
        response="Income limits vary by area...",
        retrieved_docs=[{"source": "PIH 2024-01", "title": "Income Limits"}],
        rating=1,
        comment="Very helpful!",
        model_used="llama3.1:8b-instruct-q8_0"
    )

    print(f"Saved feedback ID: {feedback_id}")

    # Get stats
    stats = db.get_feedback_stats()
    print(f"\nFeedback Stats:")
    print(f"  Total: {stats['total']}")
    print(f"  Helpful: {stats['helpful']} ({stats['helpful_pct']}%)")

    db.close()
