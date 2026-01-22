"""
HousingMind Analytics

Analyze feedback data to identify improvement opportunities and
export training data for fine-tuning.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from feedback_db import FeedbackDB, DEFAULT_DB_PATH


def analyze_feedback(db: FeedbackDB) -> Dict:
    """
    Generate comprehensive analytics from collected feedback.

    Args:
        db: FeedbackDB instance

    Returns:
        Dictionary with analytics results
    """
    results = {
        'generated_at': datetime.now().isoformat(),
        'stats': db.get_feedback_stats(),
        'common_queries': db.get_common_queries(limit=15),
        'problem_queries': db.get_problem_queries(limit=15),
        'source_effectiveness': db.get_source_effectiveness(limit=20)
    }

    return results


def print_analytics_report(results: Dict):
    """Print a formatted analytics report."""
    print("\n" + "=" * 60)
    print("  HousingMind Analytics Report")
    print("=" * 60)
    print(f"Generated: {results['generated_at']}")

    stats = results['stats']
    print("\n### Overall Statistics ###")
    print(f"Total queries: {stats['total']}")
    print(f"Rated queries: {stats['rated']}")

    if stats['rated'] > 0:
        print(f"Helpful: {stats['helpful']} ({stats['helpful_pct']}%)")
        print(f"Not helpful: {stats['not_helpful']} ({100 - stats['helpful_pct']}%)")

    if stats['avg_response_time_ms']:
        print(f"Avg response time: {stats['avg_response_time_ms']}ms")

    # Recent activity
    if stats.get('recent_feedback'):
        print("\n### Recent Activity (Last 7 Days) ###")
        for row in stats['recent_feedback']:
            date = row[0] if isinstance(row, tuple) else row['date']
            count = row[1] if isinstance(row, tuple) else row['count']
            helpful = row[2] if isinstance(row, tuple) else row['helpful_count']
            pct = (helpful / count * 100) if count > 0 else 0
            print(f"  {date}: {count} queries ({helpful} helpful, {pct:.0f}%)")

    # Common queries
    if results['common_queries']:
        print("\n### Most Common Queries ###")
        for i, q in enumerate(results['common_queries'][:10], 1):
            query = q['query'][:50] + "..." if len(q['query']) > 50 else q['query']
            rating = q.get('avg_rating')
            rating_str = f", avg rating: {rating:.1f}" if rating else ""
            print(f"  {i}. \"{query}\" (asked {q['count']}x{rating_str})")

    # Problem queries
    if results['problem_queries']:
        print("\n### Queries Needing Improvement ###")
        print("(These received negative feedback)")
        for i, q in enumerate(results['problem_queries'][:10], 1):
            query = q['query'][:50] + "..." if len(q['query']) > 50 else q['query']
            print(f"  {i}. \"{query}\" (rating: {q['avg_rating']:.1f}, count: {q['count']})")

    # Source effectiveness
    if results['source_effectiveness']:
        print("\n### Source Document Effectiveness ###")
        print("(How often sources were marked helpful)")
        for s in results['source_effectiveness'][:10]:
            source = s['source_doc'][:40] + "..." if len(s['source_doc']) > 40 else s['source_doc']
            print(f"  - {source}: {s['helpful_pct']}% helpful ({s['times_used']} uses)")

    print("\n" + "=" * 60)


def export_training_data(db: FeedbackDB,
                        min_rating: int = 1,
                        output_path: str = None) -> int:
    """
    Export positive examples for fine-tuning.

    Args:
        db: FeedbackDB instance
        min_rating: Minimum rating to include
        output_path: Output file path

    Returns:
        Number of examples exported
    """
    count = db.export_training_data(min_rating, output_path)
    print(f"Exported {count} examples for training")
    return count


def export_problem_cases(db: FeedbackDB, output_path: str = None) -> int:
    """
    Export problematic queries for review.

    Args:
        db: FeedbackDB instance
        output_path: Output file path

    Returns:
        Number of cases exported
    """
    if output_path is None:
        output_path = Path(db.db_path).parent / "problem_cases.jsonl"

    problems = db.get_problem_queries(limit=100)

    with open(output_path, 'w') as f:
        for p in problems:
            f.write(json.dumps(p) + '\n')

    print(f"Exported {len(problems)} problem cases to {output_path}")
    return len(problems)


def main():
    """Run analytics from command line."""
    parser = argparse.ArgumentParser(description="HousingMind Feedback Analytics")
    parser.add_argument("--db-path", "-d",
                       default=str(DEFAULT_DB_PATH),
                       help="Path to feedback database")
    parser.add_argument("--export-training", "-t",
                       action="store_true",
                       help="Export training data")
    parser.add_argument("--export-problems", "-p",
                       action="store_true",
                       help="Export problem cases for review")
    parser.add_argument("--min-rating", "-r",
                       type=int, default=1,
                       help="Minimum rating for training export")
    parser.add_argument("--output", "-o",
                       default=None,
                       help="Output path for exports")
    parser.add_argument("--json", "-j",
                       action="store_true",
                       help="Output analytics as JSON")

    args = parser.parse_args()

    # Initialize database
    db = FeedbackDB(args.db_path)

    if args.export_training:
        export_training_data(db, args.min_rating, args.output)
    elif args.export_problems:
        export_problem_cases(db, args.output)
    else:
        # Run analytics
        results = analyze_feedback(db)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print_analytics_report(results)

    db.close()


if __name__ == "__main__":
    main()
