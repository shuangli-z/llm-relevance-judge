"""
Analyze judgments from a single model file.

Usage:
  python analyze_single.py --file /path/to/judgments.json
  python analyze_single.py --file /path/to/judgments.json --name "Opus Thinking"
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter


def load_judgments(file_path: str) -> dict:
    """Load judgments from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['judgmentRatings']


def get_score_bucket(score: float) -> str:
    """Convert score to bucket for distribution."""
    if score < 0:
        return 'failed'
    elif score == 0:
        return '0.0'
    elif score <= 0.2:
        return '0.0-0.2'
    elif score <= 0.4:
        return '0.2-0.4'
    elif score <= 0.6:
        return '0.4-0.6'
    elif score <= 0.8:
        return '0.6-0.8'
    else:
        return '0.8-1.0'


def compute_distribution(ratings: list) -> dict:
    """Compute score distribution."""
    buckets = ['0.0', '0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', 'failed']
    dist = Counter()
    for r in ratings:
        bucket = get_score_bucket(r)
        dist[bucket] += 1
    return {b: dist.get(b, 0) for b in buckets}


def draw_bar(count: int, max_count: int, width: int = 40) -> str:
    """Draw a text-based bar."""
    if max_count == 0:
        return ''
    bar_len = int((count / max_count) * width)
    return '█' * bar_len + '░' * (width - bar_len)


def analyze_model(judgments: list, model_name: str) -> dict:
    """Analyze judgments for a single model."""
    query_means = []
    all_ratings = []

    for entry in judgments:
        ratings = [float(r['rating']) for r in entry['ratings']]
        valid_ratings = [r for r in ratings if r >= 0]

        if valid_ratings:
            query_mean = np.mean(valid_ratings)
            query_means.append(query_mean)

        all_ratings.extend(ratings)

    # Compute distribution
    distribution = compute_distribution(all_ratings)

    return {
        'model': model_name,
        'macro_avg': np.mean(query_means) if query_means else -1,
        'std': np.std(query_means) if query_means else 0,
        'min': np.min(query_means) if query_means else -1,
        'max': np.max(query_means) if query_means else -1,
        'num_queries': len(judgments),
        'total_docs': len(all_ratings),
        'total_failed': sum(1 for r in all_ratings if r < 0),
        'distribution': distribution,
        'all_ratings': all_ratings
    }


def print_distribution(dist: dict, total: int):
    """Print distribution with bar chart."""
    print("\n" + "-" * 70)
    print("SCORE DISTRIBUTION")
    print("-" * 70)

    buckets_order = ['0.0', '0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

    # Find max for scaling
    max_count = max(dist.get(b, 0) for b in buckets_order)

    bar_width = 40
    for bucket in buckets_order:
        count = dist.get(bucket, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = draw_bar(count, max_count, bar_width)
        print(f"  {bucket:>8}: {bar} {count:>4} ({pct:>5.1f}%)")

    # Print failed separately if any
    failed = dist.get('failed', 0)
    if failed > 0:
        pct = (failed / total * 100) if total > 0 else 0
        print(f"  {'failed':>8}: {'!' * min(failed, bar_width):<{bar_width}} {failed:>4} ({pct:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Analyze judgments from a single model file')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to judgments JSON file')
    parser.add_argument('--name', type=str, default=None,
                        help='Display name for the model (default: extracted from filename)')
    args = parser.parse_args()

    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return

    # Extract model name from filename if not provided
    if args.name:
        model_name = args.name
    else:
        model_name = file_path.stem.replace('_judgments', '').replace('search_res_', '')

    print(f"Analyzing: {file_path}")
    print(f"Model: {model_name}")

    judgments = load_judgments(file_path)
    analysis = analyze_model(judgments, model_name)

    # Print results
    print("\n" + "=" * 70)
    print(f"JUDGMENT ANALYSIS: {model_name.upper()}")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("OVERALL STATISTICS")
    print("-" * 70)
    print(f"{'Metric':<35} {'Value':>20}")
    print("-" * 55)
    print(f"{'Macro Avg (query-level)':<35} {analysis['macro_avg']:>20.3f}")
    print(f"{'Std Dev (query means)':<35} {analysis['std']:>20.3f}")
    print(f"{'Min (query mean)':<35} {analysis['min']:>20.3f}")
    print(f"{'Max (query mean)':<35} {analysis['max']:>20.3f}")
    print(f"{'Total Queries':<35} {analysis['num_queries']:>20}")
    print(f"{'Total Documents':<35} {analysis['total_docs']:>20}")
    print(f"{'Failed Ratings':<35} {analysis['total_failed']:>20}")

    # Print distribution
    print_distribution(analysis['distribution'], analysis['total_docs'])

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
