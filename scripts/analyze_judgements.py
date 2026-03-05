"""
Analyze and compare judgments from Sonnet and Opus models.

Usage:
  python analyze_judgements.py                     # Analyze temp1.0 (original) results
  python analyze_judgements.py --version temp0    # Analyze temp0 results
  python analyze_judgements.py --version thinking # Analyze thinking mode results
  python analyze_judgements.py --sonnet FILE --opus FILE  # Custom files
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


def draw_bar(count: int, max_count: int, width: int = 30) -> str:
    """Draw a text-based bar."""
    if max_count == 0:
        return ''
    bar_len = int((count / max_count) * width)
    return '█' * bar_len + '░' * (width - bar_len)


def print_distribution(dist: dict, model_name: str, total: int):
    """Print distribution with bar chart."""
    print(f"\n{model_name} Score Distribution:")
    print("-" * 60)

    # Exclude 'failed' for max calculation in chart
    valid_counts = [v for k, v in dist.items() if k != 'failed']
    max_count = max(valid_counts) if valid_counts else 1

    buckets_order = ['0.0', '0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

    for bucket in buckets_order:
        count = dist.get(bucket, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = draw_bar(count, max_count)
        print(f"  {bucket:>8}: {bar} {count:>4} ({pct:>5.1f}%)")

    # Print failed separately if any
    failed = dist.get('failed', 0)
    if failed > 0:
        pct = (failed / total * 100) if total > 0 else 0
        print(f"  {'failed':>8}: {'!' * min(failed, 30):<30} {failed:>4} ({pct:>5.1f}%)")


def analyze_model(judgments: list, model_name: str) -> dict:
    """Analyze judgments for a single model."""
    query_means = []
    all_ratings = []
    query_stats = []

    for entry in judgments:
        query = entry['query']
        ratings = [float(r['rating']) for r in entry['ratings']]
        valid_ratings = [r for r in ratings if r >= 0]

        if valid_ratings:
            query_mean = np.mean(valid_ratings)
            query_means.append(query_mean)

        all_ratings.extend(ratings)
        query_stats.append({
            'query': query,
            'mean': np.mean(valid_ratings) if valid_ratings else -1,
            'std': np.std(valid_ratings) if valid_ratings else 0,
            'num_docs': len(ratings),
            'failed': len(ratings) - len(valid_ratings)
        })

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
        'query_stats': query_stats,
        'distribution': distribution,
        'all_ratings': all_ratings
    }


def compare_models(sonnet_judgments: list, opus_judgments: list) -> dict:
    """Compare ratings between two models."""
    comparisons = []

    for s_entry, o_entry in zip(sonnet_judgments, opus_judgments):
        query = s_entry['query']
        s_ratings = {r['docId']: float(r['rating']) for r in s_entry['ratings']}
        o_ratings = {r['docId']: float(r['rating']) for r in o_entry['ratings']}

        diffs = []
        doc_comparisons = []
        for doc_id in s_ratings:
            s_score = s_ratings[doc_id]
            o_score = o_ratings.get(doc_id, -1)
            if s_score >= 0 and o_score >= 0:
                diff = s_score - o_score
                diffs.append(diff)
                doc_comparisons.append({
                    'docId': doc_id,
                    'sonnet': s_score,
                    'opus': o_score,
                    'diff': diff
                })

        s_valid = [s_ratings[d] for d in s_ratings if s_ratings[d] >= 0]
        o_valid = [o_ratings[d] for d in o_ratings if o_ratings[d] >= 0]

        comparisons.append({
            'query': query,
            'sonnet_mean': np.mean(s_valid) if s_valid else -1,
            'opus_mean': np.mean(o_valid) if o_valid else -1,
            'mean_diff': np.mean(diffs) if diffs else 0,
            'abs_mean_diff': np.mean(np.abs(diffs)) if diffs else 0,
            'doc_comparisons': doc_comparisons
        })

    # Overall correlation
    all_sonnet = []
    all_opus = []
    for comp in comparisons:
        for doc in comp['doc_comparisons']:
            all_sonnet.append(doc['sonnet'])
            all_opus.append(doc['opus'])

    correlation = np.corrcoef(all_sonnet, all_opus)[0, 1] if len(all_sonnet) > 1 else 0

    return {
        'query_comparisons': comparisons,
        'correlation': correlation,
        'mean_abs_diff': np.mean([c['abs_mean_diff'] for c in comparisons]),
        'sonnet_higher_count': sum(1 for c in comparisons if c['mean_diff'] > 0.1),
        'opus_higher_count': sum(1 for c in comparisons if c['mean_diff'] < -0.1),
        'similar_count': sum(1 for c in comparisons if abs(c['mean_diff']) <= 0.1)
    }


def print_side_by_side_distribution(sonnet_dist: dict, opus_dist: dict, sonnet_total: int, opus_total: int):
    """Print side-by-side distribution comparison."""
    print("\n" + "-" * 80)
    print("SCORE DISTRIBUTION COMPARISON")
    print("-" * 80)

    buckets_order = ['0.0', '0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

    # Find max for scaling
    s_max = max(sonnet_dist.get(b, 0) for b in buckets_order)
    o_max = max(opus_dist.get(b, 0) for b in buckets_order)

    print(f"\n{'Range':<10} {'Sonnet':<35} {'Opus':<35}")
    print("-" * 80)

    bar_width = 20
    for bucket in buckets_order:
        s_count = sonnet_dist.get(bucket, 0)
        o_count = opus_dist.get(bucket, 0)
        s_pct = (s_count / sonnet_total * 100) if sonnet_total > 0 else 0
        o_pct = (o_count / opus_total * 100) if opus_total > 0 else 0

        s_bar = draw_bar(s_count, s_max, bar_width)
        o_bar = draw_bar(o_count, o_max, bar_width)

        print(f"{bucket:<10} {s_bar} {s_count:>3} ({s_pct:>5.1f}%)  {o_bar} {o_count:>3} ({o_pct:>5.1f}%)")

    # Failed
    s_failed = sonnet_dist.get('failed', 0)
    o_failed = opus_dist.get('failed', 0)
    if s_failed > 0 or o_failed > 0:
        s_pct = (s_failed / sonnet_total * 100) if sonnet_total > 0 else 0
        o_pct = (o_failed / opus_total * 100) if opus_total > 0 else 0
        print(f"{'failed':<10} {'!' * min(s_failed, bar_width):<{bar_width}} {s_failed:>3} ({s_pct:>5.1f}%)  {'!' * min(o_failed, bar_width):<{bar_width}} {o_failed:>3} ({o_pct:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Analyze and compare judgments from Sonnet and Opus models')
    parser.add_argument('--version', type=str, choices=['temp1.0', 'temp0', 'thinking'], default='temp1.0',
                        help='Result version: temp1.0 (original), temp0 (deterministic), or thinking (extended thinking)')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Custom suffix for result files (overrides --version)')
    parser.add_argument('--sonnet', type=str, default=None,
                        help='Custom path to Sonnet judgments file')
    parser.add_argument('--opus', type=str, default=None,
                        help='Custom path to Opus judgments file')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Custom results directory (overrides --version default)')
    args = parser.parse_args()

    # Get script directory for relative paths
    script_dir = Path(__file__).parent.parent
    default_results_dir = script_dir / 'results'

    # Version presets
    VERSION_CONFIGS = {
        'temp1.0': {
            'dir': default_results_dir / 'workbench_temp1.0',
            'suffix': ''
        },
        'temp0': {
            'dir': default_results_dir,
            'suffix': '_temp0'
        },
        'thinking': {
            'dir': default_results_dir,
            'suffix': '_thinking'
        }
    }

    # Determine directory and suffix
    config = VERSION_CONFIGS[args.version]
    results_dir = Path(args.results_dir) if args.results_dir else Path(config['dir'])
    suffix = f'_{args.suffix}' if args.suffix else config['suffix']

    # Determine file paths
    if args.sonnet:
        sonnet_file = Path(args.sonnet)
    else:
        sonnet_file = results_dir / f'search_res_sonnet{suffix}_judgments.json'

    if args.opus:
        opus_file = Path(args.opus)
    else:
        opus_file = results_dir / f'search_res_opus{suffix}_judgments.json'

    print(f"Version: {args.version}")
    print(f"Sonnet file: {sonnet_file}")
    print(f"Opus file: {opus_file}")

    if not sonnet_file.exists():
        print(f"Error: {sonnet_file} not found")
        return
    if not opus_file.exists():
        print(f"Error: {opus_file} not found")
        return

    sonnet_judgments = load_judgments(sonnet_file)
    opus_judgments = load_judgments(opus_file)

    # Analyze each model
    sonnet_analysis = analyze_model(sonnet_judgments, 'Sonnet')
    opus_analysis = analyze_model(opus_judgments, 'Opus')

    # Compare models
    comparison = compare_models(sonnet_judgments, opus_judgments)

    # Print results
    print("=" * 80)
    version_info = f" [{args.version}]"
    print(f"JUDGMENT ANALYSIS RESULTS{version_info}")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("1. OVERALL STATISTICS")
    print("-" * 80)
    print(f"{'Metric':<30} {'Sonnet':>20} {'Opus':>20}")
    print("-" * 70)
    print(f"{'Macro Avg (query-level)':<30} {sonnet_analysis['macro_avg']:>20.3f} {opus_analysis['macro_avg']:>20.3f}")
    print(f"{'Std Dev (query means)':<30} {sonnet_analysis['std']:>20.3f} {opus_analysis['std']:>20.3f}")
    print(f"{'Min (query mean)':<30} {sonnet_analysis['min']:>20.3f} {opus_analysis['min']:>20.3f}")
    print(f"{'Max (query mean)':<30} {sonnet_analysis['max']:>20.3f} {opus_analysis['max']:>20.3f}")
    print(f"{'Total Queries':<30} {sonnet_analysis['num_queries']:>20} {opus_analysis['num_queries']:>20}")
    print(f"{'Total Documents':<30} {sonnet_analysis['total_docs']:>20} {opus_analysis['total_docs']:>20}")
    print(f"{'Failed Ratings':<30} {sonnet_analysis['total_failed']:>20} {opus_analysis['total_failed']:>20}")

    # Print distribution comparison
    print_side_by_side_distribution(
        sonnet_analysis['distribution'],
        opus_analysis['distribution'],
        sonnet_analysis['total_docs'],
        opus_analysis['total_docs']
    )

    print("\n" + "-" * 80)
    print("2. MODEL AGREEMENT")
    print("-" * 80)
    print(f"Correlation (Pearson):                        {comparison['correlation']:.3f}")
    print(f"Mean Absolute Difference:                     {comparison['mean_abs_diff']:.3f}")
    print(f"Queries where Sonnet > Opus (diff > 0.1):     {comparison['sonnet_higher_count']}")
    print(f"Queries where Opus > Sonnet (diff < -0.1):    {comparison['opus_higher_count']}")
    print(f"Queries with similar scores (|diff| <= 0.1): {comparison['similar_count']}")

    # Find biggest disagreements
    print("\n" + "-" * 80)
    print("3. BIGGEST DISAGREEMENTS (by absolute difference)")
    print("-" * 80)
    sorted_comps = sorted(comparison['query_comparisons'], key=lambda x: abs(x['mean_diff']), reverse=True)
    for comp in sorted_comps[:5]:
        print(f"\nQuery: {comp['query']}")
        print(f"  Sonnet: {comp['sonnet_mean']:.3f}, Opus: {comp['opus_mean']:.3f}, Diff: {comp['mean_diff']:+.3f}")
        # Show top doc-level disagreements
        sorted_docs = sorted(comp['doc_comparisons'], key=lambda x: abs(x['diff']), reverse=True)[:3]
        for doc in sorted_docs:
            print(f"    Doc {doc['docId'][:20]}...: Sonnet={doc['sonnet']:.2f}, Opus={doc['opus']:.2f}, Diff={doc['diff']:+.2f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
