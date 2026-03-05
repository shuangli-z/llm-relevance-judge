"""
Generate judgments using chunking strategy (batch multiple docs per LLM call).

Key features:
1. Chunks docs by token limit (default 4000) - multiple docs per LLM call
2. Shuffles doc order within each query BEFORE chunking to prevent retriever bias
3. Uses Extended Thinking mode with Opus 4.6

Outputs:
1. {input_name}_opus46_chunked_judgments.json - formatted judgment ratings
2. {input_name}_opus46_chunked_llm_responses.json - full LLM responses with thinking
"""

import boto3
import json
import re
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import tiktoken

# Thread-local storage for boto3 clients
thread_local = threading.local()

def get_bedrock_client():
    """Get thread-local bedrock client."""
    if not hasattr(thread_local, 'client'):
        thread_local.client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',
        )
    return thread_local.client

# Model ID
OPUS46_MODEL_ID = 'us.anthropic.claude-opus-4-6-v1'

# Token and chunking settings
DEFAULT_TOKEN_LIMIT = 4000  # Same as OpenSearch default
THINKING_BUDGET_TOKENS = 4096
MAX_TOKENS = 16384  # Higher for batch responses

# Concurrency settings
DEFAULT_MAX_WORKERS = 3  # Lower for chunked mode (more tokens per call)

# OpenSearch NUMERIC prompt for 0-1 scoring (batch version)
SYSTEM_PROMPT = """You are an expert search relevance rater. Your task is to evaluate the relevance between search query and results with these criteria:
- Score 1.0: Perfect match, highly relevant
- Score 0.7-0.9: Very relevant with minor variations
- Score 0.4-0.6: Moderately relevant
- Score 0.1-0.3: Slightly relevant
- Score 0.0: Completely irrelevant

Evaluate based on: exact matches, semantic relevance, and overall context between the SearchText and content in Hits.

IMPORTANT: You MUST include a rating for EVERY hit provided. Evaluate each document INDEPENDENTLY.

Return ONLY a JSON object in this EXACT format:
{"ratings": [{"id": "doc_id_here", "rating_score": <score>}, ...]}
Do not include any explanation, commentary, or markdown formatting. Return only the JSON object."""

USER_CONTENT_TEMPLATE = "SearchText: {query}; Hits: {hits}"

# Initialize tokenizer (using cl100k_base as approximation for Claude)
try:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except:
    TOKENIZER = None
    print("Warning: tiktoken not available, using character-based estimation")


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    if TOKENIZER:
        return len(TOKENIZER.encode(text))
    # Fallback: estimate ~4 chars per token
    return len(text) // 4


def format_hits_json(docs: list) -> str:
    """Format multiple documents as OpenSearch hits JSON.

    Args:
        docs: List of (doc_id, doc_content) tuples
    """
    hits = [{"id": doc_id, "source": doc_content} for doc_id, doc_content in docs]
    return json.dumps(hits)


def create_chunks(query: str, docs: dict, token_limit: int) -> list:
    """
    Create chunks of documents that fit within token limit.
    Similar to OpenSearch MLInputOutputTransformer.createMLInputs logic.

    Args:
        query: The search query
        docs: Dict of {doc_id: doc_content}
        token_limit: Maximum tokens per chunk

    Returns:
        List of chunks, each chunk is a list of (doc_id, doc_content) tuples
    """
    chunks = []
    current_chunk = []

    # Convert docs dict to list for processing
    doc_items = list(docs.items())

    for doc_id, doc_content in doc_items:
        # Create temp chunk with new doc
        temp_chunk = current_chunk + [(doc_id, doc_content)]

        # Build the full message to count tokens
        hits_json = format_hits_json(temp_chunk)
        full_message = USER_CONTENT_TEMPLATE.format(query=query, hits=hits_json)
        total_tokens = count_tokens(SYSTEM_PROMPT + full_message)

        if total_tokens > token_limit:
            if not current_chunk:
                # Single doc exceeds limit, truncate it
                truncated_content = doc_content[:token_limit * 3]  # Rough truncation
                chunks.append([(doc_id, truncated_content)])
            else:
                # Save current chunk and start new one
                chunks.append(current_chunk)
                current_chunk = [(doc_id, doc_content)]
        else:
            current_chunk = temp_chunk

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def shuffle_docs(docs: dict, seed: int = None) -> dict:
    """
    Shuffle document order to prevent retriever bias.
    Adjacent docs often come from the same retriever, shuffling prevents this bias.

    Args:
        docs: Dict of {doc_id: doc_content}
        seed: Random seed for reproducibility

    Returns:
        New dict with shuffled order
    """
    items = list(docs.items())
    if seed is not None:
        random.seed(seed)
    random.shuffle(items)
    return dict(items)


def parse_scores_from_response(response_text: str, expected_doc_ids: list) -> dict:
    """
    Parse scores from batch LLM response.
    Returns dict of {doc_id: score} or empty dict if parsing fails.
    """
    parsed_text = response_text.strip()

    # Remove markdown code blocks
    if '```json' in parsed_text:
        parsed_text = parsed_text.split('```json')[1].split('```')[0].strip()
    elif '```' in parsed_text:
        parsed_text = parsed_text.split('```')[1].split('```')[0].strip()

    results = {}

    # Strategy 1: Try standard JSON parsing
    try:
        result = json.loads(parsed_text)
        ratings = result.get('ratings', [])
        for rating in ratings:
            doc_id = str(rating.get('id', ''))
            score = rating.get('rating_score')
            if score is not None and 0 <= float(score) <= 1:
                results[doc_id] = float(score)
        if results:
            return results
    except json.JSONDecodeError:
        pass

    # Strategy 2: Try to fix common JSON issues
    try:
        fixed_text = re.sub(r',\s*([}\]])', r'\1', parsed_text)
        result = json.loads(fixed_text)
        ratings = result.get('ratings', [])
        for rating in ratings:
            doc_id = str(rating.get('id', ''))
            score = rating.get('rating_score')
            if score is not None and 0 <= float(score) <= 1:
                results[doc_id] = float(score)
        if results:
            return results
    except json.JSONDecodeError:
        pass

    # Strategy 3: Use regex to extract all id/score pairs
    pattern = r'"id"\s*:\s*"([^"]+)"[^}]*"rating_score"\s*:\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, response_text)
    for doc_id, score_str in matches:
        try:
            score = float(score_str)
            if 0 <= score <= 1:
                results[doc_id] = score
        except ValueError:
            continue

    return results


def get_chunk_judgment_with_thinking(query: str, chunk: list, model_id: str, max_retries: int = 2) -> dict:
    """
    Get relevance judgments for a chunk of documents using extended thinking mode.

    Args:
        query: Search query
        chunk: List of (doc_id, doc_content) tuples
        model_id: Model ID
        max_retries: Number of retries on failure

    Returns:
        Dict with scores, responses, etc.
    """
    client = get_bedrock_client()
    hits_json = format_hits_json(chunk)
    user_content = USER_CONTENT_TEMPLATE.format(query=query, hits=hits_json)

    doc_ids = [doc_id for doc_id, _ in chunk]

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "thinking": {
            "type": "enabled",
            "budget_tokens": THINKING_BUDGET_TOKENS
        },
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": user_content
            }
        ]
    }

    response_text = None
    thinking_text = None

    for attempt in range(max_retries + 1):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read())

            # Extract thinking and text content
            content_blocks = response_body.get('content', [])
            for block in content_blocks:
                if block.get('type') == 'thinking':
                    thinking_text = block.get('thinking', '')
                elif block.get('type') == 'text':
                    response_text = block.get('text', '')

            if response_text:
                scores = parse_scores_from_response(response_text, doc_ids)

                if scores:
                    return {
                        'scores': scores,
                        'doc_ids': doc_ids,
                        'raw_response': response_text,
                        'thinking': thinking_text,
                        'success': True,
                        'chunk_size': len(chunk)
                    }
                raise ValueError(f"Could not parse scores from response: {response_text[:300]}")

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
                continue
            return {
                'scores': {},
                'doc_ids': doc_ids,
                'raw_response': response_text,
                'thinking': thinking_text,
                'error': str(e),
                'success': False,
                'chunk_size': len(chunk)
            }

    return {
        'scores': {},
        'doc_ids': doc_ids,
        'raw_response': response_text,
        'thinking': thinking_text,
        'error': 'Max retries exceeded',
        'success': False,
        'chunk_size': len(chunk)
    }


def process_chunk(args):
    """Process a single chunk (for use with ThreadPoolExecutor)."""
    query, chunk, chunk_idx, model_id = args
    result = get_chunk_judgment_with_thinking(query, chunk, model_id)
    return {
        'query': query,
        'chunk_idx': chunk_idx,
        'result': result
    }


def generate_judgments_chunked(
    input_file: str,
    output_dir: str,
    token_limit: int = DEFAULT_TOKEN_LIMIT,
    max_workers: int = DEFAULT_MAX_WORKERS,
    shuffle_seed: int = 42
):
    """
    Generate judgments using chunking strategy with parallel processing.

    Args:
        input_file: Input JSON file path
        output_dir: Output directory
        token_limit: Max tokens per chunk
        max_workers: Number of parallel workers
        shuffle_seed: Seed for shuffling (for reproducibility)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_base_name = Path(input_file).stem
    model_name = 'opus46'
    model_id = OPUS46_MODEL_ID

    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        search_data = json.load(f)

    print(f"Loaded {len(search_data)} queries from {input_file}")
    print(f"Using Chunking strategy (token_limit={token_limit})")
    print(f"Using Extended Thinking mode (budget={THINKING_BUDGET_TOKENS} tokens)")
    print(f"Shuffle seed: {shuffle_seed}")
    print(f"Max workers: {max_workers}")
    print(f"Model: {model_id}")

    # Output files with _chunked suffix
    judgments_file = output_path / f'{input_base_name}_{model_name}_chunked_judgments.json'
    responses_file = output_path / f'{input_base_name}_{model_name}_chunked_llm_responses.json'
    shuffle_log_file = output_path / f'{input_base_name}_{model_name}_chunked_shuffle_log.json'

    # Prepare all chunks with shuffling
    all_tasks = []
    shuffle_log = []  # To demonstrate shuffling
    total_docs = 0
    total_chunks = 0

    print("\n" + "="*60)
    print("Shuffling and chunking documents...")
    print("="*60)

    for entry in search_data:
        query = entry['query']
        original_docs = entry['docs']
        original_order = list(original_docs.keys())

        # Shuffle docs to prevent retriever bias
        shuffled_docs = shuffle_docs(original_docs, seed=shuffle_seed + hash(query) % 10000)
        shuffled_order = list(shuffled_docs.keys())

        # Log the shuffle for demonstration
        shuffle_entry = {
            'query': query[:50] + '...' if len(query) > 50 else query,
            'original_order': original_order[:10],  # First 10 for brevity
            'shuffled_order': shuffled_order[:10],
            'total_docs': len(original_docs)
        }
        shuffle_log.append(shuffle_entry)

        # Create chunks from shuffled docs
        chunks = create_chunks(query, shuffled_docs, token_limit)

        for chunk_idx, chunk in enumerate(chunks):
            all_tasks.append((query, chunk, chunk_idx, model_id))

        total_docs += len(original_docs)
        total_chunks += len(chunks)

    # Save shuffle log
    with open(shuffle_log_file, 'w', encoding='utf-8') as f:
        json.dump(shuffle_log, f, indent=2, ensure_ascii=False)

    print(f"\nTotal documents: {total_docs}")
    print(f"Total chunks: {total_chunks}")
    print(f"Average docs per chunk: {total_docs / total_chunks:.1f}")
    print(f"Shuffle log saved to: {shuffle_log_file}")

    # Show shuffle examples
    print("\n" + "="*60)
    print("SHUFFLE DEMONSTRATION (first 3 queries):")
    print("="*60)
    for i, entry in enumerate(shuffle_log[:3]):
        print(f"\nQuery {i+1}: {entry['query']}")
        print(f"  Original order (first 10): {entry['original_order']}")
        print(f"  Shuffled order (first 10): {entry['shuffled_order']}")
        print(f"  Total docs: {entry['total_docs']}")

    # Process chunks in parallel
    print("\n" + "="*60)
    print(f"Processing {len(all_tasks)} chunks with {max_workers} workers...")
    print("="*60)

    results_by_query = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, task): task for task in all_tasks}

        with tqdm(total=len(futures), desc="Processing chunks") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    query = result['query']
                    chunk_result = result['result']

                    # Print chunk result summary
                    scores = chunk_result.get('scores', {})
                    chunk_size = chunk_result.get('chunk_size', 0)
                    success = chunk_result.get('success', False)

                    tqdm.write(f"\n[Chunk] Query: {query[:30]}... | Docs: {chunk_size} | Scores: {len(scores)} | Success: {success}")
                    if scores:
                        sample_scores = list(scores.items())[:3]
                        tqdm.write(f"  Sample scores: {sample_scores}")

                    if query not in results_by_query:
                        results_by_query[query] = {'ratings': {}, 'responses': []}

                    # Merge scores
                    for doc_id, score in scores.items():
                        results_by_query[query]['ratings'][doc_id] = score

                    # Add response
                    results_by_query[query]['responses'].append({
                        'chunk_idx': result['chunk_idx'],
                        'doc_ids': chunk_result.get('doc_ids', []),
                        'scores': scores,
                        'raw_response': chunk_result.get('raw_response'),
                        'thinking': chunk_result.get('thinking'),
                        'success': success,
                        'error': chunk_result.get('error')
                    })

                except Exception as e:
                    task = futures[future]
                    print(f"\nError processing chunk: {e}")

                pbar.update(1)

    # Convert to output format
    judgment_ratings = []
    llm_responses = []

    for entry in search_data:
        query = entry['query']
        if query in results_by_query:
            ratings_list = [
                {'rating': f"{score:.3f}", 'docId': doc_id}
                for doc_id, score in results_by_query[query]['ratings'].items()
            ]
            judgment_ratings.append({
                'query': query,
                'ratings': ratings_list
            })
            llm_responses.append({
                'query': query,
                'chunk_responses': sorted(
                    results_by_query[query]['responses'],
                    key=lambda x: x['chunk_idx']
                )
            })

    # Save results
    with open(judgments_file, 'w', encoding='utf-8') as f:
        json.dump({'judgmentRatings': judgment_ratings}, f, indent=2, ensure_ascii=False)

    with open(responses_file, 'w', encoding='utf-8') as f:
        json.dump(llm_responses, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Judgments saved to: {judgments_file}")
    print(f"LLM responses saved to: {responses_file}")
    print(f"Shuffle log saved to: {shuffle_log_file}")
    print(f"Total queries processed: {len(judgment_ratings)}")

    total_rated = sum(len(jr['ratings']) for jr in judgment_ratings)
    print(f"Total documents rated: {total_rated}")

    # Success stats
    success_chunks = sum(
        1 for q in llm_responses
        for r in q['chunk_responses']
        if r['success']
    )
    total_chunks_processed = sum(len(q['chunk_responses']) for q in llm_responses)
    print(f"Successful chunks: {success_chunks}/{total_chunks_processed}")


if __name__ == '__main__':
    import argparse

    # Get script directory for relative paths
    script_dir = Path(__file__).parent.parent
    default_input = script_dir / 'data' / 'search_res.json'
    default_output = script_dir / 'results'

    parser = argparse.ArgumentParser(description='Generate judgments with chunking strategy')
    parser.add_argument('--input', default=str(default_input),
                        help='Input search results JSON file')
    parser.add_argument('--output-dir', default=str(default_output),
                        help='Output directory for results')
    parser.add_argument('--token-limit', type=int, default=DEFAULT_TOKEN_LIMIT,
                        help=f'Token limit per chunk (default: {DEFAULT_TOKEN_LIMIT})')
    parser.add_argument('--workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help=f'Number of parallel workers (default: {DEFAULT_MAX_WORKERS})')
    parser.add_argument('--shuffle-seed', type=int, default=42,
                        help='Random seed for shuffling (default: 42)')
    parser.add_argument('--thinking-budget', type=int, default=THINKING_BUDGET_TOKENS,
                        help=f'Thinking budget tokens (default: {THINKING_BUDGET_TOKENS})')

    args = parser.parse_args()

    if args.thinking_budget >= 1024:
        THINKING_BUDGET_TOKENS = args.thinking_budget

    generate_judgments_chunked(
        args.input,
        args.output_dir,
        args.token_limit,
        args.workers,
        args.shuffle_seed
    )
