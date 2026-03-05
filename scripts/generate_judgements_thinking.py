"""
Generate judgments using Claude Extended Thinking mode.

Uses ThreadPoolExecutor for concurrent API calls with extended thinking enabled.
Note: Extended thinking is incompatible with temperature setting.

Outputs (for each model):
1. {input_name}_{model}_thinking_judgments.json - formatted judgment ratings (0-1 scale)
2. {input_name}_{model}_thinking_llm_responses.json - full LLM responses with thinking
"""

import boto3
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

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

# Model IDs (must support extended thinking)
SONNET_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
OPUS_MODEL_ID = 'us.anthropic.claude-opus-4-5-20251101-v1:0'  # Opus 4.5 for thinking support
OPUS46_MODEL_ID = 'us.anthropic.claude-opus-4-6-v1'  # Opus 4.6

# Extended thinking settings
THINKING_BUDGET_TOKENS = 4096  # Minimum is 1024
MAX_TOKENS = 8192

# Concurrency settings
DEFAULT_MAX_WORKERS = 5  # Lower for thinking mode (longer responses)

# OpenSearch NUMERIC prompt for 0-1 scoring
SYSTEM_PROMPT = """You are an expert search relevance rater. Your task is to evaluate the relevance between search query and results with these criteria:
- Score 1.0: Perfect match, highly relevant
- Score 0.7-0.9: Very relevant with minor variations
- Score 0.4-0.6: Moderately relevant
- Score 0.1-0.3: Slightly relevant
- Score 0.0: Completely irrelevant

Evaluate based on: exact matches, semantic relevance, and overall context between the SearchText and content in Hits.
When a reference is provided, evaluate based on the relevance to both SearchText and its reference.

IMPORTANT: You MUST include a rating for EVERY hit provided.

Return ONLY a JSON object in this EXACT format:
{"ratings": [{"id": "doc_id_here", "rating_score": <score>}]}
Do not include any explanation, commentary, or markdown formatting. Return only the JSON object."""

USER_CONTENT_TEMPLATE = "SearchText - {query}; Hits - {hits}"


def format_hits_json(doc_id: str, document: str) -> str:
    """Format a single document as OpenSearch hits JSON."""
    hits = [{"_id": doc_id, "_source": {"content": document}}]
    return json.dumps(hits)


def parse_score_from_response(response_text: str, doc_id: str) -> tuple:
    """
    Parse score from LLM response with multiple fallback strategies.
    Returns (score, parsed_result) or (None, None) if parsing fails.
    """
    parsed_text = response_text.strip()

    # Remove markdown code blocks
    if '```json' in parsed_text:
        parsed_text = parsed_text.split('```json')[1].split('```')[0].strip()
    elif '```' in parsed_text:
        parsed_text = parsed_text.split('```')[1].split('```')[0].strip()

    # Strategy 1: Try standard JSON parsing
    try:
        result = json.loads(parsed_text)
        ratings = result.get('ratings', [])
        if ratings and len(ratings) > 0:
            score = ratings[0].get('rating_score')
            if score is not None and 0 <= float(score) <= 1:
                return float(score), result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Try to fix common JSON issues
    try:
        fixed_text = re.sub(r',\s*([}\]])', r'\1', parsed_text)
        result = json.loads(fixed_text)
        ratings = result.get('ratings', [])
        if ratings and len(ratings) > 0:
            score = ratings[0].get('rating_score')
            if score is not None and 0 <= float(score) <= 1:
                return float(score), result
    except json.JSONDecodeError:
        pass

    # Strategy 3: Use regex to extract rating_score directly
    patterns = [
        r'"rating_score"\s*:\s*([0-9]*\.?[0-9]+)',
        r'rating_score["\s:]+([0-9]*\.?[0-9]+)',
        r'"score"\s*:\s*([0-9]*\.?[0-9]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    return score, {"ratings": [{"id": doc_id, "rating_score": score}], "parsed_via": "regex"}
            except ValueError:
                continue

    return None, None


def get_judgment_with_thinking(query: str, document: str, doc_id: str, model_id: str, max_retries: int = 2) -> dict:
    """Get relevance judgment using extended thinking mode."""
    client = get_bedrock_client()
    hits_json = format_hits_json(doc_id, document)
    user_content = USER_CONTENT_TEMPLATE.format(query=query, hits=hits_json)

    # Build request body for InvokeModel API with extended thinking
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
                score, parsed_result = parse_score_from_response(response_text, doc_id)

                if score is not None:
                    return {
                        'score': score,
                        'raw_response': response_text,
                        'thinking': thinking_text,
                        'parsed_response': parsed_result,
                        'success': True
                    }
                raise ValueError(f"Could not parse score from response: {response_text[:200]}")

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))  # Longer backoff for thinking mode
                continue
            return {
                'score': -1,
                'raw_response': response_text,
                'thinking': thinking_text,
                'error': str(e),
                'success': False
            }

    return {'score': -1, 'raw_response': response_text, 'thinking': thinking_text, 'error': 'Max retries exceeded', 'success': False}


def process_single_doc(args):
    """Process a single document (for use with ThreadPoolExecutor)."""
    query, doc_id, doc_content, model_id = args
    result = get_judgment_with_thinking(query, doc_content, doc_id, model_id)
    return {
        'query': query,
        'doc_id': doc_id,
        'result': result
    }


def generate_judgments(input_file: str, output_dir: str, model: str = 'both', max_workers: int = DEFAULT_MAX_WORKERS):
    """Generate judgments using extended thinking mode with parallel processing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract base name from input file (e.g., search_res_10 from search_res_10.json)
    input_base_name = Path(input_file).stem

    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        search_data = json.load(f)

    print(f"Loaded {len(search_data)} queries from {input_file}")
    print(f"Using Extended Thinking mode (budget={THINKING_BUDGET_TOKENS} tokens)")
    print(f"Max workers: {max_workers}")

    models_to_run = []
    if model == 'both':
        models_to_run = [('sonnet', SONNET_MODEL_ID), ('opus', OPUS_MODEL_ID)]
    elif model == 'sonnet':
        models_to_run = [('sonnet', SONNET_MODEL_ID)]
    elif model == 'opus':
        models_to_run = [('opus', OPUS_MODEL_ID)]
    elif model == 'opus46':
        models_to_run = [('opus46', OPUS46_MODEL_ID)]

    for model_name, model_id in models_to_run:
        print(f"\n{'='*60}")
        print(f"Running with {model_name.upper()} + Extended Thinking")
        print(f"Model ID: {model_id}")
        print(f"{'='*60}")

        # Output files with _thinking suffix
        judgments_file = output_path / f'{input_base_name}_{model_name}_thinking_judgments.json'
        responses_file = output_path / f'{input_base_name}_{model_name}_thinking_llm_responses.json'

        # Prepare all tasks
        all_tasks = []
        for entry in search_data:
            query = entry['query']
            for doc_id, doc_content in entry['docs'].items():
                all_tasks.append((query, doc_id, doc_content, model_id))

        print(f"Total documents to process: {len(all_tasks)}")

        # Process in parallel
        results_by_query = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_doc, task): task for task in all_tasks}

            with tqdm(total=len(futures), desc=f"Processing ({model_name})") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        query = result['query']
                        doc_id = result['doc_id']

                        # Print LLM output
                        score = result['result']['score']
                        raw_resp = result['result'].get('raw_response', '')
                        tqdm.write(f"\n[{doc_id[:30]}] Score: {score:.3f}")
                        tqdm.write(f"  Response: {raw_resp[:200] if raw_resp else 'N/A'}")

                        if query not in results_by_query:
                            results_by_query[query] = {'ratings': [], 'responses': []}

                        # Add rating
                        results_by_query[query]['ratings'].append({
                            'rating': f"{result['result']['score']:.3f}",
                            'docId': doc_id
                        })

                        # Add response (including thinking)
                        doc_response = {
                            'docId': doc_id,
                            'score': result['result']['score'],
                            'raw_response': result['result'].get('raw_response'),
                            'thinking': result['result'].get('thinking'),
                            'parsed_response': result['result'].get('parsed_response'),
                            'success': result['result']['success']
                        }
                        if not result['result']['success']:
                            doc_response['error'] = result['result'].get('error')
                        results_by_query[query]['responses'].append(doc_response)

                    except Exception as e:
                        task = futures[future]
                        print(f"\nError processing {task[1]}: {e}")

                    pbar.update(1)

        # Convert to output format (preserving original query order)
        judgment_ratings = []
        llm_responses = []

        for entry in search_data:
            query = entry['query']
            if query in results_by_query:
                judgment_ratings.append({
                    'query': query,
                    'ratings': results_by_query[query]['ratings']
                })
                llm_responses.append({
                    'query': query,
                    'doc_responses': results_by_query[query]['responses']
                })

        # Save results
        with open(judgments_file, 'w', encoding='utf-8') as f:
            json.dump({'judgmentRatings': judgment_ratings}, f, indent=2, ensure_ascii=False)

        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(llm_responses, f, indent=2, ensure_ascii=False)

        print(f"\n{model_name.upper()} Judgments saved to {judgments_file}")
        print(f"{model_name.upper()} LLM responses (with thinking) saved to {responses_file}")
        print(f"Total queries processed: {len(judgment_ratings)}")
        total_docs = sum(len(jr['ratings']) for jr in judgment_ratings)
        print(f"Total documents rated: {total_docs}")

        # Summary stats
        success_count = sum(1 for q in llm_responses for d in q['doc_responses'] if d['success'])
        fail_count = total_docs - success_count
        print(f"Success: {success_count}, Failed: {fail_count}")

        # Thinking stats
        thinking_lengths = [
            len(d.get('thinking', '') or '')
            for q in llm_responses
            for d in q['doc_responses']
            if d.get('thinking')
        ]
        if thinking_lengths:
            avg_thinking = sum(thinking_lengths) / len(thinking_lengths)
            print(f"Avg thinking length: {avg_thinking:.0f} chars")


if __name__ == '__main__':
    import argparse

    # Get script directory for relative paths
    script_dir = Path(__file__).parent.parent
    default_input = script_dir / 'data' / 'search_res.json'
    default_output = script_dir / 'results'

    parser = argparse.ArgumentParser(description='Generate judgments with Extended Thinking mode')
    parser.add_argument('--input', default=str(default_input),
                        help='Input search results JSON file')
    parser.add_argument('--output-dir', default=str(default_output),
                        help='Output directory for results')
    parser.add_argument('--model', choices=['sonnet', 'opus', 'opus46', 'both'], default='both',
                        help='Which model to use (default: both)')
    parser.add_argument('--workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help=f'Number of parallel workers (default: {DEFAULT_MAX_WORKERS})')
    parser.add_argument('--thinking-budget', type=int, default=THINKING_BUDGET_TOKENS,
                        help=f'Thinking budget tokens (default: {THINKING_BUDGET_TOKENS}, min: 1024)')
    args = parser.parse_args()

    # Update global thinking budget if specified
    if args.thinking_budget >= 1024:
        THINKING_BUDGET_TOKENS = args.thinking_budget

    generate_judgments(args.input, args.output_dir, args.model, args.workers)
