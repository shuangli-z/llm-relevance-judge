# LLM Relevance Judge

A tool for generating and analyzing relevance judgments using Claude models via AWS Bedrock.

## Features

- **Extended Thinking Mode**: Uses Claude's extended thinking capability for more thorough relevance evaluation
- **Chunking Strategy**: Batches multiple documents per API call to reduce costs
- **Parallel Processing**: Concurrent API calls for faster processing
- **Multi-Model Support**: Supports Claude Sonnet 4.5, Opus 4.5, and Opus 4.6

## Installation

```bash
pip install -r requirements.txt
```

## AWS Credentials

This tool uses AWS Bedrock. Make sure you have AWS credentials configured:

```bash
aws configure
```

Or set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

## Usage

### Generate Judgments (Single Doc Mode)

```bash
python scripts/generate_judgements_thinking.py \
    --input data/search_res.json \
    --output-dir results/ \
    --model opus46 \
    --workers 5
```

### Generate Judgments (Chunked Mode)

```bash
python scripts/generate_judgements_chunked.py \
    --input data/search_res.json \
    --output-dir results/ \
    --token-limit 4000 \
    --workers 3
```

### Analyze Results

Compare two models:
```bash
python scripts/analyze_judgements.py \
    --sonnet results/search_res_sonnet_thinking_judgments.json \
    --opus results/search_res_opus_thinking_judgments.json
```

Analyze single model:
```bash
python scripts/analyze_single.py \
    --file results/search_res_opus46_chunked_judgments.json \
    --name "Opus 4.6"
```

## Input Format

The input JSON file should have the following structure:

```json
[
    {
        "query": "search query text",
        "docs": {
            "doc_id_1": "document content 1",
            "doc_id_2": "document content 2"
        }
    }
]
```

## Output Format

Judgments are output in the following format:

```json
{
    "judgmentRatings": [
        {
            "query": "search query text",
            "ratings": [
                {"docId": "doc_id_1", "rating": "0.850"},
                {"docId": "doc_id_2", "rating": "0.420"}
            ]
        }
    ]
}
```

## Scoring Criteria

- **1.0**: Perfect match, highly relevant
- **0.7-0.9**: Very relevant with minor variations
- **0.4-0.6**: Moderately relevant
- **0.1-0.3**: Slightly relevant
- **0.0**: Completely irrelevant

## License

MIT
