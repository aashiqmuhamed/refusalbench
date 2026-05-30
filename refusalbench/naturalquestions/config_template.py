"""
Configuration template for RefusalBench
Copy this file to config.py and fill in your credentials
"""

# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY_HERE"
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_KEY_HERE"
AWS_REGION_NAME = "us-east-1"  # or your preferred region

# OpenAI Configuration (optional - only needed if evaluating OpenAI models)
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"  # or set to None if not using OpenAI models

# Model IDs
DEFAULT_GENERATOR_MODEL = 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0'
DEFAULT_EVALUATOR_MODEL = 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0'

# Dataset Paths
# ORIGINAL_DATASET_PATH should point to a JSONL file from Natural Questions (NQ) dataset
# where all models achieved correct answers. Each line should contain:
# - query: the question from NQ
# - answer: correct answer(s) as a list
# - gold_docs or retrieved_docs: relevant context passages
# - qid: unique question identifier
ORIGINAL_DATASET_PATH = "/path/to/nq_all_models_correct.jsonl"
OUTPUT_DIR = "./output"