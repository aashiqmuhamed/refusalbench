"""
Configuration template for GaRAGe RefusalBench variant
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

# GaRAGe Dataset Paths
# ORIGINAL_DATASET_PATH should point to a JSONL file from GaRAGe dataset
# where each line contains:
# - query: the question
# - answer: correct answer(s) as a list
# - signal_docs: relevant context passages (signal)
# - noise_docs: irrelevant context passages (noise)
# - qid: unique question identifier
ORIGINAL_DATASET_PATH = "/path/to/garage_dataset.jsonl"
OUTPUT_DIR = "./garage_output"