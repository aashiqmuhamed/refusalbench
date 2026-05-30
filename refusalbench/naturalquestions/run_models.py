#!/usr/bin/env python3
"""
RefusalBench Model Evaluator using LiteLLM with AWS Bedrock and OpenAI models.
Evaluates RAG models on the RefusalBench dataset with refusal capabilities.
Implements dual evaluation logic for both answerable and unanswerable instances.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm as atqdm
import asyncio
import litellm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pathlib import Path
import datetime

# Import configuration
try:
    from config import (
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME,
        OPENAI_API_KEY, DEFAULT_EVALUATOR_MODEL
    )
    # Set up AWS credentials and OpenAI API key from config
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_REGION_NAME"] = AWS_REGION_NAME
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
except ImportError:
    print("Warning: config.py not found. Please copy config_template.py to config.py and fill in your credentials.")
    print("Using environment variables if available...")

class AsyncRAGModelEvaluator:
    """
    Class for evaluating RAG models on the RefusalBench dataset with refusal capabilities.
    Implements dual evaluation logic for both answerable and unanswerable instances.
    Uses LiteLLM for model API calls.
    """
    def __init__(self, model_ids, evaluator_engine_id, output_dir="./evaluation_results", 
                 batch_size=10, max_concurrent=20, temperature=0.1, max_tokens=2000):
        """
        Initialize the evaluator.
        
        Args:
            model_ids: List of model IDs to evaluate
            evaluator_engine_id: Model ID to use for evaluation
            output_dir: Directory to save evaluation outputs
            batch_size: Number of examples to process in parallel batches
            max_concurrent: Maximum concurrent model calls
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for model responses
        """
        self.model_ids = model_ids
        self.evaluator_engine_id = evaluator_engine_id
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        os.makedirs(output_dir, exist_ok=True)
        
        # Model generation parameters
        self.model_kwargs = {
            # "temperature": temperature,
            # # "max_tokens": max_tokens,
            # "top_p": 0.99
        }
        
        # Async processing setup
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Define valid refusal codes
        self.valid_refusal_codes = [
            "REFUSE_AMBIGUOUS_QUERY",
            "REFUSE_CONTRADICTORY_CONTEXT", 
            "REFUSE_INFO_MISSING_IN_CONTEXT",
            "REFUSE_FALSE_PREMISE_IN_QUERY",
            "REFUSE_GRANULARITY_MISMATCH",
            "REFUSE_NONFACTUAL_QUERY",
            "REFUSE_OTHER"
        ]
    
    @retry(
        wait=wait_random_exponential(min=2, max=30),
        stop=stop_after_attempt(100),
    )
    async def _call_model_async(self, model_id: str, prompt: str) -> str:
        """Call model via LiteLLM asynchronously with retry logic."""
        async with self.semaphore:
            try:
                messages = [{"role": "user", "content": prompt}]
                
                # Call LiteLLM completion
                response = await litellm.acompletion(
                    model=model_id,
                    messages=messages,
                    **self.model_kwargs
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"Error calling model {model_id}: {e}")
                raise e
    
    def load_dataset(self, jsonl_file):
        """
        Load a dataset from a JSONL file.
        
        Args:
            jsonl_file: Path to the dataset JSONL file
            
        Returns:
            List of dataset entries
        """
        dataset = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
            print(f"Loaded {len(dataset)} examples from {jsonl_file}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
        
        return dataset
    
    def load_existing_results(self, results_file=None):
        """
        Load existing evaluation results from CSV file.
        
        Args:
            results_file: Path to existing results CSV file
            
        Returns:
            DataFrame with existing results, or empty DataFrame if file doesn't exist
        """
        if results_file is None:
            results_file = os.path.join(self.output_dir, 'refusalbench_evaluation_results.csv')
        
        try:
            if os.path.exists(results_file):
                existing_df = pd.read_csv(results_file)
                print(f"Loaded {len(existing_df)} existing results from {results_file}")
                return existing_df
            else:
                print(f"No existing results file found at {results_file}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return pd.DataFrame()
    
    def get_missing_models(self, existing_results_df):
        """
        Determine which models need to be evaluated.
        
        Args:
            existing_results_df: DataFrame with existing results
            
        Returns:
            List of model IDs that need evaluation
        """
        if existing_results_df.empty:
            missing_models = self.model_ids
            print(f"No existing results - will evaluate all {len(missing_models)} models")
        else:
            existing_models = set(existing_results_df['model_id'].unique())
            requested_models = set(self.model_ids)
            missing_models = list(requested_models - existing_models)
            
            if missing_models:
                print(f"Found existing results for {len(existing_models)} models: {existing_models}")
                print(f"Will evaluate {len(missing_models)} new models: {missing_models}")
            else:
                print(f"All requested models already evaluated: {existing_models}")
        
        return missing_models
    
    def check_dataset_compatibility(self, existing_results_df, dataset, num_examples=None):
        """
        Check if the dataset is compatible with existing results.
        
        Args:
            existing_results_df: DataFrame with existing results
            dataset: Current dataset to evaluate
            num_examples: Number of examples to evaluate
            
        Returns:
            Boolean indicating compatibility
        """
        if existing_results_df.empty:
            return True
        
        # Check number of examples
        current_num_examples = num_examples if num_examples is not None else len(dataset)
        existing_num_examples = len(existing_results_df['entry_idx'].unique())
        
        if current_num_examples != existing_num_examples:
            print(f"WARNING: Dataset size mismatch!")
            print(f"Existing results: {existing_num_examples} examples")
            print(f"Current request: {current_num_examples} examples")
            print(f"Will proceed with incremental evaluation but metrics may not be comparable")
        
        return True
    
    async def get_model_answer(self, model_id, context, query):
        """
        Get answer prediction from a model using the RefusalBench prompt.
        Uses only the first retrieved document as context.
        
        Args:
            model_id: Model ID to query
            context: First context passage only
            query: User query
            
        Returns:
            Raw model response
        """
        # RefusalBench prompt template
#         prompt_template = """You are an advanced AI assistant designed to answer questions based **only** on the provided context passages. You must not use any external knowledge or prior training. Your goal is to be precise, factual, and aware of the limitations of the provided information.

# Read the following Query and Context Passages carefully.

# QUERY:
# {query}

# CONTEXT PASSAGES:
# {context}

# **INSTRUCTIONS:**
# 1. Your primary task is to answer the QUERY if, and only if, a complete and faithful answer can be derived *solely* from the facts within the CONTEXT PASSAGES.
# 2. You are permitted and expected to perform reasoning, including common-sense inference and basic mathematics, as long as it operates *exclusively* on the information provided in the context. Do not use factual knowledge external to the context.
# 3. If a direct and faithful answer is not possible due to one of the issues below, you MUST respond with **only** the corresponding refusal code. Do not add any other text or explanation.

# **REFUSAL CODES:**
#    * If a single definitive answer is impossible because of ambiguity. This can happen when:
#       * The **Query** itself has an ambiguous word or structure (e.g., "What is the *bat's* weight?").
#       * The **Context** creates ambiguity by providing multiple valid meanings or referents for a term in the Query (e.g., the Query asks about "she" after the Context mentions two different women). 
#    **RESPONSE:** `REFUSE_AMBIGUOUS_QUERY`
#    * If the CONTEXT PASSAGES contain directly contradictory or logically inconsistent information that is essential for answering the QUERY.
#    **RESPONSE:** `REFUSE_CONTRADICTORY_CONTEXT`
#    * If the CONTEXT PASSAGES lack the necessary facts to answer the QUERY, or if the content is entirely irrelevant.
#    **RESPONSE:** `REFUSE_INFO_MISSING_IN_CONTEXT`
#    * If the QUERY contains a false premise (e.g., assumes a non-existent entity or an event that didn't happen) that is contradicted by or unsupported by the CONTEXT PASSAGES.
#    **RESPONSE:** `REFUSE_FALSE_PREMISE_IN_QUERY`
#    * If there is a mismatch between the level of detail requested by the QUERY and the level provided in the CONTEXT PASSAGES (e.g., the query asks for a total, but the context only gives individual parts without enough information to sum them).
#    **RESPONSE:** `REFUSE_GRANULARITY_MISMATCH`
#    * If the QUERY asks for non-factual information (e.g., a subjective opinion, a prediction, a hypothetical, or a moral judgment) that cannot be objectively derived from the facts in the CONTEXT PASSAGES.
#    **RESPONSE:** `REFUSE_NONFACTUAL_QUERY`
#    * For any other specific issue that makes a faithful answer impossible based only on the provided information and is not covered by the codes above.
#    **RESPONSE:** `REFUSE_OTHER`

# Provide your response below:
# ANSWER:"""

        prompt_template = """
        You are an advanced AI assistant designed to answer questions based **only** on the provided context passages. You must not use any external knowledge or prior training. Your goal is to be precise, factual, and aware of the limitations of the provided information.
        
        Read the following Query and Context Passages carefully.
        
        QUERY:
        {query}
        
        CONTEXT PASSAGES:
        {context}
        
        **INSTRUCTIONS:**
        1.  Your primary task is to answer the QUERY if, and only if, a complete and faithful answer can be derived *solely* from the facts within the CONTEXT PASSAGES.
        2.  You are permitted and expected to perform reasoning, including common-sense inference and basic mathematics, as long as it operates *exclusively* on the information provided in the context.
        3.  If a direct and faithful answer is not possible due to one of the issues below, you MUST respond with **only** the corresponding refusal code. Do not add any other text or explanation.
        
        **REFUSAL CODES:**
        
        *   **`REFUSE_AMBIGUOUS_QUERY`**
            *   **When to use:** When the query or context creates multiple plausible interpretations, making a single definitive answer impossible.
            *   **Example:** Query asks about "the bat's weight" after the context mentions both a baseball bat and an animal.
        
        *   **`REFUSE_CONTRADICTORY_CONTEXT`**
            *   **When to use:** When the context contains directly contradictory or logically inconsistent information that is essential for answering the query.
            *   **Example:** One sentence states "revenue was $10M" and another states "revenue was $12M".
        
        *   **`REFUSE_INFO_MISSING_IN_CONTEXT`**
            *   **When to use:** When the context is **silent** on the matter. The specific fact, entity, or value needed to answer is **entirely absent**. Use this when the information is simply not there.
            *   **Example:** Query asks "Who is the CEO?" but the context only describes the company's products.
        
        *   **`REFUSE_FALSE_PREMISE_IN_QUERY`**
            *   **When to use:** When the query assumes a fact (e.g., an event that never happened, a non-existent object) that is contradicted by or unsupported by the context.
            *   **Example:** Query asks "Why did the company's Mars division fail?" but the context states the company has no Mars division.
        
        *   **`REFUSE_GRANULARITY_MISMATCH`**
            *   **When to use:** When the context **contains relevant information**, but it is at the **wrong scale or level of abstraction**. Answering would require a transformation (e.g., aggregation, averaging, generalization) that is impossible or invalid with the given data. This is different from `REFUSE_INFO_MISSING_IN_CONTEXT` because the information is not absent, just incompatible.
            *   **Example:** Query asks for the "average employee salary," but the context only provides individual salaries for two of the fifty employees. The information is present, but insufficient for the requested aggregation.
        
        *   **`REFUSE_NONFACTUAL_QUERY`**
            *   **When to use:** When the query asks for subjective opinions, predictions, hypotheticals, or moral judgments that cannot be objectively derived from the provided facts.
            *   **Example:** Query asks "Is this a good investment?" but the context only provides the stock price.
        
        *   **`REFUSE_OTHER`**
            *   **When to use:** For any other specific issue not covered above that makes a faithful answer impossible.
        
        Provide your response below:
        ANSWER:
        """
        
        prompt = prompt_template.format(query=query, context=context)
        response = await self._call_model_async(model_id, prompt)
        return response.strip()

    async def classify_and_evaluate_response(self, model_output, query, correct_answers=None):
        """
        Universal LLM judge to classify response type and evaluate answer quality.
        Always asks for both classification and quality score - we extract what we need.
        
        Args:
            model_output: Raw model response
            query: User query
            correct_answers: Correct answers (can be None for unanswerable instances)
            
        Returns:
            Tuple of (response_type, answer_quality_score, explanation)
        """
        # Format correct answers (even if None, we'll handle it in the prompt)
        if correct_answers:
            if isinstance(correct_answers, list):
                formatted_correct = '\n'.join(str(ans) for ans in correct_answers if ans)
            else:
                formatted_correct = str(correct_answers)
        else:
            formatted_correct = "[No reference answers provided - this may be an unanswerable instance]"
        
        # Universal prompt that always asks for both classification and quality
        universal_prompt = f"""Analyze the following model response by performing BOTH classification and quality evaluation:

STEP 1: CLASSIFICATION
Determine if this model response is:
1. An attempt to answer the query → "answer_attempt"  
2. A refusal with a specific code:
   - REFUSE_AMBIGUOUS_QUERY
   - REFUSE_CONTRADICTORY_CONTEXT
   - REFUSE_INFO_MISSING_IN_CONTEXT
   - REFUSE_FALSE_PREMISE_IN_QUERY
   - REFUSE_GRANULARITY_MISMATCH
   - REFUSE_NONFACTUAL_QUERY
   - REFUSE_OTHER

Look for refusal codes even if they appear with additional text.

STEP 2: QUALITY EVALUATION
If the response is an answer attempt AND reference answers are provided, evaluate quality:
- Score 1: Not relevant to query and reference answers
- Score 2: According to reference answers but not relevant to query  
- Score 3: Relevant to query and reference answers but contains mistakes
- Score 4: Relevant and has exact same metrics as reference, but not concise
- Score 5: Relevant to query and fully correct according to reference answers

If the response is a refusal OR no reference answers provided, use "N/A".

QUERY:
{query}

MODEL RESPONSE:
{model_output}

REFERENCE ANSWERS:
{formatted_correct}

Respond in this exact format:
CLASSIFICATION: [answer_attempt OR one of the REFUSE_* codes]
QUALITY_SCORE: [1-5 if answer_attempt with references, otherwise N/A]
EXPLANATION: [brief reasoning for both classification and score]"""

        response = await self._call_model_async(self.evaluator_engine_id, universal_prompt)
        
        # Parse the response
        try:
            lines = response.strip().split('\n')
            
            # Extract classification
            classification_line = [line for line in lines if line.startswith('CLASSIFICATION:')][0]
            classification = classification_line.split('CLASSIFICATION:')[1].strip()
            
            # Extract quality score
            score_line = [line for line in lines if line.startswith('QUALITY_SCORE:')][0]
            score_text = score_line.split('QUALITY_SCORE:')[1].strip()
            quality_score = None if score_text == "N/A" else float(score_text)
            
            # Extract explanation
            explanation_line = [line for line in lines if line.startswith('EXPLANATION:')]
            explanation = explanation_line[0].split('EXPLANATION:')[1].strip() if explanation_line else ""
            
            return classification, quality_score, explanation
            
        except (IndexError, Exception) as e:
            # Fallback parsing - look for refusal codes in the text
            response_text = response.upper()
            for code in self.valid_refusal_codes:
                if code in response_text:
                    return code, None, "Detected refusal code in response (fallback parsing)"
            
            return "answer_attempt", None, f"Could not parse response, defaulting to answer attempt. Error: {str(e)}"

    def evaluate_refusal_match(self, predicted_refusal_code, ground_truth_refusal_code):
        """
        Evaluate refusal category match (exact match only - no LLM judge needed).
        
        Args:
            predicted_refusal_code: Model's predicted refusal code
            ground_truth_refusal_code: Ground truth refusal code
            
        Returns:
            Tuple of (is_correct, explanation)
        """
        if predicted_refusal_code == ground_truth_refusal_code:
            return True, f"Exact match: {predicted_refusal_code}"
        else:
            return False, f"Mismatch: predicted={predicted_refusal_code}, ground_truth={ground_truth_refusal_code}"

    async def process_example(self, entry_idx, entry, models_to_evaluate=None):
        """Process a single example across specified models with dual evaluation logic."""
        results = []
        
        # Use specified models or all models
        models_to_evaluate = models_to_evaluate or self.model_ids
        
        # Extract data - use only first retrieved document
        retrieved_docs = entry.get('retrieved_docs', [])
        context = retrieved_docs[0] if retrieved_docs else ""  # Use only first document
        query = entry.get('query', '')
        ground_truth_label = entry.get('ground_truth_label', '')
        correct_answer = entry.get('answer', []) if ground_truth_label == "ANSWER_CORRECTLY" else None
        
        # Skip entries with no context or query
        if not context or not query:
            print(f"Skipping entry {entry_idx}: missing context or query")
            return results
            
        # Evaluate each specified model
        for model_id in models_to_evaluate:
            print(f"Evaluating {model_id} on example {entry_idx}")
            
            try:
                # Get model prediction
                model_raw_output = await self.get_model_answer(model_id, context, query)
                
                # Combined classification and evaluation in one LLM call
                model_predicted_type, answer_quality_score, evaluation_explanation = await self.classify_and_evaluate_response(
                    model_output=model_raw_output,
                    query=query,
                    correct_answers=correct_answer
                )
                
                # Initialize result structure
                result = {
                    'model_id': model_id,
                    'entry_idx': entry_idx,
                    'query': query,
                    'ground_truth_label': ground_truth_label,
                    'ground_truth_answer': correct_answer if correct_answer else None,
                    'model_raw_output': model_raw_output,
                    'model_predicted_type': model_predicted_type,
                    'answer_quality_score': None,
                    'refusal_match_correct': None,
                    'llm_evaluation_explanation': evaluation_explanation
                }
                
                # DUAL EVALUATION LOGIC
                if ground_truth_label == "ANSWER_CORRECTLY":
                    # Answerable instance
                    if model_predicted_type == "answer_attempt":
                        # Both GT and model provide answers - use quality score from combined evaluation
                        result['answer_quality_score'] = answer_quality_score
                        result['refusal_match_correct'] = None  # N/A for answer attempts
                    else:
                        # GT expects answer but model refused - False Refusal
                        result['answer_quality_score'] = None  # N/A
                        result['refusal_match_correct'] = False  # Always wrong to refuse when should answer
                        result['llm_evaluation_explanation'] = f"False Refusal: Model refused with {model_predicted_type} when should have answered"
                else:
                    # Unanswerable instance - GT is a refusal code
                    if model_predicted_type == "answer_attempt":
                        # GT expects refusal but model answered - Missed Refusal
                        result['answer_quality_score'] = None  # N/A
                        result['refusal_match_correct'] = False  # Always wrong to answer when should refuse
                        result['llm_evaluation_explanation'] = f"Missed Refusal: Model answered when should have refused with {ground_truth_label}"
                    else:
                        # Both GT and model are refusal codes - check exact match
                        is_correct, explanation = self.evaluate_refusal_match(model_predicted_type, ground_truth_label)
                        result['answer_quality_score'] = None  # N/A for refusals
                        result['refusal_match_correct'] = is_correct
                        result['llm_evaluation_explanation'] = explanation
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {model_id} on example {entry_idx}: {e}")
                # Add a failed result to keep track
                results.append({
                    'model_id': model_id,
                    'entry_idx': entry_idx,
                    'query': query,
                    'ground_truth_label': ground_truth_label,
                    'ground_truth_answer': correct_answer if correct_answer else None,
                    'model_raw_output': f"ERROR: {str(e)}",
                    'model_predicted_type': "ERROR",
                    'answer_quality_score': None,
                    'refusal_match_correct': None,
                    'llm_evaluation_explanation': "Evaluation failed due to error"
                })
                
        return results
    
    async def evaluate_models(self, dataset, num_examples=None, models_to_evaluate=None):
        """
        Evaluate specified models on the dataset using batched async processing.
        
        Args:
            dataset: Dataset to evaluate on
            num_examples: Number of examples to evaluate (None for all)
            models_to_evaluate: List of model IDs to evaluate (None for all configured models)
        """
        # Use specified models or all models
        models_to_evaluate = models_to_evaluate or self.model_ids
        
        # Limit number of examples if specified
        if num_examples is not None and num_examples < len(dataset):
            dataset = dataset[:num_examples]
        
        print(f"Evaluating {len(models_to_evaluate)} models on {len(dataset)} examples")
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i+self.batch_size]
            batch_tasks = [
                self.process_example(i+idx, entry, models_to_evaluate) 
                for idx, entry in enumerate(batch)
            ]
            
            # Wait for all tasks in the batch to complete
            batch_results = await atqdm.gather(
                *batch_tasks,
                desc=f"Processing batch {i//self.batch_size + 1}/{(len(dataset)-1)//self.batch_size + 1}"
            )
            
            # Flatten results and add to all_results
            for result_list in batch_results:
                all_results.extend(result_list)
        
        return pd.DataFrame(all_results)
    
    def compute_metrics(self, results_df):
        """
        Compute all RefusalBench metrics using the corrected evaluation logic.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            Dictionary with computed metrics per model
        """
        metrics = {}
        
        for model_id in results_df['model_id'].unique():
            model_df = results_df[results_df['model_id'] == model_id].copy()
            model_metrics = {}
            
            # Split into answerable and unanswerable instances
            answerable_df = model_df[model_df['ground_truth_label'] == 'ANSWER_CORRECTLY']
            unanswerable_df = model_df[model_df['ground_truth_label'] != 'ANSWER_CORRECTLY']
            
            # 1. Accuracy (Answerable Instances)
            if len(answerable_df) > 0:
                # Count correct answers (answer_attempt with score >= 4)
                correct_answers = len(answerable_df[
                    (answerable_df['model_predicted_type'] == 'answer_attempt') & 
                    (answerable_df['answer_quality_score'] >= 4)
                ])
                model_metrics['accuracy'] = correct_answers / len(answerable_df)
                model_metrics['num_answerable'] = len(answerable_df)
            else:
                model_metrics['accuracy'] = 0.0
                model_metrics['num_answerable'] = 0
            
            # 2. False Refusal Rate (FRR) - Answerable instances where model refused
            if len(answerable_df) > 0:
                false_refusals = len(answerable_df[
                    answerable_df['model_predicted_type'].isin(self.valid_refusal_codes)
                ])
                model_metrics['false_refusal_rate'] = false_refusals / len(answerable_df)
            else:
                model_metrics['false_refusal_rate'] = 0.0
            
            # 3. Refusal Accuracy (Unanswerable Instances) - Exact match of refusal codes
            if len(unanswerable_df) > 0:
                correct_refusals = len(unanswerable_df[
                    (unanswerable_df['refusal_match_correct'] == True)
                ])
                model_metrics['refusal_accuracy'] = correct_refusals / len(unanswerable_df)
                model_metrics['num_unanswerable'] = len(unanswerable_df)
            else:
                model_metrics['refusal_accuracy'] = 0.0
                model_metrics['num_unanswerable'] = 0
            
            # 4. Overall Refusal Rate (Unanswerable Instances) - Any refusal attempt
            if len(unanswerable_df) > 0:
                any_refusals = len(unanswerable_df[
                    unanswerable_df['model_predicted_type'].isin(self.valid_refusal_codes)
                ])
                model_metrics['overall_refusal_rate'] = any_refusals / len(unanswerable_df)
            else:
                model_metrics['overall_refusal_rate'] = 0.0
            
            # 5. Missed Refusal Rate (MRR) - Unanswerable instances where model answered
            if len(unanswerable_df) > 0:
                missed_refusals = len(unanswerable_df[
                    unanswerable_df['model_predicted_type'] == 'answer_attempt'
                ])
                model_metrics['missed_refusal_rate'] = missed_refusals / len(unanswerable_df)
            else:
                model_metrics['missed_refusal_rate'] = 0.0
            
            # 6. Calibrated Refusal Score (CRS) - Equal weights
            w_A = 0.5
            w_R = 0.5
            model_metrics['calibrated_refusal_score'] = (
                w_A * model_metrics['accuracy'] + 
                w_R * model_metrics['refusal_accuracy']
            )
            
            # 7. Confusion Matrix Data (for unanswerable instances)
            if len(unanswerable_df) > 0:
                gt_labels = unanswerable_df['ground_truth_label'].tolist()
                pred_labels = unanswerable_df['model_predicted_type'].tolist()
                model_metrics['confusion_matrix_data'] = {
                    'ground_truth': gt_labels,
                    'predicted': pred_labels
                }
            
            metrics[model_id] = model_metrics
        
        return metrics
    
    def save_results(self, results_df, output_file=None, metadata=None):
        """Save evaluation results to a CSV file with optional metadata."""
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'refusalbench_evaluation_results.csv')
            
        results_df.to_csv(output_file, index=False)
        print(f"Saved evaluation results to {output_file}")
        
        # Save metadata if provided
        if metadata:
            metadata_file = output_file.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {metadata_file}")
        
        return output_file
    
    def save_metrics(self, metrics, output_file=None, metadata=None):
        """Save computed metrics to files with optional metadata."""
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'refusalbench_metrics.json')
        
        # Convert to JSON-serializable format
        json_metrics = {}
        for model_id, model_metrics in metrics.items():
            json_metrics[model_id] = {}
            for key, value in model_metrics.items():
                if key != 'confusion_matrix_data':
                    json_metrics[model_id][key] = value
        
        # Add metadata if provided
        if metadata:
            json_metrics['_metadata'] = metadata
        
        with open(output_file, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"Saved metrics to {output_file}")
        
        # Also save confusion matrices separately
        for model_id, model_metrics in metrics.items():
            if 'confusion_matrix_data' in model_metrics:
                cm_file = os.path.join(self.output_dir, f'{model_id.replace("/", "_")}_confusion_matrix.csv')
                cm_df = pd.DataFrame(model_metrics['confusion_matrix_data'])
                cm_df.to_csv(cm_file, index=False)
                print(f"Saved confusion matrix for {model_id} to {cm_file}")
        
        return output_file
    
    def print_metrics_summary(self, metrics):
        """Print a formatted summary of metrics."""
        print("\n" + "="*80)
        print("RefusalBench EVALUATION METRICS SUMMARY")
        print("="*80)
        
        for model_id, model_metrics in metrics.items():
            print(f"\nModel: {model_id}")
            print("-" * 60)
            print(f"Answerable Instances ({model_metrics['num_answerable']} examples):")
            print(f"  Accuracy: {model_metrics['accuracy']:.3f} ({model_metrics['accuracy']*100:.1f}%)")
            print(f"  False Refusal Rate: {model_metrics['false_refusal_rate']:.3f} ({model_metrics['false_refusal_rate']*100:.1f}%)")
            
            print(f"\nUnanswerable Instances ({model_metrics['num_unanswerable']} examples):")
            print(f"  Refusal Accuracy: {model_metrics['refusal_accuracy']:.3f} ({model_metrics['refusal_accuracy']*100:.1f}%)")
            print(f"  Overall Refusal Rate: {model_metrics['overall_refusal_rate']:.3f} ({model_metrics['overall_refusal_rate']*100:.1f}%)")
            print(f"  Missed Refusal Rate: {model_metrics['missed_refusal_rate']:.3f} ({model_metrics['missed_refusal_rate']*100:.1f}%)")
            
            print(f"\nComposite Metrics:")
            print(f"  Calibrated Refusal Score: {model_metrics['calibrated_refusal_score']:.3f}")
        
        print("\n" + "="*80)
        
    async def run_async(self, dataset_path, num_examples=None, incremental=True):
        """
        Run the full evaluation workflow asynchronously with optional incremental evaluation.
        
        Args:
            dataset_path: Path to the dataset JSONL file
            num_examples: Number of examples to evaluate (None for all)
            incremental: If True, only evaluate missing models and merge with existing results
        """
        # 1. Load dataset
        print(f"Loading dataset from {dataset_path}...")
        dataset = self.load_dataset(dataset_path)
        
        if not dataset:
            print("Empty dataset, aborting evaluation")
            return {}
        
        # 2. Handle incremental evaluation
        existing_results_df = pd.DataFrame()
        models_to_evaluate = self.model_ids
        
        if incremental:
            print("\n=== INCREMENTAL EVALUATION MODE ===")
            # Load existing results
            existing_results_df = self.load_existing_results()
            
            # Check dataset compatibility
            self.check_dataset_compatibility(existing_results_df, dataset, num_examples)
            
            # Determine which models need evaluation
            models_to_evaluate = self.get_missing_models(existing_results_df)
            
            if not models_to_evaluate:
                print("All requested models already evaluated! Computing metrics on existing results...")
                # Just compute metrics on existing results
                metrics = self.compute_metrics(existing_results_df)
                self.print_metrics_summary(metrics)
                return {
                    'results': existing_results_df,
                    'metrics': metrics,
                    'results_file': os.path.join(self.output_dir, 'refusalbench_evaluation_results.csv'),
                    'metrics_file': os.path.join(self.output_dir, 'refusalbench_metrics.json'),
                    'incremental': True,
                    'new_models_evaluated': []
                }
        else:
            print("\n=== FULL EVALUATION MODE ===")
            print("Will evaluate all specified models from scratch")
        
        # 3. Evaluate missing/new models
        if models_to_evaluate:
            print(f"\nEvaluating {len(models_to_evaluate)} models on dataset...")
            new_results_df = await self.evaluate_models(dataset, num_examples, models_to_evaluate)
            
            # 4. Merge results if incremental
            if incremental and not existing_results_df.empty:
                print(f"\nMerging results:")
                print(f"  Existing results: {len(existing_results_df)} rows")
                print(f"  New results: {len(new_results_df)} rows")
                
                # Combine results
                results_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
                print(f"  Combined results: {len(results_df)} rows")
            else:
                results_df = new_results_df
        else:
            # No new models to evaluate, use existing results
            results_df = existing_results_df
        
        # 5. Compute metrics on all results
        print("\nComputing metrics...")
        metrics = self.compute_metrics(results_df)
        
        # 6. Print summary
        self.print_metrics_summary(metrics)
        
        # 7. Save results with metadata
        metadata = {
            'evaluation_timestamp': datetime.datetime.now().isoformat(),
            'dataset_path': dataset_path,
            'num_examples_evaluated': len(dataset) if num_examples is None else min(num_examples, len(dataset)),
            'incremental_mode': incremental,
            'models_in_results': list(results_df['model_id'].unique()),
            'new_models_evaluated': models_to_evaluate if models_to_evaluate else [],
            'total_predictions': len(results_df),
            'evaluator_engine': self.evaluator_engine_id,
            'batch_size': self.batch_size,
            'max_concurrent': self.max_concurrent
        }
        
        results_file = self.save_results(results_df, metadata=metadata)
        metrics_file = self.save_metrics(metrics, metadata=metadata)
        
        return {
            'results': results_df,
            'metrics': metrics,
            'results_file': results_file,
            'metrics_file': metrics_file,
            'incremental': incremental,
            'new_models_evaluated': models_to_evaluate if models_to_evaluate else []
        }
    
    def run(self, dataset_path, num_examples=None, incremental=True):
        """
        Run the evaluation synchronously with optional incremental evaluation.
        
        Args:
            dataset_path: Path to the dataset JSONL file
            num_examples: Number of examples to evaluate (None for all)
            incremental: If True, only evaluate missing models and merge with existing results
        """
        return asyncio.run(self.run_async(dataset_path, num_examples, incremental))

# Main execution function
async def main_async():
    """
    Main function to run the RefusalBench evaluation.
    """
    # Import configuration if available for defaults
    try:
        from config import DEFAULT_EVALUATOR_MODEL
        evaluation_engine = DEFAULT_EVALUATOR_MODEL
    except:
        evaluation_engine = 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0'

    # Configuration - Update these paths for your dataset
    dataset_path = './filtered_dataset/refusalbench_perturbed_reference.jsonl'
    
    # Models to evaluate
    models_to_evaluate = [
        "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        # "bedrock/us.anthropic.claude-opus-4-20250514-v1:0",
        "bedrock/us.amazon.nova-pro-v1:0",  
        "bedrock/converse/us.amazon.nova-premier-v1:0",
        "openai/gpt-4o-2024-08-06", 
        "openai/gpt-4.1-2025-04-14",
        "openai/o4-mini-2025-04-16"
    ]
    
    output_dir = './refusalbench_evaluation_results'
    
    # Number of examples to evaluate (set to None to evaluate all)
    num_examples = None
    
    # Initialize evaluator
    evaluator = AsyncRAGModelEvaluator(
        model_ids=models_to_evaluate,
        evaluator_engine_id=evaluation_engine,
        output_dir=output_dir,
        batch_size=5,  # Smaller batch size for more complex evaluation
        max_concurrent=10,  # Reduced concurrency for stability
        temperature=0.1,
        max_tokens=2000
    )
    
    # Run evaluation with incremental support
    print(f"Evaluating {len(models_to_evaluate)} models on up to {num_examples if num_examples else 'all'} examples from RefusalBench dataset...")
    results = await evaluator.run_async(dataset_path, num_examples, incremental=True)
    
    print("\nEvaluation completed!")
    print(f"New models evaluated: {results['new_models_evaluated']}")
    print(f"Total models in results: {len(results['metrics'])}")
    
    return results

def main():
    """Synchronous wrapper for the main function."""
    return asyncio.run(main_async())

if __name__ == "__main__":
    main()