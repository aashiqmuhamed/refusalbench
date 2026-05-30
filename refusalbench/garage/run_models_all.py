#!/usr/bin/env python3
"""
Hybrid RefusalBench Evaluator using GaRAGe + Refusal Metrics.
Uses RefusalBench refusal-aware prompt for ALL instances, then applies appropriate metrics:
- For answerable instances: Uses GaRAGe metrics (Eligibility, Factuality, RAF)
- For unanswerable instances: Uses RefusalBench refusal metrics (Refusal Accuracy, etc.)

FIXED: Unique Sample ID Generation
- Uses original sample_id + entry_idx concatenated for guaranteed uniqueness
- Handles missing sample_ids gracefully
- Ensures robust resume functionality

DATA STRUCTURE (precise field locations):
- refusalbench_perturbation_metadata.expected_rag_behavior: "REFUSE_AMBIGUOUS_QUERY" or "ANSWER_CORRECTLY"
- refusalbench_perturbation_metadata.perturbation_class: "P-Ambiguity", "P-Contradiction", etc.
- refusalbench_perturbation_metadata.intensity: "LOW", "MEDIUM", "HIGH"
- refusalbench_stratified_metadata.generator_model: "claude", "gpt", "nova", "deepseek"
- grounding[i].cite_X: actual passage text content
- evidence_correct[i]: "ANSWER-THE-QUESTION" for relevant passages (RAF metric)

DOMAIN/TOPIC ANALYSIS (captured in results):
- question_category: Primary domain (Finance, Science, Technology, etc.)
- topic_tag: Specific topic (arxiv, web, ent, etc.)
- question_type: Temporal nature (SLOW-CHANGING, FAST-CHANGING, etc.)
- question_complexity: Complexity type (Simple, Comparison, Multi-step, etc.)
- question_popularity: Frequency (Head, Torso, Tail)
- source_dataset: Original dataset (GaRAGe, etc.)

OUTPUT: Comprehensive CSV with all metadata for domain-specific analysis
🔄 RESUME FUNCTIONALITY: Automatically detects and skips completed evaluations
"""
import os
import json
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm as atqdm
import litellm
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import random
from collections import defaultdict
import datetime
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Import configuration
try:
    from config import (
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME,
        OPENAI_API_KEY
    )
    # Set up AWS credentials and OpenAI API key from config
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_REGION_NAME"] = AWS_REGION_NAME
    if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
except ImportError:
    print("Warning: config.py not found. Please copy config_template.py to config.py and fill in your credentials.")
    # Check if credentials are in environment
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        raise RuntimeError("AWS credentials not found. Please set up config.py or environment variables.")

class HybridRefusalBenchEvaluator:
    """
    Hybrid evaluator for RefusalBench dataset that applies:
    - GaRAGe metrics for answerable instances 
    - RefusalBench refusal metrics for unanswerable instances
    🔄 WITH RESUME FUNCTIONALITY: Automatically continues from where it left off
    """
    
    def __init__(self, model_ids: List[str], judge_model_id: str, output_dir: str = "./hybrid_evaluation_results", 
                 batch_size: int = 5, max_concurrent: int = 10):
        """
        Initialize the hybrid evaluator.
        
        Args:
            model_ids: List of RAG model IDs to evaluate
            judge_model_id: Model ID to use for judging
            output_dir: Directory to save results
            batch_size: Number of examples to process in parallel batches
            max_concurrent: Maximum concurrent model calls
        """
        self.model_ids = model_ids
        self.judge_model_id = judge_model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        
        # Async processing setup
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 🔄 RESUME FUNCTIONALITY - Track completed evaluations
        self.results_file = self.output_dir / "hybrid_evaluation_results.csv"
        self.completed_evaluations: Set[Tuple[str, str]] = set()  # (sample_id, model_id) pairs
        self.existing_results_df: Optional[pd.DataFrame] = None
        
        # Define valid refusal codes from RefusalBench
        self.valid_refusal_codes = [
            "REFUSE_AMBIGUOUS_QUERY",
            "REFUSE_CONTRADICTORY_CONTEXT", 
            "REFUSE_INFO_MISSING_IN_CONTEXT",
            "REFUSE_FALSE_PREMISE_IN_QUERY",
            "REFUSE_GRANULARITY_MISMATCH",
            "REFUSE_NONFACTUAL_QUERY",
            "REFUSE_OTHER"
        ]
        
        print("🔀 Hybrid RefusalBench Evaluator")
        print(f"📋 Uses merged GaRAGe + RefusalBench prompt for ALL instances")
        print(f"📊 Answerable instances → GaRAGe metrics (Eligibility, Factuality, RAF)")
        print(f"🚫 Unanswerable instances → RefusalBench refusal metrics")
        print(f"🔗 Context includes both relevant and irrelevant passages (as in original)")
        print(f"🤖 Models to test: {len(model_ids)}")
        print(f"⚖️  Judge model: {judge_model_id}")
        print(f"🎯 PRECISE: Field extraction based on exact data structure")
        print(f"🔄 RESUME: Automatically detects and skips completed evaluations")
        print(f"🆔 FIXED: Unique sample IDs using original_id + entry_idx")

    def load_existing_results(self) -> bool:
        """
        🔄 RESUME: Load existing results and determine what's already completed.
        Returns True if existing results were found, False otherwise.
        """
        if not self.results_file.exists():
            print("📄 No existing results found - starting fresh evaluation")
            return False
        
        try:
            self.existing_results_df = pd.read_csv(self.results_file)
            print(f"📄 Found existing results: {len(self.existing_results_df)} evaluations")
            
            # Track completed (sample_id, model_id) combinations
            for _, row in self.existing_results_df.iterrows():
                sample_id = str(row['sample_id'])
                model_id = str(row['model_id'])
                self.completed_evaluations.add((sample_id, model_id))
            
            # Report resume status
            completed_by_model = self.existing_results_df.groupby('model_id').size()
            print("🔄 RESUME STATUS by model:")
            for model_id in self.model_ids:
                count = completed_by_model.get(model_id, 0)
                print(f"   {model_id}: {count} examples completed")
            
            unique_samples = self.existing_results_df['sample_id'].nunique()
            print(f"   📊 Total unique samples with some evaluation: {unique_samples}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error loading existing results: {e}")
            print("   Starting fresh evaluation")
            self.existing_results_df = None
            self.completed_evaluations = set()
            return False

    def is_evaluation_completed(self, sample_id: str, model_id: str) -> bool:
        """🔄 RESUME: Check if a specific (sample_id, model_id) evaluation is already completed."""
        return (sample_id, model_id) in self.completed_evaluations

    def save_incremental_results(self, new_results_df: pd.DataFrame):
        """🔄 RESUME: Save results incrementally, merging with existing results."""
        try:
            if self.existing_results_df is not None and len(self.existing_results_df) > 0:
                # Merge with existing results
                combined_df = pd.concat([self.existing_results_df, new_results_df], ignore_index=True)
                # Remove duplicates based on sample_id + model_id (keep last)
                combined_df = combined_df.drop_duplicates(subset=['sample_id', 'model_id'], keep='last')
            else:
                combined_df = new_results_df
            
            # Save to file
            combined_df.to_csv(self.results_file, index=False)
            
            # Update internal tracking
            self.existing_results_df = combined_df
            for _, row in new_results_df.iterrows():
                sample_id = str(row['sample_id'])
                model_id = str(row['model_id'])
                self.completed_evaluations.add((sample_id, model_id))
            
            print(f"💾 Incremental save: {len(combined_df)} total evaluations")
            
        except Exception as e:
            print(f"⚠️  Error saving incremental results: {e}")

    def extract_field_from_entry(self, entry: Dict[str, Any], field_name: str) -> Any:
        """
        Extract field from entry based on exact data structure provided.
        PRECISE field locations (no guessing):
        """
        # Exact field mappings based on your data structure
        field_locations = {
            # Top-level fields
            'question': entry.get('question'),
            'answer_generate': entry.get('answer_generate'),
            'grounding': entry.get('grounding'),
            'evidence_correct': entry.get('evidence_correct'),
            'sample_id': entry.get('sample_id'),
            
            # Domain/topic metadata (for analysis)
            'question_category': entry.get('question_category'),  # e.g., "Finance"
            'topic_tag': entry.get('topic_tag'),                  # e.g., "arxiv"
            'question_tag': entry.get('question_tag'),            # e.g., "web"
            'question_type': entry.get('question_type'),          # e.g., "SLOW-CHANGING"
            'question_complexity': entry.get('question_complexity'), # e.g., "Comparison"
            'question_popularity': entry.get('question_popularity'),  # e.g., "Tail"
            
            # Curation metadata
            'source_dataset': entry.get('curation_notes', {}).get('source_dataset'),  # e.g., "GaRAGe"
            'curation_status': entry.get('curation_notes', {}).get('status'),         # e.g., "Pristine"
            'normalization_strategy': entry.get('curation_notes', {}).get('normalization_strategy'), # e.g., "1_signal_9_noise"
            
            # refusalbench_perturbation_metadata fields
            'perturbation_class': entry.get('refusalbench_perturbation_metadata', {}).get('perturbation_class'),
            'intensity': entry.get('refusalbench_perturbation_metadata', {}).get('intensity'),
            'expected_rag_behavior': entry.get('refusalbench_perturbation_metadata', {}).get('expected_rag_behavior'),
            'generation_successful': entry.get('refusalbench_perturbation_metadata', {}).get('generation_successful'),
            'lever_selected': entry.get('refusalbench_perturbation_metadata', {}).get('lever_selected'),
            'source_qid': entry.get('refusalbench_perturbation_metadata', {}).get('source_qid'),
            
            # refusalbench_stratified_metadata fields  
            'generator_model': entry.get('refusalbench_stratified_metadata', {}).get('generator_model'),
            'generator_model_full': entry.get('refusalbench_stratified_metadata', {}).get('generator_model_full'),
            'source_folder': entry.get('refusalbench_stratified_metadata', {}).get('source_folder'),
            'unique_id': entry.get('refusalbench_stratified_metadata', {}).get('unique_id'),
            'stratum_key': entry.get('refusalbench_stratified_metadata', {}).get('stratum_key')
        }
        
        return field_locations.get(field_name)

    def load_dataset(self, jsonl_file: str) -> List[Dict[str, Any]]:
        """Load the RefusalBench stratified dataset."""
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                dataset = [json.loads(line) for line in f]
            print(f"✅ Loaded {len(dataset)} examples from {jsonl_file}")
            
            # Verify data structure with first example
            if dataset:
                first_example = dataset[0]
                print(f"🔍 Data structure verification:")
                
                # Check key fields
                expected_behavior = self.extract_field_from_entry(first_example, 'expected_rag_behavior')
                perturbation_class = self.extract_field_from_entry(first_example, 'perturbation_class')
                intensity = self.extract_field_from_entry(first_example, 'intensity')
                generator_model = self.extract_field_from_entry(first_example, 'generator_model')
                
                print(f"   Expected behavior: {expected_behavior}")
                print(f"   Perturbation class: {perturbation_class}")
                print(f"   Intensity: {intensity}")
                print(f"   Generator model: {generator_model}")
                
                # Check grounding structure
                grounding = first_example.get('grounding', [])
                if grounding:
                    first_passage = grounding[0]
                    cite_keys = [k for k in first_passage.keys() if k.startswith('cite_')]
                    print(f"   Grounding passages: {len(grounding)}")
                    print(f"   First passage cite keys: {cite_keys}")
                
                # Check evidence labels
                evidence_labels = first_example.get('evidence_correct', [])
                relevant_count = sum(1 for label in evidence_labels if label == "ANSWER-THE-QUESTION")
                print(f"   Evidence labels: {len(evidence_labels)} total, {relevant_count} relevant")
            
            # Analyze dataset composition using precise field extraction
            answerable_count = 0
            for item in dataset:
                expected_behavior = self.extract_field_from_entry(item, 'expected_rag_behavior')
                if expected_behavior == 'ANSWER_CORRECTLY':
                    answerable_count += 1
            
            unanswerable_count = len(dataset) - answerable_count
            
            print(f"📊 Dataset composition:")
            print(f"   Answerable (GaRAGe metrics): {answerable_count}")
            print(f"   Unanswerable (Refusal metrics): {unanswerable_count}")
                    
            return dataset
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return []

    @retry(
        wait=wait_random_exponential(min=2, max=30),
        stop=stop_after_attempt(100),
    )
    async def call_model_async(self, model_id: str, prompt: str, temperature: float = 0.1) -> str:
        """Call model via LiteLLM asynchronously with retry logic."""
        async with self.semaphore:
            try:
                # Add random delay to avoid rate limits
                # await asyncio.sleep(random.uniform(0.1, 0.5))
                
                response = await litellm.acompletion(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=temperature,
                    # max_tokens=1024,
                    # timeout=60
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling {model_id}: {e}")
                raise e

    def format_search_results(self, grounding: List[Dict[str, Any]]) -> str:
        """
        Format grounding into search results string for RAG prompt.
        PRECISE formatting based on your exact data structure.
        """
        if not grounding:
            return "No search results provided."
            
        formatted_passages = []
        for i, passage in enumerate(grounding):
            cite_marker = f"[{i+1}]"
            
            if isinstance(passage, dict):
                # Your data structure: each passage has cite_X key with the actual content
                text = None
                
                # Look for cite_X field (e.g., cite_1, cite_2, etc.)
                for key, value in passage.items():
                    if key.startswith('cite_') and isinstance(value, str):
                        text = value.strip()
                        break
                
                if text is None:
                    text = f"No cite field found in passage {i+1}"
            elif isinstance(passage, str):
                text = passage.strip()
            else:
                text = str(passage)
            
            # Add citation marker and text
            formatted_passages.append(f"{cite_marker} {text}")
        
        return "\n\n".join(formatted_passages)

    async def get_model_answer_merged_prompt(self, model_id: str, entry: Dict[str, Any]) -> str:
        """Get model answer using hybrid RefusalBench + GaRAGe prompt with localized ambiguity detection."""
        prompt_template = """You are an advanced AI assistant designed to answer questions based **only** on the provided search results. You must not use any external knowledge or prior training. Your goal is to be precise, factual, and aware of the limitations of the provided information.

Read the following Query and Search Results carefully.

QUERY:
{query}

SEARCH RESULTS:
{search_results}

**CRITICAL: EXAMINE QUERY AND PASSAGES FIRST**
Before attempting to answer, carefully examine BOTH the query and each search result passage for linguistic problems that would make providing a reliable answer impossible. Do NOT ignore or filter out problematic language.

**INSTRUCTIONS:**
1. Your primary task is to answer the QUERY if, and only if, a complete and faithful answer can be derived *solely* from the facts within the SEARCH RESULTS.
2. Place citations as inline markers (e.g., [1], [2], etc.) directly within the relevant parts of the response text.
3. You are permitted to perform reasoning, including common-sense inference and basic mathematics, as long as it operates *exclusively* on the information provided in the search results.
4. If a direct and faithful answer is not possible due to one of the issues below, you MUST respond with **only** the corresponding refusal code. Do not add any other text or explanation.

**REFUSAL CODES:**

**`REFUSE_AMBIGUOUS_QUERY`**
- **When to use:** When ambiguity (in the query or relevant passages) creates sufficient uncertainty that prevents confident answering. Does this ambiguity make you uncertain about what's being asked or what the information means? Refuse if the ambiguity prevents reliable, confident answering - even if one interpretation seems more likely or has more supporting information. Answer if the ambiguity is minor or can be clearly resolved by the context, allowing confident answering.
- **Example:** Query "What is the bat's weight?" with context mentioning both baseball bats and flying bats - the query ambiguity makes it unclear what the user wants, so refuse regardless of which interpretation has more detailed information.

**`REFUSE_CONTRADICTORY_CONTEXT`**
- **When to use:** When the search results contain directly contradictory or logically inconsistent information that is essential for answering the query.
- **Example:** One result states "revenue was $10M" and another states "revenue was $12M" for the same period.

**`REFUSE_INFO_MISSING_IN_CONTEXT`**
- **When to use:** When the search results are **silent** on the matter. The specific fact, entity, or value needed to answer is **entirely absent**. Use this when the information is simply not there.
- **Example:** Query asks "Who is the CEO?" but the search results only describe the company's products.

**`REFUSE_FALSE_PREMISE_IN_QUERY`**
- **When to use:** When the query assumes a fact (e.g., an event that never happened, a non-existent object) that is contradicted by or unsupported by the search results.
- **Example:** Query asks "Why did the company's Mars division fail?" but the search results state the company has no Mars division.

**`REFUSE_GRANULARITY_MISMATCH`**
- **When to use:** When the search results **contain relevant information**, but it is at the **wrong scale or level of abstraction**. Answering would require a transformation (e.g., aggregation, averaging, generalization) that is impossible or invalid with the given data. This is different from `REFUSE_INFO_MISSING_IN_CONTEXT` because the information is not absent, just incompatible.
- **Example:** Query asks for the "average employee salary," but the search results only provide individual salaries for two of the fifty employees. The information is present, but insufficient for the requested aggregation.

**`REFUSE_NONFACTUAL_QUERY`**
- **When to use:** When the query asks for subjective opinions, predictions, hypotheticals, or moral judgments that cannot be objectively derived from the provided facts.
- **Example:** Query asks "Is this a good investment?" but the search results only provide the stock price.

**`REFUSE_OTHER`**
- **When to use:** For any other specific issue not covered above that makes a faithful answer impossible.

Provide your response below:
ANSWER:"""

        query = entry.get('question', '')
        search_results = self.format_search_results(entry.get('grounding', []))
        
        prompt = prompt_template.format(
            query=query,
            search_results=search_results
        )
        return await self.call_model_async(model_id, prompt, temperature=0.1)

    def get_relevant_passages(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relevant passages for RAF metric using evidence_correct labels.
        PRECISE extraction based on your exact data structure.
        """
        all_passages = entry.get('grounding', [])
        evidence_labels = entry.get('evidence_correct', [])
        
        # Your data structure: evidence_correct contains labels like "ANSWER-THE-QUESTION", "OUTDATED", etc.
        # Only passages labeled "ANSWER-THE-QUESTION" are relevant for RAF metric
        relevant_passages = []
        for i, passage in enumerate(all_passages):
            if i < len(evidence_labels) and evidence_labels[i] == "ANSWER-THE-QUESTION":
                relevant_passages.append(passage)
        
        return relevant_passages

    async def comprehensive_judge(self, model_answer: str, entry: Dict[str, Any], is_answerable: bool, expected_behavior: str) -> Dict[str, Any]:
        """Single comprehensive judge that evaluates all metrics in one call."""
        
        query = entry.get('question', '')
        human_answer = entry.get('answer_generate', '')
        all_passages = entry.get('grounding', [])
        relevant_passages = self.get_relevant_passages(entry)
        
        # Format contexts for the judge
        all_context = self.format_search_results(all_passages)
        relevant_context = self.format_search_results(relevant_passages) if relevant_passages else "No relevant passages identified"
        
        if is_answerable:
            # Comprehensive prompt for answerable instances
            prompt = f"""You are an expert evaluator. Analyze this model response to a query and provide comprehensive evaluation scores.

QUERY:
{query}

HUMAN REFERENCE ANSWER (Gold Standard):
{human_answer}

MODEL RESPONSE:
{model_answer}

ALL CONTEXT PASSAGES:
{all_context}

RELEVANT CONTEXT PASSAGES (for RAF evaluation):
{relevant_context}

EVALUATION TASKS:
1. CLASSIFICATION: Is this an answer attempt or a refusal?
   - "answer_attempt" if model tries to answer the query
   - One of these refusal codes if model refuses: REFUSE_AMBIGUOUS_QUERY, REFUSE_CONTRADICTORY_CONTEXT, REFUSE_INFO_MISSING_IN_CONTEXT, REFUSE_FALSE_PREMISE_IN_QUERY, REFUSE_GRANULARITY_MISMATCH, REFUSE_NONFACTUAL_QUERY, REFUSE_OTHER

2. ELIGIBILITY (if answer attempt): Does the model answer sufficiently address the user's intent?
   - 1 = No major issues (addresses the main point of the question)
   - 0 = Major issues (misunderstands intent, irrelevant answer, or fails to address main point)

3. FACTUALITY_ALL (if answer attempt): Is every factual claim in the model answer fully supported by ALL context passages?
   - 1 = Every claim is supported
   - 0 = At least one claim is not supported

4. FACTUALITY_RELEVANT (if answer attempt): Is every factual claim in the model answer fully supported by RELEVANT context passages only?
   - 1 = Every claim is supported by relevant passages
   - 0 = At least one claim is not supported by relevant passages

Respond in this exact format:
CLASSIFICATION: [answer_attempt OR refusal_code]
ELIGIBILITY: [1 or 0, or N/A if refusal]
FACTUALITY_ALL: [1 or 0, or N/A if refusal]
FACTUALITY_RELEVANT: [1 or 0, or N/A if refusal]
EXPLANATION: [Brief reasoning for your scores]"""
        else:
            # Comprehensive prompt for unanswerable instances
            prompt = f"""You are an expert evaluator. Analyze this model response to determine if it correctly refuses to answer an unanswerable query.

QUERY:
{query}

EXPECTED BEHAVIOR: {expected_behavior}

MODEL RESPONSE:
{model_answer}

CONTEXT PASSAGES:
{all_context}

EVALUATION TASKS:
1. CLASSIFICATION: What type of response is this?
   - "answer_attempt" if model tries to answer the query
   - One of these refusal codes if model refuses: REFUSE_AMBIGUOUS_QUERY, REFUSE_CONTRADICTORY_CONTEXT, REFUSE_INFO_MISSING_IN_CONTEXT, REFUSE_FALSE_PREMISE_IN_QUERY, REFUSE_GRANULARITY_MISMATCH, REFUSE_NONFACTUAL_QUERY, REFUSE_OTHER

2. REFUSAL_APPROPRIATENESS: Is the refusal appropriate and correct?
   - 1 = Correct refusal type matching expected behavior
   - 0 = Wrong refusal type or inappropriate refusal

Respond in this exact format:
CLASSIFICATION: [answer_attempt OR refusal_code]
REFUSAL_APPROPRIATENESS: [1 or 0, or N/A if answer_attempt]
EXPLANATION: [Brief reasoning for your classification and appropriateness score]"""
        
        try:
            response = await self.call_model_async(self.judge_model_id, prompt, temperature=0.0)
            return self.parse_comprehensive_judgment(response, is_answerable, expected_behavior)
        except Exception as e:
            print(f"Error in comprehensive judgment: {e}")
            return self.create_error_judgment(is_answerable)

    def parse_comprehensive_judgment(self, response: str, is_answerable: bool, expected_behavior: str) -> Dict[str, Any]:
        """Parse the comprehensive judge response."""
        try:
            lines = response.strip().split('\n')
            
            # Extract classification
            classification_line = next((line for line in lines if line.startswith('CLASSIFICATION:')), '')
            classification = classification_line.split('CLASSIFICATION:')[1].strip() if classification_line else "answer_attempt"
            
            # Extract explanation
            explanation_line = next((line for line in lines if line.startswith('EXPLANATION:')), '')
            explanation = explanation_line.split('EXPLANATION:')[1].strip() if explanation_line else "No explanation provided"
            
            result = {
                'classification': classification,
                'explanation': explanation
            }
            
            if is_answerable:
                # Parse answerable instance metrics
                eligibility_line = next((line for line in lines if line.startswith('ELIGIBILITY:')), '')
                factuality_all_line = next((line for line in lines if line.startswith('FACTUALITY_ALL:')), '')
                factuality_rel_line = next((line for line in lines if line.startswith('FACTUALITY_RELEVANT:')), '')
                
                # Handle N/A values and convert to appropriate types
                def parse_score(line, prefix):
                    if line:
                        value = line.split(f'{prefix}:')[1].strip()
                        if value.upper() == 'N/A':
                            return None
                        try:
                            return int(value)
                        except:
                            return 0
                    return 0
                
                result.update({
                    'eligibility': parse_score(eligibility_line, 'ELIGIBILITY'),
                    'factuality_all': parse_score(factuality_all_line, 'FACTUALITY_ALL'),
                    'factuality_relevant': parse_score(factuality_rel_line, 'FACTUALITY_RELEVANT')
                })
                
                # If model refused when should answer, set scores to 0
                if classification != "answer_attempt":
                    result.update({
                        'eligibility': 0,
                        'factuality_all': 0,
                        'factuality_relevant': 0
                    })
                
            else:
                # Parse unanswerable instance metrics
                appropriateness_line = next((line for line in lines if line.startswith('REFUSAL_APPROPRIATENESS:')), '')
                
                if appropriateness_line:
                    value = appropriateness_line.split('REFUSAL_APPROPRIATENESS:')[1].strip()
                    if value.upper() == 'N/A':
                        # Model answered when should refuse
                        result['refusal_appropriateness'] = 0
                    else:
                        try:
                            result['refusal_appropriateness'] = int(value)
                        except:
                            result['refusal_appropriateness'] = 0
                else:
                    result['refusal_appropriateness'] = 0
                
                # Special handling: if model answered when should refuse
                if classification == "answer_attempt":
                    result['refusal_appropriateness'] = 0  # Missed refusal
                elif classification == expected_behavior:
                    result['refusal_appropriateness'] = 1  # Correct refusal
                else:
                    result['refusal_appropriateness'] = 0  # Wrong refusal type
            
            return result
            
        except Exception as e:
            print(f"Error parsing comprehensive judgment: {e}")
            return self.create_error_judgment(is_answerable)

    def create_error_judgment(self, is_answerable: bool) -> Dict[str, Any]:
        """Create error judgment when parsing fails."""
        result = {
            'classification': 'answer_attempt',  # Default assumption
            'explanation': 'Error in judgment parsing'
        }
        
        if is_answerable:
            result.update({
                'eligibility': 0,
                'factuality_all': 0,
                'factuality_relevant': 0
            })
        else:
            result['refusal_appropriateness'] = 0
        
        return result

    def evaluate_refusal_match(self, predicted_refusal_code: str, ground_truth_refusal_code: str) -> Tuple[bool, str]:
        """Evaluate refusal category match (exact match)."""
        if predicted_refusal_code == ground_truth_refusal_code:
            return True, f"Exact match: {predicted_refusal_code}"
        else:
            return False, f"Mismatch: predicted={predicted_refusal_code}, ground_truth={ground_truth_refusal_code}"

    async def process_example(self, entry: Dict[str, Any], entry_idx: int) -> List[Dict[str, Any]]:
        """Process one example across all models using hybrid evaluation logic."""
        results = []
        
        # Extract key information using precise field extraction
        query = entry.get('question', '')
        
        # FIXED: Create unique sample ID using original_id + entry_idx
        original_sample_id = entry.get('sample_id', 'unknown')
        sample_id = f"{original_sample_id}_{entry_idx}"
        
        human_answer = entry.get('answer_generate', '')
        
        # Extract RefusalBench metadata
        expected_behavior = self.extract_field_from_entry(entry, 'expected_rag_behavior') or 'ANSWER_CORRECTLY'
        perturbation_class = self.extract_field_from_entry(entry, 'perturbation_class') or 'unknown'
        intensity = self.extract_field_from_entry(entry, 'intensity') or 'unknown'
        generator_model = self.extract_field_from_entry(entry, 'generator_model') or 'unknown'
        lever_selected = self.extract_field_from_entry(entry, 'lever_selected') or 'unknown'
        
        # Extract domain/topic metadata for analysis
        question_category = self.extract_field_from_entry(entry, 'question_category') or 'unknown'
        topic_tag = self.extract_field_from_entry(entry, 'topic_tag') or 'unknown'
        question_tag = self.extract_field_from_entry(entry, 'question_tag') or 'unknown'
        question_type = self.extract_field_from_entry(entry, 'question_type') or 'unknown'
        question_complexity = self.extract_field_from_entry(entry, 'question_complexity') or 'unknown'
        question_popularity = self.extract_field_from_entry(entry, 'question_popularity') or 'unknown'
        source_dataset = self.extract_field_from_entry(entry, 'source_dataset') or 'unknown'
        
        # Determine evaluation strategy
        is_answerable = (expected_behavior == "ANSWER_CORRECTLY")
        
        print(f"🔍 Processing {sample_id}: {'ANSWERABLE' if is_answerable else 'UNANSWERABLE'} ({expected_behavior})")
        print(f"   📂 Domain: {question_category} | Topic: {topic_tag} | Type: {question_type}")
        
        # 🔄 RESUME: Check which models still need evaluation for this example
        models_to_evaluate = []
        for model_id in self.model_ids:
            if not self.is_evaluation_completed(sample_id, model_id):
                models_to_evaluate.append(model_id)
            else:
                print(f"   ⏭️  {model_id}: Already completed (skipping)")
        
        if not models_to_evaluate:
            print(f"   ✅ All models completed for {sample_id}")
            return []
        
        for model_id in models_to_evaluate:
            try:
                # Get model prediction using merged GaRAGe + RefusalBench prompt
                model_answer = await self.get_model_answer_merged_prompt(model_id, entry)
                
                # Use comprehensive judge to evaluate the response
                judgment = await self.comprehensive_judge(model_answer, entry, is_answerable, expected_behavior)
                
                # Initialize result structure with ALL metadata for analysis
                result = {
                    # Core identification
                    'model_id': model_id,
                    'entry_idx': entry_idx,
                    'sample_id': sample_id,  # Now guaranteed unique
                    'original_sample_id': original_sample_id,  # Keep original for reference
                    'query': query,
                    'model_answer': model_answer,
                    'judge_explanation': judgment.get('explanation', ''),
                    
                    # RefusalBench metadata
                    'expected_behavior': expected_behavior,
                    'perturbation_class': perturbation_class,
                    'intensity': intensity,
                    'generator_model': generator_model,
                    'lever_selected': lever_selected,
                    'is_answerable': is_answerable,
                    
                    # DOMAIN/TOPIC METADATA (for analysis)
                    'question_category': question_category,      # Primary domain (Finance, Science, etc.)
                    'topic_tag': topic_tag,                      # Specific topic (arxiv, web, etc.)
                    'question_tag': question_tag,                # Source type (web, ent, etc.)
                    'question_type': question_type,              # SLOW-CHANGING, FAST-CHANGING, etc.
                    'question_complexity': question_complexity,  # Simple, Comparison, etc.
                    'question_popularity': question_popularity,  # Head, Torso, Tail
                    'source_dataset': source_dataset,            # GaRAGe, etc.
                    
                    # Classification results
                    'refusal_classification': judgment.get('classification', 'answer_attempt'),
                    
                    # Initialize all possible metrics
                    'eligibility_score': None,
                    'unadjusted_factuality_score': None,
                    'raf_score': None,
                    'refusal_match_correct': None,
                    'false_refusal': None,
                    'missed_refusal': None
                }
                
                if is_answerable:
                    # ANSWERABLE INSTANCE: Should answer correctly, evaluate with GaRAGe metrics
                    print(f"  📊 {model_id}: Should answer (GaRAGe metrics)...")
                    
                    if judgment.get('classification') == "answer_attempt":
                        # Model correctly attempted to answer - use judge scores
                        result.update({
                            'eligibility_score': judgment.get('eligibility', 0),
                            'unadjusted_factuality_score': judgment.get('factuality_all', 0),
                            'raf_score': judgment.get('factuality_relevant', 0),
                            'false_refusal': False
                        })
                        
                        print(f"    ✅ Answer quality: E={result['eligibility_score']}, F={result['unadjusted_factuality_score']}, RAF={result['raf_score']}")
                    else:
                        # Model incorrectly refused when should have answered - False Refusal
                        result.update({
                            'eligibility_score': 0,  # Failed to address query
                            'unadjusted_factuality_score': 0,
                            'raf_score': 0,
                            'false_refusal': True
                        })
                        print(f"    ❌ False refusal: refused with {judgment.get('classification')} when should answer")
                    
                else:
                    # UNANSWERABLE INSTANCE: Should refuse, evaluate with refusal metrics
                    print(f"  🚫 {model_id}: Should refuse ({expected_behavior})...")
                    
                    if judgment.get('classification') == "answer_attempt":
                        # Model answered when should have refused - Missed Refusal
                        result.update({
                            'refusal_match_correct': False,
                            'missed_refusal': True
                        })
                        print(f"    ❌ Missed refusal: answered when should refuse with {expected_behavior}")
                    else:
                        # Model refused - check if correct refusal type
                        is_correct = judgment.get('refusal_appropriateness', 0) == 1
                        result.update({
                            'refusal_match_correct': is_correct,
                            'missed_refusal': False
                        })
                        print(f"    {'✅' if is_correct else '⚠️'} Refusal: {judgment.get('classification')} vs {expected_behavior}")
                
                results.append(result)
                
            except Exception as e:
                print(f"  ❌ Error evaluating {model_id}: {e}")
                # Add error result with all metadata
                results.append({
                    'model_id': model_id,
                    'entry_idx': entry_idx,
                    'sample_id': sample_id,  # Now guaranteed unique
                    'original_sample_id': original_sample_id,
                    'query': query,
                    'expected_behavior': expected_behavior,
                    'perturbation_class': perturbation_class,
                    'intensity': intensity,
                    'generator_model': generator_model,
                    'lever_selected': lever_selected,
                    'question_category': question_category,
                    'topic_tag': topic_tag,
                    'question_tag': question_tag,
                    'question_type': question_type,
                    'question_complexity': question_complexity,
                    'question_popularity': question_popularity,
                    'source_dataset': source_dataset,
                    'model_answer': f"ERROR: {str(e)}",
                    'is_answerable': is_answerable,
                    'judge_explanation': 'Error occurred',
                    'eligibility_score': 0 if is_answerable else None,
                    'unadjusted_factuality_score': 0 if is_answerable else None,
                    'raf_score': 0 if is_answerable else None,
                    'refusal_classification': 'ERROR',
                    'refusal_match_correct': False if not is_answerable else None,
                    'false_refusal': None,
                    'missed_refusal': None
                })
        
        return results

    async def evaluate_dataset(self, dataset: List[Dict[str, Any]], num_examples: int = None) -> pd.DataFrame:
        """Evaluate models on dataset using batched async processing with resume functionality."""
        if num_examples is not None and num_examples < len(dataset):
            dataset = dataset[:num_examples]
        
        # 🔄 RESUME: Load existing results first
        self.load_existing_results()
        
        print(f"🚀 Evaluating {len(self.model_ids)} models on {len(dataset)} examples")
        print(f"📊 Total evaluations: {len(dataset) * len(self.model_ids)}")
        
        # 🔄 RESUME: Calculate remaining work
        total_needed = len(dataset) * len(self.model_ids)
        already_completed = len(self.completed_evaluations)
        remaining_work = total_needed - already_completed
        
        print(f"🔄 RESUME STATUS:")
        print(f"   Already completed: {already_completed}/{total_needed} evaluations")
        print(f"   Remaining work: {remaining_work} evaluations")
        
        if remaining_work == 0:
            print("✅ All evaluations already completed!")
            return self.existing_results_df if self.existing_results_df is not None else pd.DataFrame()
        
        all_results = []
        
        # Process in batches with resume functionality
        batch_count = 0
        total_batches = (len(dataset) - 1) // self.batch_size + 1
        
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i+self.batch_size]
            batch_count += 1
            
            # 🔄 RESUME: Check if any work needed in this batch
            batch_has_work = False
            for idx, entry in enumerate(batch):
                original_sample_id = entry.get('sample_id', 'unknown')
                sample_id = f"{original_sample_id}_{i+idx}"  # Use same logic as process_example
                for model_id in self.model_ids:
                    if not self.is_evaluation_completed(sample_id, model_id):
                        batch_has_work = True
                        break
                if batch_has_work:
                    break
            
            if not batch_has_work:
                print(f"⏭️  Batch {batch_count}/{total_batches}: All completed (skipping)")
                continue
            
            print(f"🔄 Processing batch {batch_count}/{total_batches} (with resume)")
            
            batch_tasks = [
                self.process_example(entry, i+idx) 
                for idx, entry in enumerate(batch)
            ]
            
            # Wait for all tasks in the batch to complete
            batch_results = await atqdm.gather(
                *batch_tasks,
                desc=f"Batch {batch_count}/{total_batches}"
            )
            
            # Flatten results and save incrementally
            batch_new_results = []
            for result_list in batch_results:
                batch_new_results.extend(result_list)
            
            if batch_new_results:
                all_results.extend(batch_new_results)
                
                # 🔄 RESUME: Save after each batch
                batch_df = pd.DataFrame(batch_new_results)
                self.save_incremental_results(batch_df)
                
                print(f"💾 Batch {batch_count} completed and saved")
        
        # Return combined results
        if self.existing_results_df is not None:
            return self.existing_results_df
        else:
            return pd.DataFrame(all_results)

    def compute_metrics(self, results_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute both GaRAGe and RefusalBench metrics by model with domain breakdowns."""
        metrics = {}
        
        for model_id in results_df['model_id'].unique():
            model_df = results_df[results_df['model_id'] == model_id].copy()
            model_metrics = {}
            
            # Split into answerable and unanswerable instances
            answerable_df = model_df[model_df['is_answerable'] == True]
            unanswerable_df = model_df[model_df['is_answerable'] == False]
            
            # === GaRAGe METRICS (Answerable Instances) ===
            if len(answerable_df) > 0:
                model_metrics['garage_eligibility'] = answerable_df['eligibility_score'].mean()
                model_metrics['garage_unadjusted_factuality'] = answerable_df['unadjusted_factuality_score'].mean()
                model_metrics['garage_raf'] = answerable_df['raf_score'].mean()
                model_metrics['false_refusal_rate'] = answerable_df['false_refusal'].mean()
                model_metrics['num_answerable'] = len(answerable_df)
            else:
                model_metrics.update({
                    'garage_eligibility': 0.0,
                    'garage_unadjusted_factuality': 0.0,
                    'garage_raf': 0.0,
                    'false_refusal_rate': 0.0,
                    'num_answerable': 0
                })
            
            # === RefusalBench METRICS (Unanswerable Instances) ===
            if len(unanswerable_df) > 0:
                model_metrics['refusal_accuracy'] = unanswerable_df['refusal_match_correct'].mean()
                model_metrics['missed_refusal_rate'] = unanswerable_df['missed_refusal'].mean()
                
                # Overall refusal rate (any refusal attempt)
                any_refusal = len(unanswerable_df[unanswerable_df['refusal_classification'] != 'answer_attempt'])
                model_metrics['overall_refusal_rate'] = any_refusal / len(unanswerable_df)
                model_metrics['num_unanswerable'] = len(unanswerable_df)
            else:
                model_metrics.update({
                    'refusal_accuracy': 0.0,
                    'missed_refusal_rate': 0.0,
                    'overall_refusal_rate': 0.0,
                    'num_unanswerable': 0
                })
            
            # === COMPOSITE METRICS ===
            # Weighted combination of GaRAGe RAF and refusal accuracy
            if model_metrics['num_answerable'] > 0 and model_metrics['num_unanswerable'] > 0:
                w_garage = model_metrics['num_answerable'] / (model_metrics['num_answerable'] + model_metrics['num_unanswerable'])
                w_refusal = model_metrics['num_unanswerable'] / (model_metrics['num_answerable'] + model_metrics['num_unanswerable'])
                
                model_metrics['hybrid_score'] = (
                    w_garage * model_metrics['garage_raf'] + 
                    w_refusal * model_metrics['refusal_accuracy']
                )
            elif model_metrics['num_answerable'] > 0:
                model_metrics['hybrid_score'] = model_metrics['garage_raf']
            elif model_metrics['num_unanswerable'] > 0:
                model_metrics['hybrid_score'] = model_metrics['refusal_accuracy']
            else:
                model_metrics['hybrid_score'] = 0.0
            
            # === PERTURBATION-SPECIFIC METRICS ===
            model_metrics['perturbation_breakdown'] = {}
            for perturb_class in model_df['perturbation_class'].unique():
                class_df = model_df[model_df['perturbation_class'] == perturb_class]
                answerable_class = class_df[class_df['is_answerable'] == True]
                unanswerable_class = class_df[class_df['is_answerable'] == False]
                
                class_metrics = {
                    'total_count': len(class_df),
                    'answerable_count': len(answerable_class),
                    'unanswerable_count': len(unanswerable_class)
                }
                
                if len(answerable_class) > 0:
                    class_metrics['garage_raf'] = answerable_class['raf_score'].mean()
                if len(unanswerable_class) > 0:
                    class_metrics['refusal_accuracy'] = unanswerable_class['refusal_match_correct'].mean()
                
                model_metrics['perturbation_breakdown'][perturb_class] = class_metrics
            
            # === DOMAIN/TOPIC BREAKDOWNS (NEW) ===
            # By question category (Finance, Science, etc.)
            model_metrics['domain_breakdown'] = {}
            for domain in model_df['question_category'].unique():
                domain_df = model_df[model_df['question_category'] == domain]
                answerable_domain = domain_df[domain_df['is_answerable'] == True]
                unanswerable_domain = domain_df[domain_df['is_answerable'] == False]
                
                domain_metrics = {
                    'total_count': len(domain_df),
                    'answerable_count': len(answerable_domain),
                    'unanswerable_count': len(unanswerable_domain)
                }
                
                if len(answerable_domain) > 0:
                    domain_metrics['garage_raf'] = answerable_domain['raf_score'].mean()
                    domain_metrics['garage_eligibility'] = answerable_domain['eligibility_score'].mean()
                if len(unanswerable_domain) > 0:
                    domain_metrics['refusal_accuracy'] = unanswerable_domain['refusal_match_correct'].mean()
                
                model_metrics['domain_breakdown'][domain] = domain_metrics
            
            # By question type (SLOW-CHANGING, etc.)
            model_metrics['question_type_breakdown'] = {}
            for qtype in model_df['question_type'].unique():
                type_df = model_df[model_df['question_type'] == qtype]
                answerable_type = type_df[type_df['is_answerable'] == True]
                unanswerable_type = type_df[type_df['is_answerable'] == False]
                
                type_metrics = {
                    'total_count': len(type_df),
                    'answerable_count': len(answerable_type),
                    'unanswerable_count': len(unanswerable_type)
                }
                
                if len(answerable_type) > 0:
                    type_metrics['garage_raf'] = answerable_type['raf_score'].mean()
                if len(unanswerable_type) > 0:
                    type_metrics['refusal_accuracy'] = unanswerable_type['refusal_match_correct'].mean()
                
                model_metrics['question_type_breakdown'][qtype] = type_metrics
            
            # By question complexity
            model_metrics['complexity_breakdown'] = {}
            for complexity in model_df['question_complexity'].unique():
                complexity_df = model_df[model_df['question_complexity'] == complexity]
                answerable_complexity = complexity_df[complexity_df['is_answerable'] == True]
                unanswerable_complexity = complexity_df[complexity_df['is_answerable'] == False]
                
                complexity_metrics = {
                    'total_count': len(complexity_df),
                    'answerable_count': len(answerable_complexity),
                    'unanswerable_count': len(unanswerable_complexity)
                }
                
                if len(answerable_complexity) > 0:
                    complexity_metrics['garage_raf'] = answerable_complexity['raf_score'].mean()
                if len(unanswerable_complexity) > 0:
                    complexity_metrics['refusal_accuracy'] = unanswerable_complexity['refusal_match_correct'].mean()
                
                model_metrics['complexity_breakdown'][complexity] = complexity_metrics
            
            metrics[model_id] = model_metrics
        
        return metrics

    def save_results(self, results_df: pd.DataFrame, metrics: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
        """Save results and metrics to files."""
        # Save raw results (already saved incrementally, but final save for consistency)
        results_df.to_csv(self.results_file, index=False)
        print(f"💾 Saved final results to {self.results_file}")
        
        # Save metrics
        metrics_file = self.output_dir / "hybrid_evaluation_metrics.json"
        
        # Convert to JSON-serializable format
        json_metrics = {}
        for model_id, model_metrics in metrics.items():
            json_metrics[model_id] = {}
            for key, value in model_metrics.items():
                if key != 'perturbation_breakdown':
                    json_metrics[model_id][key] = value
                else:
                    json_metrics[model_id][key] = value
        
        # Add metadata
        json_metrics['_metadata'] = {
            'evaluation_timestamp': datetime.datetime.now().isoformat(),
            'total_examples': len(results_df) // len(self.model_ids) if len(self.model_ids) > 0 else 0,
            'models_evaluated': self.model_ids,
            'judge_model': self.judge_model_id,
            'resume_functionality': True,
            'incremental_saves': True,
            'unique_sample_id_format': 'original_sample_id_entry_idx'
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        print(f"💾 Saved metrics to {metrics_file}")
        
        return str(self.results_file), str(metrics_file)

    def print_metrics_summary(self, metrics: Dict[str, Dict[str, Any]]):
        """Print formatted summary of hybrid metrics with domain analysis."""
        print("\n" + "="*80)
        print("HYBRID RefusalBench EVALUATION METRICS SUMMARY")
        print("="*80)
        
        for model_id, model_metrics in metrics.items():
            print(f"\nModel: {model_id}")
            print("-" * 60)
            
            print(f"📊 ANSWERABLE INSTANCES ({model_metrics['num_answerable']} examples) - GaRAGe Metrics:")
            print(f"  Eligibility Score:           {model_metrics['garage_eligibility']:.2%}")
            print(f"  Unadjusted Factuality:       {model_metrics['garage_unadjusted_factuality']:.2%}")
            print(f"  RAF Score:                   {model_metrics['garage_raf']:.2%}")
            print(f"  False Refusal Rate:          {model_metrics['false_refusal_rate']:.2%}")
            
            print(f"\n🚫 UNANSWERABLE INSTANCES ({model_metrics['num_unanswerable']} examples) - RefusalBench Metrics:")
            print(f"  Refusal Accuracy:            {model_metrics['refusal_accuracy']:.2%}")
            print(f"  Overall Refusal Rate:        {model_metrics['overall_refusal_rate']:.2%}")
            print(f"  Missed Refusal Rate:         {model_metrics['missed_refusal_rate']:.2%}")
            
            print(f"\n🔀 COMPOSITE METRICS:")
            print(f"  Hybrid Score:                {model_metrics['hybrid_score']:.2%}")
            
            # Show domain breakdown (NEW)
            domain_breakdown = model_metrics.get('domain_breakdown', {})
            if domain_breakdown:
                print(f"\n📂 DOMAIN BREAKDOWN (Top 3 by volume):")
                sorted_domains = sorted(domain_breakdown.items(), key=lambda x: x[1].get('total_count', 0), reverse=True)[:3]
                for domain_name, domain_metrics in sorted_domains:
                    raf = domain_metrics.get('garage_raf', 0)
                    ref_acc = domain_metrics.get('refusal_accuracy', 0)
                    print(f"  {domain_name}: RAF={raf:.2%}, RefAcc={ref_acc:.2%} ({domain_metrics['total_count']} examples)")
            
            # Show perturbation breakdown (top 3 most challenging)
            perturbation_breakdown = model_metrics.get('perturbation_breakdown', {})
            if perturbation_breakdown:
                print(f"\n🎯 PERTURBATION BREAKDOWN (Top 3 by volume):")
                sorted_classes = sorted(perturbation_breakdown.items(), key=lambda x: x[1].get('total_count', 0), reverse=True)[:3]
                for class_name, class_metrics in sorted_classes:
                    raf = class_metrics.get('garage_raf', 0)
                    ref_acc = class_metrics.get('refusal_accuracy', 0)
                    print(f"  {class_name}: RAF={raf:.2%}, RefAcc={ref_acc:.2%} ({class_metrics['total_count']} examples)")
            
            # Show question type breakdown
            type_breakdown = model_metrics.get('question_type_breakdown', {})
            if type_breakdown:
                print(f"\n📋 QUESTION TYPE BREAKDOWN:")
                for qtype, type_metrics in sorted(type_breakdown.items()):
                    raf = type_metrics.get('garage_raf', 0)
                    ref_acc = type_metrics.get('refusal_accuracy', 0)
                    print(f"  {qtype}: RAF={raf:.2%}, RefAcc={ref_acc:.2%} ({type_metrics['total_count']} examples)")
        
        print("\n" + "="*80)

    async def run_evaluation(self, dataset_path: str, num_examples: int = None):
        """Run the full hybrid evaluation workflow with resume functionality."""
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        if not dataset:
            print("❌ No dataset loaded, aborting evaluation")
            return None
        
        # Run evaluation with resume functionality
        results_df = await self.evaluate_dataset(dataset, num_examples)
        
        # Compute metrics
        metrics = self.compute_metrics(results_df)
        
        # Print summary
        self.print_metrics_summary(metrics)
        
        # Save results (final save)
        results_file, metrics_file = self.save_results(results_df, metrics)
        
        print(f"\n🔄 RESUME FUNCTIONALITY SUMMARY:")
        print(f"   ✅ Incremental saving after each batch")
        print(f"   ✅ Automatic resume detection on restart")
        print(f"   ✅ Skips completed (sample_id, model_id) pairs")
        print(f"   ✅ Robust against crashes and interruptions")
        print(f"   🆔 FIXED: Unique sample IDs using original_id + entry_idx")
        
        return {
            'results_df': results_df,
            'metrics': metrics,
            'results_file': results_file,
            'metrics_file': metrics_file
        }

async def main():
    """Main evaluation function."""
    # Models to evaluate (standard RAG models)
    models_to_evaluate = [
        "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/us.anthropic.claude-opus-4-20250514-v1:0",
        "bedrock/us.amazon.nova-pro-v1:0",  
        "bedrock/converse/us.amazon.nova-premier-v1:0",
        "openai/gpt-4o-2024-08-06", 
        "openai/gpt-4.1-2025-04-14",
        "openai/o4-mini-2025-04-16",
        "bedrock/converse/us.deepseek.r1-v1:0"
    ]
    
    # For initial testing, you can comment out most models and start with just 2-3:
    # models_to_evaluate = [
    #     "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
    #     "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
    # ]
    
    # Judge model for evaluation
    judge_model = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
    
    # Path to the RefusalBench stratified dataset (output from your filter)
    dataset_path = "filtered_refusalbench_analysis/refusalbench_stratified_dataset_stratified_v1.jsonl"
    
    print("🔀 Hybrid RefusalBench Evaluation Starting...")
    print(f"📂 Dataset: {dataset_path}")
    print(f"🤖 Models: {len(models_to_evaluate)}")
    print(f"⚖️  Judge: {judge_model}")
    print(f"📋 Prompt: Merged GaRAGe + RefusalBench (best of both worlds)")
    print(f"📊 Metrics: Answerable→GaRAGe, Unanswerable→Refusal")
    print(f"🔗 Context: All passages concatenated (signal + noise)")
    print(f"🎯 PRECISE: Field extraction based on exact data structure")
    print(f"🔄 RESUME: Automatically continues from previous progress")
    print(f"🆔 FIXED: Unique sample IDs using original_id + entry_idx")
    
    evaluator = HybridRefusalBenchEvaluator(
        model_ids=models_to_evaluate,
        judge_model_id=judge_model,
        output_dir="./hybrid_refusalbench_results",
        batch_size=3,  # Small batch size for complex evaluation
        max_concurrent=8  # Conservative concurrency
    )
    
    # Run evaluation - now with resume functionality and unique sample IDs!
    # You can stop and restart this script and it will continue where it left off
    await evaluator.run_evaluation(dataset_path, num_examples=None)  # Use None for full dataset
    
    print("\n🎉 Hybrid evaluation completed!")
    print("🔄 Next time you run this script, it will automatically resume from where it left off!")
    print("🆔 Sample ID uniqueness issue fixed - no more collisions!")

if __name__ == "__main__":
    asyncio.run(main())
