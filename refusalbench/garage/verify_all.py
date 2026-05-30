#!/usr/bin/env python3
"""
Multi-Model RefusalBench Perturbation Verifier using LiteLLM.
Verifies that generated perturbations correctly implement the specified 
linguistic mechanisms and achieve the target intensity levels across multiple models and datasets.
UPDATED FOR NEW REVERSIBLE DATA FORMAT AND GARAGE SCHEMA.
"""
import litellm
import json
import os
import asyncio
from typing import Dict, List, Optional, Union, Set
from tqdm.asyncio import tqdm as atqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pathlib import Path
import argparse

# Import the RefusalBench catalogue
from garage_prompt_guidelines import RefusalBenchCatalogue

# Import configuration
try:
    from config import (
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME,
        OPENAI_API_KEY
    )
    # Set up OpenAI API key if available from config
    if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
except ImportError:
    print("Warning: config.py not found. Please copy config_template.py to config.py and fill in your credentials.")

class MultiModelAsyncRefusalBenchVerifier:
    """
    Multi-Model Async RefusalBench Perturbation Verifier using LiteLLM.
    
    Verifies generated perturbations using the official RefusalBench lever catalogue
    across multiple models and datasets with resume functionality.
    UPDATED FOR NEW REVERSIBLE DATA FORMAT.
    """
    
    def __init__(self, 
                 region_name: str = 'us-east-1',
                 batch_size: int = 10,
                 max_concurrent: int = 20,
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        """Initialize the multi-model async perturbation verifier."""
        
        # Set up AWS environment variables
        os.environ["AWS_REGION_NAME"] = region_name
        
        # Model configuration
        self.model_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.99
        }
        
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        
        # Async processing setup
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize RefusalBench catalogue
        self.catalogue = RefusalBenchCatalogue()
        
        # Define available models and datasets
        self.models = [
            "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
            "bedrock/us.amazon.nova-pro-v1:0",  
            "openai/gpt-4o-2024-08-06", 
            "bedrock/converse/us.deepseek.r1-v1:0"
        ]
        
        self.datasets = [
            "dataset_claude", 
            "dataset_deepseek", 
            "dataset_nova", 
            "dataset_gpt"
        ]
        
        print(f"Initialized verifier with {len(self.catalogue.get_all_combinations())} perturbation combinations")
        print(f"Available models: {self.models}")
        print(f"Available datasets: {self.datasets}")

    def _extract_signal_noise_passages(self, perturbation_data: Dict) -> tuple:
        """
        Extract signal and noise passages from the new reversible data format.
        
        Returns:
            Tuple of (signal_passages, noise_passages)
        """
        try:
            original_garage_data = perturbation_data.get('original_garage_data', {})
            passage_mapping = perturbation_data.get('passage_mapping', {})
            
            grounding = original_garage_data.get('grounding', [])
            signal_indices = passage_mapping.get('signal_indices', [])
            noise_indices = passage_mapping.get('noise_indices', [])
            
            signal_passages = []
            noise_passages = []
            
            # Extract signal passages
            for idx in signal_indices:
                if idx < len(grounding):
                    passage_data = grounding[idx]
                    for key, value in passage_data.items():
                        if key.startswith('cite_') and isinstance(value, str):
                            signal_passages.append(value.strip())
                            break
            
            # Extract noise passages
            for idx in noise_indices:
                if idx < len(grounding):
                    passage_data = grounding[idx]
                    for key, value in passage_data.items():
                        if key.startswith('cite_') and isinstance(value, str):
                            noise_passages.append(value.strip())
                            break
            
            return signal_passages, noise_passages
            
        except Exception as e:
            print(f"Error extracting passages: {e}")
            return [], []

    def get_model_name_for_file(self, model_id: str) -> str:
        """Convert model ID to a clean filename suffix."""
        if "claude" in model_id:
            return "claude"
        elif "nova" in model_id:
            return "nova"
        elif "gpt" in model_id:
            return "gpt"
        elif "deepseek" in model_id:
            return "deepseek"
        else:
            # Fallback: use the last part after the last slash/colon
            return model_id.split("/")[-1].split(":")[0].replace(".", "_")

    def get_verification_file_path(self, dataset_folder: str, model_id: str) -> str:
        """Generate the output file path for verification results."""
        model_name = self.get_model_name_for_file(model_id)
        return f"{dataset_folder}/refusalbench_verifications_{model_name}.jsonl"

    def load_existing_verifications(self, output_file: str) -> Set[str]:
        """Load existing verification IDs to enable resume functionality."""
        existing_ids = set()
        
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        try:
                            data = json.loads(line.strip())
                            # Create a unique ID based on perturbation data
                            unique_id = self.create_perturbation_id(data)
                            existing_ids.add(unique_id)
                        except Exception as e:
                            print(f"Error reading existing verification line {line_num}: {e}")
                            continue
                            
                print(f"Found {len(existing_ids)} existing verifications in {output_file}")
            except Exception as e:
                print(f"Error loading existing verifications from {output_file}: {e}")
        
        return existing_ids

    def create_perturbation_id(self, perturbation_data: Dict) -> str:
        """Create a unique ID for a perturbation for tracking purposes."""
        try:
            # Extract key fields from new format
            original_garage_data = perturbation_data.get('original_garage_data', {})
            perturbation_metadata = perturbation_data.get('perturbation_metadata', {})
            
            original_query = original_garage_data.get('question', '')
            answer_generate_text = original_garage_data.get('answer_generate', '')
            perturbation_class = perturbation_metadata.get('perturbation_class', '')
            intensity = perturbation_metadata.get('intensity', '')
            source_qid = perturbation_metadata.get('source_qid', '')
            
            # Create unique identifier
            key_fields = [
                source_qid,
                perturbation_class,
                intensity,
                original_query[:50],  # First 50 chars of query
            ]
            return "|".join(key_fields)
        except Exception as e:
            print(f"Error creating perturbation ID: {e}")
            return "unknown_id"

    @retry(
        wait=wait_random_exponential(min=2, max=30),
        stop=stop_after_attempt(100),
    )
    async def _call_model_async(self, model_id: str, prompt: str) -> str:
        """Call specified model via LiteLLM asynchronously with retry logic."""
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
                print(f"Error calling {model_id}: {e}")
                raise e

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from model, handling common formatting issues."""
        try:
            # Remove markdown code blocks if present
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end != -1:
                    response = response[start:end].strip()
            
            # Parse JSON
            parsed = json.loads(response)
            parsed["parsing_successful"] = True
            return parsed
            
        except json.JSONDecodeError as e:
            return {
                "parsing_successful": False,
                "error": f"JSON parsing failed: {e}",
                "raw_response": response
            }
        except Exception as e:
            return {
                "parsing_successful": False,
                "error": f"Unexpected error: {e}",
                "raw_response": response
            }

    def verify_data_format(self, perturbation_data: Dict, source: str = "") -> bool:
        """Verify that the input data follows the expected new reversible format."""
        required_top_level = ['original_garage_data', 'passage_mapping', 'perturbation_metadata', 'perturbation_results']
        
        for key in required_top_level:
            if key not in perturbation_data:
                print(f"❌ FORMAT ERROR {source}: Missing top-level key '{key}'")
                return False
        
        # Check passage mapping structure
        mapping = perturbation_data['passage_mapping']
        if 'signal_indices' not in mapping or 'noise_indices' not in mapping:
            print(f"❌ FORMAT ERROR {source}: Invalid passage_mapping structure")
            return False
        
        # Check essential metadata
        metadata = perturbation_data['perturbation_metadata']
        essential_fields = ['perturbation_class', 'intensity', 'generation_successful']
        for field in essential_fields:
            if field not in metadata:
                print(f"❌ FORMAT ERROR {source}: Missing '{field}' in perturbation_metadata")
                return False
        
        return True

    def create_perturbation_id(self, perturbation_data: Dict) -> str:
        """Create a unique ID for a perturbation for tracking purposes."""
        try:
            # Handle both original format and format with verification data added
            if 'original_garage_data' in perturbation_data:
                # New reversible format
                original_garage_data = perturbation_data.get('original_garage_data', {})
                perturbation_metadata = perturbation_data.get('perturbation_metadata', {})
                
                original_query = original_garage_data.get('question', '')
                perturbation_class = perturbation_metadata.get('perturbation_class', '')
                intensity = perturbation_metadata.get('intensity', '')
                source_qid = perturbation_metadata.get('source_qid', '')
            else:
                # Fallback for other formats (shouldn't happen, but just in case)
                original_query = perturbation_data.get('original_query', '')
                perturbation_class = perturbation_data.get('perturbation_class', '')
                intensity = perturbation_data.get('intensity', '')
                source_qid = perturbation_data.get('source_qid', '')
            
            # Create unique identifier
            key_fields = [
                source_qid,
                perturbation_class,
                intensity,
                original_query[:50],  # First 50 chars of query
            ]
            return "|".join(key_fields)
        except Exception as e:
            print(f"Error creating perturbation ID: {e}")
            return "unknown_id"

    async def verify_perturbation_async(self, model_id: str, perturbation_data: Dict) -> Dict:
        """Verify a single perturbation asynchronously using the specified model."""
        
        try:
            # Verify data format first
            if not self.verify_data_format(perturbation_data, f"verify_perturbation_async({model_id})"):
                return {
                    **perturbation_data,
                    "verification_model": model_id,
                    "verification_successful": False,
                    "verification_error": "Invalid input data format"
                }
            
            # Extract required fields from new data format
            original_garage_data = perturbation_data.get('original_garage_data', {})
            perturbation_metadata = perturbation_data.get('perturbation_metadata', {})
            perturbation_results = perturbation_data.get('perturbation_results', {})
            
            original_query = original_garage_data.get('question', '')
            answer_generate_text = original_garage_data.get('answer_generate', '')
            perturbation_class = perturbation_metadata.get('perturbation_class', '')
            intensity = perturbation_metadata.get('intensity', '')
            
            # Extract signal and noise passages - FIX: correct method name
            signal_passages, noise_passages = self._extract_signal_noise_passages(perturbation_data)
            
            # Check if generation was successful
            if not perturbation_metadata.get('generation_successful', False):
                return {
                    **perturbation_data,
                    "verification_model": model_id,
                    "verification_successful": False,
                    "verification_error": "Original generation failed"
                }
            
            # Validate required fields
            if not all([original_query, answer_generate_text, perturbation_class, intensity]):
                return {
                    **perturbation_data,
                    "verification_model": model_id,
                    "verification_successful": False,
                    "verification_error": "Missing required fields"
                }
            
            # Check if perturbation results are valid
            if not perturbation_results.get('parsing_successful', False):
                return {
                    **perturbation_data,
                    "verification_model": model_id,
                    "verification_successful": False,
                    "verification_error": "Generation parsing failed"
                }
            
            # Get the verifier prompt using new signature
            prompt = self.catalogue.generate_verifier_prompt(
                perturbation_class=perturbation_class,
                intensity=intensity,
                original_query=original_query,
                answer_generate_text=answer_generate_text,
                signal_passages=signal_passages,
                noise_passages=noise_passages,
                generator_output_json=perturbation_results
            )
            
            # Call the specified model
            response = await self._call_model_async(model_id, prompt)
            
            # Parse the response
            parsed = self._parse_json_response(response)
            
            # Add metadata
            result = {
                **perturbation_data,
                "verification_model": model_id,
                "verification_successful": parsed.get("parsing_successful", False),
                "verification_response": parsed
            }
            
            return result
            
        except Exception as e:
            return {
                **perturbation_data,
                "verification_model": model_id,
                "verification_successful": False,
                "verification_error": str(e)
            }

    async def verify_batch_async(self, model_id: str, perturbations: List[Dict]) -> List[Dict]:
        """Verify a batch of perturbations asynchronously with the specified model."""
        
        tasks = []
        
        for perturbation in perturbations:
            task = self.verify_perturbation_async(model_id, perturbation)
            tasks.append(task)
        
        # Run all tasks with progress bar
        model_name = self.get_model_name_for_file(model_id)
        results = await atqdm.gather(*tasks, desc=f"Verifying {len(tasks)} perturbations with {model_name}")
        
        return results

    def test_verifier_functionality(self, sample_data: Dict) -> bool:
        """Test the verifier functionality on a sample record."""
        print(f"\n🧪 Testing verifier functionality...")
        
        try:
            # Test data format verification
            if not self.verify_data_format(sample_data, "test_sample"):
                print("❌ Test failed: Data format verification failed")
                return False
            
            # Test signal/noise extraction
            signal_passages, noise_passages = self._extract_signal_noise_passages(sample_data)
            if not signal_passages and not noise_passages:
                print("❌ Test failed: No passages extracted")
                return False
            
            print(f"✅ Extracted {len(signal_passages)} signal and {len(noise_passages)} noise passages")
            
            # Test ID creation
            perturbation_id = self.create_perturbation_id(sample_data)
            if perturbation_id == "unknown_id":
                print("❌ Test failed: Could not create perturbation ID")
                return False
            
            print(f"✅ Created perturbation ID: {perturbation_id}")
            
            # Test prompt generation (without calling model)
            original_garage_data = sample_data.get('original_garage_data', {})
            perturbation_metadata = sample_data.get('perturbation_metadata', {})
            perturbation_results = sample_data.get('perturbation_results', {})
            
            try:
                prompt = self.catalogue.generate_verifier_prompt(
                    perturbation_class=perturbation_metadata.get('perturbation_class', ''),
                    intensity=perturbation_metadata.get('intensity', ''),
                    original_query=original_garage_data.get('question', ''),
                    answer_generate_text=original_garage_data.get('answer_generate', ''),
                    signal_passages=signal_passages,
                    noise_passages=noise_passages,
                    generator_output_json=perturbation_results
                )
                
                if len(prompt) < 100:  # Sanity check
                    print("❌ Test failed: Generated prompt too short")
                    return False
                
                print(f"✅ Generated verification prompt ({len(prompt)} chars)")
                
            except Exception as e:
                print(f"❌ Test failed: Prompt generation error - {e}")
                return False
            
            print(f"✅ All verifier functionality tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            return False

    async def verify_dataset_async(self, 
                                 dataset_folder: str,
                                 model_id: str,
                                 input_filename: str = "refusalbench_perturbations.jsonl"):
        """Verify all perturbations in a dataset asynchronously with the specified model."""
        
        input_file = f"{dataset_folder}/{input_filename}"
        output_file = self.get_verification_file_path(dataset_folder, model_id)
        
        model_name = self.get_model_name_for_file(model_id)
        print(f"\nVerifying perturbations from {input_file} using {model_name}")
        print(f"Output will be saved to {output_file}")
        
        # Create output directory if it doesn't exist
        os.makedirs(dataset_folder, exist_ok=True)
        
        # Load existing verifications for resume functionality
        existing_ids = self.load_existing_verifications(output_file)
        
        # Read all perturbations
        perturbations = []
        skipped_count = 0
        format_error_count = 0
        
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist. Skipping.")
            return
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # Verify data format for first few records
                    if line_num < 3:
                        if not self.verify_data_format(data, f"{input_file}:line_{line_num}"):
                            format_error_count += 1
                            continue
                    
                    # Test functionality on first record
                    if line_num == 0:
                        if not self.test_verifier_functionality(data):
                            print(f"⚠️  Functionality test failed but continuing...")
                    
                    # Check if this perturbation has already been verified
                    perturbation_id = self.create_perturbation_id(data)
                    if perturbation_id in existing_ids:
                        skipped_count += 1
                        continue
                    
                    perturbations.append(data)
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        if format_error_count > 0:
            print(f"⚠️  {format_error_count} records failed format verification")
        
        print(f"Loaded {len(perturbations)} new perturbations to verify ({skipped_count} already verified)")
        
        if len(perturbations) == 0:
            print("No new perturbations to verify. All done!")
            return
        
        all_verifications = []
        
        # Process perturbations in batches
        for batch_start in range(0, len(perturbations), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(perturbations))
            batch = perturbations[batch_start:batch_end]
            
            print(f"\nVerifying batch {batch_start//self.batch_size + 1}/{(len(perturbations)-1)//self.batch_size + 1} with {model_name}")
            
            # Verify batch
            batch_results = await self.verify_batch_async(model_id, batch)
            all_verifications.extend(batch_results)
            
            # Append batch results to output file (for resume functionality)
            with open(output_file, 'a', encoding='utf-8') as f:
                for verification in batch_results:
                    f.write(json.dumps(verification) + '\n')
            
            print(f"Appended {len(batch_results)} verifications to {output_file}")
        
        print(f"\nCompleted verification of {len(all_verifications)} perturbations with {model_name}")
        
        # Print statistics for this run
        self._print_verification_statistics(all_verifications, model_name)

    def _print_verification_statistics(self, verifications: List[Dict], model_name: str):
        """Print detailed verification statistics for a specific model."""
        
        total = len(verifications)
        if total == 0:
            return
            
        successful_gen = sum(1 for v in verifications 
                           if v.get('perturbation_metadata', {}).get('generation_successful', False))
        successful_ver = sum(1 for v in verifications if v.get('verification_successful', False))
        
        print(f"\n{model_name} Verification Statistics:")
        print(f"Total perturbations processed: {total}")
        print(f"Successful generations: {successful_gen}")
        print(f"Successful verifications: {successful_ver}")
        print(f"Generation success rate: {successful_gen/total*100:.1f}%")
        print(f"Verification success rate: {successful_ver/total*100:.1f}%")
        
        # Break down by verification criteria
        if successful_ver > 0:
            ver_passed = []
            for v in verifications:
                if v.get('verification_successful', False):
                    response = v.get('verification_response', {})
                    if response.get('verification_result') == 'PASS':
                        ver_passed.append(v)
            
            print(f"Perturbations that passed verification: {len(ver_passed)}")
            print(f"Pass rate among verified: {len(ver_passed)/successful_ver*100:.1f}%")
        
        # Break down by perturbation type
        type_stats = {}
        for v in verifications:
            metadata = v.get('perturbation_metadata', {})
            perturbation_class = metadata.get('perturbation_class', 'Unknown')
            intensity = metadata.get('intensity', 'Unknown')
            key = f"{perturbation_class}_{intensity}"
            
            if key not in type_stats:
                type_stats[key] = {'total': 0, 'gen_success': 0, 'ver_success': 0, 'ver_pass': 0}
            
            type_stats[key]['total'] += 1
            if metadata.get('generation_successful', False):
                type_stats[key]['gen_success'] += 1
            if v.get('verification_successful', False):
                type_stats[key]['ver_success'] += 1
                response = v.get('verification_response', {})
                if response.get('verification_result') == 'PASS':
                    type_stats[key]['ver_pass'] += 1
        
        print(f"\nBreakdown by perturbation type:")
        for ptype, stats in type_stats.items():
            gen_rate = stats['gen_success']/stats['total']*100 if stats['total'] > 0 else 0
            ver_rate = stats['ver_success']/stats['total']*100 if stats['total'] > 0 else 0
            pass_rate = stats['ver_pass']/stats['ver_success']*100 if stats['ver_success'] > 0 else 0
            print(f"  {ptype}: {stats['total']} total, {gen_rate:.1f}% gen, {ver_rate:.1f}% ver, {pass_rate:.1f}% pass")

    async def verify_all_combinations_async(self, 
                                          datasets: Optional[List[str]] = None,
                                          models: Optional[List[str]] = None):
        """Verify all combinations of datasets and models."""
        
        datasets = datasets or self.datasets
        models = models or self.models
        
        print(f"Starting verification for {len(datasets)} datasets and {len(models)} models")
        print(f"Datasets: {datasets}")
        print(f"Models: {models}")
        
        total_combinations = len(datasets) * len(models)
        current_combination = 0
        
        for dataset in datasets:
            for model in models:
                current_combination += 1
                print(f"\n{'='*80}")
                print(f"COMBINATION {current_combination}/{total_combinations}: {dataset} + {self.get_model_name_for_file(model)}")
                print(f"{'='*80}")
                
                try:
                    await self.verify_dataset_async(dataset, model)
                except Exception as e:
                    print(f"Error processing {dataset} with {model}: {e}")
                    continue
        
        print(f"\n{'='*80}")
        print("ALL VERIFICATIONS COMPLETE!")
        print(f"{'='*80}")

def setup_environment():
    """Set up environment variables for API access."""
    try:
        # Try to import credentials from config
        from config import (
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME,
            OPENAI_API_KEY
        )

        # Set up AWS credentials from config
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
        os.environ["AWS_REGION_NAME"] = AWS_REGION_NAME

        # Set up OpenAI API key if available
        if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        elif "OPENAI_API_KEY" not in os.environ:
            print("Warning: OPENAI_API_KEY not configured.")
            print("OpenAI models will not be available.")

    except ImportError:
        # Check if credentials are already in environment
        if "AWS_ACCESS_KEY_ID" not in os.environ:
            print("Error: AWS credentials not found in environment or config.py")
            print("Please either:")
            print("1. Copy config_template.py to config.py and fill in your credentials")
            print("2. Set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME")
            raise RuntimeError("Missing AWS credentials")

async def main_async():
    """Async main function to run multi-model perturbation verification."""
    
    # Set up environment
    setup_environment()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Model RefusalBench Perturbation Verifier")
    parser.add_argument("--datasets", nargs="+", 
                       help="Datasets to process (default: all)",
                       choices=["dataset_claude", "dataset_deepseek", "dataset_nova", "dataset_gpt"])
    parser.add_argument("--models", nargs="+",
                       help="Models to use for verification (default: all)",
                       choices=[
                           "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
                           "bedrock/us.amazon.nova-pro-v1:0",
                           "openai/gpt-4o-2024-08-06",
                           "bedrock/converse/us.deepseek.r1-v1:0"
                       ])
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Number of perturbations to process in each batch")
    parser.add_argument("--max-concurrent", type=int, default=10,
                       help="Maximum number of concurrent API calls")
    
    args = parser.parse_args()
    
    print(f"🔍 RefusalBench Multi-Model Verifier (Updated for Reversible Format)")
    print(f"{'='*60}")
    
    print(f"\n📋 EXPECTED INPUT FORMAT:")
    print(f"   Input files should contain records with new reversible data format:")
    print(f"   - original_garage_data: {{question, answer_generate, grounding, ...}}")
    print(f"   - passage_mapping: {{signal_indices: [...], noise_indices: [...]}}")
    print(f"   - perturbation_metadata: {{perturbation_class, intensity, generation_successful, ...}}")
    print(f"   - perturbation_results: {{perturbed_query, perturbed_signal_passages, lever_selected, ...}}")
    
    print(f"\n📤 OUTPUT FORMAT:")
    print(f"   Input format + verification metadata:")
    print(f"   - verification_model: model used for verification")
    print(f"   - verification_successful: boolean")
    print(f"   - verification_response: {{verification_result: PASS/FAIL, ...}}")
    
    # Initialize verifier
    verifier = MultiModelAsyncRefusalBenchVerifier(
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        temperature=0.1,
        max_tokens=2000
    )
    
    # Verify all combinations
    await verifier.verify_all_combinations_async(
        datasets=args.datasets,
        models=args.models
    )
    
    print(f"\nMulti-model perturbation verification complete!")

def main():
    """Synchronous wrapper for the async main function."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()