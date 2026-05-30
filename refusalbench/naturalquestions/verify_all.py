#!/usr/bin/env python3
"""
Multi-Model RefusalBench Perturbation Verifier using LiteLLM.
Verifies that generated perturbations correctly implement the specified 
linguistic mechanisms and achieve the target intensity levels across multiple models and datasets.
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
from prompt_guidelines import RefusalBenchCatalogue

# Import configuration
try:
    from config import (
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME,
        OPENAI_API_KEY, DEFAULT_EVALUATOR_MODEL
    )
    # Set up credentials from config
    if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
except ImportError:
    print("Warning: config.py not found. Please copy config_template.py to config.py and fill in your credentials.")
    print("Using environment variables if available...")


class MultiModelAsyncRefusalBenchVerifier:
    """
    Multi-Model Async RefusalBench Perturbation Verifier using LiteLLM.
    
    Verifies generated perturbations using the official RefusalBench lever catalogue
    across multiple models and datasets with resume functionality.
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
        # Use key fields to create a unique identifier
        key_fields = [
            perturbation_data.get("original_query", ""),
            perturbation_data.get("perturbation_class", ""),
            perturbation_data.get("intensity", ""),
            # Add more fields if needed for uniqueness
            str(perturbation_data.get("original_context", ""))[:100]  # First 100 chars of context
        ]
        return "|".join(key_fields)

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

    async def verify_perturbation_async(self, model_id: str, perturbation_data: Dict) -> Dict:
        """Verify a single perturbation asynchronously using the specified model."""
        
        try:
            # Extract required fields
            original_query = perturbation_data.get("original_query", "")
            original_context = perturbation_data.get("original_context", "")
            original_answers = perturbation_data.get("original_answers", "")
            perturbation_class = perturbation_data.get("perturbation_class", "")
            intensity = perturbation_data.get("intensity", "")
            
            # Check if generation was successful
            if not perturbation_data.get("generation_successful", False):
                return {
                    **perturbation_data,
                    "verification_model": model_id,
                    "verification_successful": False,
                    "verification_error": "Original generation failed"
                }
            
            # Validate required fields
            if not all([original_query, original_context, perturbation_class, intensity]):
                return {
                    **perturbation_data,
                    "verification_model": model_id,
                    "verification_successful": False,
                    "verification_error": "Missing required fields"
                }
            
            # Prepare generator output for verification
            generator_output = {
                "perturbed_query": perturbation_data.get("perturbed_query", ""),
                "perturbed_context": perturbation_data.get("perturbed_context", ""),
                "lever_selected": perturbation_data.get("lever_selected", ""),
                "implementation_reasoning": perturbation_data.get("implementation_reasoning", ""),
                "intensity_achieved": perturbation_data.get("intensity_achieved", "")
            }
            
            # Get the verifier prompt from catalogue
            prompt = self.catalogue.generate_verifier_prompt(
                perturbation_class, intensity,
                original_query, original_context, original_answers,
                json.dumps(generator_output, indent=2)
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
        
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist. Skipping.")
            return
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # Check if this perturbation has already been verified
                    perturbation_id = self.create_perturbation_id(data)
                    if perturbation_id in existing_ids:
                        skipped_count += 1
                        continue
                    
                    perturbations.append(data)
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
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
            
        successful_gen = sum(1 for v in verifications if v.get('generation_successful', False))
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