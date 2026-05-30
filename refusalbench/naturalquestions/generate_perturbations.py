#!/usr/bin/env python3
"""
RefusalBench Perturbation Generator using LiteLLM with AWS Bedrock Claude models.

Generates linguistically-grounded perturbations across 6 uncertainty dimensions
with controlled qualitative intensity (LOW/MEDIUM/HIGH) using async processing.
"""

import litellm
import json
import os
import asyncio
from typing import Dict, List, Optional, Union
from tqdm.asyncio import tqdm as atqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pathlib import Path

# Import the RefusalBench catalogue
from prompt_guidelines import RefusalBenchCatalogue, get_available_combinations


class AsyncRefusalBenchGenerator:
    """
    Async RefusalBench Perturbation Generator using LiteLLM with AWS Bedrock Claude models.
    
    Generates linguistically-grounded perturbations using the official RefusalBench lever catalogue.
    """
    
    def __init__(self, 
                 region_name: str = 'us-east-1',
                 model_id: str = 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0',
                 batch_size: int = 10,
                 max_concurrent: int = 20,
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        """Initialize the async perturbation generator."""
        
        # Set up AWS environment variables
        os.environ["AWS_REGION_NAME"] = region_name
        
        # LiteLLM model configuration
        self.model_id = model_id
        self.model_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.99
        }
        
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        
        # Async processing setup
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Add rate limiting
        # self.last_request_time = 0
        # self.min_request_interval = 0.5  # 500ms between requests
                
        # Initialize RefusalBench catalogue
        self.catalogue = RefusalBenchCatalogue()
        
        # Get all available perturbation combinations
        self.combinations = get_available_combinations()
        
        print(f"Initialized generator with {len(self.combinations)} perturbation combinations")

    @retry(
    wait=wait_random_exponential(min=2, max=30),
    stop=stop_after_attempt(100),
    )
    async def _call_claude_async(self, prompt: str) -> str:
        """Call Claude model via LiteLLM asynchronously with retry logic."""
        async with self.semaphore:


            # current_time = time.time()
            # time_since_last = current_time - self.last_request_time
            # if time_since_last < self.min_request_interval:
            #     await asyncio.sleep(self.min_request_interval - time_since_last)
            # self.last_request_time = time.time()


            try:
                messages = [{"role": "user", "content": prompt}]
                
                # Call LiteLLM completion
                response = await litellm.acompletion(
                    model=self.model_id,
                    messages=messages,
                    **self.model_kwargs
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"Error calling Claude: {e}")
                raise e

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from Claude, handling common formatting issues."""
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
            print(e)
            return {
                "parsing_successful": False,
                "error": f"Unexpected error: {e}",
                "raw_response": response
            }

    async def generate_perturbation_async(self, 
                                        original_query: str, 
                                        original_context: str, 
                                        original_answers: Union[str, List[str]], 
                                        perturbation_class: str, 
                                        intensity: str) -> Dict:
        """Generate a single perturbation asynchronously using the RefusalBench catalogue."""
        
        try:
            # Get the generator prompt from catalogue (already formatted)
            prompt = self.catalogue.generate_generator_prompt(
                perturbation_class, intensity,
                original_query, original_context, original_answers
            )
            
            # Call Claude
            response = await self._call_claude_async(prompt)
            
            # Parse the response
            parsed = self._parse_json_response(response)
            
            # Add metadata
            result = {
                "original_query": original_query,
                "original_context": original_context,
                "original_answers": original_answers,
                "perturbation_class": perturbation_class,
                "intensity": intensity,
                "generation_successful": parsed.get("parsing_successful", False),
                **parsed
            }
            
            return result
            
        except Exception as e:
            return {
                "original_query": original_query,
                "original_context": original_context,
                "original_answers": original_answers,
                "perturbation_class": perturbation_class,
                "intensity": intensity,
                "generation_successful": False,
                "error": str(e)
            }

    async def generate_all_perturbations_async(self, 
                                             original_query: str, 
                                             original_context: str, 
                                             original_answers: Union[str, List[str]]) -> List[Dict]:
        """Generate all perturbations for a single base instance asynchronously.
        
        Generates exactly 1 perturbation per combination.
        Total: 6 classes × 3 intensities = 18 perturbations per base instance.
        """
        
        tasks = []
        
        # Generate exactly 1 perturbation per combination
        for perturbation_class, intensity in self.combinations:
            task = self.generate_perturbation_async(
                original_query, original_context, original_answers,
                perturbation_class, intensity
            )
            tasks.append(task)
        
        # Run all tasks with progress bar
        results = await atqdm.gather(*tasks, desc=f"Generating {len(tasks)} perturbations")
        
        return results

    async def process_dataset_async(self, 
                                  input_file: str, 
                                  output_file: str, 
                                  max_instances: int = 100):
        """Process dataset and generate perturbations asynchronously."""
        
        print(f"Processing dataset from {input_file}")
        print(f"Will generate up to {max_instances} base instances")
        print(f"Each instance will generate {len(self.combinations)} perturbations")
        
        all_perturbations = []
        processed_count = 0
        
        # Read all instances first
        instances = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if processed_count >= max_instances:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    
                    # Extract query, context, and answer using same logic as original
                    query = data['query']
                    
                    # Use the gold document as context (first one)
                    if data.get('gold_docs') and len(data['gold_docs']) > 0:
                        context = data['gold_docs'][0]  # Use first gold doc
                    else:
                        # Fallback to first retrieved doc
                        context = data['retrieved_docs'][0] if data.get('retrieved_docs') else ""
                    
                    # Use first answer as the canonical answer
                    # answer = data['answer'][0] if isinstance(data['answer'], list) else data['answer']

                    # NEW: Keep all answers
                    answers = data['answer'] if isinstance(data['answer'], list) else [data['answer']]
                    
                    # Skip entries with no context or query
                    if not context or not query:
                        print(f"Skipping entry {line_num}: missing context or query")
                        continue
                    
                    instances.append({
                        'query': query,
                        'context': context,
                        'answers': answers,
                        'qid': data.get('qid', line_num),
                        'line_num': line_num
                    })
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(instances)} valid instances")
        
        # Process instances in batches
        for batch_start in range(0, len(instances), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(instances))
            batch = instances[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//self.batch_size + 1}/{(len(instances)-1)//self.batch_size + 1}")
            
            # Create tasks for this batch
            batch_tasks = []
            for instance in batch:
                task = self.generate_all_perturbations_async(
                    instance['query'], 
                    instance['context'], 
                    instance['answers']
                )
                batch_tasks.append((task, instance))
            
            # Process batch
            batch_results = await asyncio.gather(*[task for task, _ in batch_tasks])
            
            # Add metadata and collect results
            for (_, instance), perturbations in zip(batch_tasks, batch_results):
                for perturb in perturbations:
                    perturb['source_qid'] = instance['qid']
                    perturb['source_line'] = instance['line_num']
                
                all_perturbations.extend(perturbations)
            
            # Save intermediate results
            temp_output = output_file.replace('.jsonl', f'_batch_{batch_start//self.batch_size + 1}.jsonl')
            with open(temp_output, 'w', encoding='utf-8') as temp_f:
                for perturb in all_perturbations:
                    temp_f.write(json.dumps(perturb) + '\n')
            print(f"Saved batch results to {temp_output}")
        
        # Save final results
        print(f"\nSaving {len(all_perturbations)} perturbations to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for perturbation in all_perturbations:
                f.write(json.dumps(perturbation) + '\n')
        
        # Print statistics
        successful = sum(1 for p in all_perturbations if p.get('generation_successful', False))
        print(f"\nGeneration Statistics:")
        print(f"Total perturbations attempted: {len(all_perturbations)}")
        print(f"Successful generations: {successful}")
        print(f"Failed generations: {len(all_perturbations) - successful}")
        print(f"Success rate: {successful/len(all_perturbations)*100:.1f}%")
        
        # Print breakdown by perturbation type
        type_counts = {}
        for p in all_perturbations:
            if p.get('generation_successful', False):
                key = f"{p['perturbation_class']}_{p['intensity']}"
                type_counts[key] = type_counts.get(key, 0) + 1
        
        print(f"\nBreakdown by perturbation class and intensity:")
        for perturbation_class, intensity in self.combinations:
            key = f"{perturbation_class}_{intensity}"
            count = type_counts.get(key, 0)
            print(f"  {perturbation_class} {intensity}: {count}")


async def main_async():
    """Async main function to run perturbation generation."""

    # Import configuration
    try:
        from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME, ORIGINAL_DATASET_PATH, OUTPUT_DIR
    except ImportError:
        print("Error: config.py not found. Please copy config_template.py to config.py and fill in your credentials.")
        return

    # Set up AWS credentials from config
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_REGION_NAME"] = AWS_REGION_NAME

    # Use paths from config
    # ORIGINAL_DATASET_PATH should point to a JSONL file from Natural Questions (NQ) dataset
    # where all models achieved correct answers. Each line should contain:
    # - query: the question
    # - answer: correct answer(s) as a list
    # - gold_docs or retrieved_docs: relevant context passages
    # - qid: unique question identifier
    input_file = ORIGINAL_DATASET_PATH
    output_file = os.path.join(OUTPUT_DIR, "refusalbench_perturbations.jsonl")
    
    # Initialize generator
    generator = AsyncRefusalBenchGenerator(
        model_id='bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0',
        batch_size=5,  # Process 5 instances at a time
        max_concurrent=3,  # Allow up to 20 concurrent API calls
        temperature=0.1,
        max_tokens=2000
    )
    
    # Process dataset
    await generator.process_dataset_async(
        input_file=input_file,
        output_file=output_file,
        max_instances=100  # 1 instance × 18 perturbations = 18 total for testing
    )
    
    print(f"\nPerturbation generation complete! Results saved to {output_file}")


def main():
    """Synchronous wrapper for the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()