#!/usr/bin/env python3
"""
RefusalBench Perturbation Generator using LiteLLM with AWS Bedrock Claude models.
Generates linguistically-grounded perturbations across 6 uncertainty dimensions
with controlled qualitative intensity (LOW/MEDIUM/HIGH) using async processing.
NOW WITH RESUME FUNCTIONALITY - skips already processed instances.
UPDATED FOR GARAGE SCHEMA AND NEW REFUSALBENCHCATALOGUE.
"""
import litellm
import json
import os
import asyncio
from typing import Dict, List, Optional, Union, Set, Tuple
from tqdm.asyncio import tqdm as atqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pathlib import Path

# Import the RefusalBench catalogue with correct filename
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

def get_available_combinations() -> List[Tuple[str, str]]:
    """Get all available perturbation combinations from the catalogue."""
    catalogue = RefusalBenchCatalogue()
    return catalogue.get_all_combinations()

class AsyncRefusalBenchGenerator:
    """
    Async RefusalBench Perturbation Generator using LiteLLM with AWS Bedrock Claude models.
    
    Generates linguistically-grounded perturbations using the official RefusalBench lever catalogue.
    NOW WITH RESUME FUNCTIONALITY and GARAGE SCHEMA SUPPORT.
    """
    
    def __init__(self, 
                 region_name: str = 'us-east-1',
                 model_id: str = 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0',
                 batch_size: int = 10,
                 max_concurrent: int = 20,
                 temperature: float = 0.1,
                 max_tokens: int = 2000,
                 force_restart: bool = False):
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
        self.force_restart = force_restart
        
        # Async processing setup
        self.semaphore = asyncio.Semaphore(max_concurrent)
                
        # Initialize RefusalBench catalogue
        self.catalogue = RefusalBenchCatalogue()
        
        # Get all available perturbation combinations
        self.combinations = get_available_combinations()
        
        print(f"Initialized generator with {len(self.combinations)} perturbation combinations")
        if force_restart:
            print("⚠️  FORCE RESTART MODE: Will overwrite existing results")

    def _extract_passages_from_grounding(self, grounding: List[Dict], 
                                       evidence_relevant: List[str], 
                                       evidence_correct: List[str]) -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        Extract signal and noise passages from GaRAGe grounding data using the same criteria 
        as the curation script, and return mapping indices for reversibility.
        
        Signal passages: evidence_correct[i] == "ANSWER-THE-QUESTION"
        Noise passages: everything else
        
        Returns:
            Tuple of (signal_passages, noise_passages, signal_indices, noise_indices)
        """
        signal_passages = []
        noise_passages = []
        signal_indices = []
        noise_indices = []
        
        for i, passage_data in enumerate(grounding):
            # Extract the text content from the passage
            passage_text = None
            for key, value in passage_data.items():
                if key.startswith('cite_') and isinstance(value, str):
                    passage_text = value.strip()
                    break
            
            if not passage_text:
                continue
                
            # Use the same criteria as the curation script
            is_signal = i < len(evidence_correct) and evidence_correct[i] == "ANSWER-THE-QUESTION"
            
            if is_signal:
                signal_passages.append(passage_text)
                signal_indices.append(i)
            else:
                noise_passages.append(passage_text)
                noise_indices.append(i)
        
        return signal_passages, noise_passages, signal_indices, noise_indices

    def reconstruct_original_format(self, perturbed_record: Dict) -> Dict:
        """
        Reconstruct the original GaRAGe format from a perturbed record.
        This demonstrates the reversibility of our data structure.
        
        Args:
            perturbed_record: Record with our new reversible structure
            
        Returns:
            Original GaRAGe format record
        """
        return perturbed_record['original_garage_data'].copy()

    def reconstruct_perturbed_garage_format(self, perturbed_record: Dict) -> Dict:
        """
        Reconstruct a GaRAGe-format record with perturbations applied.
        
        Args:
            perturbed_record: Record with our new reversible structure
            
        Returns:
            GaRAGe format record with perturbations applied to grounding
        """
        # Start with original data
        result = perturbed_record['original_garage_data'].copy()
        
        # Get perturbation results
        perturbation_results = perturbed_record.get('perturbation_results', {})
        mapping = perturbed_record['passage_mapping']
        
        # Apply perturbations to grounding array
        if perturbation_results.get('parsing_successful', False):
            # Apply query perturbation
            if 'perturbed_query' in perturbation_results:
                result['question'] = perturbation_results['perturbed_query']
            
            # Apply signal passage perturbations
            for perturbed_passage in perturbation_results.get('perturbed_signal_passages', []):
                original_idx = perturbed_passage.get('original_index', 0)
                if original_idx < len(mapping['signal_indices']):
                    grounding_idx = mapping['signal_indices'][original_idx]
                    # Update the cite_ field in the grounding
                    for key in result['grounding'][grounding_idx]:
                        if key.startswith('cite_'):
                            result['grounding'][grounding_idx][key] = perturbed_passage['perturbed_text']
                            break
            
            # Apply noise passage perturbations  
            for perturbed_passage in perturbation_results.get('perturbed_noise_passages', []):
                original_idx = perturbed_passage.get('original_index', 0)
                if original_idx < len(mapping['noise_indices']):
                    grounding_idx = mapping['noise_indices'][original_idx]
                    # Update the cite_ field in the grounding
                    for key in result['grounding'][grounding_idx]:
                        if key.startswith('cite_'):
                            result['grounding'][grounding_idx][key] = perturbed_passage['perturbed_text']
                            break
        
        # Add perturbation metadata to the record
        result['refusalbench_perturbation_metadata'] = perturbed_record['perturbation_metadata']
        
        return result

    def _load_existing_results(self, output_file: str) -> Dict[str, Dict]:
        """
        Load existing results and return processed instances with their perturbation counts.
        
        Returns:
            Dict mapping source_qid -> {
                'total_perturbations': int,
                'perturbation_types': set of (class, intensity) tuples,
                'is_complete': bool
            }
        """
        if not os.path.exists(output_file):
            return {}
        
        existing_results = {}
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        source_qid = data.get('perturbation_metadata', {}).get('source_qid')
                        
                        if source_qid is None:
                            continue
                            
                        # Initialize if first time seeing this qid
                        if source_qid not in existing_results:
                            existing_results[source_qid] = {
                                'total_perturbations': 0,
                                'perturbation_types': set(),
                                'is_complete': False
                            }
                        
                        # Only count successful generations
                        generation_successful = data.get('perturbation_metadata', {}).get('generation_successful', False)
                        if generation_successful:
                            existing_results[source_qid]['total_perturbations'] += 1
                            perturbation_class = data.get('perturbation_metadata', {}).get('perturbation_class')
                            intensity = data.get('perturbation_metadata', {}).get('intensity')
                            existing_results[source_qid]['perturbation_types'].add(
                                (perturbation_class, intensity)
                            )
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line {line_num} in existing results: {e}")
                        continue
                        
        except Exception as e:
            print(f"Warning: Error reading existing results: {e}")
            return {}
        
        # Check completeness for each instance
        expected_combinations = set(self.combinations)
        for qid, info in existing_results.items():
            info['is_complete'] = (
                info['perturbation_types'] == expected_combinations and
                info['total_perturbations'] == len(expected_combinations)
            )
        
        return existing_results

    def _get_missing_perturbations(self, existing_types: set) -> List[tuple]:
        """Get list of missing perturbation types for an instance."""
        expected = set(self.combinations)
        missing = expected - existing_types
        return list(missing)

    @retry(
        wait=wait_random_exponential(min=2, max=30),
        stop=stop_after_attempt(100),
    )
    async def _call_claude_async(self, prompt: str) -> str:
        """Call Claude model via LiteLLM asynchronously with retry logic."""
        async with self.semaphore:
            try:
                messages = [{"role": "user", "content": prompt}]
                
                # Call LiteLLM completion with timeout
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=self.model_id,
                        messages=messages,
                        **self.model_kwargs
                    ),
                    timeout=60.0  # 60 second timeout
                )
                
                print(".", end="", flush=True)  # Progress indicator
                return response.choices[0].message.content
                
            except asyncio.TimeoutError:
                print(f"\n⏰ API call timed out after 60 seconds")
                raise
            except Exception as e:
                print(f"\n❌ Error calling Claude: {e}")
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
                                        answer_generate_text: str,
                                        signal_passages: List[str],
                                        noise_passages: List[str],
                                        perturbation_class: str, 
                                        intensity: str) -> Dict:
        """Generate a single perturbation asynchronously using the RefusalBench catalogue."""
        
        try:
            # Get the generator prompt from catalogue with updated signature
            prompt = self.catalogue.generate_generator_prompt(
                perturbation_class=perturbation_class,
                intensity=intensity,
                original_query=original_query,
                answer_generate_text=answer_generate_text,
                signal_passages=signal_passages,
                noise_passages=noise_passages
            )
            
            # Call Claude
            response = await self._call_claude_async(prompt)
            
            # Parse the response
            parsed = self._parse_json_response(response)
            
            # Add metadata (removed duplicate keys)
            result = {
                "original_query": original_query,
                "answer_generate_text": answer_generate_text,
                "signal_passages": signal_passages,
                "noise_passages": noise_passages,
                "perturbation_class": perturbation_class,
                "intensity": intensity,
                "generation_successful": parsed.get("parsing_successful", False),
                **parsed
            }
            
            return result
            
        except Exception as e:
            return {
                "original_query": original_query,
                "answer_generate_text": answer_generate_text,
                "signal_passages": signal_passages,
                "noise_passages": noise_passages,
                "perturbation_class": perturbation_class,
                "intensity": intensity,
                "generation_successful": False,
                "error": str(e)
            }

    async def generate_perturbations_for_instance_async(self, 
                                                      original_query: str, 
                                                      answer_generate_text: str,
                                                      signal_passages: List[str],
                                                      noise_passages: List[str],
                                                      signal_indices: List[int],
                                                      noise_indices: List[int],
                                                      original_garage_data: Dict,
                                                      missing_perturbations: List[tuple] = None) -> List[Dict]:
        """
        Generate perturbations for a single instance, optionally only missing ones.
        
        Args:
            missing_perturbations: List of (class, intensity) tuples to generate.
                                 If None, generates all perturbations.
        """
        
        perturbations_to_generate = missing_perturbations or self.combinations
        
        if not perturbations_to_generate:
            return []
        
        tasks = []
        for perturbation_class, intensity in perturbations_to_generate:
            # Fixed: Remove unused parameters from method call
            task = self.generate_perturbation_async(
                original_query, answer_generate_text, signal_passages, noise_passages,
                perturbation_class, intensity
            )
            tasks.append((task, perturbation_class, intensity))
        
        # Run all tasks
        raw_results = await asyncio.gather(*[task for task, _, _ in tasks])
        
        # Build final results with proper structure
        results = []
        for (_, perturbation_class, intensity), raw_result in zip(tasks, raw_results):
            # Create the full structured result
            structured_result = {
                'original_garage_data': original_garage_data,
                'passage_mapping': {
                    'signal_indices': signal_indices,
                    'noise_indices': noise_indices
                },
                'perturbation_metadata': {
                    'perturbation_class': perturbation_class,
                    'intensity': intensity,
                    'generation_successful': raw_result.get('generation_successful', False)
                },
                'perturbation_results': raw_result
            }
            results.append(structured_result)
        
        return results

    async def process_dataset_async(self, 
                                  input_file: str, 
                                  output_file: str, 
                                  max_instances: int = 100):
        """Process dataset and generate perturbations asynchronously with resume functionality."""
        
        print(f"Processing dataset from {input_file}")
        print(f"Output file: {output_file}")
        print(f"Will process up to {max_instances} base instances")
        
        # Load existing results unless forcing restart
        existing_results = {}
        existing_complete_count = 0
        if not self.force_restart:
            existing_results = self._load_existing_results(output_file)
            if existing_results:
                existing_complete_count = sum(1 for info in existing_results.values() if info['is_complete'])
                partial_count = len(existing_results) - existing_complete_count
                print(f"📁 Found existing results:")
                print(f"   • {existing_complete_count} complete instances")
                print(f"   • {partial_count} partial instances")
                print(f"   • Will resume from where we left off")
            else:
                print("📁 No existing results found, starting fresh")
        
        # Read and filter instances from GaRAGe format
        instances_to_process = []
        instances_skipped = 0
        total_instances_seen = 0
        
        # Calculate how many more instances we need
        remaining_needed = max_instances - existing_complete_count
        
        if remaining_needed <= 0:
            print(f"✅ Already have {existing_complete_count} complete instances (target: {max_instances})")
            print("Nothing to do!")
            return
        
        print(f"🎯 Need {remaining_needed} more instances to reach target of {max_instances}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if len(instances_to_process) >= remaining_needed:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    
                    # Extract GaRAGe schema data
                    query = data['question']
                    answer_generate_text = data['answer_generate']
                    grounding = data['grounding']
                    evidence_relevant = data.get('evidence_relevant', [])
                    evidence_correct = data.get('evidence_correct', [])
                    qid = data.get('sample_id', line_num)
                    
                    # Extract signal and noise passages with mapping
                    signal_passages, noise_passages, signal_indices, noise_indices = self._extract_passages_from_grounding(
                        grounding, evidence_relevant, evidence_correct
                    )
                    
                    # Skip entries with no context or query
                    if not query or not answer_generate_text or (not signal_passages and not noise_passages):
                        print(f"Skipping entry {line_num}: missing required fields")
                        continue
                    
                    # Check if already complete
                    if qid in existing_results and existing_results[qid]['is_complete']:
                        instances_skipped += 1
                        total_instances_seen += 1
                        continue
                    
                    # Determine what perturbations are needed
                    missing_perturbations = None
                    if qid in existing_results:
                        missing_perturbations = self._get_missing_perturbations(
                            existing_results[qid]['perturbation_types']
                        )
                        if not missing_perturbations:
                            instances_skipped += 1
                            total_instances_seen += 1
                            continue
                    
                    instances_to_process.append({
                        'query': query,
                        'answer_generate_text': answer_generate_text,
                        'signal_passages': signal_passages,
                        'noise_passages': noise_passages,
                        'signal_indices': signal_indices,
                        'noise_indices': noise_indices,
                        'original_garage_data': data,  # Store full original GaRAGe record
                        'qid': qid,
                        'line_num': line_num,
                        'missing_perturbations': missing_perturbations
                    })
                    total_instances_seen += 1
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        print(f"📊 Processing summary:")
        print(f"   • Target instances: {max_instances}")
        print(f"   • Already complete: {existing_complete_count}")
        print(f"   • Instances skipped: {instances_skipped}")
        print(f"   • Instances to process: {len(instances_to_process)}")
        print(f"   • Will reach total: {existing_complete_count + len(instances_to_process)}")
        
        if not instances_to_process:
            print("✅ All instances already complete! Nothing to do.")
            return
        
        # Determine file write mode
        write_mode = 'w' if self.force_restart else 'a'
        
        # Process instances in batches
        all_new_perturbations = []
        total_batches = (len(instances_to_process) - 1) // self.batch_size + 1
        
        print(f"\n🚀 Starting batch processing: {total_batches} batches of up to {self.batch_size} instances each")
        
        for batch_start in range(0, len(instances_to_process), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(instances_to_process))
            batch = instances_to_process[batch_start:batch_end]
            current_batch_num = batch_start // self.batch_size + 1
            
            print(f"\n" + "="*50)
            print(f"🔄 BATCH {current_batch_num}/{total_batches}")
            print(f"="*50)
            
            # Create tasks for this batch
            batch_tasks = []
            for instance in batch:
                if instance['missing_perturbations'] is not None:
                    missing_count = len(instance['missing_perturbations'])
                    print(f"   • QID {instance['qid']}: generating {missing_count} missing perturbations")
                else:
                    print(f"   • QID {instance['qid']}: generating all {len(self.combinations)} perturbations")
                
                task = self.generate_perturbations_for_instance_async(
                    instance['query'], 
                    instance['answer_generate_text'],
                    instance['signal_passages'],
                    instance['noise_passages'],
                    instance['signal_indices'],
                    instance['noise_indices'],
                    instance['original_garage_data'],
                    instance['missing_perturbations']
                )
                batch_tasks.append((task, instance))
            
            # Process batch
            batch_results = await asyncio.gather(*[task for task, _ in batch_tasks])
            
            # Add metadata and collect results
            batch_perturbations = []
            for (_, instance), perturbations in zip(batch_tasks, batch_results):
                for perturb in perturbations:
                    # Add source tracking to perturbation metadata
                    perturb['perturbation_metadata']['source_qid'] = instance['qid']
                    perturb['perturbation_metadata']['source_line'] = instance['line_num']
                    perturb['perturbation_metadata']['generation_timestamp'] = None  # Could add actual timestamp
                    batch_perturbations.append(perturb)
            
            all_new_perturbations.extend(batch_perturbations)
            
            # Append results immediately to avoid loss
            with open(output_file, write_mode, encoding='utf-8') as f:
                for perturb in batch_perturbations:
                    f.write(json.dumps(perturb) + '\n')
            
            # Switch to append mode after first batch if we started with write mode
            if write_mode == 'w':
                write_mode = 'a'
            
            print(f"   ✅ Batch complete: {len(batch_perturbations)} perturbations saved")
        
        # Print final statistics
        successful = sum(1 for p in all_new_perturbations if p.get('perturbation_metadata', {}).get('generation_successful', False))
        failed = len(all_new_perturbations) - successful
        
        print(f"\n" + "="*60)
        print(f"📈 FINAL GENERATION STATISTICS (this run)")
        print(f"="*60)
        print(f"📊 Overall Performance:")
        print(f"   • Total perturbations attempted: {len(all_new_perturbations)}")
        print(f"   • Successful generations: {successful}")
        print(f"   • Failed generations: {failed}")
        if all_new_perturbations:
            print(f"   • Success rate: {successful/len(all_new_perturbations)*100:.1f}%")
        
        print(f"\n📋 Dataset Progress:")
        final_complete_count = existing_complete_count + len([i for i in instances_to_process 
                                                            if i['missing_perturbations'] is None or 
                                                            len(i['missing_perturbations']) == len(self.combinations)])
        print(f"   • Target instances: {max_instances}")
        print(f"   • Complete instances: {final_complete_count}")
        print(f"   • Progress: {final_complete_count/max_instances*100:.1f}%")
        
        # Print breakdown by perturbation type for new perturbations
        type_counts = {}
        type_failed = {}
        for p in all_new_perturbations:
            perturbation_class = p.get('perturbation_metadata', {}).get('perturbation_class', 'Unknown')
            intensity = p.get('perturbation_metadata', {}).get('intensity', 'Unknown')
            key = f"{perturbation_class}_{intensity}"
            if p.get('perturbation_metadata', {}).get('generation_successful', False):
                type_counts[key] = type_counts.get(key, 0) + 1
            else:
                type_failed[key] = type_failed.get(key, 0) + 1
        
        if type_counts or type_failed:
            print(f"\n📊 Detailed breakdown by perturbation type:")
            for perturbation_class, intensity in self.combinations:
                key = f"{perturbation_class}_{intensity}"
                success_count = type_counts.get(key, 0)
                fail_count = type_failed.get(key, 0)
                total_count = success_count + fail_count
                if total_count > 0:
                    success_rate = success_count/total_count*100 if total_count > 0 else 0
                    print(f"   • {perturbation_class} {intensity}: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        print(f"\n✅ Results saved to {output_file}")
        print(f"🎉 Generation complete!")
        print(f"="*60)

async def main_async():
    """Async main function to run perturbation generation."""

    # Set up AWS credentials from config
    try:
        from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
        os.environ["AWS_REGION_NAME"] = AWS_REGION_NAME
    except ImportError:
        # Check if credentials are in environment
        if "AWS_ACCESS_KEY_ID" not in os.environ:
            print("Error: AWS credentials not found. Please set up config.py or environment variables.")
            raise RuntimeError("Missing AWS credentials")
    
    # Hardcoded paths
    input_file = "/data/group_data/r3lit_shared/ragdynabench/refusalbench/garage/garage_base_final.jsonl"
    output_file = "/data/group_data/r3lit_shared/ragdynabench/refusalbench/garage/dataset_claude/refusalbench_perturbations.jsonl"


    from pathlib import Path
    
    # Add this before writing to the output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
        
    # Initialize generator
    generator = AsyncRefusalBenchGenerator(
        model_id='bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0',
        batch_size=5,  # Process 5 instances at a time
        max_concurrent=3,  # Allow up to 3 concurrent API calls
        temperature=0.1,
        max_tokens=2000
    )
    
    # Process dataset
    await generator.process_dataset_async(
        input_file=input_file,
        output_file=output_file,
        max_instances=100  # 2 instances for testing
    )
    
    print(f"\nPerturbation generation complete! Results saved to {output_file}")
    
    # Demonstrate reversibility with first few records
    print(f"\n--- DEMONSTRATING REVERSIBILITY ---")
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 2:  # Just show first 2 records
                    break
                try:
                    record = json.loads(line.strip())
                    
                    # Show original reconstruction
                    original = generator.reconstruct_original_format(record)
                    print(f"Record {i+1}:")
                    print(f"  Original sample_id: {original.get('sample_id', 'N/A')}")
                    print(f"  Original question: {original.get('question', 'N/A')[:50]}...")
                    
                    # Show perturbation metadata
                    metadata = record.get('perturbation_metadata', {})
                    print(f"  Perturbation: {metadata.get('perturbation_class', 'N/A')} {metadata.get('intensity', 'N/A')}")
                    print(f"  Success: {metadata.get('generation_successful', False)}")
                    
                    # Show mapping
                    mapping = record.get('passage_mapping', {})
                    print(f"  Signal indices: {mapping.get('signal_indices', [])}")
                    print(f"  Noise indices: {mapping.get('noise_indices', [])}")
                    print()
                    
                except Exception as e:
                    print(f"  Error reading record {i+1}: {e}")
                    continue

def main():
    """Synchronous wrapper for the async main function."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()