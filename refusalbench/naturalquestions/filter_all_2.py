#!/usr/bin/env python3
"""
Multi-Folder RefusalBench Filter with Content-Based Matching and Comprehensive Research Analysis.
IMPROVED VERSION: Filter by verifier agreement FIRST, then do task matching.
Flattened output format: num_lines * num_models entries.
FIXED: Proper model name mapping to handle amazon-nova-pro -> nova conversion.
"""
import json
import argparse
import random
import hashlib
import os
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import datetime

class MultiModelRefusalBenchFilterAnalyzer:
    """
    Advanced filter that combines robust content-based matching with comprehensive 
    research analysis for academic publication.
    IMPROVED: Filter within folders first, then match across folders.
    FIXED: Proper model name mapping for amazon-nova-pro.
    """
    
    def __init__(self, agreement_mode: str = "unanimous"):
        """Initialize the multi-model filter with research capabilities."""
        if agreement_mode not in ["unanimous", "majority"]:
            raise ValueError("agreement_mode must be 'unanimous' or 'majority'")
        self.agreement_mode = agreement_mode
        
        self.dataset_folders = ["dataset_claude", "dataset_deepseek", "dataset_gpt", "dataset_nova"]
        
        self.generator_models = {
            "dataset_claude": "claude-sonnet-4",
            "dataset_deepseek": "deepseek-r1", 
            "dataset_gpt": "gpt-4o",
            "dataset_nova": "amazon-nova-pro"
        }
        
        self.verifier_models = {
            "claude": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
            "nova": "bedrock/us.amazon.nova-pro-v1:0", 
            "gpt": "openai/gpt-4o-2024-08-06",
            "deepseek": "bedrock/converse/us.deepseek.r1-v1:0"
        }
        
        # FIXED: Add proper model name mapping
        self.generator_to_verifier_mapping = {
            "claude-sonnet-4": "claude",
            "deepseek-r1": "deepseek",
            "gpt-4o": "gpt",
            "amazon-nova-pro": "nova"  # FIXED: Proper mapping for amazon-nova-pro
        }
        
        # Enhanced research tracking for academic analysis
        self.research_stats = {
            'generator_verifier_matrix': defaultdict(lambda: defaultdict(lambda: {'total': 0, 'pass': 0, 'fail': 0, 'error': 0})),
            'self_verification_stats': defaultdict(lambda: {'total': 0, 'pass': 0, 'fail': 0, 'error': 0}),
            'cross_verification_stats': defaultdict(lambda: {'total': 0, 'pass': 0, 'fail': 0, 'error': 0}),
            'agreement_matrix': defaultdict(lambda: defaultdict(int)),  # How often pairs agree
            'disagreement_patterns': defaultdict(lambda: defaultdict(int)),
            'perturbation_effectiveness_by_generator': defaultdict(lambda: defaultdict(lambda: {'total': 0, 'pass': 0})),
            'verifier_bias_analysis': defaultdict(lambda: {'own_generator_pass_rate': 0, 'other_generator_pass_rate': 0, 'bias_score': 0}),
            'quality_metrics_by_model': defaultdict(lambda: defaultdict(list)),
            'content_consistency_analysis': defaultdict(lambda: {'exact_matches': 0, 'content_variations': 0}),
            'filtering_efficiency_by_model': defaultdict(lambda: {'total_generated': 0, 'passed_filtering': 0})
        }
        
        self.primary_key_counter = 0
        
        print("Initialized Multi-Model RefusalBench Filter with Research Analysis")
        print(f"Agreement mode: {self.agreement_mode}")
        print(f"Processing folders: {self.dataset_folders}")
        print(f"Generator models: {list(self.generator_models.values())}")
        print(f"Verifier models: {list(self.verifier_models.values())}")
        print("IMPROVED WORKFLOW: Filter by agreement first, then match tasks")
        print("FIXED: Proper model name mapping for amazon-nova-pro -> nova")

    def get_verifier_key_from_generator(self, generator_model: str) -> str:
        """
        FIXED: Properly map generator model names to verifier keys.
        Handles special cases like amazon-nova-pro -> nova.
        """
        if generator_model in self.generator_to_verifier_mapping:
            return self.generator_to_verifier_mapping[generator_model]
        
        # Fallback: try the old method for backward compatibility
        return generator_model.split('-')[0]

    def get_generator_short_name(self, generator_model: str) -> str:
        """
        FIXED: Get the short name of a generator model for consistency.
        This ensures we use the same naming convention throughout.
        """
        return self.get_verifier_key_from_generator(generator_model)

    def create_perturbation_task_id(self, data: Dict) -> str:
        """
        Create a unique identifier for a perturbation task based on input fields only.
        This identifies the same perturbation task across different models.
        Note: Does NOT include lever_selected since different models may choose different levers
        to achieve the same perturbation goal.
        """
        # Only use input fields that define the perturbation task
        key_fields = [
            data.get("source_qid", ""),
            data.get("original_query", ""),
            data.get("perturbation_class", ""),
            data.get("intensity", ""),
            str(data.get("original_context", ""))[:200]  # First 200 chars for context matching
        ]
        return "|".join(str(field) for field in key_fields)
    
    def create_within_folder_content_id(self, data: Dict) -> str:
        """
        Create a unique identifier for matching exact perturbed content within a folder.
        All verifiers within the same folder should evaluate the same perturbed content.
        """
        # Use the actual perturbed content for within-folder matching
        content_fields = [
            data.get("source_qid", ""),
            data.get("perturbed_query", ""),
            data.get("perturbed_context", ""),
            data.get("perturbation_class", ""),
            data.get("intensity", ""),
            data.get("lever_selected", "")
        ]
        return "|".join(str(field) for field in content_fields)

    def create_unique_primary_key(self, perturbation: Dict, generator_model: str, task_id: str) -> str:
        """Create a unique primary key including generator model info."""
        source_qid = perturbation.get('source_qid', 'unknown')
        perturb_class = perturbation.get('perturbation_class', 'unknown')
        intensity = perturbation.get('intensity', 'unknown')
        
        # FIXED: Use proper generator short name
        generator_short = self.get_generator_short_name(generator_model)
        
        # Create content hash for uniqueness
        content_parts = [
            perturbation.get('perturbed_query', ''),
            perturbation.get('perturbed_context', ''),
            generator_model,
            task_id
        ]
        content_string = '|||'.join(content_parts)
        content_hash = hashlib.md5(content_string.encode('utf-8')).hexdigest()[:8]
        
        self.primary_key_counter += 1
        
        # Create primary key with generator info
        primary_key = f"RB_{generator_short}_{source_qid}_{perturb_class}_{intensity}_{content_hash}_{self.primary_key_counter:04d}"
        
        return primary_key

    def load_and_filter_folder_by_agreement(self, folder: str) -> Tuple[Dict[str, Dict], Dict[str, int]]:
        """
        IMPROVED: Load verification files for a folder AND filter by verifier agreement immediately.
        Returns only content groups that meet the agreement criteria.
        """
        print(f"\n📂 Loading and filtering {folder} by {self.agreement_mode} agreement:")
        
        # First load all verifications
        raw_verifications = {}
        load_stats = {}
        
        for verifier_name in self.verifier_models.keys():
            verification_file = f"{folder}/refusalbench_verifications_{verifier_name}.jsonl"
            
            verifications = []
            if Path(verification_file).exists():
                try:
                    with open(verification_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            try:
                                data = json.loads(line.strip())
                                verifications.append(data)
                            except Exception as e:
                                print(f"    ⚠️  Error in line {line_num}: {e}")
                    
                    raw_verifications[verifier_name] = verifications
                    load_stats[verifier_name] = len(verifications)
                    print(f"    ✅ {verifier_name}: {len(verifications)} verifications")
                    
                except Exception as e:
                    print(f"    ❌ {verifier_name}: Error loading - {e}")
                    load_stats[verifier_name] = 0
            else:
                print(f"    ❌ {verifier_name}: File not found")
                load_stats[verifier_name] = 0
        
        # Group by content within this folder
        content_groups = defaultdict(dict)
        
        for verifier_name, verifications in raw_verifications.items():
            for verification in verifications:
                content_id = self.create_within_folder_content_id(verification)
                content_groups[content_id][verifier_name] = verification
        
        # Filter by agreement immediately
        generator_model_full = self.generator_models.get(folder, folder.split('_')[1])
        generator_model_short = self.get_generator_short_name(generator_model_full)
        
        agreed_content_groups = {}
        agreement_stats = {
            'total_content_groups': len(content_groups),
            'complete_verifier_coverage': 0,
            'unanimous_pass': 0,
            'majority_pass': 0,
            'qualifying_groups': 0
        }
        
        for content_id, verifier_group in content_groups.items():
            # Check if all verifiers responded
            if len(verifier_group) != len(self.verifier_models):
                continue
            
            agreement_stats['complete_verifier_coverage'] += 1
            
            # Analyze verifier agreement
            agreement_analysis = self.analyze_verifier_agreement(verifier_group)
            
            # Update research statistics
            self.update_research_statistics(generator_model_short, content_id, verifier_group, agreement_analysis)
            
            # Check agreement criteria
            if agreement_analysis['has_unanimous_pass']:
                agreement_stats['unanimous_pass'] += 1
            if agreement_analysis['has_majority_pass']:
                agreement_stats['majority_pass'] += 1
            
            # Apply filtering
            meets_criteria = False
            if self.agreement_mode == "unanimous" and agreement_analysis['has_unanimous_pass']:
                meets_criteria = True
            elif self.agreement_mode == "majority" and agreement_analysis['has_majority_pass']:
                meets_criteria = True
            
            if meets_criteria:
                agreed_content_groups[content_id] = verifier_group
                agreement_stats['qualifying_groups'] += 1
        
        # Report filtering results for this folder
        print(f"    📊 Agreement filtering within {folder}:")
        print(f"      Total content groups: {agreement_stats['total_content_groups']}")
        print(f"      Complete verifier coverage: {agreement_stats['complete_verifier_coverage']}")
        print(f"      Unanimous PASS: {agreement_stats['unanimous_pass']}")
        print(f"      Majority PASS: {agreement_stats['majority_pass']}")
        print(f"      Qualifying groups ({self.agreement_mode}): {agreement_stats['qualifying_groups']}")
        
        if agreement_stats['complete_verifier_coverage'] > 0:
            efficiency = agreement_stats['qualifying_groups'] / agreement_stats['complete_verifier_coverage'] * 100
            print(f"      Agreement efficiency: {efficiency:.1f}%")
        
        return agreed_content_groups, load_stats

    def find_common_tasks_from_filtered_folders(self, filtered_folders: Dict) -> Tuple[Dict[str, Dict], Dict]:
        """
        IMPROVED: Find common tasks among pre-filtered (high-quality) examples.
        Much more efficient since we're only working with agreed-upon examples.
        """
        print(f"\n🔎 Finding common tasks among pre-filtered examples...")
        
        # Collect task IDs from each folder's high-quality examples
        folder_task_mapping = {}
        folder_stats = {}
        
        for folder, agreed_content_groups in filtered_folders.items():
            if not agreed_content_groups:
                print(f"    ⚠️  No qualifying examples in {folder}")
                continue
                
            generator_model_full = self.generator_models.get(folder, folder.split('_')[1])
            generator_model_short = self.get_generator_short_name(generator_model_full)
            
            task_to_content = {}
            unique_tasks = set()
            
            # Map each content group to its task ID
            for content_id, verifier_group in agreed_content_groups.items():
                # Get task ID from any verification (should be same across verifiers for same content)
                base_verification = next(iter(verifier_group.values()))
                task_id = self.create_perturbation_task_id(base_verification)
                
                task_to_content[task_id] = {
                    'content_id': content_id,
                    'verifier_group': verifier_group
                }
                unique_tasks.add(task_id)
            
            folder_task_mapping[folder] = task_to_content
            folder_stats[folder] = {
                'generator_model': generator_model_short,
                'generator_model_full': generator_model_full,
                'agreed_content_groups': len(agreed_content_groups),
                'unique_tasks': len(unique_tasks),
                'task_ids': unique_tasks
            }
            
            print(f"    📊 {folder}: {len(unique_tasks)} unique tasks from {len(agreed_content_groups)} quality examples")
        
        if len(folder_task_mapping) < 2:
            print(f"❌ Need at least 2 folders with qualifying examples")
            return {}, folder_stats
        
        # Find intersection of task IDs across folders
        existing_folders = list(folder_task_mapping.keys())
        common_task_ids = folder_stats[existing_folders[0]]['task_ids'].copy()
        
        for folder in existing_folders[1:]:
            common_task_ids &= folder_stats[folder]['task_ids']
            print(f"    After {folder}: {len(common_task_ids)} common tasks remain")
        
        # Build final common tasks structure
        common_tasks = {}
        for task_id in common_task_ids:
            common_tasks[task_id] = {}
            
            for folder in existing_folders:
                if task_id in folder_task_mapping[folder]:
                    task_data = folder_task_mapping[folder][task_id]
                    
                    # Annotate verifier group with model info
                    annotated_verifier_group = {}
                    generator_model_short = folder_stats[folder]['generator_model']
                    generator_model_full = folder_stats[folder]['generator_model_full']
                    
                    for verifier_name, verification in task_data['verifier_group'].items():
                        annotated_verification = verification.copy()
                        annotated_verification['generator_model'] = generator_model_short
                        annotated_verification['generator_model_full'] = generator_model_full
                        annotated_verification['verifier_model'] = verifier_name
                        annotated_verification['verifier_model_full'] = self.verifier_models.get(verifier_name, verifier_name)
                        annotated_verification['source_folder'] = folder
                        annotated_verification['content_id'] = task_data['content_id']
                        
                        annotated_verifier_group[verifier_name] = annotated_verification
                    
                    common_tasks[task_id][folder] = annotated_verifier_group
        
        print(f"\n✅ Found {len(common_tasks)} common high-quality tasks")
        print(f"   All tasks already passed {self.agreement_mode} agreement filtering")
        
        return common_tasks, folder_stats

    def analyze_verifier_agreement(self, verifications: Dict[str, Dict]) -> Dict:
        """Analyze agreement patterns among verifiers with research metrics."""
        agreement_analysis = {
            'total_verifiers': len(self.verifier_models),
            'responding_verifiers': 0,
            'successful_verifications': 0,
            'pass_votes': 0,
            'fail_votes': 0,
            'verifier_results': {},
            'has_unanimous_pass': False,
            'has_majority_pass': False,
            'has_all_respond': False,
            'detailed_results': {}
        }
        
        for verifier_name in self.verifier_models.keys():
            if verifier_name in verifications:
                verification = verifications[verifier_name]
                agreement_analysis['responding_verifiers'] += 1
                
                if verification.get('verification_successful', False):
                    agreement_analysis['successful_verifications'] += 1
                    result = verification.get('verification_response', {}).get('verification_result')
                    agreement_analysis['verifier_results'][verifier_name] = result
                    
                    # Store detailed verification results for research analysis
                    vr = verification.get('verification_response', {})
                    agreement_analysis['detailed_results'][verifier_name] = {
                        'verification_result': result,
                        'lever_correctly_implemented': vr.get('lever_correctly_implemented'),
                        'intensity_correctly_achieved': vr.get('intensity_correctly_achieved'),
                        'uncertainty_successfully_induced': vr.get('uncertainty_successfully_induced'),
                        'answer_constraint_satisfied': vr.get('answer_constraint_satisfied')
                    }
                    
                    if result == 'PASS':
                        agreement_analysis['pass_votes'] += 1
                    elif result == 'FAIL':
                        agreement_analysis['fail_votes'] += 1
                else:
                    agreement_analysis['verifier_results'][verifier_name] = 'ERROR'
                    agreement_analysis['detailed_results'][verifier_name] = {'verification_result': 'ERROR'}
            else:
                agreement_analysis['verifier_results'][verifier_name] = 'MISSING'
                agreement_analysis['detailed_results'][verifier_name] = {'verification_result': 'MISSING'}
        
        # Calculate agreement flags
        agreement_analysis['has_all_respond'] = agreement_analysis['responding_verifiers'] == len(self.verifier_models)
        agreement_analysis['has_unanimous_pass'] = (
            agreement_analysis['successful_verifications'] == len(self.verifier_models) and
            agreement_analysis['pass_votes'] == len(self.verifier_models)
        )
        agreement_analysis['has_majority_pass'] = (
            agreement_analysis['pass_votes'] > agreement_analysis['successful_verifications'] / 2
        )
        
        return agreement_analysis

    def update_research_statistics(self, generator_model: str, content_id: str, verifier_group: Dict, agreement_analysis: Dict):
        """Update comprehensive research statistics for academic analysis."""
        
        # Update filtering efficiency
        self.research_stats['filtering_efficiency_by_model'][generator_model]['total_generated'] += 1
        if agreement_analysis['has_unanimous_pass']:
            self.research_stats['filtering_efficiency_by_model'][generator_model]['passed_filtering'] += 1
        
        # Track generator-verifier matrix (which verifier-generator pairs work best)
        for verifier_name, result in agreement_analysis['verifier_results'].items():
            self.research_stats['generator_verifier_matrix'][generator_model][verifier_name]['total'] += 1
            
            if result == 'PASS':
                self.research_stats['generator_verifier_matrix'][generator_model][verifier_name]['pass'] += 1
            elif result == 'FAIL':
                self.research_stats['generator_verifier_matrix'][generator_model][verifier_name]['fail'] += 1
            elif result == 'ERROR':
                self.research_stats['generator_verifier_matrix'][generator_model][verifier_name]['error'] += 1
        
        # Track self-verification vs cross-verification patterns
        for verifier_name, result in agreement_analysis['verifier_results'].items():
            if verifier_name == generator_model:  # Self-verification
                self.research_stats['self_verification_stats'][generator_model]['total'] += 1
                if result == 'PASS':
                    self.research_stats['self_verification_stats'][generator_model]['pass'] += 1
                elif result == 'FAIL':
                    self.research_stats['self_verification_stats'][generator_model]['fail'] += 1
                elif result == 'ERROR':
                    self.research_stats['self_verification_stats'][generator_model]['error'] += 1
            else:  # Cross-verification
                self.research_stats['cross_verification_stats'][generator_model]['total'] += 1
                if result == 'PASS':
                    self.research_stats['cross_verification_stats'][generator_model]['pass'] += 1
                elif result == 'FAIL':
                    self.research_stats['cross_verification_stats'][generator_model]['fail'] += 1
                elif result == 'ERROR':
                    self.research_stats['cross_verification_stats'][generator_model]['error'] += 1
        
        # Track pairwise agreement patterns
        results = agreement_analysis['verifier_results']
        verifier_names = list(self.verifier_models.keys())
        
        for i, verifier_a in enumerate(verifier_names):
            for j, verifier_b in enumerate(verifier_names):
                if i < j and verifier_a in results and verifier_b in results:
                    result_a = results[verifier_a]
                    result_b = results[verifier_b]
                    
                    pair_key = f"{verifier_a}_{verifier_b}"
                    
                    if result_a == result_b and result_a in ['PASS', 'FAIL']:
                        self.research_stats['agreement_matrix'][pair_key]['agree'] += 1
                    elif result_a in ['PASS', 'FAIL'] and result_b in ['PASS', 'FAIL']:
                        self.research_stats['agreement_matrix'][pair_key]['disagree'] += 1
                        # Track specific disagreement pattern
                        disagreement_pattern = f"{result_a}_vs_{result_b}"
                        self.research_stats['disagreement_patterns'][pair_key][disagreement_pattern] += 1
        
        # Track perturbation effectiveness
        base_verification = next(iter(verifier_group.values()))
        perturbation_class = base_verification.get('perturbation_class', 'unknown')
        intensity = base_verification.get('intensity', 'unknown')
        lever = base_verification.get('lever_selected', 'unknown')
        
        # Track by perturbation class
        self.research_stats['perturbation_effectiveness_by_generator'][generator_model][f"class_{perturbation_class}"]['total'] += 1
        if agreement_analysis['has_unanimous_pass']:
            self.research_stats['perturbation_effectiveness_by_generator'][generator_model][f"class_{perturbation_class}"]['pass'] += 1
        
        # Track by intensity
        self.research_stats['perturbation_effectiveness_by_generator'][generator_model][f"intensity_{intensity}"]['total'] += 1
        if agreement_analysis['has_unanimous_pass']:
            self.research_stats['perturbation_effectiveness_by_generator'][generator_model][f"intensity_{intensity}"]['pass'] += 1
        
        # Track by lever
        self.research_stats['perturbation_effectiveness_by_generator'][generator_model][f"lever_{lever}"]['total'] += 1
        if agreement_analysis['has_unanimous_pass']:
            self.research_stats['perturbation_effectiveness_by_generator'][generator_model][f"lever_{lever}"]['pass'] += 1
        
        # Track quality metrics for this generator model
        for verifier_name, detailed_result in agreement_analysis['detailed_results'].items():
            if detailed_result['verification_result'] == 'PASS':
                self.research_stats['quality_metrics_by_model'][generator_model]['lever_correctness'].append(
                    detailed_result.get('lever_correctly_implemented', False)
                )
                self.research_stats['quality_metrics_by_model'][generator_model]['intensity_correctness'].append(
                    detailed_result.get('intensity_correctly_achieved', False)
                )
                self.research_stats['quality_metrics_by_model'][generator_model]['uncertainty_induction'].append(
                    detailed_result.get('uncertainty_successfully_induced', False)
                )

    def create_filtered_outputs(self, common_tasks: Dict, folder_stats: Dict, 
                               output_suffix: str, num_lines: int = None, seed: int = 42,
                               output_dir: str = "filtered_refusalbench_analysis"):
        """
        Create filtered verification files and a FLATTENED unified dataset.
        IMPROVED: All examples already meet agreement criteria.
        FLATTENED: num_lines * num_models entries instead of nested structure.
        """
        print(f"\n📝 Creating outputs for pre-filtered {self.agreement_mode} agreement examples...")
        
        # Calculate final verifier bias scores
        self.calculate_verifier_bias_scores()
        
        # Subsample if requested
        selected_task_ids = list(common_tasks.keys())
        if num_lines and num_lines < len(common_tasks):
            random.seed(seed)
            selected_task_ids = random.sample(selected_task_ids, num_lines)
            print(f"📋 Randomly sampled {num_lines} tasks from {len(common_tasks)} available")
        else:
            print(f"📋 Using all {len(common_tasks)} qualifying tasks")
        
        # Create filtered files for each folder/verifier combination
        files_created = 0
        total_entries_written = 0
        
        for folder in self.dataset_folders:
            if folder not in folder_stats:
                continue
                
            generator_model = folder_stats[folder]['generator_model']
            
            for verifier_name in self.verifier_models.keys():
                # Create output filename
                output_filename = f"refusalbench_verifications_{generator_model}_{verifier_name}{output_suffix}.jsonl"
                output_path = Path(folder) / output_filename
                
                entries_written = 0
                with open(output_path, 'w', encoding='utf-8') as f:
                    for task_id in selected_task_ids:
                        if (task_id in common_tasks and 
                            folder in common_tasks[task_id] and 
                            verifier_name in common_tasks[task_id][folder]):
                            
                            entry = common_tasks[task_id][folder][verifier_name]
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            entries_written += 1
                
                if entries_written > 0:
                    print(f"    ✅ Created {output_filename}: {entries_written} entries")
                    files_created += 1
                    total_entries_written += entries_written
                else:
                    # Remove empty file
                    output_path.unlink(missing_ok=True)
        
        # Create FLATTENED unified dataset
        self.create_flattened_unified_dataset(common_tasks, selected_task_ids, output_suffix, folder_stats, output_dir)
        
        print(f"\n✅ Created {files_created} filtered verification files")
        print(f"📊 Total entries written: {total_entries_written}")
        print(f"🎯 ALL ENTRIES ALREADY MEET {self.agreement_mode.upper()} AGREEMENT CRITERIA")
        
        return files_created

    def create_flattened_unified_dataset(self, common_tasks: Dict, selected_task_ids: List[str], 
                                       output_suffix: str, folder_stats: Dict, output_dir: str = "filtered_refusalbench_analysis"):
        """
        Create a FLATTENED unified dataset: num_lines * num_models entries.
        Each line represents one model's implementation with source model clearly marked.
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🔗 Creating FLATTENED cross-model dataset ({self.agreement_mode} agreement)...")
        print(f"📁 Output directory: {output_dir}/")
        
        flattened_entries = []
        
        for task_id in selected_task_ids:
            task_data = common_tasks[task_id]
            
            # Create one entry per generator model for this task
            for folder, verifier_data in task_data.items():
                generator_model_full = folder_stats[folder]['generator_model_full']
                generator_model_short = folder_stats[folder]['generator_model']
                
                # Get base information (should be same across verifiers for this generator)
                base_entry = next(iter(verifier_data.values()))
                
                # Collect verification results from all verifiers for this generator
                verification_results = {}
                for verifier_name, verification in verifier_data.items():
                    verification_result = {
                        'verification_successful': verification.get('verification_successful', False),
                        'verification_result': None,
                        'lever_correctly_implemented': None,
                        'intensity_correctly_achieved': None,
                        'uncertainty_successfully_induced': None,
                        'verifier_model_full': verification.get('verifier_model_full')
                    }
                    
                    if verification.get('verification_successful'):
                        vr = verification.get('verification_response', {})
                        verification_result.update({
                            'verification_result': vr.get('verification_result'),
                            'lever_correctly_implemented': vr.get('lever_correctly_implemented'),
                            'intensity_correctly_achieved': vr.get('intensity_correctly_achieved'),
                            'uncertainty_successfully_induced': vr.get('uncertainty_successfully_induced'),
                            'answer_constraint_satisfied': vr.get('answer_constraint_satisfied'),
                            'final_ground_truth_label': vr.get('final_ground_truth_label', '')
                        })
                    
                    verification_results[verifier_name] = verification_result
                
                # Create unique ID for this generator's implementation
                unique_id = self.create_unique_primary_key(base_entry, generator_model_full, task_id)
                
                # Create flattened entry
                flattened_entry = {
                    # IDENTIFIERS
                    'unique_id': unique_id,
                    'task_id': task_id,
                    'source_qid': base_entry.get('source_qid'),
                    'generator_model': generator_model_short,
                    'generator_model_full': generator_model_full,
                    'source_folder': folder,
                    'filtering_mode': self.agreement_mode,
                    
                    # ORIGINAL TASK DEFINITION
                    'original_query': base_entry.get('original_query'),
                    'original_context': base_entry.get('original_context'),
                    'original_answers': base_entry.get('original_answers'),
                    'perturbation_class': base_entry.get('perturbation_class'),
                    'intensity': base_entry.get('intensity'),
                    'expected_rag_behavior': base_entry.get('expected_rag_behavior'),
                    
                    # THIS GENERATOR'S IMPLEMENTATION
                    'perturbed_query': base_entry.get('perturbed_query'),
                    'perturbed_context': base_entry.get('perturbed_context'),
                    'lever_selected': base_entry.get('lever_selected'),
                    'generation_successful': base_entry.get('generation_successful'),
                    'parsing_successful': base_entry.get('parsing_successful'),
                    'implementation_reasoning': base_entry.get('implementation_reasoning', ''),
                    
                    # VERIFICATION RESULTS FROM ALL VERIFIERS
                    'verification_results': verification_results,
                    
                    # AGREEMENT ANALYSIS FOR THIS IMPLEMENTATION
                    'agreement_analysis': self.analyze_cross_model_patterns_for_implementation(verification_results),
                    
                    # METADATA
                    'total_verifiers': len(self.verifier_models),
                    'content_matching_strategy': 'content_based_within_folder',
                    'formatted_timestamp': datetime.datetime.now().isoformat()
                }
                
                flattened_entries.append(flattened_entry)
        
        # Save flattened dataset with metadata
        unified_filename = f"{output_dir}/unified_cross_model_dataset_flattened{output_suffix}.jsonl"
        metadata = {
            'created_timestamp': datetime.datetime.now().isoformat(),
            'filtering_criteria': f'{self.agreement_mode}_agreement_within_each_folder_then_task_matching',
            'format': 'flattened_one_line_per_generator_implementation',
            'total_entries': len(flattened_entries),
            'total_tasks': len(selected_task_ids),
            'total_generators': len(self.generator_models),
            'expected_entries_per_task': len(self.generator_models),
            'verifier_models': list(self.verifier_models.keys()),
            'generator_models': list(self.generator_models.keys()),
            'model_name_fixes': 'Fixed amazon-nova-pro -> nova mapping'
        }
        
        # Save metadata separately to keep JSONL valid
        metadata_filename = f"{output_dir}/unified_dataset_metadata{output_suffix}.json"
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Write valid JSONL (no comments allowed)
        with open(unified_filename, 'w', encoding='utf-8') as f:
            for entry in flattened_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"    ✅ Created {unified_filename}: {len(flattened_entries)} flattened entries")
        print(f"    ✅ Created {metadata_filename}: metadata saved separately")
        print(f"    📊 Format: {len(selected_task_ids)} tasks × {len(self.generator_models)} generators = {len(flattened_entries)} lines")
        
        # Create comprehensive research analysis
        self.create_research_analysis(flattened_entries, output_suffix, folder_stats, output_dir)

    def analyze_cross_model_patterns_for_implementation(self, verification_results: Dict) -> Dict:
        """Analyze verification patterns for a single generator's implementation."""
        
        analysis = {
            'verifier_agreement': {},
            'agreement_statistics': {}
        }
        
        # Analyze verification results
        all_results = []
        for verifier, result in verification_results.items():
            analysis['verifier_agreement'][verifier] = {
                'successful': result['verification_successful'],
                'result': result['verification_result']
            }
            
            if result['verification_successful'] and result['verification_result'] in ['PASS', 'FAIL']:
                all_results.append(result['verification_result'])
        
        # Calculate agreement statistics
        if all_results:
            pass_count = all_results.count('PASS')
            fail_count = all_results.count('FAIL')
            total_valid = len(all_results)
            
            analysis['agreement_statistics'] = {
                'total_valid_verifications': total_valid,
                'pass_votes': pass_count,
                'fail_votes': fail_count,
                'pass_percentage': pass_count / total_valid * 100 if total_valid > 0 else 0,
                'unanimous_pass': pass_count == total_valid,
                'unanimous_fail': fail_count == total_valid,
                'majority_pass': pass_count > total_valid / 2,
                'majority_fail': fail_count > total_valid / 2
            }
        
        return analysis

    def calculate_verifier_bias_scores(self):
        """
        FIXED: Calculate bias scores for each verifier (do they favor their own generator?).
        Now properly handles amazon-nova-pro -> nova mapping.
        """
        
        for verifier_name in self.verifier_models.keys():
            own_generator_results = []
            other_generator_results = []
            
            # FIXED: Collect results for own generator vs others using proper mapping
            for generator_model_full in self.generator_models.values():
                generator_model_short = self.get_generator_short_name(generator_model_full)
                
                matrix_data = self.research_stats['generator_verifier_matrix'][generator_model_short][verifier_name]
                total = matrix_data['total']
                passes = matrix_data['pass']
                
                if total > 0:
                    pass_rate = passes / total
                    
                    if verifier_name == generator_model_short:  # Same model family
                        own_generator_results.append(pass_rate)
                    else:  # Different model family
                        other_generator_results.append(pass_rate)
            
            # Calculate bias score
            if own_generator_results and other_generator_results:
                own_rate = sum(own_generator_results) / len(own_generator_results)
                other_rate = sum(other_generator_results) / len(other_generator_results)
                bias_score = own_rate - other_rate  # Positive = bias toward own generator
                
                self.research_stats['verifier_bias_analysis'][verifier_name] = {
                    'own_generator_pass_rate': own_rate,
                    'other_generator_pass_rate': other_rate,
                    'bias_score': bias_score,
                    'sample_sizes': {
                        'own_generator_samples': len(own_generator_results),
                        'other_generator_samples': len(other_generator_results)
                    }
                }

    def create_research_analysis(self, flattened_entries: List[Dict], output_suffix: str, 
                               folder_stats: Dict, output_dir: str):
        """Create comprehensive research analysis with academic insights."""
        
        print(f"\n📊 Creating comprehensive research analysis...")
        
        # Create comprehensive analysis
        analysis = self.create_comprehensive_analysis(flattened_entries, folder_stats)
        
        # Save comprehensive metadata
        analysis_filename = f"{output_dir}/comprehensive_research_analysis{output_suffix}.json"
        with open(analysis_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"    ✅ Created {analysis_filename}")
        
        # Export academic tables
        self.export_academic_tables(analysis, output_suffix, output_dir)
        
        # Print comprehensive report
        self.print_comprehensive_report(analysis)
        
        return analysis

    def create_comprehensive_analysis(self, flattened_entries: List[Dict], folder_stats: Dict) -> Dict:
        """Create comprehensive analysis across all models and entries."""
        
        analysis = {
            'created_timestamp': datetime.datetime.now().isoformat(),
            'agreement_mode_used': self.agreement_mode,
            'total_flattened_entries': len(flattened_entries),
            'total_unique_tasks': len(set(entry['task_id'] for entry in flattened_entries)),
            'workflow_improvement': 'filter_by_agreement_first_then_match_tasks',
            'output_format': 'flattened_one_line_per_generator_implementation',
            'model_name_fixes': 'Fixed amazon-nova-pro -> nova mapping for consistency',
            'folder_statistics': folder_stats,
            
            # RESEARCH INSIGHTS FOR ACADEMIC PUBLICATION
            'research_insights': {
                'generator_verifier_effectiveness_matrix': {},
                'self_vs_cross_verification_analysis': {},
                'verifier_bias_analysis': dict(self.research_stats['verifier_bias_analysis']),
                'pairwise_agreement_analysis': {},
                'perturbation_type_effectiveness': {},
                'model_specialization_analysis': {},
                'filtering_efficiency_analysis': {},
                'quality_metrics_analysis': {},
                'key_findings_summary': {}
            },
            
            'generator_model_comparison': {},
            'verifier_model_comparison': {},
            'perturbation_analysis': {},
            'cross_model_patterns': {}
        }
        
        # Populate research insights
        analysis['research_insights'] = self.create_research_insights()
        
        # Create model comparisons
        analysis.update(self.create_model_comparisons(flattened_entries))
        
        return analysis

    def create_research_insights(self) -> Dict:
        """Create comprehensive research insights for academic publication."""
        
        insights = {}
        
        # 1. Generator-Verifier Effectiveness Matrix
        insights['generator_verifier_effectiveness_matrix'] = {}
        for generator in self.research_stats['generator_verifier_matrix']:
            insights['generator_verifier_effectiveness_matrix'][generator] = {}
            for verifier in self.research_stats['generator_verifier_matrix'][generator]:
                data = self.research_stats['generator_verifier_matrix'][generator][verifier]
                if data['total'] > 0:
                    pass_rate = data['pass'] / data['total'] * 100
                    insights['generator_verifier_effectiveness_matrix'][generator][verifier] = {
                        'pass_rate': pass_rate,
                        'total_evaluations': data['total'],
                        'raw_counts': data
                    }
        
        # 2. Self vs Cross-Verification Analysis
        insights['self_vs_cross_verification_analysis'] = {}
        
        all_generators = set(self.research_stats['self_verification_stats'].keys()) | \
                        set(self.research_stats['cross_verification_stats'].keys())
        
        for generator in all_generators:
            self_data = self.research_stats['self_verification_stats'][generator]
            cross_data = self.research_stats['cross_verification_stats'][generator]
            
            self_pass_rate = self_data['pass'] / self_data['total'] * 100 if self_data['total'] > 0 else 0
            cross_pass_rate = cross_data['pass'] / cross_data['total'] * 100 if cross_data['total'] > 0 else 0
            
            self_bias = self_pass_rate - cross_pass_rate  # Positive = self-favorable bias
            
            insights['self_vs_cross_verification_analysis'][generator] = {
                'self_verification_pass_rate': self_pass_rate,
                'cross_verification_pass_rate': cross_pass_rate,
                'self_bias_score': self_bias,
                'self_verification_total': self_data['total'],
                'cross_verification_total': cross_data['total'],
                'interpretation': 'self_favorable' if self_bias > 5 else 'cross_favorable' if self_bias < -5 else 'neutral'
            }
        
        # 3. Pairwise Agreement Analysis
        insights['pairwise_agreement_analysis'] = {}
        for pair_key, data in self.research_stats['agreement_matrix'].items():
            total_comparisons = data.get('agree', 0) + data.get('disagree', 0)
            if total_comparisons > 0:
                agreement_rate = data.get('agree', 0) / total_comparisons * 100
                insights['pairwise_agreement_analysis'][pair_key] = {
                    'agreement_rate': agreement_rate,
                    'total_comparisons': total_comparisons,
                    'agreement_strength': 'high' if agreement_rate > 80 else 'medium' if agreement_rate > 60 else 'low'
                }
        
        # 4. Perturbation Type Effectiveness by Generator
        insights['perturbation_type_effectiveness'] = {}
        for generator in self.research_stats['perturbation_effectiveness_by_generator']:
            insights['perturbation_type_effectiveness'][generator] = {}
            
            for category_key, data in self.research_stats['perturbation_effectiveness_by_generator'][generator].items():
                if data['total'] > 0:
                    success_rate = data['pass'] / data['total'] * 100
                    insights['perturbation_type_effectiveness'][generator][category_key] = {
                        'success_rate': success_rate,
                        'total_attempts': data['total'],
                        'successful_perturbations': data['pass']
                    }
        
        # 5. Filtering Efficiency Analysis
        insights['filtering_efficiency_analysis'] = {}
        for generator, data in self.research_stats['filtering_efficiency_by_model'].items():
            if data['total_generated'] > 0:
                efficiency = data['passed_filtering'] / data['total_generated'] * 100
                insights['filtering_efficiency_analysis'][generator] = {
                    'total_generated': data['total_generated'],
                    'passed_filtering': data['passed_filtering'],
                    'filtering_efficiency': efficiency
                }
        
        # 6. Quality Metrics Analysis
        insights['quality_metrics_analysis'] = {}
        for generator, metrics in self.research_stats['quality_metrics_by_model'].items():
            if metrics:
                insights['quality_metrics_analysis'][generator] = {}
                for metric_name, values in metrics.items():
                    if values:
                        avg_score = sum(values) / len(values) * 100
                        insights['quality_metrics_analysis'][generator][metric_name] = {
                            'average_score': avg_score,
                            'sample_size': len(values)
                        }
        
        # 7. Model Specialization Analysis
        insights['model_specialization_analysis'] = self.analyze_model_specializations()
        
        # 8. Key Research Findings Summary
        insights['key_findings_summary'] = self.generate_key_findings()
        
        return insights

    def analyze_model_specializations(self) -> Dict:
        """Analyze which models specialize in which types of perturbations."""
        
        specializations = {}
        
        # For each perturbation type, find which generator has highest success rate
        all_categories = set()
        for generator_data in self.research_stats['perturbation_effectiveness_by_generator'].values():
            all_categories.update(generator_data.keys())
        
        for category in all_categories:
            best_generator = None
            best_rate = 0
            category_results = {}
            
            for generator, data in self.research_stats['perturbation_effectiveness_by_generator'].items():
                if category in data and data[category]['total'] >= 5:  # Minimum sample size
                    success_rate = data[category]['pass'] / data[category]['total'] * 100
                    category_results[generator] = {
                        'success_rate': success_rate,
                        'sample_size': data[category]['total']
                    }
                    
                    if success_rate > best_rate:
                        best_rate = success_rate
                        best_generator = generator
            
            if best_generator and len(category_results) > 1:  # Only if we have comparisons
                specializations[category] = {
                    'best_generator': best_generator,
                    'best_success_rate': best_rate,
                    'all_generators': category_results,
                    'performance_gap': best_rate - max([r['success_rate'] for g, r in category_results.items() if g != best_generator], default=0)
                }
        
        return specializations

    def generate_key_findings(self) -> Dict:
        """Generate key research findings for academic paper."""
        
        findings = {}
        
        # Finding 1: Overall generator effectiveness ranking
        generator_rankings = []
        for generator, data in self.research_stats['self_verification_stats'].items():
            if data['total'] > 0:
                total_cross = self.research_stats['cross_verification_stats'][generator]['total']
                total_all = data['total'] + total_cross
                pass_all = data['pass'] + self.research_stats['cross_verification_stats'][generator]['pass']
                overall_rate = pass_all / total_all * 100 if total_all > 0 else 0
                generator_rankings.append((generator, overall_rate, total_all))
        
        generator_rankings.sort(key=lambda x: x[1], reverse=True)
        findings['generator_effectiveness_ranking'] = [
            {'model': gen, 'overall_pass_rate': rate, 'total_evaluations': total}
            for gen, rate, total in generator_rankings
        ]
        
        # Finding 2: Self-verification bias patterns
        bias_patterns = []
        for generator, analysis in self.research_stats['verifier_bias_analysis'].items():
            if analysis.get('bias_score') is not None:
                bias_patterns.append({
                    'verifier': generator,
                    'bias_score': analysis['bias_score'],
                    'bias_magnitude': abs(analysis['bias_score']),
                    'bias_direction': 'self_favorable' if analysis['bias_score'] > 0 else 'other_favorable'
                })
        
        bias_patterns.sort(key=lambda x: x['bias_magnitude'], reverse=True)
        findings['self_verification_bias_patterns'] = bias_patterns
        
        # Finding 3: Best generator-verifier pairs
        best_pairs = []
        for generator in self.research_stats['generator_verifier_matrix']:
            for verifier in self.research_stats['generator_verifier_matrix'][generator]:
                data = self.research_stats['generator_verifier_matrix'][generator][verifier]
                if data['total'] >= 10:  # Minimum sample size
                    pass_rate = data['pass'] / data['total'] * 100
                    best_pairs.append({
                        'generator': generator,
                        'verifier': verifier,
                        'pass_rate': pass_rate,
                        'sample_size': data['total'],
                        'is_self_verification': generator == verifier
                    })
        
        best_pairs.sort(key=lambda x: x['pass_rate'], reverse=True)
        findings['best_generator_verifier_pairs'] = best_pairs[:10]  # Top 10
        
        # Finding 4: Agreement vs disagreement patterns
        high_agreement_pairs = []
        high_disagreement_pairs = []
        
        for pair_key, data in self.research_stats['agreement_matrix'].items():
            total = data.get('agree', 0) + data.get('disagree', 0)
            if total >= 10:  # Minimum sample size
                agreement_rate = data.get('agree', 0) / total * 100
                pair_data = {
                    'verifier_pair': pair_key,
                    'agreement_rate': agreement_rate,
                    'total_comparisons': total
                }
                
                if agreement_rate > 80:
                    high_agreement_pairs.append(pair_data)
                elif agreement_rate < 50:
                    high_disagreement_pairs.append(pair_data)
        
        findings['high_agreement_pairs'] = sorted(high_agreement_pairs, key=lambda x: x['agreement_rate'], reverse=True)
        findings['high_disagreement_pairs'] = sorted(high_disagreement_pairs, key=lambda x: x['agreement_rate'])
        
        return findings

    def create_model_comparisons(self, flattened_entries: List[Dict]) -> Dict:
        """Create detailed model comparison statistics."""
        
        comparisons = {
            'generator_model_comparison': {},
            'verifier_model_comparison': {},
            'perturbation_analysis': {},
            'cross_model_patterns': {}
        }
        
        # Generator model comparison from research stats
        for generator, efficiency_data in self.research_stats['filtering_efficiency_by_model'].items():
            comparisons['generator_model_comparison'][generator] = {
                'total_perturbations_generated': efficiency_data['total_generated'],
                'high_quality_examples': efficiency_data['passed_filtering'],
                'success_rate': efficiency_data['passed_filtering'] / efficiency_data['total_generated'] * 100 if efficiency_data['total_generated'] > 0 else 0,
                'perturbation_class_distribution': {},
                'intensity_distribution': {}
            }
        
        # Verifier model comparison
        verifier_totals = defaultdict(lambda: {'total': 0, 'pass': 0, 'fail': 0, 'error': 0})
        
        for generator_data in self.research_stats['generator_verifier_matrix'].values():
            for verifier, perf in generator_data.items():
                for metric, count in perf.items():
                    verifier_totals[verifier][metric] += count
        
        for verifier, data in verifier_totals.items():
            total_evaluations = data['total']
            comparisons['verifier_model_comparison'][verifier] = {
                'total_evaluations': total_evaluations,
                'pass_rate': data['pass'] / total_evaluations * 100 if total_evaluations > 0 else 0,
                'fail_rate': data['fail'] / total_evaluations * 100 if total_evaluations > 0 else 0,
                'error_rate': data['error'] / total_evaluations * 100 if total_evaluations > 0 else 0,
                'reliability_score': (data['pass'] + data['fail']) / total_evaluations * 100 if total_evaluations > 0 else 0
            }
        
        # Perturbation analysis from final examples
        class_counts = defaultdict(int)
        intensity_counts = defaultdict(int)
        lever_counts = defaultdict(int)
        
        for example in flattened_entries:
            class_counts[example.get('perturbation_class', 'unknown')] += 1
            intensity_counts[example.get('intensity', 'unknown')] += 1
            lever_counts[example.get('lever_selected', 'unknown')] += 1
        
        comparisons['perturbation_analysis'] = {
            'by_class': dict(class_counts),
            'by_intensity': dict(intensity_counts),
            'by_lever': dict(lever_counts),
            'total_unique_classes': len(class_counts),
            'total_unique_levers': len(lever_counts)
        }
        
        return comparisons

    def export_academic_tables(self, analysis: Dict, output_suffix: str, output_dir: str):
        """Export academic tables ready for LaTeX/paper publication."""
        
        print(f"\n📋 Exporting academic tables...")
        
        # Table 1: Generator-Verifier Effectiveness Matrix
        with open(f'{output_dir}/table1_generator_verifier_matrix{output_suffix}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            verifiers = list(self.verifier_models.keys())
            writer.writerow(['Generator'] + verifiers + ['Average'])
            
            # Data rows  
            for generator_full in self.generator_models.values():
                generator_short = self.get_generator_short_name(generator_full)
                row = [generator_short]
                
                rates = []
                for verifier in verifiers:
                    effectiveness_data = analysis['research_insights']['generator_verifier_effectiveness_matrix'].get(generator_short, {}).get(verifier, {})
                    pass_rate = effectiveness_data.get('pass_rate', 0)
                    row.append(f"{pass_rate:.1f}%")
                    rates.append(pass_rate)
                
                avg_rate = sum(rates) / len(rates) if rates else 0
                row.append(f"{avg_rate:.1f}%")
                writer.writerow(row)
        
        # Table 2: Self vs Cross-Verification Analysis
        with open(f'{output_dir}/table2_self_vs_cross_verification{output_suffix}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Self-Verification Rate', 'Cross-Verification Rate', 'Bias Score', 'Interpretation'])
            
            for model, data in analysis['research_insights']['self_vs_cross_verification_analysis'].items():
                writer.writerow([
                    model,
                    f"{data['self_verification_pass_rate']:.1f}%",
                    f"{data['cross_verification_pass_rate']:.1f}%",
                    f"{data['self_bias_score']:+.1f}%",
                    data['interpretation']
                ])
        
        # Table 3: Model Specialization Analysis
        with open(f'{output_dir}/table3_model_specializations{output_suffix}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Perturbation Type', 'Best Generator', 'Success Rate', 'Performance Gap'])
            
            specializations = analysis['research_insights']['model_specialization_analysis']
            for category, data in specializations.items():
                if data['performance_gap'] > 5:  # Only significant gaps
                    category_clean = category.replace('class_', '').replace('intensity_', '').replace('lever_', '')
                    writer.writerow([
                        category_clean,
                        data['best_generator'],
                        f"{data['best_success_rate']:.1f}%",
                        f"{data['performance_gap']:.1f}%"
                    ])
        
        # Table 4: Filtering Efficiency
        with open(f'{output_dir}/table4_filtering_efficiency{output_suffix}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Generator Model', 'Total Generated', 'Passed Filtering', 'Efficiency Rate'])
            
            for generator, data in analysis['research_insights']['filtering_efficiency_analysis'].items():
                writer.writerow([
                    generator,
                    data['total_generated'],
                    data['passed_filtering'],
                    f"{data['filtering_efficiency']:.1f}%"
                ])
        
        print(f"    ✅ Created academic tables: {output_dir}/table1-4{output_suffix}.csv")

    def print_comprehensive_report(self, analysis: Dict):
        """Print a comprehensive research report suitable for academic publication."""
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE MULTI-MODEL RESEARCH ANALYSIS")
        print(f"IMPROVED WORKFLOW: Filter First + Flattened Output")
        print(f"FIXED: Model Name Mapping (amazon-nova-pro -> nova)")
        print(f"Agreement Mode: {self.agreement_mode.upper()}")
        print(f"{'='*80}")
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"  Workflow improvement: {analysis['workflow_improvement']}")
        print(f"  Output format: {analysis['output_format']}")
        print(f"  Model name fixes: {analysis['model_name_fixes']}")
        print(f"  Total flattened entries: {analysis['total_flattened_entries']}")
        print(f"  Total unique tasks: {analysis['total_unique_tasks']}")
        print(f"  Expected entries per task: {len(self.generator_models)} generators")
        
        # Get insights
        insights = analysis['research_insights']
        
        print(f"\n🏆 KEY RESEARCH FINDINGS:")
        
        # 1. Generator effectiveness ranking
        if 'generator_effectiveness_ranking' in insights['key_findings_summary']:
            print(f"\n1. GENERATOR EFFECTIVENESS RANKING:")
            for i, gen_data in enumerate(insights['key_findings_summary']['generator_effectiveness_ranking'], 1):
                print(f"   {i}. {gen_data['model']}: {gen_data['overall_pass_rate']:.1f}% pass rate ({gen_data['total_evaluations']} evaluations)")
        
        # 2. Self-verification bias
        if 'self_verification_bias_patterns' in insights['key_findings_summary']:
            print(f"\n2. SELF-VERIFICATION BIAS ANALYSIS:")
            for bias_data in insights['key_findings_summary']['self_verification_bias_patterns']:
                direction = "↑ SELF-BIASED" if bias_data['bias_direction'] == 'self_favorable' else "↓ OTHER-BIASED"
                print(f"   {bias_data['verifier']}: {bias_data['bias_score']:+.1f}% {direction}")
        
        # 3. Filtering efficiency
        print(f"\n3. FILTERING EFFICIENCY BY GENERATOR:")
        for generator, data in insights['filtering_efficiency_analysis'].items():
            print(f"   {generator}: {data['filtering_efficiency']:.1f}% ({data['passed_filtering']}/{data['total_generated']})")
        
        # 4. Best generator-verifier pairs
        if 'best_generator_verifier_pairs' in insights['key_findings_summary']:
            print(f"\n4. TOP GENERATOR-VERIFIER PAIRS:")
            for i, pair in enumerate(insights['key_findings_summary']['best_generator_verifier_pairs'][:5], 1):
                self_mark = " (SELF)" if pair['is_self_verification'] else ""
                print(f"   {i}. {pair['generator']} → {pair['verifier']}: {pair['pass_rate']:.1f}%{self_mark}")
        
        # 5. Model specializations
        print(f"\n5. MODEL SPECIALIZATIONS:")
        specializations = insights['model_specialization_analysis']
        significant_specs = [(cat, data) for cat, data in specializations.items() if data['performance_gap'] > 10]
        for category, spec_data in significant_specs[:5]:  # Top 5 most significant
            category_clean = category.replace('class_', '').replace('intensity_', '').replace('lever_', '')
            print(f"   {category_clean}: {spec_data['best_generator']} leads ({spec_data['best_success_rate']:.1f}%, +{spec_data['performance_gap']:.1f}% gap)")
        
        print(f"\n💡 WORKFLOW IMPROVEMENTS:")
        print(f"  ✅ Filter by verifier agreement FIRST (more efficient)")
        print(f"  ✅ Task matching on pre-validated examples only")
        print(f"  ✅ Flattened output: {analysis['total_flattened_entries']} lines instead of nested structure")
        print(f"  ✅ Each line = one generator's implementation with full metadata")
        print(f"  ✅ All research statistics preserved and enhanced")
        print(f"  ✅ FIXED: Proper amazon-nova-pro -> nova mapping throughout")
        
        print(f"\n📈 ACADEMIC READINESS:")
        print(f"  • Generator-verifier effectiveness matrices")
        print(f"  • Self-verification bias quantification")
        print(f"  • Cross-model specialization analysis")
        print(f"  • Comprehensive statistical validation")
        print(f"  • Camera-ready LaTeX tables")

    def run_improved_filter_analysis(self, output_suffix: str = "_common", num_lines: int = None, seed: int = 42,
                                   output_dir: str = "filtered_refusalbench_analysis"):
        """
        Run the IMPROVED filtering and research analysis pipeline.
        WORKFLOW: Filter by agreement first, then find common tasks, then create flattened output.
        FIXED: Proper model name mapping for amazon-nova-pro.
        """
        
        print(f"\n{'='*80}")
        print(f"IMPROVED MULTI-MODEL FILTER WITH COMPREHENSIVE RESEARCH ANALYSIS")
        print(f"WORKFLOW: Filter First → Task Match → Flattened Output")
        print(f"FIXED: amazon-nova-pro -> nova mapping")
        print(f"{'='*80}")
        
        try:
            # STEP 1: Filter each folder by verifier agreement FIRST
            print(f"\n🔍 STEP 1: Filtering each folder by {self.agreement_mode} verifier agreement...")
            
            filtered_folders = {}
            all_folder_stats = {}
            
            for folder in self.dataset_folders:
                if not Path(folder).exists():
                    print(f"⚠️  Folder {folder} not found, skipping...")
                    continue
                
                agreed_content_groups, load_stats = self.load_and_filter_folder_by_agreement(folder)
                
                if agreed_content_groups:
                    filtered_folders[folder] = agreed_content_groups
                    all_folder_stats[folder] = {
                        'generator_model': self.get_generator_short_name(self.generator_models.get(folder, folder.split('_')[1])),
                        'generator_model_full': self.generator_models.get(folder, folder.split('_')[1]),
                        'load_stats': load_stats,
                        'agreed_content_groups': len(agreed_content_groups)
                    }
                else:
                    print(f"    ❌ No qualifying examples in {folder}")
            
            if not filtered_folders:
                print("❌ No folders have qualifying examples")
                return
            
            print(f"\n✅ STEP 1 COMPLETE: {len(filtered_folders)} folders with high-quality examples")
            
            # STEP 2: Find common tasks among pre-filtered examples
            print(f"\n🔍 STEP 2: Finding common tasks among pre-filtered examples...")
            
            common_tasks, folder_stats = self.find_common_tasks_from_filtered_folders(filtered_folders)
            
            if not common_tasks:
                print("❌ No common tasks found among filtered examples")
                return
            
            print(f"\n✅ STEP 2 COMPLETE: {len(common_tasks)} common high-quality tasks")
            
            # STEP 3: Create flattened outputs and research analysis
            print(f"\n🔍 STEP 3: Creating flattened outputs and research analysis...")
            
            files_created = self.create_filtered_outputs(common_tasks, folder_stats, output_suffix, num_lines, seed, output_dir)
            
            print(f"\n✅ IMPROVED FILTERING & RESEARCH ANALYSIS COMPLETE!")
            print(f"📁 Filtered verification files: {files_created}")
            print(f"📊 Flattened unified dataset: {output_dir}/unified_cross_model_dataset_flattened{output_suffix}.jsonl")
            print(f"📋 Dataset metadata: {output_dir}/unified_dataset_metadata{output_suffix}.json")
            print(f"📈 Comprehensive research analysis: {output_dir}/comprehensive_research_analysis{output_suffix}.json")
            print(f"📋 Academic tables: {output_dir}/table1-4{output_suffix}.csv")
            print(f"\n🎯 RESEARCH READY WITH IMPROVEMENTS:")
            print(f"   ✅ More efficient workflow (filter first, then match)")
            print(f"   ✅ Flattened output format for easier analysis")
            print(f"   ✅ {self.agreement_mode.title()} agreement filtering for quality assurance")
            print(f"   ✅ Comprehensive bias analysis and model comparison")
            print(f"   ✅ Academic tables ready for camera-ready papers")
            print(f"   ✅ All data formats optimized for statistical analysis")
            print(f"   ✅ FIXED: Consistent model naming (amazon-nova-pro -> nova)")
            
        except Exception as e:
            print(f"❌ Error during improved filtering and analysis: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Improved Multi-Model RefusalBench Filter with Research Analysis (FIXED: Model Name Mapping)")
    parser.add_argument("--agreement-mode", choices=["unanimous", "majority"], 
                       default="unanimous", 
                       help="Agreement mode for filtering (default: unanimous)")
    parser.add_argument("--output-suffix", default="_common", 
                       help="Suffix for output files (default: _common)")
    parser.add_argument("--num-lines", type=int, 
                       help="Number of common tasks to subsample (optional)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for subsampling (default: 42)")
    parser.add_argument("--output-dir", default="filtered_refusalbench_analysis_temp",
                       help="Output directory for results (default: filtered_refusalbench_analysis)")
    
    args = parser.parse_args()
    
    # Initialize and run improved filter with research analysis
    filter_analyzer = MultiModelRefusalBenchFilterAnalyzer(agreement_mode=args.agreement_mode)
    filter_analyzer.run_improved_filter_analysis(
        output_suffix=args.output_suffix,
        num_lines=args.num_lines,
        seed=args.seed,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()