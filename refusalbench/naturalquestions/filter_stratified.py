#!/usr/bin/env python3
"""
RefusalBench Stratified Sampling Filter
Ensures coverage across all 18 perturbation class × intensity combinations.
Targets ~400 samples per model with at least 20 samples per stratum.
Skips task matching to preserve sample diversity.
"""
import json
import argparse
import random
import hashlib
import os
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

class StratifiedRefusalBenchFilter:
    """
    Stratified sampling filter that ensures coverage across all 18 combinations
    of perturbation classes and intensities.
    """
    
    def __init__(self, agreement_mode: str = "unanimous", seed: int = 42):
        """Initialize the stratified filter."""
        if agreement_mode not in ["unanimous", "majority"]:
            raise ValueError("agreement_mode must be 'unanimous' or 'majority'")
        self.agreement_mode = agreement_mode
        self.seed = seed
        random.seed(seed)
        
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
        
        # Model name mapping
        self.generator_to_verifier_mapping = {
            "claude-sonnet-4": "claude",
            "deepseek-r1": "deepseek",
            "gpt-4o": "gpt",
            "amazon-nova-pro": "nova"
        }
        
        # Expected perturbation classes and intensities (from RefusalBench catalogue)
        self.perturbation_classes = [
            "P-Ambiguity", "P-Contradiction", "P-MissingInfo", 
            "P-FalsePremise", "P-GranularityMismatch", "P-EpistemicMismatch"
        ]
        self.intensities = ["LOW", "MEDIUM", "HIGH"]
        
        # Stratified sampling parameters
        self.target_samples_per_model = 400
        self.min_samples_per_stratum_per_model = 22  # 22 per stratum per model = 396, then 4 to fill
        self.num_models = len(self.generator_models)
        self.num_strata = len(self.perturbation_classes) * len(self.intensities)  # 18
        
        self.primary_key_counter = 0
        
        print("🎯 Stratified RefusalBench Filter")
        print(f"Agreement mode: {self.agreement_mode}")
        print(f"Target: {self.target_samples_per_model} samples per model")
        print(f"Minimum: {self.min_samples_per_stratum_per_model} samples per stratum PER MODEL")
        print(f"Strata: {self.num_strata} (6 classes × 3 intensities)")
        print(f"Expected: {self.num_strata} × {self.min_samples_per_stratum_per_model} = {self.num_strata * self.min_samples_per_stratum_per_model} base + {self.target_samples_per_model - self.num_strata * self.min_samples_per_stratum_per_model} fill = {self.target_samples_per_model} per model")
        print(f"Total target: {self.num_models} models × {self.target_samples_per_model} = {self.target_samples_per_model * self.num_models} samples")
        print(f"Random seed: {self.seed}")
    
    def get_generator_short_name(self, generator_model: str) -> str:
        """Get the short name of a generator model."""
        if generator_model in self.generator_to_verifier_mapping:
            return self.generator_to_verifier_mapping[generator_model]
        return generator_model.split('-')[0]
    
    def get_stratum_key(self, data: Dict) -> str:
        """Get the stratum key for a sample (perturbation_class × intensity)."""
        perturb_class = data.get('perturbation_class', 'unknown')
        intensity = data.get('intensity', 'unknown').upper()
        
        # The perturbation classes should already be in the correct format (P-ClassName)
        # No normalization needed since we're using the RefusalBench catalogue names
        
        return f"{perturb_class}_{intensity}"
    
    def create_within_folder_content_id(self, data: Dict) -> str:
        """Create unique identifier for matching content within a folder."""
        content_fields = [
            data.get("source_qid", ""),
            data.get("perturbed_query", ""),
            data.get("perturbed_context", ""),
            data.get("perturbation_class", ""),
            data.get("intensity", ""),
            data.get("lever_selected", "")
        ]
        return "|".join(str(field) for field in content_fields)
    
    def create_unique_primary_key(self, perturbation: Dict, generator_model: str) -> str:
        """Create a unique primary key for this sample."""
        source_qid = perturbation.get('source_qid', 'unknown')
        perturb_class = perturbation.get('perturbation_class', 'unknown')
        intensity = perturbation.get('intensity', 'unknown')
        
        generator_short = self.get_generator_short_name(generator_model)
        
        # Create content hash for uniqueness
        content_parts = [
            perturbation.get('perturbed_query', ''),
            perturbation.get('perturbed_context', ''),
            generator_model
        ]
        content_string = '|||'.join(content_parts)
        content_hash = hashlib.md5(content_string.encode('utf-8')).hexdigest()[:8]
        
        self.primary_key_counter += 1
        
        primary_key = f"RB_{generator_short}_{source_qid}_{perturb_class}_{intensity}_{content_hash}_{self.primary_key_counter:04d}"
        
        return primary_key
    
    def analyze_verifier_agreement(self, verifications: Dict[str, Dict]) -> Dict:
        """Analyze agreement patterns among verifiers."""
        agreement_analysis = {
            'total_verifiers': len(self.verifier_models),
            'responding_verifiers': 0,
            'successful_verifications': 0,
            'pass_votes': 0,
            'fail_votes': 0,
            'verifier_results': {},
            'has_unanimous_pass': False,
            'has_majority_pass': False,
            'has_all_respond': False
        }
        
        for verifier_name in self.verifier_models.keys():
            if verifier_name in verifications:
                verification = verifications[verifier_name]
                agreement_analysis['responding_verifiers'] += 1
                
                if verification.get('verification_successful', False):
                    agreement_analysis['successful_verifications'] += 1
                    result = verification.get('verification_response', {}).get('verification_result')
                    agreement_analysis['verifier_results'][verifier_name] = result
                    
                    if result == 'PASS':
                        agreement_analysis['pass_votes'] += 1
                    elif result == 'FAIL':
                        agreement_analysis['fail_votes'] += 1
                else:
                    agreement_analysis['verifier_results'][verifier_name] = 'ERROR'
            else:
                agreement_analysis['verifier_results'][verifier_name] = 'MISSING'
        
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
    
    def load_and_filter_folder_by_agreement_stratified(self, folder: str) -> Dict[str, List[Dict]]:
        """
        Load and filter folder by agreement, organizing samples by stratum.
        Returns: Dict[stratum_key, List[flattened_samples]]
        """
        print(f"\n📂 Processing {folder}...")
        
        # Load verification files
        raw_verifications = {}
        
        for verifier_name in self.verifier_models.keys():
            verification_file = f"{folder}/refusalbench_verifications_{verifier_name}.jsonl"
            
            verifications = []
            if Path(verification_file).exists():
                try:
                    with open(verification_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                verifications.append(data)
                            except:
                                pass
                    raw_verifications[verifier_name] = verifications
                except Exception as e:
                    print(f"    ❌ Error loading {verifier_name}: {e}")
        
        if not raw_verifications:
            print(f"    ❌ No verification files loaded for {folder}")
            return {}
        
        # Group by content ID
        content_groups = defaultdict(dict)
        
        for verifier_name, verifications in raw_verifications.items():
            for verification in verifications:
                content_id = self.create_within_folder_content_id(verification)
                content_groups[content_id][verifier_name] = verification
        
        # Filter by agreement and organize by stratum
        generator_model_full = self.generator_models.get(folder, folder.split('_')[1])
        generator_model_short = self.get_generator_short_name(generator_model_full)
        
        samples_by_stratum = defaultdict(list)
        
        stats = {
            'total_content_groups': len(content_groups),
            'complete_coverage': 0,
            'qualifying_samples': 0,
            'strata_covered': set(),
            'stratum_counts': defaultdict(int)
        }
        
        for content_id, verifier_group in content_groups.items():
            # Check if all verifiers responded
            if len(verifier_group) != len(self.verifier_models):
                continue
            
            stats['complete_coverage'] += 1
            
            # Analyze agreement
            agreement_analysis = self.analyze_verifier_agreement(verifier_group)
            
            # Check if meets criteria
            meets_criteria = False
            if self.agreement_mode == "unanimous" and agreement_analysis['has_unanimous_pass']:
                meets_criteria = True
            elif self.agreement_mode == "majority" and agreement_analysis['has_majority_pass']:
                meets_criteria = True
            
            if not meets_criteria:
                continue
            
            # Get base verification to determine stratum
            base_verification = next(iter(verifier_group.values()))
            stratum_key = self.get_stratum_key(base_verification)
            
            # Create flattened entry
            flattened_sample = self.create_flattened_entry_from_verifier_group(
                content_id, verifier_group, generator_model_full, generator_model_short,
                folder, agreement_analysis
            )
            
            samples_by_stratum[stratum_key].append(flattened_sample)
            stats['qualifying_samples'] += 1
            stats['strata_covered'].add(stratum_key)
            stats['stratum_counts'][stratum_key] += 1
        
        print(f"    📊 {folder} results:")
        print(f"      Total content groups: {stats['total_content_groups']}")
        print(f"      Complete coverage: {stats['complete_coverage']}")
        print(f"      Qualifying samples: {stats['qualifying_samples']}")
        print(f"      Strata covered: {len(stats['strata_covered'])}/{self.num_strata}")
        
        # Print per-stratum counts for this model
        print(f"\n      Per-stratum breakdown for {generator_model_short}:")
        for perturb_class in self.perturbation_classes:
            for intensity in self.intensities:
                stratum = f"{perturb_class}_{intensity}"
                count = stats['stratum_counts'].get(stratum, 0)
                print(f"        {stratum}: {count}")
        
        return samples_by_stratum
    
    def create_flattened_entry_from_verifier_group(self, content_id: str, verifier_group: Dict,
                                                  generator_model_full: str, generator_model_short: str,
                                                  folder: str, agreement_analysis: Dict) -> Dict:
        """Create a flattened entry from a verifier group."""
        
        # Get base verification data
        base_entry = next(iter(verifier_group.values()))
        
        # Collect verification results from all verifiers
        verification_results = {}
        for verifier_name, verification in verifier_group.items():
            verification_result = {
                'verification_successful': verification.get('verification_successful', False),
                'verification_result': None,
                'lever_correctly_implemented': None,
                'intensity_correctly_achieved': None,
                'uncertainty_successfully_induced': None,
                'verifier_model_full': verification.get('verifier_model_full', self.verifier_models.get(verifier_name))
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
        
        # Create unique ID
        unique_id = self.create_unique_primary_key(base_entry, generator_model_full)
        
        # Create flattened entry
        flattened_entry = {
            # IDENTIFIERS
            'unique_id': unique_id,
            'task_id': 'stratified_no_task_matching',
            'source_qid': base_entry.get('source_qid'),
            'generator_model': generator_model_short,
            'generator_model_full': generator_model_full,
            'source_folder': folder,
            'filtering_mode': self.agreement_mode,
            'sampling_strategy': 'stratified',
            
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
            
            # AGREEMENT ANALYSIS
            'agreement_analysis': {
                'verifier_agreement': {
                    verifier: {'result': agreement_analysis['verifier_results'][verifier]}
                    for verifier in self.verifier_models.keys()
                },
                'agreement_statistics': {
                    'total_valid_verifications': agreement_analysis['successful_verifications'],
                    'pass_votes': agreement_analysis['pass_votes'],
                    'fail_votes': agreement_analysis['fail_votes'],
                    'unanimous_pass': agreement_analysis['has_unanimous_pass'],
                    'majority_pass': agreement_analysis['has_majority_pass']
                }
            },
            
            # METADATA
            'total_verifiers': len(self.verifier_models),
            'content_matching_strategy': 'stratified_sampling_no_task_matching',
            'formatted_timestamp': datetime.datetime.now().isoformat()
        }
        
        return flattened_entry
    
    def apply_stratified_sampling(self, all_samples_by_stratum: Dict[str, Dict[str, List[Dict]]]) -> List[Dict]:
        """
        Apply stratified sampling with per-model constraints:
        - 22 samples per stratum per model (when available)
        - 400 total samples per model
        - Total: 1600 samples across 4 models
        """
        print(f"\n🎯 Applying stratified sampling...")
        
        # First, analyze what we have
        global_stratum_counts = defaultdict(int)
        model_stratum_counts = defaultdict(lambda: defaultdict(int))
        
        for model, strata in all_samples_by_stratum.items():
            for stratum, samples in strata.items():
                count = len(samples)
                global_stratum_counts[stratum] += count
                model_stratum_counts[model][stratum] = count
        
        # Print per-model statistics before sampling
        print(f"\n📊 Available samples per model (after agreement filtering):")
        for model in sorted(all_samples_by_stratum.keys()):
            total_samples = sum(len(samples) for samples in all_samples_by_stratum[model].values())
            num_strata = len(all_samples_by_stratum[model])
            print(f"  {model}: {total_samples} samples across {num_strata} strata")
        
        # Report coverage
        print(f"\n📊 Stratum coverage across all models:")
        all_strata = set()
        for perturb_class in self.perturbation_classes:
            for intensity in self.intensities:
                stratum = f"{perturb_class}_{intensity}"
                all_strata.add(stratum)
                count = global_stratum_counts.get(stratum, 0)
                # Check if we have enough for all models to get their quota
                min_needed = self.min_samples_per_stratum_per_model * len(all_samples_by_stratum)
                status = "✅" if count >= min_needed else "⚠️" if count > 0 else "❌"
                print(f"  {status} {stratum}: {count} samples")
        
        missing_strata = all_strata - set(global_stratum_counts.keys())
        if missing_strata:
            print(f"\n⚠️  Missing strata: {sorted(missing_strata)}")
        
        # NEW STRATEGY: Sample within each model independently
        samples_to_select = []
        
        print(f"\n🎯 Sampling strategy: {self.min_samples_per_stratum_per_model} samples per stratum per model")
        
        # Process each model separately
        for model in sorted(all_samples_by_stratum.keys()):
            print(f"\n🔄 Processing {model}...")
            model_samples = []
            model_used_ids = set()
            
            # Step 1: Try to get 22 samples from each stratum for this model
            stratum_quotas = {}
            for stratum in sorted(all_strata):
                if stratum in all_samples_by_stratum[model]:
                    available = all_samples_by_stratum[model][stratum]
                    n_to_take = min(len(available), self.min_samples_per_stratum_per_model)
                    
                    if n_to_take > 0:
                        selected = random.sample(available, n_to_take)
                        model_samples.extend(selected)
                        for s in selected:
                            model_used_ids.add(s['unique_id'])
                        stratum_quotas[stratum] = n_to_take
                        
                        if n_to_take < self.min_samples_per_stratum_per_model:
                            print(f"   ⚠️  {stratum}: only {n_to_take}/{self.min_samples_per_stratum_per_model} available")
                else:
                    stratum_quotas[stratum] = 0
                    print(f"   ❌ {stratum}: no samples available")
            
            print(f"   After stratum quotas: {len(model_samples)} samples")
            
            # Step 2: Fill remaining to reach 400
            remaining_target = self.target_samples_per_model - len(model_samples)
            if remaining_target > 0:
                # Collect all unused samples from this model
                unused_samples = []
                for stratum, samples in all_samples_by_stratum[model].items():
                    for sample in samples:
                        if sample['unique_id'] not in model_used_ids:
                            unused_samples.append(sample)
                
                # Take remaining samples
                n_to_take = min(len(unused_samples), remaining_target)
                if n_to_take > 0:
                    selected = random.sample(unused_samples, n_to_take)
                    model_samples.extend(selected)
                    print(f"   Added {n_to_take} samples to reach target")
                elif len(unused_samples) == 0:
                    print(f"   No more samples available (total: {len(model_samples)})")
                
            print(f"   Final count for {model}: {len(model_samples)} samples")
            samples_to_select.extend(model_samples)
        
        # Analyze final distribution
        final_stratum_counts = defaultdict(int)
        final_model_counts = defaultdict(int)
        
        for sample in samples_to_select:
            stratum = self.get_stratum_key(sample)
            model = sample['generator_model']
            final_stratum_counts[stratum] += 1
            final_model_counts[model] += 1
        
        print(f"\n📊 Final distribution by model:")
        for model in sorted(final_model_counts.keys()):
            print(f"  {model}: {final_model_counts[model]} samples")
        
        print(f"\n📊 Final distribution by stratum:")
        for stratum in sorted(all_strata):
            count = final_stratum_counts.get(stratum, 0)
            # Check if all models got their minimum
            min_expected = self.min_samples_per_stratum_per_model * len(final_model_counts)
            status = "✅" if count >= min_expected else "⚠️" if count > 0 else "❌"
            print(f"  {status} {stratum}: {count} samples")
        
        # Add detailed model × stratum breakdown
        print(f"\n📊 Detailed model × stratum breakdown:")
        model_stratum_breakdown = defaultdict(lambda: defaultdict(int))
        for sample in samples_to_select:
            stratum = self.get_stratum_key(sample)
            model = sample['generator_model']
            model_stratum_breakdown[model][stratum] += 1
        
        # Print header
        models = sorted(final_model_counts.keys())
        print(f"{'Stratum':<30} | " + " | ".join(f"{m:>8}" for m in models) + " | Total")
        print("-" * (31 + 11 * len(models) + 8))
        
        # Print each stratum
        for stratum in sorted(all_strata):
            row = f"{stratum:<30} |"
            for model in models:
                count = model_stratum_breakdown[model].get(stratum, 0)
                row += f" {count:>8} |"
            total = final_stratum_counts.get(stratum, 0)
            row += f" {total:>5}"
            print(row)
        
        # Print totals
        print("-" * (31 + 11 * len(models) + 8))
        row = f"{'TOTAL':<30} |"
        for model in models:
            row += f" {final_model_counts[model]:>8} |"
        row += f" {len(samples_to_select):>5}"
        print(row)
        
        print(f"\n✅ Total samples selected: {len(samples_to_select)}")
        
        return samples_to_select
    
    def create_stratified_dataset(self, stratified_samples: List[Dict],
                                 output_suffix: str = "_stratified",
                                 output_dir: str = "filtered_refusalbench_analysis"):
        """Create the stratified dataset output files."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🔗 Creating stratified dataset...")
        
        # Save dataset
        unified_filename = f"{output_dir}/unified_cross_model_dataset_flattened{output_suffix}.jsonl"
        
        with open(unified_filename, 'w', encoding='utf-8') as f:
            for entry in stratified_samples:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Calculate detailed statistics
        stratum_stats = {}
        model_stats = {}
        
        for sample in stratified_samples:
            stratum = self.get_stratum_key(sample)
            model = sample['generator_model']
            
            if stratum not in stratum_stats:
                stratum_stats[stratum] = {'count': 0, 'models': set()}
            stratum_stats[stratum]['count'] += 1
            stratum_stats[stratum]['models'].add(model)
            
            if model not in model_stats:
                model_stats[model] = {'count': 0, 'strata': set()}
            model_stats[model]['count'] += 1
            model_stats[model]['strata'].add(stratum)
        
        # Save metadata
        metadata = {
            'created_timestamp': datetime.datetime.now().isoformat(),
            'filtering_strategy': f'{self.agreement_mode}_agreement_stratified_sampling',
            'format': 'flattened_one_line_per_generator_implementation',
            'total_entries': len(stratified_samples),
            'sampling_parameters': {
                'target_samples_per_model': self.target_samples_per_model,
                'min_samples_per_stratum_per_model': self.min_samples_per_stratum_per_model,
                'num_models': self.num_models,
                'num_strata': self.num_strata
            },
            'task_matching': 'skipped_for_better_coverage',
            'random_seed': self.seed,
            'verifier_models': list(self.verifier_models.keys()),
            'generator_models': list(self.generator_models.keys()),
            'stratum_statistics': {
                stratum: {
                    'count': stats['count'],
                    'models_represented': len(stats['models']),
                    'model_list': sorted(list(stats['models']))
                }
                for stratum, stats in stratum_stats.items()
            },
            'model_statistics': {
                model: {
                    'count': stats['count'],
                    'strata_covered': len(stats['strata']),
                    'missing_strata': sorted(list(
                        set(f"{pc}_{i}" for pc in self.perturbation_classes for i in self.intensities) - 
                        stats['strata']
                    ))
                }
                for model, stats in model_stats.items()
            }
        }
        
        metadata_filename = f"{output_dir}/unified_dataset_metadata{output_suffix}.json"
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Create analysis report
        analysis_filename = f"{output_dir}/stratification_analysis{output_suffix}.json"
        analysis = {
            'created_timestamp': datetime.datetime.now().isoformat(),
            'coverage_analysis': {
                'total_strata': self.num_strata,
                'covered_strata': len(stratum_stats),
                'coverage_percentage': len(stratum_stats) / self.num_strata * 100,
                'strata_meeting_minimum': sum(1 for s in stratum_stats.values() if s['count'] >= self.min_samples_per_stratum_per_model * self.num_models),
                'strata_below_minimum': [
                    {'stratum': stratum, 'count': stats['count']}
                    for stratum, stats in stratum_stats.items()
                    if stats['count'] < self.min_samples_per_stratum_per_model * self.num_models
                ],
                'missing_strata': sorted(list(
                    set(f"{pc}_{i}" for pc in self.perturbation_classes for i in self.intensities) -
                    set(stratum_stats.keys())
                ))
            },
            'model_balance': {
                model: {
                    'sample_count': stats['count'],
                    'deviation_from_target': stats['count'] - self.target_samples_per_model,
                    'strata_coverage': f"{len(stats['strata'])}/{self.num_strata}"
                }
                for model, stats in model_stats.items()
            },
            'stratum_diversity': {
                stratum: {
                    'sample_count': stats['count'],
                    'model_diversity': f"{len(stats['models'])}/{self.num_models}",
                    'is_balanced': len(stats['models']) >= 3  # At least 3 out of 4 models
                }
                for stratum, stats in stratum_stats.items()
            }
        }
        
        with open(analysis_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"    ✅ Created {unified_filename}: {len(stratified_samples)} stratified samples")
        print(f"    ✅ Created {metadata_filename}: detailed metadata")
        print(f"    ✅ Created {analysis_filename}: stratification analysis")
        
        return unified_filename, len(stratified_samples)
    
    def run_stratified_pipeline(self, output_suffix: str = "_stratified",
                               output_dir: str = "filtered_refusalbench_analysis"):
        """Run the stratified sampling pipeline."""
        print(f"\n{'='*80}")
        print(f"RefusalBench STRATIFIED SAMPLING PIPELINE")
        print(f"Target: {self.target_samples_per_model} samples per model")
        print(f"Minimum: {self.min_samples_per_stratum_per_model} samples per stratum per model")
        print(f"Strategy: Skip task matching for better coverage")
        print(f"{'='*80}")
        
        try:
            # STEP 1: Load and filter each folder by agreement
            print(f"\n🔍 STEP 1: Agreement filtering and stratum organization...")
            
            all_samples_by_stratum = {}
            
            for folder in self.dataset_folders:
                if not Path(folder).exists():
                    print(f"⚠️  Folder {folder} not found, skipping...")
                    continue
                
                # Get samples organized by stratum
                samples_by_stratum = self.load_and_filter_folder_by_agreement_stratified(folder)
                
                if samples_by_stratum:
                    model_name = self.get_generator_short_name(
                        self.generator_models.get(folder, folder.split('_')[1])
                    )
                    all_samples_by_stratum[model_name] = samples_by_stratum
            
            if not all_samples_by_stratum:
                print("❌ No samples found after agreement filtering")
                return
            
            # STEP 2: Apply stratified sampling
            print(f"\n🔍 STEP 2: Applying stratified sampling...")
            stratified_samples = self.apply_stratified_sampling(all_samples_by_stratum)
            
            if not stratified_samples:
                print("❌ No samples selected during stratified sampling")
                return
            
            # STEP 3: Create output files
            print(f"\n🔍 STEP 3: Creating stratified dataset...")
            output_file, total_entries = self.create_stratified_dataset(
                stratified_samples, output_suffix, output_dir
            )
            
            print(f"\n✅ STRATIFIED SAMPLING PIPELINE COMPLETE!")
            print(f"📁 Output file: {output_file}")
            print(f"📊 Total entries: {total_entries}")
            print(f"\n🎯 SUCCESS: Stratified dataset with coverage across perturbation classes and intensities!")
            
        except Exception as e:
            print(f"❌ Error during pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    parser = argparse.ArgumentParser(description="RefusalBench Stratified Sampling Filter")
    parser.add_argument("--agreement-mode", choices=["unanimous", "majority"], 
                       default="unanimous", 
                       help="Agreement mode for filtering (default: unanimous)")
    parser.add_argument("--output-suffix", default="_stratified", 
                       help="Suffix for output files (default: _stratified)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling (default: 42)")
    parser.add_argument("--output-dir", default="filtered_refusalbench_analysis",
                       help="Output directory (default: filtered_refusalbench_analysis)")
    
    args = parser.parse_args()
    
    # Initialize and run the stratified filter
    filter_pipeline = StratifiedRefusalBenchFilter(
        agreement_mode=args.agreement_mode,
        seed=args.seed
    )
    
    filter_pipeline.run_stratified_pipeline(
        output_suffix=args.output_suffix,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()