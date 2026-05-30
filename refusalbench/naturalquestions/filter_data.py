#!/usr/bin/env python3
"""
Enhanced RefusalBench Perturbation Filter with comprehensive metadata and unique IDs.
Takes verification outputs, filters based on agreement, and outputs in enhanced reference format.
"""
import json
import os
import hashlib
from typing import Dict, List
from pathlib import Path
from collections import defaultdict
import datetime

class EnhancedRefusalBenchReferenceFormatter:
    """
    Enhanced formatter that includes all metadata and creates unique primary keys.
    """
    
    def __init__(self, agreement_mode: str = "majority_agree"):
        """Initialize with agreement mode ('majority_agree' or 'all_agree')."""
        if agreement_mode not in ["majority_agree", "all_agree"]:
            raise ValueError("agreement_mode must be 'majority_agree' or 'all_agree'")
        
        self.agreement_mode = agreement_mode
        self.primary_key_counter = 0
        print(f"Initialized enhanced formatter with agreement mode: {agreement_mode}")
    
    def create_unique_primary_key(self, perturbation: Dict) -> str:
        """
        Create a unique primary key for each perturbation.
        Format: RB_{source_qid}_{perturbation_class}_{intensity}_{unique_hash}
        """
        source_qid = perturbation.get('source_qid', 'unknown')
        perturb_class = perturbation.get('perturbation_class', 'unknown')
        intensity = perturbation.get('intensity', 'unknown')
        lever = perturbation.get('lever_selected', 'unknown')
        
        # Create content hash for uniqueness
        content_parts = [
            perturbation.get('perturbed_query', ''),
            perturbation.get('perturbed_context', ''),
            perturbation.get('implementation_reasoning', ''),
            lever
        ]
        content_string = '|||'.join(content_parts)
        content_hash = hashlib.md5(content_string.encode('utf-8')).hexdigest()[:8]
        
        # Increment counter for absolute uniqueness
        self.primary_key_counter += 1
        
        # Create primary key
        primary_key = f"RB_{source_qid}_{perturb_class}_{intensity}_{content_hash}_{self.primary_key_counter:04d}"
        
        return primary_key
    
    def load_original_data(self, original_file: str) -> Dict[str, Dict]:
        """Load original reference dataset indexed by qid."""
        print(f"Loading original data from {original_file}")
        
        original_data = {}
        with open(original_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    qid = data.get('qid')
                    if qid:
                        original_data[qid] = data
                except Exception as e:
                    print(f"Error on line {line_num}: {e}")
        
        print(f"Loaded {len(original_data)} original examples")
        return original_data
    
    def load_verification_files(self, input_files: List[str]) -> Dict[str, List[Dict]]:
        """Load all verification files."""
        all_verifications = {}
        
        for i, input_file in enumerate(input_files):
            verifier_id = f"verifier_{i+1}"
            print(f"Loading {verifier_id} from {input_file}")
            
            verifications = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        verifications.append(data)
                    except Exception as e:
                        print(f"Error in {input_file} line {line_num}: {e}")
            
            all_verifications[verifier_id] = verifications
            print(f"Loaded {len(verifications)} verifications")
        
        return all_verifications
    
    def create_perturbation_key(self, perturbation: Dict) -> str:
        """Create unique key for matching perturbations across verifiers."""
        source_qid = perturbation.get('source_qid', 'unknown')
        perturb_class = perturbation.get('perturbation_class', 'unknown')
        intensity = perturbation.get('intensity', 'unknown')
        lever = perturbation.get('lever_selected', 'unknown')
        
        # Use source line number if available for additional uniqueness
        source_line = perturbation.get('source_line', 'unknown')
        
        return f"{source_qid}_{perturb_class}_{intensity}_{lever}_{source_line}"
    
    def group_perturbations(self, all_verifications: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Group perturbations by unique key across all verifiers."""
        perturbation_groups = defaultdict(dict)
        
        for verifier_id, verifications in all_verifications.items():
            for verification in verifications:
                key = self.create_perturbation_key(verification)
                perturbation_groups[key][verifier_id] = verification
        
        return perturbation_groups
    
    def check_agreement(self, verifications: List[Dict]) -> bool:
        """Check if verifications meet agreement criteria for PASS."""
        pass_count = 0
        valid_count = 0
        
        for verification in verifications:
            if verification.get('verification_successful', False):
                valid_count += 1
                if verification.get('verification_response', {}).get('verification_result') == 'PASS':
                    pass_count += 1
        
        if valid_count == 0:
            return False
        
        if self.agreement_mode == "all_agree":
            return pass_count == valid_count
        else:  # majority_agree
            return pass_count > (valid_count / 2)
    
    def create_enhanced_reference_example(self, perturbation: Dict, original: Dict) -> Dict:
        """
        Convert perturbation to enhanced reference format with comprehensive metadata.
        """
        # Create unique primary key
        unique_id = self.create_unique_primary_key(perturbation)
        
        # Get original retrieved docs and replace first with perturbed context
        original_retrieved = original.get('retrieved_docs', [])
        new_retrieved_docs = [perturbation.get('perturbed_context', '')]
        
        # Add remaining docs (skip first which was the gold doc)
        if len(original_retrieved) > 1:
            new_retrieved_docs.extend(original_retrieved[1:])
        
        # Get verification response data
        verification_response = perturbation.get('verification_response', {})
        
        # Create comprehensive example with all metadata
        enhanced_example = {
            # CORE IDENTIFIERS
            'unique_id': unique_id,                    # NEW: Unique primary key
            'source_qid': perturbation.get('source_qid'),  # Original question ID
            'qid': original.get('qid'),                # For backward compatibility
            
            # QUERY AND CONTEXT
            'query': perturbation.get('perturbed_query'),
            'original_query': perturbation.get('original_query'),
            'perturbed_context': perturbation.get('perturbed_context'),
            'original_context': perturbation.get('original_context'),
            
            # ANSWERS AND LABELS
            'answer': original.get('answer'),
            'original_answers': perturbation.get('original_answers'),
            'ground_truth_label': verification_response.get('final_ground_truth_label', ''),
            'expected_rag_behavior': perturbation.get('expected_rag_behavior'),
            
            # PERTURBATION METADATA
            'perturbation_class': perturbation.get('perturbation_class'),
            'intensity': perturbation.get('intensity'),
            'intensity_achieved': perturbation.get('intensity_achieved'),
            'actual_intensity_observed': verification_response.get('actual_intensity_observed'),
            'lever_selected': perturbation.get('lever_selected'),
            'implementation_reasoning': perturbation.get('implementation_reasoning', ''),
            
            # VERIFICATION METADATA
            'generation_successful': perturbation.get('generation_successful'),
            'verification_successful': perturbation.get('verification_successful'),
            'verification_result': verification_response.get('verification_result'),
            'lever_correctly_implemented': verification_response.get('lever_correctly_implemented'),
            'intensity_correctly_achieved': verification_response.get('intensity_correctly_achieved'),
            'uncertainty_successfully_induced': verification_response.get('uncertainty_successfully_induced'),
            'answer_constraint_satisfied': verification_response.get('answer_constraint_satisfied'),
            
            # RETRIEVED DOCUMENTS
            'retrieved_docs_ids': original.get('retrieved_docs_ids'),
            'retrieved_docs': new_retrieved_docs,
            
            # PROCESSING METADATA
            'source_line': perturbation.get('source_line'),
            'parsing_successful': perturbation.get('parsing_successful'),
            'constraint_analysis': verification_response.get('constraint_analysis'),
            'identified_issues': verification_response.get('identified_issues', []),
            
            # TIMESTAMP
            'formatted_timestamp': datetime.datetime.now().isoformat(),
            'agreement_mode_used': self.agreement_mode
        }
        
        return enhanced_example
    
    def filter_and_format(self, input_files: List[str], original_file: str, output_file: str) -> Dict:
        """Main processing pipeline with enhanced formatting."""
        print(f"Processing {len(input_files)} verification files")
        print(f"Agreement mode: {self.agreement_mode}")
        
        # Load data
        original_data = self.load_original_data(original_file)
        all_verifications = self.load_verification_files(input_files)
        
        # Group perturbations
        perturbation_groups = self.group_perturbations(all_verifications)
        print(f"\nFound {len(perturbation_groups)} unique perturbations")
        
        # Filter and format
        formatted_examples = []
        stats = {
            'total_perturbations': len(perturbation_groups),
            'with_all_verifiers': 0,
            'passed_agreement': 0,
            'missing_original': 0,
            'successfully_formatted': 0,
            'by_perturbation_class': defaultdict(int),
            'by_intensity': defaultdict(int),
            'by_ground_truth_label': defaultdict(int)
        }
        
        for key, verifier_group in perturbation_groups.items():
            # Must have verification from all verifiers
            if len(verifier_group) == len(input_files):
                stats['with_all_verifiers'] += 1
                
                # Check agreement
                verifications = list(verifier_group.values())
                if self.check_agreement(verifications):
                    stats['passed_agreement'] += 1
                    
                    # Get perturbation data (use first verifier as base)
                    base_perturbation = verifications[0]
                    source_qid = base_perturbation.get('source_qid')
                    
                    # Find original data
                    if source_qid in original_data:
                        original = original_data[source_qid]
                        formatted_example = self.create_enhanced_reference_example(base_perturbation, original)
                        formatted_examples.append(formatted_example)
                        stats['successfully_formatted'] += 1
                        
                        # Update category stats
                        stats['by_perturbation_class'][formatted_example.get('perturbation_class', 'unknown')] += 1
                        stats['by_intensity'][formatted_example.get('intensity', 'unknown')] += 1
                        stats['by_ground_truth_label'][formatted_example.get('ground_truth_label', 'unknown')] += 1
                        
                    else:
                        stats['missing_original'] += 1
                        print(f"Warning: Missing original data for qid {source_qid}")
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in formatted_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"\nSaved {len(formatted_examples)} enhanced examples to {output_file}")
        
        # Create metadata file
        metadata_file = output_file.replace('.jsonl', '_metadata.json')
        metadata = {
            'created_timestamp': datetime.datetime.now().isoformat(),
            'agreement_mode': self.agreement_mode,
            'input_verification_files': input_files,
            'original_reference_file': original_file,
            'total_examples': len(formatted_examples),
            'statistics': dict(stats),
            'unique_id_format': 'RB_{source_qid}_{perturbation_class}_{intensity}_{content_hash}_{counter}',
            'field_descriptions': {
                'unique_id': 'Unique primary key for each perturbation',
                'source_qid': 'Original question ID from reference dataset',
                'perturbation_class': 'Type of perturbation applied',
                'intensity': 'Intended difficulty level (LOW/MODERATE/HIGH)',
                'lever_selected': 'Specific perturbation technique used',
                'ground_truth_label': 'Expected model behavior (ANSWER_CORRECTLY or REFUSE_*)',
                'verification_result': 'Whether perturbation passed verification (PASS/FAIL)'
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Saved metadata to {metadata_file}")
        
        return stats
    
    def print_enhanced_stats(self, stats: Dict):
        """Print comprehensive processing statistics."""
        print(f"\n" + "="*80)
        print(f"ENHANCED FORMATTING STATISTICS")
        print(f"Agreement Mode: {self.agreement_mode.upper()}")
        print(f"="*80)
        
        total = stats['total_perturbations']
        with_all = stats['with_all_verifiers']
        passed = stats['passed_agreement']
        missing = stats['missing_original']
        formatted = stats['successfully_formatted']
        
        print(f"\nProcessing Summary:")
        print(f"  Total unique perturbations: {total}")
        print(f"  With all verifiers: {with_all}")
        print(f"  Passed agreement: {passed}")
        print(f"  Missing original data: {missing}")
        print(f"  Successfully formatted: {formatted}")
        
        if with_all > 0:
            print(f"  Agreement rate: {passed/with_all*100:.1f}%")
        if passed > 0:
            print(f"  Format success rate: {formatted/passed*100:.1f}%")
        
        # Category breakdowns
        print(f"\nPerturbation Class Distribution:")
        for pclass, count in stats['by_perturbation_class'].items():
            print(f"  {pclass}: {count}")
        
        print(f"\nIntensity Distribution:")
        for intensity, count in stats['by_intensity'].items():
            print(f"  {intensity}: {count}")
        
        print(f"\nGround Truth Label Distribution:")
        for label, count in stats['by_ground_truth_label'].items():
            print(f"  {label}: {count}")
    
    def run(self, input_files: List[str], original_file: str, output_file: str):
        """Run the complete enhanced pipeline."""
        stats = self.filter_and_format(input_files, original_file, output_file)
        self.print_enhanced_stats(stats)
        return stats

def main():
    """Main function with enhanced processing."""
    # Input verification files
    input_files = [
        "dataset_deepseek/refusalbench_verifications_claude.jsonl",
        "dataset_deepseek/refusalbench_verifications_deepseek.jsonl", 
        # "dataset/refusalbench_verifications_gemini.jsonl"
    ]
    
    # Original reference dataset
    original_file = "/data/group_data/r3lit_shared/ragdynabench/textgrad/examples/notebooks/reference_dataset/all_correct_reference_final.jsonl"
    
    # Output file
    output_file = "dataset_filtered_deepseek/refusalbench_perturbed_reference.jsonl"
    
    # Run enhanced formatter
    formatter = EnhancedRefusalBenchReferenceFormatter(agreement_mode="all_agree")
    
    try:
        print("Starting enhanced verification filtering and formatting...")
        formatter.run(input_files, original_file, output_file)
        print(f"\n✅ Complete! Enhanced results saved to {output_file}")
        print(f"📊 Metadata saved to {output_file.replace('.jsonl', '_metadata.json')}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print("Required files:")
        for file in input_files + [original_file]:
            exists = os.path.exists(file)
            print(f"  {file}: {'✓' if exists else '✗'}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()