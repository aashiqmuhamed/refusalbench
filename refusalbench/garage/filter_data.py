import json
import argparse
import random
from collections import defaultdict

def create_pristine_subset_final(
    input_file: str,
    output_file: str,
    target_domains: list,
    samples_per_domain: int,
    num_passages_total: int = 10,
    num_signal_passages: int = 5
):
    """
    Filters GaRAGe using the final data-driven strategy, including "Head"
    questions, to create a high-quality base set for RefusalBench perturbations.
    """
    print("--- Starting GaRAGe Dataset Curation (Final Criteria: Including Head) ---")

    # Phase 1: Filter all data into eligible pools
    eligible_pools = defaultdict(list)
    print("Phase 1: Filtering and pooling eligible candidates...")
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                data = json.loads(line)
                
                # --- Apply the Final, Refined Filtering Criteria ---
                is_validated = data.get('answer_validate') == "YES"
                is_answerable = "ANSWER-THE-QUESTION" in data.get('evidence_correct', [])
                is_simple_enough = data.get('question_complexity') in {
                    "Simple", "Simple w. condition", "Comparison", "Set"
                }
                is_stable = data.get('question_type') in {"SLOW-CHANGING", ""}
                
                # The popularity filter is now removed to include "Head" questions.

                if is_validated and is_answerable and is_simple_enough and is_stable:
                    # Also ensure there are enough passages to sample from
                    if len(data.get('grounding', [])) >= num_passages_total:
                         eligible_pools[data['question_category']].append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    
    print("\n--- Filtering Complete ---")
    print("Eligible candidates found per domain (including Head questions):")
    for domain in target_domains:
        count = len(eligible_pools.get(domain, []))
        print(f"  - {domain:<25}: {count} candidates")

    # Phase 2: Balanced sampling and controlled context composition
    print(f"\nPhase 2: Performing balanced sampling...")
    final_samples_written = 0
    final_domain_counts = defaultdict(int)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for domain in target_domains:
            candidate_pool = eligible_pools.get(domain, [])
            num_to_sample = min(samples_per_domain, len(candidate_pool))
            if num_to_sample == 0: continue

            sampled_items = random.sample(candidate_pool, num_to_sample)
            
            for sample in sampled_items:
                # --- Compose the 10-passage mixed context ---
                all_indices = list(range(len(sample['grounding'])))
                
                signal_indices = {i for i, label in enumerate(sample['evidence_correct']) if label == "ANSWER-THE-QUESTION"}
                noise_indices = {i for i in all_indices if i not in signal_indices}

                def get_priority_key(index):
                    is_cited = sample['evidence_cited'][index] == "YES"
                    is_related = sample['evidence_correct'][index] == "RELATED-INFORMATION"
                    if is_cited: return 0
                    if is_related: return 1
                    return 2

                sorted_signal = sorted(list(signal_indices), key=get_priority_key)
                sorted_noise = sorted(list(noise_indices), key=get_priority_key)

                final_signal = sorted_signal[:num_signal_passages]
                num_noise_needed = num_passages_total - len(final_signal)
                final_noise = sorted_noise[:num_noise_needed]

                final_indices = sorted(final_signal + final_noise)
                if len(final_indices) != num_passages_total: continue

                # Create the new normalized data
                curated_sample = sample.copy()
                curated_sample['grounding'] = [sample['grounding'][i] for i in final_indices]
                curated_sample['evidence_correct'] = [sample['evidence_correct'][i] for i in final_indices]
                curated_sample['evidence_cited'] = [sample['evidence_cited'][i] for i in final_indices]
                curated_sample['evidence_relevant'] = [sample['evidence_relevant'][i] for i in final_indices]
                
                curated_sample['curation_notes'] = {
                    "source_dataset": "GaRAGe", "status": "Pristine",
                    "passages_normalized_to": num_passages_total,
                    "normalization_strategy": f"{len(final_signal)}_signal_{len(final_noise)}_noise"
                }
                
                outfile.write(json.dumps(curated_sample) + '\n')
                final_samples_written += 1
                final_domain_counts[domain] += 1

    print("\n--- Curation and Writing Complete ---")
    print(f"Total pristine samples written to '{output_file}': {final_samples_written}")
    print("Final sample counts per domain:")
    for domain, count in final_domain_counts.items():
        print(f"  - {domain:<25}: {count} samples")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter GaRAGe to create a pristine base set.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the full GaRAGe dataset JSONL file.")
    parser.add_argument("--output_file", type=str, default="garage_refusalbench_base_final.jsonl", help="Path for the final curated JSONL file.")
    parser.add_argument("--domains", nargs='+', default=["Science", "Health", "Business & Industrial", "Law & Government", "Finance"], help="Domains to include.")
    parser.add_argument("--samples_per_domain", type=int, default=20, help="Samples to extract per domain.")
    args = parser.parse_args()
    random.seed(42)
    create_pristine_subset_final(
        input_file=args.input_file,
        output_file=args.output_file,
        target_domains=args.domains,
        samples_per_domain=args.samples_per_domain
    )