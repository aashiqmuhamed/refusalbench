# RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models

[![Paper](https://img.shields.io/badge/paper-EACL%202026-blue)](https://aclanthology.org/2026.eacl-long.321/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.10390-b31b1b)](https://arxiv.org/abs/2510.10390)
[![🤗 RefusalBench-NQ](https://img.shields.io/badge/🤗%20Dataset-RefusalBench--NQ-yellow)](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-NQ)
[![🤗 RefusalBench-GaRAGe](https://img.shields.io/badge/🤗%20Dataset-RefusalBench--GaRAGe-yellow)](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-GaRAGe)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)](https://www.python.org/)

## 📚 Overview

RefusalBench is a comprehensive benchmark for evaluating the selective refusal capabilities of Retrieval-Augmented Generation (RAG) systems. It systematically tests whether models can appropriately refuse to answer when faced with linguistic uncertainties, rather than generating hallucinated or incorrect responses.

### 🎯 Key Contributions

1. **First systematic benchmark** for evaluating selective refusal in RAG systems
2. **176 linguistic perturbation levers** across 6 uncertainty dimensions and 3 intensity levels
3. **Cross-model verification pipeline** ensuring high-quality perturbations
4. **Dual evaluation framework** testing both answer accuracy and refusal calibration
5. **Comprehensive analysis** of generator-evaluator bias in perturbation-based benchmarking

## 📊 Datasets

> **📥 Released benchmarks (Hugging Face):** [RefusalBench-NQ](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-NQ) (Apache-2.0) · [RefusalBench-GaRAGe](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-GaRAGe) (CC-BY-NC-4.0)
>
> ```python
> from datasets import load_dataset
> nq     = load_dataset("aashiqmuhamed/RefusalBench-NQ", split="test")      # 1,600
> garage = load_dataset("aashiqmuhamed/RefusalBench-GaRAGe", split="test")  # 1,506
> ```

RefusalBench supports two primary datasets with different characteristics:

### RefusalBench-NQ — single-document
- **Source**: Natural Questions (Kwiatkowski et al., 2019) with KILT gold passages
- **Released**: 🤗 [`aashiqmuhamed/RefusalBench-NQ`](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-NQ) — **1,600 instances** (`test` split) from 100 source questions, balanced across the 18 class×intensity strata and 4 generators (400 each)
- **Pipeline code**: `refusalbench/naturalquestions/`
- **Model input**: `perturbed_query` + `perturbed_context` (single passage)

### RefusalBench-GaRAGe — multi-document
- **Source**: [GaRAGe](https://arxiv.org/abs/2506.07671) (Sorodoc et al., 2025)
- **Released**: 🤗 [`aashiqmuhamed/RefusalBench-GaRAGe`](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-GaRAGe) — **1,506 instances** (`test` split), naturally imbalanced, across 5 domains (Science, Health, Business & Industrial, Law & Government, Finance)
- **Pipeline code**: `refusalbench/garage/`
- **Model input**: `query` + `grounding` (10 passages: up to 5 signal + noise distractors)

See each dataset card for the full field schema.

## 🗂️ Repository Structure

```
refusalbench/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── refusalbench/                      # Main codebase
│   ├── naturalquestions/              # NQ dataset pipeline
│   │   ├── config_template.py         # Configuration template
│   │   ├── prompt_guidelines.py       # RefusalBenchCatalogue with 176 perturbation levers
│   │   ├── generate_perturbations.py  # Async perturbation generation
│   │   ├── verify_all.py              # Multi-model verification
│   │   ├── filter_data.py             # Cross-model agreement filtering
│   │   ├── filter_all_2.py            # Enhanced filtering with metadata
│   │   ├── filter_stratified.py       # Stratified sampling
│   │   ├── run_models.py              # Single model evaluation
│   │   ├── run_models_all.py          # Batch model evaluation
│   │   └── extract_metrics_final.py   # Metrics computation & visualization
│   │
│   └── garage/                         # GaRAGe dataset pipeline
│       ├── config_template.py         # GaRAGe-specific config
│       ├── filter_data.py             # Initial data filtering
│       ├── garage_generate_perturbations.py  # Multi-passage perturbations
│       ├── filter_all_stratified.py   # Stratified sampling for GaRAGe
│       ├── run_models_all.py          # GaRAGe evaluation
│       └── verify_all.py              # GaRAGe verification
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for local models)
- API access to at least one LLM provider (OpenAI, Anthropic, AWS Bedrock, etc.)

### Setup

```bash
# Clone the repository
git clone https://github.com/aashiqmuhamed/refusalbench.git
cd refusalbench

# Install dependencies
pip install -r requirements.txt

# Configure API credentials for NQ dataset
cp refusalbench/naturalquestions/config_template.py refusalbench/naturalquestions/config.py
# Edit config.py with your API keys

# Configure for GaRAGe dataset (if using)
cp refusalbench/garage/config_template.py refusalbench/garage/config.py
# Edit config.py with your API keys
```

### Configuration

Edit the `config.py` file(s) with your credentials:

```python
# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID = "your-access-key"
AWS_SECRET_ACCESS_KEY = "your-secret-key"
AWS_REGION_NAME = "us-east-1"

# OpenAI Configuration
OPENAI_API_KEY = "your-openai-key"

# Model IDs for different stages
DEFAULT_GENERATOR_MODEL = "anthropic/claude-3-sonnet"
DEFAULT_VERIFIER_MODEL = "openai/gpt-4"
DEFAULT_EVALUATOR_MODEL = "anthropic/claude-3-opus"
```

## 🚀 Usage

### Complete Pipeline

The RefusalBench pipeline consists of 5 main stages:

#### 1. Generate Perturbations
Transform high-quality QA pairs into challenging perturbations:

```bash
cd refusalbench/naturalquestions
python generate_perturbations.py \
  --input-file data/nq_reference.jsonl \
  --output-file output/perturbations_raw.jsonl \
  --model claude-3-sonnet \
  --max-instances 1000
```

#### 2. Verify Quality
Use multiple models to verify perturbation quality:

```bash
python verify_all.py \
  --input-file output/perturbations_raw.jsonl \
  --output-file output/perturbations_verified.jsonl \
  --verifier-models "claude-3-opus,gpt-4,deepseek"
```

#### 3. Filter & Create Final Dataset
Apply cross-model agreement filtering:

```bash
# Option 1: Require unanimous agreement
python filter_data.py \
  --verification-files output/verified_*.jsonl \
  --agreement-mode unanimous \
  --output-file output/refusalbench_final.jsonl

# Option 2: Stratified sampling for balanced evaluation
python filter_stratified.py \
  --input-file output/refusalbench_final.jsonl \
  --output-file output/refusalbench_stratified.jsonl \
  --samples-per-stratum 22
```

#### 4. Evaluate Models
Run RAG models on the benchmark:

```bash
python run_models_all.py \
  --dataset output/refusalbench_stratified.jsonl \
  --models "claude-3.5-sonnet,gpt-4o,nova-pro" \
  --output-dir results/
```

#### 5. Analyze Results
Compute metrics and generate visualizations:

```bash
python extract_metrics_final.py \
  --results-dir results/ \
  --output-dir analysis/
```

## 📈 Perturbation Taxonomy

RefusalBench implements 176 linguistic perturbation levers (6 classes × 3 intensities, ≈10 levers each):

### Perturbation Classes

| Class | Description | Example Levers | Expected Behavior |
|-------|-------------|----------------|-------------------|
| **P-Ambiguity** | Introduces query/context ambiguities | Lexical Polysemy, Scope Ambiguity, Pronoun Resolution | `REFUSE_AMBIGUOUS_QUERY` |
| **P-Contradiction** | Creates conflicting information | Direct Negation, Temporal Conflict, Causal Reversal | `REFUSE_CONTRADICTORY_CONTEXT` |
| **P-MissingInfo** | Removes essential information | Entity Removal, Relationship Deletion, Value Omission | `REFUSE_INFO_MISSING_IN_CONTEXT` |
| **P-FalsePremise** | Embeds false assumptions | Counterfactual Entity, Impossible Action, False Attribution | `REFUSE_FALSE_PREMISE_IN_QUERY` |
| **P-GranularityMismatch** | Creates scale mismatches | Over-specification, Category-Instance Swap, Unit Confusion | `REFUSE_GRANULARITY_MISMATCH` |
| **P-EpistemicMismatch** | Non-factual queries | Subjective Transform, Future Speculation, Opinion Request | `REFUSE_NONFACTUAL_QUERY` |

### Intensity Levels

- **LOW**: Subtle perturbations, often still answerable
- **MEDIUM**: Clear uncertainties requiring careful judgment
- **HIGH**: Obvious issues demanding refusal

## 📊 Evaluation Metrics

### Primary Metrics

1. **Answer Accuracy** (for answerable instances)
   - Measures correctness when the model attempts to answer
   - Score ≥ 4 on 1-5 scale indicates correct answer

2. **Refusal Accuracy** (for unanswerable instances)
   - Exact match with expected refusal category
   - Tracks both binary refusal and category classification

3. **Calibrated Refusal Score (CRS)**
   - Composite metric balancing answer and refusal performance
   - Formula: `CRS = w₁ * AnswerAcc + w₂ * RefusalAcc - w₃ * FalseRefusalRate`

### Secondary Metrics

- **False Refusal Rate (FRR)**: Refusing when should answer
- **Missed Refusal Rate (MRR)**: Answering when should refuse
- **Refusal Calibration**: Correlation between confidence and correctness
- **Intensity Degradation**: Performance change across intensity levels

## 🔬 Key Findings

Our experiments reveal several important insights:

1. **Performance Degradation**: All models show decreased performance as perturbation intensity increases
2. **Construct Separation**: Answer accuracy and refusal classification are distinct cognitive capabilities
3. **Generator-Evaluator Bias**: Models may show self-preference when evaluating their own perturbations
4. **Model Specialization**: Some models excel at answering, others at refusing appropriately

## 📝 Output Formats

### Released dataset format (RefusalBench-NQ)
```json
{
  "id": "RB-NQ_claude_4925057086725798331_P-Ambiguity_HIGH_bd0591c4_1145",
  "source_id": "4925057086725798331",
  "generator_model": "claude",
  "perturbation_class": "P-Ambiguity",
  "intensity": "HIGH",
  "expected_rag_behavior": "REFUSE_AMBIGUOUS_QUERY",
  "lever_selected": "Pure Homonymy Clash",
  "original_query": "who wrote yakkity yak don't talk back",
  "original_context": "\"Yakety Yak\" is a song written by Jerry Leiber and Mike Stoller ...",
  "original_answers": ["Jerry Leiber and Mike Stoller", "Jerry Leiber", "Mike Stoller"],
  "perturbed_query": "who wrote yakkity yak don't talk back",
  "perturbed_context": "\"Yakety Yak\" ... \"Yakkity Yak Don't Talk Back\" is a song written by Tommy Johnson ...",
  "implementation_reasoning": "Applied the Pure Homonymy Clash lever: two near-identically titled songs with different writers ...",
  "verifier_votes": {"claude": "PASS", "nova": "PASS", "gpt": "PASS", "deepseek": "PASS"}
}
```
> RefusalBench-GaRAGe shares the core fields and adds `query`, `grounding` (10 passages), `reference_answer`, `signal_indices`/`noise_indices`, and the GaRAGe annotations. See the dataset cards for the full schema.

### Evaluation Result Format
```json
{
  "model_id": "claude-3.5-sonnet",
  "unique_id": "RB_nq_001_P-Ambiguity_MEDIUM_lexical_0001",
  "model_response": "I cannot answer this question due to ambiguity...",
  "model_predicted_type": "REFUSE_AMBIGUOUS_QUERY",
  "answer_quality_score": null,
  "refusal_match_correct": true,
  "evaluation_metadata": {...}
}
```

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

- Additional perturbation levers
- Support for more datasets
- Multilingual extensions
- New evaluation metrics
- Model-specific optimizations

Please open an issue or pull request on GitHub.

## 📖 Citation

If you use RefusalBench in your research, please cite:

```bibtex
@inproceedings{muhamed-etal-2026-refusalbench,
    title = "{R}efusal{B}ench: Generative Evaluation of Selective Refusal in Grounded Language Models",
    author = "Muhamed, Aashiq  and
      Ribeiro, Leonardo F. R.  and
      Dreyer, Markus  and
      Smith, Virginia  and
      Diab, Mona T.",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.321/",
    doi = "10.18653/v1/2026.eacl-long.321",
    pages = "6811--6856",
    ISBN = "979-8-89176-380-7"
}
```

## 🏆 Acknowledgments

We thank the creators of the Natural Questions and GaRAGe datasets.

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.


## 🔗 Links

- [Paper (ACL Anthology, EACL 2026)](https://aclanthology.org/2026.eacl-long.321/)
- [Paper (arXiv preprint)](https://arxiv.org/abs/2510.10390)
- [RefusalBench-NQ (🤗 Datasets)](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-NQ)
- [RefusalBench-GaRAGe (🤗 Datasets)](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-GaRAGe)