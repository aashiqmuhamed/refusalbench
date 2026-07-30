

# RefusalBench: Evaluación Generativa del Rechazo Selectivo en Modelos de Lenguaje Fundamentados

[![Paper](https://img.shields.io/badge/paper-EACL%202026-blue)](https://aclanthology.org/2026.eacl-long.321/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.10390-b31b1b)](https://arxiv.org/abs/2510.10390)
[![🤗 RefusalBench-NQ](https://img.shields.io/badge/🤗%20Dataset-RefusalBench--NQ-yellow)](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-NQ)
[![🤗 RefusalBench-GaRAGe](https://img.shields.io/badge/🤗%20Dataset-RefusalBench--GaRAGe-yellow)](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-GaRAGe)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)](https://www.python.org/)

## 📚 Descripción General

RefusalBench es una evaluación exhaustiva para medir las capacidades de rechazo selectivo de los sistemas de Generación Aumentada por Recuperación (RAG). Prueba sistemáticamente si los modelos pueden rechazar apropiadamente responder cuando se enfrentan a incertidumbres lingüísticas, en lugar de generar respuestas alucinadas o incorrectas.

### 🎯 Contribuciones Clave

1. **Primer conjunto de evaluación sistemático** para medir el rechazo selectivo en sistemas RAG
2. **176 palancas de perturbación lingüística** a lo largo de 6 dimensiones de incertidumbre y 3 niveles de intensidad
3. **Flujo de verificación cruzada entre modelos** que garantiza perturbaciones de alta calidad
4. **Marco de evaluación dual** que prueba tanto la precisión de las respuestas como la calibración del rechazo
5. **Análisis integral** del sesgo generador-evaluador en las evaluaciones basadas en perturbaciones

## 📊 Conjuntos de Datos

> **📥 Conjuntos publicados (Hugging Face):** [RefusalBench-NQ](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-NQ) (Apache-2.0) · [RefusalBench-GaRAGe](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-GaRAGe) (CC-BY-NC-4.0)
>
> ```python
> from datasets import load_dataset
> nq     = load_dataset("aashiqmuhamed/RefusalBench-NQ", split="test")      # 1,600
> garage = load_dataset("aashiqmuhamed/RefusalBench-GaRAGe", split="test")  # 1,506
> ```

RefusalBench admite dos conjuntos de datos principales con características diferentes:

### RefusalBench-NQ — documento único
- **Fuente**: Natural Questions (Kwiatkowski et al., 2019) con pasajes de oro KILT
- **Publicado**: 🤗 [`aashiqmuhamed/RefusalBench-NQ`](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-NQ) — **1,600 instancias** (`split` de `test`) a partir de 100 preguntas de origen, equilibradas en los 18 estratos clase×intensidad y 4 generadores (400 cada uno)
- **Código del flujo**: `refusalbench/naturalquestions/`
- **Entrada del modelo**: `perturbed_query` + `perturbed_context` (único pasaje)

### RefusalBench-GaRAGe — multi-documento
- **Fuente**: [GaRAGe](https://arxiv.org/abs/2506.07671) (Sorodoc et al., 2025)
- **Publicado**: 🤗 [`aashiqmuhamed/RefusalBench-GaRAGe`](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-GaRAGe) — **1,506 instancias** (`split` de `test`), naturalmente desequilibrado, a lo largo de 5 dominios (Ciencia, Salud, Negocios e Industrial, Leyes y Gobierno, Finanzas)
- **Código del flujo**: `refusalbench/garage/`
- **Entrada del modelo**: `query` + `grounding` (10 pasajes: hasta 5 señales + distractores de ruido)

Consulte la tarjeta de cada conjunto de datos para ver el esquema completo de los campos.

## 🗂️ Estructura del Repositorio

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

## 🔧 Instalación

### Requisitos Previos
- Python 3.8+
- GPU compatible con CUDA (recomendado para modelos locales)
- Acceso API a al menos un proveedor de LLM (OpenAI, Anthropic, AWS Bedrock, etc.)

### Configuración Inicial

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

### Configuración

Edite el(los) archivo(s) `config.py` con sus credenciales:

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

## 🚀 Uso

### Flujo de Trabajo Completo

El flujo de trabajo de RefusalBench consta de 5 etapas principales:

#### 1. Generación de Perturbaciones
Transforme pares de preguntas-respuestas de alta calidad en perturbaciones desafiantes:

```bash
cd refusalbench/naturalquestions
python generate_perturbations.py \
  --input-file data/nq_reference.jsonl \
  --output-file output/perturbations_raw.jsonl \
  --model claude-3-sonnet \
  --max-instances 1000
```

#### 2. Verificación de Calidad
Use múltiples modelos para verificar la calidad de las perturbaciones:

```bash
python verify_all.py \
  --input-file output/perturbations_raw.jsonl \
  --output-file output/perturbations_verified.jsonl \
  --verifier-models "claude-3-opus,gpt-4,deepseek"
```

#### 3. Filtrado y Creación del Conjunto de Datos Final
Aplique filtrado por acuerdo entre modelos:

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

#### 4. Evaluación de Modelos
Ejecute modelos RAG en el conjunto de evaluación:

```bash
python run_models_all.py \
  --dataset output/refusalbench_stratified.jsonl \
  --models "claude-3.5-sonnet,gpt-4o,nova-pro" \
  --output-dir results/
```

#### 5. Análisis de Resultados
Calcule métricas y genere visualizaciones:

```bash
python extract_metrics_final.py \
  --results-dir results/ \
  --output-dir analysis/
```

## 📈 Taxonomía de Perturbaciones

RefusalBench implementa 176 palancas de perturbación lingüística (6 clases × 3 intensidades, ≈10 palancas cada una):

### Clases de Perturbación

| Clase | Descripción | Palancas de Ejemplo | Comportamiento Esperado |
|-------|-------------|-------------------|----------------------|
| **P-Ambiguity** | Introduce ambigüedades en la consulta/contexto | Polisemia léxica, Ambigüedad de alcance, Resolución de pronombres | `REFUSE_AMBIGUOUS_QUERY` |
| **P-Contradiction** | Crea información conflictiva | Negación directa, Conflicto temporal, Inversión causal | `REFUSE_CONTRADICTORY_CONTEXT` |
| **P-MissingInfo** | Elimina información esencial | Eliminación de entidad, Eliminación de relación, Omisión de valor | `REFUSE_INFO_MISSING_IN_CONTEXT` |
| **P-FalsePremise** | Incrusta supuestos falsos | Entidad contrafactual, Acción imposible, Atribución falsa | `REFUSE_FALSE_PREMISE_IN_QUERY` |
| **P-GranularityMismatch** | Crea desajustes de escala | Sobreespecificación, Intercambio categoría-instancia, Confusión de unidades | `REFUSE_GRANULARITY_MISMATCH` |
| **P-EpistemicMismatch** | Consultas no factuales | Transformación subjetiva, Especulación futura, Solicitud de opinión | `REFUSE_NONFACTUAL_QUERY` |

### Niveles de Intensidad

- **BAJO**: Perturbaciones sutiles, a menudo aún respondibles
- **MEDIO**: Incertidumbres claras que requieren un juicio cuidadoso
- **ALTO**: Problemas evidentes que exigen un rechazo

## 📊 Métricas de Evaluación

### Métricas Principales

1. **Precisión de Respuesta** (para instancias respondibles)
   - Mide la corrección cuando el modelo intenta responder
   - Una puntuación ≥ 4 en escala de 1-5 indica respuesta correcta

2. **Precisión de Rechazo** (para instancias no respondibles)
   - Coincidencia exacta con la categoría de rechazo esperada
   - Rastrea tanto el rechazo binario como la clasificación por categoría

3. **Puntuación de Rechazo Calibrado (CRS)**
   - Métrica compuesta que equilibra el rendimiento de respuesta y rechazo
   - Fórmula: `CRS = w₁ * PrecisiónRespuesta + w₂ * PrecisiónRechazo - w₃ * TasaRechazoFalso`

### Métricas Secundarias

- **Tasa de Rechazo Falso (FRR)**: Rechazar cuando debería responder
- **Tasa de Rechazo No Detectado (MRR)**: Responder cuando debería rechazar
- **Calibración de Rechazo**: Correlación entre confianza y corrección
- **Degradación por Intensidad**: Cambio de rendimiento a lo largo de los niveles de intensidad

## 🔬 Hallazgos Clave

Nuestros experimentos revelan varias percepciones importantes:

1. **Degradación del Rendimiento**: Todos los modelos muestran un descenso en el rendimiento a medida que aumenta la intensidad de la perturbación
2. **Separación de Constructos**: La precisión de respuesta y la clasificación de rechazo son capacidades cognitivas distintas
3. **Sesgo Generador-Evaluador**: Los modelos pueden mostrar preferencia propia al evaluar sus propias perturbaciones
4. **Especialización de Modelos**: Algunos modelos destacan en responder, otros en rechazar apropiadamente

## 📝 Formatos de Salida

### Formato del conjunto de datos publicado (RefusalBench-NQ)
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
> RefusalBench-GaRAGe comparte los campos centrales y añade `query`, `grounding` (10 pasajes), `reference_answer`, `signal_indices`/`noise_indices` y las anotaciones de GaRAGe. Consulte las tarjetas de los conjuntos de datos para ver el esquema completo.

### Formato de Resultados de Evaluación
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

## 🤝 Contribuciones

¡Agradecemos las contribuciones! Áreas de particular interés:

- Palancas de perturbación adicionales
- Soporte para más conjuntos de datos
- Extensiones multilingües
- Nuevas métricas de evaluación
- Optimizaciones específicas para modelos

Por favor, abra un problema o una solicitud de extracción en GitHub.

## 📖 Citación

Si utiliza RefusalBench en su investigación, por favor cite:

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

## 🏆 Agradecimientos

Agradecemos a los creadores de los conjuntos de datos Natural Questions y GaRAGe.

## 📄 Licencia

Licencia Apache 2.0 - consulte [LICENSE](LICENSE) para más detalles.


## 🔗 Enlaces

- [Paper (ACL Anthology, EACL 2026)](https://aclanthology.org/2026.eacl-long.321/)
- [Paper (arXiv preprint)](https://arxiv.org/abs/2510.10390)
- [RefusalBench-NQ (🤗 Datasets)](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-NQ)
- [RefusalBench-GaRAGe (🤗 Datasets)](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-GaRAGe)
