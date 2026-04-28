# SeqLLM Bandits

This repository explores bandit-based algorithms for routing tasks in multi-stage Large Language Model (LLM) pipelines. It focuses on optimizing model selection for telecommunications and medical datasets, balancing performance and cost.

## Overview

The project implements various bandit strategies, including NeuralUCB, NeuralLinUCB, and sequential bandits, to dynamically select LLMs in pipelines. Experiments compare cost-unaware (agnostic) and cost-aware (budgeted) approaches across two domains: telecom data processing and medical diagnosis.

Key features include:
- Bandit algorithms for adaptive model routing
- Support for multi-stage pipelines
- Embedding and prompt generation utilities
- Regret analysis and evaluation tools

## Directory Structure

```
.
├── config/
│   └── keys.py                 # API keys and configuration
├── data/
│   ├── telecon_data/
│   │   ├── inputs/             # Raw telecom input texts
│   │   └── labels/             # Corresponding labels
│   └── (additional medical data files)
├── experiments/
│   ├── medical/
│   │   ├── agnostic/           # Cost-unaware baselines
│   │   └── budgeted/           # Cost-aware experiments
│   └── telecom/
│       ├── agnostic/           # Cost-unaware baselines
│       └── budgeted/           # Cost-aware experiments
├── src/
│   ├── api/
│   │   └── client.py           # LLM API client with retries
│   ├── embedding/
│   │   ├── embed_med.py        # Medical data embeddings
│   │   └── embed_tele.py       # Telecom data embeddings
│   ├── prompts/
│   │   ├── prompt_maker.py     # Base prompt templates
│   │   └── prompt_maker_seq.py # Sequential prompt helpers
│   ├── regrets/
│   │   ├── final_rand_med.py   # Medical regret calculations
│   │   ├── optimal_rand_tele.py
│   │   ├── optimal_rand_seq_tele.py
│   │   ├── sum_call.py         # Single-stage regrets
│   │   └── sum_call_seq.py     # Sequential regrets
│   ├── token_prediction/
│   │   ├── best_length_model.pth
│   │   ├── model_names.json    # Supported LLM models
│   │   └── tok_length_predict.py # Token length predictor
│   └── utils/
│       └── helper.py           # Utility functions
├── README.md
├── requirements.txt            # Python dependencies
└── (other files)
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Eddie-gi/LLM-Pipeline-Routing.git
   cd seq_llm_bandits
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys in `config/keys.py` with your LLM provider credentials (e.g., OpenAI, Anthropic).

## Usage

### Running Experiments

Navigate to the experiments directory and run a specific script. For example, to run a telecom agnostic baseline:

```bash
cd experiments/telecom/agnostic
python rand_gpt_tele.py
```

For budgeted experiments:

```bash
cd experiments/telecom/budgeted
python seqgpt_budgeted_tele.py
```

### Data Inspection

Use the data inspection utility:

```bash
python read.py
```

### Embedding Data

Generate embeddings for medical data:

```python
from src.embedding.embed_med import embed_medical_data
embed_medical_data()
```

## Experiments

The experiments are organized by domain (medical/telecom) and cost-awareness (agnostic/budgeted).

- **Agnostic**: Cost-unaware baselines that optimize for accuracy without considering API costs. Includes NeuralUCB, NeuralLinUCB, RandGPT, and sequential bandits.

- **Budgeted**: Cost-aware variants that balance performance and expenses. For telecom, includes 2-stage and 3-stage pipelines with multiple baselines.

Medical experiments focus on diagnosis tasks, while telecom experiments handle data processing pipelines. Each script evaluates regret and performance metrics.

## Contributing

This is a research project. For questions or contributions, please open an issue or pull request.