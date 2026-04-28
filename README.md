# SeqLLM Bandits

This repository implements bandit-based model selection for multi-stage LLM pipelines on telecommunications and medical datasets, along with utilities for data handling, embedding, and evaluation.

## Directory Structure
.
├── config/
│   └── keys.py                  # API keys and configuration
├── data/
│   ├── diagnoses_100.pkl        # Gold labels (medical)
│   ├── input_reports_100.pkl    # Input texts (medical)
│   └── telecon_data/            # Raw telecom data
├── experiments/
│   ├── medical/
│   │   ├── agnostic/                   # Cost-unaware baselines
│   │   │   ├── neucb_med.py                # NeuralUCB
│   │   │   ├── neulinucb_med.py            # NeuralLinUCB
│   │   │   ├── rand_gpts.py                # RandGPT
│   │   │   ├── rand_seq_gpts.py            # Rand-SeqGPT
│   │   │   ├── sequential_bandits.py       # SeqBandit
│   │   │   └── utils/
│   │   └── budgeted/                   # Cost-aware baselines
│   │       ├── two_subtask/                # 2-stage pipeline (with 4 baselines)
│   │       └── three_subtask/              # 3-stage pipeline (with 4 baselines)
│   └── telecom/
│       ├── agnostic/                   # Cost-unaware baselines
│       │   ├── neucb_tele.py               # NeuralUCB
│       │   ├── neulinucb_tele.py           # NeuralLinUCB
│       │   ├── rand_gpt_tele.py            # RandGPT
│       │   ├── sequential_bandits_tele.py  # SeqBandit
│       │   └── utils/
│       └── budgeted/                   # Cost-aware baselines
│           ├── neucb_budgeted_tele.py      # NeuralUCB budgeted
│           ├── neulinucb_budgeted_tele.py  # NeuralLinUCB budgeted
│           ├── randgpt_budget_tele.py      # RandGPT budgeted
│           ├── seqgpt_budgeted_tele.py     # SeqGPT budgeted
│           └── utils/
├── read.py                                 # Data inspection utility
├── README.md                               # This overview
├── requirements.txt                        # Python dependencies
└── src/
    ├── api/
    │   └── client.py                   # LLM API wrappers with retry logic
    ├── embedding/
    │   ├── embed_med.py                # Medical embedding routines
    │   └── embed_tele.py               # Telecom embedding routines
    ├── prompts/
    │   ├── prompt_maker.py             # Base prompt templates
    │   └── prompt_maker_seq.py         # SeqBandit prompt helpers
    ├── regrets/
    │   ├── final_rand_med.py           # Rand-SeqGPT evaluation (medical)
    │   ├── optimal_rand_tele.py        # RandGPT evaluation (telecom)
    │   ├── optimal_rand_seq_tele.py    # Rand-SeqGPT (telecom)
    │   ├── sum_call.py                 # Single-stage regret logic
    │   └── sum_call_seq.py             # SeqBandit regret logic
    ├── token_prediction/
    │   ├── model_names.json            # Supported LLM names
    │   ├── best_length_model.pth       # Trained BERT regressor
    │   └── tok_length_predict.py       # Token-length predictor
    └── utils/
        └── helper.py             # Shared utility functions