# Predictive Workload Autoscaler

ML-powered, workload-aware datacenter autoscaling simulator using Random Forest, LSTM, and GraphSAGE forecasts with optional LLM-based workload classification.

## Overview

This project simulates heterogeneous server fleets and predicts near-future CPU behavior to make proactive autoscaling decisions.  
The autoscaler optimizes for fewer active servers, lower power/cost, and safe utilization limits.

## Key Features

- Multi-model forecasting: `RandomForest`, `LSTM`, `GraphSAGE`
- Workload-aware autoscaling with server affinity rules
- Optional LLM advisor for dominant workload classification
- Docker-first workflow for reproducible training/evaluation
- JSON/CSV outputs for experiment tracking

## Project Structure

```text
core/       Autoscaler engine, traffic generation, metrics, workload advisor
models/     ML model implementations and serialized model artifacts
data/       Generated metric datasets (snapshot + wide format)
logs/       Training and evaluation result JSON files
evidence/   Supporting run artifacts
train.py    Main training/evaluation pipeline
```

## Quick Start (Docker)

```bash
docker compose build
docker compose run --rm train
docker compose run --rm eval
```

Optional LLM-assisted evaluation:

```bash
export OPENAI_API_KEY="your_key_here"
docker compose run --rm eval_llm
```

## Quick Start (Local Python)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

Evaluation only (uses saved models):

```bash
python train.py --eval-only
```

LLM-assisted evaluation:

```bash
export OPENAI_API_KEY="your_key_here"
python train.py --eval-only --use-llm --llm-model gpt-4.1-mini
```

## Outputs

- `data/metrics_snapshot.csv`
- `data/metrics_wide.csv`
- `models/*.pkl`
- `logs/training_results.json`
- `logs/eval_results.json`
