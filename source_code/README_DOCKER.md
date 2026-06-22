# Docker Usage

## 1) Build image

```bash
docker compose build
```

## 2) Full training pipeline

```bash
docker compose run --rm train
```

This command generates/updates:

- `data/metrics_snapshot.csv`
- `data/metrics_wide.csv`
- `models/*.pkl`
- `logs/training_results.json`

## 3) Eval only (saved models)

```bash
docker compose run --rm eval
```

This command generates/updates:

- `logs/eval_results.json`

## 4) Eval + LLM advisor (Gemini, free-tier friendly)

Set Gemini key in terminal:

```bash
export GEMINI_API_KEY="your_key_here"
```

Then run:

```bash
docker compose run --rm eval_llm
```

`eval_llm` service now uses these defaults:

- model: `gemini-2.5-flash`
- endpoint: `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
- LLM interval: `120` tick (rate-limit için daha güvenli)

Successful run indicator:

```text
LLM Advisor | online=True | calls=3 | success=3 | failed=0
```

Result file:

- `logs/eval_results.json`

## 5) Teammate without Docker

If one teammate does not use Docker, they can still:

- pull latest outputs (`data/`, `models/`, `logs/`)
- inspect JSON/CSV results
- run directly with Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --eval-only

# optional LLM mode (Gemini)
export GEMINI_API_KEY="your_key_here"
python train.py --eval-only --use-llm \
  --llm-model gemini-2.5-flash \
  --llm-endpoint https://generativelanguage.googleapis.com/v1beta/openai/chat/completions \
  --llm-interval-ticks 120
```
