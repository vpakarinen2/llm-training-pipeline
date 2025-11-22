# Full LLM Training Pipeline

## Workflow

#### data → config → train → evaluate → compare

## Setup

### Create Python Environment

```
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

## Run

### Training

```
python scripts/train.py --config configs/train_example.yaml --device auto
```

### Evaluation

```
python scripts/evaluate.py --config configs/train_example.yaml

python scripts/evaluate.py --config configs/train_example.yaml --checkpoint configs/outputs/example_run/step_2 (checkpoint)
```

## Author

Ville Pakarinen (@vpakarinen2)
