# RL-Path: Reinforcement Learning for Pathway Steering

RL-Path is a reproducible reinforcement-learning **research simulator** for studying ordered drug-action policies over pathway-level state vectors.

The maintained project asks an algorithmic question:

> Can a DQN learn sequential actions that reduce simulated activity on a specified subset of pathways, under a finite step budget and a simple action-breadth penalty?

The repository combines:

- DGIdb drug-gene interactions;
- Reactome gene-pathway mappings;
- a normalized drug-to-pathway coverage matrix;
- a transparent pathway-state simulator;
- a Deep Q-Network (DQN);
- random and myopic-greedy baselines;
- paired evaluation across identical simulator seeds;
- reproducibility metadata, tests, CI, and Docker.

> **Important:** this is not a clinical treatment-recommendation system. The pathway "effects" are normalized network-coverage values, not measured pharmacodynamic effects. The action cost is a mathematical breadth penalty, not a toxicity model.

## State, action, transition, reward

### Observation

The default observation contains:

```text
current pathway activity
+ disease-pathway mask
+ remaining episode fraction
```

The disease mask is included because it determines which pathways contribute to the reward. The remaining fraction is included because the task has a finite horizon.

### Action

One discrete drug index from the processed drug-to-pathway matrix.

### Transition

For action `a`:

```text
next_state = clip(state - alpha * effect[a] + Gaussian noise, 0, 1)
```

### Reward

```text
reduction in disease-pathway MSE
- per-step penalty
- action-breadth cost
```

The reward is an engineering objective for the simulator. It is not a clinical endpoint.

## Why the quality upgrade matters

The original repository had several research-software weaknesses:

- a machine-specific absolute path in `train.py`;
- process-global NumPy seeding inside the DQN;
- random-action sampling and transition noise sharing one RNG stream;
- baseline evaluations run sequentially on one environment RNG;
- checkpoints without observation/action metadata;
- object-array `.npz` files loaded with `allow_pickle=True`;
- a path typo in the PSC rollout;
- generated artifacts and downloaded external datasets committed to the repository;
- no tests or CI.

The maintained version addresses those issues while keeping the immediate-effect pathway-steering model simple and explicit.

## Repository structure

```text
.
├── src/
│   ├── env.py
│   ├── dqn.py
│   ├── baselines.py
│   ├── evaluation.py
│   └── preprocess.py
├── scripts/
│   ├── psc_rollout.py
│   ├── psc_pathways_and_drugs.py
│   └── run_seed_sweep.py
├── tests/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
├── train.py
├── evaluate.py
├── EXPERIMENTS.md
├── MODEL_CARD.md
├── pyproject.toml
├── requirements.txt
└── Dockerfile
```

## Installation

```bash
git clone https://github.com/seirana/RL-Path-Reinforcement-Learning-for-Pathway-Steering.git
cd RL-Path-Reinforcement-Learning-for-Pathway-Steering

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

For tests and linting:

```bash
python -m pip install -e ".[dev]"
```

## External data

The repository no longer vendors large downloaded copies of DGIdb, HGNC, or Reactome files.

For the core training pipeline place:

```text
data/raw/dgidb_interactions.tsv
data/raw/Ensembl2Reactome.txt
```

The preprocessing step maps DGIdb gene symbols to Ensembl IDs using `mygene` and caches those mappings under `data/processed/`.

See [data/raw/README.md](data/raw/README.md) for provenance requirements.

## Train

```bash
rlpath-train \
  --episodes 400 \
  --steps 10 \
  --top-drugs 60 \
  --top-pathways 40 \
  --eval-rollouts 30 \
  --seed 42
```

Equivalent:

```bash
python train.py ...
```

By default, the disease-pathway mask is resampled each training episode and included in the observation. Use `--fixed-disease-mask` to keep one sampled mask across episodes.

The maintained observation layout differs from the historical version, so old `dqn.pt` checkpoints should be retrained.

## Evaluate

```bash
rlpath-evaluate \
  --steps 10 \
  --top-drugs 60 \
  --top-pathways 40 \
  --n-rollouts 30 \
  --seed 10000
```

Evaluation compares:

- DQN;
- myopic greedy;
- random.

For each rollout seed, each policy receives a fresh environment generated from the same seed. Random action selection uses a separate RNG stream from transition noise, so the random policy does not shift the noise sequence simply by sampling actions.

The output includes:

- mean return;
- sample standard deviation;
- standard error;
- 95% normal-approximation confidence interval;
- final disease-pathway MSE;
- paired DQN-minus-greedy and DQN-minus-random return differences.

## Independent training seeds

A single DQN training run does not characterize optimization variability.

```bash
rlpath-seed-sweep \
  --seeds 11,22,33,44,55 \
  -- \
  --episodes 400 \
  --eval-rollouts 30
```

See [EXPERIMENTS.md](EXPERIMENTS.md).

## PSC-oriented research scripts

Two optional PSC-oriented scripts are retained:

```bash
rlpath-psc-pathways --help
rlpath-psc-rollout --help
```

`psc_pathways_and_drugs.py` computes pathway-overlap research signals from local PSC/HGNC/DGIdb/Reactome inputs.

`psc_rollout.py` creates a disease mask from pathway-name keyword matches and then runs the trained policy in that simulator context.

These outputs are not treatment recommendations. Keyword matches are a heuristic way to select a simulator target set; they are not a validated PSC pathway model.

## Reproducibility

Randomness is separated across:

- simulator state/noise;
- random-policy actions;
- DQN exploration;
- replay-buffer sampling;
- PyTorch model initialization.

Checkpoints store:

- observation dimension;
- action count;
- training seed;
- DQN configuration;
- model state.

Training also writes `run_metadata.json` with CLI parameters and software versions.

## Tests

```bash
python -m pytest
python -m ruff check src tests scripts train.py evaluate.py
```

Tests cover:

- environment/input validation;
- explicit reward context in the observation;
- independent random-action and transition-noise RNG streams;
- deterministic replay sampling;
- checkpoint compatibility;
- paired evaluation statistics;
- safe effect-matrix serialization;
- PSC pathway keyword matching.

GitHub Actions runs the maintained project on Python 3.10, 3.11, and 3.12 and builds the Docker image.

## Docker

```bash
docker build -t rl-path-pathway-steering .
```

For a real preprocessing/training run, mount the external input data:

```bash
mkdir -p data/processed artifacts

docker run --rm \
  -v "$PWD/data/raw:/app/data/raw:ro" \
  -v "$PWD/data/processed:/app/data/processed" \
  -v "$PWD/artifacts:/app/artifacts" \
  rl-path-pathway-steering \
  --episodes 400 \
  --seed 42
```

## Interpretation limits

The repository can support claims about behavior **inside the defined simulator**, such as whether a learned policy has higher simulated return than included baselines under the reported seeds.

It does not establish drug efficacy, safety, toxicity, dosing, synergy, patient benefit, or a clinically valid treatment order.

See [MODEL_CARD.md](MODEL_CARD.md).

## License

No explicit license file is currently included. Repository visibility alone does not grant reuse rights.
