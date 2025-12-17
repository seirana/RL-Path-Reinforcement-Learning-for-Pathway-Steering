# 🧬 RL-Path Project Report

## Project title
**RL-Path: Reinforcement Learning for Pathway Steering (Drug → Gene → Pathway)**

## 1) Question
In bioinformatics and drug discovery, we often want to choose interventions (drugs or target perturbations) that shift a biological system away from a disease state and toward a healthier state. This is naturally a **sequential** problem: one intervention can change the system and influence what should be applied next.

**Question:**  
Can a reinforcement learning agent learn a policy (a sequence of drug choices) that reduces the activity of “disease-associated” pathways under a fixed step budget and cost penalties?

## 2) Solution (what we built)
We built a small, interpretable RL environment where:

- **State** = pathway activity vector `s ∈ [0,1]^P`
- **Action** = choose a drug (from DGIdb)
- **Transition** = apply drug effect to pathways using a drug→gene→pathway mapping
- **Reward** = improvement in disease-pathway activity minus penalties (step + action cost)

We trained a **Deep Q-Network (DQN)** and compared it to simple baselines (random policy, greedy one-step policy).

## 3) Data
### DGIdb drug–gene interactions
Download:
```text
https://www.dgidb.org/data/latest/interactions.tsv
```

Used for:
- mapping each drug to a set of target genes
- defining the discrete action space

### Reactome gene → pathway mapping
Download:
```text
https://download.reactome.org/current/Ensembl2Reactome.txt
```

Used for:
- mapping Ensembl gene IDs to Reactome pathway IDs/names
- building pathway-level state representation

### Bridging identifiers
DGIdb interaction genes are typically **gene symbols**, while this Reactome mapping is **Ensembl-based**.  
We map `symbol → Ensembl` using the `mygene` Python package (mygene.info) and cache results locally.

## 4) Method
### 4.1 Preprocessing
1. Load DGIdb interactions (drug, gene_symbol).
2. Convert `gene_symbol → Ensembl` using mygene.info.
3. Join with Reactome Ensembl2Reactome to obtain `drug → pathway` hits.
4. Create a matrix `E` of shape `(N_drugs, P_pathways)` where `E[d, p]` is the normalized number of drug target genes in pathway `p`.

### 4.2 Environment definition
- **Initialize**: choose a subset of pathways as “disease pathways”; start them with higher activity.
- **Step**: selecting a drug reduces pathway activities proportional to `E[action]`.
- **Reward**: improvement in disease-pathway mean squared error (MSE) relative to the healthy target, minus:
  - step penalty
  - action cost (encourages narrower interventions)

### 4.3 Learning algorithm
We use a standard **DQN**:
- MLP Q-network
- Replay buffer
- Target network
- Epsilon-greedy exploration

## 5) Why reinforcement learning here?
A greedy drug choice can be shortsighted: the best immediate improvement might block later improvements (or require many costly steps). RL explicitly optimizes **long-term return** over multiple steps, which is closer to real intervention planning and combinatorial optimization.

## 6) Results
This is a small project, so results are reported as:

- training curve (episodic return vs episode)
- evaluation summary comparing DQN vs greedy vs random

Expected outcome:
- DQN should outperform random and often match or beat greedy when multi-step planning matters.

Artifacts produced:
- `artifacts/learning_curve.png`
- `artifacts/metrics.json`
- `artifacts/eval_summary.json`
- `artifacts/dqn.pt`
- cached gene mapping table in `data/processed/symbol_to_ensembl.tsv`

## 7) How to run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py
python train.py --episodes 400 --steps 10 --top_drugs 60 --top_pathways 40
python evaluate.py --steps 10 --top_drugs 60 --top_pathways 40 --n_rollouts 30
```

## 8) Limitations
- The disease state is synthetic (to keep this one-day feasible).
- We model drug effects via pathway coverage, not directionality (agonist vs antagonist).
- Mapping relies on external symbol→Ensembl conversion (mygene.info).

## 9) Extensions (next weekend-ready)
- Use your PSC gene list (WES/scDRS hits) to define disease pathways via enrichment.
- Add drug metadata for action costs (approval status, known indications).
- Include directionality by parsing interaction types (inhibit/activate) when available.
- Replace synthetic initialization with pathway scores computed from real expression (bulk or pseudo-bulk).

