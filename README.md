# 🧠 RL-Path: Reinforcement Learning for Pathway Steering (Drug → Gene → Pathway)

RL-Path is a **bioinformatics reinforcement learning project** that learns a *sequence of drug interventions* to steer a simulated disease state toward a healthier state.

It uses **public data** to build a drug→gene→pathway graph and turns it into an RL environment:
- **DGIdb** drug–gene interactions (**actions = drugs**)
- **Reactome** gene → pathway mappings (**state = pathway activity vector**)
---

## 📌 1. Research Question

Can an RL agent learn an intervention policy (a sequence of drugs) that:
- reduces activity of “disease-associated” pathways, and
- does so under a cost/penalty constraint (toxicity / number of steps)?

---

## 💡 2. Proposed Solution

I build a lightIight, data-driven Markov Decision Process (MDP):

- **State**: pathway activity vector `s ∈ [0,1]^P`
- **Action**: choose a drug from DGIdb (`N` drugs)
- **Transition**: drug perturbs pathways according to its gene targets mapped to Reactome pathways
- **Reward**: improves closeness to a healthy target state while penalizing costly actions

I train a small **DQN (Deep Q-Network)** agent and compare it to a greedy baseline.

---

## ⚙️ 3. Methodology

### Data sources (downloadable)

DGIdb (TSV):
```text
https://www.dgidb.org/data/latest/interactions.tsv
```

Reactome mapping (TSV):
```text
https://download.reactome.org/current/Ensembl2Reactome.txt
```

> Note: DGIdb uses gene symbols; Reactome mapping is Ensembl-based. I map symbols → Ensembl using `mygene` (mygene.info API) and cache results locally.

### Environment design

- I pick the top `N` drugs (by number of unique target genes) and top `P` pathways (by coverage).
- I precompute a **drug→pathway effect matrix** `E ∈ R^{N×P}`.
- I define a synthetic “disease state” by assigning higher starting activity to a subset of pathways (“disease pathways”).
- A drug action decreases pathway activity proportional to `E[action]` (with optional noise and diminishing returns).

---

## 🔁 4. Evaluation Workflow

| Step | Description |
|------|-------------|
| Data download | Fetch DGIdb interactions + Reactome mapping |
| Preprocess | Map gene symbols → Ensembl, join to pathways |
| Build effects | Compute drug→pathway effect matrix |
| Train | DQN learns a policy over `T` steps |
| Evaluate | Compare DQN vs greedy baseline |
| Artifacts | Save metrics + plots + example trajectories |

---

## 🧱 5. Repository Structure

```text
RL-Path/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ src/
│  ├─ preprocess.py
│  ├─ env.py
│  ├─ dqn.py
│  └─ baselines.py
├─ train.py
├─ evaluate.py
├─ report.md
└─ artifacts/              
```

---

## 🧩 6. Usage

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download + preprocess

### Download data

https://www.dgidb.org/data/latest/interactions.tsv  stored at ./data/raw as dgidb_interactions.tsv
https://reactome.org/download/current/Ensembl2Reactome.txt stored at ./data/raw/ as Ensembl2Reactome.txt

### Train

```bash
python train.py --episodes 400 --steps 10 --top_drugs 60 --top_pathways 40
```

### Evaluate

```bash
python evaluate.py --steps 10 --top_drugs 60 --top_pathways 40
```

Artifacts land in `artifacts/`:
- `learning_curve.png`
- `policy_rollouts.json`
- `metrics.json`

---

## 🧪 7. Expected Results

After training on the drug–gene–pathway network, using
(1) PSC WES risk genes and
(2) PSC disease-relevant pathways (immune and fibrosis),

you should observe:

- Stable learning behavior of the DQN agent, with increasing episodic return as the policy improves.

- Improved pathway control compared to random or greedy baselines, measured as a larger reduction in PSC-associated pathway activity under the same step budget.

- Ordered sequences of drugs (rather than a flat ranking), reflecting the fact that pathway states change after each intervention.

- Biologically interpretable drug orders, where early drugs tend to target immune/inflammatory pathways and later drugs increasingly affect fibrosis or tissue remodeling pathways.

- Coverage of PSC disease mechanisms, showing that the learned policy preferentially selects drugs whose targets overlap PSC risk genes and PSC-enriched pathways.

- Overall, the trained model produces state-dependent treatment sequences that demonstrate how drug order matters when pathway activity evolves over time.
