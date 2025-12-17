# 🧠 RL-Path: Reinforcement Learning for Pathway Steering (Drug → Gene → Pathway)

RL-Path is a **small bioinformatics reinforcement learning project** that learns a *sequence of drug interventions* to steer a simulated disease state toward a healthier state.

It uses **public data** to build a drug→gene→pathway graph and turns it into an RL environment:
- **DGIdb** drug–gene interactions (**actions = drugs**)
- **Reactome** gene → pathway mappings (**state = pathway activity vector**)

The goal is to keep the project **one-day doable**, interpretable, and aligned with your drug/network + pathway background.

---

## 📌 1. Research Question

Can an RL agent learn an intervention policy (a sequence of drugs) that:
- reduces activity of “disease-associated” pathways, and
- does so under a cost/penalty constraint (toxicity / number of steps)?

---

## 💡 2. Proposed Solution

We build a lightweight, data-driven Markov Decision Process (MDP):

- **State**: pathway activity vector `s ∈ [0,1]^P`
- **Action**: choose a drug from DGIdb (`N` drugs)
- **Transition**: drug perturbs pathways according to its gene targets mapped to Reactome pathways
- **Reward**: improves closeness to a healthy target state while penalizing costly actions

We train a small **DQN (Deep Q-Network)** agent and compare it to a greedy baseline.

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

> Note: DGIdb uses gene symbols; Reactome mapping is Ensembl-based. We map symbols → Ensembl using `mygene` (mygene.info API) and cache results locally.

### Environment design

- We pick the top `N` drugs (by number of unique target genes) and top `P` pathways (by coverage).
- We precompute a **drug→pathway effect matrix** `E ∈ R^{N×P}`.
- We define a synthetic “disease state” by assigning higher starting activity to a subset of pathways (“disease pathways”).
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
├─ LICENSE
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

You should see:
- increasing episodic return for DQN
- DQN achieves better final “health distance” than greedy under the same step budget
- interpretable sequences of drugs (actions) that cover disease pathways

---

## 👩‍💻 Author

Developed by Seirana, generated with assistance from Leo.
