#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""

import re
import numpy as np

from src.preprocess import load_effects
from src.env import PathwaySteeringEnv
from src.dqn import DQNAgent, DQNConfig

TOP_DRUGS = 60
TOP_PATHWAYS = 40
STEPS = 10

# Keywords you care about (we'll match by substring / regex)
PSC_KEYWORDS = {
    "immune": [
        "MHC", "antigen", "interferon", "IFN", "TNF", "NF-kB", "NFκB",
        "T cell", "T-cell", "macrophage"
    ],
    "fibrosis": [
        "TGF", "ECM", "extracellular matrix", "collagen", "wound",
        "cholangiocyte", "proliferation"
    ],
}

def find_matches(pathway_names, keywords):
    hits = []
    for i, name in enumerate(pathway_names):
        for kw in keywords:
            if re.search(re.escape(kw), name, flags=re.IGNORECASE):
                hits.append((i, name, kw))
                break
    return hits

def main():
    # Load the exact effects + names used for training
    npz_path = f"./data/rocessed/drug_pathway_effects_N{TOP_DRUGS}_P{TOP_PATHWAYS}.npz"
    em = load_effects(npz_path)

    env = PathwaySteeringEnv(
        effects=em.effects,
        drug_names=em.drug_names,
        pathway_names=em.pathway_names,
        steps=STEPS,
        seed=42,
    )

    agent = DQNAgent(obs_dim=env.n_pathways, n_actions=env.n_actions, cfg=DQNConfig(), seed=42)
    agent.load("artifacts/dqn.pt")

    # 1) Show what matches exist in YOUR trained pathway set
    immune_hits = find_matches(env.pathway_names, PSC_KEYWORDS["immune"])
    fibrosis_hits = find_matches(env.pathway_names, PSC_KEYWORDS["fibrosis"])

    print("\n=== Matched IMMUNE-related pathways in training set ===")
    for i, name, kw in immune_hits:
        print(f"[{i:02d}] {name}   (matched: {kw})")

    print("\n=== Matched FIBROSIS-related pathways in training set ===")
    for i, name, kw in fibrosis_hits:
        print(f"[{i:02d}] {name}   (matched: {kw})")

    if not immune_hits and not fibrosis_hits:
        print("\nNo matches found in the selected TOP_PATHWAYS list.")
        print("=> Increase --top_pathways or build a PSC-specific pathway panel and retrain.")
        return

    # 2) Build a PSC-like initial state (baseline + set matched pathways high)
    start = np.full(env.n_pathways, 0.35, dtype=np.float32)  # baseline
    for i, _, _ in immune_hits:
        start[i] = 0.90
    for i, _, _ in fibrosis_hits:
        start[i] = 0.85

    # 3) Roll out greedy DQN actions to get drug order
    env.reset()
    env.state = start.copy()

    seq = []
    obs = env.state.copy()
    done = False

    while not done:
        a = agent.act(obs, greedy=True)
        step = env.step(a)
        seq.append(step.info["drug"])
        obs = step.obs
        done = step.done

    print("\n=== Suggested drug order (DQN greedy rollout) ===")
    for t, d in enumerate(seq, 1):
        print(f"{t:02d}. {d}")

if __name__ == "__main__":
    main()
