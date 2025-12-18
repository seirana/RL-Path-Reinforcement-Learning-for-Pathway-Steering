#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""

"""Evaluate trained DQN vs baselines and save rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.preprocess import load_effects
from src.env import PathwaySteeringEnv
from src.dqn import DQNAgent, DQNConfig
from src.baselines import rollout


def main() -> None:
    ap = argparse.ArgumentParser(description="RL-Path: evaluate")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_drugs", type=int, default=60)
    ap.add_argument("--top_pathways", type=int, default=40)
    ap.add_argument("--model", type=str, default="artifacts/dqn.pt")
    ap.add_argument("--outdir", type=str, default="artifacts")
    ap.add_argument("--n_rollouts", type=int, default=30)
    args = ap.parse_args()

    eff_path = Path(f"data/processed/drug_pathway_effects_N{args.top_drugs}_P{args.top_pathways}.npz")
    if not eff_path.exists():
        raise FileNotFoundError(
            f"Missing effect matrix: {eff_path}. Run train.py first (or ensure preprocessing ran)."

        )
    em = load_effects(eff_path)

    env = PathwaySteeringEnv(
        effects=em.effects,
        drug_names=em.drug_names,
        pathway_names=em.pathway_names,
        steps=args.steps,
        seed=args.seed,
    )

    agent = DQNAgent(obs_dim=env.n_pathways, n_actions=env.n_actions, cfg=DQNConfig(), seed=args.seed)
    agent.load(args.model)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def dqn_rollout() -> float:
        obs = env.reset()
        tot = 0.0
        done = False
        while not done:
            a = agent.act(obs, greedy=True)
            res = env.step(a)
            tot += res.reward
            obs = res.obs
            done = res.done
        return float(tot)

    dqn_rets = [dqn_rollout() for _ in range(args.n_rollouts)]
    greedy_rets = [rollout(env, policy="greedy")[0] for _ in range(args.n_rollouts)]
    rand_rets = [rollout(env, policy="random")[0] for _ in range(args.n_rollouts)]

    summary = {
        "n_rollouts": args.n_rollouts,
        "dqn": {"mean": float(np.mean(dqn_rets)), "std": float(np.std(dqn_rets))},
        "greedy": {"mean": float(np.mean(greedy_rets)), "std": float(np.std(greedy_rets))},
        "random": {"mean": float(np.mean(rand_rets)), "std": float(np.std(rand_rets))},
    }
    (outdir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
