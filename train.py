#!/usr/bin/env python3
"""Train DQN on the pathway-steering environment.

Typical usage:
    python scripts/download_data.py
    python train.py --episodes 400 --steps 10 --top_drugs 60 --top_pathways 40

This will create:
- data/processed/drug_pathway_effects.npz
- artifacts/learning_curve.png
- artifacts/dqn.pt
- artifacts/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preprocess import (
    build_effect_matrix,
    load_dgidb_interactions,
    load_reactome_ensembl2reactome,
    save_effects,
    load_effects,
)
from src.env import PathwaySteeringEnv
from src.dqn import DQNAgent, DQNConfig
from src.baselines import rollout


def ensure_effects(top_drugs: int, top_pathways: int, seed: int) -> Path:
    proc = Path("data/processed")
    proc.mkdir(parents=True, exist_ok=True)
    outpath = proc / f"drug_pathway_effects_N{top_drugs}_P{top_pathways}.npz"
    if outpath.exists():
        return outpath

    raw = Path("/home/shashemi/Documents/000-Xodam/ML_AI_Github_Projects/Reinforcement Learning/data/raw")
    dgidb = raw / "dgidb_interactions.tsv"
    react = raw / "Ensembl2Reactome.txt"
    if not dgidb.exists() or not react.exists():
        raise FileNotFoundError(
            "Missing raw data. Run: python scripts/download_data.py"
        )

    dgidb_df = load_dgidb_interactions(dgidb)
    react_df = load_reactome_ensembl2reactome(react)
    em = build_effect_matrix(dgidb_df, react_df, top_drugs=top_drugs, top_pathways=top_pathways, seed=seed)
    np.savez_compressed(
        outpath,
        effects=em.effects,
        drug_names=np.array(em.drug_names, dtype=object),
        pathway_ids=np.array(em.pathway_ids, dtype=object),
        pathway_names=np.array(em.pathway_names, dtype=object),
    )
    return outpath


def main() -> None:
    ap = argparse.ArgumentParser(description="RL-Path: DQN training")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_drugs", type=int, default=60)
    ap.add_argument("--top_pathways", type=int, default=40)
    ap.add_argument("--outdir", type=str, default="artifacts")
    args = ap.parse_args()

    eff_path = ensure_effects(args.top_drugs, args.top_pathways, args.seed)
    em = load_effects(eff_path)

    env = PathwaySteeringEnv(
        effects=em.effects,
        drug_names=em.drug_names,
        pathway_names=em.pathway_names,
        steps=args.steps,
        seed=args.seed,
    )

    cfg = DQNConfig()
    agent = DQNAgent(obs_dim=env.n_pathways, n_actions=env.n_actions, cfg=cfg, seed=args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    returns = []
    losses = []

    # Warm-up: some random experience
    for _ in range(200):
        obs = env.reset()
        done = False
        while not done:
            a = env.sample_action()
            res = env.step(a)
            agent.push(obs, a, res.reward, res.obs, res.done)
            obs = res.obs
            done = res.done

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(obs)
            res = env.step(a)
            agent.push(obs, a, res.reward, res.obs, res.done)
            upd = agent.update()
            if not np.isnan(upd.get("loss", np.nan)):
                losses.append(upd["loss"])
            obs = res.obs
            ep_ret += res.reward
            done = res.done

        returns.append(ep_ret)

        if ep % 50 == 0:
            print(f"ep={ep:4d} return={np.mean(returns[-20:]): .4f} eps={agent.epsilon():.3f}")

    # Save model
    agent.save(str(outdir / "dqn.pt"))

    # Baselines
    greedy_ret, greedy_actions = rollout(env, policy="greedy")
    rand_ret, rand_actions = rollout(env, policy="random")
    dqn_ret, dqn_actions = rollout(env, policy="random")  # placeholder for compatibility

    # Evaluate DQN greedy rollout
    obs = env.reset()
    tot = 0.0
    acts = []
    done = False
    while not done:
        a = agent.act(obs, greedy=True)
        res = env.step(a)
        tot += res.reward
        acts.append(res.info["drug"])
        obs = res.obs
        done = res.done
    dqn_ret, dqn_actions = float(tot), acts

    metrics = {
        "episodes": args.episodes,
        "steps": args.steps,
        "top_drugs": args.top_drugs,
        "top_pathways": args.top_pathways,
        "seed": args.seed,
        "return_mean_last20": float(np.mean(returns[-20:])),
        "greedy_return": float(greedy_ret),
        "random_return": float(rand_ret),
        "dqn_greedy_return": float(dqn_ret),
        "example_actions": {
            "greedy": greedy_actions,
            "random": rand_actions,
            "dqn": dqn_actions,
        },
    }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "returns.json").write_text(json.dumps(returns, indent=2), encoding="utf-8")

    # Plot learning curve
    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN Training Return")
    plt.savefig(outdir / "learning_curve.png", bbox_inches="tight")
    plt.close()

    print("Done. Artifacts in:", outdir.resolve())


if __name__ == "__main__":
    main()
