#!/usr/bin/env python3
"""Evaluate a trained RL-Path DQN against paired simulator baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dqn import DQNAgent, DQNConfig
from src.env import PathwaySteeringEnv
from src.evaluation import evaluate_policies
from src.preprocess import EffectMatrix, load_effects


def build_env(
    args: argparse.Namespace,
    effect_matrix: EffectMatrix,
    *,
    seed: int,
) -> PathwaySteeringEnv:
    return PathwaySteeringEnv(
        effects=effect_matrix.effects,
        drug_names=effect_matrix.drug_names,
        pathway_names=effect_matrix.pathway_names,
        steps=args.steps,
        seed=seed,
        alpha=args.alpha,
        step_penalty=args.step_penalty,
        action_cost_scale=args.action_cost_scale,
        noise=args.noise,
        disease_pathway_frac=args.disease_pathway_frac,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RL-Path DQN, greedy, and random policies "
            "on paired simulator seeds."
        )
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--top-drugs", "--top_drugs", dest="top_drugs", type=int, default=60)
    parser.add_argument(
        "--top-pathways",
        "--top_pathways",
        dest="top_pathways",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/dqn.pt"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument("--n-rollouts", "--n_rollouts", dest="n_rollouts", type=int, default=30)

    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--step-penalty", type=float, default=0.02)
    parser.add_argument("--action-cost-scale", type=float, default=0.05)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--disease-pathway-frac", type=float, default=0.35)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--min-replay", type=int, default=1_000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=10_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.n_rollouts <= 0:
        raise ValueError("n-rollouts must be greater than 0")

    effect_path = args.processed_dir / (
        f"drug_pathway_effects_N{args.top_drugs}_P{args.top_pathways}.npz"
    )
    if not effect_path.exists():
        raise FileNotFoundError(
            f"Missing effect matrix: {effect_path}. "
            "Run train.py first or create the processed matrix."
        )

    effect_matrix = load_effects(effect_path)
    reference_env = build_env(
        args,
        effect_matrix,
        seed=args.seed,
    )

    config = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        min_replay=args.min_replay,
        target_update=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        hidden_dim=args.hidden_dim,
        grad_clip_norm=args.grad_clip_norm,
    )
    agent = DQNAgent(
        obs_dim=reference_env.obs_dim,
        n_actions=reference_env.n_actions,
        cfg=config,
        seed=args.seed,
    )
    agent.load(args.model)

    summary = evaluate_policies(
        make_env=lambda rollout_seed: build_env(
            args,
            effect_matrix,
            seed=rollout_seed,
        ),
        agent=agent,
        n_rollouts=args.n_rollouts,
        seed=args.seed,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    output_path = args.outdir / "eval_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
