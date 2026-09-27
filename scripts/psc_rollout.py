#!/usr/bin/env python3
"""Run a PSC-focused simulated DQN rollout.

This script is a research demonstration. It does not produce a treatment
recommendation and the simulated drug/pathway effects are not clinically
validated intervention effects.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from src.dqn import DQNAgent, DQNConfig
from src.env import PathwaySteeringEnv
from src.preprocess import load_effects


PSC_KEYWORDS = {
    "immune": [
        "MHC",
        "antigen",
        "interferon",
        "IFN",
        "TNF",
        "NF-kB",
        "NFκB",
        "T cell",
        "T-cell",
        "macrophage",
    ],
    "fibrosis": [
        "TGF",
        "ECM",
        "extracellular matrix",
        "collagen",
        "wound",
        "cholangiocyte",
        "proliferation",
    ],
}


def find_matches(
    pathway_names: list[str],
    keywords: list[str],
) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    for index, name in enumerate(pathway_names):
        for keyword in keywords:
            if re.search(
                re.escape(keyword),
                name,
                flags=re.IGNORECASE,
            ):
                hits.append((index, name, keyword))
                break
    return hits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PSC-keyword-focused rollout in the RL-Path research simulator."
        )
    )
    parser.add_argument(
        "--effects",
        type=Path,
        default=Path(
            "data/processed/drug_pathway_effects_N60_P40.npz"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/dqn.pt"),
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    effect_matrix = load_effects(args.effects)

    immune_hits = find_matches(
        effect_matrix.pathway_names,
        PSC_KEYWORDS["immune"],
    )
    fibrosis_hits = find_matches(
        effect_matrix.pathway_names,
        PSC_KEYWORDS["fibrosis"],
    )

    print("\n=== Matched immune-related pathways in simulator panel ===")
    for index, name, keyword in immune_hits:
        print(f"[{index:02d}] {name} (matched: {keyword})")

    print("\n=== Matched fibrosis-related pathways in simulator panel ===")
    for index, name, keyword in fibrosis_hits:
        print(f"[{index:02d}] {name} (matched: {keyword})")

    matched_indices = sorted(
        {
            index
            for index, _, _ in (
                immune_hits + fibrosis_hits
            )
        }
    )
    if not matched_indices:
        print(
            "\nNo keyword matches found in the selected pathway panel. "
            "Increase the panel size or construct a separately validated "
            "PSC-specific pathway panel."
        )
        return 0

    disease_mask = np.zeros(
        len(effect_matrix.pathway_names),
        dtype=bool,
    )
    disease_mask[matched_indices] = True

    env = PathwaySteeringEnv(
        effects=effect_matrix.effects,
        drug_names=effect_matrix.drug_names,
        pathway_names=effect_matrix.pathway_names,
        steps=args.steps,
        seed=args.seed,
        disease_mask=disease_mask,
    )

    agent = DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        cfg=DQNConfig(),
        seed=args.seed,
    )
    agent.load(args.model)

    start = np.full(
        env.n_pathways,
        0.35,
        dtype=np.float32,
    )
    for index, _, _ in immune_hits:
        start[index] = 0.90
    for index, _, _ in fibrosis_hits:
        start[index] = 0.85

    obs = env.reset(initial_state=start)
    sequence: list[str] = []
    done = False

    while not done:
        action = agent.act(obs, greedy=True)
        result = env.step(action)
        sequence.append(str(result.info["drug"]))
        obs = result.obs
        done = result.done

    print("\n=== Simulated DQN action sequence ===")
    print(
        "Research simulation only; this is not a treatment recommendation."
    )
    for step_number, drug in enumerate(sequence, start=1):
        print(f"{step_number:02d}. {drug}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
