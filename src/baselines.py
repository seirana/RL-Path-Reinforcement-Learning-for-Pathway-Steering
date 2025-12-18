"""Baselines for RL-Path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .env import PathwaySteeringEnv


def greedy_one_step(env: PathwaySteeringEnv, obs: np.ndarray) -> int:
    """Choose the action that maximizes *immediate* reward (one-step lookahead)."""
    best_a, best_r = 0, -1e9
    # Evaluate reward from current state for each action (deterministically, without noise)
    s = obs
    for a in range(env.n_actions):
        eff = env.effects[a]
        s2 = np.clip(s - env.alpha * eff, 0.0, 1.0)
        prev = float(np.mean((s[env.disease_mask]) ** 2))
        new = float(np.mean((s2[env.disease_mask]) ** 2))
        improvement = prev - new
        action_cost = float(env.action_cost_scale * (eff.sum() / (env.n_pathways + 1e-6)))
        r = float(improvement - env.step_penalty - action_cost)
        if r > best_r:
            best_r, best_a = r, a
    return best_a


def rollout(env: PathwaySteeringEnv, policy: str = "random") -> Tuple[float, List[str]]:
    obs = env.reset()
    total = 0.0
    actions: List[str] = []
    done = False
    while not done:
        if policy == "random":
            a = env.sample_action()
        elif policy == "greedy":
            a = greedy_one_step(env, obs)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        res = env.step(a)
        obs = res.obs
        total += res.reward
        actions.append(res.info["drug"])
        done = res.done
    return float(total), actions
