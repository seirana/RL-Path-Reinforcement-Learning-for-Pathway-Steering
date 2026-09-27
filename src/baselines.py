"""Baselines for the RL-Path simulator."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .env import PathwaySteeringEnv


def greedy_one_step(
    env: PathwaySteeringEnv,
    obs: np.ndarray,
) -> int:
    """Choose the action with the best noise-free immediate reward."""

    if obs.shape != (env.obs_dim,):
        raise ValueError(
            f"obs must have shape ({env.obs_dim},), got {obs.shape}"
        )

    rewards = [
        env.expected_reward_for_action(action)
        for action in range(env.n_actions)
    ]
    return int(np.argmax(rewards))


def rollout(
    env: PathwaySteeringEnv,
    policy: str = "random",
) -> Tuple[float, List[str]]:
    obs = env.reset()
    total = 0.0
    actions: List[str] = []
    done = False

    while not done:
        if policy == "random":
            action = env.sample_action()
        elif policy == "greedy":
            action = greedy_one_step(env, obs)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        result = env.step(action)
        obs = result.obs
        total += result.reward
        actions.append(str(result.info["drug"]))
        done = result.done

    return float(total), actions
