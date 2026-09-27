"""Transparent pathway-steering reinforcement-learning environment.

This module defines a research simulator, not a pharmacological model.

Observation
-----------
The default observation concatenates:
1. current pathway activity;
2. the disease-pathway mask used by the reward;
3. the remaining episode fraction.

Action
------
One discrete drug index.

Reward
------
Reduction in mean-squared activity on the simulated disease pathways, minus a
per-step penalty and a breadth-based action cost.

The disease mask is exposed because it affects the reward. The remaining horizon
is exposed because the task has a finite episode length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, object]


class PathwaySteeringEnv:
    def __init__(
        self,
        effects: np.ndarray,
        drug_names: List[str],
        pathway_names: List[str],
        steps: int = 10,
        seed: int = 42,
        alpha: float = 0.8,
        step_penalty: float = 0.02,
        action_cost_scale: float = 0.05,
        noise: float = 0.01,
        disease_pathway_frac: float = 0.35,
        disease_mask: Optional[np.ndarray] = None,
        include_disease_mask: bool = True,
        include_remaining_fraction: bool = True,
    ) -> None:
        effects_array = np.asarray(effects, dtype=np.float32)
        if effects_array.ndim != 2:
            raise ValueError("effects must have shape (N_drugs, N_pathways)")
        if effects_array.shape[0] == 0 or effects_array.shape[1] == 0:
            raise ValueError("effects must contain at least one drug and pathway")
        if not np.all(np.isfinite(effects_array)):
            raise ValueError("effects must contain only finite values")
        if np.any(effects_array < 0.0) or np.any(effects_array > 1.0):
            raise ValueError("effects values must be in [0, 1]")

        self.effects = effects_array
        self.drug_names = list(drug_names)
        self.pathway_names = list(pathway_names)
        self.n_actions, self.n_pathways = self.effects.shape

        if len(self.drug_names) != self.n_actions:
            raise ValueError("drug_names length must match effects.shape[0]")
        if len(self.pathway_names) != self.n_pathways:
            raise ValueError("pathway_names length must match effects.shape[1]")

        self.max_steps = int(steps)
        if self.max_steps <= 0:
            raise ValueError("steps must be greater than 0")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if step_penalty < 0:
            raise ValueError("step_penalty must be non-negative")
        if action_cost_scale < 0:
            raise ValueError("action_cost_scale must be non-negative")
        if noise < 0:
            raise ValueError("noise must be non-negative")
        if not 0.0 < disease_pathway_frac <= 1.0:
            raise ValueError("disease_pathway_frac must be in (0, 1]")

        self.seed = int(seed)
        self._set_rngs(self.seed)

        self.alpha = float(alpha)
        self.step_penalty = float(step_penalty)
        self.action_cost_scale = float(action_cost_scale)
        self.noise = float(noise)
        self.disease_pathway_frac = float(disease_pathway_frac)
        self.include_disease_mask = bool(include_disease_mask)
        self.include_remaining_fraction = bool(include_remaining_fraction)

        if disease_mask is None:
            self.disease_mask = self._make_disease_mask()
            self._fixed_disease_mask = False
        else:
            mask = np.asarray(disease_mask, dtype=bool)
            if mask.shape != (self.n_pathways,):
                raise ValueError(
                    "disease_mask must have shape "
                    f"({self.n_pathways},), got {mask.shape}"
                )
            if not np.any(mask):
                raise ValueError("disease_mask must select at least one pathway")
            self.disease_mask = mask.copy()
            self._fixed_disease_mask = True

        self.target = np.zeros(self.n_pathways, dtype=np.float32)
        self.t = 0
        self.state = np.zeros(self.n_pathways, dtype=np.float32)

        self.obs_dim = self.n_pathways
        if self.include_disease_mask:
            self.obs_dim += self.n_pathways
        if self.include_remaining_fraction:
            self.obs_dim += 1

    def _set_rngs(self, seed: int) -> None:
        sequence = np.random.SeedSequence(int(seed))
        transition_seed, action_seed = sequence.spawn(2)
        self.rng = np.random.default_rng(transition_seed)
        self.action_rng = np.random.default_rng(action_seed)

    def _make_disease_mask(self) -> np.ndarray:
        k = max(
            1,
            int(round(self.n_pathways * self.disease_pathway_frac)),
        )
        idx = self.rng.choice(
            self.n_pathways,
            size=k,
            replace=False,
        )
        mask = np.zeros(self.n_pathways, dtype=bool)
        mask[idx] = True
        return mask

    def _remaining_fraction(self) -> np.ndarray:
        remaining = max(self.max_steps - self.t, 0)
        return np.array(
            [remaining / float(self.max_steps)],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        chunks = [self.state.astype(np.float32, copy=True)]
        if self.include_disease_mask:
            chunks.append(self.disease_mask.astype(np.float32))
        if self.include_remaining_fraction:
            chunks.append(self._remaining_fraction())
        return np.concatenate(chunks).astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        resample_disease_mask: bool = False,
        initial_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
            self._set_rngs(self.seed)

        if resample_disease_mask:
            if self._fixed_disease_mask:
                raise ValueError(
                    "Cannot resample a user-supplied fixed disease_mask"
                )
            self.disease_mask = self._make_disease_mask()

        self.t = 0

        if initial_state is not None:
            state = np.asarray(initial_state, dtype=np.float32)
            if state.shape != (self.n_pathways,):
                raise ValueError(
                    "initial_state must have shape "
                    f"({self.n_pathways},), got {state.shape}"
                )
            if not np.all(np.isfinite(state)):
                raise ValueError("initial_state must contain only finite values")
            self.state = np.clip(state, 0.0, 1.0).astype(np.float32)
            return self._get_obs()

        state = self.rng.uniform(
            0.25,
            0.55,
            size=self.n_pathways,
        ).astype(np.float32)
        state[self.disease_mask] = self.rng.uniform(
            0.65,
            0.95,
            size=int(self.disease_mask.sum()),
        ).astype(np.float32)
        self.state = state
        return self._get_obs()

    def disease_mse(self) -> float:
        return float(
            np.mean(
                (
                    self.state[self.disease_mask]
                    - self.target[self.disease_mask]
                )
                ** 2
            )
        )

    def action_cost(self, action: int) -> float:
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action out of range: {action}")
        effect = self.effects[action]
        return float(
            self.action_cost_scale
            * (effect.sum() / (self.n_pathways + 1e-6))
        )

    def expected_reward_for_action(self, action: int) -> float:
        """One-step reward with transition noise set to zero."""

        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action out of range: {action}")

        state = self.state
        effect = self.effects[action]
        next_state = np.clip(
            state - self.alpha * effect,
            0.0,
            1.0,
        )
        previous = float(
            np.mean((state[self.disease_mask]) ** 2)
        )
        new = float(
            np.mean((next_state[self.disease_mask]) ** 2)
        )
        return float(
            previous
            - new
            - self.step_penalty
            - self.action_cost(action)
        )

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action out of range: {action}")
        if self.t >= self.max_steps:
            raise RuntimeError(
                "Episode is already complete; call reset() before step()."
            )

        self.t += 1
        state = self.state.astype(np.float32, copy=True)
        previous_mse = self.disease_mse()

        effect = self.effects[action]
        noise = self.rng.normal(
            0.0,
            self.noise,
            size=self.n_pathways,
        ).astype(np.float32)
        next_state = np.clip(
            state - self.alpha * effect + noise,
            0.0,
            1.0,
        ).astype(np.float32)

        self.state = next_state
        new_mse = self.disease_mse()
        improvement = previous_mse - new_mse
        action_cost = self.action_cost(action)
        reward = float(
            improvement
            - self.step_penalty
            - action_cost
        )

        done = self.t >= self.max_steps
        info: Dict[str, object] = {
            "t": self.t,
            "drug": self.drug_names[action],
            "prev_disease_mse": previous_mse,
            "new_disease_mse": new_mse,
            "improvement": improvement,
            "action_cost": action_cost,
            "effect_sum": float(effect.sum()),
        }
        return StepResult(
            obs=self._get_obs(),
            reward=reward,
            done=done,
            info=info,
        )

    def sample_action(self) -> int:
        return int(
            self.action_rng.integers(
                0,
                self.n_actions,
            )
        )
