"""RL environment: pathway steering with drug actions.

This is a tiny, transparent environment (Gym-like API but no dependency).
State: pathway activity vector in [0,1]^P
Action: choose a drug index in [0, N_drugs)
Reward: move toward healthy target while paying action cost

The transition uses a precomputed drug→pathway effect matrix E:
- action decreases pathway activity for pathways it covers (scaled)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict


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
    ) -> None:
        """
        Parameters
        ----------
        effects
            Drug→pathway matrix in [0,1], shape (N_drugs, N_pathways).
        alpha
            Strength of intervention effect.
        step_penalty
            Constant penalty each step (discourages long sequences).
        action_cost_scale
            Penalty proportional to action “breadth” (row sum).
        noise
            Small Gaussian noise in transitions.
        disease_pathway_frac
            Fraction of pathways treated as “disease-high” at start.
        """
        self.effects = effects.astype(np.float32)
        self.drug_names = list(drug_names)
        self.pathway_names = list(pathway_names)
        self.n_actions, self.n_pathways = self.effects.shape
        self.max_steps = int(steps)

        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.step_penalty = float(step_penalty)
        self.action_cost_scale = float(action_cost_scale)
        self.noise = float(noise)

        self.disease_pathway_frac = float(disease_pathway_frac)
        self.disease_mask = self._make_disease_mask()

        # Targets: “healthy” is low activity on disease pathways
        self.target = np.zeros(self.n_pathways, dtype=np.float32)

        self.t = 0
        self.state = np.zeros(self.n_pathways, dtype=np.float32)

    def _make_disease_mask(self) -> np.ndarray:
        k = max(1, int(round(self.n_pathways * self.disease_pathway_frac)))
        idx = self.rng.choice(self.n_pathways, size=k, replace=False)
        mask = np.zeros(self.n_pathways, dtype=bool)
        mask[idx] = True
        return mask

    def reset(self) -> np.ndarray:
        self.t = 0
        # Disease pathways start high, others moderate
        s = self.rng.uniform(0.25, 0.55, size=self.n_pathways).astype(np.float32)
        s[self.disease_mask] = self.rng.uniform(0.65, 0.95, size=int(self.disease_mask.sum())).astype(np.float32)
        self.state = s
        return self.state.copy()

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action out of range: {action}")

        self.t += 1
        s = self.state

        eff = self.effects[action]  # in [0,1]
        # Main effect: decrease activity for covered pathways
        delta = -self.alpha * eff

        # Add a tiny stabilizing term so non-covered pathways don't drift too much
        eps = self.rng.normal(0.0, self.noise, size=self.n_pathways).astype(np.float32)
        s2 = np.clip(s + delta + eps, 0.0, 1.0).astype(np.float32)

        # Reward: reduce distance to target, emphasize disease pathways
        prev_dist = float(np.mean((s[self.disease_mask] - self.target[self.disease_mask]) ** 2))
        new_dist = float(np.mean((s2[self.disease_mask] - self.target[self.disease_mask]) ** 2))
        improvement = prev_dist - new_dist

        # Action cost: broad drugs are more costly (row sum is 1 after normalization, but keep flexible)
        action_cost = float(self.action_cost_scale * (eff.sum() / (self.n_pathways + 1e-6)))

        reward = float(improvement - self.step_penalty - action_cost)

        self.state = s2
        done = self.t >= self.max_steps

        info = {
            "t": self.t,
            "drug": self.drug_names[action],
            "prev_disease_mse": prev_dist,
            "new_disease_mse": new_dist,
            "improvement": improvement,
            "action_cost": action_cost,
        }
        return StepResult(obs=s2.copy(), reward=reward, done=done, info=info)

    def sample_action(self) -> int:
        return int(self.rng.integers(0, self.n_actions))
