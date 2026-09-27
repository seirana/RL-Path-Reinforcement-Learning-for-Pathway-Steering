# Experiment protocol

RL-Path is a reinforcement-learning **simulation**. Its drug-to-pathway matrix is derived from normalized network coverage rather than measured treatment effects.

## Primary policy comparison

The maintained evaluator compares:

1. DQN;
2. myopic greedy;
3. random.

Each policy is evaluated on a fresh environment with the same rollout seed.

This pairs:

- disease-mask construction;
- initial-state random draws;
- transition-noise draws.

The environment uses a separate random-action RNG, so random-policy action sampling does not alter the transition-noise stream.

## Reported statistics

For each policy the evaluator reports:

- mean return;
- sample standard deviation;
- standard error;
- a 95% normal-approximation confidence interval;
- final simulated disease-pathway MSE.

It also reports paired return differences:

- DQN minus greedy;
- DQN minus random.

These intervals characterize simulator-rollout variability. They are not confidence intervals for clinical outcomes.

## Independent training seeds

Reinforcement-learning optimization can vary by seed. Run several independent training seeds:

```bash
rlpath-seed-sweep \
  --seeds 11,22,33,44,55 \
  -- \
  --episodes 400 \
  --eval-rollouts 30
```

Each run is stored separately and the sweep summarizes DQN evaluation-return means across training seeds.

## Suggested simulator ablations

Because the simulator is deliberately simple, useful ablations include:

### Transition noise

```bash
rlpath-train --noise 0.0 --outdir artifacts/no_noise
rlpath-train --noise 0.05 --outdir artifacts/noise_005
```

### Action breadth penalty

```bash
rlpath-train --action-cost-scale 0.0 --outdir artifacts/no_action_cost
rlpath-train --action-cost-scale 0.10 --outdir artifacts/action_cost_010
```

### Disease-mask training regime

```bash
rlpath-train --outdir artifacts/resampled_masks
rlpath-train --fixed-disease-mask --outdir artifacts/fixed_mask
```

### Intervention strength

```bash
rlpath-train --alpha 0.4 --outdir artifacts/alpha_04
rlpath-train --alpha 0.8 --outdir artifacts/alpha_08
```

These experiments probe assumptions of the simulator. They do not identify real-world dose-response or toxicity relationships.

## Reproducibility artifacts

A training run writes:

- `dqn.pt`;
- `metrics.json`;
- `eval_summary.json`;
- `returns.json`;
- `losses.json`;
- `learning_curve.png`;
- `run_metadata.json`.

Generated artifacts are excluded from version control.

## Interpretation

Appropriate:

> Under the specified simulator, model configuration, and evaluation seeds, the DQN produced the reported return distribution relative to the included baselines.

Not appropriate:

> The DQN discovered an effective or safe drug sequence for PSC or another disease.
