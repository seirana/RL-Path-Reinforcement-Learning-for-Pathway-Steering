# Model and simulator card

## Intended use

RL-Path is intended for research on reinforcement-learning algorithms in a transparent pathway-steering simulator.

It is suitable for:

- RL method development;
- simulator-ablation experiments;
- reproducibility demonstrations;
- comparison with simple algorithmic baselines.

It is not intended for clinical decision support, prescribing, dosing, patient stratification, or real-world treatment selection.

## State

The default observation contains:

- current pathway activity;
- the disease-pathway mask used by the reward;
- the remaining episode fraction.

These variables are observable to the policy because they influence the finite-horizon decision problem.

## Actions

Each action is a drug label from the processed DGIdb/Reactome-derived matrix.

Presence in the action space does not imply that a drug is appropriate, effective, safe, or indicated for the simulated disease context.

## Drug-to-pathway matrix

The preprocessing pipeline:

1. reads DGIdb drug-gene interactions;
2. maps gene symbols to Ensembl IDs;
3. joins those genes to Reactome pathways;
4. counts pathway coverage per drug;
5. normalizes each drug row.

The resulting values do not encode dose, direction of regulation, tissue context, exposure, pharmacokinetics, adverse effects, or clinical evidence.

## Transition

The simulator subtracts the selected row of the drug-to-pathway matrix from pathway activity after scaling by `alpha`, then adds optional Gaussian noise and clips to `[0, 1]`.

This is a mathematical state-transition rule, not a validated biological dynamical system.

## Reward

Reward is based on reduction in mean-squared activity over the simulated disease-pathway subset, minus:

- a constant step penalty;
- a breadth-based action cost.

The breadth cost is not toxicity.

## Agent

The maintained agent is a DQN with:

- two hidden fully connected layers;
- experience replay;
- epsilon-greedy exploration;
- target network;
- gradient clipping.

## Evaluation

DQN is compared with:

- a random policy;
- a myopic greedy policy that selects the action with the best noise-free one-step reward.

Policies are evaluated on paired simulator seeds. Independent training-seed sweeps are also supported.

## PSC-oriented scripts

PSC scripts use pathway-name keyword matches and local PSC gene files to build research views.

Those heuristics are not equivalent to a validated disease model, and produced action sequences are not treatment recommendations.

## Known limitations

- no patient-level state;
- no dosing or schedule units;
- no pharmacokinetic/pharmacodynamic model;
- no adverse-event or toxicity model;
- no mechanistic directionality of drug effects;
- no causal treatment-effect identification;
- no clinical outcome labels;
- no validated drug-drug interaction model;
- no external or prospective clinical validation;
- preprocessing depends on external database versions and gene-identifier mapping;
- DQN performance can be sensitive to hyperparameters and seed.

## Reproducibility

For any reported result, preserve:

- Git commit SHA;
- exact DGIdb/Reactome versions or retrieval dates;
- raw-data checksums;
- symbol-mapping cache;
- CLI arguments;
- training seeds;
- evaluation seed range;
- software versions;
- generated run metadata and metrics.
