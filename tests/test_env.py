import numpy as np
import pytest

from src.env import PathwaySteeringEnv


def make_env(**kwargs):
    return PathwaySteeringEnv(
        effects=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        drug_names=["drug_a", "drug_b"],
        pathway_names=["path_a", "path_b"],
        disease_mask=np.array([True, False]),
        noise=0.0,
        step_penalty=0.0,
        action_cost_scale=0.0,
        **kwargs,
    )


def test_observation_exposes_reward_context_and_horizon():
    env = make_env(steps=4)
    obs = env.reset(
        initial_state=np.array(
            [0.8, 0.2],
            dtype=np.float32,
        )
    )

    assert env.obs_dim == 5
    np.testing.assert_allclose(
        obs,
        np.array(
            [0.8, 0.2, 1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
    )


def test_random_action_rng_does_not_shift_transition_noise():
    kwargs = dict(
        effects=np.array(
            [[0.2, 0.1]],
            dtype=np.float32,
        ),
        drug_names=["drug_a"],
        pathway_names=["path_a", "path_b"],
        disease_mask=np.array([True, False]),
        steps=2,
        seed=123,
        noise=0.05,
    )
    env_a = PathwaySteeringEnv(**kwargs)
    env_b = PathwaySteeringEnv(**kwargs)

    start = np.array([0.8, 0.4], dtype=np.float32)
    env_a.reset(initial_state=start)
    env_b.reset(initial_state=start)

    for _ in range(20):
        env_a.sample_action()

    env_a.step(0)
    env_b.step(0)

    np.testing.assert_allclose(env_a.state, env_b.state)


def test_expected_reward_matches_noise_free_step():
    env = make_env(steps=2)
    start = np.array([0.8, 0.2], dtype=np.float32)
    env.reset(initial_state=start)

    expected = env.expected_reward_for_action(0)
    result = env.step(0)

    assert result.reward == pytest.approx(expected)


def test_step_after_done_is_rejected():
    env = make_env(steps=1)
    env.reset()
    env.step(0)

    with pytest.raises(RuntimeError, match="already complete"):
        env.step(0)


def test_invalid_effect_values_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PathwaySteeringEnv(
            effects=np.array([[1.2]], dtype=np.float32),
            drug_names=["drug"],
            pathway_names=["path"],
        )


def test_fixed_disease_mask_cannot_be_resampled():
    env = make_env()

    with pytest.raises(ValueError, match="fixed disease_mask"):
        env.reset(resample_disease_mask=True)
