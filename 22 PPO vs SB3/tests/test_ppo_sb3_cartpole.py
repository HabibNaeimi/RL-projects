import math
import numpy as np

from ppo_sb3_cartpole import (
    NUM_ENVS,
    STEPS_PER_ENV,
    MINIBATCH_SIZE,
    UPDATE_EPOCHS,
    NUM_UPDATES,
    ROLLOUT_SIZE,
    TOTAL_TIMESTEPS,
    CLIP_EPSILON,
    TARGET_KL,
    MAX_GRAD_NORM,
    SEED,
    EpisodeReturnCallback,
    make_training_env,
    build_model,
    evaluate_model,
)


def test_training_geometry():
    assert ROLLOUT_SIZE == NUM_ENVS * STEPS_PER_ENV
    assert ROLLOUT_SIZE == 1024
    assert ROLLOUT_SIZE % MINIBATCH_SIZE == 0
    assert TOTAL_TIMESTEPS == NUM_UPDATES * ROLLOUT_SIZE
    assert TOTAL_TIMESTEPS == 102_400


def test_model_uses_requested_ppo_configuration():
    env = make_training_env(SEED)

    try:
        model = build_model(env, SEED)

        assert model.n_envs == NUM_ENVS
        assert model.n_steps == STEPS_PER_ENV
        assert model.batch_size == MINIBATCH_SIZE
        assert model.n_epochs == UPDATE_EPOCHS
        assert math.isclose(model.clip_range(1.0), CLIP_EPSILON)
        assert math.isclose(model.target_kl, TARGET_KL)
        assert math.isclose(model.max_grad_norm, MAX_GRAD_NORM)
        assert model.normalize_advantage is True
        assert model.clip_range_vf is None
    finally:
        env.close()


def test_callback_collects_completed_episodes_only():
    callback = EpisodeReturnCallback()
    callback.locals = {
        "infos": [
            {},
            {"episode": {"r": 12.0, "l": 12}},
            {"episode": {"r": 25.0, "l": 25}},
        ]
    }

    assert callback._on_step() is True
    assert callback.episode_returns == [12.0, 25.0]
    assert callback.episode_lengths == [12, 25]


def test_evaluation_returns_one_result_per_episode():
    class AlwaysLeftModel:
        def predict(self, observation, deterministic):
            return np.asarray(0), None

    returns = evaluate_model(
        model=AlwaysLeftModel(),
        num_episodes=3,
        base_seed=100,
        deterministic=False,
    )

    assert len(returns) == 3
    assert np.isfinite(returns).all()
    assert all(0.0 < episode_return <= 500.0 for episode_return in returns)