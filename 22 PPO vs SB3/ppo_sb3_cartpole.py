import numpy as np
import gymnasium as gym
import torch
from torch import nn
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

ENV_ID = "CartPole-v1"

NUM_ENVS = 4
STEPS_PER_ENV = 256
MINIBATCH_SIZE = 64
UPDATE_EPOCHS = 4
NUM_UPDATES = 100

ROLLOUT_SIZE = NUM_ENVS * STEPS_PER_ENV

TOTAL_TIMESTEPS = ROLLOUT_SIZE * NUM_UPDATES

LEARNING_RATE = 3e-4

GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.01

EVALUATION_EPISODES = 20
MOVING_AVERAGE_WINDOW = 50
SEED = 32268

np.random.seed(SEED)
torch.manual_seed(SEED)


class EpisodeReturnCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.episode_returns = []
        self.episode_lengths = []

    def _on_step(self):
        for info in self.locals["infos"]:
            episode = info.get("episode")
            if episode is not None:  
                self.episode_lengths.append(int(info["episode"]["l"]))
                self.episode_returns.append(float(info["episode"]["r"]))
        return True
    

def make_training_env(seed, n_envs=NUM_ENVS, env_id=ENV_ID):
    training_env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
    )
    return training_env


def build_model(training_env, seed):
    policy_kwargs = {
        "net_arch": {
            "pi": [32],
            "vf": [32],
        },
        "activation_fn": nn.Tanh,
        "ortho_init": False,
    }
    model = PPO(
        "MlpPolicy",
        training_env,
        policy_kwargs=policy_kwargs,
        learning_rate=LEARNING_RATE,
        n_steps=STEPS_PER_ENV,
        batch_size=MINIBATCH_SIZE,
        n_epochs=UPDATE_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_EPSILON,
        normalize_advantage=True,
        ent_coef=ENTROPY_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,
        vf_coef=VALUE_COEF,
        seed=seed,
        verbose=1,
        device="cpu",
    )
    return model


def evaluate_model(
        model,
        num_episodes,
        base_seed,
        deterministic,
):
    evaluation_env = gym.make(ENV_ID)
    episode_returns = []

    for episode_index in range(num_episodes):
        observation, _ = evaluation_env.reset(
            seed=base_seed + episode_index
        )

        episode_return = 0.0

        while True:
            action, _ = model.predict(observation, deterministic=deterministic)
            action_scaler = action.item()
            next_observation, reward, terminated, truncated, _ = (
                evaluation_env.step(action_scaler)
            )

            episode_return += reward
            observation = next_observation

            if terminated or truncated:
                break
        
        episode_returns.append(episode_return)
    
    evaluation_env.close()
    return episode_returns


def moving_average(values, window=50):
    """
    Returns one average for each complete window.
    """
    if window <= 0:
        raise ValueError("Window must be a positive value!")
    
    if len(values) < window:
        return np.array([], dtype=np.float64)
    
    averages =[]

    for end in range(window, len(values) + 1):
        start = end - window
        window_values = values[start:end]
        average = np.mean(window_values)
        averages.append(average)

    return np.asarray(averages, dtype=np.float64)
    

def plot_training_history(
        episode_returns,
        window=50,
        output_path="ppo_sb3_cartpole.png"
):
    raw_episode_numbers = np.arange(1, len(episode_returns) + 1)
    averaged_returns = moving_average(episode_returns, window)
    averaged_episode_numbers = np.arange(window, len(episode_returns) + 1)
   
    plt.figure(figsize=(10, 5))
    plt.plot(raw_episode_numbers, episode_returns, alpha=0.3)
    plt.plot(averaged_episode_numbers, averaged_returns, linewidth=2, color='red', label='Moving Average')
    plt.xlabel("Training episode")
    plt.ylabel("Episode return")
    plt.title("PPO Comparison with Stable-Baselines3")
    plt.ylim(0, 510)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    training_env = make_training_env(SEED)
    callback = EpisodeReturnCallback()
    model = build_model(training_env, SEED)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        log_interval=10,
    )
    assert model.num_timesteps == TOTAL_TIMESTEPS

    episode_lengths = callback.episode_lengths
    episode_returns = callback.episode_returns
    number_of_completed_episodes = len(episode_returns)

    if number_of_completed_episodes < 50:
        raise ValueError("Training completed fewer than 50 episodes.")
    
    plot_training_history(
        episode_returns,
        MOVING_AVERAGE_WINDOW
    )

    mean_first_50_returns = np.mean(episode_returns[:50])
    mean_final_50_returns = np.mean(episode_returns[-50:])      
    mean_first_50_lengths = np.mean(episode_lengths[:50])
    mean_final_50_lengths = np.mean(episode_lengths[-50:])      

    stochastic_returns = evaluate_model(
        model=model,
        num_episodes=EVALUATION_EPISODES,
        base_seed=SEED + NUM_UPDATES,
        deterministic=False,
    )

    deterministic_returns = evaluate_model(
        model=model,
        num_episodes=EVALUATION_EPISODES,
        base_seed=SEED + NUM_UPDATES,
        deterministic=True,
    )

    print('number of completed training episodes:', number_of_completed_episodes)
    print('mean of first 50 training returns:', mean_first_50_returns)
    print('mean of final  50 training returns:', mean_final_50_returns)
    print('mean of first 50 training lengths:', mean_first_50_lengths)
    print('mean of final  50 training lengths:', mean_final_50_lengths)
    print('mean stochastic evaluation return = ', np.mean(stochastic_returns))
    print('std stochastic evaluation return = ', np.std(stochastic_returns))
    print('mean deterministic evaluation return = ', np.mean(deterministic_returns))
    print('std deterministic evaluation return = ', np.std(deterministic_returns))

    training_env.close()
if __name__ == "__main__":
    main()
