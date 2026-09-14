import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F

CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 4

NUM_EPISODES = 1000
REPORT_EVERY = 50
MOVING_AVERAGE_WINDOW = 50
EVALUATION_EPISODES = 20

ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 1e-3

GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
SEED = 32268

np.random.seed(SEED)
torch.manual_seed(SEED)


class PolicyNetwork(nn.Module):
    def __init__(
            self,
            obs_dim,
            n_actions,
            hidden_dim=32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_actions),
        )

    def forward(self, observation):
        """
        returns the logits produced by self.network
        """
        return self.network(observation)


class ValueNetwork(nn.Module):
    def __init__(
            self, 
            obs_dim,
            hidden_dim=32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
    
    def forward(self, observation):
        raw_values = self.network(observation)

        return raw_values.squeeze(-1)                    # makes (T, 1) → (T,)


def collect_episode(env, policy, seed=None):

    states = []
    actions =[]
    rewards = []
    next_states = []
    terminated_flags = []
    old_actions_log_probs = []

   
    observation, _ = env.reset(seed=seed)

    episode_return = 0.0
    episode_length = 0

    while True:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
        with torch.no_grad():
            logits = policy(observation_tensor)         # [1, n_actions]
            distribution = Categorical(logits=logits)
            action_tensor = distribution.sample()       # [1]
            log_probs = distribution.log_prob(action_tensor)
        
        action = action_tensor.item()
        next_observation, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        states.append(observation.copy())
        next_states.append(next_observation.copy())
        terminated_flags.append(terminated)
        old_actions_log_probs.append(log_probs.item())

        episode_length += 1
        episode_return += reward

        observation = next_observation

        if terminated or truncated:
            break
    
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "terminated": terminated_flags,
        "episode_return": episode_return,
        "episode_length": episode_length,
        "old_actions_log_probs": old_actions_log_probs,
    }


@torch.no_grad()
def compute_gae(
    rewards, 
    values,
    next_values,
    terminated,
    gamma,
    gae_lambda,
):
    """
    All inputs have shape [T]. T is the episode length.

    Returns:
        advantages:   [T]
        value_targets: [T]
        td_errors:    [T]
    """

    assert rewards.ndim == 1
    assert values.shape == rewards.shape
    assert next_values.shape == rewards.shape
    assert terminated.shape == rewards.shape

    # 1.0 when the transition did NOT truly terminate, 0.0 when terminated=True
    bootstrap_mask = (~terminated).to(rewards.dtype)    
    td_errors = rewards - values + gamma * bootstrap_mask * next_values 

    advantages = torch.zeros_like(rewards)

    gae = torch.zeros(                          # zero-dimensional scalar
        (),
        dtype=rewards.dtype,
        device=rewards.device,
    )

    for t in reversed(range(rewards.shape[0])):
        gae = td_errors[t] + gamma * gae_lambda *  bootstrap_mask[t] * gae
        advantages[t] = gae
    
    value_target = advantages + values

    return advantages, value_target, td_errors



def episode_to_tensors(episode):
    states = torch.as_tensor(
        np.asarray(episode["states"]), dtype=torch.float32)
    next_states = torch.as_tensor(
        np.asarray(episode["next_states"]), dtype=torch.float32)
    rewards = torch.as_tensor(
        np.asarray(episode["rewards"]), dtype=torch.float32)
    actions = torch.as_tensor(
        np.asarray(episode["actions"]), dtype=torch.int64)
    terminated = torch.as_tensor(
        np.asarray(episode["terminated"]), dtype=torch.bool)
    old_actions_log_probs = torch.as_tensor(
        np.asarray(episode["old_actions_log_probs"]), dtype=torch.float32)
    
    return states, actions, rewards, next_states, terminated, old_actions_log_probs


def calculate_ppo_terms(
        new_actions_log_probs,
        old_actions_log_probs,
        advantages,
        clip_eps,    
):
    """
    All input tensors have shape [T].

    Returns:
        log_ratios:             [T]
        ratios:                 [T]
        clipped_ratios:         [T]
        unclipped_surrogate:    [T]
        clipped_surrogate:      [T]
        conservative_surrogate: [T]
    """
    assert new_actions_log_probs.shape == old_actions_log_probs.shape
    assert new_actions_log_probs.shape == advantages.shape
    assert not old_actions_log_probs.requires_grad
    assert not advantages.requires_grad

    log_ratios = new_actions_log_probs - old_actions_log_probs
    ratios = torch.exp(log_ratios)
    clipped_ratios = torch.clamp(ratios, min=1.0 - clip_eps, max=1.0 + clip_eps)
    unclipped_surrogate = ratios * advantages
    clipped_surrogate = clipped_ratios * advantages
    conservative_surrogate = torch.minimum(unclipped_surrogate, clipped_surrogate)  # ppo objective

    return {
        "log_ratios": log_ratios,
        "ratios": ratios,
        "clipped_ratios": clipped_ratios,
        "unclipped_surrogate": unclipped_surrogate,
        "clipped_surrogate": clipped_surrogate,
        "conservative_surrogate": conservative_surrogate,
    }


def update_ppo(
        policy,
        value_network,
        actor_optimizer,
        critic_optimizer,
        episode,
        gamma,
        gae_lambda,
        entropy_coef,
        clip_eps,
        update_epochs,
):
    (
        states,
        actions,
        rewards,
        next_states,
        terminated,
        old_log_probs,
    ) = episode_to_tensors(episode)

    T = rewards.shape[0]

    assert states.shape == (T, policy.obs_dim)
    assert next_states.shape == states.shape
    assert actions.shape == (T,)
    assert old_log_probs.shape == (T,)
    assert not old_log_probs.requires_grad

    with torch.no_grad():
        rollout_values = value_network(states)
        rollout_next_values = value_network(next_states)

        raw_advantages, value_targets, _ = compute_gae(rewards, rollout_values, rollout_next_values,
                                                       terminated, gamma, gae_lambda)
        # Normalizing raw_advantages once.
        actor_advantages = (raw_advantages - raw_advantages.mean())/(raw_advantages.std(unbiased=False) + torch.finfo(raw_advantages.dtype).eps)

    final_metrics = None

    for update_epoch in range(update_epochs):
        logits = policy(states)
        distribution = Categorical(logits=logits)
        new_log_probs = distribution.log_prob(actions)

        entropies = distribution.entropy()

        ppo_terms = calculate_ppo_terms(
            new_actions_log_probs=new_log_probs,
            old_actions_log_probs=old_log_probs,
            advantages=actor_advantages,
            clip_eps=clip_eps)
        ratios = ppo_terms["ratios"]
        conservative_surrogate = ppo_terms["conservative_surrogate"]

        actor_loss = -conservative_surrogate.mean() - entropy_coef * entropies.mean()
        assert actor_loss.ndim == 0

        # The critic target stays fixed across all update epochs.
        predicted_values = value_network(states)
        critic_loss = F.mse_loss(predicted_values, value_targets)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        with torch.no_grad():
            # Fraction of samples outside the clipping interval.
            clip_fraction = ((ratios - 1.0).abs() > clip_eps).float().mean()
            final_metrics = {
                "update_epoch": update_epoch + 1,
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "mean_entropy": entropies.mean().item(),
                "mean_ratio": ratios.mean().item(),
                "min_ratio": ratios.min().item(),
                "max_ratio": ratios.max().item(),
                "clip_fraction": clip_fraction.item(),
            }
    return final_metrics


def train(
        env,
        policy,
        value_network,
        actor_optimizer,
        critic_optimizer,
        seed,
        num_episodes,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        entropy_coef=ENTROPY_COEF,
):
    episode_returns = []

    for index in range(1, num_episodes + 1):
        if seed is not None:
            episode_seed = seed + index
        else:
            episode_seed = None

        episode = collect_episode(env=env, policy=policy, seed=episode_seed)
        update_results = update_ppo(
            policy=policy, 
            value_network=value_network,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            episode=episode,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            clip_eps=CLIP_EPSILON,
            update_epochs=UPDATE_EPOCHS,
        )

        episode_returns.append(episode["episode_return"])

        if index % REPORT_EVERY == 0:
            print('Episode number:', (index))
            print('Mean of the most recent returns:', np.mean(episode_returns[-REPORT_EVERY:]))
            print('Most recent actor loss:', update_results["actor_loss"])
            print('Most recent critic loss:', update_results["critic_loss"])
            print("Mean of most recent policy entropy:", update_results["mean_entropy"])

            print("Final update epoch:", update_results["update_epoch"])
            print("Mean ratio:", update_results["mean_ratio"])
            print("Ratio range:", (
                update_results["min_ratio"],
                update_results["max_ratio"],
            ))
            print("Clip fraction:", update_results["clip_fraction"])

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
        output_path="ppo_clipped_cartpole.png"
):
    raw_episode_numbers = np.arange(1, len(episode_returns) + 1)
    averaged_returns = moving_average(episode_returns, window)
    averaged_episode_numbers = np.arange(window, len(episode_returns) + 1)
   
    plt.figure(figsize=(10, 5))
    plt.plot(raw_episode_numbers, episode_returns, alpha=0.3)
    plt.plot(averaged_episode_numbers, averaged_returns, linewidth=2, color='red', label='Moving Average')
    plt.xlabel("Training episode")
    plt.ylabel("Episode return")
    plt.title("PPO Probability Ratios and Clipped Policy Updates")
    plt.ylim(0, 510)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_policy(
        env,
        policy,
        num_episodes=20,
        base_seed=None,
):
    evaluation_returns = []

    with torch.no_grad():
        for episode_index in range(num_episodes):
            if base_seed is not None: 
                episode_seed = base_seed + episode_index
            else: 
                episode_seed = None

            episode = collect_episode(env, policy, episode_seed)
            evaluation_returns.append(episode['episode_return'])

    return evaluation_returns


def main():

    env = gym.make("CartPole-v1")                    # creating environment.

    obs_dim = env.observation_space.shape[0]         # Observation space dimensions
    n_actions = env.action_space.n                   # Action counts

    print("Observation dimension:", obs_dim)
    print("Number of actions:", n_actions)

    policy = PolicyNetwork(obs_dim, n_actions)
    actor_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=ACTOR_LEARNING_RATE
    )

    value_network = ValueNetwork(obs_dim=obs_dim)
    critic_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=CRITIC_LEARNING_RATE,
    )

    episode_returns = train(
        env=env,
        policy=policy,
        value_network=value_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        num_episodes=NUM_EPISODES,
        gamma=GAMMA,
        seed=SEED,
        gae_lambda=GAE_LAMBDA,
        entropy_coef=ENTROPY_COEF,
    )

    plot_training_history(
        episode_returns,
        MOVING_AVERAGE_WINDOW
    )
    
    evaluation_env = gym.make("CartPole-v1")

    evaluation_returns = evaluate_policy(
        evaluation_env,
        policy,
        num_episodes=EVALUATION_EPISODES,
        base_seed=SEED + NUM_EPISODES,
    )

    print('mean of first 50 training returns', np.mean(episode_returns[:50]))
    print('mean of final  50 training returns', np.mean(episode_returns[-50:]))
    print('mean evaluation return', np.mean(evaluation_returns))

    env.close()
    evaluation_env.close()

if __name__ == "__main__":
    main()


