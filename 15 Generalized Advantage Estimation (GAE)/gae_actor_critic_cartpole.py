import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F

NUM_EPISODES = 1000
REPORT_EVERY = 50
MOVING_AVERAGE_WINDOW = 50
EVALUATION_EPISODES = 20

ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 1e-3
# POLICY_LEARNING_RATE = 1e-3
# VALUE_LEARNING_RATE = 1e-3

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

   
    observation, _ = env.reset(seed=seed)

    episode_return = 0.0
    episode_length = 0

    while True:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
        with torch.no_grad():
            logits = policy(observation_tensor)         # [1, n_actions]
            distribution = Categorical(logits=logits)
            action_tensor = distribution.sample()       # [1]
        
        action = action_tensor.item()
        next_observation, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        states.append(observation.copy())
        next_states.append(next_observation.copy())
        terminated_flags.append(terminated)

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
    
    return states, actions, rewards, next_states, terminated


def update_actor_critic(
        policy,
        value_network,
        actor_optimizer,
        critic_optimizer,
        episode,
        gamma,
        gae_lambda,
        entropy_coef,
):
    (
        states,
        actions,
        rewards,
        next_states,
        terminated,
    ) = episode_to_tensors(episode=episode)

    T = rewards.shape[0]

    assert states.shape[0] == T
    assert actions.shape[0] == T
    assert next_states.shape[0] == T
    assert terminated.shape[0] == T

    with torch.no_grad():
        rollout_values = value_network(states)
        rollout_next_values = value_network(next_states)

        raw_advantages, value_targets, td_errors = compute_gae(
            rewards=rewards,
            values=rollout_values,
            next_values=rollout_next_values,
            terminated=terminated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    assert raw_advantages.shape == (T,)
    assert value_targets.shape == (T,)
    assert not raw_advantages.requires_grad
    assert not value_targets.requires_grad

    actor_advantages = (raw_advantages - raw_advantages.mean())/(raw_advantages.std(unbiased=False) + torch.finfo(raw_advantages.dtype).eps)

    logits = policy(states)
    distribution = Categorical(logits=logits)
    log_probs = distribution.log_prob(actions)
    entropies = distribution.entropy()

    assert log_probs.shape == (T,)
    assert log_probs.requires_grad
    assert not actor_advantages.requires_grad

    actor_loss = -(log_probs * actor_advantages).mean() - (entropy_coef * entropies.mean())

    predicted_values = value_network(states)

    assert predicted_values.shape == (T,)
    assert predicted_values.requires_grad
    assert not value_targets.requires_grad

    critic_loss = F.mse_loss(predicted_values, value_targets)

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    actor_loss.backward()
    critic_loss.backward()

    actor_optimizer.step()
    critic_optimizer.step()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "mean_entropy": entropies.mean().item(),
        "mean_raw_advantages": raw_advantages.mean().item(),
        "mean_td_error": td_errors.mean().item(),
    }


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
        update_results = update_actor_critic(
            policy=policy, 
            value_network=value_network,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            episode=episode,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
        )

        episode_returns.append(episode["episode_return"])

        if index % REPORT_EVERY == 0:
            print('Episode number:', (index))
            print('Mean of the most recent returns:', np.mean(episode_returns[-REPORT_EVERY:]))
            print('Most recent actor loss:', update_results["actor_loss"])
            print('Most recent critic loss:', update_results["critic_loss"])
            print("Mean of most recent policy entropy:", update_results["mean_entropy"])
            print("Mean of most recent raw advantages:", update_results["mean_raw_advantages"])

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
        output_path="gae_actor_critic_cartpole_training.png"
):
    raw_episode_numbers = np.arange(1, len(episode_returns) + 1)
    averaged_returns = moving_average(episode_returns, window)
    averaged_episode_numbers = np.arange(window, len(episode_returns) + 1)
   
    plt.figure(figsize=(10, 5))
    plt.plot(raw_episode_numbers, episode_returns, alpha=0.3)
    plt.plot(averaged_episode_numbers, averaged_returns, linewidth=2, color='red', label='Moving Average')
    plt.xlabel("Training episode")
    plt.ylabel("Episode return")
    plt.title("Actor–Critic with Generalized Advantage Estimation")
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


