import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt

NUM_EPISODES = 1000
REPORT_EVERY = 50
MOVING_AVERAGE_WINDOW = 50
EVALUATION_EPISODES = 20
POLICY_LEARNING_RATE = 1e-3
VALUE_LEARNING_RATE = 1e-3
GAMMA = 0.99
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



def choose_action(policy, observation):
    """
        Returns:
            action: Python int
            log_prob: scalar tensor that remains connected to the policy
            probs: detached tensor used only for inspection
            entropy: entropy tensor for loss calculation
    """

    observation_tensor = torch.from_numpy(observation).float()
    logits = policy(observation_tensor)                                 # Creating logits from policy
    distribution = Categorical(logits=logits)                           # Creating distributions
    action_tensor = distribution.sample()                               # Sampling from policy, not env!
    log_prob = distribution.log_prob(action_tensor)
    probs = distribution.probs.detach()                                 # detaching the tensor
    entropy = distribution.entropy()                                    # defining entropy

    return action_tensor.item(), log_prob, probs, entropy
     



def collect_episode(env, policy, seed=None):

    observation, info = env.reset(seed=seed)

    observations = []
    actions =[]
    rewards = []
    log_probs_list = []
    entropies = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, log_prob, _, entropy = choose_action(policy, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        observations.append(observation.copy())
        log_probs_list.append(log_prob)
        entropies.append(entropy)
        observation = next_observation
    
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs_list,
        "entropies": entropies,
        "episode_return": float(sum(rewards)),
        "episode_length": len(rewards),
    }



def compute_td_target(reward, next_value, terminated, gamma=GAMMA):
    """
    Returns a detached scalar TD target.
    Bootstrap when:
        terminated is False
    This includes time-limit truncation.
    """
    reward_tensor = torch.as_tensor(reward, dtype=torch.float32)
    continuation = 0.0 if terminated else 1.0
    detached_next_value = next_value.detach()
    target = reward_tensor + gamma * continuation * detached_next_value

    return target



def compute_actor_critic_losses(
        log_prob,
        entropy,
        value, 
        td_target,
        entropy_coef=ENTROPY_COEF,
):
    """
    Returns:
        actor_loss: scalar tensor
        critic_loss: scalar tensor
        td_error: scalar tensor
    """

    td_error = td_target - value
    detached_td_error = td_error.detach()
    actor_loss = -(log_prob * detached_td_error) - entropy_coef * entropy
    critic_loss = td_error.square()

    return actor_loss, critic_loss, td_error



def train_one_actor_critic_episode(
        env,
        policy,
        value_network,
        policy_optimizer,
        value_optimizer,
        gamma=GAMMA,
        entropy_coef=ENTROPY_COEF,
        seed=None,
):
    observation, info = env.reset(seed=seed)

    episode_return = 0.0
    episode_length = 0

    actor_losses = []
    critic_losses = []
    absolute_td_errors = []
    entropies = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, log_prob, _, entropy = choose_action(
            policy=policy, 
            observation=observation
            )
        
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        predicted_value = value_network(observation_tensor)
        
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation_tensor = torch.as_tensor(next_observation, dtype=torch.float32)
        next_predicted_value = value_network(next_observation_tensor)
        
        td_target = compute_td_target(
            reward=reward,
            next_value=next_predicted_value,
            terminated=terminated,
            gamma=gamma,
        )
        actor_loss, critic_loss, td_error = compute_actor_critic_losses(
            log_prob=log_prob,
            entropy=entropy,
            value=predicted_value,
            td_target=td_target,
            entropy_coef=entropy_coef,
        )

        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        policy_optimizer.step()
        value_optimizer.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        absolute_td_errors.append(td_error.detach().abs().item())
        entropies.append(entropy.detach().item())

        episode_length += 1
        episode_return += reward
        observation = next_observation

    return {
        "episode_return": episode_return,
        "episode_length": episode_length,
        "mean_actor_losses": float(np.mean(actor_losses)),
        "mean_critic_losses": float(np.mean(critic_losses)),
        "mean_absolute_td_errors": float(np.mean(absolute_td_errors)),
        "mean_entropy": float(np.mean(entropies)),
    }
    


def train_actor_critic(
        env,
        policy,
        value_network,
        policy_optimizer,
        value_optimizer,
        num_episodes,
        gamma,
        entropy_coef,
        base_seed=None,
        report_every=50,
):
    history = {
        "episode_returns": [],
        "actor_losses": [],
        "critic_losses": [],
        "mean_absolute_td_errors": [],
        "mean_entropy": [],
    }

    for index in range(num_episodes):
        if base_seed is not None:
            episode_seed = base_seed + index
        else:
            episode_seed = None

        episode_results = train_one_actor_critic_episode(
            env=env,
            policy=policy,
            value_network=value_network,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            gamma=gamma,
            entropy_coef=entropy_coef,
            seed=episode_seed,
        )
        history["episode_returns"].append(episode_results["episode_return"])
        history["actor_losses"].append(episode_results["mean_actor_losses"])
        history["critic_losses"].append(episode_results["mean_critic_losses"])
        history["mean_absolute_td_errors"].append(episode_results["mean_absolute_td_errors"])
        history["mean_entropy"].append(episode_results["mean_entropy"])

        if (index + 1) % report_every == 0:
            print('episode number:', (index + 1))
            print('mean of the most recent returns:', np.mean(history["episode_returns"][-report_every:]))
            print('most recent actor loss:', episode_results["mean_actor_losses"])
            print('most recent critic loss:', episode_results["mean_critic_losses"])
            print("Most recent mean policy entropy:", episode_results["mean_entropy"])

    return history



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
        output_path="actor_critic_td0_cartpole.png"
):
    raw_episode_numbers = np.arange(1, len(episode_returns) + 1)
    averaged_returns = moving_average(episode_returns, window)
    averaged_episode_numbers = np.arange(window, len(episode_returns) + 1)
   
    plt.figure(figsize=(10, 5))
    plt.plot(raw_episode_numbers, episode_returns, alpha=0.3)
    plt.plot(averaged_episode_numbers, averaged_returns, linewidth=2, color='red', label='Moving Average')
    plt.xlabel("Training episode")
    plt.ylabel("Episode return")
    plt.title("Online Actor–Critic with One-Step TD Error")
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
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=POLICY_LEARNING_RATE
    )

    value_network = ValueNetwork(obs_dim=obs_dim)
    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=VALUE_LEARNING_RATE,
    )

    history = train_actor_critic(
        env=env,
        policy=policy,
        value_network=value_network,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        num_episodes=NUM_EPISODES,
        gamma=GAMMA,
        base_seed=SEED,
        report_every=REPORT_EVERY,
        entropy_coef=ENTROPY_COEF,
    )

    plot_training_history(
        history['episode_returns'],
        MOVING_AVERAGE_WINDOW
    )
    
    evaluation_env = gym.make("CartPole-v1")

    evaluation_returns = evaluate_policy(
        evaluation_env,
        policy,
        num_episodes=EVALUATION_EPISODES,
        base_seed=SEED + NUM_EPISODES,
    )

    print('mean of first 50 training returns', np.mean(history["episode_returns"][:50]))
    print('mean of final  50 training returns', np.mean(history["episode_returns"][-50:]))
    print('mean evaluation return', np.mean(evaluation_returns))

    env.close()
    evaluation_env.close()

if __name__ == "__main__":
    main()


