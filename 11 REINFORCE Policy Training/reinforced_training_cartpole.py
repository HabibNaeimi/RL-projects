import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt

NUM_EPISODES = 800
REPORT_EVERY = 50
MOVING_AVERAGE_WINDOW = 50
EVALUATION_EPISODES = 20
LEARNING_RATE = 1E-2
GAMMA = 0.99
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


def choose_action(policy, observation):
    """
        Returns:
            action: Python int
            log_prob: scalar tensor that remains connected to the policy
            probs: detached tensor used only for inspection
    """

    observation_tensor = torch.from_numpy(observation).float()
    logits = policy(observation_tensor)                                 # Creating logits from policy
    distribution = Categorical(logits=logits)                           # Creating distributions
    action_tensor = distribution.sample()                               # Sampling from policy, not env!
    log_prob = distribution.log_prob(action_tensor)
    probs = distribution.probs.detach()                                 # detached tensor
    return action_tensor.item(), log_prob, probs


def collect_episode(env, policy, seed=None):

    observation, info = env.reset(seed=seed)
    observations = []
    actions =[]
    rewards = []
    log_probs_list = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, log_prob, _ = choose_action(policy, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        observations.append(observation.copy())
        log_probs_list.append(log_prob)
        observation = next_observation
    
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs_list,
        "episode_return": float(sum(rewards)),
        "episode_length": len(rewards)
    }


def compute_discounted_returns(rewards, gamma=GAMMA):
    """
    Args:
        rewards: rewards ordered from t=0 to t=T-1
        gamma: discount factor
    Returns:
        A float32 tensor of shape (T,), ordered from G_0 to G_{T-1}.    
    """
    discounted_returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        discounted_returns.append(running_return)
    
    discounted_returns.reverse()
    discounted_tensor = torch.tensor(discounted_returns, dtype=torch.float32)
    return discounted_tensor


def compute_reinforce_loss(log_probs, returns):
    """
    Args:
        log_probs: list of scalar tensors connected to the policy
        returns: tensor of shape (T,)

    Returns:
        Scalar REINFORCE loss tensor    
    """
    # creating a tensor of shape (T,) while preserving autograd.
    log_probs_tensor = torch.stack(log_probs)

    assert log_probs_tensor.shape == returns.shape   # dimension checking 

    loss = -(log_probs_tensor * returns).sum()
    return loss


def policy_update(optimizer, episode, gamma=GAMMA):

    returns = compute_discounted_returns(episode['rewards'], gamma)
    loss = compute_reinforce_loss(
        episode["log_probs"],
        returns,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()



def train_reinforce(
        env,
        policy,
        optimizer,
        num_episodes,
        gamma=GAMMA,
        base_seed=None,
        report_every=50
):
    episode_returns = []
    losses = []

    for episode_index in range(num_episodes):
        if base_seed is not None: 
            episode_seed = base_seed + episode_index
        else: 
            episode_seed = None

        episode = collect_episode(env, policy, episode_seed)
        loss_value = policy_update(
            optimizer,
            episode,
            gamma
        )
        losses.append(loss_value)
        episode_returns.append(episode["episode_return"])

        if (episode_index + 1) % report_every == 0:
            print('episode number:', (episode_index + 1))
            print('mean of the most recent returns:', np.mean(episode_returns[-report_every:]))
            print('most recent loss:', loss_value)
        
    return {
        "episode_returns": episode_returns,
        "losses": losses,
    }



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
        output_path="reinforce_learning_curve.png"
):
    raw_episode_numbers = np.arange(1, len(episode_returns) + 1)
    averaged_returns = moving_average(episode_returns, window)
    averaged_episode_numbers = np.arange(window, len(episode_returns) + 1)
   
    plt.figure(figsize=(10, 5))
    plt.plot(raw_episode_numbers, episode_returns, alpha=0.3)
    plt.plot(averaged_episode_numbers, averaged_returns, linewidth=2, color='red', label='Moving Average')
    plt.xlabel("Training episode")
    plt.ylabel("Episode return")
    plt.title("REINFORCE training on CartPole-v1")
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
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=LEARNING_RATE
    )

    history = train_reinforce(
        env, 
        policy,
        optimizer,
        NUM_EPISODES,
        GAMMA,
        SEED,
        REPORT_EVERY,
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


