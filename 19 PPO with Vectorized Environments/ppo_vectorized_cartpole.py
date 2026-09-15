import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F

NUM_ENVS = 4
STEPS_PER_ENV = 256
ROLLOUT_BATCH_SIZE = NUM_ENVS * STEPS_PER_ENV
assert ROLLOUT_BATCH_SIZE == 1024

MINIBATCH_SIZE = 64

NUM_UPDATES = 100
REPORT_EVERY_UPDATES = 10

CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 4

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


def collect_vector_rollout(
        vector_env,
        policy,
        observations,
        running_episode_lengths,
        running_episode_returns,
        steps_per_env,
):
    """
    Collect exactly steps_per_env transitions from every environment.

    Returns:
        rollout
        observations_for_next_rollout
        running_episode_returns
        running_episode_lengths
        completed_episode_returns
        completed_episode_lengths
    """
    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    terminated_flags_list = []
    episode_end_flags_list = []
    old_log_probs_list = []

    completed_episode_returns = []
    completed_episode_lengths = []

    N = vector_env.num_envs
    assert running_episode_returns.shape == (N,)
    assert running_episode_lengths.shape == (N,)

    for _ in range(steps_per_env):
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32) # [N, 4]
        
        with torch.no_grad():
            logits = policy(observations_tensor)                    # [N, 2]
            distribution = Categorical(logits=logits)
            action_tensor = distribution.sample()                   # [N]
            old_log_probs = distribution.log_prob(action_tensor)    # [N]

        actions = action_tensor.cpu().numpy()
        next_observations, reward, terminated, truncated, info = vector_env.step(actions)
        episode_end = np.logical_or(terminated, truncated)

        actions_list.append(actions)
        rewards_list.append(reward)
        states_list.append(observations.copy())
        episode_end_flags_list.append(episode_end)
        terminated_flags_list.append(terminated)
        old_log_probs_list.append(old_log_probs.cpu().numpy().copy())

        running_episode_lengths += 1
        running_episode_returns += reward

        transition_next_states = next_observations.copy()

        for env_index in range(vector_env.num_envs):
            if episode_end[env_index]:
                completed_episode_returns.append(float(running_episode_returns[env_index]))
                completed_episode_lengths.append(int(running_episode_lengths[env_index]))

                running_episode_lengths[env_index] = 0
                running_episode_returns[env_index] = 0

                transition_next_states[env_index] = info["final_obs"][env_index]

        next_states_list.append(transition_next_states)
        observations = next_observations
    
    rollout = {
        "states": states_list,
        "actions": actions_list,
        "rewards": rewards_list,
        "next_states": next_states_list,
        "terminated": terminated_flags_list,
        "episode_ends": episode_end_flags_list,
        "old_log_probs": old_log_probs_list,
    }

    return (
        rollout,
        observations,
        running_episode_returns,
        running_episode_lengths,
        completed_episode_returns,
        completed_episode_lengths,
    )



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


def make_vector_env(num_envs):
    """
    Return a SyncVectorEnv containing num_envs CartPole environments.

    Requirements:
    - create independent environment instances;
    - use SAME_STEP autoreset explicitly.
    """
    # List of functions that create environments.
    env_fns = [
        lambda: gym.make("CartPole-v1")
        for _ in range(num_envs)
    ]
    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    envs = gym.vector.SyncVectorEnv(
        env_fns,
        autoreset_mode=autoreset_mode)
    return envs


def make_minibatch_indices(
        num_samples,
        minibatch_size,
        device=None,
):
    """
    Return a list of one-dimensional index tensors.
    By splitting one permutation into minibatches.
    Requirements:
    - shuffle all indices;
    - include every index exactly once;
    - permit a smaller final minibatch;
    - reject nonpositive arguments.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive.")
    
    permutation = torch.randperm(num_samples, device=device)
    indices = []
    for start in range(0, num_samples, minibatch_size):
        end = start + minibatch_size
        indices.append(permutation[start:end])
    return indices



@torch.no_grad()
def compute_gae(
    rewards, 
    values,
    next_values,
    terminated,
    episode_ends,
    gamma,
    gae_lambda,
):
    """
    Implementation of Vectorized Boundary-aware GAE.
    All inputs have shape [T, N]. T is STEPS_PER_ENV and N is NUM_ENVS.

    Returns:
        advantages:   [T, N]
        value_targets: [T, N]
        td_errors:    [T, N]
    """
    # Dim checks
    assert rewards.ndim == 2
    T, N = rewards.shape
    assert values.shape == rewards.shape
    assert next_values.shape == rewards.shape
    assert terminated.shape == rewards.shape
    assert episode_ends.shape == rewards.shape

    # Zero for termination; otherwise 1. (1 - terminated), shape [T, N]
    bootstrap_mask = (~terminated).to(rewards.dtype) 
    # Zero for termination or truncation; otherwise 1. (1 - episode_ends), shape [T, N]
    trace_mask = (~episode_ends).to(rewards.dtype)

    td_errors = rewards - values + gamma * bootstrap_mask * next_values 

    advantages = torch.zeros_like(rewards)

    gae = torch.zeros(                          # N dimensional vector
        N,
        dtype=rewards.dtype,
        device=rewards.device,
    )
    for t in reversed(range(T)):
        gae = td_errors[t] + gamma * gae_lambda *  trace_mask[t] * gae
        advantages[t] = gae
    value_target = advantages + values

    return advantages, value_target, td_errors



def flatten_vector_batch(
        states,
        actions,
        old_log_probs,
        advantages,
        value_targets,
):
    """
    Flatten T and N into one training dimension while preserving alignment.
    D = observations dimension
    """
    T, N, D = states.shape
    states = states.reshape(T * N, D)
    actions = actions.reshape(T * N)
    old_log_probs = old_log_probs.reshape(T * N)
    advantages = advantages.reshape(T * N)
    value_targets = value_targets.reshape(T * N)

    return states, actions, old_log_probs, advantages, value_targets
        

def rollout_to_tensors(rollout):
    states = torch.as_tensor(
        np.asarray(rollout["states"]), dtype=torch.float32)
    next_states = torch.as_tensor(
        np.asarray(rollout["next_states"]), dtype=torch.float32)
    rewards = torch.as_tensor(
        np.asarray(rollout["rewards"]), dtype=torch.float32)
    actions = torch.as_tensor(
        np.asarray(rollout["actions"]), dtype=torch.int64)
    terminated = torch.as_tensor(
        np.asarray(rollout["terminated"]), dtype=torch.bool)
    episode_ends  = torch.as_tensor(
        np.asarray(rollout["episode_ends"]), dtype=torch.bool)
    old_log_probs = torch.as_tensor(
        np.asarray(rollout["old_log_probs"]), dtype=torch.float32)
    
    return states, actions, rewards, next_states, terminated, episode_ends, old_log_probs


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


def update_ppo_from_rollout(
        policy,
        value_network,
        actor_optimizer,
        critic_optimizer,
        rollout,
        gamma,
        gae_lambda,
        entropy_coef,
        clip_eps,
        update_epochs,
        minibatch_size,
):
    (
        states,
        actions,
        rewards,
        next_states,
        terminated,
        episode_ends,
        old_log_probs,
    ) = rollout_to_tensors(rollout)

    T, N = rewards.shape

    assert states.shape == (T, N, policy.obs_dim)
    assert next_states.shape == states.shape
    assert actions.shape == (T, N)
    assert old_log_probs.shape == (T, N)
    assert episode_ends.shape == (T, N)
    assert not old_log_probs.requires_grad

    with torch.no_grad():
        values = value_network(states)              # [T, N]
        next_values = value_network(next_states)    # [T, N]

        raw_advantages, value_targets, _ = compute_gae(rewards, values, next_values,
                                                       terminated, episode_ends, gamma, gae_lambda)
        # Normalizing raw_advantages once.
        actor_advantages = (raw_advantages - raw_advantages.mean())/(raw_advantages.std(unbiased=False) + torch.finfo(raw_advantages.dtype).eps)

    states, actions, old_log_probs, actor_advantages, value_targets = flatten_vector_batch(
        states,
        actions,
        old_log_probs,
        actor_advantages,
        value_targets,
    )
    
    final_metrics = None
    optimizer_steps = 0
    rollout_size = T * N

    for update_epoch in range(update_epochs):
        is_final_epoch = update_epoch == update_epochs - 1
        if is_final_epoch:
            total_samples = 0
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            entropy_sum = 0.0
            ratio_sum = 0.0
            clipped_sample_count = 0
            minimum_ratio = float("inf")
            maximum_ratio = float("-inf")

        minibatches = make_minibatch_indices(
            num_samples=rollout_size,
            minibatch_size=minibatch_size,
            device=states.device,
        )
        for indices in minibatches:
            state = states[indices]
            action = actions[indices]
            old_log_prob = old_log_probs[indices]
            advantage = actor_advantages[indices]
            value_target = value_targets[indices]

            logits = policy(state)
            distribution = Categorical(logits=logits)
            new_log_prob = distribution.log_prob(action)

            entropies = distribution.entropy()

            ppo_terms = calculate_ppo_terms(
                new_actions_log_probs=new_log_prob,
                old_actions_log_probs=old_log_prob,
                advantages=advantage,
                clip_eps=clip_eps)
            ratios = ppo_terms["ratios"]
            conservative_surrogate = ppo_terms["conservative_surrogate"]

            actor_loss = -conservative_surrogate.mean() - entropy_coef * entropies.mean()
            assert actor_loss.ndim == 0
            
            # The critic target stays fixed across all minibatches and epochs.
            predicted_values = value_network(state)
            critic_loss = F.mse_loss(predicted_values, value_target)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            optimizer_steps += 1

            batch_size = indices.numel()

            if is_final_epoch:
                total_samples += batch_size

                actor_loss_sum += actor_loss.item() * batch_size
                critic_loss_sum += critic_loss.item() * batch_size

                entropy_sum += entropies.sum().item()
                ratio_sum += ratios.sum().item()

                clipped_sample_count += (
                    (ratios - 1.0).abs() > clip_eps
                ).sum().item()

                minimum_ratio = min(minimum_ratio, ratios.min().item())
                maximum_ratio = max(maximum_ratio, ratios.max().item())

        if is_final_epoch:
            assert total_samples == rollout_size
            final_metrics = {
                "update_epoch": update_epoch + 1,
                "actor_loss": actor_loss_sum / total_samples,
                "critic_loss": critic_loss_sum / total_samples,
                "mean_entropy": entropy_sum / total_samples,
                "mean_ratio": ratio_sum / total_samples,
                "min_ratio": minimum_ratio,
                "max_ratio": maximum_ratio,
                "clip_fraction": clipped_sample_count / total_samples,
                "optimizer_steps": optimizer_steps,
            }
    return final_metrics


def train(
        vector_env,
        policy,
        value_network,
        actor_optimizer,
        critic_optimizer,
        seed,
        num_updates,
        steps_per_env=STEPS_PER_ENV,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        entropy_coef=ENTROPY_COEF,
        minibatch_size=MINIBATCH_SIZE,
):
    N = vector_env.num_envs
    episode_returns = []
    episode_lengths = []
    running_episode_returns = np.zeros(N)
    running_episode_lengths = np.zeros(N)
    total_environment_steps = 0

    observations, _ = vector_env.reset(seed=seed)

    for update_index in range(1, num_updates + 1):

        (
            rollout,
            observations,
            running_episode_returns,
            running_episode_lengths,
            completed_returns,
            completed_lengths,
        ) = collect_vector_rollout(
            vector_env=vector_env,
            policy=policy,
            observations=observations,
            running_episode_returns=running_episode_returns,
            running_episode_lengths=running_episode_lengths,
            steps_per_env=steps_per_env,
        )        

        total_environment_steps += steps_per_env * vector_env.num_envs

        episode_returns.extend(completed_returns)
        episode_lengths.extend(completed_lengths)

        update_results = update_ppo_from_rollout(
            policy=policy, 
            value_network=value_network,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            rollout=rollout,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            clip_eps=CLIP_EPSILON,
            update_epochs=UPDATE_EPOCHS,
            minibatch_size=minibatch_size
        )

        if update_index % REPORT_EVERY_UPDATES == 0:
            print(' Update number:', (update_index))
            print('Mean of the most recent returns:', np.mean(episode_returns[-MOVING_AVERAGE_WINDOW:]))
            print('Most recent actor loss:', update_results["actor_loss"])
            print('Most recent critic loss:', update_results["critic_loss"])
            print("Mean of most recent policy entropy:", update_results["mean_entropy"])
            print("optimizer_steps:", update_results["optimizer_steps"])
            print("Final update epoch:", update_results["update_epoch"])
            print("Mean ratio:", update_results["mean_ratio"])
            print("Ratio range:", (
                update_results["min_ratio"],
                update_results["max_ratio"],
            ))
            print("Clip fraction:", update_results["clip_fraction"])
            print("Environment steps:", total_environment_steps)
            print(
                "Completed episodes:",
                len(episode_returns),
            )
            
    return episode_returns, episode_lengths


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
        output_path="ppo_vectorized_cartpole.png"
):
    raw_episode_numbers = np.arange(1, len(episode_returns) + 1)
    averaged_returns = moving_average(episode_returns, window)
    averaged_episode_numbers = np.arange(window, len(episode_returns) + 1)
   
    plt.figure(figsize=(10, 5))
    plt.plot(raw_episode_numbers, episode_returns, alpha=0.3)
    plt.plot(averaged_episode_numbers, averaged_returns, linewidth=2, color='red', label='Moving Average')
    plt.xlabel("Training episode")
    plt.ylabel("Episode return")
    plt.title("PPO with Vectorized Environments and Shuffled Minibatches")
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
    vector_env = make_vector_env(NUM_ENVS)                  # creating environment.
    obs_dim = vector_env.single_observation_space.shape[0]  # Observation space dimensions
    n_actions = vector_env.single_action_space.n            # Action counts

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

    episode_returns, episode_lengths = train(
        vector_env=vector_env,
        policy=policy,
        value_network=value_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        num_updates=NUM_UPDATES,
        gamma=GAMMA,
        steps_per_env=STEPS_PER_ENV,
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
        base_seed=SEED + NUM_UPDATES,
    )

    print('mean of first 50 training returns', np.mean(episode_returns[:50]))
    print('mean of final  50 training returns', np.mean(episode_returns[-50:]))
    print('mean evaluation return', np.mean(evaluation_returns))

    vector_env.close()
    evaluation_env.close()

if __name__ == "__main__":
    main()
