"""
Monte Carlo with Linear Function Approximator

This module demonstrates how to implement Monte Carlo reinforcement learning
using a linear function approximator instead of a Q-table.

The value function is approximated as: V(s) = w^T * phi(s)
where phi(s) is a feature vector and w are the weights to be learned.
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from linear_approximator import LinearFunctionApproximator
class LinearMonteCarloAgent:
    """
    Monte Carlo agent with linear function approximator.
    """
    
    def __init__(self, action_space_n, num_features=8, alpha=0.01, gamma=0.99, epsilon=0.1, scale=100.0):
        """
        Initialize the agent.
        
        Args:
            action_space_n: Number of possible actions
            num_features: Number of features for approximator
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration parameter
        """
        self.action_space_n = action_space_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        
        # One approximator per action (pass scale to feature normalizer)
        self.approximators = [
            LinearFunctionApproximator(num_features, learning_rate=alpha, scale=scale)
            for _ in range(action_space_n)
        ]
    
    def get_action_values(self, state):
        """
        Get Q-values for all actions in a state.
        
        Args:
            state: Current state
            
        Returns:
            Array of Q-values for each action
        """
        q_values = np.array([
            approximator.predict(state)
            for approximator in self.approximators
        ])
        return q_values
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_n - 1)
        
        q_values = self.get_action_values(state)
        return np.argmax(q_values)
    
    def update_episode(self, episode_trajectory):
        """
        Update approximators using an episode trajectory.
        Monte Carlo uses the full return (sum of discounted rewards).
        
        Args:
            episode_trajectory: List of (state, action, reward) tuples
        """
        # Calculate returns (discounted cumulative rewards)
        returns = []
        G = 0
        
        # Work backwards through the episode
        for state, action, reward in reversed(episode_trajectory):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Update each state-action pair
        for (state, action, reward), G in zip(episode_trajectory, returns):
            self.approximators[action].update(state, G)


def train_with_linear_approximator(env, num_episodes=1000, num_features=8, 
                                   alpha=0.01, gamma=0.99, epsilon=0.1,
                                   epsilon_decay=0.995, epsilon_min=0.01,
                                   scale=100.0):
    """
    Train an agent using Monte Carlo with linear function approximation.
    
    Args:
        env: Gymnasium environment
        num_episodes: Number of training episodes
        num_features: Number of features for the approximator
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Initial exploration parameter
        epsilon_decay: Decay rate for epsilon
        epsilon_min: Minimum epsilon value
        
    Returns:
        agent: Trained agent
        rewards: List of rewards per episode
    """
    agent = LinearMonteCarloAgent(
        env.action_space.n,
        num_features=num_features,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        scale=scale,
    )
    
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_trajectory = []
        episode_reward = 0
        episode_rewards_list = []
        done = False
        
        # Collect episode
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_trajectory.append((state, action, reward))
            episode_reward += reward
            episode_rewards_list.append(reward)
            state = next_state
        
        # Update agent using collected episode
        agent.update_episode(episode_trajectory)

        # Record discounted return for the episode (G_0 = sum_t gamma^t * r_t)
        G = 0.0
        for r in reversed(episode_rewards_list):
            G = r + agent.gamma * G
        rewards.append(G)
        
        # Decay epsilon
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])   
    return agent, rewards


def evaluate_agent(agent, env, num_episodes=100):
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained agent
        env: Gymnasium environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        List of rewards per episode
    """
    rewards = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Greedy evaluation
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_rewards_list = []
        done = False

        while not done:
            q_values = agent.get_action_values(state)
            action = np.argmax(q_values)
            next_state, reward, done, _, _ = env.step(action)

            episode_rewards_list.append(reward)
            state = next_state

        # compute discounted return for episode
        G = 0.0
        for r in reversed(episode_rewards_list):
            G = r + agent.gamma * G
        rewards.append(G)
    
    agent.epsilon = original_epsilon
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Evaluation - Average reward (discounted): {avg_reward:.2f} ± {std_reward:.2f}")
    
    return rewards


if __name__ == "__main__":
    import gymnasium as gym
    
    # Example: FrozenLake environment
    print("Creating environment...")
    env = gym.make("FrozenLake-v1", is_slippery=True)
    
    print("\nTraining Monte Carlo agent with linear function approximator...")
    agent, training_rewards = train_with_linear_approximator(
        env,
        num_episodes=500,
        num_features=8,
        alpha=0.01,
        gamma=0.99,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("\nEvaluating trained agent...")
    eval_rewards = evaluate_agent(agent, env, num_episodes=100)
    
    print("\nPlotting results...")
    plot_training_progress(training_rewards, "linear_mc_training.png")
    
    print("\nTraining complete!")
