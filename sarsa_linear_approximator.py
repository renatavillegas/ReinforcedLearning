"""
SARSA with Linear Function Approximator

This module demonstrates how to implement SARSA (State-Action-Reward-State-Action)
reinforcement learning using a linear function approximator instead of a Q-table.

The Q-function is approximated as: Q(s,a) = w_a^T * phi(s)
where phi(s) is a feature vector and w_a are the weights for action a.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

# Reuse shared approximator
from linear_approximator import LinearFunctionApproximator
class SARSALinearAgent:
    """
    SARSA agent with linear function approximator.
    Uses on-policy learning with actual next action selection.
    """
    
    def __init__(self, action_space_n, num_features=8, alpha=0.01, gamma=0.99, epsilon=0.1):
        """
        Initialize the SARSA agent.
        
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
        
        # One approximator per action
        self.approximators = [
            LinearFunctionApproximator(num_features, alpha)
            for _ in range(action_space_n)
        ]
    
    def get_q_values(self, state):
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
        
        q_values = self.get_q_values(state)
        return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-values using SARSA update rule.
        
        SARSA: Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (already selected)
            done: Whether episode is done
        """
        q_current = self.approximators[action].predict(state)
        
        if done:
            td_target = reward
        else:
            q_next = self.approximators[next_action].predict(next_state)
            td_target = reward + self.gamma * q_next
        
        # Update the approximator for the taken action
        self.approximators[action].update(state, td_target)


def train_sarsa_linear(env, num_episodes=1000, num_features=8, 
                       alpha=0.01, gamma=0.99, epsilon=0.1,
                       epsilon_decay=0.995, epsilon_min=0.01):
    """
    Train an agent using SARSA with linear function approximation.
    
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
    agent = SARSALinearAgent(
        env.action_space.n,
        num_features=num_features,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        action = agent.select_action(state)
        episode_reward = 0
        done = False
        
        while not done:
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            # Select next action (SARSA: on-policy)
            next_action = agent.select_action(next_state)
            
            # Update Q-values
            agent.update(state, action, reward, next_state, next_action, done)
            
            # Move to next state-action
            state = next_state
            action = next_action
        
        rewards.append(episode_reward)
        
        # Decay epsilon
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last 100): {avg_reward:.2f}")
    
    return agent, rewards


def evaluate_sarsa_agent(agent, env, num_episodes=100):
    """
    Evaluate a trained SARSA agent.
    
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
        episode_reward = 0
        done = False
        
        while not done:
            q_values = agent.get_q_values(state)
            action = np.argmax(q_values)
            next_state, reward, done, _, _ = env.step(action)
            
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
    
    agent.epsilon = original_epsilon
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Evaluation - Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    return rewards


if __name__ == "__main__":
    import gymnasium as gym
    
    # Example: FrozenLake environment
    print("Creating environment...")
    env = gym.make("FrozenLake-v1", is_slippery=True)
    
    print("\nTraining SARSA agent with linear function approximator...")
    agent, training_rewards = train_sarsa_linear(
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
    eval_rewards = evaluate_sarsa_agent(agent, env, num_episodes=100)
    
    print("\nPlotting results...")
    plot_sarsa_training(training_rewards, "sarsa_linear_training.png")
    
    print("\nTraining complete!")
