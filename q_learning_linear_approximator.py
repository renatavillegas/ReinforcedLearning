"""
Q-Learning with Linear Function Approximator

This module demonstrates how to implement Q-Learning reinforcement learning
using a linear function approximator instead of a Q-table.

The Q-function is approximated as: Q(s,a) = w_a^T * phi(s)
where phi(s) is a feature vector and w_a are the weights for action a.

Q-Learning is an off-policy algorithm that learns the optimal Q-values
even when following an exploratory policy.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from linear_approximator import LinearFunctionApproximator
class QLearningLinearAgent:
    """
    Q-Learning agent with linear function approximator.
    Uses off-policy learning with maximum next action.
    """
    
    def __init__(self, action_space_n, num_features=8, alpha=0.01, gamma=0.99, epsilon=0.1):
        """
        Initialize the Q-Learning agent.
        
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
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using Q-Learning update rule.
        
        Q-Learning: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        
        Off-policy: learns optimal Q-values regardless of the policy followed.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        q_current = self.approximators[action].predict(state)
        
        if done:
            td_target = reward
        else:
            # Off-policy: use maximum Q-value of next state (not the actual next action taken)
            q_next_values = self.get_q_values(next_state)
            max_q_next = np.max(q_next_values)
            td_target = reward + self.gamma * max_q_next
        
        # Update the approximator for the taken action
        self.approximators[action].update(state, td_target)


def train_q_learning_linear(env, num_episodes=1000, num_features=8, 
                            alpha=0.01, gamma=0.99, epsilon=0.1,
                            epsilon_decay=0.995, epsilon_min=0.01):
    """
    Train an agent using Q-Learning with linear function approximation.
    
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
    agent = QLearningLinearAgent(
        env.action_space.n,
        num_features=num_features,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action using epsilon-greedy policy
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            # Update Q-values (off-policy)
            agent.update(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
        
        rewards.append(episode_reward)
        
        # Decay epsilon
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last 100): {avg_reward:.2f}")
    
    return agent, rewards


def evaluate_q_learning_agent(agent, env, num_episodes=100):
    """
    Evaluate a trained Q-Learning agent.
    
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
    
    print("\nTraining Q-Learning agent with linear function approximator...")
    agent, training_rewards = train_q_learning_linear(
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
    eval_rewards = evaluate_q_learning_agent(agent, env, num_episodes=100)
    
    print("\nPlotting results...")
    plot_q_learning_training(training_rewards, "q_learning_linear_training.png")
    
    print("\nTraining complete!")
