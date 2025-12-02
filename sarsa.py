import gymnasium as gym
import numpy as np
import random
from collections import defaultdict

def discretize_state(state, decimals=1):
    """Discretizes the continuous state by rounding values."""
    return tuple(np.round(state, decimals))

def epsilon_greedy(Q, state, action_space, epsilon):
    """Epsilon-greedy policy."""
    if random.random() < epsilon:
        return action_space.sample()
    return np.argmax(Q[state])

def sarsa_train(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Train an agent using the SARSA (State-Action-Reward-State-Action) algorithm.
    
    Args:
        env: Gymnasium environment
        num_episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration parameter (epsilon-greedy)
    
    Returns:
        Q: Trained Q-table (dictionary)
        rewards: List of accumulated rewards per episode
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        s = discretize_state(state)
        action = epsilon_greedy(Q, s, env.action_space, epsilon)
        
        episode_reward = 0
        done = False
        
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            s_next = discretize_state(next_state)
            next_action = epsilon_greedy(Q, s_next, env.action_space, epsilon)
            
            # SARSA update: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))
            Q[s][action] += alpha * (reward + gamma * Q[s_next][next_action] - Q[s][action])
            
            s = s_next
            action = next_action
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return dict(Q), rewards

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    n_episodes = 1000
    
    # Train the agent
    Q, training_rewards = sarsa_train(env, n_episodes, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    # Evaluate the trained agent
    total_rewards = 0
    for _ in range(100):
        state, _ = env.reset()
        done = False
        while not done:
            s = discretize_state(state)
            action = np.argmax(Q[s])
            state, reward, done, _, _ = env.step(action)
            total_rewards += reward
    
    print("Average reward after training:", total_rewards / 100)