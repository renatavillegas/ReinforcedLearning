import random
import numpy as np
from collections import defaultdict

def discretize_state(state):
    return tuple(np.round(state, 1))

def epsilon_greedy_policy(Q, state, action_space, epsilon):
    if random.random() < epsilon:
        return action_space.sample()
    return np.argmax(Q[state])

def monte_carlo_train(env, num_train_episodes=2000, gamma=0.95, epsilon=0.1, N0=0.1):
    """
    Train an agent using the Monte Carlo algorithm.
    
    Args:
        env: Gymnasium environment
        num_train_episodes: Number of episodes to train
        gamma: Discount factor
        epsilon: Initial exploration parameter
        N0: Parameter for adaptive epsilon-greedy
    
    Returns:
        Q: Trained Q-table (dictionary)
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Initialize Q(S,a) = 0
    variable_epsilon = epsilon
    returns = defaultdict(list)
    N_sa = defaultdict(int)
    N_s = defaultdict(int)
    for episode in range(num_train_episodes):
        state, _ = env.reset()
        episode_data = []
        terminated = False

        # Collect episode data
        while not terminated:
            s = discretize_state(state)
            N_s[s] += 1  # Update number of visits in this state
            variable_epsilon = N0 / (N0 + N_s[s]) 
            action = epsilon_greedy_policy(Q, s, env.action_space, variable_epsilon)
            next_state, reward, terminated, _, _ = env.step(action)
            episode_data.append((s, action, reward))
            state = next_state

        # Update Q-values using collected returns
        G = 0
        visited = set()
        for t in reversed(range(len(episode_data))):
            s, a, r = episode_data[t]
            G = gamma * G + r
            if (s, a) not in visited:
                N_sa[(s,a)] += 1
                returns[(s, a)].append(G)
                alpha = 1 / N_sa[(s,a)]
                Q[s][a] += alpha * (G - Q[s][a])
                visited.add((s, a))
    return dict(Q)