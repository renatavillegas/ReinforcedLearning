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

def get_value_function(Q):
    V = {}
    for s in Q:
        V[s] = np.max(Q[s])
    return V

def q_learning_train(env, num_episodes=2000, gamma=0.95, alpha=0.3, epsilon=0.9, epsilon_decay=0.99, epsilon_min=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        #initialize S
        s = discretize_state(state)
        done = False
        total_reward = 0
        while not done:
            #chose action following the behavior
            action = epsilon_greedy(Q, s, env.action_space, epsilon)
            # Observe R, S after taking this action.
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            s_next = discretize_state(next_state)
            # Choose best action
            best_next_action = np.argmax(Q[s_next])
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * Q[s_next][best_next_action]
            #Calculate td_error
            td_error = td_target - Q[s][action]
            # update Q
            Q[s][action] += alpha * td_error
            s = s_next
        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon - epsilon_decay*epsilon)

    return dict(Q), rewards


