import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))
alpha = 0.1      # taxa de aprendizado
gamma = 0.99     # fator de desconto
epsilon = 0.1    # exploração

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

n_episodes = 1000

for episode in range(n_episodes):
    state = env.reset()[0]
    action = choose_action(state)

    done = False
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = choose_action(next_state)

        # Atualização SARSA
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

        state = next_state
        action = next_action
total_rewards = 0
for _ in range(100):
    state = env.reset()[0]
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _, _ = env.step(action)
        total_rewards += reward

print("Recompensa média após treinamento:", total_rewards / 100)