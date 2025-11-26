import numpy as np

# Example
states = ['A', 'B', 'C', 'D', 'E']
V = {s: 0.0 for s in states}  # Initialize value function
alpha = 0.1   # Learning rate
gamma = 0.1   # Discount factor

# Simulated experience: (current_state, reward, next_state)
experience = [
    ('B', -1, 'C'),
    ('C', -1, 'D'),
    ('D', -1, 'A'),
    ('A', -1, 'D'),
    ('D', 1, 'E'),  # Terminal state E with reward 1
]

for s, r, s_next in experience:
    V[s] = V[s] + alpha * (r + gamma * V[s_next] - V[s])

print(V)