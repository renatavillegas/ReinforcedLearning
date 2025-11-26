#Simple example of a grid labirint
import numpy as np
#define labirint
rows, cols = 4, 4
states = [(r, c) for r in range(rows) for c in range(cols)]
actions = ['up', 'down', 'left', 'right']
gamma = 1.0
goal = (0,0)
#initial V = 0 
V = {s: 0 for s in states}
#threshold to check convergence
threshold = 1e-4
#state transition
def next_state(s, a):
    r, c = s
    if a == 'up' and r > 0: r -= 1
    elif a == 'down' and r < rows - 1: r += 1
    elif a == 'left' and c > 0: c -= 1
    elif a == 'right' and c < cols - 1: c += 1
    return (r, c)
# define reward 
def reward(s):
    return 0 if s == goal else -1

#To get the policy
def expected_value(a):
    s_prime = next_state(s, a)
    return reward(s_prime) + gamma * V[s_prime]

# Value Iteration
while True:
    delta = 0
    new_V = {}
    for s in states:
        if s == goal:
            new_V[s] = 0
        else:
            best_value = max(
                reward(next_state(s, a)) + gamma * V[next_state(s, a)]
                for a in actions
            )
            new_V[s] = best_value
            delta = max(delta, abs(V[s] - best_value))
    V = new_V
    if delta < threshold:
        break

#print values: 
for r in range(rows):
    print([round(V[(r, c)], 2) for c in range(cols)])

#find optimal policy - now that we know v*, go to each state and calculate 
#the max value to get the goal. 
policy = {}
for s in states:
    if s == goal:
        policy[s] = 'GOAL'
    else:
        best_action = max(actions,key=expected_value)
        policy[s] = best_action
#print 
print ("Best policy:")
for r in range(rows):
    print([policy[(r, c)] for c in range(cols)])
