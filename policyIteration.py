#Simple example of a grid labirint
import numpy as np
#define labirint
rows, cols = 5, 5
states = [(r, c) for r in range(rows) for c in range(cols)]
actions = ['up', 'down', 'left', 'right']
gamma = 0.9 # if gamma=1, you can't garantee convergence! 
goals = {(1,1), (3,3)}
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
    return 0 if s in goals else -1
#To improve the policy
def expected_value(a):
    s_prime = next_state(s, a)
    return reward(s_prime) + gamma * V[s_prime]


#initialize a policy (ramdom)
policy = {s: np.random.choice(actions) for s in states}
print(policy)
for goal in goals: policy[goal] = 'GOAL'
#initialize values as 0
V = {s: 0 for s in states}
#add a iteration stop condition
max_iterations =100
iteration =0
while iteration < max_iterations:
    iteration+=1
    # Policy Evaluation
    while True:
        new_V = {}
        delta = 0
        for s in states:
            if s in goals:
                new_V[s] = 0
            else:
                a = policy[s]
                s_prime = next_state(s, a)
                new_V[s] = reward(s) + gamma * V[s_prime]
                delta = max(delta, abs(V[s] - new_V[s]))
        V = new_V
        if delta < threshold:
            break
#here the policy is evaluated, 
    policy_stable = True
    for s in states:
        if s in goals:
            continue
        old_action = policy[s]
        best_action = max(actions,key=expected_value)
        policy[s] = best_action
        if old_action != best_action:
            policy_stable = False
    if policy_stable:
        break
print("best policy:")
for r in range(rows):
    print([policy[(r, c)] for c in range(cols)])

print("\nBest values:")
for r in range(rows):
    print([round(V[(r, c)], 2) for c in range(cols)])
print("total iterations=", iteration)