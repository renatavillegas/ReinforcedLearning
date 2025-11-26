#Simple example of a grid labirint
import numpy as np
#define labirint
rows, cols = 4, 5
states = [(r, c) for r in range(rows) for c in range(cols)]
actions = ['up', 'down', 'left', 'right']
gamma = 0.9 # if gamma=1, you can't garantee convergence! 
goals = {(0,0), (3,1)}
threshold = 1e-5
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
    return reward(s_prime) + gamma * Q[s_prime]


#initialize a policy (ramdom)
policy = {s: np.random.choice(actions) for s in states}
#print(policy)
for goal in goals: policy[goal] = 'GOAL'

#initialize Q values as 0
Q = {s: {a: 0 for a in actions} for s in states}
for s in goals:
    for a in actions:
        Q[s][a] = 0
#Q = {
#    (0, 0): {'up': 0, 'down': 0},
#    (0, 1): {'up': 0, 'down': 0}
#}
#add a iteration stop condition
max_iterations =100
iteration =0
while iteration < max_iterations: #Policy iteration
    iteration+=1
    # Policy Evaluation
    while True:
        new_Q = {s: {a: 0 for a in actions} for s in states} #Q-table to update action-value 
        delta = 0
        for s in states:
            if s in goals: #don't pass through terminal states
                continue
            for a in actions: #check all possible actions
                s_next = next_state(s, a)
                if s_next in goals: #if it's in terminal state, there is future value
                    q_next = 0
                else:
                    qnext = Q[s_next][policy[s_next]] #next q following the policy
                new_Q[s][a] = reward(s_next) + gamma * qnext #Belman Q(s, a)
                delta = max(delta, abs(Q[s][a] - new_Q[s][a]))
        Q = new_Q #update Q-table
        if delta < threshold: #check if converged
            break
#here the policy has converged, 
    policy_stable = True
    for s in states:
        if s in goals:
            continue
        old_action = policy[s]
        best_action = max(actions, key=lambda a: Q[s][a]) #get best action based on Q
        policy[s] = best_action 
        if old_action != best_action:
            policy_stable = False
    if policy_stable: # Check if the policy didn't change
        break
print("best policy:")
for r in range(rows):
    print([policy[(r, c)] for c in range(cols)])

print("\nBest values:")
for r in range(rows):
    print([round(max(Q[(r, c)].values()), 2) if (r, c) not in goals else 0 for c in range(cols)])
print("total iterations=", iteration)