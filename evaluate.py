import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from trading_env import TradingEnv
from trading_deterministic_env import DeterministicTradingEnv
from import_data import generate_intraday_prices
from monte_carlo import monte_carlo_train, discretize_state
from q_learning import q_learning_train, discretize_state

# Hiperparams 
window_size = 2
num_train_episodes = 360*10 #10 years of experience
num_eval_episodes = 360*1 #1 yaar
gamma = 0.99
alpha = 0.05
N0=10
epsilon = 1
hours_per_day = 10
start_price = 10.22
epsilon_decay = 0.0001

def evaluate(env, Q, num_episodes=100):
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            s = discretize_state(state)
            action = np.argmax(Q[s]) if s in Q else env.action_space.sample()
            next_state, reward, terminated, _, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    return total_rewards

def plot_rewards(rewards, filename="MonteCarlo_rewards.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward per day")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Policy Evaluation Monte Carlo")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)

def plot_values(V,filename="MonteCarloValueFunction.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(V))), list(V.values()))
    plt.title("Optimal Value Function V*(s)")
    plt.xlabel("State index")
    plt.ylabel("Value")
    plt.savefig(filename)

def get_value_function(Q):
    V = {}
    for s in Q:
        V[s] = np.max(Q[s])
    return V

if __name__ == "__main__":
    #Stocrastic environment 
    prices = generate_intraday_prices(num_eval_episodes, hours_per_day, start_price)
    env = TradingEnv(data=prices, window_size=window_size)
    print("Stocrastic Monte Carlo Training...")
    Q_mc = monte_carlo_train(env, num_train_episodes, gamma, epsilon, N0)
    V_mc = get_value_function(Q_mc)
    rewards_mc = evaluate(env, Q_mc, num_eval_episodes)
    plot_rewards(rewards_mc)
    plot_values(V_mc)
    print("Stocrastic Monte Carlo Training Done")
    print("Stocrastic Q-Learning Training...")
    Q_ql, rewards_ql = q_learning_train(env, num_train_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=0.2)
    V_ql = get_value_function(Q_ql)
    rewards_ql = evaluate(env, Q_ql, num_eval_episodes)
    plot_rewards(rewards_ql, filename="QLearning_rewards_stochastic.png")
    plot_values(V_ql, filename="QLearning_ValueFunction.png")
        
    #deterministic environment
    det_env = DeterministicTradingEnv(n_steps=hours_per_day, start_price=start_price, window_size=window_size)
    print("Deterministic Monte Carlo Training...")
    det_Q = monte_carlo_train(det_env, num_train_episodes, gamma, epsilon, N0)
    det_V = get_value_function(det_Q)
    det_rewards = evaluate(det_env, det_Q, num_eval_episodes)
    plot_rewards(det_rewards, filename="MonteCarlo_rewards_deterministic.png")
    plot_values(det_V, filename="MonteCarloValueFunction_deterministic.png")
    print("Deterministic Monte Carlo Training Done")
