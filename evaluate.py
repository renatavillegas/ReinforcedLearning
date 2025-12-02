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
from sarsa import sarsa_train

# Hyperparameters
window_size = 2
num_train_episodes = 360*10  # 10 years of training experience
num_eval_episodes = 360*1    # 1 year of evaluation

# Monte Carlo hyperparameters
gamma_mc = 0.99
epsilon_mc = 0.999
N0_mc = 10

# Q-Learning hyperparameters
gamma_ql = 0.99
alpha_ql = 0.1
epsilon_ql = 0.999
epsilon_decay_ql = 0.005
epsilon_min_ql = 0.2

# SARSA hyperparameters
gamma_sarsa = 0.8   
alpha_sarsa = 0.3
epsilon_sarsa = 0.999

# Common parameters
hours_per_day = 10
start_price = 10.22

def evaluate(env, Q, num_episodes=100):
    """
    Evaluate a trained agent on the environment.
    
    Args:
        env: Gymnasium environment
        Q: Trained Q-table (dictionary)
        num_episodes: Number of evaluation episodes
        
    Returns:
        List of total rewards per episode
    """
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
    """Plot episode rewards over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Policy Evaluation - Episode Rewards")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)

def plot_values(V, filename="MonteCarloValueFunction.png"):
    """Plot the value function."""
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(V))), list(V.values()))
    plt.title("Optimal Value Function V*(s)")
    plt.xlabel("State index")
    plt.ylabel("Value")
    plt.savefig(filename)

def get_value_function(Q):
    """Extract value function V(s) = max_a Q(s,a) from Q-table."""
    V = {}
    for s in Q:
        V[s] = np.max(Q[s])
    return V

if __name__ == "__main__":
    # Stochastic environment 
    prices = generate_intraday_prices(num_eval_episodes, hours_per_day, start_price)
    env = TradingEnv(data=prices, window_size=window_size)
    
    print("Stochastic Monte Carlo Training...")
    Q_mc = monte_carlo_train(env, num_train_episodes, gamma_mc, epsilon_mc, N0_mc)
    V_mc = get_value_function(Q_mc)
    rewards_mc = evaluate(env, Q_mc, num_eval_episodes)
    plot_rewards(rewards_mc)
    plot_values(V_mc)
    print("Stochastic Monte Carlo Training Done")
    
    print("Stochastic Q-Learning Training...")
    Q_ql, rewards_ql = q_learning_train(env, num_train_episodes, gamma=gamma_ql, alpha=alpha_ql, epsilon=epsilon_ql, epsilon_decay=epsilon_decay_ql, epsilon_min=epsilon_min_ql)
    V_ql = get_value_function(Q_ql)
    rewards_ql = evaluate(env, Q_ql, num_eval_episodes)
    plot_rewards(rewards_ql, filename="QLearning_rewards_stochastic.png")
    plot_values(V_ql, filename="QLearning_ValueFunction.png")
    print("Stochastic Q-Learning Training Done")
    
    print("Stochastic SARSA Training...")
    Q_sarsa, rewards_sarsa = sarsa_train(env, num_train_episodes, alpha=alpha_sarsa, gamma=gamma_sarsa, epsilon=epsilon_sarsa)
    V_sarsa = get_value_function(Q_sarsa)
    rewards_sarsa_eval = evaluate(env, Q_sarsa, num_eval_episodes)
    plot_rewards(rewards_sarsa_eval, filename="SARSA_rewards_stochastic.png")
    plot_values(V_sarsa, filename="SARSA_ValueFunction.png")
    print("Stochastic SARSA Training Done")
        
    # Deterministic environment
    det_env = DeterministicTradingEnv(n_steps=hours_per_day, start_price=start_price, window_size=window_size)
    print("Deterministic Monte Carlo Training...")
    det_Q = monte_carlo_train(det_env, num_train_episodes, gamma_mc, epsilon_mc, N0_mc)
    det_V = get_value_function(det_Q)
    det_rewards = evaluate(det_env, det_Q, num_eval_episodes)
    plot_rewards(det_rewards, filename="MonteCarlo_rewards_deterministic.png")
    plot_values(det_V, filename="MonteCarloValueFunction_deterministic.png")
    print("Deterministic Monte Carlo Training Done")
