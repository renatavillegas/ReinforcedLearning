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
from monte_carlo_linear_approximator import train_with_linear_approximator, evaluate_agent
from sarsa_linear_approximator import train_sarsa_linear, evaluate_sarsa_agent
from q_learning_linear_approximator import train_q_learning_linear, evaluate_q_learning_agent

# Hyperparameters
window_size = 2
# Default episode counts (can be overridden per-environment)
num_train_episodes = 360  # 10 years of training experience
num_eval_episodes = 36    # 1 year of evaluation

# Per-environment episode counts
stochastic_train_episodes = num_train_episodes
stochastic_eval_episodes = num_eval_episodes
deterministic_train_episodes = num_train_episodes
deterministic_eval_episodes = num_eval_episodes

# Monte Carlo hyperparameters (tabular)
gamma_mc = 0.99
epsilon_mc = 0.999
N0_mc = 10

# Q-Learning hyperparameters (tabular)
gamma_ql = 0.99
alpha_ql = 0.1
epsilon_ql = 0.999
epsilon_decay_ql = 0.005
epsilon_min_ql = 0.2

# SARSA hyperparameters (tabular)
gamma_sarsa = 0.8
alpha_sarsa = 0.3
epsilon_sarsa = 0.999

# Linear function-approximator defaults
linear_num_features = 8
linear_alpha = 0.01
linear_epsilon = 0.1
linear_epsilon_decay = 0.995
linear_epsilon_min = 0.01

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

def plot_rewards(rewards, filename="MonteCarlo_rewards.png", title="Training Progress"):
    """
    Generic plotting function for training progress.
    Works with all algorithms (Q-table and approximators).
    
    Args:
        rewards: List of rewards per episode
        filename: Output filename
        title: Title for the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label="Episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{title} - Episode Rewards")
    plt.grid(True)
    
    # Moving average
    plt.subplot(1, 2, 2)
    window = 5
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg, label=f"Moving average (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"{title} - Moving Average")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

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
    Q_mc = monte_carlo_train(env, stochastic_train_episodes, gamma_mc, epsilon_mc, N0_mc)
    V_mc = get_value_function(Q_mc)
    rewards_mc = evaluate(env, Q_mc, num_eval_episodes)
    plot_rewards(rewards_mc, filename="MonteCarlo_rewards_stochastic.png", title="Stochastic Monte Carlo (Q-table)")
    plot_values(V_mc, filename="MonteCarloValueFunction_stochastic.png")
    print("Stochastic Monte Carlo Training Done")
    
    print("Stochastic Q-Learning Training...")
    Q_ql, rewards_ql = q_learning_train(
        env,
        stochastic_train_episodes,
        gamma=gamma_ql,
        alpha=alpha_ql,
        epsilon=epsilon_ql,
        epsilon_decay=epsilon_decay_ql,
        epsilon_min=epsilon_min_ql,
    )
    V_ql = get_value_function(Q_ql)
    rewards_ql = evaluate(env, Q_ql, num_eval_episodes)
    plot_rewards(rewards_ql, filename="QLearning_rewards_stochastic.png", title="Stochastic Q-Learning (Q-table)")
    plot_values(V_ql, filename="QLearning_ValueFunction_stochastic.png")
    print("Stochastic Q-Learning Training Done")
    
    print("Stochastic SARSA Training...")
    Q_sarsa, rewards_sarsa = sarsa_train(
        env,
        stochastic_train_episodes,
        alpha=alpha_sarsa,
        gamma=gamma_sarsa,
        epsilon=epsilon_sarsa,
    )
    V_sarsa = get_value_function(Q_sarsa)
    rewards_sarsa_eval = evaluate(env, Q_sarsa, num_eval_episodes)
    plot_rewards(rewards_sarsa_eval, filename="SARSA_rewards_stochastic.png", title="Stochastic SARSA (Q-table)")
    plot_values(V_sarsa, filename="SARSA_ValueFunction_stochastic.png")
    print("Stochastic SARSA Training Done")
        
    # Deterministic environment
    det_env = DeterministicTradingEnv(n_steps=hours_per_day, start_price=start_price, window_size=window_size)
    print("Deterministic Monte Carlo Training...")
    det_Q = monte_carlo_train(det_env, deterministic_train_episodes, gamma_mc, epsilon_mc, N0_mc)
    det_V = get_value_function(det_Q)
    det_rewards = evaluate(det_env, det_Q, num_eval_episodes)
    plot_rewards(det_rewards, filename="MonteCarlo_rewards_deterministic.png", title="Deterministic Monte Carlo (Q-table)")
    plot_values(det_V, filename="MonteCarloValueFunction_deterministic.png")
    print("Deterministic Monte Carlo Training Done")
    
    print("Deterministic Q-Learning Training...")
    det_Q_ql, det_rewards_ql = q_learning_train(
        det_env,
        deterministic_train_episodes,
        gamma=gamma_ql,
        alpha=alpha_ql,
        epsilon=epsilon_ql,
        epsilon_decay=epsilon_decay_ql,
        epsilon_min=epsilon_min_ql,
    )
    det_V_ql = get_value_function(det_Q_ql)
    det_rewards_ql_eval = evaluate(det_env, det_Q_ql, num_eval_episodes)
    plot_rewards(det_rewards_ql_eval, filename="QLearning_rewards_deterministic.png", title="Deterministic Q-Learning (Q-table)")
    plot_values(det_V_ql, filename="QLearning_ValueFunction_deterministic.png")
    print("Deterministic Q-Learning Training Done")
    
    print("Deterministic SARSA Training...")
    det_Q_sarsa, det_rewards_sarsa = sarsa_train(
        det_env,
        deterministic_train_episodes,
        alpha=alpha_sarsa,
        gamma=gamma_sarsa,
        epsilon=epsilon_sarsa,
    )
    det_V_sarsa = get_value_function(det_Q_sarsa)
    det_rewards_sarsa_eval = evaluate(det_env, det_Q_sarsa, num_eval_episodes)
    plot_rewards(det_rewards_sarsa_eval, filename="SARSA_rewards_deterministic.png", title="Deterministic SARSA (Q-table)")
    plot_values(det_V_sarsa, filename="SARSA_ValueFunction_deterministic.png")
    print("Deterministic SARSA Training Done")
    
    # Monte Carlo with Linear Function Approximator
    print("Stochastic Monte Carlo with Linear Approximator Training...")
    agent_mc_lin, rewards_mc_lin = train_with_linear_approximator(
        env,
        num_episodes=stochastic_train_episodes,
        num_features=linear_num_features,
        alpha=linear_alpha,
        gamma=gamma_mc,
        epsilon=linear_epsilon,
        epsilon_decay=linear_epsilon_decay,
        epsilon_min=linear_epsilon_min,
        scale=start_price,
    )
    rewards_mc_lin_eval = evaluate_agent(agent_mc_lin, env, stochastic_eval_episodes)
    plot_rewards(rewards_mc_lin, filename="MC_LinearApproximator_training_stochastic.png", title="Stochastic MC with Linear Approximator")
    print("Stochastic Monte Carlo with Linear Approximator Training Done")
    
    print("Deterministic Monte Carlo with Linear Approximator Training...")
    agent_mc_lin_det, rewards_mc_lin_det = train_with_linear_approximator(
        det_env,
        num_episodes=deterministic_train_episodes,
        num_features=linear_num_features,
        alpha=linear_alpha,
        gamma=gamma_mc,
        epsilon=linear_epsilon,
        epsilon_decay=linear_epsilon_decay,
        epsilon_min=linear_epsilon_min,
        scale=start_price,
    )
    rewards_mc_lin_det_eval = evaluate_agent(agent_mc_lin_det, det_env, deterministic_eval_episodes)
    plot_rewards(rewards_mc_lin_det, filename="MC_LinearApproximator_training_deterministic.png", title="Deterministic MC with Linear Approximator")
    print("Deterministic Monte Carlo with Linear Approximator Training Done")
    
    # SARSA with Linear Function Approximator
    print("Stochastic SARSA with Linear Approximator Training...")
    agent_sarsa_lin, rewards_sarsa_lin = train_sarsa_linear(
        env,
        num_episodes=stochastic_train_episodes,
        num_features=linear_num_features,
        alpha=linear_alpha,
        gamma=gamma_sarsa,
        epsilon=linear_epsilon,
        epsilon_decay=linear_epsilon_decay,
        epsilon_min=linear_epsilon_min,
        scale=start_price,
    )
    rewards_sarsa_lin_eval = evaluate_sarsa_agent(agent_sarsa_lin, env, stochastic_eval_episodes)
    plot_rewards(rewards_sarsa_lin, filename="SARSA_LinearApproximator_training_stochastic.png", title="Stochastic SARSA with Linear Approximator")
    print("Stochastic SARSA with Linear Approximator Training Done")
    
    print("Deterministic SARSA with Linear Approximator Training...")
    agent_sarsa_lin_det, rewards_sarsa_lin_det = train_sarsa_linear(
        det_env,
        num_episodes=deterministic_train_episodes,
        num_features=linear_num_features,
        alpha=linear_alpha,
        gamma=gamma_sarsa,
        epsilon=linear_epsilon,
        epsilon_decay=linear_epsilon_decay,
        epsilon_min=linear_epsilon_min,
        scale=start_price,
    )
    rewards_sarsa_lin_det_eval = evaluate_sarsa_agent(agent_sarsa_lin_det, det_env, deterministic_eval_episodes)
    plot_rewards(rewards_sarsa_lin_det, filename="SARSA_LinearApproximator_training_deterministic.png", title="Deterministic SARSA with Linear Approximator")
    print("Deterministic SARSA with Linear Approximator Training Done")
    
    # Q-Learning with Linear Function Approximator
    print("Stochastic Q-Learning with Linear Approximator Training...")
    agent_ql_lin, rewards_ql_lin = train_q_learning_linear(
        env,
        num_episodes=stochastic_train_episodes,
        num_features=linear_num_features,
        alpha=linear_alpha,
        gamma=gamma_ql,
        epsilon=linear_epsilon,
        epsilon_decay=linear_epsilon_decay,
        epsilon_min=linear_epsilon_min,
        scale=start_price,
    )
    rewards_ql_lin_eval = evaluate_q_learning_agent(agent_ql_lin, env, stochastic_eval_episodes)
    plot_rewards(rewards_ql_lin, filename="QLearning_LinearApproximator_training_stochastic.png", title="Stochastic Q-Learning with Linear Approximator")
    print("Stochastic Q-Learning with Linear Approximator Training Done")
    
    print("Deterministic Q-Learning with Linear Approximator Training...")
    agent_ql_lin_det, rewards_ql_lin_det = train_q_learning_linear(
        det_env,
        num_episodes=deterministic_train_episodes,
        num_features=linear_num_features,
        alpha=linear_alpha,
        gamma=gamma_ql,
        epsilon=linear_epsilon,
        epsilon_decay=linear_epsilon_decay,
        epsilon_min=linear_epsilon_min,
        scale=start_price,
    )
    rewards_ql_lin_det_eval = evaluate_q_learning_agent(agent_ql_lin_det, det_env, deterministic_eval_episodes)
    plot_rewards(rewards_ql_lin_det, filename="QLearning_LinearApproximator_training_deterministic.png", title="Deterministic Q-Learning with Linear Approximator")
    print("Deterministic Q-Learning with Linear Approximator Training Done")
