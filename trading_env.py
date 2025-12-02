import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    """
    Stochastic trading environment using historical intraday price data.
    
    The agent can take three actions: Hold (0), Buy (1), Sell (2).
    Actions are constrained by current position to prevent multiple simultaneous positions.
    """
    
    def __init__(self, data, window_size=3):
        """
        Initialize the trading environment.
        
        Args:
            data: 2D array of prices with shape (num_days, hours_per_day)
            window_size: Number of historical prices to include in observation
        """
        super(TradingEnv, self).__init__()
        self.data = data  # Data format: (days, hours)
        self.num_days, self.hours_per_day = data.shape
        self.window_size = window_size
        self.current_day = 0
        self.current_step = window_size
        self.initial_cash = 100
        self.cash = self.initial_cash
        self.shares = 0
        self.position = 0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size + 1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell

    def reset(self, seed=None, options=None):
        """Reset the environment to a random day."""
        self.current_day = np.random.randint(0, self.num_days)  # Choose a random day
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares = 0
        self.position = 0
        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 = Hold, 1 = Buy, 2 = Sell
            
        Returns:
            observation, reward, done, truncated, info
        """
        price = self.data[self.current_day, self.current_step]
        if action == 1 and self.position <= 0:  # Buy
            self.shares = self.cash / price
            self.cash = 0
            self.position = 1
        elif action == 2 and self.position >= 0:  # Sell
            self.cash = self.shares * price
            self.shares = 0
            self.position = -1

        portfolio_value = self.cash + self.shares * price
        reward = portfolio_value - self.initial_cash

        self.current_step += 1
        done = self.current_step >= self.hours_per_day  # End of day
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def _get_obs(self):
        """Get current observation: window of prices + current position."""
        window = self.data[self.current_day, self.current_step - self.window_size:self.current_step]
        return np.append(window, self.position).astype(np.float32)