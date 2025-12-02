import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DeterministicTradingEnv(gym.Env):
    """
    Deterministic trading environment using the Rulkov Map for price generation.
    
    The Rulkov Map is a chaotic dynamical system that generates deterministic,
    reproducible price movements without randomness.
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_steps=10, start_price=10.0,
                 alpha=4.0, beta=10.0, sigma=0.01, mu=0.001,
                 window_size=3):
        """
        Initialize the deterministic trading environment.
        
        Args:
            n_steps: Number of steps per episode (trading hours)
            start_price: Initial asset price
            alpha, beta, sigma, mu: Rulkov Map parameters
            window_size: Number of historical prices to include in observation
        """
        super(DeterministicTradingEnv, self).__init__()
        self.n_steps = n_steps
        self.start_price = start_price
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.mu = mu
        self.window_size = window_size
        self.initial_cash = 100
        self.cash = self.initial_cash

        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # Hold (0), Buy (1), Sell (2)
        # State: last N prices + cash + asset holdings
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.window_size + 2,), dtype=np.float32
        )

        self.reset()

    def f(self, x, y, alpha):
        """Rulkov Map function."""
        if x <= 0:
            return alpha / (1 - x) + y
        elif 0 < x < (alpha + y):
            return alpha + y
        else:
            return -1

    def rulkov_map(self, x, y):
        """Update state using Rulkov Map dynamics."""
        x_next = self.f(x, y + self.beta, self.alpha)
        y_next = y - self.mu * (x_next + 1) + self.mu * self.sigma
        return x_next, y_next

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.x = -1.0
        self.y = -3.5
        self.t = 0
        self.cash = self.initial_cash
        self.asset = 0.0
        self.price = self.start_price
        # Initial price history
        self.price_history = [self.price] * self.window_size
        state = np.array(self.price_history + [self.cash, self.asset], dtype=np.float32)
        return state, {}

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 = Hold, 1 = Buy, 2 = Sell
            
        Returns:
            observation, reward, done, truncated, info
        """
        # Update price via Rulkov Map
        self.x, self.y = self.rulkov_map(self.x, self.y)
        self.price *= np.exp(self.x * 0.001)

        # Update price history
        self.price_history.append(self.price)
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)

        # Execute action
        if action == 1 and self.cash > 0:  # Buy
            self.asset += self.cash / self.price
            self.cash = 0
        elif action == 2 and self.asset > 0:  # Sell
            self.cash += self.asset * self.price
            self.asset = 0

        # Calculate reward: portfolio value change
        portfolio_value = self.cash + self.asset * self.price
        reward = portfolio_value - self.initial_cash

        # Advance time
        self.t += 1
        done = self.t >= self.n_steps

        state = np.array(self.price_history + [self.cash, self.asset], dtype=np.float32)
        return state, reward, done, False, {}

    def render(self, mode="human"):
        """Render the current environment state."""
        print(f"Step {self.t}: Price={self.price:.2f}, Cash={self.cash:.2f}, Asset={self.asset:.2f}")