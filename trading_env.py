import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, window_size=3):
        super(TradingEnv, self).__init__()
        self.data = data  # dados no formato (dias, horas)
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
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        self.current_day = np.random.randint(0, self.num_days)  # escolhe um dia aleatório
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares = 0
        self.position = 0
        return self._get_obs(), {}

    def step(self, action):
        price = self.data[self.current_day, self.current_step]
        if action == 1 and self.position <= 0:  # comprar
            self.shares = self.cash / price
            self.cash = 0
            self.position = 1
        elif action == 2 and self.position >= 0:  # vender
            self.cash = self.shares * price
            self.shares = 0
            self.position = -1

        portfolio_value = self.cash + self.shares * price
        reward = portfolio_value - self.initial_cash

        self.current_step += 1
        done = self.current_step >= self.hours_per_day  # fim do dia
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def _get_obs(self):
        window = self.data[self.current_day, self.current_step - self.window_size:self.current_step]
        return np.append(window, self.position).astype(np.float32)