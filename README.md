Problem:
Operate in the stock market – buy or sell assets based on historical data
Adapted from https://gym-trading-env.readthedocs.io/

MDP Formulation:
- States: Current holdings and prices. To make this problem Marcovian, we need to set a window size to keep some prices to help in the inference(buy/sell).
- Actions: Buy, Sell, Hold.
- Transitions: Price changes
- Rewards: The reward is defined by profit/loss of each action
- Discount(gamma): long-term gains. 

Simplfications in the model: 
- The prices are updated hourly and are updated 10 times a day. 
- The prices start with the same value each day.
- The prices std_dev is constant. 
- Here we sell/buy all stocks in portifolium, which means it is the same as considering only one stock.

Environment characterists: 
Episodic - As each day is treated as a complete episode. 
Terminal States - There are not terminal states, but we could set one, for example, 
choose to stop at a maximum profit or maximum lose. 
The states are continuous, as they are defined as the price. 
The environment is stocrastic as the price vary ramdonly with a normal distribution. 
This envrionment is partially observable, as we have information of only a window size of prices. 

Environment Parameters: 
window_size = 3 : number of hours to keep in history used by the agent to take the decision
num_train_episodes = 2500 : 2500 days of training
num_eval_episodes = 61 : Days of evaluation
gamma = 0.9 : 
epsilon = 1 : Initial epsilon value
N0 - epsilon greedy update parameter
num_days = 61 : Number of days to create data
hours_per_day = 10 : operational hours per day
start_price = 10.22 : Initial stok value

## Stochastic Trading Environment

The `TradingEnv` generates realistic stochastic price movements from historical intraday data using random walks.

### Price Dynamics:
- Prices are generated with a **normal distribution** with configurable mean return and volatility
- Data format: 2D array (num_days, hours_per_day) of price values
- Each day represents one episode

### Key Characteristics:
- **Stochastic**: Prices vary randomly with normal distribution
- **Continuous State**: Window of historical prices + current position
- **Discrete Actions**: Hold (0), Buy (1), Sell (2)
- **Reward**: Portfolio value change (cash + shares*price - initial_cash)
- **Episode Length**: Fixed steps per day (hours_per_day)
- **Partially Observable**: Agent sees only recent price history (window_size)

### Environment Parameters:
- `window_size = 3`: Historical prices retained in state
- `hours_per_day = 10`: Steps per episode
- `num_days = 360`: Days of generated data
- `start_price = 10.22`: Initial asset price
- `initial_cash = 100`: Starting capital

### Position Tracking:
- `position = 0`: No position (holding cash)
- `position = 1`: Long position (holding shares)
- Position prevents agent from doubling positions (can't buy if already in long)

### Data Generation:
The intraday prices are generated using `generate_intraday_prices()` function with:
- **mean return**: μ (typically 0.001)
- **volatility**: σ (typically 0.01)
- Random walk: `price[t] = price[t-1] * exp(μ + σ * Z)` where Z ~ N(0,1)

## Deterministic Trading Environment

The `DeterministicTradingEnv` uses the **Rulkov Map** (a chaotic dynamical system) to generate price movements deterministically.
Refs:

### Rulkov Map Dynamics:
- **x_next = f(x, y + β, α)**: Price evolution function
- **y_next = y - μ(x_next + 1) + μσ**: Coupling variable evolution
- Parameters: α=4.0, β=10.0, σ=0.01, μ=0.001

### Key Characteristics:
- **Deterministic**: Prices follow a fixed chaotic pattern (no randomness)
- **Continuous State**: Window of prices + cash + asset holdings
- **Discrete Actions**: Hold (0), Buy (1), Sell (2)
- **Reward**: Portfolio value change (cash + asset*price - initial_cash)
- **Episode Length**: Fixed number of steps (n_steps)

### Environment Parameters:
- `n_steps = 10`: Steps per episode (e.g., trading hours)
- `start_price = 10.22`: Initial asset price
- `alpha, beta, sigma, mu`: Rulkov Map parameters for price dynamics
- `window_size = 3`: Historical prices retained in state
- `initial_cash = 100`: Starting capital

Discussions:
What happen if we have fewer train episodes? 
What happen if we change N0 on Monte Carlo? 
What happen if we change the mean return?
What happen if we change the volaty?
What happen with the Rulkov Map parameters on agent performance?
How does deterministic vs stochastic environment affect learning speed?
