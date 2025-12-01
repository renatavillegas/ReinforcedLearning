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

## Deterministic Trading Environment

The `DeterministicTradingEnv` is an alternative environment that uses the **Rulkov Map** (a chaotic dynamical system) to generate price movements deterministically.

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

### Advantages vs Stochastic:
- Reproducible behavior for testing/debugging
- No randomness in price generation
- Useful for understanding agent performance without variance
- Faster training convergence expected

Discussions:
What happen if we have fewer train episodes? 
What happen if we change N0 on Monte Carlo? 
What happen if we change the mean return?
What happen if we change the volaty?
What happen with the Rulkov Map parameters on agent performance?
How does deterministic vs stochastic environment affect learning speed?
