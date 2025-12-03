import numpy as np
import matplotlib.pyplot as plt

mean_return = -0.001  # Average hourly return (negative = downward trend)
volatility = 0.003    # Standard deviation of returns (3% volatility)

def generate_intraday_prices(num_days=100, hours_per_day=10, start_price=10):
    """
    Generate intraday price data using geometric random walk.
    
    Args:
        num_days: Number of trading days to generate
        hours_per_day: Number of hourly steps per day
        start_price: Initial price for all days
        
    Returns:
        numpy array of shape (num_days, hours_per_day) with price values
    """
    prices = []
    for _ in range(num_days):
        returns = np.random.normal(loc=mean_return, scale=volatility, size=hours_per_day)
        day_prices = start_price * np.exp(np.cumsum(returns))
        prices.append(day_prices)
    plot_prices(np.array(prices), num_days)
    return np.array(prices)


def plot_prices(prices, num_days):
    """
    Plot daily price variations (percentage and absolute).
    
    Args:
        prices: 2D array of shape (num_days, hours_per_day)
        num_days: Number of days for labeling
    """
    # Calculate daily variations
    # Absolute change (close - open) and percentage change
    open_prices = prices[:, 0]
    close_prices = prices[:, -1]
    daily_abs_change = close_prices - open_prices
    daily_pct_change = (close_prices / open_prices - 1) * 100

    # Figure 1: Daily percentage price variation
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(1, num_days+1), daily_pct_change, color=['#2ca02c' if x>=0 else '#d62728' for x in daily_pct_change], width=0.8)

    ax.set_title('Daily Percentage Price Variation (Close vs. Open)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Variation (%)', fontsize=12)

    # Horizontal line at 0%
    ax.axhline(0, color='black', linewidth=1)

    # Better x-axis labels (mark every 5 days)
    ax.set_xticks(np.arange(1, num_days+1, 5))

    # Text with basic statistics
    mean_change = np.mean(daily_pct_change)
    std_change = np.std(daily_pct_change)
    ax.text(0.99, 0.02, f'Mean: {mean_change:.2f}%\nStd Dev: {std_change:.2f}%', transform=ax.transAxes,
            ha='right', va='bottom', bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Figure 2: Daily absolute price variation
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(np.arange(1, num_days+1), daily_abs_change, marker='o', linewidth=1.5, color='#1f77b4')
    ax2.set_title('Daily Absolute Price Variation (Close - Open)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Δ Price', fontsize=12)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xticks(np.arange(1, num_days+1, 5))
    plt.tight_layout()
    plt.show()
