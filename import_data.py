import numpy as np
import matplotlib.pyplot as plt

mean_return = -0.01  # average hourly return (negative = downward trend)
volatility = 0.03     # standard deviation of returns (3% volatility)

def generate_intraday_prices(num_days=100, hours_per_day=10, start_price=10):
    prices = []
    for _ in range(num_days):
        returns = np.random.normal(loc=mean_return, scale=volatility, size=hours_per_day)
        day_prices = start_price * np.exp(np.cumsum(returns))
        prices.append(day_prices)
    plot_prices(np.array(prices), num_days)
    return np.array(prices)


def plot_prices(prices, num_days):
    # ----- Cálculo da variação diária -----
    # Variação absoluta (fechamento - abertura) e percentual
    open_prices = prices[:, 0]
    close_prices = prices[:, -1]
    daily_abs_change = close_prices - open_prices
    daily_pct_change = (close_prices / open_prices - 1) * 100

    # ----- Figura: Variação do preço por dia (percentual) -----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(1, num_days+1), daily_pct_change, color=['#2ca02c' if x>=0 else '#d62728' for x in daily_pct_change], width=0.8)

    ax.set_title('Variação percentual diária do preço (fechamento vs. abertura)')
    ax.set_xlabel('Dia')
    ax.set_ylabel('Variação (%)')

    # Linha na horizontal em 0%
    ax.axhline(0, color='black', linewidth=1)

    # Melhores rótulos no eixo x (marcar a cada 5 dias)
    ax.set_xticks(np.arange(1, num_days+1, 5))

    # Texto com estatísticas básicas
    mean_change = np.mean(daily_pct_change)
    std_change = np.std(daily_pct_change)
    ax.text(0.99, 0.02, f'Média: {mean_change:.2f}%\nDesvio-padrão: {std_change:.2f}%', transform=ax.transAxes,
            ha='right', va='bottom', bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))

    plt.tight_layout()
    output_file = 'variacao_diaria_percentual.png'
    plt.savefig(output_file, dpi=220)

    # Também salvar a série de variação absoluta (opcional, segunda figura)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(np.arange(1, num_days+1), daily_abs_change, marker='o', linewidth=1.5, color='#1f77b4')
    ax2.set_title('Variação absoluta diária do preço (fechamento - abertura)')
    ax2.set_xlabel('Dia')
    ax2.set_ylabel('Δ Preço')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xticks(np.arange(1, num_days+1, 5))
    plt.tight_layout()
    output_file2 = 'variacao_diaria_absoluta.png'
    plt.savefig(output_file2, dpi=220)
