import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# parâmetros
gamma = 0.9
alpha = 0.7
lmbda = 0.6
num_episodes = 10

# sequência de estados e recompensas de um episódio
states = ['S0', 'S1', 'S2', 'S3']   # S3 é terminal
rewards = [1, 0, 2]                 # recompensas entre estados

# valores iniciais
V_td0 = {s: 0.0 for s in states}
V_lambda = {s: 0.0 for s in states}       # forward TD(λ)
V_mc = {s: 0.0 for s in states}
V_backward = {s: 0.0 for s in states}     # backward TD(λ)
V_exact = {s: 0.0 for s in states}        # exact online TD(λ)

# armazenar histórico para plotar
history_td0 = {s: [] for s in states}
history_lambda = {s: [] for s in states}
history_mc = {s: [] for s in states}
history_backward = {s: [] for s in states}
history_exact = {s: [] for s in states}


# --- Função auxiliar: n-step return ---
def n_step_return(t, n, V):
    G = 0
    for k in range(n):
        if t + k < len(rewards):
            G += (gamma ** k) * rewards[t + k]
    if t + n < len(states):
        G += (gamma ** n) * V[states[t + n]]
    return G


# --- Forward view TD(λ) ---
def lambda_return(t, V, lmbda):
    T = len(rewards)
    G_lambda = 0
    total_weight = 0
    for n in range(1, T - t + 1):
        weight = (1 - lmbda) * (lmbda ** (n - 1))
        G_lambda += weight * n_step_return(t, n, V)
        total_weight += weight
    final_weight = (lmbda ** (T - t))
    G_lambda += final_weight * n_step_return(t, T - t, V)
    total_weight += final_weight
    return G_lambda / total_weight


# --- TD(0) (online) ---
def td0_update(V):
    for t in range(len(rewards)):
        s, s_next, r = states[t], states[t + 1], rewards[t]
        V[s] += alpha * (r + gamma * V[s_next] - V[s])
    return V


# --- Monte Carlo (episódico) ---
def monte_carlo_update(V):
    T = len(rewards)
    for t in range(T):
        G = 0
        discount = 1
        for k in range(t, T):
            G += discount * rewards[k]
            discount *= gamma
        s = states[t]
        V[s] += alpha * (G - V[s])
    return V


# --- Backward TD(λ) ---
def backward_td_lambda_update(V):
    E = {s: 0.0 for s in states}  # eligibility traces
    for t in range(len(rewards)):
        s, s_next, r = states[t], states[t + 1], rewards[t]
        delta = r + gamma * V[s_next] - V[s]
        E[s] += 1  # incrementar o traço do estado atual
        for s2 in states:
            V[s2] += alpha * delta * E[s2]
            E[s2] *= gamma * lmbda  # decaimento dos traços
    return V


# --- Exact Online TD(λ) (Van Seijen & Sutton, 2014) ---
def exact_online_td_lambda_update(V):
    E = {s: 0.0 for s in states}
    V_old = V.copy()
    for t in range(len(rewards)):
        s, s_next, r = states[t], states[t + 1], rewards[t]
        delta = r + gamma * V[s_next] - V[s]

        # atualização dos traços de elegibilidade
        for s2 in states:
            if s2 == s:
                E[s2] = gamma * lmbda * E[s2] + 1
            else:
                E[s2] = gamma * lmbda * E[s2]

        # aplicar a correção de Van Seijen & Sutton - aqui ele tá corrigindo todos os estados!
        for s2 in states:
            correction = V[s2] - V_old[s2]
            V[s2] += alpha * (delta + correction) * E[s2] - alpha * correction

        V_old = V.copy()
    return V


# --- Rodar múltiplos episódios ---
for ep in range(num_episodes):
    # TD(0)
    V_td0 = td0_update(V_td0)

    # Forward TD(λ)
    for t in range(len(rewards)):
        G_l = lambda_return(t, V_lambda, lmbda)
        V_lambda[states[t]] += alpha * (G_l - V_lambda[states[t]])

    # Monte Carlo
    V_mc = monte_carlo_update(V_mc)

    # Backward TD(λ)
    V_backward = backward_td_lambda_update(V_backward)

    # Exact Online TD(λ)
    V_exact = exact_online_td_lambda_update(V_exact)

    # salvar histórico
    for s in states:
        history_td0[s].append(V_td0[s])
        history_lambda[s].append(V_lambda[s])
        history_mc[s].append(V_mc[s])
        history_backward[s].append(V_backward[s])
        history_exact[s].append(V_exact[s])


# --- Exibir resultados finais ---
print("\n=== Valores finais ===")
for s in states:
    print(f"{s}: TD(0)={V_td0[s]:.4f} | Forward TD(λ)={V_lambda[s]:.4f} | "
          f"Backward TD(λ)={V_backward[s]:.4f} | Exact Online TD(λ)={V_exact[s]:.4f} | MC={V_mc[s]:.4f}")

# --- Gráfico ---
plt.figure(figsize=(21, 15))
for s in ['S0', 'S1', 'S2']:
    plt.plot(history_td0[s], '--', label=f'TD(0) {s}')
    plt.plot(history_lambda[s], '-', label=f'Forward TD(λ) {s}')
    plt.plot(history_backward[s], '-.', label=f'Backward TD(λ) {s}')
    plt.plot(history_exact[s], ':', label=f'Exact TD(λ) {s}')
    plt.plot(history_mc[s], '.', label=f'MC {s}')
plt.xlabel("Episódios")
plt.ylabel("Valor estimado V(s)")
plt.title(f"Comparação TD(0), TD(λ) (forward/backward/exato) e Monte Carlo — Convergência dos valores")
plt.legend()
plt.grid(True)
plt.savefig("grafico_td_lambda_completo.png")
