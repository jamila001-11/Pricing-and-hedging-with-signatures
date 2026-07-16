import numpy as np

from models.payoffs import compute_payoff_polynomial


def bs_price(t, S, coef, sigma, T):
    """Prix Black-Scholes en forme close du payoff polynomial (papier, Sec 5.1).

    V_t = sum_k c_k S_t^k exp( 1/2 k(k-1) sigma^2 (T-t) ).
    """
    tau = T - t
    return sum(
        coef[k] * S**k * np.exp(0.5 * k * (k - 1) * sigma**2 * tau)
        for k in range(len(coef))
    )


def bs_delta(t, S, coef, sigma, T):
    """Delta Black-Scholes du payoff polynomial.

    Delta_t = sum_k c_k k S_t^{k-1} exp( 1/2 k(k-1) sigma^2 (T-t) ).
    Le terme k=0 est identiquement nul.
    """
    tau = T - t
    total = np.zeros_like(np.asarray(S, dtype=float))
    for k in range(1, len(coef)):
        total = total + coef[k] * k * S**(k - 1) * np.exp(0.5 * k * (k - 1) * sigma**2 * tau)
    return total


def bs_pnl(S, coef, sigma, T, S0):
    """P&L de couverture auto-financee (delta discret) pour un lot S : (I, J+1)."""
    t_grid = np.linspace(0, T, S.shape[1])
    dS = np.diff(S, axis=1)                       # (I, J)
    t_j = t_grid[:-1][None, :]                    # (1, J), broadcast sur les trajectoires
    S_j = S[:, :-1]                               # delta detenu sur [t_j, t_{j+1})
    delta_j = bs_delta(t_j, S_j, coef, sigma, T)
    V0 = bs_price(0.0, S0, coef, sigma, T)
    return V0 + np.sum(delta_j * dS, axis=1) - compute_payoff_polynomial(S, coef)


def pnl_stats(pnl):
    """Renvoie (MSE, moyenne) du vecteur de P&L."""
    return float(np.mean(pnl**2)), float(np.mean(pnl))
