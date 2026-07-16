"""Benchmark Black-Scholes naif du call asiatique geometrique (papier, Sec. 5.3).

Couvre en supposant une volatilite CONSTANTE sigma_BS = sqrt(V0) -- donc en ignorant
entierement la dynamique a retard. C'est le pendant asiatique de
`black_scholes.compute_naive_bs_lookback_delta_bergomi`, et la reference contre
laquelle se mesurent Linear/Fourier BCS et REG.

Fondement : sous BS a volatilite constante, log G_T = (1/T) int_0^T log S_t dt est
GAUSSIEN, de variance conditionnelle sigma_eff^2 = V0 tau^3 / (3 T^2). Le prix suit
alors une formule de type Black-Scholes a volatilite effective reduite (Kemna & Vorst
1990), implementee par `black_scholes.compute_call_bs_asian`. Le prix initial associe
est compute_call_bs_asian(S=1, moy_S=1, K, t=0, T, V0).
"""

import numpy as np

from fourier.black_scholes import compute_call_bs_asian


def bs_asian_delta(S, moy_S, K, t, T=0.5, V0=0.0625, h=1e-4):
    """Delta du call asiatique geometrique BS, par differences finies centrees sur S.

    `moy_S` (moyenne geometrique deja realisee) est une variable d'etat distincte et
    reste fixe pendant la derivation. Precision O(h^2), coherent avec le choix deja
    fait pour le lookback (`black_scholes.bs_lookback_delta`).
    """
    price_up = compute_call_bs_asian(S + h, moy_S, K, t, T, V0)
    price_dn = compute_call_bs_asian(S - h, moy_S, K, t, T, V0)
    return (price_up - price_dn) / (2 * h)


def compute_naive_bs_asian_delta(S, K, T, J, V0):
    """Deltas de la couverture naive du call asiatique, de forme (I, J).

    Suit la moyenne geometrique glissante bar_S_t, previsible : bar_S[:, j] n'agrege
    que S_0..S_{j-1}, comme le payoff (`models.compute_payoff_asian`) et le delta
    Fourier asiatique.
    """
    I = S.shape[0]
    dt = T / J

    log_cumsum = np.zeros(I)
    bar_S = np.zeros((I, J))
    for j in range(J):
        bar_S[:, j] = np.exp(log_cumsum / T)
        log_cumsum += np.log(S[:, j]) * dt

    deltas = np.zeros((I, J))
    for j in range(J):
        deltas[:, j] = bs_asian_delta(S[:, j], bar_S[:, j], K, j * dt, T, V0)
    return deltas
