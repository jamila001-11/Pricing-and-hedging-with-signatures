"""Modele de volatilite a retard / delayed equation (papier, Sec. 5.3).

    dSigma_t = kappa (theta - Sigma_t) dt + (eta + alpha Sigma_t + varsigma Z_t) dW_t
    dZ_t     = (Sigma_t - lam Z_t) dt,   Z_0 = 0

Z_t = int_0^t e^{-lam (t-s)} Sigma_s ds est une moyenne exponentiellement ponderee du
passe de Sigma : c'est elle qui cree la memoire du modele. Contrairement au CIR,
Sigma n'est PAS contraint positif (il descend jusqu'a ~ -0.07 avec les parametres du
papier) -- ce qui compte pour tout code qui divise par Sigma_t.

La representation signature exacte de ce Sigma est dans
`signatures.volatility_models.build_sigma_DE`.
"""

import numpy as np


def simulate_DE(Sigma0, Z0, kappa, theta, eta, alpha, varsigma, lam, T, J, I, seed):
    """Schema d'Euler du modele a retard.

    Renvoie (Sigma, Z, xi) : Sigma et Z de forme (I, J+1), xi (I, J) la matrice des
    aleas standards du brownien W. xi est renvoye car la couverture Fourier a besoin
    du chemin W_hat = (t, W) construit a partir du MEME bruit (cf. `models.build_W_hat`).
    """
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((I, J))

    Sigma = np.zeros((I, J + 1))
    Z = np.zeros((I, J + 1))
    Sigma[:, 0] = Sigma0
    Z[:, 0] = Z0
    dt = T / J

    for j in range(J):
        Sigma[:, j + 1] = (
            Sigma[:, j]
            + kappa * (theta - Sigma[:, j]) * dt
            + (eta + alpha * Sigma[:, j] + varsigma * Z[:, j]) * np.sqrt(dt) * xi[:, j]
        )
        Z[:, j + 1] = Z[:, j] + (Sigma[:, j] - lam * Z[:, j]) * dt

    return Sigma, Z, xi


def simulate_S_given_DE(S0, Sigma_path, xi, rho, T, J, I, seed):
    """Prix sous la volatilite a retard, correle au bruit de Sigma par rho.

    `xi` doit etre le bruit renvoye par `simulate_DE` pour `Sigma_path` : la
    correlation rho n'a de sens que contre le VRAI brownien de la volatilite.
    Le `seed` n'ensemence que la composante independante Z2.
    """
    rng = np.random.default_rng(seed)
    Z2 = rng.standard_normal((I, J))
    B_noise = rho * xi + np.sqrt(1 - rho**2) * Z2

    S = np.zeros((I, J + 1))
    S[:, 0] = S0
    dt = T / J
    for j in range(J):
        S[:, j + 1] = S[:, j] * np.exp(
            -0.5 * Sigma_path[:, j] ** 2 * dt + Sigma_path[:, j] * np.sqrt(dt) * B_noise[:, j]
        )
    return S
