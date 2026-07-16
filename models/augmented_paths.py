"""Chemins augmentes servant de support aux signatures (papier, Sec. 5.2-5.3).

La representation signature de la volatilite s'ecrit Sigma_t = <sigma, S(W_hat)_{0,t}>
avec W_hat = (t, W) : il faut donc reconstruire le chemin brownien augmente a partir
du MEME bruit que celui qui a pilote la volatilite simulee. Le chemin Y_hat = (t, Sigma)
sert lui de cible observee a la calibration de l'Algorithme 2.
"""

import numpy as np

from models.simulation import heston_noise


def build_W_hat(Z, T):
    """Chemin brownien augmente W_hat = (t, W), de forme (I, J+1, 2).

    `Z` (I, J) est la matrice d'aleas standards ; W est la somme cumulee des
    increments sqrt(dt) Z, prefixee par W_0 = 0.
    """
    I, J = Z.shape
    dt = T / J
    W = np.cumsum(Z * np.sqrt(dt), axis=1)
    W = np.concatenate([np.zeros((I, 1)), W], axis=1)
    t_grid = np.linspace(0, T, J + 1)
    return np.stack([np.tile(t_grid, (I, 1)), W], axis=-1)


def build_Y_hat(vol_path, T):
    """Chemin de volatilite augmente Y_hat = (t, Sigma), de forme (I, J+1, 2).

    `vol_path` (I, J+1) est la volatilite simulee : sqrt(V) sous Heston, Sigma sous
    le modele a retard.
    """
    I, Jp1 = vol_path.shape
    t_grid = np.linspace(0, T, Jp1)
    return np.stack([np.tile(t_grid, (I, 1)), vol_path], axis=-1)


def heston_variance_brownian(T, J, I, seed):
    """Rejoue Z1, le bruit de la variance des trajectoires de `models.simulate_heston`.

    `simulate_heston` ne renvoie pas Z1, dont la couverture Fourier a besoin pour
    reconstruire W_hat sur les MEMES trajectoires. On le rejoue via `heston_noise`,
    que `simulate_heston` utilise lui aussi : l'alea a une source unique, donc appeler
    ici avec les memes (J, I, seed) que la simulation redonne exactement son Z1.

    `T` n'intervient pas dans le tirage ; il n'est present que pour rendre les appels
    symetriques de ceux de `simulate_heston`.
    """
    Z1, _ = heston_noise(I, J, seed)
    return Z1
