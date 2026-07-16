"""Calibration du tenseur de volatilite sur la signature esperee (papier, Algorithme 2).

On apprend sigma_hat tel que Sigma_t = <sigma_hat, S(W_hat)_{0,t}> en ne voyant que
des MOMENTS : on apparie la signature esperee observee du chemin Y_hat = (t, Sigma)
avec sa prediction theorique, obtenue en propageant sigma_hat dans les tenseurs ell^v
et en la contractant contre E[S(W_hat)_{0,t}] (formule fermee de Fawcett).

C'est l'ingredient qui distingue Fourier REG (sigma calibre sur les moments) de
Fourier BCS (sigma exact, connu en forme close via `signatures.volatility_models`).

Convention du mot vide -- subtile et essentielle. `sigma_words(M)` EXCLUT le mot
vide : le niveau initial sigma[()] = Sigma_0 n'est pas calibre. En effet le mot (2,)
de la signature de Y_hat mesure l'INCREMENT Sigma_t - Sigma_0, et l'on a bien
<sigma prive du mot vide, E[W_hat_t]> = E[Sigma_t] - Sigma_0. Le niveau de base est
donc restaure separement avant l'injection dans la machinerie Fourier.
"""

import numpy as np

from signatures.tensor_algebra import generate_words
from signatures.tensor_dict import (
    append_letter_dict,
    contract_with_Esig,
    exp_tensor_dict,
    half_shuffle_dict,
    words_of_length,
)


def sigma_words(M):
    """Mots portant les coefficients calibres : longueur 1..M sur {1,2} (sans le mot vide)."""
    return generate_words(M, 2)[1:]


# =====================================================================
# Signature esperee du mouvement brownien augmente (formule fermee)
# =====================================================================

def fawcett_E_W_hat(t, M):
    """E[S(W_hat)_{0,t}] pour W_hat = (t, W), en forme close (Fawcett 2003).

    E[S(W_hat)_{0,t}] = exp_(x)( t e_1 + (t/2) e_2 (x) e_2 ), exponentielle pour le
    produit de CONCATENATION (et non le produit shuffle). Le terme (t/2) e_2 e_2 est
    la variation quadratique de W.
    """
    return exp_tensor_dict({(1,): t, (2, 2): t / 2}, M)


def E_W_hat_all_times(t_grid, M):
    """Liste de E[S(W_hat)_{0,t}] pour chaque t de la grille."""
    return [fawcett_E_W_hat(t, M) for t in t_grid]


# =====================================================================
# Algorithme 2 : propagation de sigma_hat et perte de moments
# =====================================================================

def build_ell(sigma_hat, Mbar, M_trunc):
    """Tenseurs ell^v pour tous les mots v de longueur 0..Mbar (Algorithme 2).

    Recurrence des mots courts vers les mots longs :
      - ell^{v.1} = ell^v concatene a droite par e_1   (integration contre dt)
      - ell^{v.2} = ell^v > sigma_hat                  (half-shuffle : integration contre dSigma)
    de sorte que <ell^v, S(W_hat)_{0,t}> = S(Y_hat)^v_{0,t} avec Y_hat = (t, Sigma).

    `M_trunc` tronque les tenseurs ell^v eux-memes ; il doit valoir au moins
    Mbar + (ordre de sigma_hat) pour ne rien perdre.
    """
    L = {(): {(): 1.0}}
    for length in range(Mbar):
        for v in words_of_length(length, 2):
            ell_v = L[v]
            L[v + (1,)] = append_letter_dict(ell_v, 1)
            L[v + (2,)] = half_shuffle_dict(ell_v, sigma_hat, M_trunc)
    return L


def compute_loss(sigma_hat, Mbar, M_trunc, E_W_list, Ybar_stream, w2i_Y):
    """Perte de l'Algorithme 2 : somme sur (instant, mot v) des ecarts predit/observe au carre.

    `Ybar_stream` (J, siglen) est la signature esperee empirique de Y_hat en mode
    stream (un vecteur par instant t_1..t_J), `E_W_list` la liste des E[S(W_hat)_{0,t}]
    aux memes instants, et `w2i_Y` l'index des mots dans les colonnes de Ybar_stream.
    """
    L_tensors = build_ell(sigma_hat, Mbar, M_trunc)
    words_v = [w for m in range(1, Mbar + 1) for w in words_of_length(m, 2)]

    loss = 0.0
    for j, E_Wj in enumerate(E_W_list):
        for v in words_v:
            pred = contract_with_Esig(L_tensors[v], E_Wj)
            loss += (pred - Ybar_stream[j, w2i_Y[v]]) ** 2
    return loss


# =====================================================================
# Parametrage vecteur <-> dict et optimisation
# =====================================================================

def vec_to_sigma_dict(theta, words_sigma):
    """Vecteur optimisable -> dict mot -> coefficient (mots dans l'ordre de `words_sigma`)."""
    return {w: theta[i] for i, w in enumerate(words_sigma)}


def sigma_dict_to_vec(sigma_dict, words_sigma):
    """Dict -> vecteur optimisable. Les mots absents (dont le mot vide) sont ignores."""
    return np.array([sigma_dict.get(w, 0.0) for w in words_sigma])


def make_loss_fn(words_sigma, Mbar, M_trunc, E_W_list, Ybar_stream, w2i_Y):
    """Ferme la perte sur ses donnees : renvoie loss_fn(theta) -> float."""
    def loss_fn(theta):
        sigma_hat = vec_to_sigma_dict(theta, words_sigma)
        return compute_loss(sigma_hat, Mbar, M_trunc, E_W_list, Ybar_stream, w2i_Y)

    return loss_fn


def numerical_gradient(theta, loss_fn, eps=1e-5):
    """Gradient par differences finies avant (la perte n'est pas differentiee analytiquement)."""
    grad = np.zeros_like(theta)
    f0 = loss_fn(theta)
    for i in range(len(theta)):
        theta_eps = theta.copy()
        theta_eps[i] += eps
        grad[i] = (loss_fn(theta_eps) - f0) / eps
    return grad


def calibrate_sigma(
    loss_fn, sigma_init_dict, words_sigma, n_restarts=3, steps=100, lr=1e-2, seed=0, verbose=True
):
    """Minimise `loss_fn` par descente Adam multi-start (Remark 5.0.2).

    Le premier redemarrage part de `sigma_init_dict` (typiquement le sigma exact du
    modele), les suivants d'un tirage aleatoire. Renvoie (theta, loss) du meilleur.
    """
    n = len(words_sigma)
    rng = np.random.default_rng(seed)
    best_theta, best_loss = None, np.inf

    for r in range(n_restarts):
        if r == 0:
            theta = sigma_dict_to_vec(sigma_init_dict, words_sigma)
        else:
            theta = 0.05 * rng.standard_normal(n)

        m, v = np.zeros(n), np.zeros(n)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            grad = numerical_gradient(theta, loss_fn)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad * grad
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        final_loss = loss_fn(theta)
        if verbose:
            print(f"  restart {r}: loss finale = {final_loss:.6e}")
        if final_loss < best_loss:
            best_loss, best_theta = final_loss, theta.copy()

    return best_theta, best_loss
