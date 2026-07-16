from itertools import product
from math import comb, factorial

import numpy as np

from signatures import generate_words, shuffle, word_to_index

from linear_hedging.lead_lag import build_path_LL, signature_transform


def sig_words(M, d=3):
    """Mots de longueur 1..M (sans le mot vide), alignes sur les colonnes de iisignature.sig.

    `signatures.generate_words` inclut le mot vide en position 0, absent de la
    sortie de iisignature : on l'enleve pour que les indices coincident.
    """
    return generate_words(M, d)[1:]


# =====================================================================
# Linear BCS  (regression empirique du P&L, trajectoire par trajectoire)
# =====================================================================

def hedging_word_indices(M, d=3):
    """Indices des mots de hedging v.(lettre prix-lead) dans la signature d'ordre M.

    Mots valides = {1,...,d-1}^* suivis de la lettre d (= S^lead). La derniere
    lettre etant le prix, le P&L auto-finance est previsible.
    """
    words = sig_words(M, d)
    return [
        i for i, w in enumerate(words)
        if w[-1] == d and all(x in range(1, d) for x in w[:-1])
    ]


def feature_matrix(sig, valid_idx):
    """Matrice de regression [1 | colonnes de hedging] pour Linear BCS."""
    ones = np.ones((sig.shape[0], 1))
    return np.column_stack([ones, sig[:, valid_idx]])


def linear_bcs(sig_train, sig_test, valid_idx, xi_train, xi_test):
    """Linear BCS : moindres carres du P&L empirique sur les features de signature.

    Renvoie (theta, pnl_test). theta[0] est la richesse initiale apprise X0.
    """
    A_train = feature_matrix(sig_train, valid_idx)
    A_test = feature_matrix(sig_test, valid_idx)
    theta, *_ = np.linalg.lstsq(A_train, xi_train, rcond=None)
    pnl = A_test @ theta - xi_test
    return theta, pnl


# =====================================================================
# Linear REG  (eq. 3.8-3.9 : n'utilise que la signature esperee E[S^LL])
# =====================================================================

# `shuffle` est importe de signatures.tensor_dict : representation creuse
# (dict mot -> coefficient), adaptee aux ordres eleves (E[S^LL] est calculee a
# l'ordre 2M, ou la representation dense serait enorme). Reexporte ici pour
# preserver `from linear_hedging import shuffle`.


def xi_coef(m, coef, S0):
    """Coefficient du mot prix-lag (lettre 2)^m dans le payoff linearise (papier p.20)."""
    c = sum(coef[k] * comb(k, m) * S0**(k - m) for k in range(len(coef)))
    return factorial(m) * c


def build_xi(coef, S0, price_letter=2):
    """Payoff polynomial linearise EXACT : xi[(price_letter,)*m] = m! a_m.

    Identite <(price_letter,)*m, S^LL> = (S_T - S0)^m / m! (binome de S_T^k
    autour de S0). Le mot vide (m=0) porte la constante.
    """
    return {(price_letter,) * m: xi_coef(m, coef, S0) for m in range(len(coef))}


def hedging_words(M, lead_letter=3, base_letters=(1, 2)):
    """Base de hedging alpha : mot vide (pour X0) puis v.lead_letter.

    v parcourt base_letters^m pour m = 0..M-1 ; lead_letter = canal prix-lead.
    """
    HW = [()]
    for m in range(M):
        for v in product(base_letters, repeat=m):
            HW.append(v + (lead_letter,))
    return HW


def expected_signature(S, T, order, d=3):
    """E[S^LL] : moyenne empirique des signatures lead-lag jusqu'a `order`.

    Renvoie un dict mot -> valeur ; le mot vide vaut 1.0 par convention.
    """
    vec = signature_transform(build_path_LL(S, T), order).mean(axis=0)
    E_sig = {(): 1.0}
    for i, w in enumerate(sig_words(order, d)):
        E_sig[w] = vec[i]
    return E_sig


def build_reg_system(E_sig, HW, xi):
    """Forme quadratique de (3.9) : J(theta) = theta^T Q theta - 2 b^T theta.

    Q_jk = <HW_j shuffle HW_k, E[S^LL]>,  b_j = <HW_j shuffle xi, E[S^LL]>.
    """
    def contract(sh):
        return sum(c * E_sig.get(w, 0.0) for w, c in sh.items())

    p = len(HW)
    Q = np.zeros((p, p))
    b = np.zeros(p)
    for j in range(p):
        for k in range(j, p):
            Q[j, k] = Q[k, j] = contract(shuffle(HW[j], HW[k]))
        b[j] = sum(c_xi * contract(shuffle(HW[j], w_xi)) for w_xi, c_xi in xi.items())
    return Q, b


def solve_reg(Q, b):
    """Minimum global de la forme quadratique : theta = Q^{-1} b (coincide avec BCS)."""
    return np.linalg.solve(Q, b)


def linear_reg_gradient_descent(Q, b, n_restarts=10, steps=4000, lr=1e-2, seed=0):
    """Minimise J(theta) = theta^T Q theta - 2 b^T theta par descente Adam multi-start.

    Reproduit la methode d'optimisation du papier (Remark 5.0.2 : 10 restarts,
    on garde le meilleur) plutot que la solution fermee Q theta = b.
    Renvoie (theta, J_min).
    """
    p = Q.shape[0]
    rng = np.random.default_rng(seed)
    best_theta, best_J = None, np.inf

    for _ in range(n_restarts):
        theta = 0.01 * rng.standard_normal(p)
        m, v = np.zeros(p), np.zeros(p)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            grad = 2 * (Q @ theta - b)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad * grad
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        J_val = theta @ Q @ theta - 2 * b @ theta
        if J_val < best_J:
            best_J, best_theta = J_val, theta.copy()

    return best_theta, best_J


def reg_pnl(theta, S_test, HW, T, M, payoff_test, d=3):
    """P&L hors-echantillon de Linear REG : X0 + <alpha, S^LL> - payoff.

    theta[0] = X0 (mot vide), theta[1:] = coefficients des mots de hedging HW[1:].
    """
    sig = signature_transform(build_path_LL(S_test, T), M)
    w2idx = word_to_index(sig_words(M, d))
    cols = [w2idx[w] for w in HW[1:]]
    return theta[0] + sig[:, cols] @ theta[1:] - payoff_test
