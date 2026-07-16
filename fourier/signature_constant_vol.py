"""Pricing et couverture Fourier-signature a tenseur de volatilite CONSTANT (Sec. 5.2-5.3).

Pendant de `signature_volatility.solve_riccati_rescaled_all_steps` (Bergomi), ou
sigma_t depend du temps et doit etre recalcule a chaque etage de RK4. Sous Heston
(CIR) et sous le modele a retard, la representation signature de la volatilite est un
tenseur sigma FIXE : Sigma_t = <sigma, S(W_hat)_{0,t}>, toute la dependance en t etant
portee par la signature. On peut donc le rescaler une seule fois avant l'integration,
ce qui rend le solveur nettement plus rapide.

Le rescaling est celui de `signature_volatility` : psi_tilde = psi * dt^{|w|/2} et
sig_tilde = sig / dt^{|w|/2}, qui laisse <psi, sig> invariant tout en ramenant les
normes a O(1).

Ces fonctions servent aussi bien a Fourier BCS (sigma exact, via
`signatures.volatility_models`) qu'a Fourier REG (sigma calibre, via
`signatures.calibration`) : seul le tenseur sigma passe en argument change.
"""

import numpy as np
import scipy.stats as st

from fourier.black_scholes import bs_asian_cf, bs_cf, compute_call_bs_asian, compute_X0_bs
from fourier.signature_volatility import build_scale_vector, normalize_tensor, riccati_rhs_rescaled
from signatures.tensor_algebra import projection, zeros_tensor


def rescale_stream_signature(sig_stream, words, word_idx, dt):
    """Prepare la signature stream de W_hat pour les fonctions de delta ci-dessous.

    `sig_stream` (I, J, n_mots_non_vides) vient de iisignature en mode stream, qui
    n'emet ni le mot vide ni l'instant t_0. On prefixe donc une signature nulle a t_0
    -> (I, J+1, .), puis on divise par le scale (les fonctions de delta indexent
    [:, :J, :], soit les valeurs previsibles t_0..t_{J-1}).

    Division et non multiplication : <psi, sig> = <psi_tilde, sig_tilde> avec
    psi_tilde = psi * scale impose sig_tilde = sig / scale.
    """
    scale = build_scale_vector(words, dt)
    words_no_empty = [w for w in words if len(w) > 0]
    scale_no_empty = np.array([scale[word_idx[w]] for w in words_no_empty])

    I = sig_stream.shape[0]
    sig_full = np.zeros((I, sig_stream.shape[1] + 1, sig_stream.shape[-1]), dtype=sig_stream.dtype)
    sig_full[:, 1:, :] = sig_stream
    return sig_full / scale_no_empty[None, None, :]


def solve_riccati_rescaled_all_steps_constant(
    u, rho, f, T, J, M, words, word_idx, sigma_arr, N_steps=10
):
    """Riccati rescale a rebours, a tenseur de volatilite constant.

    Renvoie psi_all_tilde (J+1, len(words)) : psi rescale a chaque instant de la grille.

    f(t, u) attend le temps CALENDAIRE, alors que l'integration se fait a rebours en
    tau = T - t : d'ou les t = T - tau passes aux etages de RK4. Invisible pour le call
    europeen (f = iu, constante), mais fatal pour l'asiatique (f = iu (T-t)/T).
    """
    dt = T / J
    h = dt / N_steps
    scale = build_scale_vector(words, dt)
    sigma_tilde = normalize_tensor(sigma_arr, scale)  # sigma fixe : rescale UNE FOIS

    psi_tilde = zeros_tensor(M)
    psi_all_tilde = np.zeros((J + 1, len(words)), dtype=complex)

    tau = 0.0
    for j in range(J):
        for _ in range(N_steps):
            t1, t2, t4 = T - tau, T - (tau + h / 2), T - (tau + h)
            k1 = riccati_rhs_rescaled(psi_tilde, t1, u, sigma_tilde, rho, f, words, word_idx, M, dt)
            k2 = riccati_rhs_rescaled(psi_tilde + h / 2 * k1, t2, u, sigma_tilde, rho, f, words, word_idx, M, dt)
            k3 = riccati_rhs_rescaled(psi_tilde + h / 2 * k2, t2, u, sigma_tilde, rho, f, words, word_idx, M, dt)
            k4 = riccati_rhs_rescaled(psi_tilde + h * k3, t4, u, sigma_tilde, rho, f, words, word_idx, M, dt)
            psi_tilde += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            tau += h
        psi_all_tilde[j + 1] = psi_tilde

    return psi_all_tilde


# =====================================================================
# Call europeen (Sec. 5.2)
# =====================================================================

def compute_X0_sig_fourier_constant_rescaled(
    rho, f, T, J, M, words, word_idx, sigma_arr, V0=0.0625, K=1, n_laguerre=32, N_steps=10
):
    """Richesse initiale X0* du call europeen (Sec. 3.1), variate de controle Black-Scholes."""
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    X0_bs = compute_X0_bs(V0=V0, T=T, K=K)

    integral = 0.0
    for k in range(n_laguerre):
        uk = x[k] - i / 2
        psi_all_tilde = solve_riccati_rescaled_all_steps_constant(
            uk, rho, f, T, J, M, words, word_idx, sigma_arr, N_steps=N_steps
        )
        # mot vide : pas de de-rescaling a faire (dt^0 = 1)
        phi_sig = np.exp(psi_all_tilde[J][word_idx[()]])
        phi_bs = bs_cf(uk, T, V0)
        integral += w[k] * np.exp(x[k]) * np.real((phi_sig - phi_bs) / (0.25 + x[k] ** 2))

    return X0_bs - K * integral / np.pi


def compute_delta_sig_fourier_constant_rescaled(
    rho, S, K, f, sig_W_tilde, sigma_t_all, T, J, M, words, word_idx, sigma_arr,
    V0=0.0625, N_steps=10, n_laguerre=32,
):
    """Delta / alpha* du call europeen (Sec. 3.1), pour f(t, u) = iu.

    `sigma_t_all` (I, J+1) est la volatilite VRAIE simulee, pas sa reconstruction par
    signature : elle apparait au denominateur du terme de correlation.
    """
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    I = len(S)
    dt = T / J
    delta_all = np.zeros((I, J))

    taus = T - np.arange(J) * dt
    log_S = np.log(S[:, :J])
    S_j = S[:, :J]
    sigma_t = sigma_t_all[:, :J]

    for k in range(n_laguerre):
        uk = x[k] - i / 2
        psi_all_tilde = solve_riccati_rescaled_all_steps_constant(
            uk, rho, f, T, J, M, words, word_idx, sigma_arr, N_steps=N_steps
        )
        psi_rev = psi_all_tilde[1:][::-1]
        brackets = psi_rev[:, 0] + np.einsum("ijk,jk->ij", sig_W_tilde[:, :J, :], psi_rev[:, 1:])

        phi_sig = np.exp(brackets)
        phi_bs_all = np.array([bs_cf(uk, tau, V0) for tau in taus])
        M_t = phi_sig * np.exp(i * uk * log_S)
        M_t_bs = phi_bs_all[None, :] * np.exp(i * uk * log_S)

        proj2_all = np.array(
            [projection(psi_all_tilde[J - j], 2, words, word_idx, M) / np.sqrt(dt) for j in range(J)]
        )
        proj_brackets = proj2_all[:, 0] + np.einsum(
            "ijk,jk->ij", sig_W_tilde[:, :J, :], proj2_all[:, 1:]
        )

        zeta = i * uk / S_j * (M_t - M_t_bs) + rho * M_t * proj_brackets / (S_j * sigma_t)
        delta_all += w[k] * np.exp(x[k]) * np.real(zeta / (0.25 + x[k] ** 2))

    for j in range(J):
        tau = taus[j]
        d1 = (np.log(S_j[:, j] / K) + 0.5 * V0 * tau) / np.sqrt(tau * V0)
        delta_all[:, j] = st.norm.cdf(d1) - K / np.pi * delta_all[:, j]

    return delta_all


# =====================================================================
# Call asiatique geometrique (Sec. 5.3)
# =====================================================================

def compute_X0_sig_fourier_asian_constant_rescaled(
    rho, f, T, J, M, words, word_idx, sigma_arr, V0=0.0625, K=1, n_laguerre=32, N_steps=10
):
    """Richesse initiale X0* du call asiatique geometrique, variate de controle BS asiatique."""
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    X0_bs_asian = compute_call_bs_asian(S=1, moy_S=1, K=K, t=0, T=T, V0=V0)

    integral = 0.0
    for k in range(n_laguerre):
        uk = x[k] - i / 2
        psi_all_tilde = solve_riccati_rescaled_all_steps_constant(
            uk, rho, f, T, J, M, words, word_idx, sigma_arr, N_steps=N_steps
        )
        phi_sig = np.exp(psi_all_tilde[J][word_idx[()]])
        phi_bs = bs_asian_cf(uk, tau=T, T=T, V0=V0)
        integral += w[k] * np.exp(x[k]) * np.real((phi_sig - phi_bs) / (0.25 + x[k] ** 2))

    return X0_bs_asian - K * integral / np.pi


def compute_delta_sig_fourier_asian_constant_rescaled(
    rho, S, K, f, sig_W_tilde, sigma_t_all, T, J, M, words, word_idx, sigma_arr,
    V0=0.0625, N_steps=10, n_laguerre=32,
):
    """Delta / alpha* du call asiatique geometrique, pour f(t, u) = iu (T-t)/T.

    `sigma_t_all` (I, J+1) est la volatilite VRAIE simulee. Sous le modele a retard
    elle change de signe : la proteger par np.maximum(., eps) detruirait le signe et
    ferait exploser le terme de correlation -- utiliser un plancher signe.
    """
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    I = len(S)
    dt = T / J
    delta_all = np.zeros((I, J))

    taus = T - np.arange(J) * dt
    t_array = np.arange(J) * dt
    log_S = np.log(S[:, :J])
    S_j = S[:, :J]
    sigma_t = sigma_t_all[:, :J]

    # moyenne geometrique glissante bar_S, previsible (bar_S[:, j] n'utilise que S_0..S_{j-1})
    log_cumsum = np.zeros(I)
    bar_S = np.zeros((I, J))
    for j in range(J):
        bar_S[:, j] = np.exp(log_cumsum / T)
        log_cumsum += np.log(S[:, j]) * dt

    for k in range(n_laguerre):
        uk = x[k] - i / 2
        psi_all_tilde = solve_riccati_rescaled_all_steps_constant(
            uk, rho, f, T, J, M, words, word_idx, sigma_arr, N_steps=N_steps
        )
        psi_rev = psi_all_tilde[1:][::-1]
        brackets = psi_rev[:, 0] + np.einsum("ijk,jk->ij", sig_W_tilde[:, :J, :], psi_rev[:, 1:])

        phi_sig = np.exp(brackets)
        phi_bs_all = np.array([bs_asian_cf(uk, tau, T, V0) for tau in taus])
        asian_exponent = i * uk * (T - t_array) / T * log_S + i * uk * np.log(bar_S)
        M_t = phi_sig * np.exp(asian_exponent)
        M_t_bs = phi_bs_all[None, :] * np.exp(asian_exponent)

        proj2_all = np.array(
            [projection(psi_all_tilde[J - j], 2, words, word_idx, M) / np.sqrt(dt) for j in range(J)]
        )
        proj_brackets = proj2_all[:, 0] + np.einsum(
            "ijk,jk->ij", sig_W_tilde[:, :J, :], proj2_all[:, 1:]
        )

        f_val_t = i * uk * (T - t_array)[None, :] / T
        zeta = f_val_t / S_j * (M_t - M_t_bs) + rho * M_t * proj_brackets / (S_j * sigma_t)
        delta_all += w[k] * np.exp(x[k]) * np.real(zeta / (0.25 + x[k] ** 2))

    # variate de controle : delta BS asiatique analytique
    sigma_eff = np.sqrt(V0 * taus**3 / (3 * T**2))
    sigma_eff = np.where(taus == 0, 1e-10, sigma_eff)
    d2 = (np.log(bar_S / K) + taus * log_S / T - 0.5 * V0 * taus**2 / (2 * T)) / sigma_eff
    d1 = d2 + sigma_eff
    adjustment = np.exp(taus * log_S / T - 0.5 * V0 * (taus**2 / (2 * T) - taus**3 / (3 * T**2)))
    delta_bs = st.norm.cdf(d1) * (taus / (T * S_j)) * bar_S * adjustment

    return delta_bs - K / np.pi * delta_all


# =====================================================================
# P&L
# =====================================================================

def signed_floor(sigma_t, eps=1e-4):
    """Plancher en VALEUR ABSOLUE preservant le signe, pour les denominateurs en sigma_t.

    Sous le modele a retard Sigma_t n'est pas contraint positif : np.maximum(., eps)
    y basculerait le signe des trajectoires negatives.
    """
    return np.where(np.abs(sigma_t) < eps, eps * np.where(sigma_t < 0, -1.0, 1.0), sigma_t)


def hedging_pnl(X0, delta_all, S, payoff):
    """P&L de couverture auto-financee : X0 + sum_j delta_j (S_{j+1} - S_j) - payoff."""
    dS = np.diff(S, axis=1)
    return X0 + np.sum(delta_all * dS, axis=1) - payoff
