import numpy as np
import scipy.stats as st

from fourier.black_scholes import bs_asian_cf, bs_cf, compute_call_bs_asian, compute_X0_bs
from signatures.bergomi_kernel import compute_sigma_t
from signatures.tensor_algebra import projection, shuffle_product, zeros_tensor


def bracket(l, sig_W, words, word_idx, sig_word_idx=None):
    """
    Compute <l, signature> for one time slice of signatures without the empty word.

    sig_W is expected to have shape (n_paths, n_non_empty_words).
    """
    if sig_word_idx is None:
        words_no_empty = [w for w in words if len(w) > 0]
        sig_word_idx = {w: i for i, w in enumerate(words_no_empty)}

    result = l[word_idx[()]] * np.ones(sig_W.shape[0], dtype=complex)
    for w in words:
        if len(w) == 0:
            continue
        result += l[word_idx[w]] * sig_W[:, sig_word_idx[w]]
    return result


def riccati_rhs_bergomi(psi, t, u, sigma_t, rho, f, words, word_idx, M):
    f_val = f(t, u)
    proj2 = projection(psi, 2, words, word_idx, M)
    return (
        0.5 * shuffle_product(proj2, proj2, words, word_idx, M)
        + rho * f_val * shuffle_product(sigma_t, proj2, words, word_idx, M)
        + 0.5 * projection(proj2, 2, words, word_idx, M)
        + projection(psi, 1, words, word_idx, M)
        + 0.5 * (f_val**2 - f_val) * shuffle_product(sigma_t, sigma_t, words, word_idx, M)
    )


def solve_riccati_bergomi(
    u, rho, f, varsigma, f0, epsilon, H, T, tau_target, M, words, word_idx, N_steps=1000
):
    if tau_target <= 0:
        return zeros_tensor(M)

    h = tau_target / N_steps
    tau = 0.0
    psi = zeros_tensor(M)

    for _ in range(N_steps):
        t1 = T - tau
        t2 = T - (tau + h / 2)
        t4 = T - (tau + h)

        sigma1 = compute_sigma_t(t1, varsigma, f0, epsilon, H, M, words, word_idx)
        sigma2 = compute_sigma_t(t2, varsigma, f0, epsilon, H, M, words, word_idx)
        sigma4 = compute_sigma_t(t4, varsigma, f0, epsilon, H, M, words, word_idx)

        k1 = riccati_rhs_bergomi(psi, t1, u, sigma1, rho, f, words, word_idx, M)
        k2 = riccati_rhs_bergomi(psi + h / 2 * k1, t2, u, sigma2, rho, f, words, word_idx, M)
        k3 = riccati_rhs_bergomi(psi + h / 2 * k2, t2, u, sigma2, rho, f, words, word_idx, M)
        k4 = riccati_rhs_bergomi(psi + h * k3, t4, u, sigma4, rho, f, words, word_idx, M)

        psi += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        tau += h

    return psi


def solve_riccati_bergomi_learned(
    u, rho, f, sigma_learned, T, tau_target, M, words, word_idx, N_steps=100
):
    if tau_target <= 0:
        return zeros_tensor(M)

    h = tau_target / N_steps
    tau = 0.0
    psi = zeros_tensor(M)

    for _ in range(N_steps):
        t1 = T - tau
        t2 = T - (tau + h / 2)
        t4 = T - (tau + h)

        k1 = riccati_rhs_bergomi(psi, t1, u, sigma_learned, rho, f, words, word_idx, M)
        k2 = riccati_rhs_bergomi(psi + h / 2 * k1, t2, u, sigma_learned, rho, f, words, word_idx, M)
        k3 = riccati_rhs_bergomi(psi + h / 2 * k2, t2, u, sigma_learned, rho, f, words, word_idx, M)
        k4 = riccati_rhs_bergomi(psi + h * k3, t4, u, sigma_learned, rho, f, words, word_idx, M)

        psi += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        tau += h

    return psi


def solve_riccati_all_steps(u, rho, J, f, sigma_learned, T, M, words, word_idx, N_steps=100):
    dt = T / J
    h = dt / N_steps
    psi = zeros_tensor(M)
    psi_all = np.zeros((J + 1, len(words)), dtype=complex)

    for j in range(J):
        t = j * dt
        for _ in range(N_steps):
            k1 = riccati_rhs_bergomi(psi, t, u, sigma_learned, rho, f, words, word_idx, M)
            k2 = riccati_rhs_bergomi(psi + h / 2 * k1, t + h / 2, u, sigma_learned, rho, f, words, word_idx, M)
            k3 = riccati_rhs_bergomi(psi + h / 2 * k2, t + h / 2, u, sigma_learned, rho, f, words, word_idx, M)
            k4 = riccati_rhs_bergomi(psi + h * k3, t + h, u, sigma_learned, rho, f, words, word_idx, M)
            psi += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        psi_all[j + 1] = psi

    return psi_all


def compute_X0_sig_fourier(
    rho, f, sigma_learned, T, M, words, word_idx, V0=0.0625, K=1, n_laguerre=32, N_steps=100
):
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    X0_bs = compute_X0_bs(V0=V0, T=T, K=K)

    integral = 0.0
    for k in range(n_laguerre):
        uk = x[k] - i / 2
        psi = solve_riccati_bergomi_learned(
            u=uk,
            rho=rho,
            f=f,
            sigma_learned=sigma_learned,
            T=T,
            tau_target=T,
            M=M,
            words=words,
            word_idx=word_idx,
            N_steps=N_steps,
        )
        phi_sig = np.exp(psi[word_idx[()]])
        phi_bs = bs_cf(uk, T, V0)
        integral += w[k] * np.exp(x[k]) * np.real((phi_sig - phi_bs) / (0.25 + x[k] ** 2))

    return X0_bs - K * integral / np.pi


def compute_delta_sig_fourier_vectorized(
    rho,
    S,
    K,
    f,
    sig_W_hat,
    sigma_learned,
    sigma_t_all,
    T,
    J,
    M,
    words,
    word_idx,
    V0=0.0625,
    N_steps=100,
    n_laguerre=128,
):
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    I = len(S)
    delta_all = np.zeros((I, J))

    dt = T / J
    taus = T - np.arange(J) * dt
    log_S = np.log(S[:, :J])
    S_j = S[:, :J]
    sigma_t = sigma_t_all[:, :J]

    for k in range(n_laguerre):
        u_k = x[k] - i / 2
        psi_all = solve_riccati_all_steps(
            u_k, rho, J, f, sigma_learned, T, M, words, word_idx, N_steps=N_steps
        )

        psi_rev = psi_all[1:][::-1]
        brackets = psi_rev[:, 0] + np.einsum("ijk,jk->ij", sig_W_hat[:, :J, :], psi_rev[:, 1:])

        phi_sig = np.exp(brackets)
        phi_bs_all = np.array([bs_cf(u_k, tau, V0) for tau in taus])
        M_t = phi_sig * np.exp(i * u_k * log_S)
        M_t_bs = phi_bs_all[None, :] * np.exp(i * u_k * log_S)

        proj2_all = np.array([projection(psi_all[J - j], 2, words, word_idx, M) for j in range(J)])
        proj_brackets = proj2_all[:, 0] + np.einsum(
            "ijk,jk->ij", sig_W_hat[:, :J, :], proj2_all[:, 1:]
        )

        zeta = i * u_k / S_j * (M_t - M_t_bs) + rho * M_t * proj_brackets / (S_j * sigma_t)
        integrand = np.real(zeta / (0.25 + x[k] ** 2))
        delta_all += w[k] * np.exp(x[k]) * integrand

    for j in range(J):
        tau = taus[j]
        d1 = (np.log(S_j[:, j] / K) + 0.5 * V0 * tau) / np.sqrt(tau * V0)
        delta_all[:, j] = st.norm.cdf(d1) - K / np.pi * delta_all[:, j]

    return delta_all


def compute_delta_sig_fourier_asian_vectorized(
    rho,
    S,
    K,
    f,
    sig_W_hat,
    sigma_learned,
    sigma_t_all,
    T,
    J,
    M,
    words,
    word_idx,
    V0=0.0625,
    N_steps=100,
    n_laguerre=128,
):
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    I = len(S)
    dt = T / J
    delta_all = np.zeros((I, J))

    taus = T - np.arange(J) * dt
    t_array = np.arange(J) * dt
    log_S = np.log(S[:, :J])
    S_j = S[:, :J]
    sigma_recon = sigma_learned[0] + np.einsum("ijk,k->ij", sig_W_hat[:, :J, :], sigma_learned[1:])

    log_cumsum = np.zeros(I)
    bar_S = np.zeros((I, J))
    for j in range(J):
        bar_S[:, j] = np.exp(log_cumsum / T)
        log_cumsum += np.log(S[:, j]) * dt

    for k in range(n_laguerre):
        u_k = x[k] - i / 2
        psi_all = solve_riccati_all_steps(
            u_k, rho, J, f, sigma_learned, T, M, words, word_idx, N_steps=N_steps
        )

        psi_rev = psi_all[1:][::-1]
        brackets = psi_rev[:, 0] + np.einsum("ijk,jk->ij", sig_W_hat[:, :J, :], psi_rev[:, 1:])

        phi_sig = np.exp(brackets)
        phi_bs_all = np.array([bs_asian_cf(u_k, tau, T, V0) for tau in taus])
        asian_exponent = 1j * u_k * (T - t_array) / T * log_S + 1j * u_k * np.log(bar_S)
        M_t = phi_sig * np.exp(asian_exponent)
        M_t_bs = phi_bs_all[None, :] * np.exp(asian_exponent)

        proj2_all = np.array([projection(psi_all[J - j], 2, words, word_idx, M) for j in range(J)])
        proj_brackets = proj2_all[:, 0] + np.einsum(
            "ijk,jk->ij", sig_W_hat[:, :J, :], proj2_all[:, 1:]
        )

        zeta = i * u_k / S_j * (M_t - M_t_bs) + rho * M_t * proj_brackets / (S_j * sigma_recon)
        integrand = np.real(zeta / (0.25 + x[k] ** 2))
        delta_all += w[k] * np.exp(x[k]) * integrand

    sigma_eff = np.sqrt(V0 * taus**3 / (3 * T**2))
    sigma_eff = np.where(taus == 0, 1e-10, sigma_eff)
    d2 = (np.log(bar_S / K) + taus * log_S / T - 0.5 * V0 * taus**2 / (2 * T)) / sigma_eff
    d1 = d2 + sigma_eff
    adjustment = np.exp(taus * log_S / T - 0.5 * V0 * (taus**2 / (2 * T) - taus**3 / (3 * T**2)))
    delta_bs = st.norm.cdf(d1) * (taus / (T * S_j)) * bar_S * adjustment

    return delta_bs - K / np.pi * delta_all


def compute_X0_sig_fourier_asian(
    rho, f, sigma_learned, T, M, words, word_idx, V0=0.0625, K=1, n_laguerre=32, N_steps=100
):
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    X0_bs_asian = compute_call_bs_asian(S=1, moy_S=1, K=K, t=0, T=T, V0=V0)

    integral = 0.0
    for k in range(n_laguerre):
        uk = x[k] - i / 2
        psi = solve_riccati_bergomi_learned(
            u=uk,
            rho=rho,
            f=f,
            sigma_learned=sigma_learned,
            T=T,
            tau_target=T,
            M=M,
            words=words,
            word_idx=word_idx,
            N_steps=N_steps,
        )
        phi_sig = np.exp(psi[word_idx[()]])
        phi_bs = bs_asian_cf(uk, tau=T, T=T, V0=V0)
        integral += w[k] * np.exp(x[k]) * np.real((phi_sig - phi_bs) / (0.25 + x[k] ** 2))

    return X0_bs_asian - K * integral / np.pi


def build_scale_vector(words, dt):
    return np.array([dt ** (len(w) / 2) for w in words], dtype=complex)


def normalize_tensor(l, scale):
    return l * scale


def denormalize_tensor(l, scale):
    return l / scale


def riccati_rhs_rescaled(psi_tilde, t, u, sigma_tilde, rho, f, words, word_idx, M, dt):
    f_val = f(t, u)
    sqdt = np.sqrt(dt)

    proj2_raw = projection(psi_tilde, 2, words, word_idx, M)
    proj1_raw = projection(psi_tilde, 1, words, word_idx, M)
    proj2 = proj2_raw / sqdt
    proj1 = proj1_raw / sqdt
    proj22 = projection(proj2_raw, 2, words, word_idx, M) / dt

    return (
        0.5 * shuffle_product(proj2, proj2, words, word_idx, M)
        + rho * f_val * shuffle_product(sigma_tilde, proj2, words, word_idx, M)
        + 0.5 * proj22
        + proj1
        + 0.5 * (f_val**2 - f_val) * shuffle_product(sigma_tilde, sigma_tilde, words, word_idx, M)
    )


def solve_riccati_rescaled_all_steps(
    u,
    rho,
    f,
    T,
    J,
    M,
    words,
    word_idx,
    varsigma,
    f0,
    epsilon,
    H,
    scale,
    N_steps=10,
):
    dt = T / J
    h = dt / N_steps
    psi_tilde = zeros_tensor(M)
    psi_all_tilde = np.zeros((J + 1, len(words)), dtype=complex)

    tau = 0.0
    for j in range(J):
        for _ in range(N_steps):
            t1 = T - tau
            t2 = T - (tau + h / 2)
            t4 = T - (tau + h)

            sigma1 = normalize_tensor(compute_sigma_t(t1, varsigma, f0, epsilon, H, M, words, word_idx), scale)
            sigma2 = normalize_tensor(compute_sigma_t(t2, varsigma, f0, epsilon, H, M, words, word_idx), scale)
            sigma4 = normalize_tensor(compute_sigma_t(t4, varsigma, f0, epsilon, H, M, words, word_idx), scale)

            k1 = riccati_rhs_rescaled(psi_tilde, t1, u, sigma1, rho, f, words, word_idx, M, dt)
            k2 = riccati_rhs_rescaled(psi_tilde + h / 2 * k1, t2, u, sigma2, rho, f, words, word_idx, M, dt)
            k3 = riccati_rhs_rescaled(psi_tilde + h / 2 * k2, t2, u, sigma2, rho, f, words, word_idx, M, dt)
            k4 = riccati_rhs_rescaled(psi_tilde + h * k3, t4, u, sigma4, rho, f, words, word_idx, M, dt)

            psi_tilde += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            tau += h

        psi_all_tilde[j + 1] = psi_tilde

    return psi_all_tilde


def compute_X0_sig_fourier_rescaled(
    rho,
    f,
    T,
    J,
    M,
    words,
    word_idx,
    varsigma,
    f0,
    epsilon,
    H,
    scale,
    V0=0.0625,
    K=1,
    n_laguerre=32,
    N_steps=10,
):
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    X0_bs = compute_X0_bs(V0=V0, T=T, K=K)
    integral = 0.0

    for k in range(n_laguerre):
        uk = x[k] - i / 2
        psi_all_tilde = solve_riccati_rescaled_all_steps(
            uk, rho, f, T, J, M, words, word_idx, varsigma, f0, epsilon, H, scale, N_steps=N_steps
        )
        phi_sig = np.exp(psi_all_tilde[J][word_idx[()]])
        phi_bs = bs_cf(uk, T, V0)
        integral += w[k] * np.exp(x[k]) * np.real((phi_sig - phi_bs) / (0.25 + x[k] ** 2))

    return X0_bs - K * integral / np.pi


def compute_delta_sig_fourier_rescaled(
    rho,
    S,
    K,
    f,
    sig_W_tilde,
    sigma_t_all,
    T,
    J,
    M,
    words,
    word_idx,
    varsigma,
    f0,
    epsilon,
    H,
    scale,
    V0=0.0625,
    N_steps=10,
    n_laguerre=32,
):
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
        psi_all_tilde = solve_riccati_rescaled_all_steps(
            uk, rho, f, T, J, M, words, word_idx, varsigma, f0, epsilon, H, scale, N_steps=N_steps
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
        integrand = np.real(zeta / (0.25 + x[k] ** 2))
        delta_all += w[k] * np.exp(x[k]) * integrand

    for j in range(J):
        tau = taus[j]
        d1 = (np.log(S_j[:, j] / K) + 0.5 * V0 * tau) / np.sqrt(tau * V0)
        delta_all[:, j] = st.norm.cdf(d1) - K / np.pi * delta_all[:, j]

    return delta_all


def compute_X0_sig_fourier_asian_rescaled(
    rho,
    f,
    T,
    J,
    M,
    words,
    word_idx,
    varsigma,
    f0,
    epsilon,
    H,
    scale,
    V0=0.0625,
    K=1,
    n_laguerre=32,
    N_steps=10,
):
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    X0_bs_asian = compute_call_bs_asian(S=1, moy_S=1, K=K, t=0, T=T, V0=V0)
    integral = 0.0

    for k in range(n_laguerre):
        uk = x[k] - i / 2
        psi_all_tilde = solve_riccati_rescaled_all_steps(
            uk, rho, f, T, J, M, words, word_idx, varsigma, f0, epsilon, H, scale, N_steps=N_steps
        )
        phi_sig = np.exp(psi_all_tilde[J][word_idx[()]])
        phi_bs = bs_asian_cf(uk, tau=T, T=T, V0=V0)
        integral += w[k] * np.exp(x[k]) * np.real((phi_sig - phi_bs) / (0.25 + x[k] ** 2))

    return X0_bs_asian - K * integral / np.pi


def compute_delta_sig_fourier_asian_rescaled(
    rho,
    S,
    K,
    f,
    sig_W_tilde,
    sigma_t_all,
    T,
    J,
    M,
    words,
    word_idx,
    varsigma,
    f0,
    epsilon,
    H,
    scale,
    V0=0.0625,
    N_steps=10,
    n_laguerre=32,
):
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

    log_cumsum = np.zeros(I)
    bar_S = np.zeros((I, J))
    for j in range(J):
        bar_S[:, j] = np.exp(log_cumsum / T)
        log_cumsum += np.log(S[:, j]) * dt

    for k in range(n_laguerre):
        uk = x[k] - i / 2
        psi_all_tilde = solve_riccati_rescaled_all_steps(
            uk, rho, f, T, J, M, words, word_idx, varsigma, f0, epsilon, H, scale, N_steps=N_steps
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

        #zeta = i * uk / S_j * (M_t - M_t_bs) + rho * M_t * proj_brackets / (S_j * sigma_t)
        f_val_t = i * uk * (T - t_array)[None, :] / T  # f(t,u) = iu(T-t)/T, shape (1, J)
        zeta = f_val_t / S_j * (M_t - M_t_bs) + rho * M_t * proj_brackets / (S_j * sigma_t)
        integrand = np.real(zeta / (0.25 + x[k] ** 2))
        delta_all += w[k] * np.exp(x[k]) * integrand

    sigma_eff = np.sqrt(V0 * taus**3 / (3 * T**2))
    sigma_eff = np.where(taus == 0, 1e-10, sigma_eff)
    d2 = (np.log(bar_S / K) + taus * log_S / T - 0.5 * V0 * taus**2 / (2 * T)) / sigma_eff
    d1 = d2 + sigma_eff
    adjustment = np.exp(taus * log_S / T - 0.5 * V0 * (taus**2 / (2 * T) - taus**3 / (3 * T**2)))
    delta_bs = st.norm.cdf(d1) * (taus / (T * S_j)) * bar_S * adjustment

    return delta_bs - K / np.pi * delta_all
