def riccati_rhs_bergomi(psi , t , u , sigma_t , rho , f , words , word_idx , M ):
    f_val = f (t , u)
    return 0.5 * ( shuffle_product(projection(psi ,2 , words , word_idx , M ) , projection(psi ,2 , words , word_idx , M ) , words , word_idx , M ) )+\
            rho * f_val * ( shuffle_product( sigma_t  , projection (psi , 2 , words , word_idx , M ), words , word_idx , M)) +\
            0.5 * projection(projection(psi, 2, words, word_idx , M), 2, words, word_idx  , M) + projection ( psi , 1 , words , word_idx , M ) +\
            0.5 * (f_val**2-f_val) * shuffle_product( sigma_t , sigma_t , words , word_idx , M )  

def solve_riccati_bergomi(u, rho, f, varsigma, f0, epsilon, H, T, 
                          tau_target, M, words, word_idx, N_steps=1000):
    if tau_target <= 0:
        return zeros_tensor(M)
    
    h = tau_target / N_steps  # pas de temps
    tau = 0.0
    psi = zeros_tensor(M)     # condition initiale psi(0) = 0
    
    for _ in range(N_steps):
        t = T - tau            # temps courant (tau = T-t)
        
        # sigma_t au temps t courant
        sigma_t = compute_sigma_t(t, varsigma, f0, epsilon, H, M, words, word_idx)
        
        # 4 évaluations de F
        k1 = riccati_rhs_bergomi(psi, t, u, sigma_t, rho, f, words, word_idx, M)
        
        # Pour k2, on est à tau + h/2, donc t = T - (tau + h/2)
        t2 = T - (tau + h/2)
        sigma_t2 = compute_sigma_t(t2, varsigma, f0, epsilon, H, M, words, word_idx)
        k2 = riccati_rhs_bergomi(psi + h/2*k1, t2, u, sigma_t2, rho, f, words, word_idx, M)
        k3 = riccati_rhs_bergomi(psi + h/2*k2, t2, u, sigma_t2, rho, f, words, word_idx, M)
        
        # Pour k4, on est à tau + h, donc t = T - (tau + h)
        t4 = T - (tau + h)
        sigma_t4 = compute_sigma_t(t4, varsigma, f0, epsilon, H, M, words, word_idx)
        k4 = riccati_rhs_bergomi(psi + h*k3, t4, u, sigma_t4, rho, f, words, word_idx, M)
        
        psi += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        tau += h
    
    return psi

def solve_riccati_bergomi_learned(u, rho, f, sigma_learned, T, tau_target, 
                                   M, words, word_idx, N_steps=100):
    if tau_target <= 0:
        return zeros_tensor(M)
    
    h = tau_target / N_steps
    psi = zeros_tensor(M)
    
    for _ in range(N_steps):
        k1 = riccati_rhs_bergomi(psi, 0, u, sigma_learned, rho, f, words, word_idx, M)
        k2 = riccati_rhs_bergomi(psi + h/2*k1, 0, u, sigma_learned, rho, f, words, word_idx, M)
        k3 = riccati_rhs_bergomi(psi + h/2*k2, 0, u, sigma_learned, rho, f, words, word_idx, M)
        k4 = riccati_rhs_bergomi(psi + h*k3, 0, u, sigma_learned, rho, f, words, word_idx, M)
        psi += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return psi



def solve_riccati_all_steps(u , rho, J ,f, sigma_learned , T , M , words , word_idx , N_steps =100 ):
    dt = T / J
    h= dt / N_steps
    psi = zeros_tensor(M)
    psi_all = np.zeros((J+1, len (words)), dtype=complex)
    for j in range(J):
        t = j*dt
        for _ in range(N_steps):
            k1 = riccati_rhs_bergomi(psi, t, u, sigma_learned, rho, f, words, word_idx, M)
            k2 = riccati_rhs_bergomi(psi + h/2*k1, t + h/2, u, sigma_learned, rho, f, words, word_idx, M)
            k3 = riccati_rhs_bergomi(psi + h/2*k2, t + h/2, u, sigma_learned, rho, f, words, word_idx, M)
            k4 = riccati_rhs_bergomi(psi + h*k3, t + h, u, sigma_learned, rho, f, words, word_idx, M)
            psi += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        psi_all[j+1] = psi
    return psi_all

def compute_X0_sig_fourier(rho, f, sigma_learned, T, V0=0.0625, K=1,
                            M=M, words=words, word_idx=word_idx, n_laguerre=128):
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    X0_bs = compute_X0_bs(V0=V0, T=T)
    
    integral = 0
    for k in range(n_laguerre):
        uk = x[k] - i/2
        psi = solve_riccati_bergomi_learned(
            u=uk, rho=rho, f=f, sigma_learned=sigma_learned,
            T=T, tau_target=T, M=M, words=words, word_idx=word_idx)
        phi_sig = np.exp(psi[word_idx[()]])
        phi_bs = bs_cf(uk, T, V0)
        integral += w[k] * np.exp(x[k]) * np.real((phi_sig - phi_bs) / (0.25 + x[k]**2))
    
    return X0_bs - K * integral / np.pi

def compute_delta_sig_fourier_vectorized(rho, S, K, f, sig_W_hat, sigma_learned,
                                          sigma_t_all, T, J, M, words, word_idx,
                                          V0=0.0625,N_steps=100, n_laguerre=128):
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    I = len(S)
    delta_all = np.zeros((I, J))
    
    taus = T - np.arange(J) * (T/J)  # shape (J,)
    log_S = np.log(S[:, :J])          # shape (I, J)
    S_j = S[:, :J]                    # shape (I, J)
    sigma_t = sigma_t_all[:, :J]      # shape (I, J)
    # Au début de la fonction, calcule sigma reconstruit une seule fois
    sigma_recon = (sigma_learned[0] + 
                np.einsum('ijk,k->ij', sig_W_hat[:, :J, :], sigma_learned[1:]))
    # shape (I, J) — c'est <sigma_learned, W_hat> pour chaque trajectoire et pas

    for k in range(n_laguerre):
        u_k = x[k] - i/2
        psi_all = solve_riccati_all_steps(u_k, rho, J, f, sigma_learned, T,
                                           M, words, word_idx, N_steps=N_steps)

        # bracket pour tous j
        psi_rev = psi_all[1:][::-1]  # shape (J, 63)
        brackets = psi_rev[:, 0] + np.einsum('ijk,jk->ij', 
                    sig_W_hat[:, :J, :], psi_rev[:, 1:])  # (I, J)

        # M_t et M_t_bs
        phi_sig = np.exp(brackets)  # (I, J)
        phi_bs_all = np.array([bs_cf(u_k, tau, V0) for tau in taus])  # (J,)
        M_t = phi_sig * np.exp(i * u_k * log_S)          # (I, J)
        M_t_bs = phi_bs_all[None, :] * np.exp(i * u_k * log_S)  # (I, J)

        # proj_bracket pour tous j
        proj2_all = np.array([projection(psi_all[J-j], 2, words, word_idx, M) 
                               for j in range(J)])  # (J, 63)
        proj_brackets = (proj2_all[:, 0] + 
                         np.einsum('ijk,jk->ij', sig_W_hat[:, :J, :], 
                                   proj2_all[:, 1:]))  # (I, J)

        # zeta et integrand
        zeta = i*u_k/S_j * (M_t - M_t_bs) + rho * M_t * proj_brackets / (S_j * sigma_recon)
        integrand = np.real(zeta / (0.25 + x[k]**2))
        delta_all += w[k] * np.exp(x[k]) * integrand

    # Ajout du delta BS
    for j in range(J):
        tau = taus[j]
        d1 = (np.log(S_j[:, j]/K) + 0.5*V0*tau) / np.sqrt(tau*V0)
        delta_all[:, j] = st.norm.cdf(d1) - K/np.pi * delta_all[:, j]

    return delta_all