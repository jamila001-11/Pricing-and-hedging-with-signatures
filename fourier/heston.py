def heston_cf_eur(u , tau , V , kappa , theta , eta , rho):
    i=1j
    d=np.sqrt((rho * eta*u*i-kappa)**2 + eta**2*(u*i+u**2))
    g=(kappa - rho * eta *u*i +d)/(kappa - rho * eta*u*i -d)
    D=((kappa - rho * eta * i*u + d)/ eta**2) *( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)))
    C= kappa *theta * ((kappa-rho*eta *u*i +d)*tau - 2* np.log((1-g*np.exp(d*tau))/(1- g))) / eta**2
    return np.exp(C+D*V)

def compute_X0(kappa , eta , rho , T=0.5 , V0=0.0625 , n_laguerre=128):
    x , w = np.polynomial.laguerre.laggauss(n_laguerre)
    i=1j
    u = x - i/2
    cf_T_V0 = heston_cf_eur(u, T, V0, kappa=kappa,  theta=V0, eta=eta, rho=rho)
    integrand = np.real((cf_T_V0 - bs_cf(u, T , V0)) / (0.25 + x**2))
    integral = np.sum(w * np.exp(x) * integrand)
    X0_bs = compute_X0_bs(V0=V0, T=T, S0=1, K=1, r=0)
    X0 = X0_bs - integral / np.pi
    return X0

def compute_delta_vectorized(t, S_t, V_t, kappa, theta, eta, rho, T, V0=0.0625, K=1, n_laguerre=128):
    tau = T - t
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    x = x[None, :]          # (1, 128)
    w = w[None, :]          # (1, 128)
    i = 1j
    u = x - i/2             # (1, 128)
    
    S_t = S_t[:, None]      # (n_paths, 1)
    V_t = V_t[:, None]      # (n_paths, 1)
    
    # CF Heston + extraction de B = coefficient de V_t dans l'exponent
    d = np.sqrt((rho * eta * u * i - kappa)**2 + eta**2 * (u * i + u**2))
    g = (kappa - rho * eta * u * i + d) / (kappa - rho * eta * u * i - d)
    B = ((kappa - rho * eta * i * u + d) / eta**2) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
    C_exp = kappa * theta * ((kappa - rho * eta * u * i + d) * tau
                             - 2 * np.log((1 - g * np.exp(d * tau)) / (1 - g))) / eta**2
    cf     = np.exp(C_exp + B * V_t)   # (n_paths, 128)
    cf_bs  = bs_cf(u, tau, V0)         # (1, 128)
    
    d1   = (np.log(S_t/K) + 0.5*V0*tau) / np.sqrt(tau*V0)   # (n_paths, 1)
    N_d1 = st.norm.cdf(d1)                                   # (n_paths, 1)
    
    log_moneyness = np.log(S_t / K)               # (n_paths, 1)
    exp_iu_logm   = np.exp(i * u * log_moneyness) # (n_paths, 128)
    
    # Terme delta : d C / d S_t
    integrand_delta = np.real(i * u * exp_iu_logm * (cf - cf_bs) / (0.25 + x**2))
    integral_delta  = np.sum(w * np.exp(x) * integrand_delta, axis=1, keepdims=True)
    
    # Terme vega-rho : correction optimale sous Heston
    # alpha* = delta + (rho*eta/S_t) * vega,  car d<V,S>/d<S,S> = rho*eta/S_t
    # d CF_Heston / d V_t = CF * B
    integrand_vega = np.real(exp_iu_logm * cf * B / (0.25 + x**2))
    integral_vega  = np.sum(w * np.exp(x) * integrand_vega, axis=1, keepdims=True)
    
    delta = N_d1 - K / (np.pi * S_t) * (integral_delta + rho * eta * integral_vega)
    return delta.squeeze()   # (n_paths,)

def riccati_rhs(Y, u, kappa, theta, eta, rho, T):
    i=1j
    A, B, C = Y
    dA_dtau = kappa * theta * B
    dC_dtau = i * u / T
    dB_dtau = 0.5 * C**2 - 0.5 * C - kappa * B + 0.5 * (B**2) * (eta**2) + rho * eta * C * B
    return np.array([dA_dtau, dB_dtau, dC_dtau], dtype=complex)
def solve_riccati_rk4(u, kappa, theta, eta, rho, T, tau_target, N_steps=100):
    """
    Résout le système de Riccati de tau = 0 à tau = tau_target avec RK4.
    """
    # Si le temps restant est nul, les coefficients sont nuls
    if tau_target <= 0:
        return 0.0, 0.0, 0.0
        
    dt = tau_target / N_steps
    tau = 0.0
    
    # Conditions initiales à tau = 0 : A(0)=0, B(0)=0, C(0)=0
    Y = np.array([0.0, 0.0, 0.0], dtype=complex)
    
    # Boucle RK4
    for _ in range(N_steps):
        k1 = riccati_rhs(Y,           u, kappa, theta, eta, rho, T)
        k2 = riccati_rhs(Y + dt*k1/2, u, kappa, theta, eta, rho, T)
        k3 = riccati_rhs(Y + dt*k2/2, u, kappa, theta, eta, rho, T)
        k4 = riccati_rhs(Y + dt*k3,   u, kappa, theta, eta, rho, T)
        
        Y += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        tau += dt
        
    # Retourne A, B, C sous forme de scalaires complexes
    return Y[0], Y[1], Y[2]

def asian_cf(u, tau, S_t, V_t, kappa, theta, eta, rho, T):
    i = 1j
    u = np.atleast_1d(u)
    results = np.zeros(len(u), dtype=complex)
    for k, uk in enumerate(u):
        A, B, C = solve_riccati_rk4(uk, kappa, theta, eta, rho, T,
                                     tau_target=tau, N_steps=100)
        results[k] = np.exp(A + B*V_t + (C - i*uk*tau/T)*np.log(S_t))
    return results

def compute_call_asian(S, moy_S, V_t, t, K=1, T=0.5, V0=0.0625,
                       kappa=4, theta=0.0625, eta=0.7, rho=-0.7, n_laguerre=128):
    tau = T - t
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    i = 1j
    u = x - i/2
    k_t = np.log(moy_S / K)

    cf_asian = asian_cf(u, tau=tau, S_t=S, V_t=V_t,
                        kappa=kappa, theta=theta, eta=eta, rho=rho, T=T)
    cf_bs    = bs_asian_cf(u, tau=tau, T=T, V0=V0)

    integrand = np.real(np.exp(i*u*k_t) * (cf_asian - cf_bs) / (0.25 + x**2))
    integral  = np.sum(w * np.exp(x) * integrand)

    call_bs = compute_call_bs_asian(S=S, moy_S=moy_S, K=K, t=t, T=T, V0=V0)
    return call_bs - K * integral / np.pi

def compute_delta_asian_vectorized(t, S_t, V_t, bar_S_t, kappa, theta, eta, rho, T,
                                    V0=0.0625, K=1, n_laguerre=128):
    tau = T - t
    if tau <= 0:
        return np.zeros_like(S_t)

    # Grille de Gauss-Laguerre
    x, w = np.polynomial.laguerre.laggauss(n_laguerre)
    x = x[None, :]          # Shape (1, 128)
    w = w[None, :]          # Shape (1, 128)
    i = 1j
    u = x - i/2             # Shape (1, 128)

    # Redimensionnement pour le broadcasting sur toutes les trajectoires
    S_t_mod = S_t[:, None]        # Shape (n_paths, 1)
    V_t_mod = V_t[:, None]        # Shape (n_paths, 1)
    bar_S_t_mod = bar_S_t[:, None]  # Shape (n_paths, 1)

    # 1. Résoudre Riccati pour obtenir A, B, C à chaque point u
    # On boucle sur la grille de Laguerre (très rapide car N=128)
    A_arr = np.zeros(n_laguerre, dtype=complex)
    B_arr = np.zeros(n_laguerre, dtype=complex)
    C_arr = np.zeros(n_laguerre, dtype=complex)

    for k in range(n_laguerre):
        A_arr[k], B_arr[k], C_arr[k] = solve_riccati_rk4(u[0, k], kappa, theta, eta, rho, T,
                                                         tau_target=tau, N_steps=100)

    A_arr = A_arr[None, :]  # Shape (1, 128)
    B_arr = B_arr[None, :]  # Shape (1, 128)
    C_arr = C_arr[None, :]  # Shape (1, 128)

    # 2. Calculer les fonctions caractéristiques (Heston Asian et BS Asian)
    # On conserve l'expression exacte de ta fonction asian_cf
    cf_asian = np.exp(A_arr + B_arr * V_t_mod + (C_arr - i * u * tau / T) * np.log(S_t_mod))
    cf_bs    = bs_asian_cf(u, tau=tau, T=T, V0=V0)  # Shape (1, 128)

    # 3. Log-moneyness et terme exponentiel de Fourier
    """k_t = np.log(bar_S_t_mod / K)
    exp_iu_kt = np.exp(i * u * k_t)  # Shape (n_paths, 128)"""
    # À l'intérieur de compute_delta_asian_vectorized :

    k_t_complet = np.log(bar_S_t[:, None] / K) + (tau / T) * np.log(S_t[:, None])
    exp_iu_kt = np.exp(1j * u * k_t_complet)

    # 4. Intégrande delta
    # f(t,u) = i * u * tau / T selon le papier pour l'Asiatique Géométrique
    # Ne divisez PAS par S_t_mod ici, le K / (np.pi * S_t) final s'en charge !
    facteur_f = 1j * u * tau / T
    integrand_delta = np.real(facteur_f * exp_iu_kt * (cf_asian - cf_bs) / (0.25 + x**2))
    integral_delta  = np.sum(w * np.exp(x) * integrand_delta, axis=1)  # Shape (n_paths,)

    # 5. Intégrande vega-rho (Correction optimale sous Heston via le coefficient B)
    integrand_vega = np.real(exp_iu_kt * cf_asian * B_arr / (0.25 + x**2))
    integral_vega  = np.sum(w * np.exp(x) * integrand_vega, axis=1)  # Shape (n_paths,)

    # 6. Delta BS Asian analytique
    sigma_eff = np.sqrt(V0 * tau**3 / (3 * T**2))
    d2 = (np.log(bar_S_t / K) + tau * np.log(S_t) / T - 0.5 * V0 * tau**2 / (2 * T)) / sigma_eff
    d1 = d2 + sigma_eff
    adjustment = np.exp(tau * np.log(S_t) / T - 0.5 * V0 * (tau**2 / (2 * T) - tau**3 / (3 * T**2)))

    # Dérivée de l'option BS Asiatique par rapport à S_t
    delta_bs = st.norm.cdf(d1) * (tau / (T * S_t)) * bar_S_t * adjustment

    # 7. Assembler le delta global de couverture quadratique
    # En factorisant 1/S_t issu de la définition théorique de zeta_t
    delta = delta_bs - K / (np.pi * S_t) * (integral_delta + rho * eta * integral_vega)

    return delta