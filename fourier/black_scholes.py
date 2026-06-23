import numpy as np 
import scipy.stats as st
def bs_cf(u,tau , V0 ):
    i=1j 
    return np.exp(-0.5 * i *u *V0 * tau -0.5 * u**2 * V0 * tau)
def compute_X0_bs(V0=0.0625, T=0.5 , S0=1 , K=1 , r=0):
    d1 = (np.log(S0/K) + (r + 0.5 * V0) * T) / (np.sqrt(T* V0))
    d2 = d1 - np.sqrt(T*V0)
    call_price = S0 * st.norm.cdf(d1) - K * np.exp(-r*T) * st.norm.cdf(d2)
    return call_price

def bs_asian_cf(u, tau, T, V0):
    return np.exp(-0.5 * V0 * (u**2 * tau**3 / (3*T**2) + 1j * u * tau**2 / (2*T)))

def compute_call_bs_asian(S, moy_S, K, t, T=0.5, V0=0.0625):
    tau = T - t
    sigma_eff = np.sqrt(V0 * tau**3 / (3*T**2))
    d2 = (np.log(moy_S/K) + tau*np.log(S)/T - 0.5*V0*tau**2/(2*T)) / sigma_eff
    d1 = d2 +sigma_eff  # CORRECTION : + et non -
    adjustment = np.exp(tau*np.log(S)/T - 0.5*V0*(tau**2/(2*T) - tau**3/(3*T**2)))
    call_price = moy_S * st.norm.cdf(d1) * adjustment - st.norm.cdf(d2) * K
    return call_price

def bs_lookback_call(S, m, tau, r=1e-5, sigma=0.25):
    """
    Formule analytique du Call Lookback (Floating Strike) sous Black-Scholes.
    m : minimum observé jusqu'à présent
    tau : temps restant
    """
    # Éviter la division par zéro à maturité exacte
    tau = np.maximum(tau, 1e-9)

    d1 = (np.log(S / m) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    d3 = (np.log(S / m) + (-r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

    term1 = S * st.norm.cdf(d1)
    term2 = m * np.exp(-r * tau) * st.norm.cdf(d2)
    term3 = S * (sigma**2 / (2 * r)) * (
        -st.norm.cdf(-d1) + np.exp(-r * tau) * (m / S)**(2 * r / sigma**2) * st.norm.cdf(-d3)
    )
    return term1 - term2 + term3

def bs_lookback_delta(S, m, tau, r=1e-5, sigma=0.25, h=1e-4):
    """
    Calcul du Delta par différences finies centrales sur la formule exacte.
    Vectorisé pour traiter les 20 000 trajectoires d'un coup.
    """
    if tau <= 0:
        return np.ones_like(S) # A l'échéance S_T >= m_T, la pente du payoff est 1

    price_up = bs_lookback_call(S + h, m, tau, r, sigma)
    price_dn = bs_lookback_call(S - h, m, tau, r, sigma)
    return (price_up - price_dn) / (2 * h)


def compute_bs_lookback_delta_bergomi(S, V0=0.0625, T=1/12, J=126):
    """
    Benchmark Black-Scholes EXACT du Call Lookback (floating strike) sous Bergomi.

    Couvre avec le delta analytique exact du lookback (formule floating strike,
    via bs_lookback_delta), en suivant le minimum courant m_t = min(S_0, ..., S_j),
    avec une volatilite constante sigma_BS = sqrt(V0) = sqrt(f0).

    Contrairement a compute_naive_bs_lookback_delta_bergomi (qui n'utilise que le
    delta du call europeen N(d1)), on utilise ici la vraie formule lookback.

    Renvoie les deltas de couverture, shape (I, J).
    Le prix initial associe est bs_lookback_call(1.0, 1.0, T, sigma=sqrt(V0)).
    """
    sigma_bs = np.sqrt(V0)
    dt = T / J
    I = S.shape[0]

    deltas = np.zeros((I, J))
    m_t = np.copy(S[:, 0])                       # minimum courant (m_0 = S_0)
    for j in range(J):
        tau = T - j * dt                         # temps restant
        m_t = np.minimum(m_t, S[:, j])           # min observe jusqu'a t_j
        deltas[:, j] = bs_lookback_delta(S[:, j], m_t, tau, sigma=sigma_bs)
    return deltas


def compute_naive_bs_lookback_delta_bergomi(S, V0=0.0625, T=1/12, J=126, K=1):
    """
    Benchmark naïf : delta Black-Scholes du call européen ATM,
    avec volatilité constante sigma_BS = sqrt(V0) = sqrt(f0).
    Identique en structure au benchmark Heston, juste T et les
    trajectoires Bergomi changent.
    """
    sigma_bs = np.sqrt(V0)
    dt = T / J
    I = S.shape[0]
    
    deltas_naive = np.zeros((I, J))
    for j in range(J):
        t_j = j * dt
        tau_j = T - t_j
        S_j = S[:, j]
        d1 = (np.log(S_j/K) + 0.5*sigma_bs**2*tau_j) / (sigma_bs*np.sqrt(tau_j))
        deltas_naive[:, j] = st.norm.cdf(d1)
    
    return deltas_naive