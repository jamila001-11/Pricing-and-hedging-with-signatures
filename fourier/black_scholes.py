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