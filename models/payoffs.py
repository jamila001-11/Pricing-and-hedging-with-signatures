import numpy as np
def compute_payoff_europeen(S , K):
    payoff=np.maximum(S[:,-1]-K , 0)
    return payoff
def compute_payoff_asian(S, K):
    # moyenne geometrique de S_1 a S_J (coherent avec le delta Fourier cell 98)
    J = S.shape[1] - 1
    log_average = np.mean(np.log(S[:, 1:J+1]), axis=1)
    payoff_asian = np.maximum(np.exp(log_average) - K, 0)
    return payoff_asian
def compute_payoff_lookback(S):
    return S[:,-1]-np.min(S, axis=1)
def compute_payoff_polynomial(S, coef):
    # payoff polynomial en S_T : sum_k coef[k] * S_T^k (papier, Sec 5.1)
    ST = S[:, -1]
    return sum(coef[k] * ST**k for k in range(len(coef)))