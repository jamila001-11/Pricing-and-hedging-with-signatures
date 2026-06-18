import numpy as np
import scipy.special as sp

from signatures.tensor_algebra import shuffle_exp, zeros_tensor


def compute_sigma_t(t , varsigma , f0 , epsilon , H , M ,words , word_idx):
    sigma_t = zeros_tensor(M)
    u_t = varsigma**2 *(epsilon**(2*H) - (epsilon+t)**(2*H)) / (8*H)
    k_t = zeros_tensor(M)
    for i in range(M):
        coeff = (epsilon +t)**(H-0.5-i) * sp.poch(0.5-H , i)
        k_t[word_idx[tuple([1]*(i) + [2])]] = coeff * varsigma*0.5
    l_t= zeros_tensor(M)
    l_t[word_idx[()]]= u_t
    l_t += k_t
    exp =shuffle_exp(l_t, words, word_idx, M)
    sigma_t = np.sqrt(f0)* exp
    return sigma_t
