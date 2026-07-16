"""Representations signature exactes de la volatilite (papier, Sec. 5.2-5.3).

Chaque modele fournit un tenseur DETERMINISTE sigma tel que

    Sigma_t = <sigma, S(W_hat)_{0,t}>,   W_hat_t = (t, W_t),

ou toute la dependance-chemin est portee par la signature ; sigma ne fait que la
ponderer. Le tenseur obtenu s'injecte tel quel dans la machinerie Fourier
(`fourier.signature_constant_vol`) apres conversion par `dict_to_array`.

Convention de l'alphabet a 2 lettres : 1 <-> dt, 2 <-> dW (Stratonovich).
"""

from functools import lru_cache

from signatures.tensor_algebra import generate_words
from signatures.tensor_dict import add_dicts, tensor_concat, tensor_inverse_dict


# =====================================================================
# CIR / Heston (Sec. 5.2)
# =====================================================================

def make_f(theta, kappa, eta, x):
    """Recurrence f_n(v) de la representation signature de sqrt(V) sous CIR (papier p.20-21).

    x = sqrt(V0). La recurrence deconcatene v par la droite :
      - lettre 1 (dt)  : A n / x^2 * f_{n-2}(v') - kappa/2 * n * f_n(v')
      - lettre 2 (dW)  : eta/2 * n / x * f_{n-1}(v')
    avec A = (kappa theta - eta^2/4) / 2.

    Cas de base : f_n(mot vide) = 1 pour TOUT n, y compris n <= 0 -- c'est le seul
    cas de base du papier. Pour n <= 0 et v non vide on renvoie 0, ce qui tronque
    les puissances negatives de x_t issues du terme de derive A/x_t. Cette
    troncature est la raison pour laquelle l'erreur de reconstruction de sqrt(V_t)
    plafonne vers 1.2e-2 au lieu de tendre vers 0 quand M croit (cf. notebook 09) ;
    elle reste sans effet notable sur le MSE de couverture.
    """
    A = 0.5 * (kappa * theta - eta**2 / 4)

    @lru_cache(maxsize=None)
    def f(n, v):
        if n <= 0:
            return 1.0 if v == () else 0.0
        if v == ():
            return 1.0

        last, previous = v[-1], v[:-1]
        if last == 1:
            return A * n / (x**2) * f(n - 2, previous) - kappa / 2 * n * f(n, previous)
        if last == 2:
            return eta / 2 * n / x * f(n - 1, previous)
        raise ValueError(f"lettre {last} invalide (attendu 1 ou 2)")

    return f


def compute_sigma_CIR(V0, kappa, theta, eta, M):
    """Tenseur sigma tel que sqrt(V_t) ~ <sigma, S(W_hat)_{0,t}> sous CIR (Heston).

    Renvoie un dict mot -> coefficient sur l'alphabet {1, 2}, mots de longueur 0..M.
    Le mot vide vaut sqrt(V0) (car f_1(vide) = 1) : c'est le niveau de vol initial.

    Verification usuelle (V0=0.0625, kappa=2, theta=0.0625, eta=0.5) :
    sigma[()] = sigma[(2,)] = 0.25, sigma[(1,)] = -0.125.
    """
    x = V0**0.5
    f = make_f(theta, kappa, eta, x)
    # generate_words inclut le mot vide : f(1, ()) = 1 donne bien sigma[()] = x.
    return {w: x * f(1, w) for w in generate_words(M, 2)}


# =====================================================================
# Modele a retard / delayed equation (Sec. 5.3)
# =====================================================================

def build_sigma_DE(sigma0, kappa, theta, eta, alpha, varsigma, lam, M):
    """Tenseur sigma du modele a retard, resolu exactement dans l'algebre tensorielle.

    Modele (sous Ito) :
        dSigma_t = kappa (theta - Sigma_t) dt + (eta + alpha Sigma_t + varsigma Z_t) dW_t
        dZ_t     = (Sigma_t - lam Z_t) dt,   Z_0 = 0.

    La signature d'un chemin interpole lineairement est un objet de calcul de
    STRATONOVICH : on convertit d'abord la derive (correction -1/2 d_Sigma b . b
    avec b = eta + alpha Sigma + varsigma Z, d_Sigma b = alpha), d'ou les
    constantes kappa theta - alpha eta/2 et kappa + alpha^2/2 ci-dessous.

    En integrant (integrer = concatener a droite) le systeme devient lineaire a
    droite, ce qui permet d'eliminer Z puis d'inverser :
        ell^Z  = ell^Sigma (x) e_1 (x) (vide + lam e_1)^{-1}
        sigma  = ell^Sigma = ell4 (x) (vide - ell)^{-1}
    Les deux inverses sont des sommes FINIES exactes dans T^M (cf.
    `tensor_inverse_dict`), pas des approximations : aucune condition de petitesse
    sur les parametres.
    """
    ell2 = {(1,): -alpha / 2, (2,): 1.0}
    # (vide + lam e_1)^{-1} = sum_k (-lam)^k e_1^{(x)k}  <->  noyau de memoire e^{-lam t}
    E = tensor_inverse_dict({(1,): -lam}, M)
    ell3 = {(1,): varsigma}
    q = tensor_concat(tensor_concat(ell3, E, M), ell2, M)

    # ell vit dans l'ideal d'augmentation (pas de mot vide) : indispensable pour l'inverse
    ell = add_dicts({(1,): -(kappa + alpha**2 / 2)}, {(2,): alpha}, q)
    inv = tensor_inverse_dict(ell, M)

    ell4 = {(): sigma0, (1,): kappa * theta - alpha * eta / 2, (2,): eta}
    return tensor_concat(ell4, inv, M)
