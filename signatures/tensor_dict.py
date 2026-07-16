"""Algebre tensorielle en representation creuse (dict mot -> coefficient).

Complement de `tensor_algebra` (representation dense indexee par word_idx) : la
representation creuse est adaptee aux ordres eleves, ou seuls quelques mots
portent un coefficient non nul (E[S^LL] a l'ordre 8, series de Neumann des
sections 5.2-5.3, tenseurs ell^v de l'Algorithme 2).
"""

from itertools import product
from math import factorial

import numpy as np


def words_of_length(n, d=2):
    """Tous les mots de longueur exactement n sur l'alphabet {1,...,d}."""
    if n == 0:
        return [()]
    return list(product(range(1, d + 1), repeat=n))


# =====================================================================
# Produits de mots
# =====================================================================

_shuffle_memo = {}


def shuffle(u, v):
    """Produit shuffle de deux mots (recurrence par deconcatenation a droite).

    Renvoie un dict mot -> coefficient. Le resultat est memoise : ne pas le muter.
    """
    if not u:
        return {v: 1}
    if not v:
        return {u: 1}
    if (u, v) in _shuffle_memo:
        return _shuffle_memo[(u, v)]
    res = {}
    a, b = u[-1], v[-1]
    for w, c in shuffle(u[:-1], v).items():
        res[w + (a,)] = res.get(w + (a,), 0) + c
    for w, c in shuffle(u, v[:-1]).items():
        res[w + (b,)] = res.get(w + (b,), 0) + c
    _shuffle_memo[(u, v)] = res
    return res


def half_shuffle(u, v):
    """Half-shuffle u > v (papier, Def. 2.2) : shuffle(u, v[:-1]) puis concatenation de v[-1].

    u > mot_vide = 0, represente par le dictionnaire vide (tenseur nul).
    """
    if not v:
        return {}
    w, j = v[:-1], v[-1]
    return {mot + (j,): coef for mot, coef in shuffle(u, w).items()}


# =====================================================================
# Produits bilineaires de tenseurs (tronques a l'ordre M)
# =====================================================================

def shuffle_dict(d1, d2, M):
    """Produit shuffle de deux tenseurs creux, tronque a l'ordre M.

    Encode le produit ponctuel : <a shuffle b, S_t> = <a, S_t> * <b, S_t>.
    """
    result = {}
    for u, c1 in d1.items():
        for v, c2 in d2.items():
            if len(u) + len(v) > M:
                continue
            for mot, coef in shuffle(u, v).items():
                result[mot] = result.get(mot, 0) + c1 * c2 * coef
    return result


def half_shuffle_dict(d1, d2, M):
    """Half-shuffle de deux tenseurs creux, tronque a l'ordre M."""
    result = {}
    for u, c1 in d1.items():
        for v, c2 in d2.items():
            if len(u) + len(v) > M:
                continue
            for mot, coef in half_shuffle(u, v).items():
                result[mot] = result.get(mot, 0) + c1 * c2 * coef
    return result


def tensor_concat(d1, d2, M):
    """Produit tensoriel (concatenation des mots), tronque a l'ordre M.

    Encode l'integration : <ell (x) e_i, S(X)_{0,t}> = int_0^t <ell, S_{0,s}> dX^i_s.
    A ne pas confondre avec `shuffle_dict` (produit ponctuel).
    """
    result = {}
    for u, c1 in d1.items():
        for v, c2 in d2.items():
            if len(u) + len(v) > M:
                continue
            result[u + v] = result.get(u + v, 0) + c1 * c2
    return result


def append_letter_dict(ell, letter):
    """Concatene `letter` a droite de chaque mot du tenseur (= integration contre dX^letter)."""
    return {mot + (letter,): coef for mot, coef in ell.items()}


def add_dicts(*dicts, signs=None):
    """Somme terme a terme de tenseurs creux ; `signs` donne un +1/-1 par tenseur."""
    if signs is None:
        signs = [1] * len(dicts)
    result = {}
    for d, s in zip(dicts, signs):
        for w, c in d.items():
            result[w] = result.get(w, 0) + s * c
    return result


# =====================================================================
# Series : exponentielles et inverse tensoriel
# =====================================================================

def exp_tensor_dict(ell, M):
    """Exponentielle pour le produit de concatenation : sum_{n=0}^{M} ell^{(x)n} / n!."""
    result = {(): 1.0}
    power = {(): 1.0}
    for n in range(1, M + 1):
        power = tensor_concat(power, ell, M)
        for mot, coef in power.items():
            result[mot] = result.get(mot, 0) + coef / factorial(n)
    return result


def exp_shuffle_dict(ell, M):
    """Exponentielle pour le produit shuffle : sum_{n=0}^{M} ell^{shuffle n} / n!.

    Verification usuelle : <exp_shuffle(c e_1), S(W_hat)_{0,t}> = exp(c t).
    """
    result = {(): 1.0}
    power = {(): 1.0}
    for n in range(1, M + 1):
        power = shuffle_dict(power, ell, M)
        for mot, coef in power.items():
            result[mot] = result.get(mot, 0) + coef / factorial(n)
    return result


def tensor_inverse_dict(ell, M):
    """Serie de Neumann (mot_vide - ell)^{-1} = sum_{n=0}^{M} ell^{(x)n}.

    ATTENTION : passer `ell`, jamais `mot_vide - ell`. L'argument doit vivre dans
    l'ideal d'augmentation (coefficient du mot vide nul) ; c'est ce qui rend la
    somme finie et exacte dans l'algebre tronquee T^M (nilpotence graduee :
    tout mot de ell^{(x)n} a une longueur >= n, donc m^{M+1} = {0}).

    Si l'argument contient le mot vide avec coefficient 1, chaque puissance y
    contribue 1 et on obtient <mot_vide, inv> = M+1 au lieu de 1 : le tenseur est
    alors faux, et l'erreur croit avec M.
    """
    result = {(): 1.0}
    power = {(): 1.0}
    for n in range(1, M + 1):
        power = tensor_concat(power, ell, M)
        for mot, coef in power.items():
            result[mot] = result.get(mot, 0) + coef
    return result


# =====================================================================
# Contraction et conversion vers la representation dense
# =====================================================================

def contract_with_Esig(tensor_dict, E_sig):
    """<tenseur, E_sig> : somme des coef * E_sig[mot] (mots absents traites comme nuls)."""
    return sum(c * E_sig.get(w, 0.0) for w, c in tensor_dict.items())


def dict_to_array(tensor_dict, words, word_idx, dtype=complex):
    """Convertit un tenseur creux en tableau dense indexe par `word_idx`.

    Format attendu par `tensor_algebra` / `signature_volatility`. Les mots de
    `words` absents du dictionnaire valent 0.
    """
    arr = np.zeros(len(words), dtype=dtype)
    for w, c in tensor_dict.items():
        if w in word_idx:
            arr[word_idx[w]] = c
    return arr
