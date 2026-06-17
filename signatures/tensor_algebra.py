import numpy as np
from itertools import product, combinations
import math

def generate_words(M, d=2):
    """Génère tous les mots (tuples) jusqu'à la taille M pour un alphabet de taille d."""
    words = []
    for i in range(M + 1):
        words += list(product(range(1, d + 1), repeat=i))
    return words

def word_to_index(words):
    """Crée un dictionnaire pour mapper un mot à son index dans le vecteur."""
    return {w: i for i, w in enumerate(words)}

def zeros_tensor(M):
    """Initialise un tenseur de zéros complexes pour le degré M."""
    words = generate_words(M)
    return np.zeros(len(words), dtype=complex)

def get_shuffles(v, w):
    """Calcule le produit shuffle (mélange complet) de deux mots."""
    n, m = len(v), len(w)
    result = []  # list au lieu de set
    for positions in combinations(range(n+m), n):
        shuffle = [None] * (n+m)
        for i, p in enumerate(positions):
            shuffle[p] = v[i]
        j = 0
        for k in range(n+m):
            if shuffle[k] is None:
                shuffle[k] = w[j]
                j += 1
        result.append(tuple(shuffle))  # append au lieu de add
    return result

def shuffle_product(l1, l2, words, word_idx, M):
    """Remplit le tenseur 'result' avec le produit shuffle de l1 et l2."""
    result = np.zeros(len(words), dtype=complex)
    for i, v in enumerate(words):
        if l1[i] == 0:
            continue
        for j, w in enumerate(words):
            if l2[j] == 0:
                continue
            if len(v) + len(w) > M:
                continue
            for shuffle in get_shuffles(v, w):
                if shuffle in word_idx:
                    result[word_idx[shuffle]] += l1[i] * l2[j]
    return result

def shuffle_exp(l, words, word_idx, M):
    """Calcule l'exponentielle de shuffle pour un mot donné jusqu'à l'ordre M."""
    result = zeros_tensor(M)
    result[word_idx[()]] = 1.0  # terme n=0 : mot vide avec coeff 1
    
    power = zeros_tensor(M)
    power[word_idx[()]] = 1.0  # ell^{shuffle 0} = mot vide
    
    for n in range(1, M+1):
        power = shuffle_product(power, l, words, word_idx, M)  # power = ell^{shuffle n}
        result += power / math.factorial(n)
    
    return result

def projection( psi , letter , words , word_idx , M): 
    """Extrait la coordonnée d'un mot spécifique dans un tenseur."""
    psi_p = zeros_tensor(M)
    for word in words :
        if len(word) == 0:
            continue
        if word[-1] == letter :
            psi_p[word_idx[word[:-1]]] = psi[word_idx[word]]
    return psi_p   