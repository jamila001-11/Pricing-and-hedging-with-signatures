import numpy as np

def normalize_psi(psi, words, word_idx, dt):
    """
    Normalise psi en multipliant le coeff du mot w par dt^(|w|/2),
    ou dt = T/J est le pas de temps de la simulation.
    Reduit la norme de O(epsilon^{-(H+4.5)}) a O(1).
    Propriete cle : <normalize_psi(psi), normalize_sig(sig)> = <psi, sig>.
    """
    psi_norm = psi.copy()
    for w in words:
        k = len(w)
        if k > 0:
            psi_norm[word_idx[w]] *= dt ** (k / 2)
    return psi_norm


def normalize_sig(sig, words, word_idx, dt):
    """
    Normalise la signature en divisant le coeff du mot w par dt^(|w|/2).
    Complementaire a normalize_psi : <normalize_psi(psi), normalize_sig(sig)> = <psi, sig>.
    """
    sig_norm = sig.copy()
    for w in words:
        k = len(w)
        if k > 0:
            sig_norm[word_idx[w]] /= dt ** (k / 2)
    return sig_norm