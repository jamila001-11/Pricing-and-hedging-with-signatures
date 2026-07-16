"""Linearisation du payoff par regression (papier, eq. 3.8) -- Sec. 5.2 et 5.3.

Linear REG a besoin d'un tenseur xi tel que <xi, S^LL_T> ~ payoff. Pour le payoff
polynomial de la Sec. 5.1, xi est connu en FORME CLOSE (`linear_regressors.build_xi`).
Pour un call ou un asiatique, il n'y a pas de forme close : on l'apprend par moindres
carres sur la signature, ce que le papier appelle l'etape (3.8).

Le choix des mots regresses depend de la dependance-chemin du payoff :

- `fit_xi_power_words` -- payoff fonction du seul S_T (call europeen, Sec. 5.2).
  Les mots puissances (2,)^m suffisent, via <(2,)^m, S^LL> = (S_T - S0)^m / m!.

- `fit_xi_full_signature` -- payoff dependant de toute la trajectoire (asiatique
  geometrique, Sec. 5.3). La base des puissances du prix final ne peut pas capturer
  l'information temporelle : il faut les mots melant temps et prix ((1,2), (2,1,2)...).
  Sur les donnees du papier, regresser sur les 120 colonnes bat le sous-ensemble des
  mots de longueur <= 3 en train (R^2 0.965 vs 0.948) ET en test (MSE 1.03e-4 vs
  1.58e-4) : l'absence de degradation hors-echantillon exclut un surapprentissage
  severe (I=10 000 pour 120 features).
"""

import numpy as np

from signatures import word_to_index

from linear_hedging.linear_regressors import sig_words


def fit_xi_power_words(sig, y, M, d=3, price_letter=2):
    """xi par moindres carres sur les mots puissances du prix (price_letter,)^m, m = 0..M.

    `sig` (I, siglen) est la signature lead-lag tronquee a l'ordre M, `y` (I,) le payoff.
    Le mot vide fournit la constante. Renvoie le dict xi.
    """
    w2idx = word_to_index(sig_words(M, d))
    power_words = [(price_letter,) * m for m in range(M + 1)]

    cols = [np.ones(sig.shape[0])] + [sig[:, w2idx[w]] for w in power_words[1:]]
    coefs, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    return {w: coefs[m] for m, w in enumerate(power_words)}


def fit_xi_full_signature(sig, y, M, d=3):
    """xi par moindres carres sur TOUTES les colonnes de la signature (+ le mot vide).

    Adapte aux payoffs path-dependants, sans restriction a priori sur les mots
    pertinents (Lyons, Nejad & Arribas 2020). Renvoie le dict xi.
    """
    words = sig_words(M, d)
    A = np.column_stack([np.ones(sig.shape[0]), sig])
    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)

    xi = {(): coefs[0]}
    xi.update({w: coefs[i + 1] for i, w in enumerate(words)})
    return xi


def contract_xi(xi, sig, M, d=3):
    """<xi, S^LL> trajectoire par trajectoire : le payoff tel que xi le represente.

    Le mot vide porte la constante ; les autres mots lisent leur colonne de signature.
    """
    w2idx = word_to_index(sig_words(M, d))
    out = np.full(sig.shape[0], xi.get((), 0.0))
    for w, c in xi.items():
        if w == ():
            continue
        out = out + c * sig[:, w2idx[w]]
    return out


def xi_r2(xi, sig, y, M, d=3):
    """R^2 de l'approximation du payoff par xi (qualite de l'etape 3.8)."""
    approx = contract_xi(xi, sig, M, d)
    return 1 - np.sum((approx - y) ** 2) / np.sum((y - y.mean()) ** 2)
