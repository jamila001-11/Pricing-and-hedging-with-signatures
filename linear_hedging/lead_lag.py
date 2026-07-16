import numpy as np


def hoff_lead_lag(S, T):
    """Transformee lead-lag de Hoff (Flint, Hambly, Lyons 2016, Def. 2.1).

    S : (I, J+1) trajectoires echantillonnees, colonnes = temps t_0..t_J.
    Renvoie tau (2J+1,), S_lag (I, 2J+1), S_lead (I, 2J+1).

    Le chemin part et termine sur la diagonale : (x_0,x_0), (x_0,x_1),
    (x_1,x_1), ..., (x_{J-1},x_J), (x_J,x_J). Les coins diagonaux (x_j,x_j)
    portent la variation quadratique et NE sont PAS des redondances : on les garde.
    """
    I, Jp1 = S.shape
    J = Jp1 - 1
    t = np.linspace(0.0, T, Jp1)

    lag_idx, lead_idx = [0], [0]
    for j in range(J):                # j = 0 .. J-1
        lead_idx += [j + 1, j + 1]
        lag_idx += [j, j + 1]
    lag_idx, lead_idx = np.array(lag_idx), np.array(lead_idx)

    S_lag = S[:, lag_idx]
    S_lead = S[:, lead_idx]
    tau = t[lag_idx]                  # un seul canal de temps (lag)
    return tau, S_lag, S_lead


def build_path_LL(S, T):
    """Chemin a 3 canaux X = (tau, S_lag, S_lead) de forme (I, 2J+1, 3).

    Le canal t^lead est volontairement omis (canal inerte pour la couverture
    previsible) : un seul canal de temps tau = t^lag suffit. cf. l'analyse du
    notebook source (mots de hedge dans {tau, ell}^* . L).
    """
    tau, S_lag, S_lead = hoff_lead_lag(S, T)
    return np.stack([np.broadcast_to(tau, S_lag.shape), S_lag, S_lead], axis=-1)


def signature_transform(path_LL, depth, fmt=0):
    """Signature tronquee a l'ordre `depth` du chemin lead-lag (via iisignature).

    `fmt` est passe tel quel a iisignature.sig (0 = signature terminale,
    2 = mode stream). Les colonnes de la sortie correspondent aux mots de
    longueur 1..depth dans l'ordre de `signatures.generate_words(depth, d)[1:]`.
    """
    try:
        import iisignature
    except ImportError as exc:
        raise ImportError(
            "iisignature is required for signature_transform. "
            "Install it with `pip install iisignature`."
        ) from exc

    return iisignature.sig(path_LL, depth, fmt)
