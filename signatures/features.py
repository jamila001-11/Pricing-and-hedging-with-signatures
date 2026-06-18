def compute_signatures(states, depth, stream=True):
    """
    Compute path signatures with iisignature.

    With stream=True, the result contains one signature per time step after the
    initial point, which is the format used by the SNN notebooks.
    """
    try:
        import iisignature
    except ImportError as exc:
        raise ImportError(
            "iisignature is required for compute_signatures. Install it with `pip install iisignature`."
        ) from exc

    return iisignature.sig(states, depth, 2 if stream else 0)
