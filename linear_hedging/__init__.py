from linear_hedging.black_scholes import bs_delta, bs_pnl, bs_price, pnl_stats
from linear_hedging.lead_lag import build_path_LL, hoff_lead_lag, signature_transform
from linear_hedging.linear_regressors import (
    build_reg_system,
    build_xi,
    expected_signature,
    feature_matrix,
    hedging_word_indices,
    hedging_words,
    linear_bcs,
    linear_reg_gradient_descent,
    reg_pnl,
    shuffle,
    sig_words,
    solve_reg,
    xi_coef,
)
from linear_hedging.payoff_xi import (
    contract_xi,
    fit_xi_full_signature,
    fit_xi_power_words,
    xi_r2,
)
