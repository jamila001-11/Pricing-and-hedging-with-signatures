from signatures.bergomi_kernel import compute_sigma_t
from signatures.calibration import (
    build_ell,
    calibrate_sigma,
    compute_loss,
    E_W_hat_all_times,
    fawcett_E_W_hat,
    make_loss_fn,
    numerical_gradient,
    sigma_dict_to_vec,
    sigma_words,
    vec_to_sigma_dict,
)
from signatures.features import compute_signatures
from signatures.normalization import normalize_psi, normalize_sig
from signatures.tensor_algebra import (
    generate_words,
    get_shuffles,
    projection,
    shuffle_exp,
    shuffle_product,
    word_to_index,
    zeros_tensor,
)
from signatures.tensor_dict import (
    add_dicts,
    append_letter_dict,
    contract_with_Esig,
    dict_to_array,
    exp_shuffle_dict,
    exp_tensor_dict,
    half_shuffle,
    half_shuffle_dict,
    shuffle,
    shuffle_dict,
    tensor_concat,
    tensor_inverse_dict,
    words_of_length,
)
from signatures.volatility_models import build_sigma_DE, compute_sigma_CIR, make_f
