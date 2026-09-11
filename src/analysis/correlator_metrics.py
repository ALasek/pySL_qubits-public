import numpy as np


CORRELATOR_METRIC_NAMES = (
    "corr_spread_radius",
    "corr_offdiag_fraction",
    "corr_total_connected_weight",
    "corr_nn_weight",
    "corr_nn_fraction",
    "corr_commutator_spread_radius",
    "corr_total_commutator_weight",
)


def connected_correlator_metrics(C_connected, source_indices=None, eps=1e-30):
    C = np.asarray(C_connected)
    if C.ndim != 3:
        raise ValueError(f"C_connected must have shape [time, nE, nSources], got {C.shape}")

    n_env = C.shape[1]
    if source_indices is None:
        if C.shape[2] != n_env:
            raise ValueError("source_indices are required for a rectangular correlator array")
        source_indices = np.arange(n_env)
    source_indices = np.asarray(source_indices, dtype=int)
    if source_indices.shape != (C.shape[2],):
        raise ValueError(f"source_indices must have shape {(C.shape[2],)}, got {source_indices.shape}")
    distances = np.abs(np.arange(n_env)[:, None] - source_indices[None, :])
    weights = np.abs(C) ** 2
    commutator_weights = (2 * np.imag(C)) ** 2

    total = np.sum(weights, axis=(1, 2))
    distance_sq_weight = np.sum(weights * distances**2, axis=(1, 2))
    offdiag_weight = np.sum(weights * (distances > 0), axis=(1, 2))
    nn_weight = np.sum(weights * (distances == 1), axis=(1, 2))
    commutator_total = np.sum(commutator_weights, axis=(1, 2))
    commutator_distance_sq = np.sum(commutator_weights * distances**2, axis=(1, 2))

    radius = np.zeros_like(total, dtype=np.float64)
    offdiag_fraction = np.zeros_like(total, dtype=np.float64)
    nn_fraction = np.zeros_like(total, dtype=np.float64)
    commutator_radius = np.zeros_like(total, dtype=np.float64)
    valid_rows = np.all(np.isfinite(C), axis=(1, 2))

    has_total = total > eps
    radius[has_total] = np.sqrt(distance_sq_weight[has_total] / total[has_total])
    offdiag_fraction[has_total] = offdiag_weight[has_total] / total[has_total]

    has_offdiag = offdiag_weight > eps
    nn_fraction[has_offdiag] = nn_weight[has_offdiag] / offdiag_weight[has_offdiag]

    has_commutator = commutator_total > eps
    commutator_radius[has_commutator] = np.sqrt(
        commutator_distance_sq[has_commutator] / commutator_total[has_commutator]
    )

    for values in (radius, offdiag_fraction, nn_fraction, commutator_radius):
        values[~valid_rows] = np.nan
    commutator_total[~valid_rows] = np.nan

    return {
        "corr_spread_radius": radius,
        "corr_offdiag_fraction": offdiag_fraction,
        "corr_total_connected_weight": total.astype(np.float64),
        "corr_nn_weight": nn_weight.astype(np.float64),
        "corr_nn_fraction": nn_fraction,
        "corr_commutator_spread_radius": commutator_radius,
        "corr_total_commutator_weight": commutator_total.astype(np.float64),
    }
