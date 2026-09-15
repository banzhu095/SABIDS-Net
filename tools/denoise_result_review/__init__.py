"""Traceable denoising result discovery, packaging, and blinded ROI review."""

METHOD_ORDER = [
    "noisy_identity", "bm3d_standard", "tv_chambolle", "nlm", "ksvd_self",
    "dncnn_paired", "nafnet_paired", "sabids_current", "tcfl_dncnn",
]

DEEP_METHODS = {"dncnn_paired", "nafnet_paired", "sabids_current", "tcfl_dncnn"}
