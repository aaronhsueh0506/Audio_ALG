"""DeepFilterNet conditioned for residual-echo plus noise reduction.

The base DFN architecture and fixed composition remain in the pure AINR model.
This AIAEC project variant adds only an error/far feature conditioner at the
network boundary and applies the predicted mask/filter to the linear-AEC error.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AINR.DeepFilterNet2.model import DeepFilterNet2
from AIAEC.aiaec_common import AecOutput, SignalGrid, require_complex_btf


class DeepFilterNetAENR(DeepFilterNet2):
    base_reference = "local DeepFilterNet2 cascade/alpha port"
    task = "linear_aec_postfilter_res_nr_dereverb"

    def __init__(self, grid: SignalGrid, n_erb: int = 32,
                 df_bins: int | None = None, df_order: int = 5,
                 mask_lookahead: int = 1, df_lookahead: int = 1,
                 **kwargs):
        if df_bins is None:
            df_bins = 96 if grid.sample_rate == 48000 else 64
        super().__init__(
            n_fft=grid.n_fft, sr=grid.sample_rate, n_erb=n_erb,
            df_bins=df_bins, df_order=df_order,
            mask_lookahead=mask_lookahead, df_lookahead=df_lookahead,
            **kwargs,
        )
        self.grid = grid
        self.erb_condition = nn.Conv2d(2, 1, 1)
        self.spec_condition = nn.Conv2d(4, 2, 1)
        self._init_error_passthrough()

    def _init_error_passthrough(self) -> None:
        """Start exactly as the pretrained-compatible single-input DFN."""
        with torch.no_grad():
            self.erb_condition.weight.zero_()
            self.erb_condition.bias.zero_()
            self.erb_condition.weight[0, 0, 0, 0] = 1.0
            self.spec_condition.weight.zero_()
            self.spec_condition.bias.zero_()
            self.spec_condition.weight[0, 0, 0, 0] = 1.0
            self.spec_condition.weight[1, 1, 0, 0] = 1.0

    def condition_features(self, error_erb: Tensor, error_spec: Tensor,
                           far_erb: Tensor, far_spec: Tensor):
        if error_erb.shape != far_erb.shape:
            raise ValueError("error/far ERB features must have identical shapes")
        if error_spec.shape != far_spec.shape:
            raise ValueError("error/far DF features must have identical shapes")
        if error_erb.ndim != 4 or error_erb.shape[1] != 1:
            raise ValueError("ERB features must be [B,1,T,n_erb]")
        if error_spec.ndim != 4 or error_spec.shape[1] != 2:
            raise ValueError("DF features must be [B,2,T,df_bins]")
        erb = self.erb_condition(torch.cat((error_erb, far_erb), dim=1))
        spec = self.spec_condition(torch.cat((error_spec, far_spec), dim=1))
        return erb, spec

    def forward(self, linear_error: Tensor, error_erb: Tensor,
                error_spec: Tensor, far_erb: Tensor,
                far_spec: Tensor) -> AecOutput:
        require_complex_btf(linear_error, "linear_error")
        if linear_error.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")
        feat_erb, feat_spec = self.condition_features(
            error_erb, error_spec, far_erb, far_spec,
        )
        erb_mask, coefs, alpha = self.heads(feat_erb, feat_spec)
        # Pure DFN uses [B,F,T]; AIAEC's public contract is [B,T,F].
        spec_bft = linear_error.transpose(1, 2)
        enhanced = self.compose(
            spec_bft, erb_mask, coefs, alpha
        ).transpose(1, 2)
        return AecOutput(
            enhanced=enhanced, mask=erb_mask,
            auxiliary={"deep_filter_coefficients": coefs,
                       "deep_filter_alpha": alpha,
                       "conditioned_erb": feat_erb,
                       "conditioned_spec": feat_spec},
        )
