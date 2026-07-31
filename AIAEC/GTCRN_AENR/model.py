"""GTCRN conditioned on a linear-AEC error/reference pair.

This is a project AENR variant, not a model claimed by the GTCRN paper.  Every
GTCRN block after the first convolution is reused unchanged from the audited
standalone AINR implementation; only the input contract grows from one complex
spectrum to two.
"""

from __future__ import annotations

import torch
from torch import Tensor

from AINR.GTCRN.model import ConvBlock, GTCRN
from AIAEC.aiaec_common import AecOutput, SignalGrid, require_complex_btf


class GTCRNAENR(GTCRN):
    base_reference = "Xiaobin Rong et al., GTCRN (INTERSPEECH 2024)"
    task = "linear_aec_postfilter_res_nr"

    def __init__(self, grid: SignalGrid, erb_subband_1: int = 65,
                 erb_subband_2: int = 64):
        if (grid.sample_rate, grid.n_fft) != (16000, 512):
            raise ValueError(
                "GTCRN-AENR follows the upstream 16 kHz/512 grid; use the "
                "DeepFilterNet-AENR candidate for the 48 kHz route"
            )
        super().__init__(erb_subband_1, erb_subband_2,
                         nfft=grid.n_fft, fs=grid.sample_rate)
        self.grid = grid
        # One source produces [mag,re,im] then SFE(k=3) => 9 channels.
        # Error + far reference therefore produce 18. All later GTCRN layers,
        # DPGRNN widths, decoder and CRM arithmetic are unchanged.
        self.encoder.en_convs[0] = ConvBlock(
            18, 16, (1, 5), stride=(1, 2), padding=(0, 2),
        )

    @staticmethod
    def _three_features(spec: Tensor) -> Tensor:
        real = spec.real
        imag = spec.imag
        mag = (real.square() + imag.square() + 1e-12).sqrt()
        return torch.stack((mag, real, imag), dim=1)

    def forward(self, linear_error: Tensor, far_end: Tensor) -> AecOutput:
        require_complex_btf(linear_error, "linear_error")
        require_complex_btf(far_end, "far_end")
        if linear_error.shape != far_end.shape:
            raise ValueError("linear_error and far_end STFT grids must match")
        if linear_error.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")

        err = self.sfe(self.erb.bm(self._three_features(linear_error)))
        far = self.sfe(self.erb.bm(self._three_features(far_end)))
        feat, skips = self.encoder(torch.cat((err, far), dim=1))
        feat = self.dpgrnn2(self.dpgrnn1(feat))
        mask_erb = self.decoder(feat, skips)
        mask_full = self.erb.bs(mask_erb)
        enhanced_ri = self.mask(
            mask_full,
            torch.stack((linear_error.real, linear_error.imag), dim=1),
        )
        enhanced = torch.complex(enhanced_ri[:, 0], enhanced_ri[:, 1])
        mask = torch.complex(mask_full[:, 0], mask_full[:, 1])
        return AecOutput(enhanced=enhanced, mask=mask,
                         auxiliary={"erb_complex_mask": mask_erb})
