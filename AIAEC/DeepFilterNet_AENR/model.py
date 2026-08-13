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
from AIAEC.aiaec_streaming import (
    DelayRingCell,
    StreamConv2dCell,
    StreamGRUCell,
    StreamModuleCell,
    assert_streaming_ready,
)


def _padded_conv_cell(module: nn.Module) -> StreamModuleCell:
    """Streaming cell for a block laid out as ConstantPad2d -> Conv2d -> rest.

    Covers both time-extended conv layouts in this model: LookaheadConv2d
    (erb_conv0 / df_conv0 -- the pad carries the temporal lookahead AND the
    frequency padding) and SeparableConv2d with a temporal kernel (df_convp --
    the pad is left-only and the conv pads frequency itself).  The pad
    geometry is read back from the module so the cell cannot drift from the
    offline padding.
    """
    children = list(module.children())
    pad, conv = children[0], children[1]
    if not isinstance(pad, nn.ConstantPad2d) or not isinstance(conv, nn.Conv2d):
        raise TypeError("expected ConstantPad2d -> Conv2d at the block head")
    freq_left, freq_right, time_left, lookahead = pad.padding
    kt = conv.kernel_size[0]
    dt = conv.dilation[0]
    if time_left != (kt - 1) * dt - lookahead:
        raise ValueError("pad geometry disagrees with the conv time kernel")
    cell = StreamConv2dCell(conv, kt, dt, freq_left, freq_right,
                            lookahead=lookahead)
    return StreamModuleCell(cell, children[2:])


def _module_cell_step(cell: StreamModuleCell, x: Tensor) -> Tensor:
    """StreamModuleCell.step that tolerates the T=0 lookahead warm-up output.

    Conv2d rejects zero-size spatial inputs, so the per-frame tail (pointwise
    conv / norm / act) must be skipped while the conv cell is still buffering
    its lookahead frame; the conv history itself must still advance.
    """
    y = cell.conv_cell.step(x)
    if y.shape[2] == 0:
        return y
    for module in cell.per_frame:
        y = module(y)
    return y


def _squeezed_gru_step(module, gru_cell: StreamGRUCell, x: Tensor) -> Tensor:
    """One-frame replay of SqueezedGRU_S.forward with an external hidden."""
    skip_in = x
    x = module.linear_in(x)
    y = gru_cell.step(x)
    y = module.linear_out(y)
    if module.gru_skip is not None:
        y = y + module.gru_skip(skip_in)
    return y


class DeepFilterNetAENR(DeepFilterNet2):
    base_reference = "local DeepFilterNet2 cascade/alpha port"
    task = "linear_aec_postfilter_res_nr_dereverb"
    # Streaming latency in STFT hops.  The DFN2 cascade is serial: the deep
    # filter's future tap consumes the MASKED spectrum, so the mask path's own
    # lookahead adds to the DF lookahead (see dfn2_process.h,
    # dfn2_compose_stream): shipped 1 + 1 = 2 hops head-to-audio.
    stream_output_delay = 2

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

    # ------------------------------------------------------------------
    # frame-by-frame streaming (deployment reference; offline forward is
    # untouched -- everything below replays its arithmetic one hop at a time)
    # ------------------------------------------------------------------

    def create_stream_state(self) -> dict:
        """Named cells + ring/queue state consumed by forward_stream.

        The dict is the deployment RAM contract (dfn2_process.h is the C twin
        of the compose rings): two lookahead-conv histories, three GRU
        hiddens, the df_convp history, the masked-spec deep-filter ring, the
        raw spectra waiting for their one-frame-late mask, and the head queue
        realising the deep-filter lookahead.
        """
        assert_streaming_ready(self)
        if self.mask_lookahead + self.df_lookahead != self.stream_output_delay:
            raise ValueError(
                f"stream_output_delay={self.stream_output_delay} describes the "
                "shipped mask_lookahead=1/df_lookahead=1 cascade; this model "
                f"was built with {self.mask_lookahead}/{self.df_lookahead}"
            )
        return {
            "erb_conv0": _padded_conv_cell(self.encoder.erb_conv0),
            "df_conv0": _padded_conv_cell(self.encoder.df_conv0),
            "encoder_gru": StreamGRUCell(self.encoder.emb_gru.gru),
            "erb_dec_gru": StreamGRUCell(self.erb_dec.emb_gru.gru),
            "df_gru": StreamGRUCell(self.df_dec.df_gru.gru),
            "df_convp": _padded_conv_cell(self.df_dec.df_convp),
            "df_ring": DelayRingCell(self.df_order),
            # Raw input spectra whose heads have not emerged yet (depth =
            # mask_lookahead between steps).
            "pending_spec": [],
            # Masked frames whose deep-filter future frame is still missing
            # (depth = df_lookahead between steps).
            "df_queue": [],
        }

    def forward_stream(self, linear_error: Tensor, error_erb: Tensor,
                       error_spec: Tensor, far_erb: Tensor,
                       far_spec: Tensor, state: dict) -> AecOutput:
        """One new STFT frame through the two-hop-delayed DFN2 cascade.

        Every input is the T=1 slice of its offline tensor.  The returned
        ``enhanced`` frame is for input frame ``t - stream_output_delay``; the
        first ``stream_output_delay`` calls return an empty (T=0) AecOutput
        and ``flush_stream`` supplies the final two frames, replaying the
        offline right-side zero padding.  ``mask`` and the deep-filter
        auxiliaries are emitted aligned WITH the enhanced frame (same source
        frame, same delay).  ``conditioned_erb``/``conditioned_spec`` are
        omitted: they belong to the undelayed input step and would misalign
        with everything else in the output.
        """
        require_complex_btf(linear_error, "linear_error")
        if linear_error.shape[1] != 1:
            raise ValueError("forward_stream consumes exactly one frame per call")
        if linear_error.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")
        feat_erb, feat_spec = self.condition_features(
            error_erb, error_spec, far_erb, far_spec,
        )
        if "feat_zero" not in state:
            # The offline lookahead pad sits AFTER the conditioner (inside
            # LookaheadConv2d), so flush must inject zeros at this boundary --
            # a zero wav frame would still carry the conditioner bias.
            state["feat_zero"] = (torch.zeros_like(feat_erb),
                                  torch.zeros_like(feat_spec))
        state["pending_spec"].append(linear_error)
        heads = self._stream_heads(state, feat_erb, feat_spec)
        if heads is None:
            return self._stream_empty(linear_error)
        emitted = self._stream_mask_and_queue(state, heads)
        if emitted is None:
            return self._stream_empty(linear_error)
        return emitted

    def flush_stream(self, state: dict) -> AecOutput:
        """Emit the last ``stream_output_delay`` frames of the stream.

        Zero conditioned-feature frames feed the lookahead convs until every
        pending spectrum has its heads (the offline right pad), then zero
        masked-spec frames drain the deep-filter queue (the offline DF right
        pad).  Returns the remaining frames concatenated (2 for any stream
        with at least 2 input frames).
        """
        if "feat_zero" not in state:
            raise ValueError("flush_stream on a stream that never saw a frame")
        zero_erb, zero_spec = state["feat_zero"]
        outputs = []
        budget = 2 * (self.mask_lookahead + 1)
        while state["pending_spec"]:
            if budget == 0:
                raise RuntimeError("stream flush failed to drain pending frames")
            budget -= 1
            heads = self._stream_heads(state, zero_erb, zero_spec)
            if heads is None:
                continue
            emitted = self._stream_mask_and_queue(state, heads)
            if emitted is not None:
                outputs.append(emitted)
        ring_state = state["df_ring"].state_tensors()["ring"]
        zero_low = ring_state.new_zeros(ring_state.shape[0], 1, self.df_bins)
        while state["df_queue"]:
            ring = state["df_ring"].step(zero_low)
            outputs.append(self._stream_emit(state, ring[:, 0, 0]))
        if not outputs:
            raise RuntimeError("flush produced no frames; corrupt stream state")
        if len(outputs) == 1:
            return outputs[0]
        return AecOutput(
            enhanced=torch.cat([o.enhanced for o in outputs], dim=1),
            mask=torch.cat([o.mask for o in outputs], dim=2),
            auxiliary={
                key: torch.cat([o.auxiliary[key] for o in outputs], dim=1)
                for key in outputs[0].auxiliary
            },
        )

    def _stream_heads(self, state: dict, feat_erb: Tensor, feat_spec: Tensor):
        """Push one conditioned frame; heads emerge ``mask_lookahead`` late."""
        enc = self.encoder
        e0 = _module_cell_step(state["erb_conv0"], feat_erb)
        c0 = _module_cell_step(state["df_conv0"], feat_spec)
        if e0.shape[2] == 0:
            return None                       # encoder lookahead warm-up
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)
        c1 = enc.df_conv1(c0)
        b = e3.shape[0]
        e3_flat = e3.permute(0, 2, 3, 1).reshape(b, 1, -1)
        c1_flat = c1.permute(0, 2, 3, 1).reshape(b, 1, -1)
        df_emb = enc.df_fc_emb(c1_flat)
        if enc.enc_concat:
            emb = torch.cat([e3_flat, df_emb], dim=-1)
        else:
            emb = e3_flat + df_emb
        emb = _squeezed_gru_step(enc.emb_gru, state["encoder_gru"], emb)

        dec = self.erb_dec
        x = _squeezed_gru_step(dec.emb_gru, state["erb_dec_gru"], emb)
        x = x.reshape(b, 1, dec.n_erb_4, dec.enc_ch).permute(0, 3, 1, 2)
        x = dec.convt3(dec.conv3p(e3) + x)
        x = dec.convt2(dec.conv2p(e2) + x)
        x = dec.convt1(dec.conv1p(e1) + x)
        erb_mask = dec.conv0_out(dec.conv0p(e0) + x)

        dfd = self.df_dec
        c = _squeezed_gru_step(dfd.df_gru, state["df_gru"], emb)
        if dfd.df_skip is not None:
            c = c + dfd.df_skip(emb)
        c0_res = state["df_convp"].step(c0).permute(0, 2, 3, 1)
        alpha = dfd.df_fc_a(c)
        coefs = dfd.df_out(c).view(b, 1, dfd.df_bins, dfd.df_order * 2) + c0_res
        return erb_mask, coefs, alpha

    def _stream_mask_and_queue(self, state: dict, heads):
        """compose() stage 1 for the oldest pending frame, then try to emit."""
        erb_mask, coefs, alpha = heads
        spec_btf = state["pending_spec"].pop(0)
        spec_bft = spec_btf.transpose(1, 2)                     # [B,F,1]
        bin_mask = erb_mask.squeeze(1).matmul(self.erb_inv).permute(0, 2, 1)
        spec_m = spec_bft * bin_mask                            # [B,n_bins,1]
        ring = state["df_ring"].step(
            spec_m[:, : self.df_bins, 0].unsqueeze(1))          # [B,1,1,D,F]
        state["df_queue"].append({
            "spec_raw": spec_bft[:, :, 0],
            "spec_m": spec_m[:, :, 0],
            "mask": erb_mask,
            "coefs": coefs,
            "alpha": alpha,
        })
        if len(state["df_queue"]) <= self.df_lookahead:
            return None                                         # DF warm-up
        return self._stream_emit(state, ring[:, 0, 0])

    def _stream_emit(self, state: dict, ring: Tensor) -> AecOutput:
        """compose() stage 2 for the queue front, whose DF window is complete.

        ``ring`` is [B, D, df_bins] with slot d holding masked frame s-d; the
        emitted frame is e = s - df_lookahead, so FIR tap j (offline window
        position e - history + j) lives in slot d = df_order-1-j.
        """
        entry = state["df_queue"].pop(0)
        order = self.df_order
        window = ring.flip(1)                                   # [B,O,F], j-major
        coefs = entry["coefs"].view(-1, self.df_bins, order, 2)
        win_re = window.real.permute(0, 2, 1)                   # [B,F,O]
        win_im = window.imag.permute(0, 2, 1)
        df_re = (win_re * coefs[..., 0] - win_im * coefs[..., 1]).sum(-1)
        df_im = (win_im * coefs[..., 0] + win_re * coefs[..., 1]).sum(-1)
        spec_df = torch.complex(df_re, df_im)                   # [B,df_bins]
        alpha = entry["alpha"][:, 0]                            # [B,1]
        low = alpha * spec_df + (1 - alpha) * entry["spec_m"][:, : self.df_bins]
        spec_e = entry["spec_m"].clone()
        spec_e[:, : self.df_bins] = low
        spec_e = self.post_filter(entry["spec_raw"].unsqueeze(-1),
                                  spec_e.unsqueeze(-1))         # [B,n_bins,1]
        return AecOutput(
            enhanced=spec_e.transpose(1, 2),
            mask=entry["mask"],
            auxiliary={"deep_filter_coefficients": entry["coefs"],
                       "deep_filter_alpha": entry["alpha"]},
        )

    def _stream_empty(self, linear_error: Tensor) -> AecOutput:
        """Warm-up output: zero frames, shape-consistent for concatenation."""
        b = linear_error.shape[0]
        real = linear_error.real
        return AecOutput(
            enhanced=linear_error[:, :0],
            mask=real.new_zeros(b, 1, 0, self.n_erb),
            auxiliary={
                "deep_filter_coefficients": real.new_zeros(
                    b, 0, self.df_bins, self.df_order * 2),
                "deep_filter_alpha": real.new_zeros(b, 0, 1),
            },
        )
