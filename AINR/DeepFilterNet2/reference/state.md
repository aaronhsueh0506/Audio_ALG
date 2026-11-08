# State boundary

State layout version 8. Both state tensors are rank 4 with the same shape at
the input and at the `*_out` output, bound to one address; the metadata records
`state_binding = in_place_same_address`, `state_handoff` and `state_slicing`.
Versions 4, 5, 6 and 7 are retired (`RETIRED_LAYOUT_VERSIONS`); the contract
test asserts `DFN2_MODEL_IO_LAYOUT_VERSION` is not one of them.

| tensor | shape | meaning |
| --- | --- | --- |
| `h_gru` | `(1,5,1,256)` | encoder row 0, ERB decoder rows 1-2, DF decoder rows 3-4 |
| `df_convp_history` | `(1,64,5,96)` | full causal window of the DF residual convolution |

The library knows none of these shapes. The application registers buffers of these element counts (1,280 and 30,720 floats on the
shipped grid) or, on a board, its accelerator holds the state (see [C deployment boundary](../README.md#c-deployment-boundary)).

Each GRU reads its rows through a Slice of the graph input on dim 1 followed by
a squeeze, and the next `h_gru` is one Concat on dim 1. ONNX has no stacked GRU
op, so a two-layer stack still becomes two GRU nodes: five GRU nodes in all.
The three GRUs therefore share one input tensor and one quantization scale;
`inference.py calib` reports under `gru_state_slices` what that costs each GRU
(bits lost under a symmetric max-abs int8 scale). The window input has exactly
one consumer, a Slice of axis 2 with the explicit range `[1:5]` and step 1; no
state output is produced by a Slice. `../onnx_streaming_contract.py`
(`validate_inplace_state_graph`) checks this at export time, and
`../check_state_quantization.py` repeats it on an exported graph and, given a
quantization report, aborts when a state input and its output differ in dtype,
scale, zero point or shape. A window row is 96 elements, so on an int8 target
the axis-2 Slice starts at a multiple of 96 bytes; how the toolchain aligns it
is not verified on a board.

## Debugging excessive low-frequency suppression

Check these in order:

1. confirm the checkpoint reports the v6 cascade/alpha model version;
2. log ERB mask, alpha, and low-band DF output separately;
3. compare speech-only, noise-only, and steady low-frequency-noise cohorts;
4. verify error/far or clip-to-clip EMA state is not shared or leaked.

If the ERB-masked low band is healthy but the alpha-blended output collapses,
the issue is in the DF/alpha path rather than the shared dataset. If the
ERB-masked low band is already collapsed, audit low-frequency corpus coverage,
speech-only sampling, SNR distribution, and target contamination.

The upstream-alignment record is in
[`UPSTREAM_ALIGNMENT.md`](../UPSTREAM_ALIGNMENT.md).

[返回本主題](../README.md)
