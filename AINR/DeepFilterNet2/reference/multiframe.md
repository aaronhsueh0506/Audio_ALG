# Multi-frame export

`export_onnx.py --frames-per-call N` (default 1, at most 64 as in the C host) writes a graph that runs N
frames per invocation, for an accelerator where one call per hop costs more
than the extra latency. `--frames-per-call 1`, or no option, writes the same
ONNX and JSON bytes as before the option existed.

| | one frame per call | N frames per call |
| --- | --- | --- |
| `erb`, `spec` | `(1,1,3,32)`, `(1,2,3,96)` | `(1,1,N+2,32)`, `(1,2,N+2,96)`, frames `[t-(2-M) .. t+N-1+M]` |
| `erb_mask`, `df_coefs`, `df_alpha` | heads of frame `t` | heads of frames `t .. t+N-1`, oldest first |
| `h_gru`, `df_convp_history` | `(1,5,1,256)`, `(1,64,5,96)` | unchanged |
| metadata | — | `frames_per_call`, `chunk_window_alignment`; `input_feature_frames` = N+2, `output_frames_per_invocation` = N |

Head `i` reads the three window frames starting at slot `i`, exactly the
one-frame window of frame `t+i`. The three GRUs run over the N frames as one
sequence (one GRU node per layer, as for N = 1) and `h_gru_out` is the hidden
state after the last frame. `df_convp_history_out` is the last five frames of
the pathway input: the kept history and the new frames for N < 5, the new
frames for N = 5; for N > 5 the last five frames of the DF input convolution
are computed from their own slice of the `spec` window, so the state output is
a convolution result and not a Slice (the in-place rule above; the same
multiply-adds as one convolution). Every other layer works on one frame at a
time. Fed in chunks of N with the state carried between calls, the graph equals
the one-frame graph fed frame by frame. `onnx_streaming_contract.py`
(`validate_frames_per_call`) refuses a graph whose `frames_per_call` disagrees
with its window and head shapes, and a one-frame graph that records one.

`inference.py calib --frames-per-call N` exports the N-frame graph and records
its invocations (`--frames` counts invocations, the report records
`frames_per_call`).

The C host takes N as `DFN2GridSpec.frames_per_call` (1..64, a run-time field; `DFN2_GRID_ERR_FRAMES_PER_CALL` otherwise),
with the lookahead pair:

- the window holds `N + 2` frames; the first call is ready after `M + N`
  pushed frames, every later one after N more (`pre_process_freq` returns 1 on hops
  `M+N-1`, `M+2N-1`, ...); `dfn2_model_io_frame_slot()` names the slot of the
  last pushed frame. With N > 1 nothing of the one-flow-per-hop behaviour applies: the earlier hops return 0, there are no warm-up
  frames, and `frame_open()` is called only after a 1 (as are the model and `frame_finish()`);
- `frame_open()` publishes N heads per tensor, `frame_finish()` builds the N enhanced frames
  in one burst, and `post_process_freq` returns one hop per call from that burst,
  so the output trails by N - 1 more hops: latency `1 + M + D + N - 1` hops
  (`dfn2_prepost_latency_frames(grid)` + 1; 10 hops, 106.7 ms, at the shipped pair
  with N = 8). A refused commit zeroes the registered state and advances the
  generation, and `frame_bypass()` passes all N frames as the noisy spectrum;
  flushing means pushing zero hops until the last real one has been emitted (`M + D + N` of them);
- `DFN2ModelIoDescriptor` keeps its 18 fields and 72 bytes: N is
  `input_frames - 2` (`input_frames` = N + 2, metadata `input_feature_frames`),
  so `dfn2_model_io_descriptor_validate()` and `dfn2_model_run_frame()` refuse
  another N, and a descriptor made for the one-frame graph (`input_frames` = 3)
  is refused by a grid of N > 1;
- N is folded into the pool hash (a pool of another N is refused) and into the pool
  size (144,112 bytes at N = 4 on the shipped grid against 92,608 at N = 1; 64-bit host). The pipelines DFN stage and its Python twin are
  one-frame units, and the stage refuses N > 1.

`DFN2_MODEL_IO_LAYOUT_VERSION` stays 8: it names the state tensors, which N
does not change, and the window and head shapes carry N through `input_frames` in
the descriptor and `frames_per_call` in the metadata.

Tests: `../tests/test_dfn2_frames_per_call.py` (wrapper and exported graph
streamed in chunks against the one-frame graph, all pairs; a wrong state carry
fails; metadata and validator; calibration) and
`../tests/test_dfn2_frames_per_call_c.py` (the N-frame host against the
one-frame host, byte for byte, with a stand-in accelerator; reset, refusal, contract;
the 72-byte descriptor through a grid of either N).

## Mask-only (stage 1)

The cascade is stage 1 (ERB mask over the full band) then stage 2 (deep filter on the low bins, blended by `alpha`). The
mask head does not depend on the deep-filter decoder, so the same checkpoint can run stage 1 alone:

- `python3 inference.py --model ... --input a.wav --output out.wav --mask-only` writes `out.wav` (full) and
  `out_mask_only.wav` (stage 1, `model.compose_mask_only`). `--input-dir/--output-dir` does the same per file.
- `python3 export_onnx.py --model ... --output m.onnx --mask-only` and `inference.py calib ... --mask-only` write the
  stage-1 graph: inputs `erb`, `spec`, `h_gru` `(1, 3, 1, 256)`; outputs `erb_mask`, `h_gru_out`. The deep-filter decoder
  (two GRU layers, `df_skip`, `df_out`, `df_fc_a`, `df_convp`) and `df_convp_history` are gone: 5 GRU nodes become 3. The
  encoder keeps `df_conv0`, `df_conv1` and `df_fc_emb` (their output feeds the embedding the ERB decoder reads), so the
  grouped linear of `df_fc_emb` stays in the graph. Metadata: `variant: mask_only`, `state_layout_version` 9,
  `df_lookahead` 0.
- C: build every unit with `-DDFN2_MASK_ONLY` (see `dfn2_model_io.h`). The state, the accelerator boundary structs and the
  descriptor shrink accordingly, and the output is `noisy * (mask @ erb_inv)` of the head frame, `grid.mask_lookahead`
  frames behind the input. The grid's `df_lookahead` must be 0 (`DFN2_GRID_ERR_DF_LOOKAHEAD` otherwise), `df_order` is stored as 0, and `df_bins` may reach `n_bins`. The final attenuation-limit mix (`--atten-lim`, `dfn2_prepost_set_atten_lim`) stays and acts on
  the stage-1 output in Python and in C. `../tests/test_dfn2_mask_only_c.py` compares it with numpy.
- Board sample: `alg_ai3_dfn_lite.c` (`audio_sample/`, outside this repo) is the sample for this graph, separate from the
  full model's `alg_ai3_dfn.c`. Build it and every library unit with `-DDFN2_MASK_ONLY`, in an object directory of its own
  (objects of the other variant do not link: the link symbol is `dfn2_variant_mask_only`). It binds I/O layout 9: inputs
  `erb, spec, h_gru` (h_gru is `(1, 3, 1, 256)`), outputs `erb_mask, h_gru_out`. The model is read from
  `/mnt/sd/audio_model/<quant>/dfn2_lite/` and the output goes to `/mnt/sd/audio/output/DFN2_LITE/`. Its grid comes from
  `--sample-rate` and `--fft-size` through the sample's table (see [Choosing the grid](grid.md#choosing-the-grid)); not run on a board from this repo.

No retraining: the quality of stage 1 alone is whatever the jointly trained mask gives (not measured here).

[返回本主題](../README.md)
