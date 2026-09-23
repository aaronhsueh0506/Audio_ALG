# Mono AEC + RES + DeepFilterNet2

This pipeline replaces the MMSE-LSA denoiser of the conventional mono
pipeline with DeepFilterNet2 while keeping everything else of that pipeline:
the linear AEC, the AEC3 residual echo suppressor, the comfort-noise recipe
and the synthesis WOLA. It does so by *hosting* the conventional pipeline
(`mono_aec_nr_res/audio_pipeline.h`, configured `enable_nr=0`,
`enable_res=1`) and inserting the network between the host's post-RES
spectrum and the host's synthesis:

```text
mic/ref -> audio_pipeline_process_post_view()
             E = linear-AEC error spectrum        -> DFN2 feature input
             P = E * G_res + comfort noise        -> DFN2 applied here
           dfn_res_stage_process(E, P)            -> enhanced P, frame t-2
           audio_pipeline_synthesize_external()   -> sqrt-Hann iFFT / OLA
```

The network estimates from the pre-RES spectrum because it was trained on
unprocessed noisy speech and must not see the spectral holes RES cuts; it is
applied to the post-RES spectrum so far-end residue that looks like speech is
preserved by the network and left to RES, and non-speech is suppressed by
both. The host's comfort-noise fill is kept as a switch but is OFF by
default in this variant: with it on, the additive fill perturbs P's phase in
the RES-cut bins where it is largest, which breaks the estimate/apply phase
consistency the complex deep filter relies on, and no true-model result yet
supports enabling it. Turn it on only for a subjective far-end-only / silence
A/B; when on it travels through the network like every other component of P
and `atten_lim_db` bounds how far any bin can be attenuated. The dual-input
topology with the full deep filter is itself the product hypothesis: the C
gates prove it structurally, and its quality against a mask-only compose is
settled by `pipelines/tools/eval_dfn2_variants.py` once a checkpoint of the
current model contract exists.

With no model callback (or a failing one) the output is the conventional
pipeline with `enable_nr=0` delayed: bit for bit by two hops at 48 kHz
(the acceptance gate in `tests/test_audio_pipeline_dfn.c`, asserted with
comfort noise off and on, across a real matched-delay re-lock, with the
suppressor demonstrably active), and through the resampler round trip by
the bridge delay on a native grid (the same test pins the delay with a
near-end impulse, the frame schedule, and the pass-band error against the
conventional 16/8 kHz output, measured at 1e-5).

The host runs on the product's own grid: 16 kHz (fft 256, hop 128) by
default, 16 kHz/512 and 8 kHz/256 as the host offers them, or 48 kHz. Only
the DFN2 stage runs at 48 kHz. At 48 kHz the host's spectra are DFN2's grid
and the stage is applied directly (algorithmic latency three hops: two model
lookahead hops plus one synthesis-WOLA hop, 32 ms). Below 48 kHz the rate
bridge (`pipelines/dfn_rate_bridge.h`) synthesises E and P on the native
grid, resamples them up with `audio_common`'s polyphase FIR, runs the stage
on the 48 kHz 1024/512 grid every 512 new samples, and resamples the result
down; the AEC, RES and comfort noise never leave the product rate. The
bridge adds a fixed delay to the host's own output: 672 samples at 16 kHz
with hop 128 (42 ms), 629 at 16 kHz with hop 256 (39 ms), 330 at 8 kHz
(41 ms); the total algorithmic latency is that plus the host's WOLA hop
(16 kHz, hop 128: 800 samples, 50 ms). Per native hop the stage runs a fixed
schedule (16 kHz/hop 128: 0,1,1,1 frames; 16 kHz/hop 256 and 8 kHz: 1,2), so
two inferences in one hop is a real per-hop compute fact. The Python reference
`pipelines/aec_dfn2_res_pipeline.py` mirrors both paths (16 and 48 kHz
hosts).

Build and test from `pipelines/` (the directory `Makefile` forwards):

```sh
make BACKEND=kiss WERROR=1 libaudio_pipeline_dfn.a mono_aec_dfn_res
make BACKEND=kiss WERROR=1 test        # includes test_dfn_res_stage,
                                       # test_audio_pipeline_dfn and the skeleton
```

`libaudio_pipeline_dfn.a` holds the wrapper, the shared stage and the DFN2
host pre/post objects. It is linked together with `libaudio_pipeline.a` (the
host), `libaec`, `libmmse_lsa` (referenced by the host archive) and
`libaudio_common`; the board supplies the stateless model callback
(`DFN2Model`) and the exported ERB matrices (`erb_fwd.bin` / `erb_inv.bin`
from `AINR/DeepFilterNet2/export_erb_matrix.py`). `main.c` is the board
skeleton: delay-profile flags in the `aec_wav` spelling, `--erb-fwd` /
`--erb-inv`, `--sample-rate` (default 16000), `--atten-lim`, `--cng`, and a fail-open `run_accelerator()`
stub to replace with the runtime call. It publishes the model I/O descriptor
of the geometry it was compiled against; a product fills that descriptor from
its model's exported metadata instead, which is what makes a mismatched graph
detectable.

## Compute admission

`pipelines/tools/profile_dfn2_ops.py` counts 2,134,971 trainable parameters
and 3,307,136 major model MACs per 512-sample invocation, or 310.044 MMAC/s
at 48 kHz. This excludes elementwise operations, host pre/post DSP and memory
traffic; final A53/A73 admission therefore depends on the board's int8/fp
runtime. Weight storage is about 2.04 MiB int8, 4.07 MiB fp16 or 8.14 MiB
fp32.

At a 16-kHz product boundary, two input resamplers (mic and render) plus the
mono output resampler add about 4.720 MMAC/s. Whole-front-end conversion also
moves the AEC itself to 48 kHz; that extra AEC cost is separate from this
resampler count and must be included in the board measurement.

Host DSP pool (`mono_aec_dfn_res_get_mem_requirements`, matched delay,
five-filter bank, NE10): 827,552 bytes at the 16 kHz default (host plus the
rate bridge, whose own pool is 428,576 bytes including the stage) and
1,571,264 bytes at 48 kHz (the conventional MMSE-LSA pipeline on the 48 kHz
grid needs 1,608,128 bytes). Model weights and accelerator activation/
workspace memory are external to this descriptor; adding int8 weights alone
brings the known minimum to about 2.8 MiB at 16 kHz.
