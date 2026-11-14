# 板端呼叫順序

## At a glance: sequence, public/internal functions, external/internal parameters

Lifelines of the board path. The mono flow has two library objects besides the synthesis: an `Aec` and the denoiser; the
4-channel flow is one `Aec` too (`AecConfig.n_mics = 4`: four lanes, the shared delay estimator, the fusion and the gain stage in
its pool) plus the same post libraries. Every box is queried and initialised on its own. The numbered sequences further down give
the exact calls; the per-flow diagrams (with return values and the swimlane over hops) are in section 0 of
[mono 整合](../mono/integration.md#0-速覽流程時序圖公開內部函式外部內部參數),
[四麥克風 array](../array/README.md) and
[SRP／BF helper](../array/beam.md).

```
Firmware         Aec (mono: 1 mic | 4ch: n_mics = 4)       AnrDenoiser          AcIstft
   |                  |                                         |                   |
   |== boot (once; re-query every size, never cache) ==========================
   |-- aec_get_mem_requirements (0 = ok, -1 = invalid) / aec_get_mem_size, anr_get_mem_size, ac_istft_get_mem_size (0 = invalid)
   |-- platform_alloc(Sum ALIGN16(each block), 16)         pool need not be zeroed
   |-- aec_init_ex (or aec_init) -->|-- anr_init -->|-- ac_istft_init -->|   (NULL = stop)
   |== per hop ================================================================
   |-- mono: aec_process_linear, aec_process_residual(NULL)
   |   4ch: aec_process_linear, aec_get_array, your beamformer (weights W, optional E), aec_process_residual(&{W, E})
   |-- aec_get_residual -->|  v{error_spec, base_spec, r2, comfort_noise, res_gain (FULL), post_reset_serial, ...}
   |      (error_spec is the analysis seam; the gain is applied to base_spec; 4ch: the one G_res is computed over the fused spectra)
   |-- v.post_reset_serial moved: ac_istft_reset (before the CNG draw and the push; the library reseeds its own comfort noise)
   |-- anr_process_gain_with_residual_psd (mono: on base_spec; 4ch: array_sample_anr_gain, on error_spec) -->|  G_nr
   |-- aec_apply_gain(aec, G_nr, P), aec_apply_cng(aec, G_nr, P)   (P = base * min(G_nr, G_res) + comfort noise)
   |-- ac_istft_push ------------------------------------------------>|  out[hop]   (1 hop behind the input)
   |== refused call or non-finite output =======================================
   |-- aec_reset (4ch: also the four lanes, the estimator and the gain stage), anr_reset, ac_istft_reset   (all together)
   |== teardown ================================================================
   |-- anr_destroy, aec_destroy, then platform_free(pool)
```

| | Public (the firmware calls these) | Internal (do not call or include) |
|---|---|---|
| Mono | `aec_resolve_signal_grid()`/`aec_config_from_level()`, `aec_get_mem_size()`/`aec_init()`, `anr_get_mem_size()`/`anr_init()`, `ac_istft_get_mem_size()`/`ac_istft_init()`, `aec_process_linear()`, `aec_process_residual()`, `aec_get_residual()`, `anr_process_gain()`/`anr_process_gain_with_residual_psd()`, `aec_apply_gain()`, `aec_apply_cng()`, `ac_istft_push()`, resets, destroys, `aec_set_level()`, `anr_set_level()` | AEC headers other than `aec.h`, NR `noise_estimator.h`/`spp_estimator.h`/`anr_internal.h` |
| 4-channel | `aec.h` (`n_mics = AEC_ARRAY_MICS`: `aec_get_mem_requirements()`/`aec_init_ex()`, `aec_process_linear()`, `aec_get_array()`, `aec_process_residual()`, `aec_get_residual()`, `aec_array_*` diagnostics), plus the same post libraries | `aec_array_internal.h`; `bf.h`/`srp.h` and `audiolib/{doa,bf}` (used by the sample helper `array_sample_beam` only) |

Parameters: what a third party may tune (grid and rate, delay mode and bank size,
AEC and NR levels, array geometry and `num_angles` of the sample helper) and what stays inside
(time constants, step sizes, detector thresholds, DOA/BF recursion constants, NR tracker constants)
is classified field by field in section 0.3 of the mono manual and in the config references of the 4-channel manuals ([array](../array/config.md), [helper](../array/beam-settings.md)).

## Static APIs and pool sizing

Each library slices its own internals (HPF, PBFDKF x2, delay estimator, RES/post context,
internal FFTs for AEC; for the 4-channel array also the four lanes, the shared delay estimator and the gain stage; MCRA + SPP for NR) inside its single pool segment. The integrator queries
each size, aligns every segment to 16 bytes (`ALIGN16`) and places them in one block (or
several):

```c
size_t aec_sz   = aec_get_mem_size(&aec_cfg);       /* lib/aec; 0 = invalid config        */
size_t nr_sz    = anr_get_mem_size(&nr_cfg);         /* lib/nr; 0 = invalid config         */
size_t synth_sz = ac_istft_get_mem_size(&synth_cfg); /* audiolib; 0 = invalid config       */
/* + your gain/spectrum buffers and your own stages; every segment 16-byte aligned (ALIGN16) */

Aec*         aec   = aec_init(mem_aec, aec_sz, &aec_cfg);
AnrDenoiser* nr    = anr_init(mem_nr, nr_sz, &nr_cfg);
AcIstft*     istft = ac_istft_init(mem_synth, synth_sz, &synth_cfg);  /* synth_cfg.window = the table borrowed from aec_synthesis_window(); see the lifecycle below */
/* destroy() on a static instance never frees the pool. aec_destroy() only drops the
 * instance's internal references and is NULL-safe and repeatable; anr_destroy() returns at
 * once for a pool-resident instance. Call them before releasing the pool. */
```

Query the exact pool budget for any configuration without running audio:

```bash
./aec_nr_pipeline_static --print-mem-size --sample-rate 16000
./4ch_static --print-mem-size --sample-rate 16000                   # in 4ch/ (array + SRP/BF helper + NR + synthesis)
./4ch_external_static --print-mem-size --sample-rate 16000         # in 4ch/ (array + NR + synthesis)
```

**Embedded-target integration:** allocate one contiguous, 16-byte-aligned
block of the summed total from the platform allocator, pass it in place of
the desktop `malloc` — no other change. Each pool base MUST be 16-byte
aligned (the libraries assert this).

## Sequence (mono: `Aec` plus the libraries it chains)

```
1. query    AecSignalGrid grid;  AecConfig aec_cfg;
            if (aec_resolve_signal_grid(sample_rate, aec_default_fft_size(sample_rate), &grid) != AEC_OK) fail;  // AEC_ERR_ARG = no such grid
            aec_config_from_level(&aec_cfg, level, sample_rate);    // then fft_size = grid.fft_size, residual = AEC_RESIDUAL_FULL
                                                                    // (see mono/aec_nr_res/main.c derive_lane())
            AnrConfig nr_cfg = anr_config_for_level_grid(sample_rate, grid.fft_size, level);
            AcStftConfig synth_cfg;  ac_stft_config_default(&synth_cfg, grid.fft_size);
            synth_cfg.window = k_sizing_window;   // static const float k_sizing_window[1]; borrowed tables carve no pool, so any non-NULL stand-in sizes the synthesis; the AEC's own window is borrowed in step 3
            size_t aec_bytes   = aec_get_mem_size(&aec_cfg);       // 0 = invalid  (aec_get_mem_requirements() also fills the descriptor)
            size_t nr_bytes    = anr_get_mem_size(&nr_cfg);        // 0 = invalid
            size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);// 0 = invalid
            size_t buf_bytes   = 3*ALIGN16(n_freqs*sizeof(float)) + ALIGN16(n_freqs*sizeof(Complex));
            // Query these fresh, immediately before every init below — never cache a size
            // across a build/backend/config change and replay it later. See "Warnings" in memory.md.

2. allocate void* pool = platform_alloc(ALIGN16(aec_bytes) + ALIGN16(nr_bytes) +
                                       ALIGN16(synth_bytes) + buf_bytes, 16);
            // pool need NOT be zeroed — see "Dirty-pool contract" in memory.md. Pool must stay STABLE
            // and EXCLUSIVE (nothing else reads/writes it, not shared with any other
            // instance) for the entire lifetime of the handles below.

3. init     uint8_t* cur = pool;
            Aec* aec = aec_init(cur, aec_bytes, &aec_cfg);           cur += ALIGN16(aec_bytes);
            AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);      cur += ALIGN16(nr_bytes);
            int win_n;  if (!aec || aec_synthesis_window(aec, &synth_cfg.window, &win_n) != 0) fail;   // borrow the AEC's synthesis window
            AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg);  cur += ALIGN16(synth_bytes);
            // gain/spectrum buffers follow (unity, NR prior, output spectrum)
            // NULL on: a misaligned/undersized pool, an invalid config, or a sub-object init
            // failure (stderr has detail from the libraries, UNLESS they were built with
            // NO_STDIO=1 — see "NO_STDIO=1" in memory.md). A rejected init writes nothing outside
            // the pool and leaves the pool usable for a correct init.

4. process  float mic[hop], ref[hop], out[hop];   // hop = grid.hop_size
            uint32_t handled = aec_post_reset_serial(aec);
            while (have_audio()) {
                read_hop(mic, ref, hop);
                AecResidualView v = { sizeof(v) };                 // struct_size is an input; v.base_spec is the rendering base B
                int ok = aec_process_linear(aec, mic, ref) == AEC_OK &&
                         aec_process_residual(aec, NULL) == AEC_OK &&
                         aec_get_residual(aec, &v) == AEC_OK;
                // per hop: v.post_reset_serial != handled -> ac_istft_reset(istft); handled = serial
                //          (before the CNG draw and the push; the AEC reseeds its own generator and the denoiser is not reset for it);
                //          G_nr = anr_process_gain_with_residual_psd(nr, v.base_spec, residual_psd_audio);
                //          aec_apply_gain(aec, G_nr, P); aec_apply_cng(aec, G_nr, P); ac_istft_push
                //          (v.error_spec is the analysis seam, v.base_spec the rendering base)
                if (!ok || !(G_nr = mono_nr_gain(nr, &v)) ||
                    aec_apply_gain(aec, G_nr, P) != AEC_OK || aec_apply_cng(aec, G_nr, P) != AEC_OK ||
                    ac_istft_push(istft, P, out) != hop || !skn_all_finite_f32(out, hop)) {
                    aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); handled = aec_post_reset_serial(aec);
                    continue;                                       // a refused hop: reset together
                }
                write_hop(out, hop);
            }

5. reset?   aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); handled = aec_post_reset_serial(aec);
            // optional: echo-path change, stream switch; re-zeros AEC/NR/synthesis state in place
            // and restarts the comfort-noise generator; no re-validation, no pool re-touch beyond that.

6. destroy  anr_destroy(nr); aec_destroy(aec);
            // NULL-safe; idempotent for these pool-resident instances.

7. release  platform_free(pool);   // only after step 6
```

## Sequence (4-channel: the `Aec` array instance plus the libraries it feeds)

The array is one `Aec` (`AecConfig.n_mics = AEC_ARRAY_MICS`) with one descriptor (`AecMemReq`) that covers its whole pool: the four
lanes, the shared estimator, the alignment ring, the fusion buffers and the gain stage. There is nothing to hand in. Query, init and teardown:

```
1. query    AecConfig cfg;  aec_config_from_level(&cfg, level, sample_rate);    // or array_sample_default_config(&cfg, sample_rate)
            cfg.n_mics = AEC_ARRAY_MICS;  cfg.residual = AEC_RESIDUAL_FULL;     // STATE: lanes only (ULCNet); ESTIMATE: no gain
            cfg.delay_mode/delay_num_filters/capture_proxy_channel/... = ...;
            AecMemReq req;
            aec_get_mem_requirements(&cfg, &req);                    // 0 = OK, -1 = invalid; the whole array pool
            // fresh before every init_ex(); see "Warnings" in memory.md

2. allocate one 16-byte-aligned block for ALIGN16(req.bytes) (+ the post libraries, + the sample helper pool), at req.alignment

3. init     Aec* a = aec_init_ex(mem_aec, req.bytes, &cfg, &req);
            // NULL on: any expected-descriptor mismatch (descriptor_version / layout_version /
            // backend_id / build_flags_hash / alignment / reserved / bytes below the current), a
            // misaligned/undersized pool, or an invalid cfg (grid other than 16000/256, 16000/512, 48000/1024;
            // delay mode/field mismatch; residual OFF; capture_proxy_channel out of range).
            // aec_init() is init_ex(..., NULL): no descriptor check.
            // array_sample_unit_init() (array_sample_unit.h) does this sequence for the array and the helper.

   (The beamformer, post-beam NR and synthesis, the neural stage or DFN2 stage and bridge are separate
    objects with their own size queries and inits: the sample helper array_sample_beam_get_mem_size /
    array_sample_beam_init (SRP-PHAT, BF), anr_get_mem_size / anr_init,
    ac_istft_get_mem_size / ac_istft_init (the 4ch_static sample synthesis carries its own window table; optionally borrow aec_synthesis_window(a) as in mono),
    ulcnet_nn_stage_mem_size(&grid) / ulcnet_nn_stage_init(mem, bytes, &grid) (grid from ulcnet_grid_make; the stage has no STFT, so an AcStft for the raw far hop and an AcIstft for the output come with ac_*_get_mem_size / init),
    dfn_rate_bridge_default_config(host_rate, &model_grid) / dfn_rate_bridge_get_mem_requirements / dfn_rate_bridge_init
    (or dfn_res_stage_default_config(&model_grid) / dfn_res_stage_get_mem_requirements / dfn_res_stage_init when the
    host rate equals the model grid's; model_grid from dfn2_grid_make; cfg.stage.erb_fwd / erb_inv (cfg.erb_fwd / erb_inv
    on the stage path) point into the application's ERB data); the gain stage is inside the array)

4. process  aec_process_linear(a, mics, far);  aec_get_array(a, &front);   // shared delay + 4 lanes
            // your beamformer on front.lane_error_spec -> weights W (+ its spectrum E); or the helper:
            // array_sample_beam_process_with_activity(b, mics, far, vad, vad, NULL, &beam, &info);
            AecBeam beam = { W, E };  aec_process_residual(a, &beam);      // residual STATE: skipped
            aec_get_residual(a, &v);                                        // v.error_spec is the analysis seam, v.base_spec the rendering base
            // then: array_sample_anr_gain(nr, &v, scratch, 1) -> aec_apply_gain(a, G_nr, P) -> aec_apply_cng(a, G_nr, P) -> ac_istft_push
            // before the CNG draw and the push: array_sample_post_reset_due(&handled, &v) -> ac_istft_reset(istft)
            // or the helper's process_front() / process_beam() with your own mask and VAD in between

5. reset?   aec_reset(a);  anr_reset(nr);  ac_istft_reset(istft);  handled = aec_post_reset_serial(a);  [ulcnet_nn_stage_reset(stage); ac_stft_reset(far_stft); ac_istft_reset(ulc_istft);]
            // aec_reset() resets the four lanes, the estimator and the gain stage too (the level setting is kept).
            // after ANY refused call too: the integrator resets the array and its own stages together

6. destroy  array_sample_beam_destroy(b);  aec_destroy(a); ...   // helper first, then the array; NR, stage and bridge each have their own
7. release  platform_free(pool); ...                  // only after step 6
```

### Running the model on the NPU (NN applications)

The ULCNet and DFN2 examples split the hop around the model call so the
accelerator runs between `nn_begin` and `nn_end` (the model config then needs
no `infer` callback). Both stages are fed the error (or beam) spectrum; the Align-ULCNet
stage takes spectra only (the error or beam spectrum and the spectrum of the raw far hop, which
the application analyses with its own `AcStft`) and returns one output spectrum for the
application's `AcIstft`; the DFN2 stage emits one hop of samples. The model
libraries behind them hold no framing, FFT or window (audiolib's `AcStft` / `AcIstft` do that).
The ULCNet stage calls the model's `reset` callback on its own reset and on an alignment
event (a delay change, or the first transition to a solid alignment after the first hop);
the DFN2 stage and bridge call `model.reset` (when present) on their own reset only, and an
AEC delay change does not reset them:

```
// ULCNet (ulcnet/ulcnet_nn_stage.h)
ac_stft_push(far_stft, far, far_spec);                             // the application's analysis (mono: also err_stft on formed_hop)
ulcnet_nn_stage_begin_spec(stage, beam.spectrum, far_spec, delay);
if (ulcnet_nn_stage_nn_begin(stage, &job) > 0) {                   // 1 = a frame is open (UlcnetNnJob)
    rc = run_npu(&job);                                            // job outputs are NaN-prefilled; write every one
    ulcnet_nn_stage_nn_end(stage, rc);                             // rc != 0 or a non-finite output => identity
}
ulcnet_nn_stage_end_spec(stage, out_spec);                         // accepted model spectrum, else the error spectrum
ac_istft_push(ulc_istft, out_spec, out);                           // the application's WOLA, exactly one hop

// DFN2 (dfn/dfn_hop.h): up to two frames per hop through the rate bridge
dfn_hop_begin(&hop, e, p);               // P = E.G_res formed first (by the example main)
while (dfn_hop_nn_begin(&hop, &job) == 1)
    dfn_hop_nn_end(&hop, run_npu(&job));
dfn_hop_end(&hop, out);
```

The 4-channel Align-ULCNet example splits the sample helper's hop once more for a board that derives a per-bin
mask and a VAD from the lane spectra of the same hop: `array_sample_beam_process_front(b, mics, far, &front)`
(lanes), the caller reads `front.lane_error_spec[0]`, then
`array_sample_beam_process_beam(b, &front, vad_raw, vad_out, mask, &beam, &info)` (SRP, BF; `mask`
may be NULL). `info` returns the DOA (radians, NaN when none), the BF adaptation flag and the
alignment state (`info.delay`). `array_sample_beam_process_with_activity(b, mics, far, v, v, NULL, &beam, &info)` is `process_front` plus
`process_beam(front, v, v, NULL)`.

The recurrent state is the application's, not the library's. On a board the accelerator keeps it and nothing is registered with the
library; the model callback reloads quantized zeros into the accelerator's state inputs at start and whenever the generation of the object
that owns the restart count changes: `dfn_res_stage_state_generation()` (direct stage), `dfn_rate_bridge_state_generation()` (bridge),
and for Align-ULCNet the generation of the pre/post class or adapter the callback drives (`ulcnet_prepost_state_generation()`,
`ulcnet_accelerator_adapter_state_generation()`). The counter advances on a reset, on every DFN2 warm-up frame (the first `mask_lookahead` hops after a start or reset: the model runs,
its heads are discarded and the state restarts, so the callback reloads quantized zeros after each, M times in all) and on a refused
commit (a non-finite head) of a real frame; a model failure on a real frame and a committed frame leave it alone. It starts at 1 for the DFN2 objects and at 0 for the Align-ULCNet ones:
compare it, never assume the value. A float host that drives a pre/post class registers its own state buffers in that class's config
instead, and the library zeroes and checks them; the DFN2 stage and bridge register none.

A frame that is never fetched, or left open at `end`, takes the
identity, so latency does not depend on the model. Every library's pool is
sized by its own query and carved from a caller-owned block; the per-hop stage chaining adds
no heap use in the hop path.

[返回本主題](README.md)
