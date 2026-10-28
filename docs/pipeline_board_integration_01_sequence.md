> [Index](pipeline_board_integration.md) | [Next: Warnings, descriptor and teardown contract](pipeline_board_integration_02_contract.md) <!-- split-nav -->
## At a glance: sequence, public/internal functions, external/internal parameters

Lifelines of the board path. The mono flow has three library objects; the 4-channel flow has four
`Aec` lanes, one shared `DelayAec`, the `Mcab` flow module and the same post libraries; every box is
queried and initialised on its own. The numbered sequences further down give the exact calls; the
per-flow diagrams (with return values and the swimlane over hops) are in section 0 of
[`integration_mono_zh_TW.md`](integration_mono_zh_TW_01_overview_link_quickstart.md#0-速覽流程時序圖公開內部函式外部內部參數),
[`integration_4ch_core_zh_TW.md`](integration_4ch_core_zh_TW.md) and
[`integration_4ch_spatial_zh_TW.md`](integration_4ch_spatial_zh_TW.md).

```
Firmware         Aec (mono) | Aec x4, DelayAec, Mcab       AnrDenoiser          AcIstft
   |                  |                           |                   |
   |== boot (once; re-query every size, never cache) ==========================
   |-- *_get_mem_size (0 = invalid) / mcab_get_mem_requirements (0 = ok, -1 = invalid)
   |-- platform_alloc(Sum ALIGN16(each block), 16)         pool need not be zeroed
   |-- aec_init | aec_init x4, delay_aec_init (MATCHED), mcab_init_ex -->|-- aec_res_init (4ch only) -->|-- anr_init -->|-- ac_istft_init -->|   (NULL = stop)
   |== per hop ================================================================
   |-- aec_process_context, aec_get_res_context | mcab_process_front, your beamformer, mcab_fuse -->|  mono: ctx{error_spec,r2,comfort_noise,res_gain}; 4ch: frame{e,r2,cn,g_res_ctx}
   |      (mono: res_gain is the Aec's own RES gain; 4ch: g_res_ctx = NULL)
   |-- mcab_sample_res_input + aec_res_process(res, &in) (4ch only) ---->|  G_res from your AecRes
   |-- anr_process_gain + anr_get_gain ---------->|  G_nr   (4ch: mcab_sample_anr_gain)
   |-- sk_min_f32, sk_capply_gain_f32, aec_cng_fill (firmware code)
   |-- ac_istft_push ------------------------------------------------>|  out[hop]   (1 hop behind the input)
   |== refused call or non-finite output =======================================
   |-- aec_reset | mcab_reset (also resets the lanes and the estimator), aec_res_reset (4ch only), anr_reset, ac_istft_reset, rng = your comfort-noise seed   (all together)
   |== teardown ================================================================
   |-- anr_destroy, aec_destroy | mcab_destroy, aec_destroy x4, then platform_free(pool)
```

| | Public (the firmware calls these) | Internal (do not call or include) |
|---|---|---|
| Mono | `aec_resolve_signal_grid()`/`aec_config_from_level()`, `aec_get_mem_size()`/`aec_init()`, `anr_get_mem_size()`/`anr_init()`, `ac_istft_get_mem_size()`/`ac_istft_init()`, `aec_process_context()`, `aec_get_res_context()`, `anr_process_gain()`/`anr_get_gain()`, `aec_cng_fill()`, `ac_istft_push()`, resets, destroys, `aec_set_level()`, `anr_set_level()` | AEC headers other than `aec.h`/`aec_res.h`, NR `noise_estimator.h`/`spp_estimator.h`/`anr_internal.h` |
| 4-channel | `mcab.h` (`mcab_*`), `aec.h` for the lanes, `delay_aec_get_mem_size()`/`delay_aec_init()` for the estimator, plus the same post libraries | `mcab_internal.h`; `bf.h`/`srp.h` and `audiolib/{doa,bf}` (used by the sample helper `mcab_sample_beam` only) |

Parameters: what a third party may tune (grid and rate, `filter_length`, delay mode and bank size,
AEC and NR levels, array geometry and `num_angles` of the sample helper) and what stays inside
(time constants, step sizes, detector thresholds, DOA/BF recursion constants, NR tracker constants)
is classified field by field in section 0.3 of the three manuals above.

## Static APIs and pool sizing

Each library slices its own internals (HPF, PBFDKF x2, delay estimator, RES/post context,
internal FFTs for AEC; MCRA + SPP for NR) inside its single pool segment. The integrator queries
each size, aligns every segment to 16 bytes (`ALIGN16`) and places them in one block (or
several):

```c
size_t aec_sz   = aec_get_mem_size(&aec_cfg);       /* lib/aec; 0 = invalid config        */
size_t nr_sz    = anr_get_mem_size(&nr_cfg);         /* lib/nr; 0 = invalid config         */
size_t synth_sz = ac_istft_get_mem_size(&synth_cfg); /* audiolib; 0 = invalid config       */
/* + your gain/spectrum buffers and your own stages; every segment 16-byte aligned (ALIGN16) */

Aec*         aec   = aec_init(mem_aec, aec_sz, &aec_cfg);
AnrDenoiser* nr    = anr_init(mem_nr, nr_sz, &nr_cfg);
AcIstft*     istft = ac_istft_init(mem_synth, synth_sz, &synth_cfg);
/* destroy() on a static instance never frees the pool. aec_destroy() only drops the
 * instance's internal references and is NULL-safe and repeatable; anr_destroy() returns at
 * once for a pool-resident instance. Call them before releasing the pool. */
```

Query the exact pool budget for any configuration without running audio:

```bash
./aec_nr_pipeline_static --print-mem-size --sample-rate 16000
./mcab_static --print-mem-size --sample-rate 16000                   # in 4ch_aec_bf_nr_res/
```

**Embedded-target integration:** allocate one contiguous, 16-byte-aligned
block of the summed total from the platform allocator, pass it in place of
the desktop `malloc` — no other change. Each pool base MUST be 16-byte
aligned (the libraries assert this).

## Sequence (mono: `Aec` plus the libraries it chains)

```
1. query    AecSignalGrid grid;  AecConfig aec_cfg;
            if (!aec_resolve_signal_grid(sample_rate, aec_default_fft_size(sample_rate), &grid)) fail;  // 0 = no such grid
            aec_config_from_level(&aec_cfg, level, sample_rate);    // then fft_size = grid.fft_size, enable_res = 0,
                                                                    // return_res_context = 1 (see mono_aec_nr_res/main.c derive_lane())
            AnrConfig nr_cfg = anr_config_for_level_grid(sample_rate, grid.fft_size, level);
            AcStftConfig synth_cfg;  ac_stft_config_default(&synth_cfg, grid.fft_size);
            size_t aec_bytes   = aec_get_mem_size(&aec_cfg);       // 0 = invalid
            size_t nr_bytes    = anr_get_mem_size(&nr_cfg);        // 0 = invalid
            size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);// 0 = invalid
            size_t buf_bytes   = 3*ALIGN16(n_freqs*sizeof(float)) + ALIGN16(n_freqs*sizeof(Complex));
            // Query these fresh, immediately before every init below — never cache a size
            // across a build/backend/config change and replay it later. See "Warnings" below (pipeline_board_integration_02_contract.md).

2. allocate void* pool = platform_alloc(ALIGN16(aec_bytes) + ALIGN16(nr_bytes) +
                                       ALIGN16(synth_bytes) + buf_bytes, 16);
            // pool need NOT be zeroed — see "Dirty-pool contract" below (pipeline_board_integration_02_contract.md). Pool must stay STABLE
            // and EXCLUSIVE (nothing else reads/writes it, not shared with any other
            // instance) for the entire lifetime of the handles below.

3. init     uint8_t* cur = pool;
            Aec* aec = aec_init(cur, aec_bytes, &aec_cfg);           cur += ALIGN16(aec_bytes);
            AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);      cur += ALIGN16(nr_bytes);
            AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg);  cur += ALIGN16(synth_bytes);
            // gain/spectrum buffers follow (unity, total gain, NR prior, output spectrum)
            // NULL on: a misaligned/undersized pool, an invalid config, or a sub-object init
            // failure (stderr has detail from the libraries, UNLESS they were built with
            // NO_STDIO=1 — see "NO_STDIO=1" below, pipeline_board_integration_02_contract.md). A rejected init writes nothing outside
            // the pool and leaves the pool usable for a correct init.

4. process  float mic[hop], ref[hop], out[hop];   // hop = grid.hop_size
            while (have_audio()) {
                read_hop(mic, ref, hop);
                aec_process_context(aec, mic, ref);
                aec_get_res_context(aec, &c);
                // per hop: G_res = c.res_gain; nr = anr_process_gain(nr, c.error_spec, extra, NULL) + anr_get_gain(nr, NULL);
                //          sk_min_f32; sk_capply_gain_f32; aec_cng_fill; ac_istft_push
                if (!c.error_spec || !(P = mono_form(&c)) ||
                    ac_istft_push(istft, P, out) != hop || !skn_all_finite_f32(out, hop)) {
                    aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); rng = AEC_CNG_SEED;
                    continue;                                       // a refused hop: reset together
                }
                write_hop(out, hop);
            }

5. reset?   aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); rng = AEC_CNG_SEED;
            // optional: echo-path change, stream switch; re-zeros AEC/NR/synthesis state in place
            // and restarts the comfort-noise generator; no re-validation, no pool re-touch beyond that.

6. destroy  anr_destroy(nr); aec_destroy(aec);
            // NULL-safe; idempotent for these pool-resident instances.

7. release  platform_free(pool);   // only after step 6
```

## Sequence (4-channel: the `mcab` flow plus the libraries it feeds)

`Mcab` is the one object with a descriptor, and it covers its own pool only; the lanes and the
estimator are the application's. Query, init and teardown:

```
1. query    McabConfig cfg = mcab_default_config(sample_rate);
            cfg.delay_mode/delay_num_filters/capture_proxy_channel/enable_post = ...;
            AecConfig lane;   mcab_lane_config(&cfg, &lane);      // balanced level + the flow contract;
                              // or aec_config_from_level() + your own fields + mcab_apply_lane_contract(&cfg, &lane)
            McabDelayEstimatorConfig dc;  mcab_delay_estimator_config(&cfg, &dc);   // dc.required: MATCHED only
            size_t lane_bytes = aec_get_mem_size(&lane);                            // 0 = invalid
            size_t delay_bytes = dc.required ? delay_aec_get_mem_size(dc.sample_rate, dc.hop_size, dc.num_filters) : 0;
            McabMemReq req;
            mcab_get_mem_requirements(&cfg, &req);                   // 0 = OK, -1 = invalid; the flow's own pool
            // fresh before every init_ex(); see "Warnings" below (pipeline_board_integration_02_contract.md)

2. allocate one 16-byte-aligned block for 4 x ALIGN16(lane_bytes) + ALIGN16(delay_bytes) + the DelayAec struct +
            ALIGN16(req.bytes) (+ the post libraries), at req.alignment

3. init     McabLibs libs = {0};
            for (ch = 0; ch < MCAB_CHANNELS; ++ch) libs.lanes[ch] = aec_init(mem_lane[ch], lane_bytes, &lane);
            if (dc.required) { delay_aec_init(&delay, mem_delay, delay_bytes, dc.sample_rate, dc.hop_size, dc.num_filters);
                               libs.delay = &delay; }
            Mcab* p = mcab_init_ex(mem_mcab, req.bytes, &cfg, &libs, &req);
            // NULL on: any expected-descriptor mismatch (descriptor_version / layout_version /
            // backend_id / build_flags_hash / alignment / reserved / bytes below the current), a
            // misaligned/undersized pool, an invalid cfg, or a lane/estimator that does not fit the
            // flow (wrong grid or post half, lane not EXTERNAL_ALIGNED, duplicate lane, wrong bank size,
            // delay object missing/unexpected). The lanes and the estimator must be freshly
            // initialised or reset first. mcab_init() is init_ex(..., NULL): no descriptor check.
            // mcab_sample_flow_init() (mcab_sample_flow.h) is this whole sequence in one pool.

   (The beamformer, post-beam RES, NR and synthesis, the neural stage or DFN2 stage and bridge are separate
    objects with their own size queries and inits: the sample helper mcab_sample_beam_get_mem_size /
    mcab_sample_beam_init (SRP-PHAT, BF), aec_config_from_level + aec_res_get_mem_size / aec_res_init
    (config fft_size = mcab_fft_size(p); no destroy), anr_get_mem_size / anr_init,
    ac_istft_get_mem_size / ac_istft_init, ulcnet_nn_stage_mem_size / _init,
    dfn_rate_bridge_get_mem_requirements / dfn_rate_bridge_init; mcab holds no RES)

4. process  mcab_process_front(p, mics, far, &front);              // shared delay + 4 lanes
            // your beamformer on front.linear_spectra -> weights W (+ its spectrum E); or the helper:
            // mcab_sample_beam_process_with_activity(b, mics, far, vad, vad, NULL, &beam, &info);
            mcab_fuse(p, W, E, &frame);                            // optional (cfg.enable_post = 1)
            // frame.g_res_ctx = NULL (mcab has no RES); then
            // mcab_sample_res_input(&frame, &in) + aec_res_process(res, &in) -> mcab_sample_anr_gain -> sk_min_f32
            // -> sk_capply_gain_f32 -> aec_cng_fill -> ac_istft_push
            // or the helper's process_front() / process_beam() with your own mask and VAD in between

5. reset?   mcab_reset(p);  aec_res_reset(res);  anr_reset(nr);  ac_istft_reset(istft);  rng = MCAB_SAMPLE_CNG_RNG_SEED;  [ulcnet_nn_stage_reset(stage);]
            // mcab_reset() resets the four lanes and the estimator too.
            // after ANY refused call too: the integrator resets the flow and its own stages together

6. destroy  mcab_destroy(p); aec_destroy(lane) x4; ...   // helper first (mcab_sample_beam_destroy), then mcab, then the lanes;
                                                      // NR, stage and bridge each have their own
7. release  platform_free(pool); ...                  // only after step 6
```

### Running the model on the NPU (NN applications)

The ULCNet and DFN2 examples split the hop around the model call so the
accelerator runs between `nn_begin` and `nn_end` (the model config then needs
no `infer` callback; a `reset` callback is still called on an alignment
boundary and by the stage's `reset`):

```
// ULCNet (ulcnet/ulcnet_nn_stage.h)
ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, delay);   // mono: ulcnet_nn_stage_begin_time(...)
if (ulcnet_nn_stage_nn_begin(stage, &job) > 0) {                   // 1 = a frame is open (UlcnetNnJob)
    rc = run_npu(&job);                                            // job outputs are NaN-prefilled; write every one
    ulcnet_nn_stage_nn_end(stage, rc);                             // rc != 0 or a non-finite output => identity
}
ulcnet_nn_stage_end(stage, out);                                   // WOLA, exactly one hop

// DFN2 (dfn/dfn_hop.h): up to two frames per hop through the rate bridge
dfn_hop_begin(&hop, e, p);               // P = E.G_res formed first (by the example main)
while (dfn_hop_nn_begin(&hop, &job) == 1)
    dfn_hop_nn_end(&hop, run_npu(&job));
dfn_hop_end(&hop, out);
```

The 4-channel Align-ULCNet example splits the sample helper's hop once more for a board that derives a per-bin
mask and a VAD from the lane spectra of the same hop: `mcab_sample_beam_process_front(b, mics, far, &front)`
(lanes), the caller reads `front.linear_spectra[0]`, then
`mcab_sample_beam_process_beam(b, &front, vad_raw, vad_out, mask, &beam, &info)` (SRP, BF; `mask`
may be NULL). `info` returns the DOA (radians, NaN when none), the BF adaptation flag and the
alignment state (`info.delay`). `mcab_sample_beam_process_with_activity(b, mics, far, v, v, NULL, &beam, &info)` is `process_front` plus
`process_beam(front, v, v, NULL)`.

A frame that is never fetched, or left open at `end`, takes the
identity, so latency does not depend on the model. Every library's pool is
sized by its own query and carved from a caller-owned block; the per-hop stage chaining adds
no heap use in the hop path.
> [Index](pipeline_board_integration.md) | [Next: Warnings, descriptor and teardown contract](pipeline_board_integration_02_contract.md) <!-- split-nav -->
