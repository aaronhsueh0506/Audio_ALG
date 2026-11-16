# Calling the library

The order, for one mono stream (hop in, hop out; the application frames with audiolib). Nothing in the library knows a grid until
the application gives it one, so step 1 comes first and its `DFN2Grid` goes to every later call.

1. `dfn2_grid_make()` from the application's table row (or `dfn2_grid_validate()` on a grid it already holds).
2. Load the ERB matrices of that model (the layouts are described under [Choosing the grid](grid.md#choosing-the-grid); the library never
   reads files and borrows the pointers).
3. The framing: `ac_stft_config_default(&sc, grid.n_fft)` with `sc.spec_scale = dfn2_stft_scale(grid.n_fft)` and `sc.ispec_scale =
   dfn2_stft_unscale(grid.n_fft)`, then `ac_stft_get_mem_size()` / `ac_stft_init()` and `ac_istft_get_mem_size()` / `ac_istft_init()`.
4. `dfn2_prepost_config_defaults(&cfg, &grid)`, set `erb_fwd`, `erb_inv` (and `atten_lim_db` when wanted) and, on a float host, the
   state registration; `dfn2_prepost_get_mem_size()`, a 16-byte aligned pool of `req.bytes`, `dfn2_prepost_init_ex()` (a stale `req`
   is refused).
5. `dfn2_model_io_descriptor_default(&grid, &d)`: the graph contract of this build for the grid, to compare with what the accelerator
   runtime reports (`dfn2_model_io_descriptor_validate(&grid, &d)`); a model exported for another grid or lookahead pair fails it.
6. Per hop (`frames_per_call` 1): `ac_stft_push_planar()` (this spectrum is also the application spectrum and stays until the post),
   `dfn2_prepost_pre_frame()` (returns 0 on the first M hops: no job; 1 from hop M on), then for a job `dfn2_prepost_views()`, the model, and
   `dfn2_prepost_validate_run()` (0: the heads are accepted; -1: refused, the state is zeroed and the generation advanced once) or, when the run
   did not succeed, `dfn2_prepost_zero_state()` (a run that never started leaves the state alone). Then `dfn2_prepost_post_frame(p, 0, ...)` with
   an accepted run, otherwise `dfn2_prepost_post_frame_identity()` (the noisy spectrum; also the first M hops), and `ac_istft_push_planar()`;
   the return is 1 once a spectrum is emitted and 0 before (the output is zero then). The library keeps no per-call latch: the sequence above is
   the caller's, and `stream/ainr/dfn2/dfn2_stream.h` implements it for PCM and spectrum callers, N frames per call included
   (see [Multi-frame export](multiframe.md#multi-frame-export)).
7. Flush: push `M + D + 1` zero audio hops (`M + D` zero spectra, one more for the overlap-add) with the accelerator running on all of them.

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "audio_stft.h"
#include "dfn2_prepost.h"

/* The shipped model: config.ini [signal] [model]. The seven [feature] normaliser settings
 * are fixed constants inside dfn2_grid.h now (DFN2_ERB_NORM_TAU_SEC etc.), not spec fields. */
static const DFN2GridSpec SHIPPED = {
    .sample_rate = 48000, .n_fft = 1024, .n_erb = 32, .df_bins = 96, .df_order = 5,
    .mask_lookahead = 1, .df_lookahead = 1, .frames_per_call = 1,
};

/* State shapes belong to the application: the exported graph's h_gru [1,5,1,256] and
 * df_convp_history [1,64,5,df_bins]. */
enum { GRU_ELEMENTS = 5 * 256, CONVP_CHANNELS = 64, CONVP_WINDOW = 5 };

static void *alloc16(size_t bytes) { return aligned_alloc(16, (bytes + 15) & ~(size_t)15); }

/* fb_<sr>_<n_erb>.bin = the exporter's erb_matrix.bin: int32 n_bins, int32 n_bands, float32
 * fwd[n_bins][n_bands], float32 inv[n_bins][n_bands] (both bin-major). The library wants the
 * inverse band-major [n_erb][n_bins]: transpose it once. Returns 0, or -1. */
static int load_erb(const char *path, const DFN2Grid *g, float *fwd, float *inv_band_major)
{
    size_t n = (size_t)g->n_bins * (size_t)g->n_erb;
    int32_t hdr[2];
    float *inv = malloc(n * sizeof *inv);
    FILE *f = fopen(path, "rb");
    int ok = f && inv && fread(hdr, sizeof hdr[0], 2, f) == 2 &&
             hdr[0] == g->n_bins && hdr[1] == g->n_erb &&
             fread(fwd, sizeof *fwd, n, f) == n && fread(inv, sizeof *inv, n, f) == n;
    for (int k = 0; ok && k < g->n_bins; ++k)
        for (int b = 0; b < g->n_erb; ++b)
            inv_band_major[(size_t)b * g->n_bins + k] = inv[(size_t)k * g->n_erb + b];
    if (f) fclose(f);
    free(inv);
    return ok ? 0 : -1;
}

/* The accelerator: reads the windows in `in`, fills out->erb_mask / coefs / alpha and updates the
 * two state tensors in place (the state outputs alias the state inputs: read every state input
 * before writing any state output). 0 when everything was written. */
typedef int (*RunModel)(const DFN2PrepostInputs *in, DFN2PrepostOutputs *out);

/* One mono stream, `hops` hops of grid.hop samples from `in` to `out`. Returns 0, or -1. */
static int run_dfn2(const char *erb_path, RunModel run_model, const float *in, float *out, long hops)
{
    DFN2_REQUIRE_VARIANT_LINK();           /* units built with another -DDFN2_MASK_ONLY do not link */

    /* 1. the grid: what the model fixes, the rest derived and validated */
    DFN2Grid grid;
    int status = dfn2_grid_make(&grid, &SHIPPED);
    if (status != DFN2_GRID_OK) {
        fprintf(stderr, "grid refused: %s\n", dfn2_grid_status_text(status));
        return -1;
    }

    int rc = -1;
    size_t erb_n = (size_t)grid.n_bins * (size_t)grid.n_erb;
    size_t convp_elements = (size_t)CONVP_CHANNELS * CONVP_WINDOW * grid.df_bins;
    float *erb_fwd = malloc(erb_n * sizeof *erb_fwd), *erb_inv = malloc(erb_n * sizeof *erb_inv);
    float *gru = calloc(GRU_ELEMENTS, sizeof *gru);                      /* the application's state */
    float *convp = calloc(convp_elements, sizeof *convp);
    float *re = malloc(4 * (size_t)grid.n_bins * sizeof *re);            /* spectra of one hop */
    float *im = re + grid.n_bins, *out_re = im + grid.n_bins, *out_im = out_re + grid.n_bins;
    void *stft_mem = NULL, *istft_mem = NULL, *pool = NULL;
    AcStft *stft = NULL;
    AcIstft *istft = NULL;
    DFN2Prepost *p = NULL;
    if (!erb_fwd || !erb_inv || !gru || !convp || !re) goto done;
    if (load_erb(erb_path, &grid, erb_fwd, erb_inv) != 0) goto done;   /* 2. the ERB matrices (caller-owned, borrowed) */

    /* 3. the framing: audiolib, the scales from the grid */
    AcStftConfig sc;
    ac_stft_config_default(&sc, grid.n_fft);
    sc.spec_scale = dfn2_stft_scale(grid.n_fft);
    sc.ispec_scale = dfn2_stft_unscale(grid.n_fft);
    size_t stft_bytes = ac_stft_get_mem_size(&sc), istft_bytes = ac_istft_get_mem_size(&sc);
    stft_mem = alloc16(stft_bytes);
    istft_mem = alloc16(istft_bytes);
    if (!stft_mem || !istft_mem) goto done;
    stft = ac_stft_init(stft_mem, stft_bytes, &sc);
    istft = ac_istft_init(istft_mem, istft_bytes, &sc);
    if (!stft || !istft) goto done;

    /* 4. config (copies the grid), exact pool size, init inside caller memory */
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    if (dfn2_prepost_config_defaults(&cfg, &grid) != 0) goto done;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.atten_lim_db = 0.0f;               /* 0 = no attenuation limit */
    cfg.gru_hidden = gru;                  /* a float host registers its state; a board registers nothing */
    cfg.gru_hidden_elements = GRU_ELEMENTS;
    cfg.df_convp_history = convp;
    cfg.df_convp_history_elements = convp_elements;
    if (dfn2_prepost_get_mem_size(&cfg, &req) != 0) goto done;
    pool = alloc16(req.bytes);
    p = pool ? dfn2_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req) : NULL;
    if (!p) goto done;

    /* 5. the graph the accelerator runtime was built against. A board fills it from the exported
     *    model's metadata; _default() is this build's contract for the grid, and _validate()
     *    refuses a description of another grid, pair or frame count. */
    DFN2ModelIoDescriptor desc;
    if (dfn2_model_io_descriptor_default(&grid, &desc) != 0 ||
        dfn2_model_io_descriptor_validate(&grid, &desc) != 0) goto done;

    /* 6. per hop (N == 1): analysis, pre_frame, views -> model -> validate_run, post_frame, synthesis. The first
     *    mask_lookahead hops return 0 from pre_frame: no job, no model run, the state and the generation stay. */
    rc = 0;
    for (long h = 0; h < hops && rc == 0; ++h) {
        ac_stft_push_planar(stft, in + h * grid.hop, re, im);
        int need = dfn2_prepost_pre_frame(p, re, im);          /* 0 on the first M hops, 1 after */
        int commit = 0;
        if (need < 0) { rc = -1; break; }
        if (need == 1) {
            DFN2PrepostInputs xin;
            DFN2PrepostOutputs xout;
            if (dfn2_prepost_views(p, &xin, &xout) != 0) { rc = -1; break; }
            /* dfn2_prepost_state_generation(p) changed since the last frame (a reset, a refused run or a
             * failed run zeroed the state): an accelerator with its own state copy refills it first. */
            if (run_model(&xin, &xout) != 0)
                dfn2_prepost_zero_state(p);                /* the noisy spectrum below, state zeroed, generation + 1 */
            else
                commit = dfn2_prepost_validate_run(p) == 0;   /* refused: already zeroed, the noisy spectrum below */
        }
        if ((commit ? dfn2_prepost_post_frame(p, 0, re, im, out_re, out_im)
                    : dfn2_prepost_post_frame_identity(p, re, im, out_re, out_im)) < 0) { rc = -1; break; }   /* zeros for the first M + D hops */
        ac_istft_push_planar(istft, out_re, out_im, out + h * grid.hop);                         /* every hop: AC_OLA_ACCUMULATE */
    }
done:
    free(pool); free(istft_mem); free(stft_mem); free(re); free(convp); free(gru); free(erb_inv); free(erb_fwd);
    return rc;
}
```

The spectra are `torch.stft(normalized=True)` scaled: the `AcStft` / `AcIstft` of step 3 carry `dfn2_stft_scale(grid.n_fft)` and
`dfn2_stft_unscale(grid.n_fft)`, so the application passes audio-scale samples. Build every unit that includes the DFN2 headers with the same
`-DDFN2_MASK_ONLY` setting as the library; `DFN2_REQUIRE_VARIANT_LINK()` turns a mismatch into a link error. With the shipped 1/1
pair `valid` is 0 for the first two hops (the output is zeros) and the end-to-end delay is three hops. The example assumes
`frames_per_call` 1 (the table row above); the other pairs are in
[Lookahead pairs](../README.md#lookahead-pairs).

[返回本主題](../README.md)
