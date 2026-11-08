# GTCRN

Compact 16 kHz noise-reduction model on the project `512/512/256`
FFT/window/hop grid. Training and offline inference use a periodic root-Hann
window and unnormalised complex STFT, matching the model's `[B,F,T,2]`
real/imag boundary.

`gtcrn_process.h/.c` implements the deployment-side pre/post-processing on spectra:
the ERB banding in front of the graph, the mask expansion and application behind it,
the finite checks and the state generation. The STFT and ISTFT are audiolib's
(`AcStft` / `AcIstft`), run by the application. The model inference itself is
intentionally outside this C module and may run on the target accelerator. The
streaming flow consumes and emits one 256-sample hop per call; its initial boundary is
zero-padded rather than using PyTorch's offline centered reflect padding.

Run the shared C parity and round-trip gate from `AINR/`:

```bash
make test-simd
make SIMD=0 test
```

Architecture/paper notes are in `docs/architecture_notes.md`.

## C integration

The application describes the exported model in a `GTCRNGridSpec` (`n_fft`, `erb_low_bins` = `erb_subband_1`, `erb_high_bands` =
`erb_subband_2` of `config.ini`), turns it into a `GTCRNGrid` with `gtcrn_grid_make()` (`hop`, `n_bins`, `erb_bands` and
`erb_high_bins` are derived: 256, 257, 129 and 192 for the shipped 512-point grid) and passes the grid to
`gtcrn_process_config_defaults()`. The memory is the caller's (`gtcrn_process_get_mem_size()`, memory, `gtcrn_process_init_ex()`);
the pool is 3,328 bytes on the shipped grid (64-bit host) and holds neither the ERB matrices nor the state. The model is a
16 kHz model; the sample rate is not a grid field. Framing is audiolib's, configured by `ac_stft_config_default(&sc, grid.n_fft)` and
nothing else: periodic sqrt-Hann, center 0, hop `n_fft / 2`, `spec_scale = ispec_scale = 1.0f` (unnormalised, not
`torch.stft(normalized=True)`), `AC_OLA_ACCUMULATE`. Spectra are interleaved `[n_bins][2]`.

```c
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_stft.h"
#include "gtcrn_process.h"

/* The application's state shapes (the exported graph's), by GTCRNStateId: conv_enc0..2, conv_dec0..2, h_tra, h_dpgrnn. */
static const size_t STATE_ELEMENTS[GTCRN_STATE_COUNT] = {
    16 * 3 * 33, 16 * 5 * 33, 16 * 11 * 33, 16 * 11 * 33, 16 * 5 * 33, 16 * 3 * 33, 6 * 1 * 16, 4 * 33 * 8,
};

static void *alloc16(size_t bytes) { return aligned_alloc(16, (bytes + 15) & ~(size_t)15); }

/* The accelerator: mag, real_part, imag_part + the eight states -> mask_erb. `state` is the registered buffer
 * (the eight tensors back to back, in GTCRNStateId order): the graph reads every state input before it writes
 * the outputs into the same memory. */
typedef int (*RunGraph)(const float *mag, const float *real_part, const float *imag_part, float *state,
                        float *mask_erb);

static int run_gtcrn(const float *erb_fwd, const float *erb_inv, RunGraph run_graph,
                     const float *in, float *out, long hops)
{
    const GTCRNGridSpec spec = { .n_fft = 512, .erb_low_bins = 65, .erb_high_bands = 64 };   /* config.ini */
    GTCRNGrid grid;
    if (gtcrn_grid_make(&grid, &spec) != GTCRN_GRID_OK) return -1;

    int rc = -1;
    size_t total = 0;
    for (int i = 0; i < GTCRN_STATE_COUNT; ++i) total += STATE_ELEMENTS[i];
    float *state = calloc(total, sizeof *state);       /* the eight tensors back to back (a float host) */
    float *buf = malloc(sizeof *buf * (4 * (size_t)grid.n_bins + 5 * (size_t)grid.erb_bands));
    float *spectrum = buf, *enhanced = spectrum + 2 * grid.n_bins;               /* interleaved [n_bins][2] */
    float *mag = enhanced + 2 * grid.n_bins, *real_part = mag + grid.erb_bands, *imag_part = real_part + grid.erb_bands;
    float *mask_erb = imag_part + grid.erb_bands;                                /* [erb_bands][2] */
    void *stft_mem = NULL, *istft_mem = NULL, *pool = NULL;
    AcStft *stft = NULL;
    AcIstft *istft = NULL;
    GTCRNProcess *p = NULL;
    if (!state || !buf) goto done;

    AcStftConfig sc;                      /* audiolib's defaults are GTCRN's framing: sqrt-Hann, center 0, scales 1.0f */
    ac_stft_config_default(&sc, grid.n_fft);
    size_t stft_bytes = ac_stft_get_mem_size(&sc), istft_bytes = ac_istft_get_mem_size(&sc);
    stft_mem = alloc16(stft_bytes);
    istft_mem = alloc16(istft_bytes);
    if (!stft_mem || !istft_mem) goto done;
    stft = ac_stft_init(stft_mem, stft_bytes, &sc);
    istft = ac_istft_init(istft_mem, istft_bytes, &sc);
    if (!stft || !istft) goto done;

    GTCRNProcessConfig cfg;
    GTCRNProcessMemReq req;
    if (gtcrn_process_config_defaults(&cfg, &grid) != 0) goto done;
    cfg.erb_fwd = erb_fwd;                /* [erb_high_bins][erb_high_bands], borrowed */
    cfg.erb_inv = erb_inv;                /* [erb_high_bands][erb_high_bins], borrowed */
    size_t offset = 0;
    for (int i = 0; i < GTCRN_STATE_COUNT; ++i) {      /* all eight or none */
        cfg.state[i] = state + offset;
        cfg.state_elements[i] = STATE_ELEMENTS[i];
        offset += STATE_ELEMENTS[i];
    }
    if (gtcrn_process_get_mem_size(&cfg, &req) != 0) goto done;
    pool = alloc16(req.bytes);
    p = pool ? gtcrn_process_init_ex(pool, (size_t)req.bytes, &cfg, &req) : NULL;
    if (!p) goto done;

    rc = 0;
    for (long h = 0; h < hops; ++h) {
        ac_stft_push(stft, in + h * grid.hop, (Complex *)(void *)spectrum);
        gtcrn_model_input(p, spectrum, mag, real_part, imag_part);
        for (int i = 0; i < 2 * grid.erb_bands; ++i) mask_erb[i] = NAN;      /* an unwritten mask is refused */
        int run_rc = run_graph(mag, real_part, imag_part, state, mask_erb);
        if (run_rc != 0) {
            /* run_graph() itself reported failure, but may already have overwritten mask_erb
             * and/or the registered state with finite-looking garbage before discovering that
             * (e.g. a half-finished run): mask_validate/state_validate only catch NON-FINITE
             * values, so a finite garbage write would otherwise be silently ACCEPTED (both rc
             * 0, generation not advanced) and carry that garbage into the next hop undetected --
             * the accelerator's own error code is the only signal that caught it. Poison both
             * with NaN here so the already-correct finite-check mechanism rejects this hop's
             * state exactly as if run_graph() had written nothing. */
            for (int i = 0; i < 2 * grid.erb_bands; ++i) mask_erb[i] = NAN;
            for (size_t i = 0; i < total; ++i) state[i] = NAN;
        }
        /* Both validators always run, even when run_graph() itself reports failure: the
         * NaN-filled mask_erb then still trips mask_validate's own finite check, which zeroes
         * the registered state and advances the generation -- skipping them here would let a
         * failed call's partial, finite state write (e.g. state[0] left non-zero) survive into
         * the next hop undetected. */
        int mask_rc = gtcrn_process_mask_validate(p, mask_erb);
        int state_rc = gtcrn_process_state_validate(p);
        if (run_rc != 0 || mask_rc != 0 || state_rc != 0)
            memcpy(enhanced, spectrum, sizeof *enhanced * 2 * (size_t)grid.n_bins);   /* pass the spectrum through */
        else
            gtcrn_model_output(p, mask_erb, spectrum, enhanced);
        ac_istft_push(istft, (const Complex *)(const void *)enhanced, out + h * grid.hop);
    }
done:
    free(pool); free(istft_mem); free(stft_mem); free(buf); free(state);
    return rc;
}
```

`gtcrn_process_mask_validate()` runs before `gtcrn_process_state_validate()`: a refusal of either advances the state generation, and
the pair advances it at most once. `GTCRN_PROCESS_CARVE_VERSION` is 1 and `GTCRN_MODEL_LAYOUT_VERSION` 8; the pool hash folds the
whole grid and both.

## ONNX, ERB tables and calibration

Install `../requirements-export.txt` in addition to the normal model
dependencies. Unlike the offline training graph, the exported ONNX consumes
one STFT frame and carries every temporal cache explicitly, so it can run
continuously without resetting model history:

```bash
python3 export_onnx.py --model output/gtcrn_best.pth \
  --output output/gtcrn_stream.onnx --verify
python3 inference.py calib --model output/gtcrn_best.pth \
  --wav-dir /path/to/noisy_wavs --frames 8192 --format bin \
  --output calib/gtcrn
python3 export_erb_matrix.py --model output/gtcrn_best.pth \
  --output-dir output/erb --format all
```

The C library reads no file: it takes pointers to the two matrices (`erb_fwd` bin-major `[192][64]`, `erb_inv` band-major
`[64][192]`) that the application loads and keeps alive while the library uses them. `export_erb_matrix.py --runtime-bins DIR`
writes them as `erb_fwd.bin` and `erb_inv.bin` (raw little-endian float32, plus `erb_matrices.json`); the `--output-dir`
formats are for toolchain validation.

The graph takes eleven inputs: `mag`, `real`, `imag`, six `conv_*` windows,
`h_tra` and `h_dpgrnn`. It returns
the ERB-domain complex mask and every updated state tensor on each call.
Calibration captures real pre-frame state values rather than repeating zero
state.
Use `--format npz --output calib/gtcrn.npz` when a NumPy archive is needed.
BIN output contains one folder per ONNX input and one numbered file per frame.
The deployment ONNX does not contain the fixed ERB front/back end:
`gtcrn_model_input()` applies the forward matrix before inference and
`gtcrn_model_output()` applies the inverse matrix plus the complex mask afterward.
The model consumes one new STFT frame per
invocation; GTCRN must not be padded to an artificial three-frame input
because its full temporal context already lives in the explicit state
tensors.

The eight state tensors belong to the application (`enum GTCRNStateId`: `GTCRN_STATE_CONV_ENC0` ... `GTCRN_STATE_CONV_DEC2`,
`GTCRN_STATE_H_TRA`, `GTCRN_STATE_H_DPGRNN`, in graph order). Bind each state input and its `*_out` output to the same buffer, so the
updated state is the next call's input and the CPU copies nothing between invocations. The runtime must read every state input
before it writes any state output at the same address. A float host registers its buffers in `GTCRNProcessConfig` (`state[8]` and
`state_elements[8]`, elements not bytes; all eight pairs or none; they are in neither the pool size nor the hash): init and
`gtcrn_process_reset()` zero them, and after each invocation `gtcrn_process_state_validate()` finite-checks all eight in place and
returns `int`: `0` when every element is finite or nothing is registered, `-1` on a null argument or on any NaN/Inf, in which
case all eight are zeroed (the state restarts from zero) because the previous value no longer exists. A partial or mixed write that
stays finite is not detectable. A board registers nothing and keeps the state in its accelerator.

The accelerator keeps its own fixed-point copy that a reset or a refused validation does not reach.
`gtcrn_process_state_generation()` returns 1 after `gtcrn_process_init()` and advances by one on every reset and every refused
validation (modulo 2^32), whether or not state is registered: a runtime that sees a value different from the one at the previous
frame refills its state tensors with quantized zeros.

## State boundary

State layout version 8. The exported metadata carries `state_layout_version`,
kept numerically equal to `GTCRN_MODEL_LAYOUT_VERSION` in `gtcrn_process.h`, so
an integrator can refuse a graph whose state layout no longer matches the
struct it allocated. Versions 3 to 7 are retired
(`RETIRED_LAYOUT_VERSIONS` in `export_onnx.py`). The metadata also records
`state_binding = in_place_same_address`, `state_handoff` and `state_slicing`.

Every state tensor is rank 4 and has the same shape at its input and at its
`*_out` output:

| tensor | shape | meaning |
| --- | --- | --- |
| `conv_enc0`, `conv_enc1`, `conv_enc2` | `(1, 16, 3 / 5 / 11, 33)` | full causal window of each encoder depthwise convolution |
| `conv_dec0`, `conv_dec1`, `conv_dec2` | `(1, 16, 11 / 5 / 3, 33)` | the same for the decoder, in reverse order |
| `h_tra` | `(1, 6, 1, 16)` | the six TRA attention GRUs, encoder 0-2 then decoder 0-2 |
| `h_dpgrnn` | `(1, 4, 33, 8)` | the four grouped inter GRUs, `h_dpgrnn1_0`, `h_dpgrnn1_1`, `h_dpgrnn2_0`, `h_dpgrnn2_1`; the hidden batch is the frequency axis |

A window input has exactly one consumer, a Slice of axis 2 with an explicit
integer range `[1:K]` and step 1. It drops the oldest frame; the graph appends
the new frame and returns the whole concatenation, so no state output is
produced by a Slice. The stacked GRU tensors are cut with Slices on dim 1 and
joined by one Concat on dim 1. `onnx_streaming_contract.py`
(`validate_inplace_state_graph`) checks all of this at export time;
`check_state_quantization.py` repeats the check on an exported graph and, given
a quantization report, aborts when a state input and its output differ in
dtype, scale, zero point or shape.

Total state is 21,216 floats (84,864 bytes), the application's. A window row is 33 elements, so on an int8 target the
axis-2 Slice starts at a multiple of 33 bytes; how the toolchain aligns it is
not verified on a board.
