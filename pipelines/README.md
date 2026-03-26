# C Pipeline: Linear AEC → NR → RES

## Architecture

```
mic ─┐                       ┌─ aec_out ──┐              ┌─ nr_out ──┐                  ┌─ output
     ├→ AEC (linear) ────────┤            ├→ NR (MMSE) ──┤           ├→ RES (post) ─────┤
ref ─┘   PBFDKF+Shadow      └─ context   ┘  LSA+MCRA    └─ gain[]   ┘  echo×nr_gain    └─ final
```

## Modules

| Module | Library | Header | Function |
|--------|---------|--------|----------|
| AEC | libaec.a | aec.h, aec_types.h | PBFDKF adaptive filter + shadow filter |
| NR | libmmse_lsa.a | mmse_lsa_denoiser.h | MMSE-LSA + MCRA noise est + SPP |
| RES | libaec.a (included) | res_filter.h | Residual echo suppression (WOLA) |

## Parameter Alignment

All modules use unified 10ms hop:

| Parameter | Value | Note |
|-----------|-------|------|
| frame_size | 320 | 20ms @ 16kHz |
| hop_size | 160 | 10ms @ 16kHz |
| fft_size | 512 | next pow2 >= 320 |
| n_freqs | 257 | fft_size/2 + 1 |

## Integration Flow

1. **AEC (linear)**: Set `enable_res=0`, use `aec_process_ex()` to get context
2. **NR**: `mmse_lsa_process()` for denoising, `mmse_lsa_get_gain()` for per-bin gain
3. **RES**: Correct echo PSD with `echo_spec *= nr_gain`, then `res_process()`

### Echo PSD Correction

```c
const float* gain = mmse_lsa_get_gain(nr, NULL);
for (int k = 0; k < n_freqs; k++) {
    corrected_echo[k].re = ctx->echo_spec_re[k] * gain[k];
    corrected_echo[k].im = ctx->echo_spec_im[k] * gain[k];
}
res_process(res, nr_out, corrected_echo, ...);
```

NR already attenuated certain frequency bins. The echo PSD estimate must
reflect this, otherwise RES will over-suppress (seeing echo that NR already
removed). Multiplying by the NR gain corrects for this.

## NR OLA Delay

NR uses OLA (frame_size=320, hop=160), introducing 1-frame (10ms) delay.
The pipeline saves the previous AEC context and uses it when the
corresponding NR output becomes available.

## Build

```bash
# Build libraries (from Audio_ALG/pipelines/)
make libs

# Build pipeline
make

# Run
./aec_nr_pipeline mic.wav ref.wav output.wav balanced
./aec_nr_pipeline mic.wav ref.wav output.wav --aec-only
./aec_nr_pipeline mic.wav ref.wav output.wav aggressive --nr-gain -20
```

## Two Versions

### Version A: malloc (this file)
Each module uses `_create()` / `_destroy()` and manages its own memory.
Suitable for desktop testing and Linux servers.

### Version B: static memory (Novatek embedded)
- `_get_mem_size()` pre-calculates memory needed per module
- Pipeline allocates total once via PA/VA
- `_init()` assigns pre-allocated buffer, no internal malloc
- On separate branch: `feature/static-memory`

### Memory Budget (Version B, 16kHz)

| Module | Size |
|--------|------|
| AEC (linear, no RES) | ~185 KB |
| NR (MMSE-LSA + MCRA) | ~218 KB |
| RES (standalone) | ~42 KB |
| Pipeline buffers | ~10 KB |
| **Total** | **~455 KB** |
