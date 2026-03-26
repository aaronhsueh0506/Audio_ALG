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
make libs           # Version A (submodule libs)
make libs-static    # Version B (SE/ repo libs on feature/static-memory)

# Build pipeline
make                # Builds both versions

# Run Version A (malloc)
./aec_nr_pipeline mic.wav ref.wav output.wav balanced
./aec_nr_pipeline mic.wav ref.wav output.wav --aec-only
./aec_nr_pipeline mic.wav ref.wav output.wav aggressive --nr-gain -20

# Run Version B (static memory)
./aec_nr_pipeline_static mic.wav ref.wav output.wav balanced
./aec_nr_pipeline_static --print-mem-size              # Print memory budget only
./aec_nr_pipeline_static --print-mem-size aggressive   # With preset
```

## Two Versions

### Version A: malloc (`aec_nr_pipeline.c`)
Each module uses `_create()` / `_destroy()` and manages its own memory internally.
Suitable for desktop testing and Linux servers.

### Version B: static memory (`aec_nr_pipeline_static.c`)

On branch: `feature/static-memory` (all three repos: AEC, NR, Audio_ALG)

Single pre-allocated memory pool, no internal malloc:

1. Query each module's memory requirement: `_get_mem_size()`
2. Allocate one contiguous pool (malloc on desktop, PA/VA on Novatek)
3. Slice pool via pointer arithmetic, init each module: `_init()`
4. Process frames (identical logic to Version A)
5. Free the single pool at cleanup

**Static memory API pattern** (every module follows this):

```c
// Query memory size needed
size_t aec_get_mem_size(const AecConfig* config);

// Initialize in pre-allocated memory (no malloc inside)
Aec* aec_init(void* mem, size_t mem_size, const AecConfig* config);

// Destroy is no-op for static (is_static flag)
void aec_destroy(Aec* aec);
```

**Modules with static memory support:**

| Module | `_get_mem_size()` | `_init()` | Sub-modules |
|--------|-------------------|-----------|-------------|
| AEC | `aec_get_mem_size()` | `aec_init()` | HPF, PBFDKF x2, RES (optional), FFT |
| NR | `mmse_lsa_get_mem_size()` | `mmse_lsa_init()` | MCRA, SPP, FFT |
| RES | `res_get_mem_size()` | `res_init()` | FFT |
| Context | `aec_context_get_mem_size()` | `aec_context_init()` | — |
| PBFDKF | `pbfdkf_get_mem_size()` | `pbfdkf_init()` | FFT |
| HPF | `hpf_get_mem_size()` | `hpf_init()` | — |
| MCRA | `mcra_get_mem_size()` | `mcra_init()` | — |
| SPP | `spp_get_mem_size()` | `spp_init()` | — |
| FFT | `fft_get_mem_size()` | `fft_init()` | kiss_fft |

**Novatek integration:**

```c
// Replace malloc with PA/VA allocation:
// void* pool = malloc(total);
uint32_t pa;
void* pool = (void*)nvt_mem_alloc(total, &pa);
// pa = physical address (for DMA), pool = virtual address (for CPU)

// Cleanup:
// free(pool);
nvt_mem_free(pool, pa);
```

### Memory Budget (16kHz, verified)

| Module | Size |
|--------|------|
| AEC (linear, no RES) | 181.5 KB |
| AEC Context x2 | 12.3 KB |
| NR (MMSE-LSA + MCRA) | 214.8 KB |
| RES (standalone) | 41.8 KB |
| Pipeline buffers | 9.2 KB |
| **Total** | **459.7 KB** |

Run `./aec_nr_pipeline_static --print-mem-size` to get exact byte counts.

### Verification

Version A and Version B produce **bit-exact** identical output.
Both versions have been tested and confirmed to match sample-for-sample.
