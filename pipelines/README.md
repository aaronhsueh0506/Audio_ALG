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

All modules use unified 20ms frame / 10ms hop, auto-configured by sample rate:

| Parameter | 8 kHz | 16 kHz | 48 kHz | Formula |
|-----------|-------|--------|--------|---------|
| frame_size | 160 | 320 | 960 | sr × 20ms |
| hop_size | 80 | 160 | 480 | frame / 2 |
| fft_size | 256 | 512 | 1024 | next pow2 ≥ frame |
| n_freqs | 129 | 257 | 513 | fft/2 + 1 |

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

## Tunable Parameters

### AEC (`AecConfig`, see `aec_types.h`)

**Presets**: `MILD` / `BALANCED`（default）/ `AGGRESSIVE` / `MAXIMUM`

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `sample_rate` | 16000 | 8000, 16000, 48000 | 自動計算 frame/fft/hop size |
| `filter_length` | 1024 | 512, 1024, ... | 自適應濾波器長度（samples），CLI 依 mode/sr 自動設定 |
| `enable_highpass` | 1 | 0, 1 | 高通濾波器開關 |
| `highpass_cutoff_hz` | 80.0 | 50–200 | HPF 截止頻率 |
| `enable_res` | 0 (pipeline) | 0, 1 | 殘留回聲抑制（pipeline 中由外部 RES 處理） |
| `delta` | 1e-8 | — | 正則化常數，防止除零 |

**RES 參數**（pipeline 中由 `ResConfig` 控制）：

| Parameter | Balanced | Range | Description |
|-----------|----------|-------|-------------|
| `res_g_min_db` | -55.0 | -60 ~ -25 | 最小增益（最大抑制量） |
| `res_over_sub_base` | 5.0 | 1.0–10.0 | 過減因子基底 |
| `res_over_sub_scale` | 9.0 | 2.0–15.0 | ERLE 相關的過減縮放 |
| `res_dt_reduction` | 2.5 | 1.0–5.0 | Double-talk 時降低抑制量 |
| `res_spectral_floor_db` | -38.0 | -45 ~ -20 | 頻譜底噪（CNG） |
| `res_ne_protect_db` | -16.0 | -20 ~ -8 | 近端保護閾值 |
| `res_enr_scale` | 0.85 | 0.5–1.5 | ENR gain 縮放因子 |
| `res_enable_reverb` | 1 | 0, 1 | 殘響尾部估計 |
| `res_reverb_decay` | 0.65 | 0.3–0.9 | 殘響衰減係數 |
| `res_reverb_gain` | 1.4 | 0.5–2.0 | 殘響增益 |

**Shadow filter / Kalman 參數**（進階，通常不需調整）：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shadow_q_ratio` | 3.5 | Shadow 正則化比率（相對 main） |
| `shadow_copy_threshold` | 0.7 | Shadow→main 複製閾值 |
| `kalman_q_high` | 1e-3 | Kalman 處理噪聲上界 |
| `kalman_q_low` | 1e-5 | Kalman 處理噪聲下界 |
| `warmup_frames` | 80 | 收斂前暖機幀數 |

### NR (`MmseLsaConfig`, see `mmse_lsa_types.h`)

**Strength modes**: `MILD` / `BALANCED`（default）/ `AGGRESSIVE`

| Parameter | Balanced | Mild | Aggressive | Description |
|-----------|----------|------|------------|-------------|
| `g_min_db` | -15.0 | -10.0 | -20.0 | 最小增益（最大抑制量 dB） |
| `q` | 0.50 | 0.60 | 0.35 | 語音先驗機率（越低越積極抑噪） |
| `xi_min_db` | -20.0 | -15.0 | -25.0 | 先驗 SNR 下限 |
| `alpha_g` | 0.88 | 0.92 | 0.75 | 增益時間平滑（越高越平滑） |
| `alpha_attack` | 0.30 | 0.40 | 0.15 | 非對稱平滑 — attack（語音起始快速追蹤） |
| `alpha_decay` | 0.88 | 0.92 | 0.85 | 非對稱平滑 — decay（噪聲緩慢釋放） |

**MCRA 噪聲估計參數**：

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `alpha_s` | 0.95 | 0.8–0.99 | 功率譜時間平滑 |
| `alpha_d` | 0.70 | 0.5–0.9 | 噪聲更新速率 |
| `alpha_p` | 0.20 | 0.1–0.4 | SPP indicator 平滑 |
| `L` | 32 | 5–150 | 最小值追蹤視窗（幀數，×10ms） |
| `delta_db` | 10.0 | 5.0–15.0 | 偏差補償 |
| `num_init_frames` | 20 | 10–50 | 初始化靜默幀數（×10ms 為初始化期） |

**SPP 參數**：

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `alpha_xi` | 0.88 | 0.7–0.98 | Decision Directed 先驗 SNR 平滑 |

**場景轉換偵測**：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scene_change_threshold_db` | 10.0 | 高頻 γ 超過此閾值觸發偵測 |
| `scene_change_min_frames` | 5 | 連續超閾值幀數才觸發（50ms） |
| `scene_change_blend` | 0.5 | 噪聲重估混合因子 |

