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
| filter_length | 256 | 512 | 1536 | sr × 32ms |
| n_partitions | 4 | 4 | 4 | ceil(filter_length / hop) |

## Latency & Performance

| 項目 | 數值 | 說明 |
|------|------|------|
| **Algorithmic latency** | 10 ms | 1 hop（所有 sample rate 一致） |
| **NR OLA delay** | +10 ms | NR frame 處理引入額外 1 hop 延遲 |
| **Pipeline total latency** | **20 ms** | AEC hop + NR OLA delay |
| **Processing (per hop)** | < 0.5 ms | AEC + NR + RES 合計（ARM Cortex-A53 @ 1GHz 估計） |
| **RTF** | < 0.05 | 遠低於即時要求 |

### Memory Budget

| Sample Rate | AEC | Context×2 | NR | RES | Buffers | **Total** |
|-------------|-----|-----------|-----|-----|---------|-----------|
| **8 kHz** | 61.7 KB | 6.3 KB | 49.0 KB | 21.5 KB | 4.6 KB | **143.1 KB** |
| **16 kHz** | 120.7 KB | 12.3 KB | 96.3 KB | 41.8 KB | 9.2 KB | **280.4 KB** |
| **48 kHz** | 240.4 KB | 24.3 KB | 194.8 KB | 86.3 KB | 21.4 KB | **567.4 KB** |

> `filter_length=sr×32ms`。若需更長 echo path，增加 `filter_length` 會等比增加 AEC 記憶體。

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | 8000 / 16000 / 48000，自動計算 frame/fft/hop |
| `filter_length` | sr×32ms | 自適應濾波器長度（256@8k, 512@16k, 1536@48k） |
| `enable_highpass` | 1 | 高通濾波器（移除 DC + 低頻） |
| `highpass_cutoff_hz` | 80.0 | HPF 截止頻率 (Hz) |

**RES 參數**（`ResConfig`）：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `res_g_min_db` | -55.0 | 最小增益 dB（最大抑制量） |
| `res_over_sub_base` | 5.0 | 過減因子基底 |
| `res_over_sub_scale` | 9.0 | ERLE 連動的過減縮放 |
| `res_dt_reduction` | 2.5 | Double-talk 時降低過減量 |
| `res_spectral_floor_db` | -38.0 | 頻譜底噪 dB（CNG） |
| `res_ne_protect_db` | -16.0 | 近端保護閾值 dB |
| `res_enr_scale` | 0.85 | ENR gain 縮放（<1 更積極） |
| `res_enable_reverb` | 1 | 殘響尾部估計 |
| `res_reverb_decay` | 0.65 | 殘響衰減係數 |
| `res_reverb_gain` | 1.4 | 殘響增益 |

### NR (`MmseLsaConfig`, see `mmse_lsa_types.h`)

**Modes**: `MILD` / `BALANCED`（default）/ `AGGRESSIVE`

| Parameter | Balanced | Mild | Aggressive | Description |
|-----------|----------|------|------------|-------------|
| `g_min_db` | -15 | -10 | -20 | 最小增益 dB（最大抑制量） |
| `q` | 0.50 | 0.60 | 0.35 | 語音先驗機率（低→積極抑噪） |
| `xi_min_db` | -20 | -15 | -25 | 先驗 SNR 下限 dB |
| `alpha_g` | 0.88 | 0.92 | 0.75 | 增益時間平滑（高→平滑） |
| `alpha_attack` | 0.30 | 0.40 | 0.15 | Attack 平滑（語音起始追蹤） |
| `alpha_decay` | 0.88 | 0.92 | 0.85 | Decay 平滑（噪聲釋放） |

**MCRA 噪聲估計**：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_s` | 0.95 | 功率譜時間平滑 |
| `alpha_d` | 0.70 | 噪聲更新速率 |
| `L` | 32 | 最小值追蹤視窗（幀數，×10ms = 320ms） |
| `num_init_frames` | 20 | 初始化靜默幀數（200ms） |
| `scene_change_threshold_db` | 10.0 | 場景轉換偵測閾值 |

**SPP**：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_xi` | 0.88 | Decision Directed 先驗 SNR 平滑 |

---

## Troubleshooting & Tuning Guide

### AEC 相關

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **殘留回聲明顯** | RES 抑制不足 | 降低 `res_g_min_db`（如 -55→-65），提高 `res_over_sub_base`（如 5→7） |
| **殘留回聲 + 遠端持續講話** | Filter 未完全收斂 | 增加 `filter_length`（如 1024→2048），確認 mic-ref delay < filter_length |
| **殘留回聲尾部（殘響感）** | 殘響估計不足 | 提高 `res_reverb_gain`（如 1.4→2.0），提高 `res_reverb_decay`（如 0.65→0.75） |
| **近端語音被壓制（DT degradation）** | RES 過度抑制 | 提高 `res_g_min_db`（如 -55→-35），提高 `res_dt_reduction`（如 2.5→4.0），提高 `res_ne_protect_db`（如 -16→-10） |
| **輸出底噪不自然（突然靜音）** | CNG 底噪太低 | 提高 `res_spectral_floor_db`（如 -38→-25） |
| **收斂太慢** | Kalman Q 太保守 | 提高 `kalman_q_high`（如 1e-3→2e-3），減少 `warmup_frames` |
| **Filter 發散（輸出爆音）** | Kalman Q 太激進或 echo path 劇變 | 降低 `kalman_q_high`（如 1e-3→5e-4） |
| **Echo path 變化後適應慢** | Shadow filter 太保守 | 提高 `shadow_q_ratio`（如 3.5→5.0），降低 `shadow_copy_threshold`（如 0.7→0.5） |

### NR 相關

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **噪聲殘留太多** | 抑制量不夠 | 降低 `g_min_db`（如 -15→-20），降低 `q`（如 0.5→0.35） |
| **語音被吃掉** | 抑制太激進 | 提高 `g_min_db`（如 -15→-10），提高 `q`（如 0.5→0.6），提高 `alpha_g`（如 0.88→0.92） |
| **Musical noise（隨機顆粒噪聲）** | 增益抖動 | 提高 `alpha_g`（增益更平滑），提高 `alpha_decay`（釋放更慢） |
| **語音起始被截斷** | Attack 太慢 | 降低 `alpha_attack`（如 0.3→0.15），讓增益快速回升 |
| **噪聲環境切換後適應慢** | MCRA 追蹤窗太長 | 減小 `L`（如 32→16），但會增加噪聲底噪估計抖動 |
| **初始化期語音被壓** | 噪聲底噪估計偏高 | 減少 `num_init_frames`（如 20→10），但需確保前段有足夠噪聲 |
| **穩態噪聲殘留（風扇聲）** | 噪聲更新太慢 | 降低 `alpha_d`（如 0.7→0.5），讓噪聲估計更快跟上 |
| **語音段噪聲估計上升** | SPP 平滑不足 | 提高 `alpha_xi`（如 0.88→0.95），讓 SPP 更穩定判別語音 |

### Pipeline 整體

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **回聲消了但底噪變大** | NR 沒開或太保守 | 確認 NR mode 非 MILD，或降低 `g_min_db` |
| **NR 把回聲當噪聲學進去** | AEC 殘留回聲被 MCRA 當底噪 | 先確保 AEC 收斂良好，再調 NR。提高 `num_init_frames` 讓 MCRA 避開 AEC 收斂期 |
| **整體語音品質差（悶、失真）** | 多階段過度處理 | 改用 MILD preset（AEC + NR 都放鬆），只在必要時加強個別模組 |
| **處理 48kHz 音訊記憶體不足** | 模組記憶體隨 fft_size 增長 | 縮短 `filter_length`、減小 NR `L`（主要記憶體佔用） |

