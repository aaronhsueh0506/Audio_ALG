# Unified Audio Pipeline API（audio_pipeline.h）

## Context

AEC 和 NR 是兩個獨立 C 函式庫（libaec + libmmse_lsa），目前串接邏輯在 `aec_nr_pipeline.c` 的 `main()` 裡手動完成。需要包成統一 API，讓外部呼叫者可以：

1. **Initialize** — 建立完整 pipeline（preset 或自訂參數）
2. **Reset** — 重置濾波器到出廠預設（部署前 / echo path change 後）
3. **Runtime preset change** — 即時切換 AEC preset、NR 強度（state 歸零重收斂）
4. **Runtime parameter tuning** — 現場微調個別參數（即時生效，不丟 state）
5. **Process** — 餵 mic+ref，拿 output
6. **Query** — 取 ERLE、收斂狀態、delay 等
7. **Debug mode** — per-frame callback log + 中間信號 dump

## Branch

`feature/audio-pipeline-api`（三個 repo 都開：Audio_ALG、AEC、NR）

---

## 目前 API 現狀

| Module | Create | Process | Reset | Runtime Setter |
|--------|--------|---------|-------|---------------|
| AEC (libaec) | `aec_create(AecConfig*)` | `aec_process()` / `aec_process_ex()` | `aec_reset()` | **無** |
| NR (libmmse_lsa) | `mmse_lsa_create(MmseLsaConfig*)` | `mmse_lsa_process()` | `mmse_lsa_reset()` | **無** |
| RES (libaec) | `res_create(ResConfig*)` | `res_process()` | `res_reset()` | **無** |

三個模組建立後 config 不可變。需要新增 setter functions。

---

## API 設計

### Pipeline Modes

```
AEC_NR_RES:  Mic,Ref → HPF → AEC(linear) → NR → RES(NR-corrected) → Output
AEC_RES:     Mic,Ref → HPF → AEC + 內建 RES → Output
AEC_ONLY:    Mic,Ref → HPF → AEC(linear only) → Output
NR_ONLY:     Mic     → HPF → NR → Output（無 ref，無 echo cancellation）
```

Mode 在 `create` 時決定，不 runtime 切換。切模式需 `destroy` + `create`。

### 檔案清單

| 檔案 | 動作 | 說明 |
|------|------|------|
| `Audio_ALG/pipelines/audio_pipeline.h` | **新增** | 統一 API header |
| `Audio_ALG/pipelines/audio_pipeline.c` | **新增** | 實作：create/destroy/process/setters/getters/debug |
| `AEC/c_impl/include/res_filter.h` | **修改** | 新增 3 個 setter 宣告 |
| `AEC/c_impl/src/res_filter.c` | **修改** | 實作 setter（~15 行） |
| `NR/c_impl/include/mmse_lsa_denoiser.h` | **修改** | 新增 3 個 setter 宣告 |
| `NR/c_impl/src/mmse_lsa_denoiser.c` | **修改** | 實作 setter（~20 行） |
| `Audio_ALG/pipelines/Makefile` | **修改** | 加 `audio_pipeline` build target |

---

## 核心 API（audio_pipeline.h）

### 型別

```c
typedef struct AudioPipeline AudioPipeline;  /* opaque handle */

/* ====== Pipeline Mode（建立時決定，不可 runtime 切換）====== */
typedef enum {
    PIPELINE_AEC_NR_RES = 0,  /* AEC(linear) → NR → RES（完整）*/
    PIPELINE_AEC_RES    = 1,  /* AEC + 內建 RES（無 NR）*/
    PIPELINE_AEC_ONLY   = 2,  /* 純線性 AEC（無 RES、無 NR）*/
    PIPELINE_NR_ONLY    = 3   /* 純 NR（無 AEC、ref 忽略）*/
} PipelineMode;

/* ====== Pipeline Stage（debug 用）====== */
typedef enum {
    PIPELINE_STAGE_AEC = 0,   /* AEC linear output（RES 前）*/
    PIPELINE_STAGE_NR  = 1,   /* NR output（RES 前）*/
    PIPELINE_STAGE_RES = 2    /* RES output（最終）*/
} PipelineStage;

/* ====== Config ====== */
typedef struct {
    int sample_rate;             /* 16000 */
    PipelineMode mode;
    AecPreset    aec_preset;     /* MILD / BALANCED / AGGRESSIVE / MAXIMUM */
    MmseLsaNrMode nr_mode;       /* MILD / BALANCED / AGGRESSIVE */
    float res_g_min_db;          /* NAN = 用 preset 預設 */
    float nr_g_min_db;           /* NAN = 用 preset 預設 */
    int   enable_hpf;            /* 1 */
    float hpf_cutoff_hz;         /* 80.0 */
    int   enable_debug;          /* 0 = off, 1 = enable debug（保留中間信號）*/
} AudioPipelineConfig;
```

### Debug Callback

```c
/**
 * Per-frame debug callback — 每個 process() 呼叫後觸發。
 *
 * @param frame_idx     累計 frame 編號（從 0 開始）
 * @param info          本 frame 的內部指標快照
 * @param user_data     使用者自訂 context
 */
typedef struct {
    float erle;              /* 累計 ERLE (dB) */
    float erle_instant;      /* 瞬時 ERLE (dB) */
    float dt_indicator;      /* [0, 0.8] double-talk 信心值 */
    int   converged;         /* 0 or 1 */
    float divergence;        /* [0, 1] 發散指標 */
    int   delay_samples;     /* 估計延遲 (samples), -1 = 未知 */
    float far_power;         /* far-end 均功率 */
    float near_power;        /* near-end 均功率 */
    float error_power;       /* error 均功率 */
    int   n_freqs;           /* 頻率 bin 數 */
    const float* nr_gain;    /* [n_freqs] NR per-bin gain, NULL if no NR */
    const float* res_gain;   /* [n_freqs] RES per-bin gain, NULL if no RES */
    const float* echo_psd;   /* [n_freqs] echo PSD estimate */
    const float* error_psd;  /* [n_freqs] error PSD estimate */
    const float* noise_psd;  /* [n_freqs] NR noise PSD, NULL if no NR */
} AudioPipelineDebugInfo;

typedef void (*AudioPipelineDebugCb)(
    int frame_idx,
    const AudioPipelineDebugInfo* info,
    void* user_data
);
```

### Lifecycle

```c
AudioPipelineConfig audio_pipeline_default_config(int sample_rate);
AudioPipeline*      audio_pipeline_create(const AudioPipelineConfig* cfg);
void                audio_pipeline_destroy(AudioPipeline* p);
void                audio_pipeline_reset(AudioPipeline* p);  /* 重置濾波器，保留 config */
```

### Processing

```c
int audio_pipeline_process(AudioPipeline* p,
                           const float* mic,    /* [hop_size] */
                           const float* ref,    /* [hop_size], NULL for NR_ONLY */
                           float* output);      /* [hop_size] */
```

### Runtime Preset Switch（warm — 重建子模組，state 歸零重收斂）

```c
int audio_pipeline_set_aec_preset(AudioPipeline* p, AecPreset preset);
int audio_pipeline_set_nr_mode(AudioPipeline* p, MmseLsaNrMode mode);
/* Note: PipelineMode 不可 runtime 切換，只能 destroy + create */
```

### Runtime Parameter Tuning（hot — 即時生效，不丟 state）

```c
int audio_pipeline_set_res_g_min_db(AudioPipeline* p, float g_min_db);
int audio_pipeline_set_nr_g_min_db(AudioPipeline* p, float g_min_db);
int audio_pipeline_set_res_enr_scale(AudioPipeline* p, float enr_scale);
int audio_pipeline_set_res_over_sub(AudioPipeline* p, float base, float scale);
int audio_pipeline_set_nr_speech_prior(AudioPipeline* p, float q);
int audio_pipeline_set_nr_gain_smoothing(AudioPipeline* p,
                                         float alpha_g, float alpha_attack, float alpha_decay);
```

### Debug API

```c
/**
 * 註冊 debug callback（每次 process 後觸發）。
 * 設 cb=NULL 取消。需 enable_debug=1。
 */
void audio_pipeline_set_debug_cb(AudioPipeline* p,
                                  AudioPipelineDebugCb cb,
                                  void* user_data);

/**
 * 取得上一次 process() 的中間信號。
 * 需 enable_debug=1，否則回傳 -1。
 *
 * @param p       Pipeline handle
 * @param stage   要取哪一級的 output
 * @param buf     輸出 buffer [hop_size]
 * @return 0 成功, -1 無此 stage 或 debug 未啟用
 */
int audio_pipeline_get_stage_output(const AudioPipeline* p,
                                    PipelineStage stage,
                                    float* buf);

/**
 * 取得上一次 process() 的 debug info snapshot。
 * 需 enable_debug=1。
 *
 * @param p       Pipeline handle
 * @param info    輸出 info（caller 提供空間）
 * @return 0 成功, -1 debug 未啟用
 */
int audio_pipeline_get_debug_info(const AudioPipeline* p,
                                   AudioPipelineDebugInfo* info);
```

### Query

```c
int   audio_pipeline_get_hop_size(const AudioPipeline* p);
float audio_pipeline_get_erle(const AudioPipeline* p);
float audio_pipeline_get_erle_instant(const AudioPipeline* p);
int   audio_pipeline_is_converged(const AudioPipeline* p);
int   audio_pipeline_get_delay(const AudioPipeline* p);
PipelineMode  audio_pipeline_get_mode(const AudioPipeline* p);
AecPreset     audio_pipeline_get_aec_preset(const AudioPipeline* p);
MmseLsaNrMode audio_pipeline_get_nr_mode(const AudioPipeline* p);
const AecConfig* audio_pipeline_get_aec_config(const AudioPipeline* p);
const float*  audio_pipeline_get_nr_gain(const AudioPipeline* p, int* n_freqs);
const float*  audio_pipeline_get_noise_psd(const AudioPipeline* p, int* n_freqs);
```

---

## Internal Struct

```c
struct AudioPipeline {
    AudioPipelineConfig config;
    AecPreset     current_aec_preset;
    MmseLsaNrMode current_nr_mode;
    int           frame_idx;        /* 累計 frame 計數 */

    /* Sub-modules */
    Aec*             aec;           /* NULL if NR_ONLY */
    AecResContext*    ctx;           /* 當前 frame context（AEC_NR_RES 用）*/
    AecResContext*    prev_ctx;      /* 前一 frame（NR OLA 1-frame delay 對齊）*/
    int              have_prev_ctx;
    MmseLsaDenoiser* nr;            /* NULL if AEC_ONLY / AEC_RES */
    ResFilter*       res;           /* NULL if AEC_ONLY / NR_ONLY */
    Hpf*             hpf_mic;       /* NULL if HPF disabled */
    Hpf*             hpf_ref;       /* NULL if NR_ONLY or HPF disabled */

    /* Cached */
    int hop_size;
    int n_freqs;

    /* Pre-allocated processing buffers */
    float*   aec_out;               /* [hop_size] */
    float*   nr_out;                /* [hop_size] */
    float*   res_out;               /* [hop_size] */
    Complex* corrected_echo;        /* [n_freqs] */
    Complex* far_spec_c;            /* [n_freqs] */
    Complex* near_spec_c;           /* [n_freqs] */

    /* Debug（only allocated when enable_debug=1）*/
    int      debug_enabled;
    float*   dbg_aec_out;           /* [hop_size] copy of AEC stage output */
    float*   dbg_nr_out;            /* [hop_size] copy of NR stage output */
    float*   dbg_res_gain;          /* [n_freqs] last RES per-bin gain */
    AudioPipelineDebugInfo dbg_info; /* last frame's debug snapshot */
    AudioPipelineDebugCb   dbg_cb;
    void*                  dbg_user_data;
};
```

---

## 參數分類

| 類別 | 參數 | 機制 |
|------|------|------|
| **Hot（即時生效）** | `res_g_min_db`, `nr_g_min_db`, `res_enr_scale`, `res_over_sub_base/scale`, `nr_q`, `nr_alpha_g/attack/decay` | Setter → 修改子模組內部值，不 reset |
| **Warm（需 reset）** | AEC preset, NR mode, `kalman_q_*`, `shadow_*`, `warmup_frames` | Destroy + Recreate 子模組（create-before-destroy 確保失敗時不丟舊 state）|
| **Cold（需 reinit）** | `sample_rate`, `frame_size`, `fft_size`, `filter_length`, `mode` | 必須 `pipeline_destroy()` + `pipeline_create()` |

---

## 需要新增的子模組 Setter

### RES（加到 res_filter.h / res_filter.c）

```c
void res_set_g_min_db(ResFilter* res, float g_min_db);
void res_set_enr_scale(ResFilter* res, float enr_scale);
void res_set_over_sub(ResFilter* res, float base, float scale);
```

實作：直接修改 `res->g_min`、`res->enr_scale`、`res->over_sub_base/scale`。約 15 行。

### NR（加到 mmse_lsa_denoiser.h / mmse_lsa_denoiser.c）

```c
void mmse_lsa_set_g_min_db(MmseLsaDenoiser* self, float g_min_db);
void mmse_lsa_set_speech_prior(MmseLsaDenoiser* self, float q);
void mmse_lsa_set_gain_smoothing(MmseLsaDenoiser* self,
                                  float alpha_g, float alpha_attack, float alpha_decay);
```

實作：修改 `self->g_min`、SPP estimator 的 `q`、gain smoothing 參數。約 20 行。

---

## Preset Switch 策略：Create-Before-Destroy

```c
int audio_pipeline_set_aec_preset(AudioPipeline* p, AecPreset preset) {
    // 1. 用新 preset 建 config
    AecConfig new_cfg = aec_config_from_preset(preset, p->config.sample_rate);
    new_cfg.enable_res = (p->config.mode == PIPELINE_AEC_RES) ? 1 : 0;

    // 2. 建新 AEC（如果失敗，舊的還在）
    Aec* new_aec = aec_create(&new_cfg);
    if (!new_aec) return -1;

    // 3. 如果有 standalone RES，也重建
    ResFilter* new_res = NULL;
    if (p->res) {
        ResConfig rc = res_config_from_aec(&new_cfg);
        new_res = res_create(&rc);
        if (!new_res) { aec_destroy(new_aec); return -1; }
    }

    // 4. 成功 → swap
    aec_destroy(p->aec);    p->aec = new_aec;
    if (p->res) { res_destroy(p->res); p->res = new_res; }
    p->current_aec_preset = preset;
    p->have_prev_ctx = 0;  // reset delay compensation
    return 0;
}
```

---

## Debug Mode 實作細節

### enable_debug=1 時額外做的事

在 `audio_pipeline_process()` 內部：

```c
// AEC stage 完成後
if (p->debug_enabled) {
    memcpy(p->dbg_aec_out, p->aec_out, hop * sizeof(float));
}

// NR stage 完成後
if (p->debug_enabled) {
    memcpy(p->dbg_nr_out, p->nr_out, hop * sizeof(float));
}

// 收集 debug info
if (p->debug_enabled) {
    AudioPipelineDebugInfo* d = &p->dbg_info;
    d->erle = aec_get_erle(p->aec);
    d->erle_instant = aec_get_erle_instant(p->aec);
    d->converged = aec_is_converged(p->aec);
    d->delay_samples = aec_get_delay(p->aec);
    d->dt_indicator = p->prev_ctx ? p->prev_ctx->dt_indicator : 0;
    d->divergence = p->prev_ctx ? p->prev_ctx->divergence : 0;
    d->far_power = p->prev_ctx ? p->prev_ctx->far_power : 0;
    d->n_freqs = p->n_freqs;
    d->nr_gain = p->nr ? mmse_lsa_get_gain(p->nr, NULL) : NULL;
    d->echo_psd = p->res ? res_get_echo_psd(p->res) : NULL;
    d->error_psd = p->res ? res_get_error_psd(p->res) : NULL;
    d->noise_psd = p->nr ? mmse_lsa_get_noise_psd(p->nr, NULL) : NULL;

    // 觸發 callback
    if (p->dbg_cb) {
        p->dbg_cb(p->frame_idx, d, p->dbg_user_data);
    }
}
p->frame_idx++;
```

### Debug 記憶體開銷

| Buffer | Size @ 16kHz |
|--------|-------------|
| `dbg_aec_out` | 160 × 4 = 640 B |
| `dbg_nr_out` | 160 × 4 = 640 B |
| `dbg_res_gain` | 257 × 4 = 1028 B |
| `dbg_info` (struct) | ~100 B |
| **Total debug overhead** | **~2.4 KB** |

Debug mode 只在 `enable_debug=1` 時分配，正常模式零開銷。

---

## 實作步驟

### Phase 1：子模組 Setter（低風險，additive）
1. `res_filter.h/c` 加 3 個 setter
2. `mmse_lsa_denoiser.h/c` 加 3 個 setter
3. 編譯驗證不影響現有 API

### Phase 2：Pipeline 核心
1. 寫 `audio_pipeline.h`（完整 API 如上）
2. 寫 `audio_pipeline.c`：
   - `create` — 根據 mode 建立對應子模組 + buffers
   - `destroy` — 釋放所有子模組 + buffers
   - `process` — 4 種 mode 各自的處理流程
   - `reset` — 呼叫各子模組 reset
   - `default_config` — 回傳合理預設值

### Phase 3：Runtime Control
1. Preset switch（create-before-destroy pattern）
2. Hot-swap setters（直接轉發到子模組 setter）

### Phase 4：Debug Mode
1. Debug buffers 分配（在 create 中，enable_debug=1 時）
2. process 中 memcpy 中間信號
3. Debug info 收集 + callback 觸發
4. `get_stage_output` / `get_debug_info` query functions

### Phase 5：Makefile + 驗證
1. 更新 `pipelines/Makefile` 加 `libaudio_pipeline.a` target
2. 寫 `test_audio_pipeline.c` 用新 API 處理 WAV
3. 用 blind test WAV 驗證 AEC_NR_RES mode output 與原 `aec_nr_pipeline` bit-exact
4. 驗證 debug callback 正確輸出

---

## 使用範例

### 範例 1：完整 pipeline

```c
#include "audio_pipeline.h"

AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
cfg.mode = PIPELINE_AEC_NR_RES;
cfg.aec_preset = AEC_PRESET_BALANCED;
cfg.nr_mode = MMSE_LSA_NR_BALANCED;
AudioPipeline* pipe = audio_pipeline_create(&cfg);

float mic[160], ref[160], out[160];
while (read_audio(mic, ref, 160)) {
    audio_pipeline_process(pipe, mic, ref, out);
    write_audio(out, 160);
}
audio_pipeline_destroy(pipe);
```

### 範例 2：NR only

```c
AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
cfg.mode = PIPELINE_NR_ONLY;
cfg.nr_mode = MMSE_LSA_NR_AGGRESSIVE;
AudioPipeline* pipe = audio_pipeline_create(&cfg);

while (read_mic(mic, 160)) {
    audio_pipeline_process(pipe, mic, NULL, out);
    write_audio(out, 160);
}
audio_pipeline_destroy(pipe);
```

### 範例 3：現場調試 — 切 preset

```c
// 切換（state 歸零，重新收斂）
audio_pipeline_set_aec_preset(pipe, AEC_PRESET_AGGRESSIVE);
audio_pipeline_set_nr_mode(pipe, MMSE_LSA_NR_AGGRESSIVE);
```

### 範例 4：微調參數（不丟 state）

```c
audio_pipeline_set_res_g_min_db(pipe, -60.0f);
audio_pipeline_set_nr_g_min_db(pipe, -18.0f);
audio_pipeline_set_res_enr_scale(pipe, 0.7f);
```

### 範例 5：Debug mode — callback log

```c
void my_debug_cb(int frame_idx, const AudioPipelineDebugInfo* info, void* ud) {
    FILE* fp = (FILE*)ud;
    fprintf(fp, "frame=%d erle=%.1f dt=%.2f conv=%d div=%.2f delay=%d\n",
            frame_idx, info->erle, info->dt_indicator,
            info->converged, info->divergence, info->delay_samples);
}

AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
cfg.mode = PIPELINE_AEC_NR_RES;
cfg.enable_debug = 1;
AudioPipeline* pipe = audio_pipeline_create(&cfg);

FILE* log_fp = fopen("debug.log", "w");
audio_pipeline_set_debug_cb(pipe, my_debug_cb, log_fp);

while (read_audio(mic, ref, 160)) {
    audio_pipeline_process(pipe, mic, ref, out);
    write_audio(out, 160);
}
fclose(log_fp);
audio_pipeline_destroy(pipe);
```

### 範例 6：Debug mode — dump 中間信號

```c
AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
cfg.mode = PIPELINE_AEC_NR_RES;
cfg.enable_debug = 1;
AudioPipeline* pipe = audio_pipeline_create(&cfg);

float aec_stage[160], nr_stage[160];
while (read_audio(mic, ref, 160)) {
    audio_pipeline_process(pipe, mic, ref, out);

    // 取各級中間信號
    audio_pipeline_get_stage_output(pipe, PIPELINE_STAGE_AEC, aec_stage);
    audio_pipeline_get_stage_output(pipe, PIPELINE_STAGE_NR, nr_stage);
    // out 已是 RES stage output

    write_multi_channel(aec_stage, nr_stage, out, 160);
}
audio_pipeline_destroy(pipe);
```

### 範例 7：重置（echo path change）

```c
audio_pipeline_reset(pipe);  // 所有 filter 歸零，config 保留
```

---

## 記憶體 / 延遲

| 項目 | 值 |
|------|-----|
| AEC (PBFDKF + shadow + delay_est) | ~180 KB |
| NR (MMSE-LSA + MCRA + SPP) | ~60 KB |
| RES (WOLA + PSD) | ~25 KB |
| Pipeline overhead (buffers + context) | ~15 KB |
| Debug overhead (optional) | ~2.4 KB |
| **Total (AEC_NR_RES + debug)** | **~282 KB** |
| Pipeline latency | **20 ms**（AEC 10ms + NR OLA 10ms）|

---

## Error Codes

```c
#define APIPE_OK           0
#define APIPE_ERR_NULL    -1   /* NULL handle */
#define APIPE_ERR_ALLOC   -2   /* Memory allocation failed */
#define APIPE_ERR_INVALID -3   /* Invalid parameter value */
#define APIPE_ERR_MODE    -4   /* Operation not applicable to current mode */
#define APIPE_ERR_DEBUG   -5   /* Debug not enabled */
```

---

## 驗證

```bash
# 編譯
cd Audio_ALG/pipelines
make audio_pipeline

# 用 blind test WAV 驗證（AEC_NR_RES mode 應與原 pipeline bit-exact）
./test_audio_pipeline mic.wav ref.wav out_api.wav
./aec_nr_pipeline mic.wav ref.wav out_manual.wav

python3 -c "
import numpy as np, soundfile as sf
a, _ = sf.read('out_api.wav')
b, _ = sf.read('out_manual.wav')
print(f'corr={np.corrcoef(a,b)[0,1]:.6f}')  # 目標 = 1.000000
"

# 驗證 debug log
./test_audio_pipeline mic.wav ref.wav out.wav --debug
# 檢查 debug.log 內容
```
