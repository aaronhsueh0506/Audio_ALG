> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_04_errors.md) ｜ [下一份](integration_4ch_spatial_zh_TW_06_checklist.md) <!-- split-nav -->
## 6. 出貨內容 vs 範例

`pipelines/4ch_aec_bf_nr_res/` 目錄不是自解釋的。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c` | `libaudio_pipeline_4ch.a`（object 之一） | **本手冊的主體**（共用 delay 對齊 + 四條線性 AEC + SRP-PHAT + BF 單元；不含 RES／NR／合成） |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c` | `libaudio_pipeline_4ch.a`（object 之二） | 核心層，本層一定會用到 |
| `pipelines/common/se_config.c` | `libaudio_pipeline_4ch.a`（object 之三） | 共用組態 |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h` | — | 本手冊的公開 API |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.h` | — | 核心層公開 API（本 header 會 include 它） |
| `<audiolib>/doa/` | `libdoa.a` | SRP-PHAT / steering / DOA 平滑。可重用函式庫 |
| `<audiolib>/bf/` | `libbf.a` | BF beamformer。可重用函式庫 |
| `<audiolib>/common/utility/` | `libspatial_common.a` | spatial kernel 與 complex 輔助 |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_internal.h`、`4aec_projection_kernels.h` | — | 內部 header，**不是**公開 API，不要 include |

### 6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/samples/se_anr_stage.h`、`ulcnet_nn_stage.{h,c}`、`dfn_hop.h` | header-only／單檔範例碼 | NR 級的 gain 呼叫與 NN stage：沒有 ABI、沒有 layout version、不進任何 archive，可整份複製改寫 |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_static.c` | caller-owned pool 路徑的主機參考程式 | 示範單元加 RES（libaec `AecRes`）／NR／合成的 query → allocate → init_ex → process → fuse → 後級 → destroy → release（RES、NR、合成與暫存另有一塊 pool）。真正的 SRP-PHAT + BF 在這個 process 裡跑，不是固定權重的替身 |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_static.c` | **核心層**（外部 beamformer seam）的 pool 路徑參考程式 | 屬於另一份手冊。它在 front/fuse 之間塞的是固定等權重的 smoke adapter，不是 production beamformer |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c` | raw-float 錄音驗證 runner（單元加 RES／NR／合成） | 主機一次性工具，刻意走 heap `create()` 路徑（不是板端佈署示範） |

單聲道的 library 與參考程式（`pipelines/mono_aec_nr_res/main.c`、
`static_main.c`、`example_board_adapter.c`）見 `docs/integration_mono_zh_TW.md`。

### 6.3 其他（不出貨）

* `pipelines/4ch_aec_bf_nr_res/tests/` —— 測試。
* `.py` 檔（`pipeline.py`、`evaluate_recordings.py`、`evaluate_external_recordings.py`）
  —— 主機端評估與參考實作，不是 C 產品的一部分。

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_04_errors.md) ｜ [下一份](integration_4ch_spatial_zh_TW_06_checklist.md) <!-- split-nav -->
