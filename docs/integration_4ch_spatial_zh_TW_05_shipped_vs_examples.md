> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_04_errors.md) ｜ [下一份](integration_4ch_spatial_zh_TW_06_checklist.md) <!-- split-nav -->
## 6. 出貨內容 vs 範例

`pipelines/4ch_aec_bf_nr_res/` 目錄不是自解釋的。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/mcab.c` | `libaudio_pipeline_mcab.a`（唯一的 object） | flow 層：共用 delay admission、對齊環、四條注入 lane 的 front 與 `mcab_fuse()` 投影；**不含** beamformer、RES／NR／合成 |
| `pipelines/4ch_aec_bf_nr_res/mcab.h` | — | mcab 的公開 API（本手冊 helper 與所有範例都以它為基礎） |
| `lib/aec/c_impl`（`libaec.a`） | `libaec.a` | `Aec` lane 與 `DelayAec` estimator 的實作；物件由你建立、保管，注入 mcab |
| `<audiolib>/doa/` | `libdoa.a` | SRP-PHAT / steering / DOA 平滑。可重用函式庫（helper 需要） |
| `<audiolib>/bf/` | `libbf.a` | BF beamformer。可重用函式庫（helper 需要） |
| `<audiolib>/common/utility/` | `libspatial_common.a` | spatial kernel 與 complex 輔助（helper 需要） |
| `pipelines/4ch_aec_bf_nr_res/mcab_internal.h` | — | 內部 header（admission 狀態機、診斷 hook、投影 kernel），**不是**公開 API，不要 include |

### 6.2 範例碼與參考執行檔（**沒有任何一個會進產品**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/mcab_sample_beam.{h,c}` | **本手冊的 helper**（SRP-PHAT + BF + 備援 VAD 的接線） | 範例碼：沒有 ABI、沒有 layout version、不進任何 archive（Makefile 編成 `mcab_sample_beam.o`）。有自己 beamformer 的產品不需要它；板端 sample 可以像它一樣自己內聯 `doa_step()` + `bf_process_with_weights()` 兩個呼叫 |
| `pipelines/4ch_aec_bf_nr_res/mcab_sample_flow.{h,c}` | 應用端 init 順序的 builder | 把四條 lane、`DelayAec`、mcab pool 的建立與拆除寫在一處（`mcab_sample_flow_*`）；測試與 DFN／ULCNet 範例使用 |
| `pipelines/4ch_aec_bf_nr_res/mcab_sample_driver.{h,c}` | CLI 與 hop 迴圈骨架 | `mcab_raw`、`mcab_static`、`4ch_alignulcnet`、`4ch_aec_bf_dfn_res` 共用。`mcab_raw` 的數值選項由 `common/example_cli.h` 的 `example_cli_parse_int()`／`example_cli_parse_float()` 嚴格解析（整個字串必須是一個數字：拒絕空字串、開頭空白、多餘字元、超出 `int` 範圍，float 另拒絕 inf／nan）；`mcab_static` 以 `raw=0` 沿用 `atoi`／`atof`，不拒絕格式錯誤的數字 |
| `pipelines/4ch_aec_bf_nr_res/mcab_sample_post.h`、`pipelines/ulcnet/ulcnet_nn_stage.{h,c}`、`pipelines/dfn/dfn_hop.h` | header-only／單檔範例碼 | `mcab_sample_post.h`：`McabFrame` 到 RES 輸入（`mcab_sample_res_input()`）、NR 增益（`mcab_sample_anr_gain()`）與 comfort-noise 常數；ULCNet／DFN2：NN stage。沒有 ABI、沒有 layout version、不進任何 archive，可整份複製改寫 |
| `pipelines/4ch_aec_bf_nr_res/mcab_static.c`（binary `mcab_static`） | caller-owned pool 路徑的主機參考程式 | 示範 flow 加 helper 加 RES（libaec `AecRes`）／NR／合成的 query → allocate → init → process → fuse → 後級 → destroy → release（lane、estimator、mcab、helper 在一塊 pool，RES、NR、合成與暫存另有一塊）。真正的 SRP-PHAT + BF 在這個 process 裡跑，不是固定權重的替身。建置：`make -C pipelines/4ch_aec_bf_nr_res mcab_static` |
| `pipelines/4ch_aec_bf_nr_res/mcab_external_static.c`（binary `mcab_external_static`） | **不用 helper** 的 pool 路徑參考程式 | 示範 mcab 單獨使用：自己建 lane 與 estimator，在 front／fuse 之間塞的是固定等權重的 smoke adapter，不是 production beamformer。`--sample-rate`／`--fft-size`／`--capture-proxy` 以 `atoi` 解析，不拒絕格式錯誤的數字 |
| `pipelines/4ch_aec_bf_nr_res/mcab_raw.c`（binary `mcab_raw`） | raw-float 錄音驗證 runner（flow 加 helper 加 RES／NR／合成） | 主機一次性工具：輸入 raw float32 mic／ref，輸出 mono float32，結尾印一行 JSON；`--vad-u8` 提供產品 VAD。數值選項嚴格解析，錯誤的數字（例如空的 `--fixed-doa-deg`）以 exit 2 拒絕；輸出檔寫入失敗時印出訊息並以非 0 結束 |
| `pipelines/ulcnet/4ch_alignulcnet/main.c`、`pipelines/dfn/4ch_aec_bf_dfn_res/main.c` | 4ch 加 NN 後級的範例 | 用 `mcab_sample_flow` 加 helper 作前端（見各自的 README；建置轉發到 `4ch_aec_bf_nr_res/Makefile`）。兩個範例同樣嚴格解析數字選項（含 delay 選項），`12x`、`3x`、`12dB`、`16000Hz` 這類字串以 exit 2 拒絕 |

單聲道的 library 與參考程式（`pipelines/mono_aec_nr_res/main.c`——同一份原始碼建出 `aec_nr_pipeline_static` 與
`-DAEC_NR_POOL_SPLIT=1` 的 `aec_nr_pipeline` 兩個 binary——以及 `example_board_adapter.c`）見 `docs/integration_mono_zh_TW.md`。

### 6.3 其他（不出貨）

* `pipelines/4ch_aec_bf_nr_res/tests/` —— 測試：`test_mcab.c`（flow）、`test_mcab_sample_beam.c`（helper）、`test_mcab_sample_post.c`（`mcab_sample_post.h`）、`test_mcab_sample_driver.c`、
  `mcab_test_rig.h`、`conv_chain_mcab.h`（獨立寫出的參考鏈）、`dump_delay_parity.c`，以及 Python 側的 `test_pipeline.py`、`test_delay_parity.py`。
* `.py` 檔（`pipeline.py`、`evaluate_recordings.py`、`evaluate_external_recordings.py`）
  —— 主機端評估與參考實作，不是 C 產品的一部分。`pipeline.py` 是 mcab 的 Python 雙胞胎（`McabPipeline(config=None, beamformer=None, lane_config=None)`，
  以及 `mcab_apply_lane_contract()`／`mcab_lane_config()`；它不含 SRP-PHAT／BF，波束器由呼叫端提供）。

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_04_errors.md) ｜ [下一份](integration_4ch_spatial_zh_TW_06_checklist.md) <!-- split-nav -->
