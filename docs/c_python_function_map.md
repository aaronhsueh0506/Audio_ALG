# C 與 Python 函式對照表（pipelines、audiolib、AINR、AIAEC）

- 範圍：本檔所在的 Audio_ALG 樹，加上同層的 audiolib 檢出。表內所有路徑都相對於 Audio_ALG 根目錄；`audiolib/common/…` 指同層的 audiolib 檢出（pipelines/Makefile 的 `COMMON_DIR`）。
- 守衛測試：`pipelines/tests/test_c_python_map.py`
- 語言：繁體中文說明，英文識別字。

## 1. 用途

算法在 Python 端開發，C 要先與 Python 的「效果」一致，之後才能優化或改寫。要做到這件事，必須先知道「C 的哪個函式在 Python 對應哪個符號」。本表以**符號名**（不用行號，行號會漂移）提供函式級追溯，並且不寫進 C 檔。

本表回答四件事：

1. 某個 C 函式有沒有 Python 對應，是哪一個符號、對應到什麼程度（`mirror`／`partial`）。
2. 沒有對應的是哪些（`C-only`，以及只在 Python 的 `Python-only`），避免有人拿不存在的東西去對拍。
3. 這一列已知有哪些落差（S-xx／D-xx／T-xx，見第 8 節的編號對照）。
4. 這一列有沒有現成的對拍工具或測試，以及它到底是 C↔Py、C 對 C 還是 Python 對 Python。

範圍內：`pipelines/`（common、mono、DFN2、4ch、ULCNet wrapper）及其 Python 參考，`pipelines/third_party/gain/` 與 audiolib 的 doa/bf/VAD/common/utility 對外函式，audiolib（hpf、audio_resampler、audio_stft／window、fft_wrapper、audio_pre_gain、wav_io，以及 fast_math、mem_align、simd_kernels、simd_kernel_nn 四個 header-only 檔），AINR（DeepFilterNet2、GTCRN、RNNoise-ERB）與 AIAEC（Align-ULCNet、DeepVQE-S、共用 mask helper）的 C 前後處理。

範圍外（不是漏列）：AEC 與 NR 函式庫本體（本表只列 pipelines 呼叫它們的接縫；它們的函式對照在各自 repo 的 `docs/c_python_function_map.md`，建立本表時 `lib/aec/docs/` 與 `lib/nr/docs/` 內已有）、`lib/aec`／`lib/nr` 的 Python、`pipelines/tools/` 與 `pipelines/experimental/`（本表建立時為未追蹤的實驗檔）、各處 `tests/` 與 audiolib 的 `test/`（測試程式）、`third_party` 內的 static 輔助函式、Align_CRUSE 與 CAGCRN（沒有 C pre/post 類別）。

## 2. 欄位語法與關係詞彙

每一列有 7 個欄位。**語法是嚴格的**，守衛測試用同一份語法解析，解析不了就失敗（避免表格式一改測試就變成空轉）。

| 欄位 | 語法 |
|---|---|
| ID | `區碼-子碼-序號`，例如 `PL-MONO-03`。對拍腳本引用這個編號；新增列用新編號，刪列不重用舊編號。 |
| C 檔:函式 | 一個或多個反引號區塊 `` `相對路徑.c: 函式1, 函式2` ``，不同檔案用 `<br>` 隔開；Python-only 的列填 `—`。同一個 (檔案, 函式) 在全表只能出現一次。 |
| Python 檔:符號 | 一個或多個 `` `相對路徑.py:符號` ``，用 `<br>` 隔開；符號是 `名稱` 或 `類別.成員`（方法、屬性、類別內賦值、dataclass 欄位）；名稱必須是該檔自己定義或賦值的（def、class、模組層賦值），單純匯入的名稱不算，讓列指向定義處。C-only 與未能判斷的列填 `—`。 |
| 關係 | 下列五個詞之一。 |
| 已知落差 | 落差編號，`；` 隔開，或 `—`。每個編號必須在第 8 節有定義。 |
| 對拍工具／測試 | repo 內的路徑（反引號）加括號標明性質：（C↔Py）真的把 C 與 Python 對拍；（C 對 C）、（NEON 對純量）、（Py 對 Py）、（Py 契約）、（文字比對）則不是。沒有就填 `—`。 |
| 備註 | 自由文字。以「（推論）」開頭代表配對是閱讀函式名稱與角色得到的，沒有 Python 檔或測試明確點名；統計會單獨計數。 |

關係詞彙：

| 關係 | 意思 |
|---|---|
| `mirror` | 同數學：運算與意圖相同。**不代表已有數值對拍**，數值證據看「已知落差」與「對拍工具」。 |
| `partial` | 部分對應：只涵蓋該 C 函式的一部分，或 Python 把同一件事拆在別處／內聯，或兩邊所有權、粒度不同。 |
| `C-only` | 沒有 Python 對應：記憶體池、生命週期、描述子、CLI、診斷、NEON 專用核心等。 |
| `Python-only` | 只在 Python、沒有 C 函式對應（C 函式欄填 `—`）。這些列只列 pipelines 的 Python 參考與 AINR 的一個明確缺口，不窮舉所有 Python 檔。 |
| `未能判斷` | 讀了還是無法決定對應或是否存在對應，不硬湊。 |

## 3. 如何維護

1. 改了 C 或 Python 之後，從 Audio_ALG 根目錄執行：

   ```
   PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
       pipelines/tests/test_c_python_map.py
   ```

2. 失敗訊息會指出該改哪裡：`defined but not in any row`（新增了 C 函式，補一列或併入既有的 C-only 列）、`no such function definition`（C 函式改名或刪除）、`symbol not found`（Python 符號改名或刪除）、`statistics block … is stale`（表改了，統計沒跟上）。
3. 統計區的數字要跟著改。重算指令（印出可直接貼進第 4 節的列）：`python3 pipelines/tests/test_c_python_map.py --stats`。
4. 新落差編號先加進第 8 節，再在列內引用。
5. 不要把行號寫進表。要追溯行號時用符號名搜尋。
6. 找不到把握的對應，就用 `未能判斷` 並在備註寫原因，不要用 `partial` 湊。
7. 沒有 audiolib 檢出時（單獨取出的 Audio_ALG），針對 `audiolib/common/…` 的 C 檢查會略過，其餘照常；可用環境變數 `C_PY_MAP_AC` 指定檢出位置。

**證明守衛會失敗**（改動只做在暫存副本，不碰 repo）：

- 改掉表內一個 Python 符號：把文件複製到暫存目錄、改壞一個 `path.py:Symbol`，再 `C_PY_MAP_MD=<副本> python3 -m pytest -q -p no:cacheprovider pipelines/tests/test_c_python_map.py`，測試應該紅並指出列編號。
- 改掉 Python 原始碼的一個符號：複製被引用的 Python 檔到暫存的同構目錄、改名一個 def，再用 `C_PY_MAP_ROOT=<暫存根目錄> … -k python_symbol` 執行。
- 新增一個 C 函式或改名：複製 C 檔做同樣的事，應該出現 `defined but not in any row` 或 `no such function definition`。

## 4. 涵蓋率統計

計數單位是 **C 函式**（同一檔內 `#if` 變體的同名定義算一個）。`Python-only` 欄計的是 Python 符號數。數字由表內各列重算，守衛測試會比對，手改會失敗。

<!-- map:stats:begin -->
| 範圍 | C 函式總數 | 已對應 | 其中 mirror | 其中 partial | C-only | 未能判斷 | Python-only 符號 | 已對應中的推論 |
|---|---|---|---|---|---|---|---|---|
| `PL` pipelines（common、mono、DFN、4ch、ULCNet wrapper）與其 Python 參考 | 428 | 95 | 50 | 45 | 333 | 0 | 31 | 35 |
| `TP` pipelines/third_party (gain/) + audiolib (doa/bf/VAD/common/utility) | 99 | 0 | 0 | 0 | 99 | 0 | 0 | 0 |
| `AC` audiolib | 302 | 41 | 19 | 22 | 261 | 0 | 0 | 13 |
| `AINR` AINR（DFN2、GTCRN、RNNoise-ERB） | 107 | 51 | 29 | 22 | 56 | 0 | 2 | 16 |
| `AIAEC` AIAEC（Align-ULCNet、DeepVQE-S、共用 helper） | 112 | 40 | 17 | 23 | 72 | 0 | 0 | 21 |
| `TOTAL` 合計 | 1048 | 227 | 115 | 112 | 821 | 0 | 33 | 85 |
<!-- map:stats:end -->

讀法：1048 個 C 函式中，227 個有 Python 對應（mirror 115、partial 112），其中 85 個的配對是推論；821 個是 C-only；未能判斷 0 個。另有 33 個 Python 符號只在 Python。

「已對應」是**索引**，不是「已驗證一致」。真正量過數值的只有「對拍工具／測試」欄標（C↔Py）的項目，以及備註內註明為一次性量測的數字；其餘靠「已知落差」欄的編號。

## 5. 證據來源與限制

每一列的配對來自下列來源（備註裡的數字是一次性量測，量測驅動不在本 repo 內；本表沒有執行任何 DSP 計算，也沒有重量那些數字）：

1. **Python 檔內的明文**：docstring 寫「Python twin of `sk_min_f32`」「`df_common_expand_mask`」「`dfn2_model_io_push_features`」等直接點名 C 函式的句子。
2. **C 檔內的註解**：pipelines 的 C 註解（不含 `tests/`、`tools/`）指向 Python 的很少（4ch 核心指向 `pipeline.py`、`audio_pipeline.c` 的 `_PSD_SCALE`、mono_aec_dfn_res 的 `main.c` 指向 `export_onnx.py`）；AINR／AIAEC 的標頭與 .c 則大量點名 `train.py`、`model.py`、`export_onnx.py` 的符號。
3. **現有測試**：`pipelines/tests`、`pipelines/4ch_aec_bf_nr_res/tests`、`AINR/tests`、`AIAEC/tests` 內真正驅動 C 並對拍 Python 的測試，用的 C 函式與 Python 可呼叫物就是配對依據。
4. **一次性量測**：audiolib、AINR／AIAEC、pipelines 各自把 C 函式與 Python 可呼叫物配對後量測；這些驅動不在 repo 內，備註內以「一次性量測」標示，**不是常設測試**。
5. **閱讀函式名稱與角色**：備註以「（推論）」開頭，統計單獨計數。

限制：

- Python 目標解析在 Audio_ALG 內：`lib/aec/python`、`lib/nr/python` 是子模組，版本可能落後 AEC／NR 主線（lib/aec 的 Python 可能落後 AEC 主線），本表指向子模組裡實際存在的符號，因為守衛測試也只看這一份。
- 結構（struct、enum、巨集常數）不是函式，不在 C 函式表內；少數有 Python 對應的常數（如 `SE_CNG_NR_GAIN_FLOOR`）寫在備註或 Python-only 列。
- 本表不判斷哪邊該改。

### 與 pyc_parity 對拍腳本的關係

對拍腳本量的是數值（逐情境的 max abs err、p99.9、是否逐位元相同），本表回答「拿 C 的哪個函式去對 Python 的哪個符號」。兩者分工如下：

- 本表是**索引**。每列有穩定編號；對拍腳本可以在情境名稱或輸出 JSON 內引用列編號，讓一個超出容差（或標 `EXPECTED_RED(S-xx)`）的情境能回頭查到 C 函式、Python 符號與已知落差。落差編號與本表「已知落差」欄是同一套。
- 本表不含容差、不含數值判定，也不要求任何對拍腳本存在；對拍腳本也不依賴本表才能執行。
- pipelines 的對拍腳本 `pipelines/tests/pyc_parity_pipelines.sh`（資料夾 `pipelines/tests/pyc_parity/`）有四組情境，對應本表的列：4ch NR 增益對應 PL-STAGE-02、4ch 後波束 RES 對應 PL-STAGE-05、4ch 共用 delay 的混合 realign 對應 PL-4CH-07、mono 膠水對應 PL-MONO-02 與 PL-MONO-03；這些列的「對拍工具」欄引用該腳本。
- 守衛測試只保證「表指的東西還在」：Python 符號存在、C 函式存在、涵蓋完整、統計與表一致、引用的落差編號與工具路徑有效。它不檢查兩邊數值是否一致。

## 6. 涵蓋檔案

守衛測試逐檔重新列舉函式，與下表及第 7 節的列逐一比對。範圍欄：`all` 為檔內所有函式（含 static）；`extern` 只計非 static 函式（用於 third_party，其 static 輔助函式不逐一列舉）。

<!-- map:coverage:begin -->
| 檔案 | 範圍 | 函式數 |
|---|---|---|
| `pipelines/common/se_anr_stage.c` | all | 8 |
| `pipelines/common/se_config.c` | all | 5 |
| `pipelines/common/se_config.h` | all | 1 |
| `pipelines/common/se_stage.h` | all | 1 |
| `pipelines/mono_aec_nr_res/audio_pipeline.c` | all | 31 |
| `pipelines/mono_aec_nr_res/example_board_adapter.c` | all | 7 |
| `pipelines/mono_aec_nr_res/main.c` | all | 10 |
| `pipelines/mono_aec_nr_res/static_main.c` | all | 12 |
| `pipelines/dfn_example_args.h` | all | 10 |
| `pipelines/dfn_rate_bridge.c` | all | 28 |
| `pipelines/dfn_res_stage.c` | all | 24 |
| `pipelines/dfn_unit_partition.h` | all | 1 |
| `pipelines/mono_aec_dfn_res/audio_pipeline_dfn.c` | all | 26 |
| `pipelines/mono_aec_dfn_res/main.c` | all | 2 |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c` | all | 52 |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_static.c` | all | 9 |
| `pipelines/4ch_aec_bf_nr_res/4aec_projection_kernels.h` | all | 11 |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c` | all | 48 |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c` | all | 10 |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_static.c` | all | 10 |
| `pipelines/4ch_aec_bf_nr_res/mcab_sample_driver.c` | all | 12 |
| `pipelines/4ch_aec_bf_dfn_res/4aec_dfn_res.c` | all | 28 |
| `pipelines/4ch_aec_bf_dfn_res/main.c` | all | 10 |
| `pipelines/mono_alignulcnet/audio_pipeline_ulcnet.c` | all | 21 |
| `pipelines/mono_alignulcnet/main.c` | all | 6 |
| `pipelines/4ch_alignulcnet/audio_pipeline_4ch_ulcnet.c` | all | 34 |
| `pipelines/4ch_alignulcnet/main.c` | all | 11 |
| `audiolib/bf/bf.c` | extern | 15 |
| `audiolib/VAD/masker.c` | extern | 6 |
| `audiolib/VAD/vad.c` | extern | 8 |
| `audiolib/VAD/vad_api.c` | extern | 10 |
| `audiolib/doa/doa_smoother.c` | extern | 1 |
| `audiolib/doa/srp.c` | extern | 12 |
| `audiolib/doa/steering.c` | extern | 8 |
| `pipelines/third_party/gain/fix_gain.c` | extern | 7 |
| `pipelines/third_party/gain/nr_gain.c` | extern | 6 |
| `pipelines/third_party/gain/post_gain.c` | extern | 9 |
| `audiolib/common/utility/spatial_complex.c` | extern | 4 |
| `audiolib/common/utility/spatial_simd.c` | extern | 13 |
| `audiolib/common/utility/fast_math.h` | all | 8 |
| `audiolib/common/utility/mem_align.h` | all | 6 |
| `audiolib/common/utility/simd_kernel_nn.h` | all | 36 |
| `audiolib/common/utility/simd_kernels.h` | all | 59 |
| `audiolib/common/utility/simd_aec.h` | all | 79 |
| `audiolib/common/utility/simd_nr.h` | all | 19 |
| `audiolib/common/include/wav_io.h` | all | 9 |
| `audiolib/common/src/audio_pre_gain.c` | all | 9 |
| `audiolib/common/src/audio_resampler.c` | all | 21 |
| `audiolib/common/src/audio_stft.c` | all | 33 |
| `audiolib/common/src/audio_window.c` | all | 1 |
| `audiolib/common/src/fft_wrapper_ne10.c` | all | 15 |
| `audiolib/common/src/hpf.c` | all | 8 |
| `AINR/DeepFilterNet2/dfn2_model_io.c` | all | 12 |
| `AINR/DeepFilterNet2/dfn2_prepost.c` | all | 39 |
| `AINR/DeepFilterNet2/dfn2_process.c` | all | 17 |
| `AINR/GTCRN/gtcrn_process.c` | all | 19 |
| `AINR/RNNoise-ERB/gen_rnnoise_tables.c` | all | 7 |
| `AINR/RNNoise-ERB/process.c` | all | 12 |
| `AIAEC/Align_ULCNet/ulcnet_accelerator_adapter.c` | all | 9 |
| `AIAEC/Align_ULCNet/ulcnet_model_io.c` | all | 27 |
| `AIAEC/Align_ULCNet/ulcnet_model_io.h` | all | 1 |
| `AIAEC/Align_ULCNet/ulcnet_prepost.c` | all | 27 |
| `AIAEC/Align_ULCNet/ulcnet_process.c` | all | 2 |
| `AIAEC/DeepVQE_S/deepvqe_prepost.c` | all | 39 |
| `AIAEC/DeepVQE_S/deepvqe_process.c` | all | 3 |
| `AIAEC/aiaec_process.c` | all | 4 |
<!-- map:coverage:end -->

## 7. 對照表

<!-- map:rows:begin -->

### 7.1 pipelines

| ID | C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 | 備註 |
|---|---|---|---|---|---|---|
| PL-CFG-01 | `pipelines/common/se_config.c: se_grid_resolve` | `pipelines/aec_nr_pipeline.py:_project_grid` | partial | S-30 | — | C 另接受 8 kHz/256（allow_8k）；Python 只接受 16 kHz 與 48 kHz。16 kHz 預設 256、48 kHz 預設 1024 兩邊相同。 |
| PL-CFG-02 | `pipelines/common/se_config.c: se_derive` | `lib/aec/python/modules/config.py:AecConfig.from_preset`<br>`pipelines/aec_nr_pipeline.py:_project_grid` | partial | D-38 | — | C 把 grid、AecConfig（enable_res=0、return_res_context=1、delay 模式）一次導出；Python 由 `_project_grid` 與 `AecConfig.from_preset` 分兩步取得。struct 值有 10 個欄位與 Python 有效值不同（C 以 10 ms 基準寫入、carve 時 retime），carve 後有效值屬 AEC 函式庫範圍。 |
| PL-CFG-03 | `pipelines/common/se_config.c: se_grid_fixed, preset_valid, delay_request_valid`<br>`pipelines/common/se_config.h: se_flag_valid` | — | C-only | S-34；D-39 | — | 固定速率應用（DFN／ULCNet wrapper）的 grid 規則與 fail-closed 請求驗證；Python 參考不驗證範圍，也沒有 FIXED／EXTERNAL_ALIGNED delay 模式。 |
| PL-STAGE-01 | `pipelines/common/se_anr_stage.c: se_anr_compose_config` | `pipelines/aec_nr_pipeline.py:_build_denoiser` | mirror | D-23 | — | 19~21 個 NR 欄位，3 grid x 4 強度，最大相對差 7.7e-8（一次性量測，非常設測試）。L=150ms 與 alpha_decay=alpha_g 兩邊相同。 |
| PL-STAGE-02 | `pipelines/common/se_anr_stage.c: se_anr_stage_gain` | `pipelines/aec_nr_pipeline.py:run_nr_spectrum` | mirror | S-10；S-11；T-15 | `pipelines/tests/pyc_parity_pipelines.sh`（C↔Py） | C 逐 hop，Python 整段批次。mono 路徑把 C seam 餵給 Python 的 G_nr 差 8.3e-6（16k/256，標準數學建置）。 |
| PL-STAGE-03 | `pipelines/common/se_anr_stage.c: se_psd_scale` | `pipelines/aec_nr_pipeline.py:run_nr_spectrum` | mirror | — | — | （推論）r2 / 2^30 的倒數乘法；Python 在 `run_nr_spectrum` 內聯 `psd_scale = 32768.0 ** 2`。除數是 2 的冪，兩邊結果相同。 |
| PL-STAGE-04 | `pipelines/common/se_anr_stage.c: se_anr_stage_mem_size, se_anr_stage_init, se_anr_stage_set_mode, se_anr_stage_reset, se_anr_stage_destroy` | — | C-only | D-23 | — | 靜態記憶體生命週期與執行期強度切換；Python 每次呼叫 `_build_denoiser` 重建，沒有執行期 setter。 |
| PL-STAGE-05 | `pipelines/common/se_stage.h: se_res_stage_gain` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:PostBeamResidualSuppressor.process` | partial | T-16 | `pipelines/tests/pyc_parity_pipelines.sh`（C↔Py） | 後波束 RES 增益。clock_drift 在 Python 寫死 False。 |
| PL-MONO-01 | `pipelines/mono_aec_nr_res/audio_pipeline.c: analyse_hop` | `pipelines/se_stage.py:frame_from_context`<br>`pipelines/aec_nr_pipeline.py:run_aec_linear` | partial | S-05；T-17 | — | AEC 線性模式（enable_res=0、return_res_context=1）加上 SeSpecFrame 欄位填充。Python 的 `SeSpecFrame` 沒有 usable_linear 欄位（`run_nr_spectrum` 直接讀 AecResContext.usable_linear），`frame_from_context` 也不填 far_hop、delay_changed、delay_locked、clock_drift（維持預設值；4ch 的 `FourChannelFrame.spec_frame` 只補 delay_changed 與 delay_locked），所以只算部分對應。一次性量測的 10 s 錄音上 usable_linear 與 converged 旗標逐 hop 一致；S-05 另記 doubletalk 錄音上 converged 上升緣 Python 晚 1 hop（5233 hop 中 2 hop）。E 在分岔前差約 1e-4 相對，之後來自 lib/aec 接縫（不是膠水）。 |
| PL-MONO-02 | `pipelines/mono_aec_nr_res/audio_pipeline.c: synthesize_hop, audio_pipeline_synthesize` | `pipelines/aec_nr_pipeline.py:synthesize_hop` | mirror | — | `pipelines/tests/pyc_parity_pipelines.sh`（C↔Py） | 一次 IFFT + sqrt-Hann + OLA 的 hop 輸出。把 C 的 E、G_nr、G_res 餵給 Python 的 min／apply／OLA，與 C 輸出最大差 5.96e-8（16k）、7.45e-8（48k）。 |
| PL-MONO-03 | `pipelines/mono_aec_nr_res/audio_pipeline.c: audio_pipeline_process` | `pipelines/aec_nr_pipeline.py:run_res`<br>`pipelines/aec_nr_pipeline.py:total_gain`<br>`pipelines/aec_nr_pipeline.py:apply_gain`<br>`pipelines/aec_nr_pipeline.py:comfort_noise_fill` | mirror | D-37；T-17；S-10；S-11 | `pipelines/tests/pyc_parity_pipelines.sh`（C↔Py） | 逐 hop：NR 增益、min、apply、comfort noise、synthesis（aec_only 分支對應 `run_aec_linear(standalone_time_output=True)`）。comfort noise 是 Python 的高斯近似，不逐位元；整條端到端在 16k/256 於 hop 446 分岔（lib/aec 接縫）。其中 min 與 apply 的核心見 AC-SIMD-01。 |
| PL-MONO-04 | `pipelines/mono_aec_nr_res/audio_pipeline.c: audio_pipeline_process_spec` | `pipelines/aec_nr_pipeline.py:run_aec_linear` | partial | — | — | （推論）只做分析半邊（回傳 SeSpecFrame）；Python 的 `run_aec_linear` 回傳整段 context。 |
| PL-MONO-05 | `pipelines/mono_aec_nr_res/audio_pipeline.c: derive_dims_and_configs` | `pipelines/aec_nr_pipeline.py:_project_grid` | partial | S-30；D-38 | — | （推論）grid 與 AecConfig／AnrConfig 導出（透過 se_derive）；Python 無 8 kHz。 |
| PL-MONO-06 | `pipelines/mono_aec_nr_res/audio_pipeline.c: audio_pipeline_set_aec_preset, audio_pipeline_set_nr_mode, audio_pipeline_reset` | — | C-only | D-23 | — | 執行期 setter 與重置；mono Python 參考每次執行重建物件（4ch 的 Python 有 `set_aec_preset`，見 PL-4APP-03）。 |
| PL-MONO-07 | `pipelines/mono_aec_nr_res/audio_pipeline.c: audio_pipeline_now_us, audio_pipeline_backend_id, unity_gain_needed, synthesis_mem_size, pipeline_pool_size, pipeline_build, fnv1a_str, audio_pipeline_build_flags_hash, audio_pipeline_default_config, audio_pipeline_get_mem_requirements, audio_pipeline_init_ex, audio_pipeline_init, audio_pipeline_create, audio_pipeline_copy_aec_timing, audio_pipeline_destroy, audio_pipeline_hop_size, audio_pipeline_n_freqs, audio_pipeline_sample_rate, audio_pipeline_get_aec, audio_pipeline_get_nr, audio_pipeline_get_mem_breakdown, audio_pipeline_get_last_timing` | — | C-only | — | — | 靜態記憶體池尺寸／carve、build-flags hash、描述子、計時與 introspection；沒有演算法內容。 |
| PL-MONO-08 | `pipelines/mono_aec_nr_res/main.c: main`<br>`pipelines/mono_aec_nr_res/static_main.c: main` | `pipelines/aec_nr_pipeline.py:main` | partial | S-31；D-40 | — | 旗標集合不對稱：Python 沒有 moderate、--no-nr／--no-res／--no-cng；C 以位置參數給 aec-preset。static_main.c 與 main.c 同流程，差別是使用呼叫端記憶體池。 |
| PL-MONO-09 | `pipelines/mono_aec_nr_res/main.c: parse_preset, preset_name, parse_nr_mode, nr_mode_name, print_debug_status, stage_accum_add, stage_accum_row, print_stage_timing, timing_now_us`<br>`pipelines/mono_aec_nr_res/static_main.c: parse_preset, preset_name, parse_nr_mode, nr_mode_name, print_debug_status, backend_id_name, print_mem_budget, stage_accum_add, stage_accum_row, print_stage_timing, timing_now_us`<br>`pipelines/mono_aec_nr_res/example_board_adapter.c: board_mem_alloc, board_mem_free, lcg_sample, run_hops, run_one_cycle, run_negative_demonstrations, main` | — | C-only | — | — | CLI 解析、診斷列印、計時、板端轉接範例。 |
| PL-DFN-01 | `pipelines/dfn_res_stage.c: dfn_res_stage_process, dfn_res_stage_begin, dfn_res_stage_infer, dfn_res_stage_end` | `pipelines/dfn2_stage.py:Dfn2Stage.push` | mirror | D-42；S-17 | `pipelines/tests/test_dfn2_stage_c_parity.py`（C↔Py，驅動 dfn2_process.c）；`pipelines/tests/test_dfn2_stage.py`（Py 契約） | C 把一個 hop 拆成 begin／callback／end 三段；Python 的 `push` 在內部呼叫 heads。compose 路徑逐位元相同、特徵在 AINR 容差內（測試驅動的是 dfn2_process.c，不是 dfn_res_stage.c 本體）。 |
| PL-DFN-02 | `pipelines/dfn_res_stage.c: dfn_res_stage_reset` | `pipelines/dfn2_stage.py:Dfn2Stage.reset` | mirror | D-42 | — | fail-open 後的狀態語意不同：C 在 commit 被拒時就地歸零，Python 不步進遞迴狀態。 |
| PL-DFN-03 | `pipelines/dfn_res_stage.c: dfn_res_stage_set_atten_lim` | `pipelines/dfn2_stage.py:Dfn2Stage.set_atten_lim` | mirror | — | — | — |
| PL-DFN-04 | `pipelines/dfn_res_stage.c: erb_inv_is_partition_of_unity` | `pipelines/dfn2_stage.py:erb_inv_is_partition_of_unity`<br>`AINR/DeepFilterNet2/export_erb_matrix.py:require_partition_of_unity` | mirror | S-21 | `AINR/DeepFilterNet2/tests/test_dfn2_contract.py`（Py 契約） | identity（fail-open）路徑的前提：單位 band mask 展開後每個 bin 恰為 1.0。 |
| PL-DFN-05 | `pipelines/dfn_res_stage.c: dfn_res_stage_output_frame_index` | `pipelines/dfn2_stage.py:StageOutput.frame_index` | partial | — | — | （推論）C 是存取函式，Python 把來源 frame 序號放在 `StageOutput` 欄位。 |
| PL-DFN-06 | `pipelines/dfn_res_stage.c: align_up, carve, fnv1a_u32, backend_id, model_valid, resolve, fill_descriptor, dfn_res_stage_default_config, dfn_res_stage_get_mem_requirements, descriptor_matches, dfn_res_stage_init_ex, dfn_res_stage_init, dfn_res_stage_create, dfn_res_stage_destroy, dfn_res_stage_get_counters, inputs_valid` | — | C-only | — | — | 描述子、記憶體池、counters；`inputs_valid` 在 begin 前掃兩個輸入頻譜是否有限，Python 的 `Dfn2Stage.push` 只檢查 heads 輸出端（fail-open 的觸發條件），沒有輸入端掃描。 |
| PL-DFN-07 | `pipelines/dfn_rate_bridge.c: dfn_rate_bridge_process, dfn_rate_bridge_begin, dfn_rate_bridge_nn_begin, dfn_rate_bridge_nn_end, dfn_rate_bridge_finish, push_up, finish_frame` | `pipelines/dfn2_rate_adapter.py:Dfn2RateBridge.push` | mirror | D-24 | `pipelines/tests/test_dfn2_rate_bridge_c_parity.py`（C↔Py） | 兩邊用同一個 C resampler 核心，只剩 FFT 後端差；容差 float32 round-off，prefill／added delay 為相同整數。 |
| PL-DFN-08 | `pipelines/dfn_rate_bridge.c: calibrate_prefill` | `pipelines/dfn2_rate_adapter.py:Dfn2RateBridge._calibrate_prefill` | mirror | — | `pipelines/tests/test_dfn2_rate_bridge_c_parity.py`（C↔Py） | — |
| PL-DFN-09 | `pipelines/dfn_rate_bridge.c: dfn_rate_bridge_reset, clear_streams` | `pipelines/dfn2_rate_adapter.py:Dfn2RateBridge.reset` | mirror | — | — | — |
| PL-DFN-10 | `pipelines/dfn_rate_bridge.c: grid_dims, resolve, dfn_rate_bridge_added_delay_samples` | `pipelines/dfn2_rate_adapter.py:Dfn2RateBridge.__init__` | partial | — | `pipelines/tests/test_dfn2_rate_bridge_c_parity.py`（C↔Py） | （推論）grid 能力閘與 added delay（16k/128: 672、16k/256: 629、8k/128: 330）；Python 在建構時計算並對拍。 |
| PL-DFN-11 | `pipelines/dfn_rate_bridge.c: align_up, fnv_u32, borrowed_bytes, fill_descriptor, dfn_rate_bridge_default_config, dfn_rate_bridge_get_mem_requirements, descriptor_matches, dfn_rate_bridge_init_ex, dfn_rate_bridge_init, dfn_rate_bridge_create, dfn_rate_bridge_destroy, dfn_rate_bridge_hop_size, dfn_rate_bridge_n_freqs, dfn_rate_bridge_stage, dfn_rate_bridge_get_counters` | — | C-only | — | — | 描述子、記憶體池、counters、存取函式。 |
| PL-DFN-12 | `pipelines/mono_aec_dfn_res/audio_pipeline_dfn.c: mono_aec_dfn_res_process, mono_aec_dfn_res_process_begin, mono_aec_dfn_res_nn_begin, mono_aec_dfn_res_nn_end, mono_aec_dfn_res_process_end` | `pipelines/aec_dfn2_res_pipeline.py:run_dfn2_res` | mirror | D-43；D-37；T-15 | `pipelines/tests/test_dfn2_pipeline_contract.py`（Py 對 Py，identity 模型） | Python 是這條 C 管線的孿生（檔頭明載）。identity 模型下 Python 吃 C seam 對 C 輸出最大差 1.2e-7（A6，一次性量測）。Python 另有 C 沒有的實驗開關（--dfn-estimate-source、--dfn-compose）。 |
| PL-DFN-13 | `pipelines/mono_aec_dfn_res/audio_pipeline_dfn.c: mono_aec_dfn_res_set_atten_lim` | `pipelines/dfn2_stage.py:Dfn2Stage.set_atten_lim` | partial | — | — | （推論）透傳到 stage。 |
| PL-DFN-14 | `pipelines/mono_aec_dfn_res/audio_pipeline_dfn.c: mono_aec_dfn_res_lookahead_samples` | `pipelines/dfn2_stage.py:Dfn2Stage.output_delay_frames` | partial | — | — | （推論）C 回傳 host hop 加兩個 DFN2 lookahead hop（48 kHz）或 bridge 的 added delay（原生網格）；Python 只在 stage 層給 frame 數。 |
| PL-DFN-15 | `pipelines/mono_aec_dfn_res/audio_pipeline_dfn.c: align_up, fnv_u32, resolve, fill_descriptor, mono_aec_dfn_res_default_config, mono_aec_dfn_res_get_mem_requirements, descriptor_matches, mono_aec_dfn_res_init_ex, mono_aec_dfn_res_init, mono_aec_dfn_res_create, mono_aec_dfn_res_reset, mono_aec_dfn_res_destroy, fail, mono_aec_dfn_res_set_aec_preset, mono_aec_dfn_res_hop_size, mono_aec_dfn_res_get_host, mono_aec_dfn_res_get_aec, mono_aec_dfn_res_get_stage, mono_aec_dfn_res_get_bridge` | — | C-only | — | — | 描述子、記憶體池、存取函式；`set_aec_preset` 在 Python 是建構參數。 |
| PL-DFN-16 | `pipelines/mono_aec_dfn_res/main.c: run_accelerator, main`<br>`pipelines/dfn_unit_partition.h: dfn_unit_partition`<br>`pipelines/dfn_example_args.h: dfn_example_rate_known, dfn_example_rates_text, dfn_example_usage, dfn_example_parse_delay_mode, dfn_example_delay_mode_name, dfn_example_parse_int, dfn_example_parse_float, dfn_example_parse_args, dfn_example_read_matrix, dfn_example_load_matrices` | — | C-only | — | — | 板端範例的 CLI、參數解析與單位 ERB 矩陣；`run_accelerator` 是替換成板端 runtime 的樁。 |
| PL-4CH-01 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: update_shared_delay` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:SharedMatchedDelayEstimator.accumulate` | mirror | D-39 | `pipelines/4ch_aec_bf_nr_res/tests/test_delay_parity.py`（C↔Py）；`pipelines/4ch_aec_bf_nr_res/tests/dump_delay_parity.c`（C 側驅動） | delay_samples／solid／changed 逐 hop 相同（16k/256 1250/1250、16k/512 625/625、48k 937/937）；matched-filter 的 duty-cycle 排程、FIXED 與 EXTERNAL_ALIGNED 模式是 C-only（對拍時 pin 全速）。 |
| PL-4CH-02 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_nr_res_admission_age, four_aec_nr_res_admission_offer, four_aec_nr_res_pending_delay_candidate` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:SharedMatchedDelayEstimator.accumulate` | partial | D-39 | `pipelines/4ch_aec_bf_nr_res/tests/test_delay_parity.py`（C↔Py）；`pipelines/4ch_aec_bf_nr_res/tests/dump_delay_parity.c`（C 側驅動） | Path-B 三條件（變動 >32 樣本、第二次估計在 16 內、候選壽命 3 hop）；C 拆成可由測試直接呼叫的函式，Python 內聯在 `accumulate`。兩邊常數相同。 |
| PL-4CH-03 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: align_render` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:_SharedReferenceDelayLine.process` | mirror | — | `pipelines/4ch_aec_bf_nr_res/tests/test_delay_parity.py`（C↔Py）；`pipelines/4ch_aec_bf_nr_res/tests/dump_delay_parity.c`（C 側驅動） | 整個 hop 以同一個謂詞決定走對齊環或原始 render；aligned far 逐位元相同（0/160000 樣本差）。 |
| PL-4CH-04 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_front_process` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.front_process` | mirror | D-39；T-15 | — | 共用 delay、對齊 render、四條線性 AEC lane。 |
| PL-4CH-05 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: fuse_contexts` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts` | mirror | D-41；T-16 | — | 相干投影 E／Y／回音，R² 與 comfort noise 加權。R² 相量的相位下限只在 C（0<\|echo\|²≤1e-12 時釘在實軸）。 |
| PL-4CH-06 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_fuse` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.fuse`<br>`pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelFrame.spec_frame` | partial | — | — | C 的 fuse 只做投影，後波束 RES 由應用層呼叫 se_res_stage_gain；Python 的 `fuse` 把後波束 RES 也含進去。 |
| PL-4CH-07 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_nr_res_far_spec_agrees, complex_close` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts` | partial | — | `pipelines/tests/pyc_parity_pipelines.sh`（C↔Py） | 「四 lane 必須共用同一個 far 頻譜」的前提檢查。C 在混合 warm／soft realign 的 hop 以 `far_spec_mixed_hop` 旗標略過這項檢查；Python 的 `_fuse_contexts` 不做這項檢查，兩邊的融合 X 都取 lane 0 的 far_spec。 |
| PL-4CH-08 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: validate_weights`<br>`pipelines/4ch_aec_bf_nr_res/4aec_projection_kernels.h: four_aec_weights_valid, four_aec_weights_valid_scalar` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts` | partial | — | — | （推論）波束權重有限性與非零檢查；Python 在 `_fuse_contexts` 內以 np.isfinite 與 weight_sum 判斷。 |
| PL-4CH-09 | `pipelines/4ch_aec_bf_nr_res/4aec_projection_kernels.h: four_aec_projection_cmac, four_aec_projection_cmac_scalar` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts` | mirror | — | — | （推論）sum(weights * values) 的複數乘加核心。 |
| PL-4CH-10 | `pipelines/4ch_aec_bf_nr_res/4aec_projection_kernels.h: four_aec_residual_sample, four_aec_residual_vector, four_aec_residual_vector_scalar, four_aec_complex_mag2, four_aec_complex_mag2_scalar` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts` | partial | D-41 | — | \|sum(w * sqrt(r2) * exp(j*phase))\|² 的 R² 相量融合。零向量的相位下限 C 與 Python 不同（C 釘 +實軸，Python 用 np.angle）。 |
| PL-4CH-11 | `pipelines/4ch_aec_bf_nr_res/4aec_projection_kernels.h: four_aec_comfort_accumulate, four_aec_comfort_accumulate_scalar` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:_fuse_contexts` | mirror | — | — | （推論）sum(\|w\|² * comfort_noise) 的 comfort noise 融合。 |
| PL-4CH-12 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_nr_res_reset` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.reset`<br>`pipelines/4ch_aec_bf_nr_res/pipeline.py:SharedMatchedDelayEstimator.reset` | mirror | — | — | 所有 lane 與共用 delay 估計器共用同一個重置時期。 |
| PL-4CH-13 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: derive_dims_and_configs, four_aec_nr_res_default_config` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecConfig.resolved_grid` | partial | S-32；D-39 | — | （推論）grid 白名單（16k/256、16k/512、48k/1024）與預設值；Python 的設定面比 C 少（無 delay_mode、nr_mode、enable_*、legacy_amin）。 |
| PL-4CH-14 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_nr_res_stage_configs` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:PostBeamResidualSuppressor._build_gain`<br>`pipelines/aec_nr_pipeline.py:_build_denoiser` | partial | — | — | 回傳後波束 RES 與 NR 的 AecConfig／AnrConfig。 |
| PL-4CH-15 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: bind_lane_view` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:_snapshot_context` | partial | — | — | （推論）所有權語意不同：C 直接把指標指向各 lane 的 buffer（零拷貝），Python 的 `_snapshot_context` 複製一份。 |
| PL-4CH-16 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_nr_res_matched_filter_count` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.matched_filter_instance_count`<br>`pipelines/4ch_aec_bf_nr_res/pipeline.py:SharedMatchedDelayEstimator.instance_count` | mirror | — | — | （推論）結構稽核掛鉤：整條管線只有一個 matched filter。 |
| PL-4CH-17 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_nr_res_linear_aec_count` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.linear_filter_instance_count` | mirror | — | — | （推論）結構稽核掛鉤：四條線性 AEC lane。 |
| PL-4CH-18 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_nr_res_realign_warm_lane_count` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.realign_warm_lane_count` | mirror | — | — | — |
| PL-4CH-19 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_nr_res_realign_soft_lane_count` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.realign_soft_lane_count` | mirror | — | — | — |
| PL-4CH-20 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_nr_res_hop_size, four_aec_nr_res_fft_size, four_aec_nr_res_n_freqs, four_aec_nr_res_sample_rate` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.__init__` | partial | — | — | （推論）Python 在建構時把 grid 存成屬性，沒有獨立的存取方法。 |
| PL-4CH-21 | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c: four_aec_nr_res_backend_id, four_aec_nr_res_now_us, duty_leak_db_for_grid, pool_carve, carve_working_buffers, pipeline_buffer_size, pipeline_pool_size, pipeline_build, fnv1a_str, four_aec_nr_res_build_flags_hash, four_aec_nr_res_get_mem_requirements, four_aec_nr_res_init_ex, four_aec_nr_res_init, four_aec_nr_res_create, inputs_finite, max_lane_erle_windowed, complex_vector_finite, four_aec_nr_res_destroy, four_aec_nr_res_far_fft_real_compute_count, four_aec_nr_res_duty_hops_total, four_aec_nr_res_duty_hops_run, four_aec_nr_res_far_spec_provenance, four_aec_nr_res_pin_duty_full_rate, four_aec_nr_res_duty_leak_db_for_grid, four_aec_nr_res_duty_erle_leak_db, four_aec_nr_res_duty_erle_peak, four_aec_nr_res_get_mem_breakdown, four_aec_nr_res_get_last_timing`<br>`pipelines/4ch_aec_bf_nr_res/4aec_nr_res_static.c: parse_preset, preset_name, parse_nr_mode, nr_mode_name, backend_id_name, plan_stages, print_mem_budget, fill_hop, main` | — | C-only | D-39 | — | 靜態記憶體池、carve、build-flags hash、duty-cycle 排程（C-only）、far-spec provenance 與 duty 診斷、計時；static 應用的 CLI 與合成輸入。 |
| PL-4APP-01 | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c: run_post_stages` | `pipelines/4ch_aec_bf_dfn_res/pipeline.py:post_conventional` | mirror | D-37 | — | 融合 frame 之後的頻域鏈：RES 增益、NR 增益、min、apply、comfort noise、synthesis。Python 靠 `run_nr_spectrum`＋`run_res` 組成。 |
| PL-4APP-02 | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c: audio_pipeline_4ch_process, audio_pipeline_4ch_process_with_activity` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.process`<br>`pipelines/4ch_aec_bf_dfn_res/pipeline.py:post_conventional` | partial | D-43；S-32 | — | （推論）C 含 SRP-PHAT 與波束器（bf）（沒有 Python 版）；Python 的波束器是由外部供應的接縫（EqualWeightBeamformer 只供測試）。 |
| PL-4APP-03 | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c: audio_pipeline_4ch_set_aec_preset` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.set_aec_preset` | partial | — | — | 兩邊都同時重設 far-active 地板、DT 地板與 split-floor gate。 |
| PL-4APP-04 | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c: audio_pipeline_4ch_post_split_floor` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:PostBeamResidualSuppressor.set_preset` | partial | — | — | （推論）設定 split floor（far-active）；Python 的 `set_preset` 另外重設 DT 地板與 split-floor gate。 |
| PL-4APP-05 | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c: audio_pipeline_4ch_reset` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.reset`<br>`pipelines/4ch_aec_bf_nr_res/pipeline.py:PostBeamResidualSuppressor.reset` | partial | — | — | （推論）C 還重置 SRP／bf／後波束階段。 |
| PL-4APP-06 | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c: audio_pipeline_4ch_matched_filter_count` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.matched_filter_instance_count` | mirror | — | — | （推論）結構稽核計數（整條管線一個 matched filter）。 |
| PL-4APP-07 | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c: audio_pipeline_4ch_linear_aec_count` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.linear_filter_instance_count` | mirror | — | — | （推論）結構稽核計數（四條線性 AEC）。 |
| PL-4APP-08 | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c: audio_pipeline_4ch_post_res_count` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.residual_suppressor_instance_count` | mirror | — | — | （推論）結構稽核計數（一個後波束 RES）。 |
| PL-4APP-09 | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c: is_bool, validate_config, audio_pipeline_4ch_default_config, pool_carve, fnv1a_str, audio_pipeline_4ch_build_flags_hash, fill_uca_geometry, fill_ula_geometry, fill_custom_geometry, fill_stack_geometry, derive_spatial_configs, derive_post_stages, derive_pipeline_layout, audio_pipeline_4ch_get_mem_requirements, audio_pipeline_4ch_init_ex, audio_pipeline_4ch_init, audio_pipeline_4ch_create, fill_frame_info, floats_finite, auto_vad, audio_pipeline_4ch_set_nr_mode, audio_pipeline_4ch_destroy, audio_pipeline_4ch_hop_size, audio_pipeline_4ch_frame_size, audio_pipeline_4ch_fft_size, audio_pipeline_4ch_n_freqs, audio_pipeline_4ch_sample_rate, audio_pipeline_4ch_doa_sample_rate, audio_pipeline_4ch_doa_frame_size, audio_pipeline_4ch_doa_hop_size, audio_pipeline_4ch_doa_fft_size, audio_pipeline_4ch_bf_sample_rate, audio_pipeline_4ch_bf_frame_size, audio_pipeline_4ch_bf_hop_size, audio_pipeline_4ch_bf_fft_size, audio_pipeline_4ch_bf_effective_adapt_interval, audio_pipeline_4ch_bf_lambda, audio_pipeline_4ch_nr_count, audio_pipeline_4ch_spatial_backend`<br>`pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c: usage, json_float, raw_parse, raw_setup, raw_read, raw_process, raw_consume, raw_finish, raw_cleanup, main`<br>`pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_static.c: fill_hop, static_parse, static_setup, static_read, static_process, static_nonfinite, static_consume, static_finish, static_cleanup, main` | — | C-only | D-43 | — | 設定驗證、SRP／bf 陣列幾何與設定導出、記憶體池、VAD 自動判斷、DOA／bf 存取函式、CLI；沒有 Python 版。`backend_id_name`／`print_mem_budget` 的共用實作是 `mcab_sample_driver.{c,h}` 的 `mcab_backend_id_name`／`mcab_print_mem_budget`（見 PL-4APP-10）。 |
| PL-4APP-10 | `pipelines/4ch_aec_bf_nr_res/mcab_sample_driver.c: mcab_backend_id_name, mcab_print_mem_budget, mcab_scan_cli, mcab_raw_int, mcab_raw_float, mcab_spatial_set, mcab_spatial_unknown, mcab_parse_spatial_cli, mcab_delay_mode_name, mcab_set_delay_option, mcab_validate_delay_options, mcab_sample_main` | — | C-only | — | — | 四個 4ch sample 共用的 CLI 掃描器、空間／delay 選項政策、記憶體診斷與 setup/read/process/finite-check/consume/finish/cleanup 骨架。raw、static、DFN2、ULCNet 各自保留型別化設定與模型資源 adapter；只用於 sample，不進演算法 archive。`ctest:pipelines/4ch_aec_bf_nr_res/tests/test_mcab_sample_driver.c` 驗證錯誤短路、清理次數、有限值政策與 CLI 邊界；沒有 Python 版。 |
| PL-4DFN-01 | `pipelines/4ch_aec_bf_dfn_res/4aec_dfn_res.c: four_aec_dfn_res_process, four_aec_dfn_res_process_begin, four_aec_dfn_res_nn_begin, four_aec_dfn_res_nn_end, four_aec_dfn_res_process_end` | `pipelines/4ch_aec_bf_dfn_res/pipeline.py:post_dfn2` | mirror | D-43 | `pipelines/tests/test_dfn2_4ch_contract.py`（Py 對 Py，identity 模型） | Python 的 E 是 adapter 權重的相干投影；C wrapper 直接拿 bf 輸出，權重一致時是同一個訊號。契約測試是 Python 對 Python（identity 模型），不是 C 對拍。 |
| PL-4DFN-02 | `pipelines/4ch_aec_bf_dfn_res/4aec_dfn_res.c: four_aec_dfn_res_front_process` | `pipelines/4ch_aec_bf_dfn_res/pipeline.py:run_front_end` | partial | — | — | （推論）共用 delay 與四 lane 的前端；Python 的 `run_front_end` 另外跑融合與後波束 RES。 |
| PL-4DFN-03 | `pipelines/4ch_aec_bf_dfn_res/4aec_dfn_res.c: four_aec_dfn_res_set_atten_lim` | `pipelines/dfn2_stage.py:Dfn2Stage.set_atten_lim` | partial | — | — | （推論） |
| PL-4DFN-04 | `pipelines/4ch_aec_bf_dfn_res/4aec_dfn_res.c: four_aec_dfn_res_set_aec_preset` | `pipelines/4ch_aec_bf_nr_res/pipeline.py:PostBeamResidualSuppressor.set_preset` | partial | — | — | C 版設的是後波束 RES 的 preset（aec_res_set_preset）。 |
| PL-4DFN-05 | `pipelines/4ch_aec_bf_dfn_res/4aec_dfn_res.c: four_aec_dfn_res_lookahead_samples` | `pipelines/dfn2_stage.py:Dfn2Stage.output_delay_frames` | partial | — | — | （推論）host WOLA hop 加兩個 DFN2 lookahead hop（48 kHz）或 bridge 的 added delay；Python 只在 stage 層給 frame 數。 |
| PL-4DFN-06 | `pipelines/4ch_aec_bf_dfn_res/4aec_dfn_res.c: align_up, fnv_u32, add_region, resolve, fill_descriptor, four_aec_dfn_res_default_config, four_aec_dfn_res_get_mem_requirements, descriptor_matches, four_aec_dfn_res_init_ex, four_aec_dfn_res_init, four_aec_dfn_res_create, four_aec_dfn_res_reset, four_aec_dfn_res_destroy, spectrum_finite, fail, four_aec_dfn_res_hop_size, four_aec_dfn_res_get_front_end, four_aec_dfn_res_get_stage, four_aec_dfn_res_get_bridge`<br>`pipelines/4ch_aec_bf_dfn_res/main.c: run_accelerator, dfn_unknown, dfn_set, dfn_parse, dfn_setup, dfn_read, dfn_process, dfn_complete, dfn_cleanup, main` | — | C-only | — | — | 描述子、記憶體池、存取函式、板端範例。 |
| PL-ULC-01 | `pipelines/mono_alignulcnet/audio_pipeline_ulcnet.c: audio_pipeline_ulcnet_backend_id, ulcnet_derive_dims_and_config, ulcnet_framing_bytes, ulcnet_sub_pool_size, ulcnet_pipeline_build, ulcnet_fnv1a_str, audio_pipeline_ulcnet_build_flags_hash, audio_pipeline_ulcnet_default_config, audio_pipeline_ulcnet_get_mem_requirements, audio_pipeline_ulcnet_init_ex, audio_pipeline_ulcnet_init, audio_pipeline_ulcnet_create, audio_pipeline_ulcnet_process_begin, audio_pipeline_ulcnet_nn_begin, audio_pipeline_ulcnet_nn_end, audio_pipeline_ulcnet_process_end, audio_pipeline_ulcnet_process, audio_pipeline_ulcnet_reset, audio_pipeline_ulcnet_destroy, audio_pipeline_ulcnet_hop_size, audio_pipeline_ulcnet_get_aec`<br>`pipelines/mono_alignulcnet/main.c: usage, parse_delay_mode, delay_mode_name, parse_delay_profile, run_accelerator, main`<br>`pipelines/4ch_alignulcnet/audio_pipeline_4ch_ulcnet.c: is_bool, validate_config, audio_pipeline_4ch_ulcnet_default_config, pool_carve, fnv1a_str, audio_pipeline_4ch_ulcnet_build_flags_hash, fill_uca_geometry, fill_ula_geometry, fill_custom_geometry, fill_stack_geometry, derive_spatial_configs, framing_bytes, derive_pipeline_layout, audio_pipeline_4ch_ulcnet_get_mem_requirements, audio_pipeline_4ch_ulcnet_init_ex, audio_pipeline_4ch_ulcnet_init, audio_pipeline_4ch_ulcnet_create, audio_pipeline_4ch_ulcnet_set_model, audio_pipeline_4ch_ulcnet_process_front, audio_pipeline_4ch_ulcnet_process_beam, audio_pipeline_4ch_ulcnet_process_begin, audio_pipeline_4ch_ulcnet_nn_begin, audio_pipeline_4ch_ulcnet_nn_end, audio_pipeline_4ch_ulcnet_process_end, audio_pipeline_4ch_ulcnet_process_with_activity, audio_pipeline_4ch_ulcnet_reset, audio_pipeline_4ch_ulcnet_destroy, audio_pipeline_4ch_ulcnet_hop_size, audio_pipeline_4ch_ulcnet_fft_size, audio_pipeline_4ch_ulcnet_n_freqs, audio_pipeline_4ch_ulcnet_sample_rate, audio_pipeline_4ch_ulcnet_beam_state, audio_pipeline_4ch_ulcnet_last_core_timing, audio_pipeline_4ch_ulcnet_last_delay`<br>`pipelines/4ch_alignulcnet/main.c: usage, run_accelerator, ulc_unknown, ulc_set, ulc_parse, ulc_setup, ulc_read, ulc_process, ulc_complete, ulc_cleanup, main` | — | C-only | D-43；T-14 | — | ULCNet wrapper（mono 與 4ch）沒有 Python 版，也沒有 pipeline 層的 C 對 Python 對拍。模型側前後處理見 AIAEC 區。 |
| PL-PYO-01 | — | `pipelines/aec_nr_pipeline.py:run_aec_classic`<br>`pipelines/aec_nr_pipeline.py:parse_aec_mode` | Python-only | — | — | 經典模式（AEC 內建 RES）與 AEC 模式字串解析；兩條 C 管線都只用線性模式加外部 RES，沒有對應。 |
| PL-PYO-02 | — | `pipelines/aec_nr_pipeline.py:run_nr` | Python-only | — | — | 時域 NR 路徑（被 `run_nr_spectrum` 取代的 `--pipeline-mode linear`）；C 沒有。 |
| PL-PYO-03 | — | `pipelines/aec_nr_pipeline.py:comfort_noise_fill` | Python-only | D-37 | — | lib/aec `aec_cng_fill` 的高斯近似（檔內明載不逐位元）；真正的孿生是 `cng_aec3.py`（見 PL-PYO-04），函式屬 AEC 函式庫範圍。 |
| PL-PYO-04 | — | `pipelines/cng_aec3.py:Aec3ComfortNoise.add`<br>`pipelines/cng_aec3.py:Aec3ComfortNoise.draw`<br>`pipelines/cng_aec3.py:Aec3ComfortNoise.reset`<br>`pipelines/cng_aec3.py:lcg_step` | Python-only | D-37 | `pipelines/tests/test_cng_aec3.py`（文字與行為比對） | lib/aec `aec_cng_fill` 的逐位元孿生（18/18 組相同，一次性量測）；只接在 `aec_dfn2_res_pipeline.py`，mono 的 `run_res` 還沒接上。C 函式屬 AEC 函式庫範圍。 |
| PL-PYO-05 | — | `pipelines/aec_nr_pipeline.py:CNG_NR_GAIN_FLOOR`<br>`pipelines/aec_nr_pipeline.py:PROD_INJECT_ECHO_PSD` | Python-only | — | `pipelines/tests/test_cng_floor_constant.py`（原始碼文字比對） | 前者對應 C 巨集 SE_CNG_NR_GAIN_FLOOR（se_stage.h），四處以文字比對釘死；後者是 R² 折入 NR 的生產開關，C 的對應是 cfg.legacy_amin 的反向。 |
| PL-PYO-06 | — | `pipelines/se_stage.py:SeSpecFrame` | Python-only | — | — | C 結構 SeSpecFrame（se_stage.h）的 Python 版。欄位大致對應，但 C 的 usable_linear 在 Python 沒有欄位（Python 由 `run_nr_spectrum` 直接讀 AecResContext.usable_linear，`frame_from_context` 也不帶它）。 |
| PL-PYO-07 | — | `pipelines/dfn2_stage.py:load_dfn2`<br>`pipelines/dfn2_stage.py:Dfn2Assets`<br>`pipelines/dfn2_stage.py:IdentityHeads`<br>`pipelines/dfn2_stage.py:Dfn2Heads`<br>`pipelines/dfn2_stage.py:dfn2_modules`<br>`pipelines/dfn2_stage.py:Dfn2Stage.flush` | Python-only | D-42 | — | 模型載入與 heads 可呼叫物件（C 由加速器 callback 取代）、尾端 flush（C 由呼叫端餵零框）。 |
| PL-PYO-08 | — | `pipelines/dfn2_rate_adapter.py:CResampler`<br>`pipelines/dfn2_rate_adapter.py:round_trip`<br>`pipelines/dfn2_rate_adapter.py:_root_hann` | Python-only | — | — | CResampler 是 ctypes 對 C resampler 的綁定（實驗與對拍用）；round_trip 與 _root_hann 是 Python 實驗輔助。 |
| PL-PYO-09 | — | `pipelines/aec_dfn2_res_pipeline.py:MaskOnlyHeads`<br>`pipelines/aec_dfn2_res_pipeline.py:make_stage`<br>`pipelines/aec_dfn2_res_pipeline.py:main` | Python-only | D-43 | — | Python 專屬實驗開關（--dfn-estimate-source、--dfn-compose）與 CLI；C 管線只出貨 pre_res／full。 |
| PL-PYO-10 | — | `pipelines/4ch_aec_bf_nr_res/pipeline.py:Beamformer`<br>`pipelines/4ch_aec_bf_nr_res/pipeline.py:BeamformerFrame`<br>`pipelines/4ch_aec_bf_nr_res/pipeline.py:EqualWeightBeamformer`<br>`pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.process_signal`<br>`pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelAecPipeline.beamformer_configured` | Python-only | D-43 | `pipelines/4ch_aec_bf_nr_res/tests/test_pipeline.py`（Py 契約） | Python 參考刻意不含 SRP-PHAT／GSC，波束器是外部接縫；EqualWeightBeamformer 只供測試與離線評測。C 的對應是 audiolib 的 bf。 |
| PL-PYO-11 | — | `pipelines/4ch_aec_bf_nr_res/pipeline.py:SharedDelayState`<br>`pipelines/4ch_aec_bf_nr_res/pipeline.py:FrontFrame`<br>`pipelines/4ch_aec_bf_nr_res/pipeline.py:FourChannelFrame` | Python-only | — | `pipelines/4ch_aec_bf_nr_res/tests/test_pipeline.py`（Py 契約） | C 結構 FourAecNrResDelayState、FourAecFront、SeSpecFrame 的 Python 資料類別；結構不是函式，所以不在 C 函式表內。 |

### 7.2 pipelines/third_party

| ID | C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 | 備註 |
|---|---|---|---|---|---|---|
| TP-GSC-01 | `audiolib/bf/bf.c: bf_effective_adapt_interval, bf_get_mem_size, bf_init, bf_create, bf_process_with_weights, bf_test_process_with_weights_scalar_rls, bf_process, bf_reset, bf_destroy, bf_get_doa_used, bf_get_adaptive, bf_get_bin_resets, bf_p_diag_floor, bf_p_diag_ceil, bf_wa_leak_factor` | — | C-only | D-43 | — | 第三方 C 來源（`audiolib/bf`），沒有 Python 版（設計如此）。BF 有兩項非逐位元的數值強化（`wa_leak` 衰減、P-diagonal 上下限鉗制），記載於 `bf.h`/`bf.c` 的註解。 |
| TP-VAD-01 | `audiolib/VAD/vad.c: vad_get_mem_size, vad_config_is_valid, vad_init, vad_create, vad_destroy, vad_step, vad_get_raw, vad_get_out`<br>`audiolib/VAD/vad_api.c: vad_api_get_mem_size, vad_api_init, vad_api_create, vad_api_process, vad_api_get, vad_api_get_raw, vad_api_get_mask, vad_api_get_masker, vad_api_get_mask_vad, vad_api_destroy`<br>`audiolib/VAD/masker.c: masker_get_mem_size, masker_create, masker_init, masker_destroy, masker_step, masker_get_mask` | — | C-only | D-43 | — | 第三方 C 來源，repo 內沒有 Python 版（設計如此）。 |
| TP-DOA-01 | `audiolib/doa/srp.c: srp_get_mem_size, srp_init, srp_angle_to_index, srp_create_from_geometry, srp, srp2doa, doa_step, doa_get_raw, doa_get_smooth, srp_hold, srp_reset, srp_destroy`<br>`audiolib/doa/steering.c: array_geometry_create, array_geometry_create_uca, array_geometry_create_ula, array_geometry_create_custom, array_geometry_destroy, srp_create_uniform_angles, srp_build_steering, srp_destroy_steering`<br>`audiolib/doa/doa_smoother.c: doa_smoother_update` | — | C-only | D-43 | — | 第三方 C 來源，repo 內沒有 Python 版（設計如此）。SRP-PHAT 與幾何導出。 |
| TP-GAIN-01 | `pipelines/third_party/gain/fix_gain.c: fix_gain_db_to_linear, fix_gain_get_mem_size, fix_gain_create, fix_gain_init, fix_gain_reset, fix_gain_process, fix_gain_destroy`<br>`pipelines/third_party/gain/nr_gain.c: nr_gain_get_mem_size, nr_gain_create, nr_gain_init, nr_gain_process, nr_gain_get_gain, nr_gain_destroy`<br>`pipelines/third_party/gain/post_gain.c: post_gain_get_mem_size, post_gain_create, post_gain_init, post_gain_destroy, post_gain_apply, post_gain_get_raw_mask, post_gain_get_class, post_gain_get_gain, post_gain_get_stats` | — | C-only | D-43 | — | 第三方 C 來源，repo 內沒有 Python 版（設計如此）。 |
| TP-UTIL-01 | `audiolib/common/utility/spatial_complex.c: spatial_complex_add, spatial_complex_mul, spatial_complex_conj, spatial_complex_scale`<br>`audiolib/common/utility/spatial_simd.c: spatial_phat_cross_scalar, spatial_phat_cross, spatial_conj_beamform_scalar, spatial_conj_beamform, spatial_bf_projection_scalar, spatial_bf_projection, spatial_complex_sub_array_scalar, spatial_complex_sub_array, spatial_bf_effective_weights_scalar, spatial_bf_effective_weights, spatial_pair_score_accumulate_scalar, spatial_pair_score_accumulate, spatial_simd_backend` | — | C-only | D-43 | — | 複數與 NEON 空間運算輔助；無 Python 版。 |

### 7.3 audiolib

| ID | C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 | 備註 |
|---|---|---|---|---|---|---|
| AC-HPF-01 | `audiolib/common/src/hpf.c: hpf_compute_coeffs` | `lib/aec/python/modules/preprocessing.py:HighPassFilter.__init__` | mirror | D-01 | — | 2 階 Butterworth 的 bilinear 係數。C 用 float32，Python 用 float64，係數差落在 float32 捨入內。 |
| AC-HPF-02 | `audiolib/common/src/hpf.c: hpf_process` | `lib/aec/python/modules/preprocessing.py:HighPassFilter.process` | mirror | D-01；T-06 | — | Direct-Form-II 轉置。80 Hz 時 C 對 Python 最大差 8k 4.8e-6、16k 4.2e-5、48k 2.55e-4；經 AEC 的硬分支放大後端到端 -35~-63 dB（屬 AEC 函式庫範圍）。HPF 沒有專屬的 Python↔C golden，只有端到端間接覆蓋。 |
| AC-HPF-03 | `audiolib/common/src/hpf.c: hpf_reset` | `lib/aec/python/modules/preprocessing.py:HighPassFilter.reset` | mirror | — | — | — |
| AC-HPF-04 | `audiolib/common/src/hpf.c: hpf_params_valid` | — | C-only | S-04 | — | 參數域檢查（0 < cutoff < 0.45 * sr）；Python 的 `HighPassFilter` 與 `AecConfig` 不檢查，Nyquist 以上會靜默得到近靜音或 NaN。 |
| AC-HPF-05 | `audiolib/common/src/hpf.c: hpf_create, hpf_get_mem_size, hpf_init, hpf_destroy` | — | C-only | — | — | 建立／銷毀／記憶體池 API。 |
| AC-RS-01 | `audiolib/common/src/audio_resampler.c: audio_resampler_design` | `pipelines/dfn2_rate_adapter.py:AudioResampler._design` | mirror | D-24；D-25 | `pipelines/tests/test_dfn2_rate_adapter.py`（C↔Py，SIMD=0） | Blackman 窗 sinc 的有理多相濾波器設計；SIMD=0 對 Python twin 在 25 組比值 x 3 訊號上逐位元相同（一次性量測）。 |
| AC-RS-02 | `audiolib/common/src/audio_resampler.c: audio_resampler_process, rs_run_integer_up, rs_run_decimation, rs_run_rational` | `pipelines/dfn2_rate_adapter.py:AudioResampler.process` | mirror | D-24；T-07 | `pipelines/tests/test_dfn2_rate_adapter.py`（C↔Py，SIMD=0） | SIMD=0 逐位元相同；NEON 因四路累加順序不同，最大差 5.36e-7（24k→8k DC）。多聲道與分塊呼叫的一致性只有一次性量測。 |
| AC-RS-03 | `audiolib/common/src/audio_resampler.c: rs_dot1, rs_dot4` | `pipelines/dfn2_rate_adapter.py:AudioResampler._dot` | mirror | D-24 | `pipelines/tests/test_dfn2_rate_adapter.py`（C↔Py，SIMD=0） | NEON 版 rs_dot4 的累加順序與 Python 不同。 |
| AC-RS-04 | `audiolib/common/src/audio_resampler.c: audio_resampler_reset` | `pipelines/dfn2_rate_adapter.py:AudioResampler.reset` | mirror | — | — | — |
| AC-RS-05 | `audiolib/common/src/audio_resampler.c: audio_resampler_latency_input_frames` | `pipelines/dfn2_rate_adapter.py:AudioResampler.latency_input_frames` | mirror | — | — | — |
| AC-RS-06 | `audiolib/common/src/audio_resampler.c: audio_resampler_output_bound` | `pipelines/dfn2_rate_adapter.py:AudioResampler.output_bound` | mirror | — | — | — |
| AC-RS-07 | `audiolib/common/src/audio_resampler.c: audio_resampler_rate_supported` | `pipelines/dfn2_rate_adapter.py:SUPPORTED_RATES` | partial | — | — | （推論）C 是函式，Python 是常數元組（8/16/24/32/48 kHz）。 |
| AC-RS-08 | `audiolib/common/src/audio_resampler.c: audio_gcd, audio_resampler_layout, audio_resampler_get_mem_size, audio_resampler_carve, audio_resampler_init, audio_resampler_create, audio_resampler_destroy, audio_resampler_input_rate, audio_resampler_output_rate, audio_resampler_channels` | — | C-only | — | — | 記憶體池、carve、gcd、存取函式。 |
| AC-STFT-01 | `audiolib/common/src/audio_window.c: ac_window_build` | `pipelines/aec_nr_pipeline.py:synth_window` | mirror | D-26 | — | periodic sqrt-Hann。C 以 float32 公式計算，Python 先算 float64 再轉 float32；n=512 時 259/512 項差 1 個 float32 位元（最大 7.1e-7）。 |
| AC-STFT-02 | `audiolib/common/src/audio_stft.c: ac_stft_push, ac_stft_push_planar, ac_stft_flush, ac_stft_flush_planar` | `AIAEC/aiaec_streaming.py:StreamSTFT.push`<br>`AIAEC/aiaec_streaming.py:StreamSTFT.flush` | partial | D-33；D-34 | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py） | center=1 時對 Python 串流孿生最大差 9.5e-7（16k/512/256）；center=0 的滾動框第 j 幀（j>=1）與 center=1 的第 j 幀逐位元相同，第 0 幀本來就不同。center=0 沒有同名的 Python 孿生（訓練端用 torch.stft）。 |
| AC-STFT-03 | `audiolib/common/src/audio_stft.c: ac_stft_frame, ac_stft_frame_planar, frame_interleaved, frame_planar, transform` | `AIAEC/aiaec_streaming.py:StreamSTFT._emit` | partial | D-33 | — | （推論）單框加窗與正向 FFT；Python 以 torch.fft 計算。 |
| AC-STFT-04 | `audiolib/common/src/audio_stft.c: history_append, build_head_segment, build_tail_segment` | `AIAEC/aiaec_streaming.py:StreamSTFT.push`<br>`AIAEC/aiaec_streaming.py:StreamSTFT.flush` | partial | D-34 | — | （推論）center=1 的前後 reflect 補齊與歷史緩衝。 |
| AC-STFT-05 | `audiolib/common/src/audio_stft.c: ac_stft_reset` | `AIAEC/aiaec_streaming.py:StreamSTFT.reset` | mirror | — | — | — |
| AC-STFT-06 | `audiolib/common/src/audio_stft.c: ac_istft_push, ac_istft_push_planar, overlap_add` | `AIAEC/aiaec_streaming.py:StreamISTFT.push`<br>`pipelines/aec_nr_pipeline.py:synthesize_hop` | partial | D-27；T-08 | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py） | 兩種正規化：ENV_NORM（除以窗平方包絡）對應 `StreamISTFT`；ACCUMULATE（不除）對應管線的 `synthesize_hop`。對拍的是 consumer 測試與一次性量測，audiolib 自己的 test_stft 沒有 Python 對拍。 |
| AC-STFT-07 | `audiolib/common/src/audio_stft.c: ac_istft_flush` | `AIAEC/aiaec_streaming.py:StreamISTFT.flush` | mirror | D-27 | — | 最後一個 hop 的尾端 <=8 個樣本被包絡除法放大（n=512 時 1.73e-5），C 與 Python 孿生同形。 |
| AC-STFT-08 | `audiolib/common/src/audio_stft.c: ac_istft_reset` | `AIAEC/aiaec_streaming.py:StreamISTFT.reset` | mirror | — | — | — |
| AC-STFT-09 | `audiolib/common/src/audio_stft.c: config_valid, layout_walk, take, carve_tables, ac_stft_config_default, ac_stft_latency_samples, ac_stft_get_mem_size, ac_stft_init, ac_stft_window, ac_stft_get_config, istft_arrays, ac_istft_get_mem_size, ac_istft_init, ac_istft_window, ac_istft_get_config` | — | C-only | — | — | 設定、記憶體佈局、存取函式。 |
| AC-FFT-01 | `audiolib/common/src/fft_wrapper_ne10.c: fft_size_in_range, fft_create, fft_get_mem_size, fft_init, fft_destroy, fft_get_n_freqs, fft_forward, fft_inverse, fft_forward_scratch, fft_inverse_scratch, fft_magnitude, fft_power, fft_phase, fft_from_mag_phase, fft_apply_gain` | — | C-only | D-31；T-09 | — | NE10 後端的 FFT 包裝。外部參照是 numpy.fft（正向絕對差 <=3.95e-5 @ n=8192，一次性量測），repo 內沒有同名 Python 符號；沒有專屬的 Python golden。 |
| AC-GAIN-01 | `audiolib/common/src/audio_pre_gain.c: audio_pre_gain_convert, audio_pre_gain_get_mem_size, audio_pre_gain_init, audio_pre_gain_create, audio_pre_gain_destroy, audio_pre_gain_set_db, audio_pre_gain_get_db, audio_pre_gain_get_linear, audio_pre_gain_process` | — | C-only | D-31；T-10 | — | 前置增益；repo 內沒有消費者也沒有 Python 孿生（powf(10, dB/20) 對 float64 相對差 4.99e-7）。 |
| AC-WAV-01 | `audiolib/common/include/wav_io.h: wav_io_file_size, wav_io_skip_chunk, wav_open_read, wav_read_float, wav_close_read, wav_open_write, wav_write_float, wav_finalize_write, wav_close_write` | — | C-only | D-29；D-31 | — | WAV 讀寫（header-only）。量化慣例與 soundfile／NR／評測端不同，跨 C／Python 的 WAV 位元比對天生帶 1~2 LSB 偏差。 |
| AC-MATH-01 | `audiolib/common/utility/fast_math.h: fast_exp, fast_exp_neg, fast_log, fast_log10, fast_sqrt, exp1_approx, clip_f, max_f` | — | C-only | D-28 | — | fast_exp／fast_log／fast_sqrt 等近似；參照是 libm（USE_STANDARD_MATH 建置），沒有 Python 孿生。 |
| AC-MEM-01 | `audiolib/common/utility/mem_align.h: ck_add_size, ck_mul_size, ck_align16_size, ck_field_size, ck_field_size_reps, ac_memreq_abi_hash` | — | C-only | — | — | 尺寸運算的溢位檢查與對齊工具；`ac_memreq_abi_hash` 為 native-pointer pool descriptor 加入32-bit ABI識別，64-bit既有hash不變。 |
| AC-SIMD-01 | `audiolib/common/utility/simd_kernels.h: sk_min_f32, sk_min_f32_scalar` | `pipelines/aec_nr_pipeline.py:total_gain` | partial | D-30 | — | `total_gain` 不是純粹的 `sk_min_f32` 呼叫：Python 檔內明載核心是 `sk_min_f32` 的孿生，但函式本身還有多增益源的乘積組合與分支停用的路徑，核心以外的部分沒有 C 對應。核心本身有限正常值逐位元相同，NaN 與 +-0 的回傳規則不同（增益恆為有限正數，生產不可觀）。 |
| AC-SIMD-02 | `audiolib/common/utility/simd_kernels.h: sk_capply_gain_f32, sk_capply_gain_f32_scalar` | `pipelines/aec_nr_pipeline.py:apply_gain` | mirror | D-30 | — | S(f) = E(f) * g_total；Python 檔內明載是 `sk_capply_gain_f32` 的孿生。 |
| AC-SIMD-03 | `audiolib/common/utility/simd_kernels.h: sk_wola_accumulate_f32, sk_wola_accumulate_f32_scalar` | `pipelines/aec_nr_pipeline.py:synthesize_hop` | partial | T-08 | — | （推論）acc += x * w 的重疊相加累加；Python 在 `synthesize_hop` 內聯。 |
| AC-SIMD-04 | `audiolib/common/utility/simd_kernels.h: sk_noise_update_f32, sk_noise_update_f32_scalar` | `lib/nr/python/core/noise_estimators/mcra.py:McraNoiseEstimator.update` | partial | — | — | （推論）遞迴噪聲 PSD 更新。是 `update` 內的一步；初始化與場景轉換的落差屬 NR 函式庫範圍。 |
| AC-SIMD-05 | `audiolib/common/utility/simd_kernels.h: sk__fast_sqrt_elem, sk_ema_f32_scalar, sk_ema_f32, sk__cquad_load, sk__cquad_store, sk_cadd_f32_scalar, sk_cadd_f32, sk_sq_scale_f32_scalar, sk_sq_scale_f32, sk_clip_f32_scalar, sk_clip_f32, sk_fast_sqrt_f32_scalar, sk_fast_sqrt_f32, sk__fast_exp_elem, sk__fast_exp_neg_elem, sk__fast_log_elem, sk__fast_log10_elem, sk__exp1_approx_elem, sk_fast_exp_f32_scalar, sk_fast_exp_neg_f32_scalar, sk_fast_log_f32_scalar, sk_exp1_approx_f32_scalar, sk__fast_exp_vec, sk_fast_exp_f32, sk_fast_exp_neg_f32, sk__fast_log_vec, sk_fast_log_f32, sk_exp1_approx_f32, sk__clip_scale_to_s16_elem, sk_clip_scale_to_s16_arr_scalar, sk_clip_scale_to_s16, sk_clip_scale_to_s16_arr, sk_cmag_f32_scalar, sk_cmag_f32, sk_linear_to_db_f32, sk_asym_ema_f32_scalar, sk_asym_ema_f32, sk_scale_f32_scalar, sk_scale_f32, sk_argmax_f32_scalar, sk_argmax_f32, sk_argmax_i32, sk_argmax_sq_f32_scalar, sk_argmax_sq_f32, sk_cmac_fma_f32_scalar, sk_cmac_fma_ring_f32_scalar, sk_cmac_fma_ring_f32, sk_cmac_f32_scalar, sk_cmac_f32, sk_csub_f32_scalar, sk_csub_f32` | — | C-only | D-31；T-10 | — | 其餘元素運算核心（ema、cmag、clip、fast_* 向量版、linear_to_db、clip_scale_to_s16 等）：Python 對應是 lib/nr 與 lib/aec 函式內的 numpy 內聯式，不在本表範圍；本身只有 NEON 對純量的自測。包含 AEC 的 argmax（平方峰值／原始值／整數）、cmac（fma 版＋非 fma 版）與 csub；外層在 `delay_aec.c`／`filter_analyzer.c`／`render_signal_analyzer.c`／`utility/simd_aec.h`／`pbfdkf.c`／`aec_post.c`／`linear_filter_output.c`／pipelines 的 `4aec_projection_kernels.h` 是薄 wrapper 或直接呼叫，不影響這些檔案原有的列。`render_signal_analyzer.c` 的 max_abs 呼叫 `utility/simd_aec.h` 的 `sk_max_abs_f32`（見 AC-SIMD-07），不在本列。 |
| AC-SIMD-06 | `audiolib/common/utility/simd_kernel_nn.h: skn_all_finite_f32_scalar, skn_all_finite_cf32_scalar, skn__exp_field, skn__all_finite_view, skn_all_finite_f32, skn_all_finite_cf32, skn_fill_f32_scalar, skn_fill_f32, skn_mul_f32_scalar, skn_mul_f32, skn_power_scale_f32_scalar, skn_power_scale_f32, skn_mag_eps_f32_scalar, skn_mag_eps_f32, skn_deinterleave_cf32_scalar, skn_deinterleave_scale_cf32_scalar, skn_interleave_cf32_scalar, skn_interleave_scale_cf32_scalar, skn_deinterleave_cf32, skn_deinterleave_scale_cf32, skn_interleave_cf32, skn_interleave_scale_cf32, skn_div_floor_f32_scalar, skn_div_floor_f32, skn__ctaps_mac_range, skn_ctaps_mac_f32_scalar, skn_ctaps_mac_f32, skn_erb_expand_mask_f32_scalar, skn_erb_expand_mask_f32_neon, skn_erb_expand_mask_f32, skn_erb_band_accumulate_f32_scalar, skn_erb_band_accumulate_nch_f32_scalar, skn_erb_band_accumulate_nch_f32, skn_all_finite_rows_f32_scalar, skn_all_finite_rows_f32` | — | C-only | D-31；T-10 | — | AINR／AIAEC 前後處理用的核心（有限性掃描、planar 與 interleaved 轉換、power scale、ctaps_mac 等），只有 NEON 對純量的自測。包含 DFN2/DFN3 共用的 ERB mask 展開與 RNNoise-ERB/GTCRN 共用的 ERB band 累加；外層是各自檔案裡的薄 wrapper（見 AINR-DFN2-05／AINR-RN-02／AINR-GTCRN-03）。`skn_erb_forward_banded_f32` 列在 AINR-DFN2-02（它有自己的 partial 列指到 Python，同一 (檔案,函式) 不在兩列重複）。 |
| AC-SIMD-07 | `audiolib/common/utility/simd_aec.h: sk__cabs_np_elem,sk__cmag2_np_elem,sk__cabs_np_neon4,sk__cmag2_np_neon4,sk_cabs_np_f32_scalar,sk_cabs_np_f32,sk_cmag2_np_f32_scalar,sk_cmag2_np_f32,sk_cmag2_np_acc_f32_scalar,sk_cmag2_np_acc_f32,sk_wupdate_nlms_f32_scalar,sk_wupdate_nlms_f32,sk_wupdate_kf_f32_scalar,sk_wupdate_kf_f32,sk__pairwise_sum_leaf_scalar,sk_pairwise_sum_f32_scalar,sk__pairwise_sum_leaf_neon,sk_pairwise_sum_f32,sk__sum_sq_leaf_scalar,sk_sum_sq_pairwise_f32_scalar,sk__sum_sq_leaf_neon,sk_sum_sq_pairwise_f32,sk__coherence_ema_gate_elem,sk_coherence_ema_gate_f32_scalar,sk_coherence_ema_gate_f32,sk_ema_delta_f32_scalar,sk_ema_delta_f32,sk_n2_track_f32_scalar,sk_n2_track_f32,sk_n2_initial_track_f32_scalar,sk_n2_initial_track_f32,sk_mask_zero_f32_scalar,sk_mask_zero_f32,sk__pairwise_sum_tailfold_leaf_scalar,sk_pairwise_sum_tailfold_f32_scalar,sk__pairwise_sum_tailfold_leaf_neon,sk_pairwise_sum_tailfold_f32,sk_pairwise_sum_tailfold_b_f32_scalar,sk_pairwise_sum_tailfold_b_f32,sk__noise_spectrum_update_elem,sk_noise_spectrum_update_f32_scalar,sk_noise_spectrum_update_f32,sk__erl_bin_update_elem,sk_erl_bin_update_f32_scalar,sk_erl_bin_update_f32,sk_dec1_floorintmin_s32_scalar,sk_dec1_floorintmin_s32,sk__erl_hold_expire_elem,sk_erl_hold_expire_f32_scalar,sk_erl_hold_expire_f32,sk__no_audible_echo_gain_elem,sk_no_audible_echo_gain_f32_scalar,sk_no_audible_echo_gain_f32,sk_max_abs_f32_scalar,sk_max_abs_f32,sk__noise_floor_track_elem,sk_noise_floor_track_f32_scalar,sk_noise_floor_track_f32,sk_raise_to_f32_scalar,sk_raise_to_f32,sk__stationary_elem,sk_stationary_mask_u8_scalar,sk_stationary_mask_u8,sk_h_error_refresh_f32_scalar,sk_h_error_refresh_f32,sk_erle_accumulate_f32_scalar,sk_erle_accumulate_f32,sk_erle_release_f32_scalar,sk_erle_release_f32,sk_peak_counters_i64_scalar,sk_peak_counters_i64,sk_stationarity_hangover_i32_scalar,sk_stationarity_hangover_i32,sk_stationarity_smooth_u8_scalar,sk_stationarity_smooth_u8,sk_stationarity_flags_u8_scalar,sk_stationarity_flags_u8,sk_saturation_counts_f32_scalar,sk_saturation_counts_f32` | — | C-only | D-31；T-10 | `ctest:AEC/c_impl/test/simd_selftest_aec.c` | AEC 專用 NEON／純量核心（NLMS／Kalman 權重更新、複數模平方、CNG EMA、ERL 追蹤、pairwise 加總、穩態遮罩等），另有 8 組 kernel（16 個 scalar／NEON 入口）來自 `pbfdkf.c`、`subband_erle.c`、`render_signal_analyzer.c`、`stationarity_estimator.c`、`saturation.c` 的內嵌區塊；有狀態的 wrapper 留在各模組。AEC 呼叫端直接包含 `utility/simd_aec.h`；本列統計共用核心，`AEC/docs/c_python_function_map.md` §2.4 僅說明共用範圍，不重複列入。`render_signal_analyzer.c` 的 `rsa_max_abs` 是 `sk_max_abs_f32`／`sk_max_abs_f32_scalar` 的薄 wrapper；NaN 契約（任何位置的 NaN 都不參與最大值，numpy 則回傳 NaN）見標頭。 |
| AC-SIMD-08 | `audiolib/common/utility/simd_nr.h: skr_gain_prepare_elem,skr_gain_prepare_scalar,skr_gain_prepare_neon,skr_final_blend_elem,skr_final_blend_scalar,skr_final_blend_neon,skr_final_store_elem,skr_final_store_scalar,skr_final_store_neon,skr_min_ordered_scalar,skr_min_ordered_fast_neon,skr_min_ordered_neon,skr_vmin_lt,skr_ring_min_elem,skr_ring_min_scalar,skr_ring_min_neon,skr_restart_inert_elem,skr_restart_inert_scalar,skr_restart_inert_neon` | — | C-only | D-31；T-10 | `ctest:NR/c_impl/test/test_anr_kernels.c` | NR 專用 NEON／純量核心（MMSE-LSA gain 的 xi/(1+xi) 反推、log 域 final gain blend/store、noise tracker 的有序最小值／4-bin 融合 ring-min／inert bin 重啟）。前綴是 `skr_`（不是 `sk_`/`skn_`），避免撞名。**ANR_VECTOR 的派發邏輯不在這裡**：`gain_prepare`／`noise_estimator_min_ordered`／`_ring_min`／`_restart_inert` 四個 `#if ANR_VECTOR` 分派 wrapper 留在 NR，因為 NR 另有 `ANR_REFERENCE_KERNELS` 這個獨立於 `SK_HAVE_NEON` 的強制純量旋鈕（`NR/python/tools/test_bit_exact.py` 用它產生參照建置），改成直接依 `SK_HAVE_NEON` 派發會讓它悄悄失效（見 `simd_nr.h` 檔首註解）。`final_blend`/`final_store` 在 `#if defined(USE_FAST_GAIN_SMOOTHING) && ANR_VECTOR` 下直接呼叫 `skr_final_blend_neon`/`skr_final_store_neon`。`spp_dd_*`（spp_estimator.c）不在此列：它吃整個不透明的 `SppEstimator*` 並直接寫入三個私有陣列。Python 對應關係見 `NR/docs/c_python_function_map.md`。 |

### 7.4 AINR

| ID | C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 | 備註 |
|---|---|---|---|---|---|---|
| AINR-DFN2-01 | `AINR/DeepFilterNet2/dfn2_process.c: df_common_features, dfn2_compute_features` | `AINR/DeepFilterNet2/train.py:extract_dfn2_features`<br>`pipelines/dfn2_stage.py:Dfn2Stage._features` | mirror | S-17；D-35；T-13 | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py）；`pipelines/tests/test_dfn2_stage_c_parity.py`（C↔Py） | 1499 幀含數位靜音：ERB 特徵 7.82e-6、SPEC 2.38e-6、EMA 狀態 3.0e-4 dB。補數係數 `1-alpha` 的取整造成約 2e-6 的系統偏差；既有測試只涵蓋 9~30 幀。 |
| AINR-DFN2-02 | `audiolib/common/utility/simd_kernel_nn.h: skn_erb_forward_banded_f32` | `AINR/DeepFilterNet2/train.py:erb_band_db` | partial | — | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py） | （推論）帶狀 ERB 前向矩陣乘；dB 轉換在 `df_common_features`。核心是共用標頭的 `skn_erb_forward_banded_f32`；DFN3 呼叫同一個函式，呼叫點在 `dfn3_process.c`（該檔不在本表的涵蓋清單內，故無對應列）。 |
| AINR-DFN2-03 | `AINR/DeepFilterNet2/dfn2_process.c: dfn2_analysis` | `AINR/DeepFilterNet2/inference.py:process_file` | partial | D-33 | — | center=False 串流 STFT 對 torch.stft(center=True) 的第 k 幀（k>=1）最大差 1.9e-6；第 0 幀（零歷史對 reflect）不同。 |
| AINR-DFN2-04 | `AINR/DeepFilterNet2/dfn2_process.c: dfn2_synthesis` | `AINR/DeepFilterNet2/inference.py:process_file` | partial | D-33 | — | （推論）對 torch.istft(center=True) 內部區最大差 4.8e-7。 |
| AINR-DFN2-05 | `AINR/DeepFilterNet2/dfn2_process.c: df_common_expand_mask, df_common_expand_mask_scalar, df_common_expand_mask_neon` | `pipelines/dfn2_stage.py:expand_erb_mask`<br>`AINR/DeepFilterNet2/model.py:DeepFilterNet2.compose` | mirror | S-21 | `AINR/DeepFilterNet2/tests/test_dfn2_contract.py`（Py 契約） | ERB mask 以 band-major 逆矩陣展開到 bin；矩陣版面是契約（`--output-dir` 匯出包的 inverse 版面不同）。 |
| AINR-DFN2-06 | `AINR/DeepFilterNet2/dfn2_process.c: df_common_deep_filter` | `AINR/DeepFilterNet2/model.py:deep_filter_apply` | mirror | — | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py） | 深度濾波器的複數 FIR；compose 整體對 model.compose 最大差 2.98e-8。 |
| AINR-DFN2-07 | `AINR/DeepFilterNet2/dfn2_process.c: dfn2_compose` | `AINR/DeepFilterNet2/model.py:DeepFilterNet2.compose` | mirror | — | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py） | — |
| AINR-DFN2-08 | `AINR/DeepFilterNet2/dfn2_process.c: dfn2_compose_stream` | `pipelines/dfn2_stage.py:Dfn2Stage._compose` | mirror | D-42 | `pipelines/tests/test_dfn2_stage_c_parity.py`（C↔Py） | 每個 emitted frame 逐位元相同。 |
| AINR-DFN2-09 | `AINR/DeepFilterNet2/dfn2_process.c: df_common_atten_lim, dfn2_apply_atten_lim` | `AINR/DeepFilterNet2/inference.py:apply_atten_lim` | mirror | — | — | atten_lim 的 +12/-12/0/40/100 dB 逐位元相同（3 dB 差 1.19e-7）。 |
| AINR-DFN2-10 | — | `AINR/DeepFilterNet2/model.py:DeepFilterNet2.post_filter` | Python-only | S-16 | — | Valin post-filter（--pf-beta／config `mask_pf`，出貨關閉）。C 沒有對應符號；出貨路徑（mask_pf=false）兩邊無差。 |
| AINR-DFN2-11 | `AINR/DeepFilterNet2/dfn2_process.c: dfn2_set_erb_matrices`<br>`AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_prepost_set_erb_matrices` | `AINR/DeepFilterNet2/export_erb_matrix.py:write_runtime_bins` | partial | S-21 | `AINR/DeepFilterNet2/tests/test_dfn2_contract.py`（Py 契約） | C 載入呼叫端提供的 ERB 矩陣（bin-major forward、band-major inverse）；寫出端是 `--runtime-bins` 路徑。 |
| AINR-DFN2-12 | `AINR/DeepFilterNet2/dfn2_process.c: erb_spans, dfn2_framing_config, dfn2_state_init, dfn2_simd_backend` | — | C-only | — | — | band 範圍計算、framing 設定、狀態初始化、SIMD 後端查詢。 |
| AINR-DFN2-13 | `AINR/DeepFilterNet2/dfn2_model_io.c: dfn2_model_io_push_erb_window, dfn2_model_io_push_spec_window, dfn2_model_io_push_features` | `AINR/DeepFilterNet2/export_onnx.py:feature_windows`<br>`pipelines/dfn2_stage.py:Dfn2Stage._push_window` | mirror | T-14 | — | [t-1, t, t+1] 圖輸入視窗滑動；40 窗含零 flush 逐位元相同（一次性量測）。 |
| AINR-DFN2-14 | `AINR/DeepFilterNet2/dfn2_model_io.c: dfn2_model_io_init, dfn2_model_io_descriptor_default, dfn2_model_io_descriptor_validate, all_finite, dfn2_model_io_validate_arrays, dfn2_model_io_validate_state, dfn2_model_io_reset, dfn2_model_io_zero_recurrent, dfn2_model_io_state_generation` | — | C-only | T-14 | — | 描述子驗證、遞迴狀態的有限性檢查與歸零、state generation；只有 C 對 C 或錄製 digest 的測試。 |
| AINR-DFN2-15 | `AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_prepost_pre_process_freq, dfn2_prepost_pre_process_freq_dual, dfn2_prepost_frame_inputs, dfn2_prepost_frame_commit, dfn2_prepost_post_process_freq, pp_begin_frame, pp_close_frame` | `pipelines/dfn2_stage.py:Dfn2Stage.push` | partial | D-42 | `pipelines/tests/test_dfn2_stage.py`（Py 契約） | FREQ 模式；Python 的孿生是雙輸入版（估計頻譜 A、應用頻譜 B），C 類別的單一暫存對在孿生中拆成兩個。 |
| AINR-DFN2-16 | `AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_prepost_frame_skip` | `pipelines/dfn2_stage.py:Dfn2Stage.push` | partial | D-42 | — | fail-open：單位 mask、零 tap、alpha 0。C 在 commit 被拒時就地歸零遞迴狀態，Python 不步進狀態。 |
| AINR-DFN2-17 | `AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_prepost_reset` | `pipelines/dfn2_stage.py:Dfn2Stage.reset` | partial | D-42 | — | （推論）重置類別與其 DFN2State；Python 重置自己的環與狀態，fail-open 後的狀態語意見 D-42。 |
| AINR-DFN2-18 | `AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_prepost_set_atten_lim` | `pipelines/dfn2_stage.py:Dfn2Stage.set_atten_lim` | mirror | — | — | — |
| AINR-DFN2-19 | `AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_prepost_output_frame_index` | `pipelines/dfn2_stage.py:StageOutput.frame_index` | partial | — | — | （推論）C 是存取函式，Python 把來源 frame 序號放在 `StageOutput.frame_index`。 |
| AINR-DFN2-20 | `AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_prepost_model_lookahead_frames` | `pipelines/dfn2_stage.py:MODEL_LOOKAHEAD` | partial | — | — | （推論）C 是存取函式，Python 是常數。 |
| AINR-DFN2-21 | `AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_prepost_hop_size` | `pipelines/dfn2_stage.py:HOP` | partial | — | — | （推論）C 是存取函式，Python 是常數。 |
| AINR-DFN2-22 | `AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_prepost_num_bins` | `pipelines/dfn2_stage.py:N_BINS` | partial | — | — | （推論）C 是存取函式，Python 是常數。 |
| AINR-DFN2-23 | `AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_model_run_frame` | `pipelines/dfn2_stage.py:Dfn2Heads.__call__` | partial | — | — | （推論）加速器 callback 邊界；Python 端是 heads 可呼叫物件。 |
| AINR-DFN2-24 | `AINR/DeepFilterNet2/dfn2_prepost.c: dfn2_prepost_pre_process, dfn2_prepost_post_process` | — | C-only | D-33 | — | TIME 模式（由類別自己擁有 STFT／WOLA）；Python 側由 inference.py 的 torch.stft／istft 取代，見 AINR-DFN2-03、04。 |
| AINR-DFN2-25 | `AINR/DeepFilterNet2/dfn2_prepost.c: pp_carve, pp_fnv1a_u32, pp_fnv1a_str, pp_build_hash, pp_framing_sizes, pp_layout, dfn2_prepost_config_defaults, pp_check_config, dfn2_prepost_get_mem_size, pp_reset_states, pp_clear_frame, pp_zero_staging, dfn2_prepost_init_ex, dfn2_prepost_init, dfn2_prepost_create, dfn2_prepost_destroy, dfn2_prepost_state_generation, dfn2_prepost_io_mode, dfn2_prepost_layout_version, pp_all_finite, pp_fill_nan` | — | C-only | T-14 | — | 記憶體池佈局、設定檢查、build hash、生命週期、存取函式。 |
| AINR-GTCRN-01 | `AINR/GTCRN/gtcrn_process.c: gtcrn_model_input` | `AINR/GTCRN/stream_model.py:stream_features` | mirror | — | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py） | 主機側 ERB 前端（mag／re／im）；mag 3.8e-6、re／im 9.5e-7（一次性量測）。 |
| AINR-GTCRN-02 | `AINR/GTCRN/gtcrn_process.c: gtcrn_model_output` | `AINR/GTCRN/stream_model.py:host_synthesis` | mirror | — | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py） | 複數比值遮罩套用與 ERB 逆展開；200 幀逐位元相同。 |
| AINR-GTCRN-03 | `AINR/GTCRN/gtcrn_process.c: erb_band_channel_scalar, erb_band3, erb_band3_neon` | `AINR/GTCRN/model.py:ERB.bm` | mirror | S-33 | — | （推論）低頻 bin 直通、高頻 192 個 bin 折成 64 個 band（band merge）。C 用匯出的前向表，Python 用 nn.Linear 權重；表由 export_erb_matrix.py 的 runtime-bins 路徑匯出。 |
| AINR-GTCRN-04 | `AINR/GTCRN/gtcrn_process.c: mask_expand, mask_expand_scalar, mask_expand_neon` | `AINR/GTCRN/model.py:ERB.bs` | mirror | S-33 | — | （推論）64 band 逆展開回 192 個高頻 bin（band split，band-major 逆表）。 |
| AINR-GTCRN-05 | `AINR/GTCRN/gtcrn_process.c: gtcrn_analysis, gtcrn_synthesis` | `AINR/GTCRN/inference.py:process_file` | partial | D-33 | — | （推論）center=False 串流 STFT／ISTFT 對 torch.stft／istft(center=True) 最大差 1.6e-7（相對峰值）。 |
| AINR-GTCRN-06 | `AINR/GTCRN/gtcrn_process.c: gtcrn_framing_config, gtcrn_process_init, gtcrn_model_state_init, gtcrn_model_state_reset, gtcrn_model_state_generation, all_finite, state_tensors, gtcrn_model_state_validate, gtcrn_simd_backend` | — | C-only | — | — | 狀態張量初始化、驗證與 generation、framing 設定、SIMD 後端查詢。 |
| AINR-RN-01 | `AINR/RNNoise-ERB/process.c: rnnoise_analysis` | `AINR/RNNoise-ERB/train.py:stft` | mirror | D-32 | `AINR/RNNoise-ERB/tests/test_python_c_features.py`（C↔Py） | caller-owned AcStft 對完整 512-sample frame 做 center=False、normalized=True 分析；窗、縮放與 framing 不變，FFT 經 audiolib NE10。 |
| AINR-RN-02 | `AINR/RNNoise-ERB/process.c: rnnoise_compute_features` | `AINR/RNNoise-ERB/train.py:extract_model_features` | mirror | S-17；T-13 | `AINR/RNNoise-ERB/tests/test_python_c_features.py`（C↔Py） | 相同頻譜輸入的 feature/state 與 FFT 收斂前逐位元相同；FFT 捨入經低能量 band 的 log 放大，與 torch 非逐位元。 |
| AINR-RN-03 | `AINR/RNNoise-ERB/process.c: rnnoise_expand_gains` | `AINR/RNNoise-ERB/train.py:apply_erb_gains_batch` | partial | — | — | bin_gains = gains @ erb_inv.T；Python 同函式還含乘上頻譜與邊界 frame 補 1。對 `gains @ erb_inv.T` 差 5.96e-8（1 ULP）。 |
| AINR-RN-04 | `AINR/RNNoise-ERB/process.c: rnnoise_synthesis` | `AINR/RNNoise-ERB/train.py:istft` | mirror | D-32 | `AINR/RNNoise-ERB/tests/test_python_c_features.py`（C↔Py） | 保留 spec *= gain*sqrt(N) 的順序與頻譜修改；caller-owned AcIstft 執行 IFFT、root-Hann 與 ACCUMULATE overlap-add，首幀即輸出一 hop。 |
| AINR-RN-05 | `AINR/RNNoise-ERB/process.c: rnnoise_apply_atten_lim` | `AINR/RNNoise-ERB/inference.py:apply_atten_lim` | mirror | — | `AINR/RNNoise-ERB/tests/test_python_c_features.py`（C↔Py） | — |
| AINR-RN-06 | `AINR/RNNoise-ERB/process.c: rnnoise_framing_config` | — | C-only | D-32 | `AINR/RNNoise-ERB/test/test_rnnoise_framing.c` | 建立固定 512/512/256、center=0、正向 N^-0.5、反向 scale=1 的外部 framing 設定；RNNoise synthesis 自己完成反正規化。 |
| AINR-RN-07 | `AINR/RNNoise-ERB/process.c: rnnoise_state_init, rnnoise_model_state_init, hidden_finite, rnnoise_model_state_validate, rnnoise_model_state_inherit, rnnoise_simd_backend` | — | C-only | — | `AINR/RNNoise-ERB/test/test_rnnoise_framing.c` | framing 綁定／驗證與重置、模型狀態初始化／驗證／inherit、SIMD 後端查詢。pre/post layout=2，GRU tensor layout=1。 |
| AINR-RN-08 | `AINR/RNNoise-ERB/gen_rnnoise_tables.c: freq2erb` | `AINR/RNNoise-ERB/train.py:freq2erb` | mirror | — | — | — |
| AINR-RN-09 | `AINR/RNNoise-ERB/gen_rnnoise_tables.c: erb2freq` | `AINR/RNNoise-ERB/train.py:erb2freq` | mirror | — | — | — |
| AINR-RN-10 | `AINR/RNNoise-ERB/gen_rnnoise_tables.c: compute_erb_tables` | `AINR/RNNoise-ERB/train.py:erb_bandborder`<br>`AINR/RNNoise-ERB/train.py:compute_erb_matrix` | mirror | — | — | ERB 邊界與前向／逆矩陣在 16k/22 band/N=512 上逐位元相同（一次性量測）。 |
| AINR-RN-11 | `AINR/RNNoise-ERB/gen_rnnoise_tables.c: compute_window_table, print_float_hex, print_header, main` | — | C-only | D-26 | — | 表產生器的視窗表與輸出格式化。視窗對 torch.sqrt(hann_window) 差 8.08e-7（僅窗邊緣）。 |
| AINR-RN-12 | — | `AINR/RNNoise-ERB/inference.py:valin_post_filter` | Python-only | S-14 | — | Valin post-filter（--pf-beta，預設 0）。C 沒有對應符號；出貨路徑（beta=0）兩邊無差。 |

### 7.5 AIAEC

| ID | C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 | 備註 |
|---|---|---|---|---|---|---|
| AIAEC-UL-01 | `AIAEC/Align_ULCNet/ulcnet_model_io.c: ulcnet_model_io_prepare, front_end, front_end_scalar, front_end_neon, front_end_tail_one` | `AIAEC/Align_ULCNet/export_onnx.py:stream_features` | mirror | D-34 | `AIAEC/tests/test_ulcnet_model_io_c.py`（C 契約）；`AIAEC/tests/test_prepost_kernels.py`（NEON 對純量） | 有號冪壓縮、幅度與壓縮域相位方向。16k/257 bin 與 48k/513 bin：error_mag 4.77e-7、far_mag 9.54e-7（一次性量測）。 |
| AIAEC-UL-02 | `AIAEC/Align_ULCNet/ulcnet_model_io.c: ulcnet_model_io_commit, back_end, back_end_scalar, back_end_neon` | `AIAEC/Align_ULCNet/export_onnx.py:host_back_end` | mirror | D-34 | `AIAEC/tests/test_ulcnet_model_io_c.py`（C 契約）；`AIAEC/tests/test_prepost_kernels.py`（NEON 對純量） | 壓縮誤差乘上圖輸出的平面複數 mask，再做逆有號冪。commit 相對誤差 8e-8~5.8e-7。 |
| AIAEC-UL-03 | `AIAEC/Align_ULCNet/ulcnet_model_io.h: ulcnet_model_io_signed_pow` | `AIAEC/Align_ULCNet/model.py:_signed_power` | mirror | — | — | 檔內註解明載是 model.py `_signed_power` 的唯一 C 副本。 |
| AIAEC-UL-04 | `AIAEC/Align_ULCNet/ulcnet_prepost.c: ulcnet_prepost_pre_process, ulcnet_prepost_pre_process_freq` | `AIAEC/aiaec_streaming.py:StreamSTFT.push` | partial | D-34；T-14 | `AIAEC/tests/test_ulcnet_prepost_c.py`（C 對 C） | （推論）pre/post 類別擁有的是 center=0 滾動框，Python 孿生是 center=1；類別本身只有 C 對 C 的 memcmp 等價測試。 |
| AIAEC-UL-05 | `AIAEC/Align_ULCNet/ulcnet_prepost.c: ulcnet_prepost_post_process, ulcnet_prepost_post_process_freq` | `AIAEC/aiaec_streaming.py:StreamISTFT.push` | partial | D-34；T-14 | `AIAEC/tests/test_ulcnet_prepost_c.py`（C 對 C） | （推論）synthesis 半邊（WOLA）；STFT 側是 center=0 滾動框，Python 孿生為 center=1。 |
| AIAEC-UL-06 | `AIAEC/Align_ULCNet/ulcnet_prepost.c: ulcnet_prepost_frame_inputs` | `AIAEC/Align_ULCNet/export_onnx.py:stream_features` | partial | T-14 | — | （推論）把一框頻譜送進 model I/O 的 prepare。 |
| AIAEC-UL-07 | `AIAEC/Align_ULCNet/ulcnet_prepost.c: ulcnet_prepost_frame_commit` | `AIAEC/Align_ULCNet/export_onnx.py:host_back_end` | partial | T-14 | — | （推論）把圖輸出送進 model I/O 的 commit。 |
| AIAEC-UL-08 | `AIAEC/Align_ULCNet/ulcnet_process.c: ulcnet_framing_config` | `AIAEC/aiaec_streaming.py:StreamSTFT.__init__` | partial | D-34 | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py） | AcStft／AcIstft 的 ULCNet 設定（窗、hop、正規化模式）。該測試以這個設定驅動 AcStft／AcIstft，逐框對 `StreamSTFT`／`StreamISTFT` 比對（一次性量測：center=0 滾動框第 j 幀對 `StreamSTFT(center=True)` 最大差 9.5e-7）。 |
| AIAEC-UL-09 | `AIAEC/Align_ULCNet/ulcnet_prepost.c: pp_carve, pp_fnv1a, pp_build_hash, pp_framing_sizes, pp_layout, ulcnet_prepost_config_defaults, pp_check_config, pp_descriptor_for, ulcnet_prepost_get_mem_size, ulcnet_prepost_init_ex, ulcnet_prepost_init, ulcnet_prepost_create, ulcnet_prepost_destroy, ulcnet_prepost_reset, ulcnet_prepost_state_generation, ulcnet_prepost_hop_size, ulcnet_prepost_num_bins, ulcnet_prepost_io_mode, ulcnet_prepost_descriptor, pp_close_frame, ulcnet_prepost_frame_skip`<br>`AIAEC/Align_ULCNet/ulcnet_process.c: ulcnet_frame_is_finite`<br>`AIAEC/Align_ULCNet/ulcnet_model_io.c: checked_add, checked_mul, align_up, add_float_region, add_float_regions, compute_counts, ulcnet_model_io_descriptor_default, ulcnet_model_io_align_up, ulcnet_far_input_mode_name, ulcnet_model_io_descriptor_validate, ulcnet_model_io_get_mem_requirements, carve_float, ulcnet_model_io_init, clear_recurrent_state, ulcnet_model_io_reset, ulcnet_model_io_state_generation, frame_is_finite, ulcnet_model_io_descriptor`<br>`AIAEC/Align_ULCNet/ulcnet_accelerator_adapter.c: adapter_config, adapter_layout, ulcnet_accelerator_adapter_get_mem_size, ulcnet_accelerator_adapter_init, infer, reset, ulcnet_accelerator_adapter_state_generation, ulcnet_accelerator_adapter_descriptor, ulcnet_accelerator_adapter_model` | — | C-only | T-14 | — | 記憶體池佈局、描述子驗證、遞迴狀態環（in-place）、有限性掃描、加速器 adapter 與 callback 步進；沒有 Python 孿生。 |
| AIAEC-DV-01 | `AIAEC/DeepVQE_S/deepvqe_prepost.c: compress_planar` | `AIAEC/aiaec_common.py:compressed_ri_feature` | mirror | — | `AIAEC/tests/test_prepost_kernels.py`（NEON 對純量） | 壓縮 mic／far 最大差 4.77e-7（一次性量測）。 |
| AIAEC-DV-02 | `AIAEC/DeepVQE_S/deepvqe_process.c: deepvqe_ccm_init, deepvqe_ccm_process, ccm_edge_bin` | `AIAEC/DeepVQE_S/model.py:DeepVQES._apply_ccm_frame`<br>`AIAEC/aiaec_common.py:apply_causal_tf_filter`<br>`AIAEC/_export_common.py:COMPOSITION` | mirror | — | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py） | 複數卷積遮罩（CCM）；最大差 7.63e-6（峰值 95）。兩個邊緣 bin 的越界 tap 略過，與乘零不同。 |
| AIAEC-DV-03 | `AIAEC/DeepVQE_S/deepvqe_prepost.c: deepvqe_prepost_pre_process, deepvqe_prepost_pre_process_freq, deepvqe_prepost_frame_inputs` | `AIAEC/aiaec_streaming.py:StreamSTFT.push`<br>`AIAEC/aiaec_common.py:compressed_ri_feature` | partial | D-34；T-14 | `AIAEC/tests/test_deepvqe_prepost_c.py`（C 對 C） | （推論）center=0 滾動框加壓縮特徵。 |
| AIAEC-DV-04 | `AIAEC/DeepVQE_S/deepvqe_prepost.c: deepvqe_prepost_post_process, deepvqe_prepost_post_process_freq, deepvqe_prepost_frame_commit` | `AIAEC/DeepVQE_S/model.py:DeepVQES._apply_ccm_frame`<br>`AIAEC/aiaec_streaming.py:StreamISTFT.push` | partial | D-34；T-14 | `AIAEC/tests/test_deepvqe_prepost_c.py`（C 對 C） | （推論）CCM 輸出頻譜經 WOLA 合成。 |
| AIAEC-DV-05 | `AIAEC/DeepVQE_S/deepvqe_prepost.c: deepvqe_prepost_state_shape, deepvqe_prepost_state_elements, deepvqe_prepost_state_name, deepvqe_prepost_state_name_out, deepvqe_prepost_skip_policy_name, dv_valid_depth` | `AIAEC/_streaming_export.py:StatelessOneFrameAIAEC.state_names` | partial | T-14 | — | （推論）C 側的狀態張量名稱與形狀逐字對應匯出器的 input_names；契約變動要同時 bump layout 版本。 |
| AIAEC-DV-06 | `AIAEC/DeepVQE_S/deepvqe_prepost.c: deepvqe_prepost_descriptor_default, deepvqe_prepost_descriptor_validate, deepvqe_prepost_descriptor` | `AIAEC/_streaming_export.py:c_descriptor` | partial | T-14 | — | （推論）描述子對匯出器寫入的 metadata。 |
| AIAEC-DV-07 | `AIAEC/DeepVQE_S/deepvqe_prepost.c: pp_carve, pp_fnv1a, pp_build_hash, pp_framing_config, pp_layout, deepvqe_prepost_config_defaults, pp_check_config, deepvqe_prepost_get_mem_size, deepvqe_prepost_init_ex, deepvqe_prepost_init, deepvqe_prepost_create, deepvqe_prepost_destroy, zero_state, deepvqe_prepost_reset, deepvqe_prepost_state_generation, deepvqe_prepost_hop_size, deepvqe_prepost_num_bins, deepvqe_prepost_io_mode, fill_nan, all_finite, pp_close_frame, newest_frame_finite, deepvqe_prepost_frame_skip` | — | C-only | T-14 | — | 記憶體池佈局、設定檢查、build hash、生命週期、有限性掃描、NaN 填充；沒有 Python 孿生。 |
| AIAEC-MASK-01 | `AIAEC/aiaec_process.c: aiaec_apply_complex_mask` | `AIAEC/aiaec_common.py:complex_mask`<br>`AIAEC/_export_common.py:COMPOSITION` | mirror | — | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py，逐位元） | CAGCRN 的 host 複數 mask 乘法（`_export_common.COMPOSITION` 表指定）。 |
| AIAEC-MASK-02 | `AIAEC/aiaec_process.c: aiaec_apply_ulcnet_compressed_mask, aiaec_signed_pow` | `AIAEC/Align_ULCNet/export_onnx.py:host_back_end`<br>`AIAEC/_export_common.py:COMPOSITION` | mirror | D-34 | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py，<2e-5） | Align-ULCNet 的 host 後處理（`_export_common.COMPOSITION` 表指定），與 AIAEC-UL-02 的 back_end 是同一數學。 |
| AIAEC-MASK-03 | `AIAEC/aiaec_process.c: aiaec_apply_real_mask` | `AIAEC/Align_CRUSE/model.py:AlignCRUSE.forward`<br>`AIAEC/_export_common.py:COMPOSITION` | partial | D-36 | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py，逐位元，對內聯 torch 算式） | 實數幅度 mask（Align-CRUSE 的 `enhanced = microphone * mask`）。Align-CRUSE 沒有 C pre/post 類別，C 只提供這個共用 helper。 |

<!-- map:rows:end -->

## 8. 落差編號對照

**S** 為 Python/C 落差（部分未獨立驗證），**D** 為文件或註解明載的刻意差異，**T** 為缺少對拍的項目。此處只留一行摘要。

<!-- map:gaps:begin -->
| 編號 | 摘要 |
|---|---|
| D-01 | mic 路徑 HPF：C float32 DF2T 對 Python float64，非逐位元（文件化）。 |
| D-23 | Python 與 C 對 NR 強度名稱的呈現方式不同（文件化）。 |
| D-24 | audio_resampler 的 NEON 版與 Python twin 非逐位元（累加順序）；SIMD=0 逐位元相同。 |
| D-25 | 本專案 house resampler 與訓練／評估端 torchaudio、NR np.interp 不是同一個濾波器。 |
| D-26 | ac_window_build（float32 公式）與 numpy／torch 窗非逐位元。 |
| D-27 | AcIstft ENV_NORM 的 flush 最後一個 hop 尾端被包絡除法放大，C 與 Python twin 同形。 |
| D-28 | fast_math 近似對 libm：最大相對誤差約 4e-3（fast_exp），文件化。 |
| D-29 | wav_io 的量化慣例與 soundfile／NR／評測端不同，帶 1~2 LSB 偏差。 |
| D-30 | sk_min／sk_capply_gain 的邊界語意（NaN、+-0）與 numpy 不同；增益恆為有限正數，生產不可觀。 |
| D-31 | fft_wrapper、fast_math、simd_kernels、wav_io、audio_pre_gain 無 Python 孿生。 |
| D-32 | RNNoise 經 audiolib NE10 FFT／IFFT；數學定義、窗與縮放不變，butterfly 捨入與 torch 仍非逐位元。低能量 band 的 log 會放大微小誤差；窗本身的差異另見 D-26。 |
| D-33 | 離線 Python 用 center=True，C 串流用 center=False，第 0 幀不同。 |
| D-34 | AIAEC pre/post 類別用 center=0 滾動框，Python 訓練／評測用 center=1。 |
| D-35 | spec_norm_eps：C 保留 1e-12，trainer 沒有。 |
| D-36 | Align_CRUSE、CAGCRN 沒有 C pre/post 類別，只共用 aiaec_process.c 的 mask helper。 |
| D-37 | mono comfort noise：Python 用高斯近似，C 用 LCG；cng_aec3 才是逐位元孿生。 |
| D-38 | se_derive 寫出的 AecConfig struct 值與 Python 有效值有 10 個欄位不同（10 ms 基準、carve 時 retime）。 |
| D-39 | 4ch 共用 estimator 的 C-only 排程（duty-cycle）與 delay 模式（FIXED、EXTERNAL_ALIGNED、delay_num_filters）。 |
| D-40 | 8 kHz mono 只有 C；mono CLI 旗標不對等（另見 S-30、S-31）。 |
| D-41 | R² 相量融合的相位下限：C 釘在 +實軸，Python 用 np.angle。 |
| D-42 | DFN2 fail-open：Python 不步進遞迴狀態，C 在 commit 被拒時就地歸零。 |
| D-43 | third_party（gain）與 audiolib（SRP-PHAT、BF、VAD）與 ULCNet wrapper 沒有 Python 版（設計如此）；Python 另有實驗專用開關。 |
| S-04 | HPF 參數域檢查（0 < cutoff < 0.45 * sr）只在 C；Python 的 HighPassFilter／AecConfig 不檢查。 |
| S-05 | AecResContext.filter_converged 在上升緣晚 1 hop：Python 在 update_convergence 之前建構 ctx。 |
| S-10 | NR 的 shared xi-ratio 在 bin 功率近 0 時 gain 與 Python 差 0.12~0.43（屬 NR 函式庫範圍；此處僅因 se_anr_stage_gain 會呼叫它而列出）。 |
| S-11 | NR MCRA 初始噪聲 PSD 下限 1e-10 只在 C（屬 NR 函式庫範圍）。 |
| S-14 | RNNoise-ERB 的 Valin post-filter（--pf-beta）只有 Python，C 沒有；出貨 beta=0 兩邊無差。 |
| S-16 | DFN2 的 Valin post-filter（config `mask_pf`）只有 Python，C 沒有；出貨 mask_pf=false 兩邊無差。 |
| S-17 | EMA 補數 1-alpha 的 float32 取整：C 在 float32 算 1-alpha，Python 在 double 算完再轉，造成約 2e-6 的系統偏差。 |
| S-21 | DFN2 `export_erb_matrix.py --output-dir` 匯出包（工具鏈驗證用）的 inverse 版面與 C 期待的 band-major 不同；C 載入 `--runtime-bins` 寫出的檔。 |
| S-30 | Python mono 參考管線不支援 8 kHz，C 支援 8k/256；未獨立驗證。 |
| S-31 | mono CLI 旗標不對等（moderate、--no-nr、--no-res、--no-cng 只在 C）；未獨立驗證。 |
| S-32 | 4ch Python 參考的設定面比 C 少（無 delay_mode、nr_mode、enable_*、legacy_amin 等）；未獨立驗證。 |
| S-33 | GTCRN `export_erb_matrix.py` 的 bin／npy 匯出包（工具鏈驗證用）與 C 載入的 `--runtime-bins` 檔不同；未獨立驗證。 |
| S-34 | 設定範圍檢查不對稱：C validate 嚴格 fail-closed，Python 幾乎不驗。 |
| T-06 | HPF 沒有 Python↔C 專屬 golden，只有端到端間接覆蓋。 |
| T-07 | resampler 既有 Python 對拍覆蓋不足（補量後無落差）。 |
| T-08 | audiolib 的 test_stft 驗窗公式、延遲、API 契約與縮放入口，沒有對 Python 的逐位元對拍。 |
| T-09 | FFT 沒有專屬的 Python golden。 |
| T-10 | 沒有 Python 對照的 C 元件（audio_pre_gain、mem_align、部分 simd 核心等）。 |
| T-13 | 標頭常數沒有依名稱對 config／train.py 的漂移守衛。 |
| T-14 | 沒有 Python 孿生的 C 狀態機（prepost、model_io、adapter）。 |
| T-15 | pipelines 內真正執行 C 並與 Python 對拍的常設測試只有 delay、DFN2 stage、rate bridge（與 resampler twin）幾個；mono 管線組合、se_derive／se_config、se_anr_stage、4ch fuse 與後波束 RES、ULCNet wrapper 沒有常設對拍。 |
| T-16 | 4ch 後波束 RES 的 clock_drift 輸入：C 傳共用 estimator 旗標，Python 寫死 False；無測試。 |
| T-17 | mono 端到端與上游 AEC 分岔（歸屬 lib/aec 接縫，非 pipeline 膠水）。 |
<!-- map:gaps:end -->

## 9. 未能判斷

目前沒有。這不表示每一列都有證據：標「（推論）」的列（第 4 節統計的最後一欄）是只憑函式名稱與角色得出的配對，證據最弱，要引用它們的對拍結果前應先確認。

