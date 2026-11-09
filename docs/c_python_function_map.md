# C 與 Python 函式對照表（pipelines、audiolib、AINR、AIAEC）

- 範圍：本檔所在的 Audio_ALG 樹，加上同層的 audiolib 檢出。表內所有路徑都相對於 Audio_ALG 根目錄；`audiolib/common/…` 指同層的 audiolib 檢出（pipelines/Makefile 的 `COMMON_DIR`）。
- 守衛測試：`pipelines/tests/test_c_python_map.py`
- 語言：繁體中文說明，英文識別字。
- 命名橋：強度在 C 稱 level，在 Python 稱 preset（AEC）或 strength（NR）：**C level == Python preset**，值名與整數相同；欄位 1（C）用 level 名，欄位 2（Python）保留 `from_preset`、`set_preset`、`apply_strength` 等原名，所以兩欄同一件事名字不同。Python 檔內的註解仍可能寫舊的 C 名（`aec_config_from_preset`、`anr_config_for_mode`、`aec_res_set_preset`），以本條為準。

## 1. 用途

算法在 Python 端開發，C 要先與 Python 的「效果」一致，之後才能優化或改寫。要做到這件事，必須先知道「C 的哪個函式在 Python 對應哪個符號」。本表以**符號名**（不用行號，行號會漂移）提供函式級追溯，並且不寫進 C 檔。

本表回答四件事：

1. 某個 C 函式有沒有 Python 對應，是哪一個符號、對應到什麼程度（`mirror`／`partial`）。
2. 沒有對應的是哪些（`C-only`，以及只在 Python 的 `Python-only`），避免有人拿不存在的東西去對拍。
3. 這一列已知有哪些落差（S-xx／D-xx／T-xx，見第 8 節的編號對照）。
4. 這一列有沒有現成的對拍工具或測試，以及它到底是 C↔Py、C 對 C 還是 Python 對 Python。

範圍內：`pipelines/`（common、mono、`dfn/`、`ulcnet/`、`4ch/`（`mcab` 與其 sample helper）及各範例）及其 Python 參考，audiolib 的 doa/bf/VAD/common/utility 對外函式，audiolib（hpf、audio_resampler、audio_stft／window、fft_wrapper、wav_io，以及 fast_math、mem_align、simd_kernels、simd_kernel_nn 四個 header-only 檔），AINR（DeepFilterNet2、GTCRN、RNNoise-ERB）與 AIAEC（Align-ULCNet、DeepVQE-S、共用 mask helper）的 C 前後處理。

範圍外（不是漏列）：AEC 與 NR 函式庫本體（本表只列 pipelines 呼叫它們的接縫；它們的函式對照在各自 repo 的 `docs/c_python_function_map.md`，建立本表時 `lib/aec/docs/` 與 `lib/nr/docs/` 內已有）、`lib/aec`／`lib/nr` 的 Python、`pipelines/tools/`（離線評測與 bench 工具）、各處 `tests/` 與 audiolib 的 `test/`（測試程式）、audiolib doa/bf/VAD 內的 static 輔助函式、Align_CRUSE 與 CAGCRN（沒有 C pre/post 類別）。

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

2. 失敗訊息會指出該改哪裡：`defined but not in any row`（新增了 C 函式，補一列或併入既有的 C-only 列）、`no such function definition`（C 函式改名或刪除）、`symbol not found`（Python 符號改名或刪除）、`statistics block … is stale` 或 `the reading of the statistics is stale`（表改了，統計或「讀法」一行沒跟上）。
3. 統計區的數字與其下的「讀法」一行要跟著改。重算指令（印出可直接貼進第 4 節的列與「讀法」一行）：`python3 pipelines/tests/test_c_python_map.py --stats`。
4. 新落差編號先加進第 8 節，再在列內引用。
5. 不要把行號寫進表。要追溯行號時用符號名搜尋。
6. 找不到把握的對應，就用 `未能判斷` 並在備註寫原因，不要用 `partial` 湊。
7. 沒有 audiolib 檢出時（單獨取出的 Audio_ALG），針對 `audiolib/common/…` 的 C 檢查會略過，其餘照常；可用環境變數 `C_PY_MAP_AC` 指定檢出位置。

**證明守衛會失敗**（改動只做在暫存副本，不碰 repo）：

- 改掉表內一個 Python 符號：把文件複製到暫存目錄、改壞一個 `path.py:Symbol`，再 `C_PY_MAP_MD=<副本> python3 -m pytest -q -p no:cacheprovider pipelines/tests/test_c_python_map.py`，測試應該紅並指出列編號。
- 改掉「讀法」一行的一個數字：同樣用 `C_PY_MAP_MD=<副本>` 執行 `-k prose`，測試應該紅並印出應有的一行。
- 改掉 Python 原始碼的一個符號：複製被引用的 Python 檔到暫存的同構目錄、改名一個 def，再用 `C_PY_MAP_ROOT=<暫存根目錄> … -k python_symbol` 執行。
- 新增一個 C 函式或改名：複製 C 檔做同樣的事，應該出現 `defined but not in any row` 或 `no such function definition`。

## 4. 涵蓋率統計

計數單位是 **C 函式**（同一檔內 `#if` 變體的同名定義算一個）。`Python-only` 欄計的是 Python 符號數。數字由表內各列重算，守衛測試會比對統計表與「讀法」一行，手改會失敗。

<!-- map:stats:begin -->
| 範圍 | C 函式總數 | 已對應 | 其中 mirror | 其中 partial | C-only | 未能判斷 | Python-only 符號 | 已對應中的推論 |
|---|---|---|---|---|---|---|---|---|
| `PL` pipelines（common、mono、DFN、4ch、ULCNet 範例）與其 Python 參考 | 344 | 90 | 42 | 48 | 254 | 0 | 30 | 36 |
| `TP` audiolib (doa/bf/VAD/common/utility) | 65 | 0 | 0 | 0 | 65 | 0 | 0 | 0 |
| `AC` audiolib | 297 | 39 | 19 | 20 | 258 | 0 | 0 | 12 |
| `AINR` AINR（DFN2、GTCRN、RNNoise-ERB） | 158 | 54 | 28 | 26 | 104 | 0 | 2 | 20 |
| `AIAEC` AIAEC（Align-ULCNet、DeepVQE-S、共用 helper） | 106 | 34 | 18 | 16 | 72 | 0 | 0 | 15 |
| `TOTAL` 合計 | 970 | 217 | 107 | 110 | 753 | 0 | 32 | 83 |
<!-- map:stats:end -->

讀法：970 個 C 函式中，217 個有 Python 對應（mirror 107、partial 110），其中 83 個的配對是推論；753 個是 C-only；未能判斷 0 個。另有 32 個 Python 符號只在 Python。

「已對應」是**索引**，不是「已驗證一致」。真正量過數值的只有「對拍工具／測試」欄標（C↔Py）的項目，以及備註內註明為一次性量測的數字；其餘靠「已知落差」欄的編號。

## 5. 證據來源與限制

每一列的配對來自下列來源（備註裡的數字是一次性量測，量測驅動不在本 repo 內；本表沒有執行任何 DSP 計算，也沒有重量那些數字）：

1. **Python 檔內的明文**：docstring 寫「Python twin of `sk_min_f32`」「`df_common_expand_mask`」「`dfn2_model_io_push_features`」等直接點名 C 函式的句子。
2. **C 檔內的註解**：pipelines 的 C 註解（不含 `tests/`、`tools/`）指向 Python 的很少（`mcab` 的註解指向 `4ch/pipeline.py`）；AINR／AIAEC 的標頭與 .c 則大量點名 `train.py`、`model.py`、`export_onnx.py` 的符號。
3. **現有測試**：`pipelines/tests`、`pipelines/dfn/tests`、`pipelines/dfn/*/tests`、`pipelines/4ch/tests`、`pipelines/mono/aec_nr_res/tests`、`pipelines/common/tests`、`AINR/tests`、`AIAEC/tests` 內真正驅動 C 並對拍 Python 的測試，用的 C 函式與 Python 可呼叫物就是配對依據。
4. **一次性量測**：audiolib、AINR／AIAEC、pipelines 各自把 C 函式與 Python 可呼叫物配對後量測；這些驅動不在 repo 內，備註內以「一次性量測」標示，**不是常設測試**。
5. **閱讀函式名稱與角色**：備註以「（推論）」開頭，統計單獨計數。

限制：

- Python 目標解析在 Audio_ALG 內：`lib/aec/python`、`lib/nr/python` 是子模組，版本可能落後 AEC／NR 主線（lib/aec 的 Python 可能落後 AEC 主線），本表指向子模組裡實際存在的符號，因為守衛測試也只看這一份。
- 結構（struct、enum、巨集常數）不是函式，不在 C 函式表內；少數有 Python 對應的常數（如 `MCAB_SAMPLE_CNG_NR_GAIN_FLOOR`）寫在備註或 Python-only 列。
- 本表不判斷哪邊該改。

### 與 pyc_parity 對拍腳本的關係

對拍腳本量的是數值（逐情境的 max abs err、p99.9、是否逐位元相同），本表回答「拿 C 的哪個函式去對 Python 的哪個符號」。兩者分工如下：

- 本表是**索引**。每列有穩定編號；對拍腳本可以在情境名稱或輸出 JSON 內引用列編號，讓一個超出容差（或標 `EXPECTED_RED(S-xx)`）的情境能回頭查到 C 函式、Python 符號與已知落差。落差編號與本表「已知落差」欄是同一套。
- 本表不含容差、不含數值判定，也不要求任何對拍腳本存在；對拍腳本也不依賴本表才能執行。
- pipelines 的對拍腳本 `pipelines/tests/pyc_parity_pipelines.sh`（資料夾 `pipelines/tests/pyc_parity/`）有四組情境，對應本表的列：4ch NR 增益對應 PL-STAGE-02、4ch 後波束 RES 對應 PL-STAGE-05、4ch 共用 delay 的混合 realign 對應 PL-MCAB-07、mono 膠水（`mono_form` 與 `ac_istft_push` 合成）對應 PL-STAGE-07 與 AC-STFT-06；這些列的「對拍工具」欄引用該腳本。
- 守衛測試只保證「表指的東西還在」：Python 符號存在、C 函式存在、涵蓋完整、統計與表一致、引用的落差編號與工具路徑有效。它不檢查兩邊數值是否一致。

## 6. 涵蓋檔案

守衛測試逐檔重新列舉函式，與下表及第 7 節的列逐一比對。範圍欄：`all` 為檔內所有函式（含 static）；`extern` 只計非 static 函式（用於 audiolib 的 doa/bf/VAD，其 static 輔助函式不逐一列舉）。

<!-- map:coverage:begin -->
| 檔案 | 範圍 | 函式數 |
|---|---|---|
| `pipelines/dfn/dfn_hop.h` | all | 9 |
| `pipelines/dfn/dfn_hop_pool.h` | all | 3 |
| `pipelines/ulcnet/ulcnet_nn_stage.c` | all | 18 |
| `pipelines/common/example_cli.h` | all | 8 |
| `pipelines/mono/aec_nr_res/example_board_adapter.c` | all | 12 |
| `pipelines/mono/aec_nr_res/main.c` | all | 23 |
| `pipelines/dfn/dfn_example_args.h` | all | 10 |
| `pipelines/dfn/dfn_rate_bridge.c` | all | 32 |
| `pipelines/dfn/dfn_res_stage.c` | all | 26 |
| `pipelines/dfn/dfn_unit_partition.h` | all | 1 |
| `pipelines/mono/aec_dfn_res/main.c` | all | 9 |
| `pipelines/4ch/mcab.c` | all | 56 |
| `pipelines/4ch/mcab_internal.h` | all | 13 |
| `pipelines/4ch/mcab_sample_flow.c` | all | 6 |
| `pipelines/4ch/mcab_sample_beam.c` | all | 23 |
| `pipelines/4ch/mcab_sample_post.h` | all | 3 |
| `pipelines/4ch/mcab_external_static.c` | all | 15 |
| `pipelines/4ch/mcab_raw.c` | all | 17 |
| `pipelines/4ch/mcab_static.c` | all | 16 |
| `pipelines/4ch/mcab_sample_driver.c` | all | 7 |
| `pipelines/4ch/mcab_sample_driver.h` | all | 2 |
| `pipelines/4ch/aec_bf_dfn_res/main.c` | all | 14 |
| `pipelines/mono/alignulcnet/main.c` | all | 7 |
| `pipelines/4ch/alignulcnet/main.c` | all | 14 |
| `audiolib/bf/bf.c` | extern | 15 |
| `audiolib/VAD/masker.c` | extern | 6 |
| `audiolib/VAD/vad.c` | extern | 8 |
| `audiolib/VAD/vad_api.c` | extern | 8 |
| `audiolib/doa/doa_smoother.c` | extern | 1 |
| `audiolib/doa/srp.c` | extern | 10 |
| `audiolib/common/utility/spatial_complex.c` | extern | 4 |
| `audiolib/common/utility/spatial_simd.c` | extern | 13 |
| `audiolib/common/utility/fast_math.h` | all | 8 |
| `audiolib/common/utility/mem_align.h` | all | 6 |
| `audiolib/common/utility/simd_kernel_nn.h` | all | 36 |
| `audiolib/common/utility/simd_kernels.h` | all | 73 |
| `audiolib/common/utility/simd_aec.h` | all | 79 |
| `audiolib/common/utility/simd_nr.h` | all | 19 |
| `audiolib/common/include/wav_io.h` | all | 9 |
| `audiolib/common/src/audio_resampler.c` | all | 18 |
| `audiolib/common/src/audio_stft.c` | all | 29 |
| `audiolib/common/src/audio_window.c` | all | 1 |
| `audiolib/common/src/fft_wrapper_ne10.c` | all | 12 |
| `audiolib/common/src/hpf.c` | all | 8 |
| `lib/ainr/dfn2/dfn2_model_io.c` | all | 18 |
| `lib/ainr/dfn2/dfn2_model_io_priv.h` | all | 1 |
| `lib/ainr/dfn2/dfn2_prepost.c` | all | 39 |
| `lib/ainr/dfn2/dfn2_prepost.h` | all | 6 |
| `lib/ainr/dfn2/dfn2_process.c` | all | 28 |
| `lib/ainr/gtcrn/gtcrn_process.c` | all | 30 |
| `lib/ainr/rnnoise_erb/process.c` | all | 33 |
| `lib/ainr/rnnoise_erb/process.h` | all | 2 |
| `lib/aiaec/ulcnet/ulcnet_accelerator_adapter.c` | all | 9 |
| `lib/aiaec/ulcnet/ulcnet_model_io.c` | all | 27 |
| `lib/aiaec/ulcnet/ulcnet_model_io.h` | all | 1 |
| `lib/aiaec/ulcnet/ulcnet_prepost.c` | all | 24 |
| `lib/aiaec/ulcnet/ulcnet_process.c` | all | 1 |
| `lib/aiaec/deepvqe_s/deepvqe_prepost.c` | all | 35 |
| `lib/aiaec/deepvqe_s/deepvqe_prepost.h` | all | 1 |
| `lib/aiaec/deepvqe_s/deepvqe_process.c` | all | 4 |
| `lib/aiaec/cagcrn/cagcrn_process.c` | all | 1 |
| `lib/aiaec/align_cruse/align_cruse_process.c` | all | 1 |
| `lib/aiaec/ulcnet/ulcnet_mask_process.c` | all | 2 |
<!-- map:coverage:end -->

## 7. 對照表

<!-- map:rows:begin -->

### 7.1 pipelines

| ID | C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 | 備註 |
|---|---|---|---|---|---|---|
| PL-CFG-02 | `pipelines/mono/aec_nr_res/main.c: derive_lane` | `lib/aec/python/modules/config.py:AecConfig.from_preset`<br>`pipelines/mono/aec_nr_res/pipeline.py:_project_grid` | partial | S-30；D-38 | — | mono lane：grid 取自 lib/aec 的 `aec_resolve_signal_grid`（fft 0 用 `aec_default_fft_size`；C 另接受 8 kHz/256，Python 只接受 16 kHz 與 48 kHz），AecConfig 是 `aec_config_from_level`（對應 Python 的 `from_preset`：C level == Python preset）加明寫欄位（fft_size、MATCHED delay、enable_res=0、return_res_context=1）；其餘範圍檢查由 lib/aec 在 `aec_get_mem_size`／`aec_init` 做。Python 由 `_project_grid` 與 `AecConfig.from_preset` 分兩步取得。struct 值有 10 個欄位與 Python 有效值不同（C 以 10 ms 基準寫入、carve 時 retime），carve 後有效值屬 AEC 函式庫範圍。 |
| PL-CFG-04 | `pipelines/common/example_cli.h: example_cli_delay_options_default, example_cli_delay_mode_name, example_cli_is_option, example_cli_option_value, example_cli_parse_int, example_cli_parse_float, example_cli_delay_set_option, example_cli_delay_validate` | — | C-only | — | — | 範例程式（mono／4ch DFN2、mono／4ch ULCNet、4ch sample driver 的嚴格數值）共用的命令列數值與 delay 選項：數字一律整串嚴格解析（空字串、前置空白、`12x`、超出 int 或非有限值以 exit 2 拒絕），再檢查 mode、n 與 fixed-delay 的組合；非 MATCHED 模式把 n 設為 DA_NUM_FILTERS。選項的值是下一個 token；該 token 以 `--` 開頭時（`example_cli_is_option`）視為缺值，`example_cli_option_value` 回傳 NULL 並在 stderr 印出訊息，呼叫端以 exit 2 結束；單獨的 `-` 與負數是值。沒有 Python 版。 |
| PL-STAGE-02 | `pipelines/4ch/mcab_sample_post.h: mcab_sample_anr_gain`<br>`pipelines/mono/aec_nr_res/main.c: mono_nr_gain` | `pipelines/mono/aec_nr_res/pipeline.py:run_nr_spectrum` | mirror | — | `pipelines/tests/pyc_parity_pipelines.sh`（C↔Py） | C 逐 hop，Python 整段批次。R² 以 r2 / 2^30 的倒數乘法（`sk_scale_f32`，除數是 2 的冪，與 Python 的 `psd_scale = 32768.0 ** 2` 除法結果相同）在 usable_linear 時折入雜訊底。降噪器兩邊都是 NR 函式庫的強度組態（C 稱 level，範例鏈用 `anr_config_for_level_grid`；Python 稱 strength，其 `_build_denoiser` 用 v3_2_config.yaml 加 `apply_strength`），pipeline 不另覆寫欄位。mono 路徑把 C seam 餵給 Python 的 G_nr 差 8.3e-6（16k/256，標準數學建置）。 |
| PL-STAGE-05 | `pipelines/4ch/mcab_sample_post.h: mcab_sample_res_input` | `pipelines/4ch/pipeline.py:PostBeamResidualSuppressor.process` | partial | T-16 | `pipelines/tests/pyc_parity_pipelines.sh`（C↔Py） | 後波束 RES 的 lib/aec 輸入（`AecResInput`，指標借用 McabFrame）；4ch 範例的 `raw_chain`／`static_chain` 與 mcab_external_static.c 的 `post_hop` 填好後直接呼叫 `aec_res_process`；單元本身不含 RES。clock_drift 在 Python 寫死 False。 |
| PL-STAGE-07 | `pipelines/mono/aec_nr_res/main.c: mono_form, follow_post_reset` | `pipelines/mono/aec_nr_res/pipeline.py:run_res`<br>`pipelines/mono/aec_nr_res/pipeline.py:total_gain`<br>`pipelines/mono/aec_nr_res/pipeline.py:apply_gain`<br>`pipelines/mono/aec_nr_res/pipeline.py:comfort_noise_fill` | mirror | T-17 | `pipelines/tests/pyc_parity_pipelines.sh`（C↔Py） | 各範例 main 自己寫的逐 hop 後級（沒有共用的 chain 型別）：RES 增益取 `aec_get_res_context` 的 `ctx.res_gain`（`--no-res` 時為 unity）、NR 增益 `mono_nr_gain`（`--no-nr` 時為 unity）、`sk_min_f32`、`sk_capply_gain_f32`、comfort noise（`aec_cng_fill`，只在 `with_res` 且 AEC 設定允許時）；任一 stage 拒絕就回 NULL，由 `process_hop` 重置全部。Python 的 RES 增益同樣只來自 `ctx.res_gain`（`run_res`）。渲染的頻譜是 AEC 的渲染基底 B（`aec_get_output_base`／`ctx.output_base_spec`，未觸發輸出 guard 的 hop 上就是 E），NR 估計與增益都讀 B；`follow_post_reset` 對應 `run_res` 內對 `ctx.post_reset_serial` 的處理（變了就清 OLA 尾巴並把 comfort noise 種子設回 `AEC_CNG_SEED`，在該 hop 的第一次抽樣與合成之前；NR 不動）。comfort noise 兩邊都是 AEC 函式庫的同一個產生器（Python 呼叫 `modules/cng.py:cng_fill`），`--no-nr` 且 RES 開啟時整條等於 `aec_process()`（C、Python 各自對自己的第二個 `Aec` 逐位元相同）。合成（`ac_istft_push`）不在 `mono_form` 內，對應 `synthesize_hop`，見 AC-STFT-06；對拍腳本 `d.mono.*.min_apply_ola` 的註解記載 48k/1024 最大差 7.5e-8。其中 min 與 apply 的核心見 AC-SIMD-01。 |
| PL-MONO-08 | `pipelines/mono/aec_nr_res/main.c: main` | `pipelines/mono/aec_nr_res/pipeline.py:main` | partial | S-31；D-40 | — | 旗標集合不對稱：Python 沒有 moderate、--no-nr／--no-res／--no-cng；C 以位置參數給 aec-level。main 直接組合：`derive_lane` 導 lane、`Aec`（`aec_process_context`）→ `aec_get_res_context` → `mono_form`（RES 增益取 AEC 自己的 `ctx.res_gain`、NR 增益、min、apply、comfort noise，見 PL-STAGE-07）→ `ac_istft_push`；同一份 main.c 編出兩個 binary：aec_nr_pipeline_static 用一個呼叫端記憶體池，aec_nr_pipeline（-DAEC_NR_POOL_SPLIT=1）把 AEC 與後級分成兩塊。 |
| PL-MONO-09 | `pipelines/mono/aec_nr_res/main.c: parse_level, level_name, parse_nr_level, nr_level_name, is_nr_level_option, rig_free, post_plan, post_bytes, pool_bytes, rig_build, rig_reset, print_debug_status, stage_accum_add, stage_accum_row, print_stage_timing, timing_now_us, process_hop, print_mem_budget`<br>`pipelines/mono/aec_nr_res/example_board_adapter.c: board_mem_alloc, board_mem_free, mono_plan, mono_init, mono_hop, mono_reset, mono_destroy, lcg_sample, run_hops, run_one_cycle, run_negative_demonstrations, main` | — | C-only | — | — | CLI 解析（值的位置出現 `--` 開頭的 token 以 `example_cli_option_value` 判為缺值，exit 2；未知 level 名稱仍退回 balanced）、lane 與後級（NR、合成、增益緩衝）的組裝（`post_plan`、rig_* 與 pool 切分）、逐 hop 組合（process_hop，其中增益運算在 `mono_form`，見 PL-STAGE-07）、診斷列印、計時、板端轉接範例。 |
| PL-DFN-01 | `pipelines/dfn/dfn_res_stage.c: dfn_res_stage_process, dfn_res_stage_begin, dfn_res_stage_infer, dfn_res_stage_end` | `pipelines/dfn/dfn2_stage.py:Dfn2Stage.push` | mirror | D-42；S-17 | `pipelines/dfn/tests/test_dfn2_stage_c_parity.py`（C↔Py，驅動 dfn2_process.c）；`pipelines/dfn/tests/test_dfn2_stage.py`（Py 契約） | C 把一個 hop 拆成 begin／callback／end 三段；Python 的 `push` 在內部呼叫 heads。`dfn2_enhance_spectrum_stream` 路徑逐位元相同、特徵在 AINR 容差內（測試驅動的是 dfn2_process.c，不是 dfn_res_stage.c 本體）。 |
| PL-DFN-02 | `pipelines/dfn/dfn_res_stage.c: dfn_res_stage_reset` | `pipelines/dfn/dfn2_stage.py:Dfn2Stage.reset` | mirror | D-42 | — | 重置 DFN2 類別與計數並呼叫 `model.reset`；fail-open 後的狀態語意不同（D-42）。 |
| PL-DFN-03 | `pipelines/dfn/dfn_res_stage.c: dfn_res_stage_set_atten_lim` | `pipelines/dfn/dfn2_stage.py:Dfn2Stage.set_atten_lim` | mirror | — | — | — |
| PL-DFN-04 | `pipelines/dfn/dfn_res_stage.c: erb_inv_is_partition_of_unity` | `pipelines/dfn/dfn2_stage.py:erb_inv_is_partition_of_unity`<br>`AINR/DeepFilterNet2/export_erb_matrix.py:require_partition_of_unity` | mirror | S-21 | `AINR/DeepFilterNet2/tests/test_dfn2_contract.py`（Py 契約） | identity（fail-open）路徑的前提：單位 band mask 展開後每個 bin 恰為 1.0。 |
| PL-DFN-05 | `pipelines/dfn/dfn_res_stage.c: dfn_res_stage_output_frame_index` | `pipelines/dfn/dfn2_stage.py:StageOutput.frame_index` | partial | — | — | （推論）C 是存取函式，Python 把來源 frame 序號放在 `StageOutput` 欄位。 |
| PL-DFN-06 | `pipelines/dfn/dfn_res_stage.c: align_up, carve, fnv1a_u32, backend_id, model_valid, resolve, fill_descriptor, dfn_res_stage_default_config, dfn_res_stage_get_mem_requirements, descriptor_matches, dfn_res_stage_init_ex, dfn_res_stage_init, dfn_res_stage_create, dfn_res_stage_destroy, dfn_res_stage_get_counters, dfn_res_stage_grid, dfn_res_stage_state_generation, inputs_valid` | — | C-only | — | — | 描述子、記憶體池、counters；`dfn_res_stage_state_generation` 是模型遞迴狀態的重啟計數（stage 不註冊狀態，持有狀態的模型回呼在值改變時重載零）；`inputs_valid` 在 begin 前掃兩個輸入頻譜是否有限，Python 的 `Dfn2Stage.push` 只檢查 heads 輸出端（fail-open 的觸發條件），沒有輸入端掃描。 |
| PL-DFN-07 | `pipelines/dfn/dfn_rate_bridge.c: dfn_rate_bridge_process, dfn_rate_bridge_begin, dfn_rate_bridge_nn_begin, dfn_rate_bridge_nn_end, dfn_rate_bridge_finish, push_up, finish_frame` | `pipelines/dfn/dfn2_rate_adapter.py:Dfn2RateBridge.push` | mirror | D-24 | `pipelines/dfn/tests/test_dfn2_rate_bridge_c_parity.py`（C↔Py） | 兩邊用同一個 C resampler 核心，只剩 FFT 後端差；容差 float32 round-off，prefill／added delay 為相同整數。 |
| PL-DFN-08 | `pipelines/dfn/dfn_rate_bridge.c: calibrate_prefill` | `pipelines/dfn/dfn2_rate_adapter.py:Dfn2RateBridge._calibrate_prefill` | mirror | — | `pipelines/dfn/tests/test_dfn2_rate_bridge_c_parity.py`（C↔Py） | — |
| PL-DFN-09 | `pipelines/dfn/dfn_rate_bridge.c: dfn_rate_bridge_reset, clear_streams` | `pipelines/dfn/dfn2_rate_adapter.py:Dfn2RateBridge.reset` | mirror | — | — | — |
| PL-DFN-13 | `pipelines/dfn/dfn_rate_bridge.c: dfn_rate_bridge_reset_synthesis` | `pipelines/dfn/dfn2_rate_adapter.py:Dfn2RateBridge.reset_synthesis` | mirror | — | `pipelines/dfn/tests/test_dfn2_rate_bridge_c_parity.py`（C↔Py）；`pipelines/dfn/tests/test_dfn_rate_bridge.c`；`pipelines/mono/aec_dfn_res/tests/test_mono_dfn.c`；`pipelines/mono/aec_dfn_res/tests/test_dfn2_pipeline_contract.py` | AEC 硬重置事件（`aec_post_reset_serial` 動了）時只清橋內兩個原生格點合成物件（estimate／apply 兩路）的 overlap-add 尾巴；DFN2、兩個 resampler、48 kHz FIFO／分析歷史／合成、輸出 FIFO 與 prefill 都不動，added delay 不變。合成在 `begin()`／`push()` 開頭，前面沒有 lookahead，所以在該 hop 的 `begin()` 之前呼叫即可，不需要像 48 kHz 路徑那樣延遲 L 個 hop。hop 進行中（`begin()` 之後）C 回 -1 且不清。 |
| PL-DFN-10 | `pipelines/dfn/dfn_rate_bridge.c: grid_dims, resolve, dfn_rate_bridge_added_delay_samples, dfn_rate_bridge_added_delay_exact` | `pipelines/dfn/dfn2_rate_adapter.py:Dfn2RateBridge.__init__` | partial | — | `pipelines/dfn/tests/test_dfn2_rate_bridge_c_parity.py`（C↔Py） | （推論）grid 能力閘與 added delay（取整後 16k/128: 672、16k/256: 629、8k/128: 330；精確值為 `dfn_rate_bridge_added_delay_exact` 回傳的分子除以 6 個原生 sample，例如 pair 0/1 在 16k/128 為 501 1/3）；Python 在建構時計算 `added_delay` 與 `added_delay_exact` 並對拍。 |
| PL-DFN-11 | `pipelines/dfn/dfn_rate_bridge.c: align_up, fnv_u32, borrowed_bytes, fill_descriptor, dfn_rate_bridge_default_config, dfn_rate_bridge_get_mem_requirements, descriptor_matches, dfn_rate_bridge_init_ex, dfn_rate_bridge_init, dfn_rate_bridge_create, dfn_rate_bridge_destroy, dfn_rate_bridge_hop_size, dfn_rate_bridge_n_freqs, dfn_rate_bridge_stage, dfn_rate_bridge_get_counters, dfn_rate_bridge_get_host_hops, dfn_rate_bridge_state_generation` | — | C-only | — | — | 描述子、記憶體池、counters、存取函式（`dfn_rate_bridge_state_generation` 轉發 stage 的值）；`dfn_rate_bridge_get_host_hops` 是診斷用的 tap（原生格點的 E／P 合成 hop，Python 孿生以 `Dfn2RateBridge.host_hops` 屬性暴露）。 |
| PL-DFN-12 | `pipelines/mono/aec_dfn_res/main.c: rig_process, follow_post_reset`<br>`pipelines/dfn/dfn_hop.h: dfn_hop_begin, dfn_hop_nn_begin, dfn_hop_nn_end, dfn_hop_end, dfn_hop_process` | `pipelines/mono/aec_dfn_res/pipeline.py:run_dfn2_res` | mirror | D-43；T-15 | `pipelines/mono/aec_dfn_res/tests/test_dfn2_pipeline_contract.py`（Py 對 Py，identity 模型） | Python 是這條 C 組合的孿生（`mono/aec_dfn_res/pipeline.py` 檔頭明載）：`Aec` → `aec_get_res_context` → `rig_process` 內聯的 `sk_min_f32`（min(1, G_res)）、`sk_capply_gain_f32` 與 `aec_cng_fill`（RES 增益加 comfort noise，不跑 NR、不合成）→ DFN2 hop。identity 模型下 Python 吃 C seam 對 C 輸出最大差 1.2e-7（一次性量測）。Python 另有 C 沒有的實驗開關（--dfn-estimate-source、--dfn-compose）。 |
| PL-DFN-16 | `pipelines/mono/aec_dfn_res/main.c: run_accelerator, derive_lane, make_plan, rig_free, rig_build, rig_reset, main`<br>`pipelines/dfn/dfn_unit_partition.h: dfn_unit_partition`<br>`pipelines/dfn/dfn_example_args.h: dfn_example_rate_known, dfn_example_rates_text, dfn_example_model_rows_text, dfn_example_model_row, dfn_example_usage, dfn_example_resolve_model_grid, dfn_example_parse_args, dfn_example_erb_free, dfn_example_read_matrix, dfn_example_load_matrices` | — | C-only | T-13 | — | 板端範例的 CLI、參數解析、記憶體計畫與 ERB 矩陣來源：`--erb-fwd`、`--erb-inv` 兩個無標頭的 float32 原始檔（各剛好 n_bins x n_erb 個值，由 `dfn_example_read_matrix` 檢查，開不了或個數不符回傳失敗）讀進一塊執行期配置的對齊記憶體，函式庫只借用其中的兩個指標（`dfn_example_load_matrices`；`dfn_example_erb_free` 在借用它們的物件銷毀之後才釋放）；兩個都沒給時用單位 ERB 矩陣（`dfn_unit_partition`，只做 fail-open 冒煙測試）。`run_accelerator` 是替換成板端 runtime 的樁。模型 grid 的 spec 取自檔內寫死的 `DFN_EXAMPLE_MODEL_ROWS`（`dfn_example_model_row`、`dfn_example_resolve_model_grid`），再交給 `dfn2_grid_make`。 |
| PL-DFN-17 | `pipelines/dfn/dfn_hop.h: dfn_hop_init_stage, dfn_hop_init_bridge, dfn_hop_reset, dfn_hop_stage` | — | C-only | — | — | `DfnHop` 的接線（48 kHz 走 stage 加呼叫端的 `AcIstft`，其餘走 rate bridge）、重置與存取；非擁有指標，沒有 Python 對應。 |
| PL-DFN-18 | `pipelines/dfn/dfn_hop_pool.h: dfn_hop_pool_plan, dfn_hop_pool_bytes, dfn_hop_pool_build` | — | C-only | — | — | 兩個 DFN2 範例 main 共用的 pool 尾段：48 kHz 為 stage、48 kHz 合成與 stage 輸出頻譜，其他取樣率為 rate bridge；依序切出並接上 `DfnHop`（4ch 版開 stage 路徑的 finite 檢查）。沒有 Python 版。 |
| PL-MCAB-01 | `pipelines/4ch/mcab.c: update_shared_delay` | `pipelines/4ch/pipeline.py:SharedMatchedDelayEstimator.accumulate` | mirror | D-39 | `pipelines/4ch/tests/test_delay_parity.py`（C↔Py）；`pipelines/4ch/tests/dump_delay_parity.c`（C 側驅動） | delay_samples／solid／changed 逐 hop 相同（16k/256 1250/1250、16k/512 625/625、48k 937/937）；matched-filter 的 duty-cycle 排程、FIXED 與 EXTERNAL_ALIGNED 模式是 C-only（對拍時 pin 全速）。共用 estimator（`DelayAec`）由應用程式建立後注入。 |
| PL-MCAB-02 | `pipelines/4ch/mcab.c: mcab_admission_age, mcab_admission_offer, mcab_pending_delay_candidate` | `pipelines/4ch/pipeline.py:SharedMatchedDelayEstimator.accumulate` | partial | D-39 | `pipelines/4ch/tests/test_delay_parity.py`（C↔Py）；`pipelines/4ch/tests/dump_delay_parity.c`（C 側驅動） | Path-B 三條件（變動 >32 樣本、第二次估計在 16 內、候選壽命 3 hop）；C 拆成可由測試直接呼叫的函式，Python 內聯在 `accumulate`。兩邊常數相同。 |
| PL-MCAB-03 | `pipelines/4ch/mcab.c: align_render` | `pipelines/4ch/pipeline.py:_SharedReferenceDelayLine.process` | mirror | — | `pipelines/4ch/tests/test_delay_parity.py`（C↔Py）；`pipelines/4ch/tests/dump_delay_parity.c`（C 側驅動） | 整個 hop 以同一個謂詞決定走對齊環或原始 render；aligned far 逐位元相同（0/160000 樣本差）。 |
| PL-MCAB-04 | `pipelines/4ch/mcab.c: mcab_process_front` | `pipelines/4ch/pipeline.py:McabPipeline.process_front` | mirror | D-39；T-15 | — | 共用 delay、對齊 render、四條線性 AEC lane（lane 的參數由應用程式組成，以 `mcab_apply_lane_contract()` 蓋上 flow 的契約欄位後建立並注入）。 |
| PL-MCAB-05 | `pipelines/4ch/mcab.c: fuse_contexts` | `pipelines/4ch/pipeline.py:_fuse_contexts` | partial | D-41；T-16 | — | 相干投影 E／Y／回音，R² 與 comfort noise 加權。R² 相量的相位下限只在 C（0<\|echo\|²≤1e-12 時釘在實軸）。C 另檢查投影結果全為有限值（有限但極大的權重會讓平方溢位），不是就拒絕；Python 不檢查。 |
| PL-MCAB-06 | `pipelines/4ch/mcab.c: mcab_fuse` | `pipelines/4ch/pipeline.py:McabPipeline.fuse` | partial | T-16 | — | fuse 只做投影，結果是 mcab.h 的 McabFrame，不算 RES 增益（`g_res_ctx` 為 NULL）；Python 的 `fuse` 同樣只做投影（`context.res_gain` 為 None）。後波束 RES 由呼叫端建：Python 用自己的 `PostBeamResidualSuppressor`（吃 `McabFrame.aligned_render`），C 範例用 lib/aec 的 `AecRes`。波束器權重一律來自呼叫端（sample helper 的形式見 PL-MCAB-26）。C 允許同一個 front 再 fuse 一次，Python 的 `fuse` 每個 front 只接受一次（第二次報 out of order）；C 在投影溢位時回 `MCAB_DSP_ERROR` 並重置 flow。Python 的 McabFrame 帶投影後的 context 與 `aligned_render`，不是 C McabFrame 的逐欄對應。 |
| PL-MCAB-07 | `pipelines/4ch/mcab.c: mcab_far_spec_agrees, complex_close` | `pipelines/4ch/pipeline.py:_fuse_contexts` | partial | — | `pipelines/tests/pyc_parity_pipelines.sh`（C↔Py） | 「四 lane 必須共用同一個 far 頻譜」的前提檢查。C 在混合 warm／soft realign 的 hop 以 `far_spec_mixed_hop` 旗標略過這項檢查；Python 的 `_fuse_contexts` 不做這項檢查，兩邊的融合 X 都取 lane 0 的 far_spec。 |
| PL-MCAB-08 | `pipelines/4ch/mcab_internal.h: mcab_weights_valid, mcab_weights_valid_scalar` | `pipelines/4ch/pipeline.py:_fuse_contexts` | partial | — | — | （推論）波束權重有限性與非零檢查（`mcab_fuse` 直接呼叫）；Python 在 `_fuse_contexts` 內以 np.isfinite 與 weight_sum 判斷。 |
| PL-MCAB-09 | `pipelines/4ch/mcab_internal.h: mcab_projection_cmac, mcab_projection_cmac_scalar` | `pipelines/4ch/pipeline.py:_fuse_contexts` | mirror | — | — | （推論）sum(weights * values) 的複數乘加核心。 |
| PL-MCAB-10 | `pipelines/4ch/mcab_internal.h: mcab_residual_sample, mcab_residual_vector, mcab_residual_vector_scalar, mcab_complex_mag2, mcab_complex_mag2_scalar` | `pipelines/4ch/pipeline.py:_fuse_contexts` | partial | D-41 | — | \|sum(w * sqrt(r2) * exp(j*phase))\|² 的 R² 相量融合。0<\|echo\|²≤1e-12 時相位下限 C 與 Python 不同（C 釘 +實軸，Python 用 np.angle）；正好為 0 時兩邊都是 +實軸。 |
| PL-MCAB-11 | `pipelines/4ch/mcab_internal.h: mcab_comfort_accumulate, mcab_comfort_accumulate_scalar` | `pipelines/4ch/pipeline.py:_fuse_contexts` | mirror | — | — | （推論）sum(\|w\|² * comfort_noise) 的 comfort noise 融合。 |
| PL-MCAB-12 | `pipelines/4ch/mcab.c: mcab_reset` | `pipelines/4ch/pipeline.py:McabPipeline.reset`<br>`pipelines/4ch/pipeline.py:SharedMatchedDelayEstimator.reset` | mirror | — | — | 所有 lane 與共用 delay 估計器共用同一個重置時期；C 透過各 lib 自己的 `aec_reset`／`delay_aec_reset` 重置被注入的物件。 |
| PL-MCAB-13 | `pipelines/4ch/mcab.c: flow_grid, delay_request_valid, derive_flow, mcab_default_config, mcab_delay_estimator_config` | `pipelines/4ch/pipeline.py:McabConfig.resolved_grid` | partial | S-32；D-39 | — | （推論）flow 的參數：grid 白名單（16k/256、16k/512、48k/1024）、delay 請求規則與預設值，以及共用 estimator 的設定；Python 的設定面比 C 少（無 delay_mode、delay_num_filters、fixed_delay_samples、delay_est_period_s、enable_post）。lane 的參數不在 flow 設定內（PL-MCAB-32）。 |
| PL-MCAB-32 | `pipelines/4ch/mcab.c: mcab_apply_lane_contract, mcab_lane_config` | `pipelines/4ch/pipeline.py:mcab_apply_lane_contract`<br>`pipelines/4ch/pipeline.py:mcab_lane_config` | partial | — | — | lane 的參數（level、filter length 等）由應用程式決定，flow 只蓋上契約欄位且不動其他欄位、可重複套用。C 的契約：fft_size、EXTERNAL_ALIGNED（delay_num_filters=DA_NUM_FILTERS、fixed_delay_samples=-1、enable_delay_est=0）、enable_res=0、return_res_context=1、spatial context（enable_post 決定）、delay_est_period_s；Python 的契約：frame／hop、enable_delay_est=False、fixed_delay_samples=-1、enable_res=False、return_res_context=True（Python 沒有 spatial context 欄位）。預設 lane：C 為 balanced level，Python 為 PBFDKF、52／64 ms filter。兩邊都拒絕其他取樣率的 lane。 |
| PL-MCAB-33 | `pipelines/4ch/mcab.c: project_render_base, lane_contributes` | `pipelines/4ch/pipeline.py:_fuse_contexts` | mirror | — | `pipelines/4ch/tests/test_mcab.c`（C，double 參考）；`tests/test_pipeline.py::test_rendering_base_is_the_weighted_lane_bases`（Python） | 渲染基底 B：沒有任何有權重的 lane 觸發輸出 guard 時 B 就是分析波束 E 本身（C 同一指標；Python 同一陣列），否則為 sum(w * 各 lane 的 output_base_spec)；`guard_lanes`／`guard_active` 是有權重且觸發的 lane。`post_reset_serial` 兩邊都是四條 lane 計數的和取模 2^32。C 的 E 可能是呼叫端的波束（`trusted_beamformed_error`），Python 一律由權重重建。 |
| PL-MCAB-34 | `pipelines/4ch/mcab_sample_post.h: mcab_sample_post_reset_due` | — | C-only | — | `pipelines/4ch/tests/test_mcab.c`（full-mode twin，含忽略 serial 的突變） | 範例層：融合輸出的合成者記住已處理的 `post_reset_serial`，變了就在該 hop 的 comfort-noise 抽樣與 iFFT 前清 OLA 尾巴並把種子設回 `AEC_CNG_SEED`。Python 端由使用融合 context 的 mono 後級函式依 `post_reset_serial` 處理。 |
| PL-MCAB-15 | `pipelines/4ch/mcab.c: bind_lane_view` | `pipelines/4ch/pipeline.py:_snapshot_context` | partial | — | — | （推論）所有權語意不同：C 直接把指標指向各 lane 的 buffer（零拷貝），Python 的 `_snapshot_context` 複製一份。 |
| PL-MCAB-16 | `pipelines/4ch/mcab.c: mcab_matched_filter_count` | `pipelines/4ch/pipeline.py:McabPipeline.matched_filter_instance_count`<br>`pipelines/4ch/pipeline.py:SharedMatchedDelayEstimator.instance_count` | partial | — | — | （推論）結構稽核掛鉤：MATCHED 時整條管線只有一個 matched filter；MATCHED 以外 C 回 0，Python 恆為 1（Python 只有 MATCHED）。 |
| PL-MCAB-17 | `pipelines/4ch/mcab.c: mcab_linear_aec_count` | `pipelines/4ch/pipeline.py:McabPipeline.linear_filter_instance_count` | mirror | — | — | （推論）結構稽核掛鉤：四條線性 AEC lane。 |
| PL-MCAB-18 | `pipelines/4ch/mcab.c: mcab_realign_warm_lane_count` | `pipelines/4ch/pipeline.py:McabPipeline.realign_warm_lane_count` | mirror | — | — | — |
| PL-MCAB-19 | `pipelines/4ch/mcab.c: mcab_realign_soft_lane_count` | `pipelines/4ch/pipeline.py:McabPipeline.realign_soft_lane_count` | mirror | — | — | — |
| PL-MCAB-20 | `pipelines/4ch/mcab.c: mcab_hop_size, mcab_fft_size, mcab_n_freqs, mcab_sample_rate, mcab_enable_post` | `pipelines/4ch/pipeline.py:McabPipeline.__init__` | partial | — | — | （推論）Python 在建構時把 grid 存成屬性，沒有獨立的存取方法。 |
| PL-MCAB-21 | `pipelines/4ch/mcab.c: mcab_backend_id, mcab_now_us, pool_carve, flow_buffer_size, carve_buffers, fnv1a_str, mcab_build_flags_hash, mcab_get_mem_breakdown, mcab_get_mem_requirements, lane_matches, libs_match, mcab_init_ex, mcab_init, mcab_create, max_lane_erle_windowed, complex_vector_finite, mcab_destroy, mcab_far_fft_real_compute_count, mcab_duty_hops_total, mcab_duty_hops_run, mcab_far_spec_provenance, mcab_pin_duty_full_rate, mcab_duty_leak_db_for_grid, mcab_duty_erle_leak_db, mcab_duty_erle_peak, mcab_get_last_timing`<br>`pipelines/4ch/mcab_internal.h: mcab_count_add_sat, mcab_far_fft_delta_matches` | — | C-only | D-39 | — | flow 自己的記憶體池、carve、build-flags hash、被注入物件的驗證（lane 的 grid／context／對齊方式、estimator 的 bank 與 48 kHz scratch）、duty-cycle 排程（C-only）、far-spec provenance 與 duty 診斷、計時。 |
| PL-MCAB-22 | `pipelines/4ch/mcab_sample_beam.c: mcab_sample_beam_process, mcab_sample_beam_process_with_activity` | `pipelines/4ch/pipeline.py:McabPipeline.process` | partial | D-43；S-32 | — | （推論）sample helper：C 含 SRP-PHAT 與波束器（bf）（沒有 Python 版），輸出是 BF 後頻譜與權重，不含 fuse、RES、NR、合成；Python 的波束器是由外部供應的接縫（EqualWeightBeamformer 只供測試）。 |
| PL-MCAB-23 | `pipelines/4ch/mcab_sample_beam.c: mcab_sample_beam_default_params, flag_valid, params_valid, fill_uca_geometry, fill_ula_geometry, mcab_sample_spatial_config, helper_layout, mcab_sample_beam_get_mem_size, take, mcab_sample_beam_init, mcab_sample_beam_destroy, fill_info, mcab_sample_beam_process_beam, floats_finite, auto_vad, mcab_sample_beam_bf_effective_adapt_interval, mcab_sample_beam_bf_lambda, mcab_sample_beam_spatial_backend` | — | C-only | D-43 | — | sample helper（不在模組 archive）：參數驗證、SRP／bf 陣列幾何與設定導出、helper 自己的池、VAD 自動判斷、DOA／bf 存取函式；沒有 Python 版。 |
| PL-MCAB-24 | `pipelines/4ch/mcab_sample_beam.c: mcab_sample_beam_reset` | `pipelines/4ch/pipeline.py:McabPipeline.reset` | partial | — | — | （推論）C 重置 flow、SRP、bf 與備援 VAD；Python 的 `reset` 重置共用 delay 與四條 lane。RES、NR 與合成不在 helper 內，由呼叫端一起重置。 |
| PL-MCAB-25 | `pipelines/4ch/mcab_sample_beam.c: mcab_sample_beam_process_front` | `pipelines/4ch/pipeline.py:McabPipeline.process_front` | partial | D-39 | — | （推論）呼叫 `mcab_process_front`（PL-MCAB-04）；C 另管 helper 的 phase 守衛（`process_beam` 要有待處理的 front）。 |
| PL-MCAB-26 | `pipelines/4ch/mcab_sample_beam.c: mcab_sample_beam_fuse` | `pipelines/4ch/pipeline.py:McabPipeline.fuse` | partial | T-16 | — | （推論）helper 用自己的權重與 BF 後頻譜呼叫 `mcab_fuse`（PL-MCAB-06）填 McabFrame（`g_res_ctx` 為 NULL，RES 由呼叫端算）；C 在 fuse 失敗時重置 flow、SRP 與 BF，Python 的 `fuse` 由呼叫端傳入 McabFront 與外部 BF 結果。 |
| PL-MCAB-27 | `pipelines/4ch/mcab_sample_flow.c: flow_layout, mcab_sample_flow_get_mem_size, take, mcab_sample_flow_init, mcab_sample_flow_create, mcab_sample_flow_destroy` | — | C-only | — | — | sample 層的應用端 init 順序（四條 lane、共用 estimator、mcab 自己的池放在同一塊池內）；DFN2／ULCNet 範例與測試用它，raw／static 範例自己寫出同樣的順序；沒有 Python 版。 |
| PL-MCAB-28 | `pipelines/4ch/mcab_sample_driver.c: mcab_sample_backend_name, mcab_sample_print_mem_budget, mcab_sample_scan_cli, mcab_sample_spatial_set, mcab_sample_spatial_unknown, mcab_sample_parse_spatial_cli, mcab_sample_main`<br>`pipelines/4ch/mcab_sample_driver.h: mcab_sample_resolved_fft, mcab_sample_unit_memory` | — | C-only | — | — | 四個 4ch sample 共用的 CLI 掃描器、空間選項政策、記憶體診斷與 setup/read/process/finite-check/consume/finish/cleanup 骨架。raw、static、DFN2、ULCNet 各自保留型別化設定與模型資源 adapter；只用於 sample，不進演算法 archive。數值選項一律用 `common/example_cli.h` 的 `example_cli_parse_int()`／`example_cli_parse_float()` 嚴格解析（`raw` 只決定選項集合與解析時的 grid、半徑、檔案檢查）。`ctest:pipelines/4ch/tests/test_mcab_sample_driver.c` 驗證錯誤短路、清理次數、有限值政策與 CLI 邊界（兩種 `raw` 下的壞數字與缺值）；沒有 Python 版。 |
| PL-MCAB-29 | `pipelines/4ch/mcab_external_static.c: parse_level, level_name, nr_level_name, backend_id_name, flow_plan, flow_destroy, flow_init, post_plan, post_init, post_destroy, print_mem_budget, fill_hop, int_option, main`<br>`pipelines/4ch/mcab_raw.c: usage, json_float, unit_plan, unit_destroy, unit_init, raw_parse, raw_setup, raw_read, raw_process, raw_consume, raw_finish, raw_cleanup, main`<br>`pipelines/4ch/mcab_static.c: fill_hop, unit_plan, unit_destroy, unit_init, static_parse, static_setup, static_read, static_process, static_consume, static_finish, static_cleanup, main` | — | C-only | D-43 | — | 範例：應用端 init 順序（`flow_plan`／`flow_init`、`unit_plan`／`unit_init`：`mcab_lane_config` + `aec_init` ×4、`DelayAec`、`mcab_init_ex`，raw／static 再加 sample helper）、拆除順序、CLI（數值選項嚴格解析，值的位置出現 `--` 開頭的 token 視為缺值，未知 `--nr-level`（或別名 `--nr-preset`）名稱、未知 AEC level 與第二個位置參數皆 exit 2；`tests/test_cli_strict.sh`）、合成輸入與記憶體診斷；沒有 Python 版。 |
| PL-MCAB-30 | `pipelines/4ch/mcab_raw.c: raw_post, raw_chain`<br>`pipelines/4ch/mcab_static.c: static_hop, static_chain`<br>`pipelines/4ch/mcab_external_static.c: post_hop` | `pipelines/4ch/aec_bf_dfn_res/pipeline.py:post_conventional` | partial | — | — | （推論）範例層的後波束鏈：raw／static 用 `mcab_sample_beam_fuse`（不含 RES），再由範例自己的 `raw_chain`／`static_chain` 以 `mcab_sample_res_input` 與 `aec_res_process` 算 RES 增益、做 NR 增益、`sk_min_f32`、`sk_capply_gain_f32`、comfort noise 與 `ac_istft_push`；`raw_post`／`static_hop` 在任一步失敗時重置 flow、helper 與後級（含範例自己的 AecRes）。mcab_external_static.c 用自己的固定權重呼叫 `mcab_fuse`，`post_hop` 做同樣的鏈。Python 靠 `run_nr_spectrum`＋`run_res` 組成。 |
| PL-MCAB-31 | `pipelines/4ch/mcab_raw.c: post_plan, raw_post_reset`<br>`pipelines/4ch/mcab_static.c: post_plan, static_post_reset` | — | C-only | — | — | `post_plan` 算範例後級（RES、NR、合成、兩個增益向量與輸出頻譜）的池大小；`raw_post_reset`／`static_post_reset` 重置 RES、NR、合成與 comfort noise 種子。 |
| PL-4DFN-06 | `pipelines/4ch/aec_bf_dfn_res/main.c: run_accelerator, dfn_unknown, dfn_set, dfn_parse, dfn_plan, dfn_reset, dfn_cleanup, dfn_build, dfn_setup, dfn_read, dfn_complete, main` | — | C-only | — | — | 板端範例：CLI、記憶體計畫（單元、P 緩衝、stage 或 bridge 各自的池）、組裝、重置（單元、comfort noise 種子、DfnHop 一起）、清理。 |
| PL-4DFN-07 | `pipelines/4ch/aec_bf_dfn_res/main.c: dfn_process, dfn_form` | `pipelines/4ch/aec_bf_dfn_res/pipeline.py:run_front_end`<br>`pipelines/4ch/aec_bf_dfn_res/pipeline.py:post_dfn2` | partial | D-43 | `pipelines/4ch/aec_bf_dfn_res/tests/test_dfn2_4ch_contract.py`（Py 對 Py，identity 模型） | （推論）逐 hop：`mcab_sample_beam_process`（共用 delay、四條線性 AEC、SRP-PHAT、BF）→ `mcab_sample_beam_fuse`（不含 RES；範例自己的 AecRes 算 G_res）→ `dfn_form`（P = E·min(1, G_res)，`aec_cng_fill`，無 NR）→ `dfn_hop_process`。Python 的 E 是 `EqualWeightBeamformer` 權重的相干投影，後波束 RES 由 `McabPipeline` 持有（`run_front_end` 呼叫它的 `process`）；權重一致時是同一個訊號。契約測試是 Python 對 Python（identity 模型），不是 C 對拍。 |
| PL-ULC-01 | `pipelines/ulcnet/ulcnet_nn_stage.c: framing_config, borrowed_bytes, spectra_bytes, ulcnet_nn_stage_mem_size, ulcnet_nn_stage_init, ulcnet_nn_stage_grid, ulcnet_nn_stage_set_model, ulcnet_nn_stage_reset, gate_delay, start_frame, begin_check, ulcnet_nn_stage_begin, ulcnet_nn_stage_begin_complex, ulcnet_nn_stage_begin_time, ulcnet_nn_stage_nn_begin, ulcnet_nn_stage_nn_end, ulcnet_nn_stage_end, ulcnet_nn_stage_process`<br>`pipelines/mono/alignulcnet/main.c: usage, grid_row, parse_options, run_accelerator, derive_lane, process_hop, main`<br>`pipelines/4ch/alignulcnet/main.c: usage, grid_row, run_accelerator, ulcnet_example_unit_config, ulc_unknown, ulc_set, ulc_parse, ulc_setup, ulc_read, ulc_hop, ulc_process, ulc_complete, ulc_cleanup, main` | — | C-only | D-43；D-44；T-14 | — | ULCNet 後濾波 stage（遠端 STFT、模型呼叫政策、WOLA）與兩個範例 main（mono：`Aec` 加 stage；4ch：單元加 stage）沒有 Python 版，也沒有 pipeline 層的 C 對 Python 對拍。stage 的 grid 是應用端給的 `UlcnetGrid`（兩個 main 以 `--sample-rate` 16000／48000 經 `grid_row` 選列）。模型側前後處理見 AIAEC 區。`gate_delay` 的對齊事件政策（模型 reset＋1 個恆等框）Python 推論沒有（D-44）；推論對拍只比沒有對齊事件的區段，區段定義見 `AIAEC/Align_ULCNet/README.md` 的 C/Python parity 一節。 |
| PL-PYO-01 | — | `pipelines/mono/aec_nr_res/pipeline.py:run_aec_classic`<br>`pipelines/mono/aec_nr_res/pipeline.py:parse_aec_mode` | Python-only | — | — | 經典模式（AEC 內建 RES）與 AEC 模式字串解析；兩條 C 管線都只用線性模式加外部 RES，沒有對應。 |
| PL-PYO-02 | — | `pipelines/mono/aec_nr_res/pipeline.py:run_nr` | Python-only | — | — | 時域 NR 路徑（被 `run_nr_spectrum` 取代的 `--pipeline-mode linear`）；C 沒有。 |
| PL-PYO-03 | — | `pipelines/mono/aec_nr_res/pipeline.py:comfort_noise_fill` | Python-only | — | — | lib/aec `aec_cng_fill` 的逐位元孿生（呼叫 AEC 函式庫自己的 `modules/cng.py:cng_fill`，種子 `AEC_CNG_SEED`；回傳新的 rng 狀態）；`pyc_parity` 列 `d.mono.cng_pipeline` 比對 C 為 400 hop 逐位元相同。 |
| PL-PYO-04 | — | `pipelines/common/cng_aec3.py:Aec3ComfortNoise.fill`<br>`pipelines/common/cng_aec3.py:Aec3ComfortNoise.draw`<br>`pipelines/common/cng_aec3.py:Aec3ComfortNoise.reset`<br>`pipelines/common/cng_aec3.py:lcg_step` | Python-only | — | `pipelines/common/tests/test_cng_aec3.py`（文字與行為比對） | lib/aec `aec_cng_fill` 的逐位元孿生（`fill()` 與 `draw()` 共用 AEC 函式庫 `modules/cng.py` 的數值核心；`pyc_parity` 列 `d.mono.cng_twin` 18/18 組逐位元相同）；接在 `mono/aec_dfn_res/pipeline.py`，種子 `AEC_CNG_SEED`（42），硬重置事件（`post_reset_serial` 改變）時 `reset()`。C 函式屬 AEC 函式庫範圍。 |
| PL-PYO-05 | — | `pipelines/mono/aec_nr_res/pipeline.py:CNG_NR_GAIN_FLOOR`<br>`pipelines/mono/aec_nr_res/pipeline.py:PROD_INJECT_ECHO_PSD` | Python-only | — | `pipelines/common/tests/test_cng_floor_constant.py`（原始碼文字比對） | 前者對應各 C 程式本地的 `*CNG_NR_GAIN_FLOOR` 巨集（4ch 為 mcab_sample_post.h 的 MCAB_SAMPLE_CNG_NR_GAIN_FLOOR），所有定義與兩個 C 見證以文字比對釘死；後者是 R² 折入 NR 的生產開關，C 的對應是 cfg.legacy_amin 的反向。 |
| PL-PYO-07 | — | `pipelines/dfn/dfn2_stage.py:load_dfn2`<br>`pipelines/dfn/dfn2_stage.py:Dfn2Assets`<br>`pipelines/dfn/dfn2_stage.py:IdentityHeads`<br>`pipelines/dfn/dfn2_stage.py:Dfn2Heads`<br>`pipelines/dfn/dfn2_stage.py:dfn2_modules`<br>`pipelines/dfn/dfn2_stage.py:Dfn2Stage.flush` | Python-only | D-42 | — | 模型載入與 heads 可呼叫物件（C 由加速器 callback 取代）、尾端 flush（C 由呼叫端餵零框）。 |
| PL-PYO-08 | — | `pipelines/dfn/dfn2_rate_adapter.py:CResampler`<br>`pipelines/dfn/dfn2_rate_adapter.py:round_trip`<br>`pipelines/dfn/dfn2_rate_adapter.py:_root_hann` | Python-only | — | — | CResampler 是 ctypes 對 C resampler 的綁定（實驗與對拍用）；round_trip 與 _root_hann 是 Python 實驗輔助。 |
| PL-PYO-09 | — | `pipelines/mono/aec_dfn_res/pipeline.py:MaskOnlyHeads`<br>`pipelines/mono/aec_dfn_res/pipeline.py:make_stage`<br>`pipelines/mono/aec_dfn_res/pipeline.py:main` | Python-only | D-43 | — | Python 專屬實驗開關（--dfn-estimate-source、--dfn-compose）與 CLI；C 管線只出貨 pre_res／full。 |
| PL-PYO-10 | — | `pipelines/4ch/pipeline.py:Beamformer`<br>`pipelines/4ch/pipeline.py:McabBeam`<br>`pipelines/4ch/pipeline.py:EqualWeightBeamformer`<br>`pipelines/4ch/pipeline.py:McabPipeline.process_signal`<br>`pipelines/4ch/pipeline.py:McabPipeline.beamformer_configured` | Python-only | D-43 | `pipelines/4ch/tests/test_pipeline.py`（Py 契約） | Python 參考刻意不含 SRP-PHAT／GSC，波束器是外部接縫；EqualWeightBeamformer 只供測試與離線評測。C 的對應是 audiolib 的 bf。 |
| PL-PYO-11 | — | `pipelines/4ch/pipeline.py:McabDelayState`<br>`pipelines/4ch/pipeline.py:McabFront`<br>`pipelines/4ch/pipeline.py:McabFrame` | Python-only | — | `pipelines/4ch/tests/test_pipeline.py`（Py 契約） | C 結構 McabDelayState、McabFront、McabFrame 的 Python 資料類別（`McabFront` 另帶 `aligned_render`，供 `PostBeamResidualSuppressor` 作 render 輸入）；結構不是函式，所以不在 C 函式表內。 |

### 7.2 audiolib（doa、bf、VAD、utility）

| ID | C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 | 備註 |
|---|---|---|---|---|---|---|
| TP-GSC-01 | `audiolib/bf/bf.c: bf_effective_adapt_interval, bf_get_mem_size, bf_init, bf_create, bf_process_with_weights, bf_test_process_with_weights_scalar_rls, bf_process, bf_reset, bf_destroy, bf_get_doa_used, bf_get_adaptive, bf_get_bin_resets, bf_p_diag_floor, bf_p_diag_ceil, bf_wa_leak_factor` | — | C-only | D-43 | — | 第三方 C 來源（`audiolib/bf`），沒有 Python 版（設計如此）。BF 有兩項非逐位元的數值強化（`wa_leak` 衰減、P-diagonal 上下限鉗制），記載於 `bf.h`/`bf.c` 的註解。 |
| TP-VAD-01 | `audiolib/VAD/vad.c: vad_get_mem_size, vad_config_is_valid, vad_init, vad_create, vad_destroy, vad_step, vad_get_raw, vad_get_out`<br>`audiolib/VAD/vad_api.c: vad_api_get_mem_size, vad_api_init, vad_api_create, vad_api_process, vad_api_get, vad_api_get_raw, vad_api_get_mask, vad_api_destroy`<br>`audiolib/VAD/masker.c: masker_get_mem_size, masker_create, masker_init, masker_destroy, masker_step, masker_get_mask` | — | C-only | D-43 | — | 第三方 C 來源，repo 內沒有 Python 版（設計如此）。 |
| TP-DOA-01 | `audiolib/doa/srp.c: srp_get_mem_size, srp_init, srp_create_from_geometry, srp, srp2doa, doa_step, doa_get_raw, doa_get_smooth, srp_reset, srp_destroy`<br>`audiolib/doa/doa_smoother.c: doa_smoother_update` | — | C-only | D-43 | — | 第三方 C 來源，repo 內沒有 Python 版（設計如此）。SRP-PHAT 與幾何導出。 |
| TP-UTIL-01 | `audiolib/common/utility/spatial_complex.c: spatial_complex_add, spatial_complex_mul, spatial_complex_conj, spatial_complex_scale`<br>`audiolib/common/utility/spatial_simd.c: spatial_phat_cross_scalar, spatial_phat_cross, spatial_conj_beamform_scalar, spatial_conj_beamform, spatial_bf_projection_scalar, spatial_bf_projection, spatial_complex_sub_array_scalar, spatial_complex_sub_array, spatial_bf_effective_weights_scalar, spatial_bf_effective_weights, spatial_pair_score_accumulate_scalar, spatial_pair_score_accumulate, spatial_simd_backend` | — | C-only | D-43 | — | 複數與 NEON 空間運算輔助；無 Python 版。 |

### 7.3 audiolib

| ID | C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 | 備註 |
|---|---|---|---|---|---|---|
| AC-HPF-01 | `audiolib/common/src/hpf.c: hpf_compute_coeffs` | `lib/aec/python/modules/preprocessing.py:HighPassFilter.__init__` | mirror | D-01 | — | 2 階 Butterworth 的 bilinear 係數。C 用 float32，Python 用 float64，係數差落在 float32 捨入內。 |
| AC-HPF-02 | `audiolib/common/src/hpf.c: hpf_process` | `lib/aec/python/modules/preprocessing.py:HighPassFilter.process` | mirror | D-01；T-06 | — | Direct-Form-II 轉置。80 Hz 時 C 對 Python 最大差 8k 4.8e-6、16k 4.2e-5、48k 2.55e-4；經 AEC 的硬分支放大後端到端 -35~-63 dB（屬 AEC 函式庫範圍）。HPF 沒有專屬的 Python↔C golden，只有端到端間接覆蓋。 |
| AC-HPF-03 | `audiolib/common/src/hpf.c: hpf_reset` | `lib/aec/python/modules/preprocessing.py:HighPassFilter.reset` | mirror | — | — | — |
| AC-HPF-04 | `audiolib/common/src/hpf.c: hpf_params_valid` | — | C-only | S-04 | — | 參數域檢查（0 < cutoff < 0.45 * sr）；Python 的 `HighPassFilter` 與 `AecConfig` 不檢查，Nyquist 以上會靜默得到近靜音或 NaN。 |
| AC-HPF-05 | `audiolib/common/src/hpf.c: hpf_create, hpf_get_mem_size, hpf_init, hpf_destroy` | — | C-only | — | — | 建立／銷毀／記憶體池 API。 |
| AC-RS-01 | `audiolib/common/src/audio_resampler.c: audio_resampler_design` | `pipelines/dfn/dfn2_rate_adapter.py:AudioResampler._design` | mirror | D-24；D-25 | `pipelines/dfn/tests/test_dfn2_rate_adapter.py`（C↔Py，SIMD=0） | Blackman 窗 sinc 的有理多相濾波器設計；SIMD=0 對 Python twin 在 25 組比值 x 3 訊號上逐位元相同（一次性量測）。 |
| AC-RS-02 | `audiolib/common/src/audio_resampler.c: audio_resampler_process, rs_run_integer_up, rs_run_decimation, rs_run_rational` | `pipelines/dfn/dfn2_rate_adapter.py:AudioResampler.process` | mirror | D-24；T-07 | `pipelines/dfn/tests/test_dfn2_rate_adapter.py`（C↔Py，SIMD=0） | SIMD=0 逐位元相同；NEON 因四路累加順序不同，最大差 5.36e-7（24k→8k DC）。多聲道與分塊呼叫的一致性只有一次性量測。 |
| AC-RS-03 | `audiolib/common/src/audio_resampler.c: rs_dot1, rs_dot4` | `pipelines/dfn/dfn2_rate_adapter.py:AudioResampler._dot` | mirror | D-24 | `pipelines/dfn/tests/test_dfn2_rate_adapter.py`（C↔Py，SIMD=0） | NEON 版 rs_dot4 的累加順序與 Python 不同。 |
| AC-RS-04 | `audiolib/common/src/audio_resampler.c: audio_resampler_reset` | `pipelines/dfn/dfn2_rate_adapter.py:AudioResampler.reset` | mirror | — | — | — |
| AC-RS-05 | `audiolib/common/src/audio_resampler.c: audio_resampler_latency_input_frames` | `pipelines/dfn/dfn2_rate_adapter.py:AudioResampler.latency_input_frames` | mirror | — | — | — |
| AC-RS-06 | `audiolib/common/src/audio_resampler.c: audio_resampler_output_bound` | `pipelines/dfn/dfn2_rate_adapter.py:AudioResampler.output_bound` | mirror | — | — | — |
| AC-RS-07 | `audiolib/common/src/audio_resampler.c: audio_resampler_rate_supported` | `pipelines/dfn/dfn2_rate_adapter.py:SUPPORTED_RATES` | partial | — | — | （推論）C 是函式，Python 是常數元組（8/16/24/32/48 kHz）。 |
| AC-RS-08 | `audiolib/common/src/audio_resampler.c: audio_gcd, audio_resampler_layout, audio_resampler_get_mem_size, audio_resampler_carve, audio_resampler_init, audio_resampler_create, audio_resampler_destroy` | — | C-only | — | — | 記憶體池、carve、gcd、存取函式。 |
| AC-STFT-01 | `audiolib/common/src/audio_window.c: ac_window_build` | `pipelines/mono/aec_nr_res/pipeline.py:synth_window` | mirror | D-26 | — | periodic sqrt-Hann。C 以 float32 公式計算，Python 先算 float64 再轉 float32；n=512 時 259/512 項差 1 個 float32 位元（最大 7.1e-7）。 |
| AC-STFT-02 | `audiolib/common/src/audio_stft.c: ac_stft_push, ac_stft_push_planar, ac_stft_flush_planar` | `AIAEC/aiaec_streaming.py:StreamSTFT.push`<br>`AIAEC/aiaec_streaming.py:StreamSTFT.flush` | partial | D-33；D-34 | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py） | center=1 時對 Python 串流孿生最大差 9.5e-7（16k/512/256）；center=0 的滾動框第 j 幀（j>=1）與 center=1 的第 j 幀逐位元相同，第 0 幀本來就不同。center=0 沒有同名的 Python 孿生（訓練端用 torch.stft）。 |
| AC-STFT-03 | `audiolib/common/src/audio_stft.c: ac_stft_frame_planar, frame_interleaved, frame_planar, transform` | `AIAEC/aiaec_streaming.py:StreamSTFT._emit` | partial | D-33 | — | （推論）單框加窗與正向 FFT；Python 以 torch.fft 計算。 |
| AC-STFT-04 | `audiolib/common/src/audio_stft.c: history_append, build_head_segment, build_tail_segment` | `AIAEC/aiaec_streaming.py:StreamSTFT.push`<br>`AIAEC/aiaec_streaming.py:StreamSTFT.flush` | partial | D-34 | — | （推論）center=1 的前後 reflect 補齊與歷史緩衝。 |
| AC-STFT-05 | `audiolib/common/src/audio_stft.c: ac_stft_reset` | `AIAEC/aiaec_streaming.py:StreamSTFT.reset` | mirror | — | — | — |
| AC-STFT-06 | `audiolib/common/src/audio_stft.c: ac_istft_push, ac_istft_push_planar, overlap_add` | `AIAEC/aiaec_streaming.py:StreamISTFT.push`<br>`pipelines/mono/aec_nr_res/pipeline.py:synthesize_hop` | partial | D-27；T-08 | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py） | 兩種正規化：ENV_NORM（除以窗平方包絡）對應 `StreamISTFT`；ACCUMULATE（不除）對應管線的 `synthesize_hop`。對拍的是 consumer 測試與一次性量測，audiolib 自己的 test_stft 沒有 Python 對拍。 |
| AC-STFT-07 | `audiolib/common/src/audio_stft.c: ac_istft_flush` | `AIAEC/aiaec_streaming.py:StreamISTFT.flush` | mirror | D-27 | — | 最後一個 hop 的尾端 <=8 個樣本被包絡除法放大（n=512 時 1.73e-5），C 與 Python 孿生同形。 |
| AC-STFT-08 | `audiolib/common/src/audio_stft.c: ac_istft_reset` | `AIAEC/aiaec_streaming.py:StreamISTFT.reset` | mirror | — | — | — |
| AC-STFT-09 | `audiolib/common/src/audio_stft.c: config_valid, layout_walk, take, carve_tables, ac_stft_config_default, ac_stft_get_mem_size, ac_stft_init, ac_stft_window, ac_stft_get_config, istft_arrays, ac_istft_get_mem_size, ac_istft_init, ac_istft_get_config` | — | C-only | — | — | 設定、記憶體佈局、存取函式。 |
| AC-FFT-01 | `audiolib/common/src/fft_wrapper_ne10.c: fft_size_in_range, fft_create, fft_get_mem_size, fft_init, fft_destroy, fft_get_n_freqs, fft_forward, fft_inverse, fft_forward_scratch, fft_inverse_scratch, fft_power, fft_apply_gain` | — | C-only | D-31；T-09 | — | NE10 後端的 FFT 包裝。外部參照是 numpy.fft（正向絕對差 <=3.95e-5 @ n=8192，一次性量測），repo 內沒有同名 Python 符號；沒有專屬的 Python golden。 |
| AC-WAV-01 | `audiolib/common/include/wav_io.h: wav_io_file_size, wav_io_skip_chunk, wav_open_read, wav_read_float, wav_close_read, wav_open_write, wav_write_float, wav_finalize_write, wav_close_write` | — | C-only | D-29；D-31 | — | WAV 讀寫（header-only）。量化慣例與 soundfile／NR／評測端不同，跨 C／Python 的 WAV 位元比對天生帶 1~2 LSB 偏差。 |
| AC-MATH-01 | `audiolib/common/utility/fast_math.h: fast_exp, fast_exp_neg, fast_log, fast_log10, fast_sqrt, exp1_approx, clip_f, max_f` | — | C-only | D-28 | — | fast_exp／fast_log／fast_sqrt 等近似；參照是 libm（USE_STANDARD_MATH 建置），沒有 Python 孿生。 |
| AC-MEM-01 | `audiolib/common/utility/mem_align.h: ck_add_size, ck_mul_size, ck_align16_size, ck_field_size, ck_field_size_reps, ac_memreq_abi_hash` | — | C-only | — | — | 尺寸運算的溢位檢查與對齊工具；`ac_memreq_abi_hash` 為 native-pointer pool descriptor 加入32-bit ABI識別，64-bit既有hash不變。 |
| AC-SIMD-01 | `audiolib/common/utility/simd_kernels.h: sk_min_f32, sk_min_f32_scalar` | `pipelines/mono/aec_nr_res/pipeline.py:total_gain` | partial | D-30 | — | `total_gain` 不是純粹的 `sk_min_f32` 呼叫：Python 檔內明載核心是 `sk_min_f32` 的孿生，但函式本身還有多增益源的乘積組合與分支停用的路徑，核心以外的部分沒有 C 對應。核心本身有限正常值逐位元相同，NaN 與 +-0 的回傳規則不同（增益恆為有限正數，生產不可觀）。 |
| AC-SIMD-02 | `audiolib/common/utility/simd_kernels.h: sk_capply_gain_f32, sk_capply_gain_f32_scalar` | `pipelines/mono/aec_nr_res/pipeline.py:apply_gain` | mirror | D-30 | — | S(f) = E(f) * g_total；Python 檔內明載是 `sk_capply_gain_f32` 的孿生。 |
| AC-SIMD-03 | `audiolib/common/utility/simd_kernels.h: sk_wola_accumulate_f32, sk_wola_accumulate_f32_scalar` | `pipelines/mono/aec_nr_res/pipeline.py:synthesize_hop` | partial | T-08 | — | （推論）acc += x * w 的重疊相加累加；Python 在 `synthesize_hop` 內聯。 |
| AC-SIMD-04 | `audiolib/common/utility/simd_kernels.h: sk_noise_update_f32, sk_noise_update_f32_scalar` | `lib/nr/python/core/noise_estimators/mcra.py:McraNoiseEstimator.update` | partial | — | — | （推論）遞迴噪聲 PSD 更新。是 `update` 內的一步；初始化與場景轉換的落差屬 NR 函式庫範圍。 |
| AC-SIMD-05 | `audiolib/common/utility/simd_kernels.h: sk__fast_sqrt_elem, sk_ema_f32_scalar, sk_ema_f32, sk__cquad_load, sk__cquad_store, sk_cadd_f32_scalar, sk_cadd_f32, sk_sq_scale_f32_scalar, sk_sq_scale_f32, sk_clip_f32_scalar, sk_clip_f32, sk_fast_sqrt_f32_scalar, sk_fast_sqrt_f32, sk__fast_exp_elem, sk__fast_exp_neg_elem, sk__fast_log_elem, sk__fast_log10_elem, sk__exp1_approx_elem, sk_fast_exp_f32_scalar, sk_fast_exp_neg_f32_scalar, sk_fast_log_f32_scalar, sk_exp1_approx_f32_scalar, sk__fast_exp_vec, sk_fast_exp_f32, sk_fast_exp_neg_f32, sk__fast_log_vec, sk_fast_log_f32, sk_exp1_approx_f32, sk__clip_scale_to_s16_elem, sk_clip_scale_to_s16_arr_scalar, sk_clip_scale_to_s16, sk_clip_scale_to_s16_arr, sk_cmag_f32_scalar, sk_cmag_f32, sk_linear_to_db_f32, sk_asym_ema_f32_scalar, sk_asym_ema_f32, sk_scale_f32_scalar, sk_scale_f32, sk_argmax_f32_scalar, sk_argmax_f32, sk_argmax_i32, sk_argmax_sq_f32_scalar, sk_argmax_sq_f32, sk_cmac_fma_f32_scalar, sk_cmac_fma_ring_f32_scalar, sk_cmac_fma_ring_f32, sk_cmac_f32_scalar, sk_cmac_f32, sk_csub_f32_scalar, sk_csub_f32` | — | C-only | D-31；T-10 | — | 其餘元素運算核心（ema、cmag、clip、fast_* 向量版、linear_to_db、clip_scale_to_s16 等）：Python 對應是 lib/nr 與 lib/aec 函式內的 numpy 內聯式，不在本表範圍；本身只有 NEON 對純量的自測。包含 AEC 的 argmax（平方峰值／原始值／整數）、cmac（fma 版＋非 fma 版）與 csub；外層在 `delay_aec.c`／`filter_analyzer.c`／`render_signal_analyzer.c`／`utility/simd_aec.h`／`pbfdkf.c`／`aec_post.c`／`linear_filter_output.c`／pipelines 的 `mcab_internal.h` 是薄 wrapper 或直接呼叫，不影響這些檔案原有的列。`render_signal_analyzer.c` 的 max_abs 呼叫 `utility/simd_aec.h` 的 `sk_max_abs_f32`（見 AC-SIMD-07），不在本列。 |
| AC-SIMD-06 | `audiolib/common/utility/simd_kernel_nn.h: skn_all_finite_f32_scalar, skn_all_finite_cf32_scalar, skn__exp_field, skn__all_finite_view, skn_all_finite_f32, skn_all_finite_cf32, skn_fill_f32_scalar, skn_fill_f32, skn_mul_f32_scalar, skn_mul_f32, skn_power_scale_f32_scalar, skn_power_scale_f32, skn_mag_eps_f32_scalar, skn_mag_eps_f32, skn_deinterleave_cf32_scalar, skn_deinterleave_scale_cf32_scalar, skn_interleave_cf32_scalar, skn_interleave_scale_cf32_scalar, skn_deinterleave_cf32, skn_deinterleave_scale_cf32, skn_interleave_cf32, skn_interleave_scale_cf32, skn_div_floor_f32_scalar, skn_div_floor_f32, skn__ctaps_mac_range, skn_ctaps_mac_f32_scalar, skn_ctaps_mac_f32, skn_erb_expand_mask_f32_scalar, skn_erb_expand_mask_f32_neon, skn_erb_expand_mask_f32, skn_erb_band_accumulate_f32_scalar, skn_erb_band_accumulate_nch_f32_scalar, skn_erb_band_accumulate_nch_f32, skn_all_finite_rows_f32_scalar, skn_all_finite_rows_f32` | — | C-only | D-31；T-10 | — | AINR／AIAEC 前後處理用的核心（有限性掃描、planar 與 interleaved 轉換、power scale、ctaps_mac 等），只有 NEON 對純量的自測。包含 DFN2/DFN3 共用的 ERB mask 展開與 RNNoise-ERB/GTCRN 共用的 ERB band 累加；外層是各自檔案裡的薄 wrapper（見 AINR-DFN2-05／AINR-RN-02／AINR-GTCRN-03）。`skn_erb_forward_banded_f32` 列在 AINR-DFN2-02（它有自己的 partial 列指到 Python，同一 (檔案,函式) 不在兩列重複）。 |
| AC-SIMD-07 | `audiolib/common/utility/simd_aec.h: sk__cabs_np_elem,sk__cmag2_np_elem,sk__cabs_np_neon4,sk__cmag2_np_neon4,sk_cabs_np_f32_scalar,sk_cabs_np_f32,sk_cmag2_np_f32_scalar,sk_cmag2_np_f32,sk_cmag2_np_acc_f32_scalar,sk_cmag2_np_acc_f32,sk_wupdate_nlms_f32_scalar,sk_wupdate_nlms_f32,sk_wupdate_kf_f32_scalar,sk_wupdate_kf_f32,sk__pairwise_sum_leaf_scalar,sk_pairwise_sum_f32_scalar,sk__pairwise_sum_leaf_neon,sk_pairwise_sum_f32,sk__sum_sq_leaf_scalar,sk_sum_sq_pairwise_f32_scalar,sk__sum_sq_leaf_neon,sk_sum_sq_pairwise_f32,sk__coherence_ema_gate_elem,sk_coherence_ema_gate_f32_scalar,sk_coherence_ema_gate_f32,sk_ema_delta_f32_scalar,sk_ema_delta_f32,sk_n2_track_f32_scalar,sk_n2_track_f32,sk_n2_initial_track_f32_scalar,sk_n2_initial_track_f32,sk_mask_zero_f32_scalar,sk_mask_zero_f32,sk__pairwise_sum_tailfold_leaf_scalar,sk_pairwise_sum_tailfold_f32_scalar,sk__pairwise_sum_tailfold_leaf_neon,sk_pairwise_sum_tailfold_f32,sk_pairwise_sum_tailfold_b_f32_scalar,sk_pairwise_sum_tailfold_b_f32,sk__noise_spectrum_update_elem,sk_noise_spectrum_update_f32_scalar,sk_noise_spectrum_update_f32,sk__erl_bin_update_elem,sk_erl_bin_update_f32_scalar,sk_erl_bin_update_f32,sk_dec1_floorintmin_s32_scalar,sk_dec1_floorintmin_s32,sk__erl_hold_expire_elem,sk_erl_hold_expire_f32_scalar,sk_erl_hold_expire_f32,sk__no_audible_echo_gain_elem,sk_no_audible_echo_gain_f32_scalar,sk_no_audible_echo_gain_f32,sk_max_abs_f32_scalar,sk_max_abs_f32,sk__noise_floor_track_elem,sk_noise_floor_track_f32_scalar,sk_noise_floor_track_f32,sk_raise_to_f32_scalar,sk_raise_to_f32,sk__stationary_elem,sk_stationary_mask_u8_scalar,sk_stationary_mask_u8,sk_h_error_refresh_f32_scalar,sk_h_error_refresh_f32,sk_erle_accumulate_f32_scalar,sk_erle_accumulate_f32,sk_erle_release_f32_scalar,sk_erle_release_f32,sk_peak_counters_i64_scalar,sk_peak_counters_i64,sk_stationarity_hangover_i32_scalar,sk_stationarity_hangover_i32,sk_stationarity_smooth_u8_scalar,sk_stationarity_smooth_u8,sk_stationarity_flags_u8_scalar,sk_stationarity_flags_u8,sk_saturation_counts_f32_scalar,sk_saturation_counts_f32` | — | C-only | D-31；T-10 | `ctest:AEC/c_impl/test/simd_selftest_aec.c` | AEC 專用 NEON／純量核心（NLMS／Kalman 權重更新、複數模平方、CNG EMA、ERL 追蹤、pairwise 加總、穩態遮罩等），另有 8 組 kernel（16 個 scalar／NEON 入口）來自 `pbfdkf.c`、`subband_erle.c`、`render_signal_analyzer.c`、`stationarity_estimator.c`、`saturation.c` 的內嵌區塊；有狀態的 wrapper 留在各模組。AEC 呼叫端直接包含 `utility/simd_aec.h`；本列統計共用核心，`AEC/docs/c_python_function_map.md` §2.4 僅說明共用範圍，不重複列入。`render_signal_analyzer.c` 的 `rsa_max_abs` 是 `sk_max_abs_f32`／`sk_max_abs_f32_scalar` 的薄 wrapper；NaN 契約（任何位置的 NaN 都不參與最大值，numpy 則回傳 NaN）見標頭。 |
| AC-SIMD-08 | `audiolib/common/utility/simd_nr.h: skr_gain_prepare_elem,skr_gain_prepare_scalar,skr_gain_prepare_neon,skr_final_blend_elem,skr_final_blend_scalar,skr_final_blend_neon,skr_final_store_elem,skr_final_store_scalar,skr_final_store_neon,skr_min_ordered_scalar,skr_min_ordered_fast_neon,skr_min_ordered_neon,skr_vmin_lt,skr_ring_min_elem,skr_ring_min_scalar,skr_ring_min_neon,skr_restart_inert_elem,skr_restart_inert_scalar,skr_restart_inert_neon` | — | C-only | D-31；T-10 | `ctest:NR/c_impl/test/test_anr_kernels.c` | NR 專用 NEON／純量核心（MMSE-LSA gain 的 xi/(1+xi) 反推、log 域 final gain blend/store、noise tracker 的有序最小值／4-bin 融合 ring-min／inert bin 重啟）。前綴是 `skr_`（不是 `sk_`/`skn_`），避免撞名。**ANR_VECTOR 的派發邏輯不在這裡**：`gain_prepare`／`noise_estimator_min_ordered`／`_ring_min`／`_restart_inert` 四個 `#if ANR_VECTOR` 分派 wrapper 留在 NR，因為 NR 另有 `ANR_REFERENCE_KERNELS` 這個獨立於 `SK_HAVE_NEON` 的強制純量旋鈕（`NR/python/tools/test_bit_exact.py` 用它產生參照建置），改成直接依 `SK_HAVE_NEON` 派發會讓它悄悄失效（見 `simd_nr.h` 檔首註解）。`final_blend`/`final_store` 在 `#if defined(USE_FAST_GAIN_SMOOTHING) && ANR_VECTOR` 下直接呼叫 `skr_final_blend_neon`/`skr_final_store_neon`。`spp_dd_*`（spp_estimator.c）不在此列：它吃整個不透明的 `SppEstimator*` 並直接寫入三個私有陣列。Python 對應關係見 `NR/docs/c_python_function_map.md`。 |
| AC-SIMD-09 | `audiolib/common/utility/simd_kernels.h: sk_a32_vdivq_f32, sk_a32_vsqrtq_f32, sk_a32_vrndmq_f32, sk_a32_vaddvq_f32, sk_a32_vmaxvq_f32, sk_a32_vpaddq_f32, sk_a32_vaddvq_u32, sk_a32_vmaxvq_u32, sk_a32_vminvq_u32, sk_a32_vminvq_u8, sk_a32_vaddlvq_u8, sk_a32_vceqzq_s32, sk_a32_vzip1q_f32, sk_a32_vzip2q_f32` | — | C-only | D-31 | — | 32 位元 ARM 上 AArch64 專用 NEON intrinsic 的替身（逐 lane 精確除法與開根號、跨 lane 加總與取最大最小、zip）；AArch64 不使用（除了自測的 `SK_A32_SHIM_HOST_TEST`）。只有 `common/test/simd_a32_shim_selftest.c` 對 C 參考實作與 AArch64 intrinsic 的比對，沒有 Python 對應。 |

### 7.4 AINR

| ID | C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 | 備註 |
|---|---|---|---|---|---|---|
| AINR-DFN2-01 | `lib/ainr/dfn2/dfn2_process.c: df_common_features, dfn2_compute_features` | `AINR/DeepFilterNet2/train.py:extract_dfn2_features`<br>`pipelines/dfn/dfn2_stage.py:Dfn2Stage._features` | mirror | S-17；D-35 | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py）；`pipelines/dfn/tests/test_dfn2_stage_c_parity.py`（C↔Py） | 1499 幀含數位靜音：ERB 特徵 7.82e-6、SPEC 2.38e-6、EMA 狀態 3.0e-4 dB。補數係數 `1-alpha` 的取整造成約 2e-6 的系統偏差；既有測試只涵蓋 9~30 幀。 |
| AINR-DFN2-02 | `audiolib/common/utility/simd_kernel_nn.h: skn_erb_forward_banded_f32` | `AINR/DeepFilterNet2/train.py:erb_band_db` | partial | — | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py） | （推論）帶狀 ERB 前向矩陣乘；dB 轉換在 `df_common_features`。核心是共用標頭的 `skn_erb_forward_banded_f32`；DFN3 呼叫同一個函式，呼叫點在 `dfn3_process.c`（該檔不在本表的涵蓋清單內，故無對應列）。 |
| AINR-DFN2-05 | `lib/ainr/dfn2/dfn2_process.c: df_common_expand_mask, df_common_expand_mask_scalar, df_common_expand_mask_neon, df_mask_spectrum` | `pipelines/dfn/dfn2_stage.py:expand_erb_mask`<br>`AINR/DeepFilterNet2/model.py:DeepFilterNet2.compose`<br>`AINR/DeepFilterNet2/model.py:DeepFilterNet2.apply_erb_mask` | mirror | S-21 | `AINR/DeepFilterNet2/tests/test_dfn2_contract.py`（Py 契約） | ERB mask 以 band-major 逆矩陣展開到 bin；矩陣版面是契約（`--output-dir` 匯出包的 inverse 版面不同）。`df_mask_spectrum` 只在 `-DDFN2_MASK_ONLY` 建置存在（= `apply_erb_mask`，由 `AINR/tests/test_dfn2_mask_only_c.py` 對 numpy 對拍）。 |
| AINR-DFN2-06 | `lib/ainr/dfn2/dfn2_process.c: df_common_deep_filter` | `AINR/DeepFilterNet2/model.py:deep_filter_apply` | mirror | — | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py） | 深度濾波器的複數 FIR；`dfn2_enhance_spectrum` 整體對 `model.compose` 最大差 2.98e-8。 |
| AINR-DFN2-07 | `lib/ainr/dfn2/dfn2_process.c: dfn2_enhance_spectrum` | `AINR/DeepFilterNet2/model.py:DeepFilterNet2.compose` | mirror | — | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py） | — |
| AINR-DFN2-08 | `lib/ainr/dfn2/dfn2_process.c: dfn2_enhance_spectrum_stream` | `pipelines/dfn/dfn2_stage.py:Dfn2Stage._compose` | mirror | D-42 | `pipelines/dfn/tests/test_dfn2_stage_c_parity.py`（C↔Py） | 每個 emitted frame 逐位元相同。 |
| AINR-DFN2-09 | `lib/ainr/dfn2/dfn2_process.c: df_common_atten_lim, dfn2_apply_atten_lim` | `AINR/DeepFilterNet2/inference.py:apply_atten_lim` | mirror | — | — | atten_lim 的 +12/-12/0/40/100 dB 逐位元相同（3 dB 差 1.19e-7）。 |
| AINR-DFN2-10 | — | `AINR/DeepFilterNet2/model.py:DeepFilterNet2.post_filter` | Python-only | S-16 | — | Valin post-filter（--pf-beta／config `mask_pf`，出貨關閉）。C 沒有對應符號；出貨路徑（mask_pf=false）兩邊無差。 |
| AINR-DFN2-11 | `lib/ainr/dfn2/dfn2_process.c: dfn2_set_erb_matrices`<br>`lib/ainr/dfn2/dfn2_prepost.c: dfn2_prepost_set_erb_matrices` | `AINR/DeepFilterNet2/export_erb_matrix.py:write_runtime_bins` | partial | S-21 | `AINR/DeepFilterNet2/tests/test_dfn2_contract.py`（Py 契約） | C 只借用呼叫端給的兩個矩陣指標（bin-major forward、band-major inverse），檔案由應用端讀入；`write_runtime_bins` 寫出的 `erb_fwd.bin`、`erb_inv.bin` 是無標頭的 float32 原始檔，版面與這兩個指標一致，範例程式的 `--erb-fwd`、`--erb-inv` 直接讀這兩個檔（S-21）。 |
| AINR-DFN2-12 | `lib/ainr/dfn2/dfn2_process.c: erb_spans, dfn2_state_mem_size, state_carve, state_layout, dfn2_state_init, norm_seed_linear, state_begin, dfn2_state_reset, dfn2_simd_backend` | — | C-only | — | — | band 範圍計算；狀態區塊（`dfn2_state_mem_size` 與 `dfn2_state_init` 走同一次 `state_layout` 切分，大小與切分不會不一致）、起始與重置（`state_begin` 把兩組正規化器放回線性起始值）、SIMD 後端查詢。 |
| AINR-DFN2-13 | `lib/ainr/dfn2/dfn2_model_io.c: dfn2_model_io_push_erb_window, dfn2_model_io_push_spec_window, dfn2_model_io_push_features` | `AINR/DeepFilterNet2/export_onnx.py:feature_windows`<br>`pipelines/dfn/dfn2_stage.py:Dfn2Stage._push_window` | mirror | T-14 | — | [t-1, t, t+1] 圖輸入視窗滑動；40 窗含零 flush 逐位元相同（一次性量測）。每次 invocation N 個 frame（`grid.frames_per_call`）時視窗為 N+2 個 frame，第一次呼叫在推入 M+N 個 frame 後就緒、之後每 N 個一次；Python 孿生是 `export_onnx.py:feature_chunk_windows`（見 AINR-DFN2-26）。 |
| AINR-DFN2-14 | `lib/ainr/dfn2/dfn2_model_io.c: dfn2_model_io_mem_size, io_carve, io_layout, dfn2_model_io_init, zero_registered, dfn2_model_io_ready_frames, dfn2_model_io_descriptor_default, dfn2_model_io_descriptor_validate, all_finite, dfn2_model_io_validate_arrays, dfn2_model_io_validate_state, dfn2_model_io_reset, dfn2_model_io_zero_recurrent, dfn2_model_io_state_generation`<br>`lib/ainr/dfn2/dfn2_model_io_priv.h: dfn2_registration_check` | — | C-only | T-14 | — | 描述子驗證（描述子的 grid 欄位取自呼叫端傳入的 `DFN2Grid`）、視窗區塊的大小與切分（`dfn2_model_io_mem_size` 與 init 走同一次 `io_layout`）、第一次加速器呼叫需要的 feature frame 數（`dfn2_model_io_ready_frames()` = `mask_lookahead` + `frames_per_call`）、已註冊遞迴狀態的有限性檢查與歸零、state generation、註冊規則檢查（`dfn2_registration_check`，函式庫私有的 inline 函式，`dfn2_model_io.c` 與 `dfn2_prepost.c` 共用；h_gru 一組、完整版再加 df_convp_history 一組，全有或全無，`DFN2_MASK_ONLY` 時任何 df_convp_history 值都無效）；只有 C 對 C 或錄製 digest 的測試。 |
| AINR-DFN2-15 | `lib/ainr/dfn2/dfn2_prepost.c: dfn2_prepost_pre_process_freq, dfn2_prepost_pre_process_freq_dual, dfn2_prepost_frame_open, dfn2_prepost_frame_finish, dfn2_prepost_post_process_freq, pp_begin_frame, pp_close_frame` | `pipelines/dfn/dfn2_stage.py:Dfn2Stage.push` | partial | D-42 | `pipelines/dfn/tests/test_dfn2_stage.py`（Py 契約） | 頻域入口；Python 的孿生是雙輸入版（估計頻譜 A、應用頻譜 B），C 類別的單一暫存對在孿生中拆成兩個。 |
| AINR-DFN2-16 | `lib/ainr/dfn2/dfn2_prepost.c: dfn2_prepost_frame_bypass` | `pipelines/dfn/dfn2_stage.py:Dfn2Stage.push` | partial | D-42 | — | fail-open：單位 mask、零 tap、alpha 0。C 在 `dfn2_prepost_frame_finish` 被拒時就地歸零已註冊的遞迴狀態並遞增 state generation，Python 不步進狀態。N = 1 且 M ≥ 1 時，C 對每個暖機幀（init 或 reset 後前 M 拍）的 `frame_finish`／`frame_bypass` 同樣歸零並遞增 generation（丟棄 heads、回 0），Python 沒有暖機幀。 |
| AINR-DFN2-17 | `lib/ainr/dfn2/dfn2_prepost.c: dfn2_prepost_reset` | `pipelines/dfn/dfn2_stage.py:Dfn2Stage.reset` | partial | D-42 | — | （推論）重置類別與其 DFN2State；Python 重置自己的環與狀態，fail-open 後的狀態語意見 D-42。 |
| AINR-DFN2-18 | `lib/ainr/dfn2/dfn2_prepost.c: dfn2_prepost_set_atten_lim` | `pipelines/dfn/dfn2_stage.py:Dfn2Stage.set_atten_lim` | mirror | — | — | — |
| AINR-DFN2-19 | `lib/ainr/dfn2/dfn2_prepost.c: dfn2_prepost_output_frame_index` | `pipelines/dfn/dfn2_stage.py:StageOutput.frame_index` | partial | — | — | （推論）C 是存取函式，Python 把來源 frame 序號放在 `StageOutput.frame_index`。 |
| AINR-DFN2-20 | `lib/ainr/dfn2/dfn2_prepost.c: dfn2_prepost_model_lookahead_frames`<br>`lib/ainr/dfn2/dfn2_prepost.h: dfn2_prepost_latency_frames` | `pipelines/dfn/dfn2_stage.py:MODEL_LOOKAHEAD` | partial | — | — | （推論）C 是存取函式，Python 是常數。N 個 frame 每次 invocation 時 C 回 M+D+N-1（`dfn2_prepost_latency_frames(grid)`，標頭內的 inline 函式，取 grid 的兩個 lookahead 與 `frames_per_call`），`dfn2_stage.py` 的 `MODEL_LOOKAHEAD` 仍是 M+D，因為 pipelines 固定 N=1。 |
| AINR-DFN2-21 | `lib/ainr/dfn2/dfn2_prepost.c: dfn2_prepost_hop_size` | `pipelines/dfn/dfn2_stage.py:HOP` | partial | — | — | （推論）C 是存取函式，Python 是常數。 |
| AINR-DFN2-22 | `lib/ainr/dfn2/dfn2_prepost.c: dfn2_prepost_num_bins` | `pipelines/dfn/dfn2_stage.py:N_BINS` | partial | — | — | （推論）C 是存取函式，Python 是常數。 |
| AINR-DFN2-23 | `lib/ainr/dfn2/dfn2_prepost.c: dfn2_model_run_frame` | `pipelines/dfn/dfn2_stage.py:Dfn2Heads.__call__` | partial | — | — | （推論）加速器 callback 邊界；Python 端是 heads 可呼叫物件。 |
| AINR-DFN2-25 | `lib/ainr/dfn2/dfn2_prepost.c: pp_carve, pp_fnv1a_u32, pp_fnv1a_str, pp_build_hash, pp_layout, dfn2_prepost_config_defaults, pp_check_config, dfn2_prepost_get_mem_size, pp_reset_states, pp_clear_frame, pp_zero_staging, dfn2_prepost_init_ex, dfn2_prepost_init, dfn2_prepost_create, dfn2_prepost_destroy, dfn2_prepost_state_generation, dfn2_prepost_grid, dfn2_prepost_layout_version, pp_all_finite, pp_fill_nan` | — | C-only | T-14 | — | 記憶體池佈局、設定檢查（含已註冊遞迴狀態的全有或全無）、build hash（折入整個 grid：lookahead pair、frames_per_call、兩個 alpha、兩組正規化器設定，加變體與版本）、生命週期、存取函式（`dfn2_prepost_grid()` 回傳實例持有的 grid）。 |
| AINR-DFN2-26 | `lib/ainr/dfn2/dfn2_model_io.c: dfn2_model_io_frame_slot`<br>`lib/ainr/dfn2/dfn2_prepost.c: pp_input_slot, pp_deliver_slot, pp_deliver` | `AINR/DeepFilterNet2/export_onnx.py:feature_chunk_windows` | partial | — | `AINR/tests/test_dfn2_frames_per_call_c.py`（C 對 C） | （推論）每次 invocation N 個 frame：最後一個推入 frame 完成的是第幾個 head（`index % N`）、輸入視窗與輸出 head 的槽位，以及一次 `frame_finish` 後把 N 個 frame 依序合成並逐 hop 交出；Python 側只有匯出端的 N+2 視窗切法，沒有 host 孿生（`dfn2_stage.py` 固定 N=1）。N=1 時槽位恆為 0。 |
| AINR-DFN2-27 | `lib/ainr/dfn2/dfn2_process.c: dfn2_norm_alpha` | `AINR/DeepFilterNet2/train.py:make_norm_alpha` | mirror | — | `AINR/tests/test_dfn2_grid.py`（C↔Py） | 兩個 EMA 係數 = exp(−(hop／sr)／tau) 取三位小數、進位到 1.0 就多一位。C 以 `fma` 對確切的二進位值取捨並平手取偶數，重現 Python `round()`（輔助函式 `round_decimals`，Python 內建函式，表中無對應符號）。tau 取自 grid 的 `erb_norm_tau_sec`／`spec_norm_tau_sec`（出貨模型為 1.0 s）。 |
| AINR-DFN2-28 | `lib/ainr/dfn2/dfn2_process.c: dfn2_grid_make, dfn2_grid_validate` | `AINR/DeepFilterNet2/train.py:validate_signal_config`<br>`AINR/DeepFilterNet2/train.py:analysis_scale` | partial | — | `AINR/tests/test_dfn2_grid.py`（C↔Py） | （推論）由 `DFN2GridSpec` 的 8 個欄位（sample_rate、n_fft、n_erb、df_bins、df_order、兩個 lookahead、frames_per_call；正規化器設定已改為 `dfn2_grid.h` 內的固定常數）推導 hop、bins、ring 與兩個 alpha，並拒絕非法 grid。Python 的 `validate_signal_config` 檢查同一批規則（n_erb 為 4 的倍數、df_bins 範圍、lookahead 範圍），但 `win_len`／`hop_len` 可另設；C 固定 hop = n_fft／2，另外只有 C 檢查 n_fft 是否為 audiolib 接受的大小、2·n_erb ≤ n_bins、df_order 與 frames_per_call 的範圍、兩個 alpha 嚴格落在 (0, 1)，以及推導欄位一致性。`analysis_scale` = 1／√n_fft 對應 `dfn2_stft_scale(n_fft)`（應用端 AcStft 的 `spec_scale`；`dfn2_stft_unscale(n_fft)` 給 AcIstft 的 `ispec_scale`）。 |
| AINR-DFN2-29 | `lib/ainr/dfn2/dfn2_process.c: round_decimals, norm_alpha_ok, state_ready, dfn2_grid_status_text` | — | C-only | — | — | Python `round()` 的 C 重現、EMA 係數是否落在 (0, 1) 的判斷、state 是否已記錄 grid 的判斷、狀態碼的文字（記錄用）。 |
| AINR-DFN2-30 | `lib/ainr/dfn2/dfn2_prepost.h: dfn2_prepost_erb_window_elements, dfn2_prepost_spec_window_elements, dfn2_prepost_erb_mask_elements, dfn2_prepost_coefs_elements, dfn2_prepost_alpha_elements` | — | C-only | — | — | 加速器邊界各張量的 float 個數（隨 grid 的 n_erb、df_bins 與 `frames_per_call`）；Python 端的形狀在匯出的 `input_schema`／`output_schema`。標頭內的 inline 函式，`dfn2_prepost_latency_frames` 見 AINR-DFN2-20。 |
| AINR-GTCRN-01 | `lib/ainr/gtcrn/gtcrn_process.c: gtcrn_model_input` | `AINR/GTCRN/stream_model.py:stream_features` | mirror | — | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py） | 主機側 ERB 前端（mag／re／im）；mag 3.8e-6、re／im 9.5e-7（一次性量測）。 |
| AINR-GTCRN-02 | `lib/ainr/gtcrn/gtcrn_process.c: gtcrn_model_output` | `AINR/GTCRN/stream_model.py:host_synthesis` | mirror | — | `AINR/tests/test_python_c_prepost_parity.py`（C↔Py） | 複數比值遮罩套用與 ERB 逆展開；200 幀逐位元相同。 |
| AINR-GTCRN-03 | `lib/ainr/gtcrn/gtcrn_process.c: erb_band_channel_scalar, erb_band3, erb_band3_neon` | `AINR/GTCRN/model.py:ERB.bm` | mirror | S-33 | — | （推論）低頻 bin 直通、高頻 192 個 bin 折成 64 個 band（band merge）。C 用呼叫端給的前向矩陣（`cfg.erb_fwd`），Python 用 nn.Linear 權重；矩陣由 export_erb_matrix.py 的 runtime-bins 路徑匯出為原始檔。 |
| AINR-GTCRN-04 | `lib/ainr/gtcrn/gtcrn_process.c: mask_expand, mask_expand_scalar, mask_expand_neon` | `AINR/GTCRN/model.py:ERB.bs` | mirror | S-33 | — | （推論）64 band 逆展開回 192 個高頻 bin（band split，band-major 逆矩陣 `cfg.erb_inv`）。 |
| AINR-GTCRN-06 | `lib/ainr/gtcrn/gtcrn_process.c: pp_carve, pp_fnv1a_u32, pp_build_hash, pp_layout, gtcrn_process_config_defaults, pp_registration_ok, pp_check_config, gtcrn_process_get_mem_size, pp_zero_registered, gtcrn_process_init, gtcrn_process_init_ex, gtcrn_process_grid, gtcrn_process_set_erb_matrices, gtcrn_process_reset, gtcrn_process_state_generation, gtcrn_process_state_validate, gtcrn_process_mask_validate, gtcrn_simd_backend` | — | C-only | T-14 | — | 記憶體池佈局與 build hash、設定檢查（八個狀態張量全有或全無）、生命週期與 state generation、輸出驗證（先 `mask_validate` 驗 mask，再 `state_validate` 驗已註冊的八個狀態張量；任一非有限就歸零已註冊狀態並遞增 generation）、ERB 矩陣替換、SIMD 後端查詢；沒有 Python 孿生。 |
| AINR-GTCRN-07 | `lib/ainr/gtcrn/gtcrn_process.c: n_fft_accepted, gtcrn_grid_make, gtcrn_grid_validate, gtcrn_grid_status_text` | — | C-only | — | — | 由 `GTCRNGridSpec`（n_fft、erb_low_bins、erb_high_bands）推導 hop、n_bins、erb_bands、erb_high_bins，並拒絕 audiolib 不接受的 n_fft 與超出範圍的 ERB 設定；GTCRN 是 16 kHz 模型，取樣率不在 grid 內。 |
| AINR-RN-02 | `lib/ainr/rnnoise_erb/process.c: erb_forward_banded, rnnoise_compute_features` | `AINR/RNNoise-ERB/train.py:extract_model_features` | mirror | S-17；D-32 | `AINR/RNNoise-ERB/tests/test_python_c_features.py`（C↔Py） | 相同頻譜輸入下 feature 與正規化器狀態在 float32 容差內一致（測試 rtol 3e-5；矩陣乘法的累加順序不同，不是逐位元）；FFT 捨入經低能量 band 的 log 放大，與 torch 非逐位元。`erb_forward_banded` 只加總前向矩陣的非零帶，等於略去確切零權重的稠密乘積。 |
| AINR-RN-03 | `lib/ainr/rnnoise_erb/process.c: rnnoise_expand_gains` | `AINR/RNNoise-ERB/train.py:apply_erb_gains_batch` | partial | — | — | bin_gains = gains @ erb_inv.T；Python 同函式還含乘上頻譜與邊界 frame 補 1。對 `gains @ erb_inv.T` 差 5.96e-8（1 ULP）。 |
| AINR-RN-04 | `lib/ainr/rnnoise_erb/process.c: rnnoise_apply_gains` | `AINR/RNNoise-ERB/train.py:istft` | partial | D-32 | `AINR/RNNoise-ERB/tests/test_python_c_features.py`（C↔Py） | C 只做頻域乘法 spec *= bin_gain*sqrt(N)（增益與 `istft` 的 ×sqrt(N) 合成一次乘法，對 `re*(gain*scale)` 逐位元相同；bin_gains 由 `rnnoise_expand_gains` 給）；IFFT、root-Hann 與 ACCUMULATE overlap-add 由應用端的 AcIstft（`ispec_scale` 1.0）執行，所以 Python 的 `istft` 在 C 側拆成兩處。測試把兩者接起來對 `istft(spec*gains)`，首幀即輸出一 hop。 |
| AINR-RN-05 | `lib/ainr/rnnoise_erb/process.c: rnnoise_apply_atten_lim` | `AINR/RNNoise-ERB/inference.py:apply_atten_lim` | mirror | — | `AINR/RNNoise-ERB/tests/test_python_c_features.py`（C↔Py）；`AINR/RNNoise-ERB/test/test_rnnoise_features.c`（C 對 C：衰減 dB 非有限或 state／gains 為 NULL 時回 -1 且 gains 不改動，<=0 為停用） | — |
| AINR-RN-07 | `lib/ainr/rnnoise_erb/process.c: fnv1a_u32, fnv1a_f32, fnv1a_f64, fnv1a_str, build_hash, registration_ok, check_config, st_carve, state_layout, erb_spans, norm_seed_linear, seed_norm_states, zero_registered, registered_finite, rnnoise_config_defaults, rnnoise_get_mem_size, rnnoise_init, rnnoise_init_ex, rnnoise_reset, rnnoise_state_generation, rnnoise_frame_finish, rnnoise_simd_backend` | — | C-only | — | `AINR/RNNoise-ERB/test/test_rnnoise_features.c`（C 對 C） | 記憶體池佈局與 build hash、設定檢查（h1／h2／h3 全有或全無）、ERB 矩陣的非零帶範圍、正規化器起始值、重置與 state generation、`rnnoise_frame_finish`（band gains 與已註冊狀態有限性檢查，失敗時歸零並讓呼叫端改用單位增益）、SIMD 後端查詢。pre/post layout=2，GRU tensor layout=1。 |
| AINR-RN-12 | — | `AINR/RNNoise-ERB/inference.py:valin_post_filter` | Python-only | S-14 | — | Valin post-filter（--pf-beta，預設 0）。C 沒有對應符號；出貨路徑（beta=0）兩邊無差。 |
| AINR-RN-13 | `lib/ainr/rnnoise_erb/process.c: rnnoise_norm_alpha` | `AINR/RNNoise-ERB/train.py:make_norm_alpha` | mirror | — | `AINR/RNNoise-ERB/tests/test_python_c_features.py`（C↔Py） | 兩個 EMA 係數取三位小數、進位到 1.0 就多一位；C 與 `make_norm_alpha` 的 float32 值相同。 |
| AINR-RN-14 | `lib/ainr/rnnoise_erb/process.c: fft_size_valid, round_decimals, rnnoise_grid_make, rnnoise_grid_validate, rnnoise_grid_status_text` | — | C-only | — | `AINR/RNNoise-ERB/test/test_rnnoise_framing.c`（C 對 C） | 由 `RNNoiseGridSpec`（config.ini 的 [signal]、[feature] 值）推導 hop、bins、spec_bins、conv_delay、兩個 alpha 與尺度，並拒絕非法 grid；n_fft 以 audiolib 是否接受為準，ERB 矩陣不在 grid 內（由應用端讀入）。 |
| AINR-RN-15 | `lib/ainr/rnnoise_erb/process.h: rnnoise_erb_window_elements, rnnoise_spec_window_elements` | — | C-only | — | — | 加速器邊界兩個輸入張量的 float 個數，由 grid 算出：`rnnoise_erb_window_elements` = `RNNOISE_MODEL_INPUT_FRAMES`（3）× `n_bands`，`rnnoise_spec_window_elements` = `RNNOISE_MODEL_INPUT_FRAMES` × 2 × `spec_bins`；應用端據此配置 `out_erb`、`out_spec` 緩衝。標頭內的 inline 函式；Python 端的形狀在匯出的 `input_schema`（`erb_input[1,3,n_bands]`、`spec_input[1,3,2,spec_bins]`）。 |

### 7.5 AIAEC

| ID | C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 | 備註 |
|---|---|---|---|---|---|---|
| AIAEC-UL-01 | `lib/aiaec/ulcnet/ulcnet_model_io.c: ulcnet_model_io_prepare, front_end, front_end_scalar, front_end_neon, front_end_tail_one` | `AIAEC/Align_ULCNet/export_onnx.py:stream_features` | mirror | D-34 | `AIAEC/tests/test_ulcnet_model_io_c.py`（C 契約）；`AIAEC/tests/test_prepost_kernels.py`（NEON 對純量） | 有號冪壓縮、幅度與壓縮域相位方向。16k/257 bin 與 48k/513 bin：error_mag 4.77e-7、far_mag 9.54e-7（一次性量測）。 |
| AIAEC-UL-02 | `lib/aiaec/ulcnet/ulcnet_model_io.c: ulcnet_model_io_enhance_spectrum, back_end, back_end_scalar, back_end_neon` | `AIAEC/Align_ULCNet/export_onnx.py:host_back_end` | mirror | D-34 | `AIAEC/tests/test_ulcnet_model_io_c.py`（C 契約）；`AIAEC/tests/test_prepost_kernels.py`（NEON 對純量） | 壓縮誤差乘上圖輸出的平面複數 mask，再做逆有號冪。enhance_spectrum 相對誤差 8e-8~5.8e-7。 |
| AIAEC-UL-03 | `lib/aiaec/ulcnet/ulcnet_model_io.h: ulcnet_model_io_signed_pow` | `AIAEC/Align_ULCNet/model.py:_signed_power` | mirror | — | — | 檔內註解明載是 model.py `_signed_power` 的唯一 C 副本。 |
| AIAEC-UL-06 | `lib/aiaec/ulcnet/ulcnet_prepost.c: ulcnet_prepost_pre_process_freq, ulcnet_prepost_frame_open` | `AIAEC/Align_ULCNet/export_onnx.py:stream_features` | partial | T-14 | `AIAEC/tests/test_ulcnet_prepost_c.py`（C 對 C） | （推論）暫存一個 hop 的誤差頻譜與遠端頻譜，再送進 model I/O 的 prepare；頻譜由應用端的 AcStft 產生，類別不含框架。 |
| AIAEC-UL-07 | `lib/aiaec/ulcnet/ulcnet_prepost.c: ulcnet_prepost_frame_finish, ulcnet_prepost_post_process_freq` | `AIAEC/Align_ULCNet/export_onnx.py:host_back_end` | partial | T-14 | `AIAEC/tests/test_ulcnet_prepost_c.py`（C 對 C） | （推論）把圖輸出送進 model I/O 的 `enhance_spectrum`，再交出增強後頻譜；合成由應用端的 AcIstft 完成。 |
| AIAEC-UL-09 | `lib/aiaec/ulcnet/ulcnet_prepost.c: pp_carve, pp_fnv1a, pp_build_hash, pp_layout, pp_buffers, ulcnet_prepost_config_defaults, pp_check_config, pp_descriptor_for, ulcnet_prepost_get_mem_size, ulcnet_prepost_init_ex, ulcnet_prepost_init, ulcnet_prepost_create, ulcnet_prepost_destroy, ulcnet_prepost_reset, ulcnet_prepost_state_generation, ulcnet_prepost_hop_size, ulcnet_prepost_num_bins, ulcnet_prepost_descriptor, pp_close_frame, ulcnet_prepost_frame_bypass`<br>`lib/aiaec/ulcnet/ulcnet_process.c: ulcnet_frame_is_finite`<br>`lib/aiaec/ulcnet/ulcnet_model_io.c: mio_carve, mio_layout, ulcnet_grid_make, ulcnet_grid_validate, ulcnet_model_io_descriptor_default, ulcnet_model_io_descriptor_grid, ulcnet_far_input_mode_name, ulcnet_model_io_descriptor_validate, ulcnet_model_io_get_mem_requirements, ulcnet_model_io_init, ulcnet_model_io_state_buffers_validate, zero_registered, all_finite_or_empty, clear_recurrent_state, ulcnet_model_io_reset, ulcnet_model_io_state_generation, frame_is_finite, ulcnet_model_io_descriptor`<br>`lib/aiaec/ulcnet/ulcnet_accelerator_adapter.c: adapter_config, adapter_layout, ulcnet_accelerator_adapter_get_mem_size, ulcnet_accelerator_adapter_init, infer, reset, ulcnet_accelerator_adapter_state_generation, ulcnet_accelerator_adapter_descriptor, ulcnet_accelerator_adapter_model` | — | C-only | T-14 | — | 記憶體池佈局、`UlcnetGrid` 與描述子驗證、已註冊遞迴狀態（全有或全無；函式庫只做有限性掃描與歸零，狀態形狀由應用端持有）、加速器 adapter 與 callback 步進；沒有 Python 孿生。 |
| AIAEC-DV-01 | `lib/aiaec/deepvqe_s/deepvqe_prepost.c: compress_planar` | `AIAEC/aiaec_common.py:compressed_ri_feature` | mirror | — | `AIAEC/tests/test_prepost_kernels.py`（NEON 對純量） | 壓縮 mic／far 最大差 4.77e-7（一次性量測）。 |
| AIAEC-DV-02 | `lib/aiaec/deepvqe_s/deepvqe_process.c: deepvqe_ccm_init, deepvqe_ccm_reset, deepvqe_ccm_process, ccm_edge_bin` | `AIAEC/DeepVQE_S/model.py:DeepVQES._apply_ccm_frame`<br>`AIAEC/aiaec_common.py:apply_causal_tf_filter`<br>`AIAEC/_export_common.py:COMPOSITION` | mirror | — | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py） | 複數卷積遮罩（CCM）；最大差 7.63e-6（峰值 95）。兩個邊緣 bin 的越界 tap 略過，與乘零不同。兩個環由呼叫端借出，`deepvqe_ccm_reset` 只清環。 |
| AIAEC-DV-03 | `lib/aiaec/deepvqe_s/deepvqe_prepost.c: deepvqe_prepost_pre_process_freq, deepvqe_prepost_frame_open` | `AIAEC/aiaec_common.py:compressed_ri_feature` | partial | T-14 | `AIAEC/tests/test_deepvqe_prepost_c.py`（C 對 C） | （推論）暫存一個 hop 的麥克風與遠端頻譜，再做壓縮特徵；頻譜由應用端的 AcStft 產生。 |
| AIAEC-DV-04 | `lib/aiaec/deepvqe_s/deepvqe_prepost.c: deepvqe_prepost_post_process_freq, deepvqe_prepost_frame_finish` | `AIAEC/DeepVQE_S/model.py:DeepVQES._apply_ccm_frame` | partial | T-14 | `AIAEC/tests/test_deepvqe_prepost_c.py`（C 對 C） | （推論）圖輸出的 CCM taps 套到原始麥克風頻譜的環上，再交出增強後頻譜；合成由應用端的 AcIstft 完成。 |
| AIAEC-DV-05 | `lib/aiaec/deepvqe_s/deepvqe_prepost.c: deepvqe_prepost_state_name, deepvqe_prepost_state_name_out, deepvqe_prepost_skip_policy_name, dv_valid_depth` | `AIAEC/_streaming_export.py:StatelessOneFrameAIAEC.state_names` | partial | T-14 | — | （推論）C 側的狀態張量名稱逐字對應匯出器的 input_names；形狀由應用端持有，契約變動要同時 bump layout 版本。 |
| AIAEC-DV-06 | `lib/aiaec/deepvqe_s/deepvqe_prepost.c: deepvqe_prepost_descriptor_default, deepvqe_prepost_descriptor_validate, deepvqe_prepost_descriptor` | `AIAEC/_streaming_export.py:c_descriptor` | partial | T-14 | — | （推論）描述子對匯出器寫入的 metadata。 |
| AIAEC-DV-07 | `lib/aiaec/deepvqe_s/deepvqe_prepost.c: pp_carve, pp_fnv1a, pp_build_hash, pp_layout, deepvqe_grid_make, deepvqe_grid_validate, deepvqe_prepost_config_defaults, pp_registration_ok, pp_check_config, deepvqe_prepost_get_mem_size, deepvqe_prepost_init_ex, deepvqe_prepost_init, deepvqe_prepost_create, deepvqe_prepost_destroy, zero_registered, zero_state, deepvqe_prepost_reset, deepvqe_prepost_state_generation, deepvqe_prepost_hop_size, deepvqe_prepost_num_bins, fill_nan, all_finite, deepvqe_prepost_frame_bypass` | — | C-only | T-14 | — | 記憶體池佈局、`DeepVqeGrid` 與設定檢查、已註冊狀態檢查（十六個張量全有或全無）、build hash、生命週期、有限性掃描、NaN 填充；`deepvqe_prepost_frame_bypass` 輸出靜音（fail closed）；沒有 Python 孿生。 |
| AIAEC-DV-08 | `lib/aiaec/deepvqe_s/deepvqe_prepost.h: deepvqe_prepost_taps_elements` | — | C-only | — | — | CCM taps 張量的 float 個數，由 grid 算出：`n_bins` × `DEEPVQE_TIME_ORDER` × `DEEPVQE_FREQ_TAPS` × 2；應用端據此配置 taps 緩衝。標頭內的 inline 函式；Python 端的形狀在匯出的 `output_schema`。 |
| AIAEC-MASK-01 | `lib/aiaec/cagcrn/cagcrn_process.c: aiaec_apply_complex_mask` | `AIAEC/aiaec_common.py:complex_mask`<br>`AIAEC/_export_common.py:COMPOSITION` | mirror | — | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py，逐位元） | CAGCRN 的 host 複數 mask 乘法（`_export_common.COMPOSITION` 表指定）。 |
| AIAEC-MASK-02 | `lib/aiaec/ulcnet/ulcnet_mask_process.c: aiaec_apply_ulcnet_compressed_mask, aiaec_signed_pow` | `AIAEC/Align_ULCNet/export_onnx.py:host_back_end`<br>`AIAEC/_export_common.py:COMPOSITION` | mirror | D-34 | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py，<2e-5） | Align-ULCNet 的 host 後處理（`_export_common.COMPOSITION` 表指定），與 AIAEC-UL-02 的 back_end 是同一數學。 |
| AIAEC-MASK-03 | `lib/aiaec/align_cruse/align_cruse_process.c: aiaec_apply_real_mask` | `AIAEC/Align_CRUSE/model.py:AlignCRUSE.forward`<br>`AIAEC/_export_common.py:COMPOSITION` | partial | D-36 | `AIAEC/tests/test_ulcnet_process_c.py`（C↔Py，逐位元，對內聯 torch 算式） | 實數幅度 mask（Align-CRUSE 的 `enhanced = microphone * mask`）。Align-CRUSE 沒有 C pre/post 類別，C 只提供這個共用 helper。 |

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
| D-31 | fft_wrapper、fast_math、simd_kernels、wav_io 無 Python 孿生。 |
| D-32 | RNNoise 的 FFT／IFFT 由應用端經 audiolib NE10 執行；數學定義、窗與縮放不變，butterfly 捨入與 torch 仍非逐位元。低能量 band 的 log 會放大微小誤差；窗本身的差異另見 D-26。 |
| D-33 | 離線 Python 用 center=True，C 串流用 center=False，第 0 幀不同。 |
| D-34 | AIAEC 應用端的 AcStft 用 center=0 滾動框，Python 訓練／評測用 center=1。 |
| D-35 | spec_norm_eps：C 保留 1e-12，trainer 沒有。 |
| D-36 | Align_CRUSE、CAGCRN 沒有 C pre/post 類別，只各自帶一份 mask helper（`lib/aiaec/{align_cruse,cagcrn}/`，split 自舊 aiaec_process.c）。 |
| D-38 | 範例組出的 AecConfig（`aec_config_from_level` 加欄位）struct 值與 Python 有效值有 10 個欄位不同（10 ms 基準、carve 時 retime）。 |
| D-39 | 4ch 共用 estimator 的 C-only 排程（duty-cycle）與 delay 模式（FIXED、EXTERNAL_ALIGNED、delay_num_filters）。 |
| D-40 | 8 kHz mono 只有 C；mono CLI 旗標不對等（另見 S-30、S-31）。 |
| D-41 | R² 相量融合的相位下限：C 釘在 +實軸，Python 用 np.angle。 |
| D-42 | DFN2 fail-open：Python 不步進遞迴狀態；C 在輸出被拒時，把已註冊的遞迴狀態就地歸零並遞增 state generation，未註冊狀態的 stage 與 rate bridge 只遞增 generation，由持有狀態的模型回呼重載零；每個暖機幀（N = 1、M ≥ 1）也各歸零並遞增一次（heads 丟棄，音訊不變），Python 沒有這一步（它的 heads 不經加速器）。 |
| D-43 | audiolib（SRP-PHAT、BF、VAD）與 ULCNet NN stage 範例沒有 Python 版（設計如此）；Python 另有實驗專用開關。 |
| D-44 | ULCNet NN stage 在對齊事件（`delay.changed`，或第一次轉為 solid）呼叫 model->reset 並插入 `ULCNET_NN_STAGE_REPRIME_FRAMES` 個恆等框；Python 串流推論（`AIAEC/Align_ULCNet/_streaming.py`）不重設模型狀態、不插恆等框，`lib/aec/python` 也沒有 `AecLinearContext.delay_state` 的對應存取器（只有 `get_post_reset_serial`）。重設政策目前只在 C（有文件的分岔）。 |
| S-04 | HPF 參數域檢查（0 < cutoff < 0.45 * sr）只在 C；Python 的 HighPassFilter／AecConfig 不檢查。 |
| S-05 | AecResContext.filter_converged 在上升緣晚 1 hop：Python 在 update_convergence 之前建構 ctx。 |
| S-14 | RNNoise-ERB 的 Valin post-filter（--pf-beta）只有 Python，C 沒有；出貨 beta=0 兩邊無差。 |
| S-16 | DFN2 的 Valin post-filter（config `mask_pf`）只有 Python，C 沒有；出貨 mask_pf=false 兩邊無差。 |
| S-17 | EMA 補數 1-alpha 的 float32 取整：C 在 float32 算 1-alpha，Python 在 double 算完再轉，造成約 2e-6 的系統偏差。 |
| S-21 | DFN2 `export_erb_matrix.py --output-dir` 匯出包（工具鏈驗證用）的 inverse 版面與 C 期待的 band-major 不同；`--runtime-bins` 寫出 `erb_fwd.bin`、`erb_inv.bin` 兩個原始檔，版面與 C 的矩陣指標一致；範例程式以 `--erb-fwd`、`--erb-inv` 直接讀這兩個檔。 |
| S-30 | Python mono 參考管線不支援 8 kHz，C 支援 8k/256；未獨立驗證。 |
| S-31 | mono CLI 旗標不對等（moderate、--no-nr、--no-res、--no-cng 只在 C）；未獨立驗證。 |
| S-32 | 4ch Python 參考的設定面比 C 少（無 delay_mode、delay_num_filters、fixed_delay_samples、delay_est_period_s、enable_post 等）；未獨立驗證。 |
| S-33 | GTCRN `export_erb_matrix.py` 的 bin／npy 匯出包（工具鏈驗證用）與 C 的 `erb_fwd`／`erb_inv` 指標期待的版面（`--runtime-bins` 寫出的原始檔）不同；未獨立驗證。 |
| S-34 | 設定範圍檢查不對稱：C validate 嚴格 fail-closed，Python 幾乎不驗。 |
| T-06 | HPF 沒有 Python↔C 專屬 golden，只有端到端間接覆蓋。 |
| T-07 | resampler 既有 Python 對拍覆蓋不足（補量後無落差）。 |
| T-08 | audiolib 的 test_stft 驗窗公式、延遲、API 契約與縮放入口，沒有對 Python 的逐位元對拍。 |
| T-09 | FFT 沒有專屬的 Python golden。 |
| T-10 | 沒有 Python 對照的 C 元件（mem_align、部分 simd 核心等）。 |
| T-13 | 應用端寫死的 spec 列（`DFN_EXAMPLE_MODEL_ROWS`）沒有依名稱對 `AINR/DeepFilterNet2/config.ini` 的漂移守衛；測試用的 spec 已有（DFN 的 `pipelines/dfn/tests/test_dfn_test_grid_pin.py`，RNNoise-ERB 的 `AINR/RNNoise-ERB/tests/test_feature_contract.py`）。 |
| T-14 | 沒有 Python 孿生的 C 狀態機（prepost、model_io、adapter）。 |
| T-15 | pipelines 內執行 C 並與 Python 對拍的常設測試有 delay、DFN2 stage、rate bridge（與 resampler twin），以及 `pipelines/tests/pyc_parity_pipelines.sh`（4ch NR 增益、4ch 後波束 RES、4ch 共用 delay 的混合 realign、mono NR／min／apply／CNG／OLA 膠水）；lane 組合（derive_lane）、4ch fuse 本身（只靠以 C 融合框餵入的 a.nr4／b.res4 間接覆蓋）、DFN 組合範例（Py 對 Py）、ULCNet NN stage 沒有常設 C↔Py 對拍。 |
| T-16 | 4ch 後波束 RES 的 clock_drift 輸入：C 傳共用 estimator 旗標，Python 寫死 False；無測試。 |
| T-17 | mono 端到端與上游 AEC 分岔（歸屬 lib/aec 接縫，非 pipeline 膠水）。 |
<!-- map:gaps:end -->

## 9. 未能判斷

目前沒有。這不表示每一列都有證據：標「（推論）」的列（第 4 節統計的最後一欄）是只憑函式名稱與角色得出的配對，證據最弱，要引用它們的對拍結果前應先確認。

