# 單聲道整合手冊（`Aec` + 逐個初始化的函式庫，沒有 mono 函式庫）

適用對象：要把單聲道 AEC → NR → RES 連進自己產品的**函式庫使用者**。

本文只講「什麼時候呼叫哪個 function」「每個參數怎麼設」。演算法內部推導不在本文範圍。

> **路徑慣例**：本文所有路徑都相對於 Audio_ALG repository 根目錄。
> `audiolib` 是與 Audio_ALG 平行的另一個 checkout，本文寫成 `<audiolib>/`；
> 從 `pipelines/` 出發的相對路徑是 `../../audiolib/common`。
>
> **數值來源**：本文所有數字都是從當前 source 讀出，或以當前 checkout 實際建置後量測。
> 記憶體數字**必須**用 API 重新查詢，不要抄本文（見 [§4.5](integration_mono_zh_TW_02_lifecycle_config_errors.md#45-實測記憶體僅供量級參考務必自己重查)）。

---

## 目錄 <!-- split-toc -->
- [單聲道整合手冊（`Aec` + 逐個初始化的函式庫，沒有 mono 函式庫）](#單聲道整合手冊aec--逐個初始化的函式庫沒有-mono-函式庫) <!-- split-toc -->
  - [0. 速覽：流程時序圖、公開／內部函式、外部／內部參數](integration_mono_zh_TW_01_overview_link_quickstart.md#0-速覽流程時序圖公開內部函式外部內部參數) <!-- split-toc -->
    - [0.1 流程時序圖（mono：`Aec` 加逐個初始化的函式庫）](integration_mono_zh_TW_01_overview_link_quickstart.md#01-流程時序圖monoaec-加逐個初始化的函式庫) <!-- split-toc -->
    - [0.2 公開函式與內部函式](integration_mono_zh_TW_01_overview_link_quickstart.md#02-公開函式與內部函式) <!-- split-toc -->
    - [0.3 參數面：外部與內部](integration_mono_zh_TW_01_overview_link_quickstart.md#03-參數面外部與內部) <!-- split-toc -->
  - [1. 你連結什麼、你 `#include` 什麼](integration_mono_zh_TW_01_overview_link_quickstart.md#1-你連結什麼你-include-什麼) <!-- split-toc -->
    - [1.1 構成：每個函式庫各自初始化，每個 hop 的串接由你寫](integration_mono_zh_TW_01_overview_link_quickstart.md#11-構成每個函式庫各自初始化每個-hop-的串接由你寫) <!-- split-toc -->
    - [1.2 `--aec-only`](integration_mono_zh_TW_01_overview_link_quickstart.md#12---aec-only) <!-- split-toc -->
    - [1.3 Header closure](integration_mono_zh_TW_01_overview_link_quickstart.md#13-header-closure) <!-- split-toc -->
    - [1.4 編譯需要的 `-I` 清單（consumer 端最小集合）](integration_mono_zh_TW_01_overview_link_quickstart.md#14-編譯需要的--i-清單consumer-端最小集合) <!-- split-toc -->
    - [1.5 Archive link order](integration_mono_zh_TW_01_overview_link_quickstart.md#15-archive-link-order) <!-- split-toc -->
  - [2. Quick start](integration_mono_zh_TW_01_overview_link_quickstart.md#2-quick-start) <!-- split-toc -->
  - [3. Lifecycle](integration_mono_zh_TW_02_lifecycle_config_errors.md#3-lifecycle) <!-- split-toc -->
    - [3.1 執行期唯讀存取](integration_mono_zh_TW_02_lifecycle_config_errors.md#31-執行期唯讀存取) <!-- split-toc -->
    - [3.1b 逐階段耗時（診斷用）](integration_mono_zh_TW_02_lifecycle_config_errors.md#31b-逐階段耗時診斷用) <!-- split-toc -->
    - [3.2 執行期強度控制](integration_mono_zh_TW_02_lifecycle_config_errors.md#32-執行期強度控制) <!-- split-toc -->
  - [4. Config 完整參考](integration_mono_zh_TW_02_lifecycle_config_errors.md#4-config-完整參考) <!-- split-toc -->
    - [4.1 `SeRequest` → `se_derive()`](integration_mono_zh_TW_02_lifecycle_config_errors.md#41-serequest--se_derive) <!-- split-toc -->
    - [4.2 後級選項（範例程式的欄位與 CLI 旗標）](integration_mono_zh_TW_02_lifecycle_config_errors.md#42-後級選項範例程式的欄位與-cli-旗標) <!-- split-toc -->
    - [4.3 Grid（由 `sample_rate` + `fft_size` 唯一決定）](integration_mono_zh_TW_02_lifecycle_config_errors.md#43-grid由-sample_rate--fft_size-唯一決定) <!-- split-toc -->
    - [4.4 哪些欄位會改變記憶體用量](integration_mono_zh_TW_02_lifecycle_config_errors.md#44-哪些欄位會改變記憶體用量) <!-- split-toc -->
    - [4.4b 診斷用的分項](integration_mono_zh_TW_02_lifecycle_config_errors.md#44b-診斷用的分項) <!-- split-toc -->
    - [4.5 實測記憶體（僅供量級參考，務必自己重查）](integration_mono_zh_TW_02_lifecycle_config_errors.md#45-實測記憶體僅供量級參考務必自己重查) <!-- split-toc -->
  - [5. 錯誤處理](integration_mono_zh_TW_02_lifecycle_config_errors.md#5-錯誤處理) <!-- split-toc -->
    - [5.1 錯誤語意總表](integration_mono_zh_TW_02_lifecycle_config_errors.md#51-錯誤語意總表) <!-- split-toc -->
    - [5.2 init 的拒絕清單](integration_mono_zh_TW_02_lifecycle_config_errors.md#52-init-的拒絕清單) <!-- split-toc -->
    - [5.3 `destroy()` 的語意](integration_mono_zh_TW_02_lifecycle_config_errors.md#53-destroy-的語意) <!-- split-toc -->
    - [5.4 診斷輸出去向](integration_mono_zh_TW_02_lifecycle_config_errors.md#54-診斷輸出去向) <!-- split-toc -->
  - [6. 出貨內容 vs 範例](#6-出貨內容-vs-範例) <!-- split-toc -->
    - [6.1 函式庫（會進產品）](#61-函式庫會進產品) <!-- split-toc -->
    - [6.2 範例碼與參考執行檔（**沒有任何一個會進產品 archive**）](#62-範例碼與參考執行檔沒有任何一個會進產品-archive) <!-- split-toc -->
    - [6.3 測試（不出貨）](#63-測試不出貨) <!-- split-toc -->
  - [7. 整合檢查清單](#7-整合檢查清單) <!-- split-toc -->
  - [8. 版本與相容性](#8-版本與相容性) <!-- split-toc -->
## 6. 出貨內容 vs 範例

`pipelines/` 目錄不是自解釋的。下表區分「函式庫」與「參考程式」。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `lib/aec`、`lib/nr`、audiolib | `libaec.a`、`libanr.a`、`libaudio_common.a` | 線性 AEC（含 AEC3 殘留 gain context）、降噪、FFT／STFT／合成 |
| `pipelines/common/se_config.c` / `.h` | `se_config.o`（4ch 單元的 archive 也含它） | 格點解析與 `AecConfig` 推導（`se_derive()`）；各應用的規則（rate 集合、delay 檢查、lane 對齊、seam context）以 `SeRules` 傳入 |
| `pipelines/common/se_stage.h` | header-only | `SeSpecFrame`、`se_res_stage_gain()`、`se_frame_from_aec()`、CNG 常數 |

### 6.2 範例碼與參考執行檔（**沒有任何一個會進產品 archive**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/samples/se_anr_stage.h` | header-only 的 NR 級 gain 呼叫（`se_anr_stage_gain()`） | 把 AEC 的殘留回音功率摺進降噪器的 noise floor 並回傳 NR 增益；可整份複製改寫，沒有 ABI |
| `pipelines/mono_aec_nr_res/main.c` | heap 路徑 CLI（WAV in / WAV out） | 示範 `Aec` + `AnrDenoiser` + `AcIstft` 各自初始化（`aec`、`nr` 與合成/暫存各一塊 heap）與每 hop 串接（`mono_form()`）；離線比對用 |
| `pipelines/mono_aec_nr_res/static_main.c` | caller-owned pool 路徑 CLI | 示範 query → allocate（AEC、降噪器、合成、暫存在**一塊**對齊 pool 內依序切）→ init → process → destroy → release；另提供 `--print-mem-size` |
| `pipelines/mono_aec_nr_res/example_board_adapter.c` | 板端 adapter 的**主機模擬** | 示範完整呼叫序列（AEC、降噪器、合成、暫存在一塊 arena 內）、pool 大小加總與拒絕路徑。裡面每個 `board_mem_*` 都是主機上的假實作，不是平台程式碼。標了 `// BOARD:` 的地方就是你要換成真實平台程式碼的位置 |

四麥克風的 library 與參考程式是**另外兩份**手冊的主題：
`docs/integration_4ch_core_zh_TW.md`、`docs/integration_4ch_spatial_zh_TW.md`。

### 6.3 測試（不出貨）

`pipelines/mono_aec_nr_res/tests/test_mono_chain.c`（照 `static_main.c` 的展開順序建 Rig，並把兩支 mono 程式當黑箱跑，與手寫 oracle `conv_chain_mono.h` 逐位元比對）、
`pipelines/mono_aec_nr_res/tests/test_no_stdio_stack.c`。

---

## 7. 整合檢查清單

建置與連結：

- [ ] `libaec.a` / `libanr.a` / `libaudio_common.a` 與你的 `se_config.o` 是用**同一組** backend 與編譯選項建出來的
- [ ] link 順序照 [§1.5](integration_mono_zh_TW_01_overview_link_quickstart.md#15-archive-link-order)
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用各 Makefile 的 `print-lib-path` 查的，不是硬寫的
- [ ] 你的 TU 用 `gnu99`（或以上的 GNU 方言）

初始化：

- [ ] 每次 init 之前都重新呼叫 `aec_get_mem_size()`、`anr_get_mem_size()`、`ac_istft_get_mem_size()`，沒有快取大小
- [ ] 每一塊的大小都檢查過 `!= 0`
- [ ] pool 16-byte 對齊，大小 ≥ 各塊 `ALIGN16` 後相加（AEC、降噪器、合成、暫存，加上你自己的 stage）
- [ ] `aec_init()`、`anr_init()`、`ac_istft_init()` 回 `NULL` 時你的程式會停下來，不會繼續跑

執行期：

- [ ] `mic` / `ref` / `out` 三個 buffer 都剛好 `grid.hop_size` 個 float
- [ ] hop 大小是查來的，不是寫死的
- [ ] `se_frame_from_aec()`、`g_res_ctx`、`se_anr_stage_gain()` 的回傳值，以及合成輸出的有限性（`se_hop_is_finite()`）都有檢查
- [ ] 任何一步被拒，或換回聲路徑／換串流時，`aec_reset()`、`anr_reset()`、`ac_istft_reset()` 與 comfort-noise RNG 的重設**一起**做
- [ ] 同一個 instance 沒有被多 thread 同時 process
- [ ] 降噪器的 reset／destroy 與 `Aec` 同進退，沒有只重設其中一個

收尾：

- [ ] `destroy()` 在釋放 pool **之前**
- [ ] pool 在 handle 存活期間沒有被搬動、覆寫或共用

板端額外：

- [ ] 若 image 不能連 stdio，函式庫是以 `NO_STDIO=1` 建的
- [ ] 已確認 pool 不需要預先清零（dirty pool 可接受）
- [ ] bring-up log 有保留 init 的拒絕訊息（NO_STDIO build 就沒有了，此時只能靠回傳值）

---

## 8. 版本與相容性

mono 沒有 descriptor，也沒有 layout version：`Aec`、`AnrDenoiser`、`AecRes`、`AcIstft` 只提供大小查詢，
`se_anr_stage.h` 與各範例程式是範例碼。需要 descriptor 閘的出貨單元只有 4ch 的 `AudioPipeline4Ch`
（`AudioPipeline4ChMemReq`，`AUDIO_PIPELINE_4CH_LAYOUT_VERSION`）與 DFN2 的 stage／bridge，
見 [`integration_4ch_spatial_zh_TW.md`](integration_4ch_spatial_zh_TW.md) 與
[`pipeline_board_integration.md`](pipeline_board_integration.md)。

給整合者的兩條規則：

1. **大小永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用。
2. **升級 library 之後一定要重跑一次大小查詢**，並確認你的 pool 預算仍然足夠。
