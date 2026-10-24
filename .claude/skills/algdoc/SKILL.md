---
name: algdoc
description: 演算法改動後同步文件與使用者說明 — 更新 Audio_ALG/docs/html 的 HTML 文件站（API 用法、程式碼流程 block diagram、訊號泳道圖、模型 I/O 標記），並校正 md 文件的過期敘述
---

# algdoc — 演算法改動後的文件同步

每次演算法/介面改動後執行。目標：文件站與 md 文件永遠反映**當前程式碼**，
不留過期敘述。

## 存放結構

文件站**拆散在各自 repo**，不集中在 `Audio_ALG/docs/html/`：

- **元件頁在各自 repo 的 `docs/html/`**：
  - `AEC/docs/html/aec.html`
  - `NR/docs/html/nr.html`
  - `audiolib/common/docs/html/audio_common.html`
  - `audiolib/docs/html/spatial_chain.html`（audiolib 的空間鏈：VAD／DOA／BF，
    與 `common/` 不同層，路徑不要混用）
  - 新元件（新的 AEC/NR/audiolib/common 子模組或子系統）比照辦理：建在該
    元件自己的 repo `docs/html/` 下，不要放進 Audio_ALG。
- **`Audio_ALG/docs/html/` 只留自身內容**：`index.html`（總覽+導覽）、
  `conventions.html`（含 layout／carve 版號一覽）與 `conventions_latency.html`、
  `onnx_prepost.html`、`pipeline_*.html`、`ainr_*.html`、`aiaec_*.html` 與其
  `_usage`／`_sequence`／`_params`／`_detail` 附屬頁（Audio_ALG 自有的
  pipelines/AINR/AIAEC 才留在這裡；新的 pipeline/AINR/AIAEC 頁面也建在這裡）。
  整個 repo 的架構概觀頁 `architecture.html`、`architecture_aec.html`、
  `architecture_detail.html` 在 `Audio_ALG/docs/`（`html/` 的上一層）。

**連結規約（SE 樹 sibling 相對路徑，四個 repo 同層 `SE/{AEC,NR,
audiolib,Audio_ALG}`）**：

- 從 `Audio_ALG/docs/html/*.html` 連到元件頁：
  `../../../AEC/docs/html/aec.html`、`../../../NR/docs/html/nr.html`、
  `../../../audiolib/common/docs/html/audio_common.html`、
  `../../../audiolib/docs/html/spatial_chain.html`；連到架構概觀頁：
  `../architecture.html`。
- 從 `audiolib/common/docs/html/*.html` 連到空間鏈頁：
  `../../../docs/html/spatial_chain.html`；從 `audiolib/docs/html/*.html` 連到 common
  頁：`../../common/docs/html/audio_common.html`。
- 從元件頁（`aec.html`/`nr.html`/`audio_common.html`）連回 Audio_ALG index：
  `../../../Audio_ALG/docs/html/index.html`；元件頁彼此互連同理用
  `../../../<repo>/docs/html/<page>.html`；連到自己則用檔名(`aec.html`
  這種同目錄相對連結)。
- 這是**固定慣例**，不要每次改動時重新發明——新增元件頁一律套用上述模式。

## 步驟

1. **找出受影響元件**：`git diff` 未提交改動 + 最近的 commits（AEC、NR、
   audiolib、Audio_ALG 四個 repo 都要看）。對應到下方頁面清單。
2. **逐頁更新**受影響的頁面（規格見下）；元件頁去對應 repo 的
   `docs/html/` 找，Audio_ALG 自身頁面去 `Audio_ALG/docs/html/` 找。新元件
   （新 pipeline、新 model、新 seam API）→ 新頁面 + 更新 `index.html` 與
   `conventions.html` 的對照表（連結套用上面的 sibling 相對路徑規約）。
3. **校正 md 文件**：受影響的 README、`docs/*.md`（尤其
   `align_ulcnet_embedded_streaming_design_zh_TW.md`（索引檔，第 6–11 章在同名 `_01_`／`_02_` part 檔）、
   `aiaec_streaming_readiness_zh_TW.md`、AEC 的 `c_user_manual_zh_TW.md` 與
   `nn_integration_interface.md`）。已被實作推翻的「現況」敘述必須改寫成
   「實作前查證紀錄」或直接更新，並標日期。
4. **驗證**：頁面內的每個 file:line / API 名稱 / 常數，用 grep 對當前程式
   碼確認過再寫；模型數字（參數量、state 大小、延遲）優先引用測試實測值。
   跨 repo 相對連結另外用檔案系統存在性逐一檢查（考慮該頁自身的新位置）。

## HTML 文件站規格（docs/html/）

- **自包含**：純 HTML+inline CSS，零外部資源（無 CDN、無 JS 依賴）。
- **每頁小**：目標 < 30 KB；內容多就拆頁再互連。
- **內連**：相對連結互連；每頁頂部有回 Audio_ALG `index.html` 的導覽列 +
  同類頁面連結（元件頁跨 repo 用上面「存放結構」一節的 sibling 相對路徑）；
  原始碼引用寫成 `repo/path/file.c:line` 文字（不做 file:// 連結）。
- **每頁固定結構**：
  1. 概要（這個元件是什麼、在整鏈的位置）
  2. API 用法（呼叫順序 + 最小 C/Python 範例碼）
  3. **Block diagram**（程式碼流程圖：用 `<pre>` ASCII，對應實際函式呼叫
     順序，節點旁標函式名）
  4. **訊號泳道圖**（swimlane：橫軸 hop/時間，泳道 = caller / C 前處理 /
     NPU(model) / C 後處理…，標出每 hop 誰產生什麼、延遲幾個 hop）
  5. **I/O 表**（模型頁必備）：每個輸入/輸出 tensor 的名稱、shape、dtype、
     一次 invocation 要幾個 frame、對應的時域樣本數
  6. **State 表**（串流元件必備）：跨呼叫保存的每個 state 的 shape/大小/
     誰負責保存（C struct / NPU runtime / host）
  7. 延遲與 warm-up 行為
  8. 相關檔案與測試的路徑清單
- **模型頁額外要求**：標明 ONNX 邊界——graph 內做什麼、graph 外（C 前後
  處理）做什麼、哪些 state 由 runtime 在 invocation 間搬運；若該模型還沒
  有 C 前後處理，在 `onnx_prepost.html` 的缺口表記一列。

## 頁面清單（現行；新增元件時擴充）

新頁面必須同時列進 `Audio_ALG/docs/html/index.html` 的導覽列與頁面索引表、
`Audio_ALG/docs/README.md`，以及本清單；各頁導覽列要與同類頁一致。

**元件頁（各自 repo 的 `docs/html/`，見上面存放結構一節）**：

- AEC（`AEC/docs/html/`）：`aec.html`、`aec_detail.html`（API 細節）、
  `aec_usage.html`（呼叫順序與範例）、`aec_sequence.html`（時序圖）、
  `aec_params.html`（外部參數）、`aec_params_internal.html`（內部參數）、
  `aec_c_api.html`、`aec_c_api_params.html`、`aec_res.html`（RES 級獨立介面）
- NR（`NR/docs/html/`）：`nr.html`、`nr_c_api.html`、`nr_c_api_params.html`
- audiolib common（`audiolib/common/docs/html/`）：`audio_common.html`、
  `audio_common_detail.html`、`audio_common_params.html`、`audio_common_c_api.html`
- audiolib 空間鏈（`audiolib/docs/html/`）：`spatial_chain.html`、`doa_srp.html`、
  `bf.html`、`vad.html`

**`Audio_ALG/docs/html/`（本站；index/conventions/onnx_prepost +
Audio_ALG 自有 pipelines/AINR/AIAEC）**：

- 總覽：`index.html`（總覽+導覽）、`conventions.html`（全鏈 grid/window/framing 與
  layout／carve 版號對照）、`conventions_latency.html`（全鏈延遲預算表）、
  `onnx_prepost.html`（ONNX 邊界與前後處理現況/缺口表）
- 整合：`integration_example.html`、`integration_sequence.html`（整合時序圖）、
  `integration_example_detail.html`（release build 的 DEBUG 觀測面）
- mono：`pipeline_mono.html`、`pipeline_mono_params.html`、`pipeline_mono_detail.html`
- 4ch：`pipeline_4ch.html`、`pipeline_4ch_params.html`、
  `pipeline_4ch_detail.html`（4ch 的執行期強度控制與逐階段耗時）
- ULCNet pipeline：`pipeline_ulcnet_mono.html`、`pipeline_ulcnet_4ch.html`、
  `pipeline_ulcnet_detail.html`（共用 adapter 與 reprime 細節）、
  `pipeline_ulcnet_usage.html`、`pipeline_ulcnet_sequence.html`、`pipeline_ulcnet_params.html`
- DFN2 pipeline：`pipeline_dfn_mono.html`、`pipeline_dfn_4ch.html`、
  `pipeline_dfn_detail.html`（入口語意、rate bridge 與測試內容）、
  `pipeline_dfn_usage.html`、`pipeline_dfn_sequence.html`、`pipeline_dfn_params.html`
- AINR：`ainr_dfn2.html`、`ainr_dfn2_detail.html`（pool、callback 邊界、建置）、
  `ainr_dfn2_usage.html`、`ainr_dfn3.html`、`ainr_dfn3_usage.html`、`ainr_gtcrn.html`、
  `ainr_gtcrn_usage.html`、`ainr_rnnoise_erb.html`、`ainr_rnnoise_erb_usage.html`
- AIAEC：`aiaec_align_ulcnet.html`、`aiaec_align_ulcnet_detail.html`（far branch、PTQ、
  layout 重匯出）、`aiaec_align_ulcnet_usage.html`、`aiaec_align_cruse.html`、
  `aiaec_align_cruse_usage.html`、`aiaec_deepvqe_s.html`、`aiaec_deepvqe_s_usage.html`、
  `aiaec_cagcrn.html`、`aiaec_cagcrn_usage.html`

**`Audio_ALG/docs/`**：`architecture.html`、`architecture_aec.html`、`architecture_detail.html`
（整個 repo 的架構概觀；不在 `html/` 內）。

## 歸屬與命名要點（寫頁面前先對照程式碼）

- **RES 歸屬**：mono 的 RES 增益在 `aec_process_context()` 內算出，經 `se_frame_from_aec()` 以
  `frame.g_res_ctx` 交出（強度 `aec_set_preset()`）；4ch 單元 `AudioPipeline4Ch` 只做到 `fuse()`
  （`g_res_ctx` 為 NULL），要 RES 的應用自己用 `aec_res_init()` 建立 `AecRes`、每 hop 呼叫
  `se_res_stage_gain()`（強度 `aec_res_set_preset()`）。不要寫成「單元擁有 RES」，也不要把
  「所有 stage 分開呼叫」套用到 mono。
- **波束器**一律稱 `bf`（`audiolib/bf`，RLS；結構為 GSC 形式，僅在 prose 說明結構時提及）；
  不要在表格或標籤裡用 GSC 當元件名。
- **版號**只認 header 的 `#define`；文件寫出 `MACRO = N` 時由
  `pipelines/tests/test_doc_layout_versions.py` 核對（layout 與 carve 兩張表），新增帶版號的
  class 時把它的 header 加進該測試的來源表。

## 硬規則

- 廠商/平台名稱一律不得出現（用「目標平台」）；不得出現絕對個人路徑。
- 中文敘述 + 英文識別字；數字一律可追溯（測試輸出或程式碼常數）。
- 不確定的事實寫「待量測/待確認」，不要編。
- 文件站只描述**已存在**的程式碼；規劃中的東西放 md 設計文件，不進 HTML。
