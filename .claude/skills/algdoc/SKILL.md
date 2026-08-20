---
name: algdoc
description: 演算法改動後同步文件與使用者說明 — 更新 Audio_ALG/docs/html 的 HTML 文件站（API 用法、程式碼流程 block diagram、訊號泳道圖、模型 I/O 標記），並校正 md 文件的過期敘述
---

# algdoc — 演算法改動後的文件同步

每次演算法/介面改動後執行。目標：文件站與 md 文件永遠反映**當前程式碼**，
不留過期敘述。

## 存放結構（2026-08-14 起）

文件站**拆散在各自 repo**，不再全部集中在 `Audio_ALG/docs/html/`：

- **元件頁在各自 repo 的 `docs/html/`**：
  - `AEC/docs/html/aec.html`
  - `NR/docs/html/nr.html`
  - `audio_common/docs/html/audio_common.html`
  - 新元件（新的 AEC/NR/audio_common 子模組或子系統）比照辦理：建在該
    元件自己的 repo `docs/html/` 下，不要放進 Audio_ALG。
- **`Audio_ALG/docs/html/` 只留自身內容**：`index.html`（總覽+導覽）、
  `conventions.html`、`onnx_prepost.html`、四個 `pipeline_*.html`、四個
  `ainr_*.html`、四個 `aiaec_*.html`（Audio_ALG 自有的 pipelines/AINR/AIAEC
  才留在這裡；新的 pipeline/AINR/AIAEC 頁面也建在這裡）。

**連結規約（SE 樹 sibling 相對路徑，四個 repo 同層 `SE/{AEC,NR,
audio_common,Audio_ALG}`）**：

- 從 `Audio_ALG/docs/html/*.html` 連到元件頁：
  `../../../AEC/docs/html/aec.html`、`../../../NR/docs/html/nr.html`、
  `../../../audio_common/docs/html/audio_common.html`。
- 從元件頁（`aec.html`/`nr.html`/`audio_common.html`）連回 Audio_ALG index：
  `../../../Audio_ALG/docs/html/index.html`；元件頁彼此互連同理用
  `../../../<repo>/docs/html/<page>.html`；連到自己則用檔名(`aec.html`
  這種同目錄相對連結)。
- 這是**固定慣例**，不要每次改動時重新發明——新增元件頁一律套用上述模式。

## 步驟

1. **找出受影響元件**：`git diff` 未提交改動 + 最近的 commits（AEC、NR、
   audio_common、Audio_ALG 四個 repo 都要看）。對應到下方頁面清單。
2. **逐頁更新**受影響的頁面（規格見下）；元件頁去對應 repo 的
   `docs/html/` 找，Audio_ALG 自身頁面去 `Audio_ALG/docs/html/` 找。新元件
   （新 pipeline、新 model、新 seam API）→ 新頁面 + 更新 `index.html` 與
   `conventions.html` 的對照表（連結套用上面的 sibling 相對路徑規約）。
3. **校正 md 文件**：受影響的 README、`docs/*.md`（尤其
   `align_ulcnet_embedded_streaming_design_zh_TW.md`、
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

**元件頁（各自 repo 的 `docs/html/`，見上面存放結構一節）**：
`AEC/docs/html/aec.html`、`NR/docs/html/nr.html`、
`audio_common/docs/html/audio_common.html`

**`Audio_ALG/docs/html/`（本站；index/conventions/onnx_prepost +
Audio_ALG 自有 pipelines/AINR/AIAEC）**：

`index.html`（總覽+導覽）、`conventions.html`（全鏈 grid/window/framing/
延遲對照）、`onnx_prepost.html`（ONNX 邊界與前後處理現況/缺口表）、
`pipeline_mono.html`、`pipeline_4ch.html`、
`pipeline_ulcnet_mono.html`、`pipeline_ulcnet_4ch.html`、
`ainr_dfn2.html`、`ainr_gtcrn.html`、`ainr_rnnoise_erb.html`、
`aiaec_align_ulcnet.html`、`aiaec_align_cruse.html`、`aiaec_deepvqe_s.html`、
`aiaec_cagcrn.html`

## 硬規則

- 廠商/平台名稱一律不得出現（用「目標平台」）；不得出現絕對個人路徑。
- 中文敘述 + 英文識別字；數字一律可追溯（測試輸出或程式碼常數）。
- 不確定的事實寫「待量測/待確認」，不要編。
- 文件站只描述**已存在**的程式碼；規劃中的東西放 md 設計文件，不進 HTML。
