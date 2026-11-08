---
name: algdoc
description: 核對音訊程式改動，更新四個專案的使用者手冊及離線導覽。
---

# 音訊手冊維護

以公開標頭、實際 caller、Makefile 和模型匯出器為準。先確認改動影響的使用者動作，再更新對應的段落、範例或圖。

## 文件入口

- 每個可獨立交付的專案從本庫的 `docs/html/index.html` 開始。Git 根目錄 README 提供最短建置步驟及手冊連結。
- Audio_ALG 首頁維持建置、pipeline、模型、底層函式庫四項任務。新增模型時更新 `docs/html/models/index.html`，新增 pipeline 時更新 `docs/html/pipelines/index.html`。
- 各模型首頁在 `models/<模型>/index.html`，各路由在 `pipelines/{mono,4ch}/<後級>/index.html`。細部 grid、tensor、state、時序和參數表依實際需要放在主題的 `reference/`。同一模型共用的頻域契約只維護一份。
- 模型目錄的 Markdown 負責訓練、匯出或完整可編譯的整合範例。離線手冊說明訊號來源、初始化和每 hop 的呼叫順序，連至對應範例。共用文件站 CSS 用 `docs/html/manual.css`，正文與導覽都可直接從本機檔案開啟。
- 共用 DSP、spatial、AEC、NR 的詳情由各自 repo 負責。Audio_ALG 的 `libraries/index.html` 提供各庫的 sibling 連結入口。原始碼目錄使用現行網址。需支援舊網址時，在 `docs/maintenance/offline/aliases.tsv` 登錄映射，由離線相容交付工具產生轉址。

## 內容和驗證

1. 先讀產品路徑，確認由誰配置 pool、作 FFT、維護 state、執行推論，以及錯誤和 reset 後的第一個 hop 如何輸出。選能解釋這個邊界的一個範例或圖即可。
2. 分清單模型 PCM stream、頻域 core 和前後還有 AEC／beamformer 的 pipeline。DFN2 單幀與多幀要寫明觸發推論的 hop 和頻譜、PCM 各自的延遲。裝置持有 state 的範例須涵蓋 generation 同步與推論失敗後回填量化零。
3. 四聲道 MCAB 的四條 AEC lane 和共享 delay 由 caller 提供，beamformer 與 RES 在應用層。Mono 的 RES 增益由 AEC 交出。命名、grid 白名單、layout／carve 版號以所在標頭為準。時序數字標出對應 grid、checkpoint 與 lookahead。
4. 保留使用者需要的故障處理、有效限制、授權與設定。開發交接、逐輪修改紀錄、測試清單和單次跑分放在維護端。部署頁簡記已驗證的環境，不將 host 替身或 ONNX CPU 推算成實板驗證。
5. 更新後核對所有本機相對連結、錨點及移動過的舊網址。版號和文件中的範例若受既有檢查讀取，讓檢查指到新正文並實跑。檢查首頁能走到受影響的模型或路由，且單庫離線時不依賴其他 repo 提供樣式或導覽。
