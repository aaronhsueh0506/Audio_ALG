# RNNoise-ERB 開發紀錄

## 2026-04 — feature/wav-perceptual-loss branch

### 動機
- 訓練 100hr 後 VCTK PESQ 輸給原始 RNNoise paper
- 推測原因: IRM-domain MSE 是 PESQ 的 proxy，不是直接 optimize 音訊品質
- DeepFilterNet 不需手動調 over/under weight，是因為它的 loss 本來就在 signal domain

### 設計決策
- **gen_dataset 改存 WAV pair** (`noisy/000000.wav` + `clean/000000.wav`)，不存 `.pt` features
  - 理由: 後續可能訓練 DeepFilterNet、Conformer 等其他 model，WAV 一份資料能重複用
- **訓練改成 on-the-fly STFT → ERB → model → ISTFT → perceptual loss**
- **Loss = multi-resolution STFT loss** (DeepFilterNet 風格: spectral convergence + log magnitude，FFT sizes = 512/256/1024)
- Model 輸出維持 ERB gains，沒改架構（這是第一階段）

### 代碼結構
- `dataset.py`:
  - `DNS4Dataset(cfg, return_raw=True)` → 跳過 STFT，回傳 raw `(noisy, clean)` 給 gen_dataset 用
  - `WavPairDataset(data_dir)` → 訓練時讀 WAV pair
- `gen_dataset.py`: 存 WAV，meta.json，resume 靠 count `noisy/*.wav`
- `train.py`:
  - `extract_erb_features` / `apply_erb_gains_batch` / `multi_res_stft_loss` 三個新 helper
  - `--wav-data` flag 切 perceptual loss mode
  - 舊 `--precomputed` (.pt) 和 online mode 仍保留
- `config.ini`: 新增 `[perceptual_loss]` section
- 所有 entry-point 加 `sys.path.insert` 以支援任意目錄執行

### 待辦
- 見 `TODO.md`

### 比較曾試過但放棄的方向
- `feature/audio-domain-loss` (已刪除): 混合 loss (perceptual + IRM 正則化)，仍用 online DNS4Dataset，沒存 WAV
  - 放棄理由: 未來訓其他 model 還是要重 gen 一次資料；想一次到位
