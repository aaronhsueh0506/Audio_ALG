# RNNoise-ERB 開發紀錄

## 2026-07 — dual absolute-ERB + complex spectrum v2

### 問題
- `log_erb_shared_online_cmvn_v1` 在實際穩態 noise + 約 0 dB SNR 仍出現
  speech 被全壓到 0，單靠 22 個 coarse ERB 不足以區分語音細頻率/週期性。
- DeepFilterNet 的 ERB normalization 並不是單獨使用；它還有 complex
  spectrogram branch。只複製 ERB norm 會遺失主要訊息。
- 舊 waveform-only loss 沒有直接約束 gain，容易走向過度抑制。

### 決策
- Feature version 改為 `log_erb_abs_cplx_0_4k_v2`，舊 checkpoint 強制拒絕。
- ERB branch 改為 absolute log-energy 固定 affine scaling，完全不做 temporal EMA。
- 新增 0–4 kHz、129-bin complex branch；只對 magnitude 做 per-bin causal EMA
  unit norm，real/imag 保留細頻率與相位資訊。
- 模型新增 complex frequency encoder/fusion，但 runtime 仍只輸出 ERB gains。
- 依實際既有 checkpoint 容量使用 3 層 GRU(128)，不增加 VAD head。
- Loss 改為 waveform multi-resolution STFT + direct ERB IRM，並提高
  speech-active frame 的 gain loss 權重。
- 不同 shuffled WAV 間重置 EMA/GRU state；單一 WAV 內仍依 frame causal 更新。
- `denoise.py --dump-debug` 可輸出 feature/raw gain/post gain；
  `--pf-beta` 預設關閉。

### 驗證
- C/Python contract 固定 v2 常數與 shape。
- 4096-frame stationary regression 要求 ERB/complex 兩路皆不得歸零。
- 這是 checkpoint-incompatible architecture change，需重訓與重新匯出 ONNX。

## 2026-07 — shared broadband runtime normalization（v1，已被 v2 取代）

### 問題
- 單一 ERB input 沿用 DeepFilterNet per-band temporal EMA 時，每個 band 都會被
  自己的 running mean 扣除；穩態訊號的平均頻譜與絕對 level 因此趨近全零。
- DeepFilterNet 尚有 complex spectrum branch 保留資訊，本模型沒有，兩者不能
  假設等價。
- 舊訓練每個 3 秒 clip 重設 feature state，但 runtime stream 長時間持續更新，
  造成 feature distribution mismatch。

### 決策
- Feature version 改為 `log_erb_shared_online_cmvn_v1`。
- 保持 22 維輸入；所有 ERB bands 共用一組 broadband scalar mean/variance。
- Feature 使用更新前 state，之後才用當前 frame 的 band-average 更新 state。
- 當時訓練讓 normalizer state 跨 batch clips 延續；v2 已修正為不同 WAV 必須重置。
- Checkpoint 保存並驗證 feature version/tau/init/floor/clip；舊 checkpoint 必須重訓。
- Python/C 預設：tau=10s、mean=-75dB、std init=20dB、std floor=6dB、clip=5。

### 驗證
- `make test-features`：C recurrence reference + 4096-frame stationary regression。
- `make test-feature-python`：previous-state、chunk equivalence、stationary envelope、
  scalar-state shape。

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
