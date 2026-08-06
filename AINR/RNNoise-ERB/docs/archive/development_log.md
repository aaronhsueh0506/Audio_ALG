# RNNoise-ERB 開發紀錄（歷史封存）

## 2026-07 — 修正 erb_bandborder() 最小 band 寬度保證（v5）

### 問題
- 追查「RNNoise-ERB vs DeepFilterNet2 兩套 ERB 公式是否相同」時,直接
  對照原作者 [Rikorose/DeepFilterNet `libDF/src/lib.rs`](https://github.com/Rikorose/DeepFilterNet/blob/main/libDF/src/lib.rs)
  的 `freq2erb`/`erb2freq`,確認跟本專案(以及 `aaronhsueh0506/
  DeepFilterNet-Keras` 的 `bandERB.ipynb`)完全一致 —— 這部分本來就對。
- 但原作者的 band-寬度演算法(`erb_fb()`,序列式逐一往後借)跟
  `bandERB.ipynb` 的 `ERBBand()`(「每隔一個 band 補 2」規則,只檢查
  `nb[i+2]-nb[i]>=2`,不檢查 `nb[i+1]-nb[i]`)不是同一套,兩者算出的 band
  寬度分佈不同。且原作者的 filterbank 本身是矩形不重疊(`df/modules.py`
  的 `erb_fb()`),跟本專案與 `bandERB.ipynb` 的三角重疊 filterbank 也不同
  ——這是設計選擇差異,决定保持三角形。
- 但覆盤 `ERBBand()` 的「每隔一個 band 補 2」規則時,發現它**不保證每個
  band 都 ≥2 bin**:在 sr=16000/n_fft=512/n_bands=22(本專案實際設定)下
  實測算出一個寬度僅 1 bin 的 band(`nfftborder` 相鄰兩個值只差 1)。

### 決策
- `erb_bandborder()`(`train.py`)、`gen_rnnoise_tables.c`、
  `test_rnnoise_tables.c`(獨立 drift-guard 複製)三邊同步改成嚴格
  greedy-forward 演算法:從 `ideal[i]`(跟原本一樣,ERB-rate linspace +
  `round((cutoff+bw/2)/bw)`)開始,若 `ideal[i] < prev_border +
  min_bins_per_band` 就往前推進到 `prev_border + min_bins_per_band`,從不
  後退。這個風格跟本檔案自己 `compute_erb_bands()`(停用中的 hybrid-band
  路徑用)一致,只是套用到 `erb_bandborder()` 的「N borders → N bands,
  頭尾兩欄 edge-doubling」慣例上。
- ERB-rate 公式(`freq2erb`/`erb2freq`,9.265/24.7 常數)、三角 filterbank
  構造(`compute_erb_matrix`,mode=0 forward 頭尾×2 / mode=1 inverse 乾淨
  partition-of-unity)維持不變——這兩者已經跟原作者/`bandERB.ipynb` 一致
  或是刻意的設計選擇,不在這次修正範圍。
- Feature version 改為 `log_erb_dfn_mean_cplx_unit_0_4k_v5`;input/output
  shape 不變,但 `erb_fwd`/`erb_inv` 矩陣數值改變,checkpoint-incompatible。

### 驗證
- 新增 regression test(`tests/test_feature_norm.py`
  `test_erb_bandborder_guarantees_min_bins_per_band`):對 4 組不同
  sr/n_fft/n_bands 組合逐一檢查每個 band 寬度 `>=2`,而非只抽查「每隔一
  個」。
- `make test`(tables drift-guard 兩層 + feature contract + 4096-frame C
  獨立參考 regression)、`make test-feature-python`(11 個 unittest + `Python`/`C`
  golden-vector parity)、`make test-loss-python` 全部重跑,PASS。
- `export_erb_matrix.py` 重跑確認 mode=1 partition-of-unity 仍是
  `max|rowsum-1|=0`、mode=0/mode=1 仍是頭尾 ×2 關係。

## 2026-07 — 拿掉 v3 的 erb_norm_clip/spec_clip，精確對齊 DeepFilterNet（v4）

### 問題
- v3 的雙路 normalization 公式已對齊原作 DeepFilterNet libDF
  (`band_mean_norm_erb`/`band_unit_norm`)，但額外保留了 v1 舊 broadband
  CMVN 沿用下來的 `erb_norm_clip=±5`/`spec_clip=±10`，README 明載為「部署
  數值安全界線」。
- 直接讀 upstream [Rikorose/DeepFilterNet `libDF/src/lib.rs`](https://github.com/Rikorose/DeepFilterNet/blob/main/libDF/src/lib.rs)
  確認 `band_mean_norm_erb`/`band_unit_norm` 兩個函式完全沒有 clip/clamp；
  DeepFilterNet 3 Python 訓練碼（`df/modules.py`、`deepfilternet3.py`）裡
  所有 `clamp`/`clip` 呼叫也都是 gain/mask/SNR 用途，沒有一處作用在這兩條
  feature 分支的輸出上。
- 本 repo 自己的 `AINR/DeepFilterNet2` port（`causal_ema_db_norm`/
  `causal_ema_mag_norm`/`extract_dfn2_features`）逐行比對後，`[feature]`
  的每個常數（tau/alpha/init/scale/eps）都跟 RNNoise-ERB 完全一致，**唯一
  差異就是這兩個 clip**——DFN2 這邊完全沒有它們。

### 決策
- Feature version 改為 `log_erb_dfn_mean_cplx_unit_0_4k_v4`；輸入 shape
  不變（ERB `[B,T,22]`、complex `[B,T,2,129]`），純粹是 normalization
  semantics 改變，checkpoint-incompatible。
- `train.py` 的 `normalize_log_erb`/`normalize_complex_spectrum` 拿掉
  `clip` 參數與 `.clamp(-clip, clip)`；`process.h`/`process.c` 同步拿掉
  `RNNOISE_ERB_NORM_CLIP`/`RNNOISE_SPEC_CLIP` 及對應的飽和判斷；
  `config.ini`、`export_onnx.py`、checkpoint contract
  (`require_checkpoint_feature_config`)、四個 Python 測試與
  `test_rnnoise_features.c` 全部同步移除。
- 這是刻意推翻 v3 當初的工程判斷（不是修 bug）：好處是兩路公式現在對
  upstream 是逐行忠實移植，沒有任何本專案自行加上的偏離；代價是拿掉了
  「冷啟動 EMA 未收斂或極端 transient 把 feature 值打到離譜範圍」這個
  安全網，需要重訓後實際觀察是否會出現數值不穩定。

### 驗證
- `grep` 確認整個元件 Python/C/config/測試裡不再有任何 `erb_norm_clip`/
  `spec_clip`/`RNNOISE_ERB_NORM_CLIP`/`RNNOISE_SPEC_CLIP` 的功能性引用
  （只剩解釋這個改動本身的註解）。
- **這是 checkpoint-incompatible 的架構/語意改動，需要重訓與重新匯出
  ONNX**；本輪只更新程式碼與測試，尚未重訓。

## 2026-07 — dataset_gen 單一來源與目錄清理

- `dataset_gen/gen_dataset.py` 每次產生一種指定 sample rate；
  `[signal] sr` 可由 `--sample-rate` 覆寫。16 kHz 供 RNNoise-ERB/GTCRN，
  48 kHz 供 DeepFilterNet2。
- generator 預設使用 OS 產生的新 seed；指定非負 `--seed` 時才固定可重現。
- 修正 RIR pre-delay：DRR target 的 dry path 會對齊 trimmed RIR direct peak。
- RIR full/target kernel 分別做 L2 normalization；noise SNR 改以 clean target
  RMS 為基準，與 DeepFilterNet `RandReverbSim` / `mix_audio_signal` 對齊。
- 刪除 RNNoise-ERB 內重複的 `dataset.py`、`gen_dataset.py`、
  `pack_dataset.py`；augmentation、指定-rate generation、resample 與
  packing 統一由 `../dataset_gen/` 維護。
- packed tensor loader 移到共用 `dataset_gen.packed_dataset`；RNNoise
  training 只保留模型與訓練責任。
- generation-only config sections 從模型 `config.ini` 移除，改用
  `../dataset_gen/config.ini`。
- 仍有效的 Python regression tests 收進 `tests/`；四個測試分別保護
  feature constants、normalization/model shapes、DF3 loss/pure-noise 與
  Python/C golden-vector parity，沒有可安全刪除的重複測試。

## 2026-07 — DeepFilterNet 3 MultiResSpecLoss-only

- 移除 direct ERB IRM、speech-active frame weight 與 `f_under`；訓練 total loss
  只剩單一 `MultiResSpecLoss`。
- 對齊 DeepFilterNet 3 production 設定：FFT 256/512/1024/2048、γ=0.3，
  compressed magnitude/complex MSE factor 各 500，各 resolution 加總不平均。
- 不移植 local-SNR loss：RNNoise-ERB 沒有 local-SNR/VAD output head。
- pure-noise 使用 clean waveform=0 的正常 MRSL target，沒有額外權重或
  clean-energy 除法；新增 finite loss/gradient regression test。
- Checkpoint 新增 `loss_version` 與 loss config gate，防止續訓舊 objective
  的 optimizer/scheduler state。
- 上游對齊來源：[Rikorose/DeepFilterNet `df/loss.py`](https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/loss.py)
  與 repository 內的 `models/DeepFilterNet3.zip/config.ini`。

## 2026-07 — DeepFilterNet-aligned dual preprocessing v3

- Feature version 改為 `log_erb_dfn_mean_cplx_unit_0_4k_v3`；雙路 input shape
  仍為 ERB `[B,T,22]` 與 complex `[B,T,2,129]`，不增加 absolute-level 維度。
- ERB 依 DeepFilterNet/Keras 公式改為每-band causal EMA mean norm：先更新
  mean，再輸出 `(erb_db - mean) / 40`。
- Complex 以原作者 Rikorose/DeepFilterNet `libDF::band_unit_norm` 為權威：
  每個 bin 以自己的 magnitude 更新 EMA，再做 `X[k] / sqrt(state[k])`。
  不採用 DeepFilterNet-Keras notebook 的 frame-level `np.linalg.norm()` 寫法。
- Checkpoint/config gate 新增 `win_len` 與兩路 v3 normalization 常數。
- 新增 Python/C golden-vector parity test，比對 STFT、兩路 feature 與
  normalization state。

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
