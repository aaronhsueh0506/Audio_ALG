# RNNoise-ERB

以 RNNoise v0.2 的 Conv+GRU 方向為基礎的本地噪音抑制模型，使用
mean-normalized log-ERB 與低頻 complex spectrum 雙路特徵；並非官方 v0.2
或 DeepFilterNet 的逐層相容移植。

## 架構

- **ERB 輸入**: 22 個 log-energy bands，做每-band causal EMA mean norm
- **Complex 輸入**: 0–4 kHz 的 129 個 real/imag bins，做每-bin magnitude EMA unit norm
- **模型**: ERB temporal conv + complex frequency encoder/temporal conv → fusion → 3 層 GRU(128)
- **輸出**: 22 個 gain masks [0, 1]
- **參數量**: 376,254（無 VAD head）
- **Latency**: 16ms lookahead（`lookahead_frames=1`，gain 對應 3-frame window 中間幀）

### Feature preprocessing

目前 feature contract 為 `log_erb_dfn_mean_cplx_unit_0_4k_v3`：

```text
normalized complex STFT
├─ power → triangular ERB → 10*log10(E+1e-10)
│          → per-band EMA mean → (dB - mean) / 40 → clip
└─ bins 0..4 kHz → per-bin |X[k]| EMA
                    → real/imag / sqrt(EMA+eps) → clip
```

ERB 路徑對穩態訊號會逐步趨近 0，這是原作 DeepFilterNet 的預期行為；
辨識穩態噪音的細頻率結構、相位及部分 level 資訊由 complex 路徑保留。
`X[k] / sqrt(EMA(|X[k]|))` 不是完全 gain-invariant；穩態下輸入振幅放大 `a` 倍，
complex feature 約放大 `sqrt(a)` 倍，因此不需要額外 absolute-level input。
同一 stream 分 chunk 處理時必須同時傳遞 ERB EMA、complex EMA 與 GRU state；
不同 WAV 之間必須重置。
兩路主公式、state 初始值與先更新再輸出的順序對齊原作 `libDF`；
本專案額外保留 ERB `±5` 與 complex `±10` clip 作為部署數值安全界線。
公式來源：[Rikorose/DeepFilterNet `libDF/src/lib.rs`](https://github.com/Rikorose/DeepFilterNet/blob/main/libDF/src/lib.rs)。

Feature 常數在 `config.ini [feature]`，C 部署常數固定於 `process.h`。
修改後需同步兩邊並重訓。

## 環境安裝

```bash
# 建議使用 Python 3.9+
pip install -r requirements.txt
```

依賴套件：
- `torch` >= 1.13
- `torchaudio` >= 0.13
- `numpy`
- `tqdm`

匯出 ONNX 額外需要（訓練不需要）：
```bash
pip install onnx onnxoptimizer onnxruntime
```

## 資料集準備 (DNS4)

本專案使用 [DNS Challenge 4](https://github.com/microsoft/DNS-Challenge) 資料集。下載後目錄結構如下：

```
datasets_fullband/
├── clean_fullband/              ← config.ini 的 speech_dir
│   ├── emotional_speech/
│   ├── french_speech/
│   ├── german_speech/
│   ├── italian_speech/
│   ├── read_speech/
│   ├── russian_speech/
│   ├── spanish_speech/
│   ├── vctk_wav48_silence_trimmed/
│   └── ...
├── noise_fullband/              ← config.ini 的 noise_dir
│   ├── audioset/
│   ├── freesound/
│   └── ...
└── impulse_responses/           ← config.ini 的 rir_dir (optional)
    ├── SLR26/
    ├── SLR28/
    └── ...
```

### 注意事項

- 所有音檔必須是 **`.wav` 格式**（程式只掃描 `*.wav`）
- 音檔可以是任意 sample rate，程式會自動 resample 到 16kHz
- **RIR 為 optional**：若不使用，可將 `rir_dir` 設為空字串或刪除該行。建議使用以提升 dereverberation 效果
- RIR 會自動用 Schroeder 積分法估算 RT60，只保留 `rt60_min` ~ `rt60_max` (預設 0.1s ~ 1.3s) 範圍內的 RIR
- **RIR RT60 快取**：首次掃描後會自動存成 `.rir_cache_*.json`，後續相同設定直接讀取，大幅加速初始化
- 不一定要用 DNS4，**任何符合上述結構的語音/噪音資料集都可以**（只要是 wav 檔放在對應目錄下）

## 設定 config.ini

訓練前需要修改 `config.ini` 中的資料路徑：

```ini
[paths]
speech_dir = /your/path/to/datasets_fullband/clean_fullband
noise_dir = /your/path/to/datasets_fullband/noise_fullband
rir_dir = /your/path/to/datasets_fullband/impulse_responses
output_dir = ./output
```

其他常用參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `[training] epochs` | 100 | 訓練 epoch 數 |
| `[training] batch_size` | 128 | Batch size |
| `[training] lr` | 3e-4 | Learning rate |
| `[training] device` | cuda | 訓練裝置 (`cuda` 或 `cpu`) |
| `[training] epoch_size` | 0 | 每 epoch sample 上限；0 表示使用全部資料 |
| `[audio] segment_sec` | 3.0 | 每筆訓練音檔長度 (秒) |
| `[rir] p_rir` | 0.1 | 套用 RIR 的機率 |
| `[feature] erb_norm_tau_sec` | 1.0 | ERB per-band mean EMA 時間常數 |
| `[feature] erb_norm_init_lo_db` | -60.0 | 最低 ERB band EMA 初值 |
| `[feature] erb_norm_init_hi_db` | -90.0 | 最高 ERB band EMA 初值 |
| `[feature] erb_norm_scale_db` | 40.0 | mean-subtracted ERB 縮放尺度 |
| `[feature] spec_max_hz` | 4000 | complex branch 頻率上限 |
| `[feature] spec_norm_tau_sec` | 1.0 | complex per-bin magnitude EMA 時間常數 |

改動任何 feature 常數都需要同步修改 `process.h` 並重新訓練。Checkpoint 會保存
`feature_version` 與所有 normalization 常數；train resume、denoise 和 ONNX export
都會拒絕缺少版本的舊 checkpoint，或拒絕與 runtime config 不一致的 checkpoint。
匯出的 ONNX model metadata 也會帶上相同 feature contract，供部署端檢查。

> v1/legacy ERB-only 與 v2 absolute-ERB checkpoint/ONNX 都無法沿用。v3 雖然
> input shape 不變，normalization semantics 已改變，必須重訓後重新匯出。

## 訓練

目前訓練只接受 `pack_dataset.py` 產生的 raw noisy/clean WAV tensor；STFT、
log-ERB 與 runtime normalization 會在訓練時即時計算。完整流程：

```bash
# Step 1: 產生 2-channel WAV pairs（ch0=noisy, ch1=clean）
python3 gen_dataset.py --config config.ini --output data --hours 25 --workers 4

# Step 2: 打包，避免訓練時逐 WAV I/O
python3 pack_dataset.py --input data/pairs --output data/packed.pt --dtype float32

# Step 3: 訓練
python3 train.py --config config.ini --packed-data data/packed.pt

# 可選：指定 GPU、降低 RAM、或載入同一目錄下多個 packed files
python3 train.py --config config.ini --packed-dir data/packed_shards --gpu 0 --mmap
```

`gen_dataset.py` 常用參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--hours` | 8.3 | 目標音檔總時數（自動 round 到最近整數倍 epoch） |
| `--output` | `data` | 輸出目錄 |
| `--workers` | 4 | DataLoader workers；0 表示單程序 |
| `--resume` | false | 從 `pairs/` 最大編號後繼續寫入 |
| `--seed` | 42 | 隨機種子（-1 關閉） |

### 從斷點續訓

訓練中途中斷後，可以從最後的 checkpoint 繼續：

```bash
python3 train.py --config config.ini --packed-data data/packed.pt \
    --resume output/rnnoise_epoch5.pth
```

續訓會恢復：model weights、optimizer 狀態、scheduler 狀態、epoch 計數、best_val_loss。

### 訓練輸出

訓練結果儲存在 `output/` 目錄：

```
output/
├── rnnoise_epoch1.pth      # 每 epoch 的 checkpoint
├── rnnoise_epoch2.pth
├── ...
└── rnnoise_best.pth        # 最佳 validation loss 的模型
```

## 推論 (Denoise)

對單一音檔進行降噪：

```bash
python3 denoise.py --config config.ini --model output/rnnoise_best.pth \
                   --input noisy.wav --output clean.wav
```

`--pf-beta` 預設為 0（關閉），避免在還沒驗證 raw gain 前再加劇抑制。
若要定位「全壓成 0」是 feature、model 或 post-filter 造成：

```bash
python3 denoise.py --config config.ini --model output/rnnoise_best.pth \
  --input failing.wav --output debug.wav --dump-debug debug/failing.npz
```

dump 含 `erb_db`、兩路 features、`raw_gains`、`post_gains`
與輸入/輸出 waveform。這是 debug 用；checkpoint version gate 則是部署防呆。

## ONNX 匯出

將訓練好的模型匯出為 ONNX 格式（用於部署）：

```bash
# 匯出
python3 export_onnx.py --config config.ini --model output/rnnoise_best.pth \
                       --output output/rnnoise.onnx

# 匯出 + 驗證 (比較 PyTorch 與 ONNX 輸出)
python3 export_onnx.py --config config.ini --model output/rnnoise_best.pth \
                       --output output/rnnoise.onnx --verify
```

Streaming ONNX 輸入為 `erb_input[1,3,22]`、`spec_input[1,3,2,129]`
與三組 GRU hidden state；輸出 gains 與更新後 hidden state。

## ERB 矩陣匯出

匯出 ERB 轉換矩陣（C 部署用）：

```bash
# 匯出所有格式 (npy + C header)
python3 export_erb_matrix.py --config config.ini --format all
```

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `train.py` | 模型定義與 packed raw-WAV tensor 訓練迴圈 |
| `gen_dataset.py` | 離線產生 2-channel noisy/clean WAV pairs |
| `pack_dataset.py` | 將 WAV pairs 打包成訓練用 `.pt` tensor |
| `dataset.py` | DNS4 augmentation 與 packed/WAV dataset readers |
| `denoise.py` | 推論腳本（單檔降噪） |
| `export_onnx.py` | ONNX 匯出（streaming 推論格式） |
| `export_erb_matrix.py` | ERB 矩陣匯出（npy / C header） |
| `config.ini` | 所有超參數設定 |
| `process.c` / `process.h` | C 前後處理實作（嵌入式部署用） |
| `requirements.txt` | Python 依賴套件 |

## 表格 drift-guard(C 部署表格的兩層契約)

`rnnoise_tables_gen.h` 的編譯期常數表(ERB fwd/inv、nfftborder、Hann window)
有兩層 drift 契約(round-3 審查 B09;round-4 P2-4 接進 make):

```bash
make test-tables   # 兩層都建置+執行,任一 FAIL 即非零退出
```

- **Layer 1(canonical,預設)**:獨立重算 vs header 逐 byte `memcmp`——
  本機/pinned CI 工具鏈的 bit-exact 契約。
- **Layer 2(portable,`-DRNN_TABLES_PORTABLE`)**:數學性質檢查(有限值、
  nfftborder 單調且端點釘死、erb_inv partition-of-unity、erb_fwd 端帶 2x
  關係、Hann 落在 [0,1] 且滿足公式對稱)+ 實測定界的 ULP 門檻——
  **recompute-vs-table 為 256 ULP、Hann 鏡像對稱為 4096 ULP(實測 434)**,
  兩者是不同檢查的不同門檻,皆為 garbage-detector 而非位元契約。

## Feature tests

```bash
make test                 # tables + config/Python/C constants + C feature-state contract
make test-feature-python  # 需 torch training environment
```

C test 會以獨立 recurrence 對照 `rnnoise_compute_features()`，並讓固定非平坦
spectrum 運行 4096 frames，確認 ERB mean norm 收旂為 0，complex 路徑仍非零。
Python test 另驗證兩路 state/chunk equivalence、ERB 穩態收旂、complex 穩態可觀測性、
雙路 shape 與 model forward contract；`test_python_c_features.py` 以 golden vectors 對照
Python/C 的 STFT、雙路特徵與最終 normalization state。
