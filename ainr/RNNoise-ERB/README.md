# RNNoise-ERB

基於 RNNoise v0.2 架構的噪音抑制模型，使用 ERB (Equivalent Rectangular Bandwidth) 頻帶作為特徵。

## 架構

- **輸入**: 22 個 ERB band 的 normalized log energy (每 frame 16ms)
- **模型**: Conv1d(k=3, causal) + Conv1d(k=1) → 3 層 GRU(128) → concat → Dense → sigmoid
- **輸出**: 22 個 band 的 gain mask [0, 1]
- **參數量**: ~350K
- **Latency**: 16ms (1 hop, 0 lookahead — causal Conv1d)

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
| `[training] batch_size` | 32 | Batch size |
| `[training] lr` | 1e-3 | Learning rate |
| `[training] device` | cuda | 訓練裝置 (`cuda` 或 `cpu`) |
| `[training] epoch_size` | 10000 | 每 epoch 隨機抽取的 sample 數 (10000 × 3s = 8.3hr/epoch) |
| `[audio] segment_sec` | 3.0 | 每筆訓練音檔長度 (秒) |
| `[rir] p_rir` | 0.8 | 套用 RIR 的機率 |

## 訓練

### 基本訓練（Online 模式）

即時做 augmentation + 訓練，不需預生成資料：

```bash
# GPU 訓練
python3 train.py --config config.ini

# 指定 GPU ID
python3 train.py --config config.ini --gpu 0

# CPU 訓練
python3 train.py --config config.ini --device cpu

# 指定隨機種子
python3 train.py --config config.ini --seed 123
```

### 離線預生成資料 + 快速訓練（Precomputed 模式）

先用 `gen_dataset.py` 跑一次 augmentation pipeline，存成 `.pt` shard 檔，後續訓練直接讀取：

```bash
# Step 1: 預生成 25 小時資料（自動取最近整數倍 epoch，不會有殘餘筆數）
python3 gen_dataset.py --config config.ini --output data/ --hours 25

# Step 2: 用預生成資料訓練（無即時 I/O + DSP，速度大幅提升）
python3 train.py --config config.ini --precomputed data/
```

`gen_dataset.py` 參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--hours` | 8.3 | 目標音檔總時數（自動 round 到最近整數倍 epoch） |
| `--output` | `data` | 輸出目錄 |
| `--n-shards` | 10 | 分成幾個 shard 檔 |
| `--seed` | 42 | 隨機種子（-1 關閉） |

### 從斷點續訓

訓練中途中斷後，可以從最後的 checkpoint 繼續：

```bash
python3 train.py --config config.ini --resume output/rnnoise_epoch5.pth
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

## ERB 矩陣匯出

匯出 ERB 轉換矩陣（C 部署用）：

```bash
# 匯出所有格式 (npy + C header)
python3 export_erb_matrix.py --config config.ini --format all
```

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `train.py` | 訓練腳本（模型定義 + 訓練迴圈，支援 online / precomputed 兩種模式） |
| `gen_dataset.py` | 離線預生成訓練資料（存成 .pt shard 檔） |
| `dataset.py` | DNS4 資料集 + augmentation pipeline + PrecomputedDataset |
| `denoise.py` | 推論腳本（單檔降噪） |
| `export_onnx.py` | ONNX 匯出（streaming 推論格式） |
| `export_erb_matrix.py` | ERB 矩陣匯出（npy / C header） |
| `config.ini` | 所有超參數設定 |
| `process.c` / `process.h` | C 前後處理實作（嵌入式部署用） |
| `requirements.txt` | Python 依賴套件 |
