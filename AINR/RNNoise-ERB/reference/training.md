# 環境安裝

請在模型目錄執行本頁的 Python 指令。

```bash
# 建議使用 Python 3.9+
pip install -r requirements.txt
```

依賴套件：
- `torch` >= 1.13, < 2.9
- `torchaudio` >= 0.13, < 2.9（2.9 會忽略 `PCM_F`/32-bit WAV 參數）
- `soundfile` >= 0.12
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
- RIR 會自動用 Schroeder 積分法估算 RT60，只保留 `rt60_min` ~ `rt60_max` (`dataset_gen/config.example.ini` 預設 0.2s ~ 1.0s) 範圍內的 RIR
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
| `[feature] erb_norm_tau_sec` | 1.0 | ERB per-band mean EMA 時間常數 |
| `[feature] erb_norm_init_lo_db` | -20 | 最低 ERB band EMA 初值 |
| `[feature] erb_norm_init_hi_db` | -45 | 最高 ERB band EMA 初值 |
| `[feature] erb_norm_scale_db` | 40.0 | mean-subtracted ERB 縮放尺度 |
| `[feature] spec_max_hz` | 4000 | complex branch 頻率上限 |
| `[feature] spec_norm_tau_sec` | 1.0 | complex per-bin magnitude EMA 時間常數 |

改動任何 feature 常數都需要同步修改應用端的 `RNNoiseGridSpec` 並重新訓練。Checkpoint 會保存
`feature_version` 與所有 normalization 常數；train resume、denoise 和 ONNX export
都會拒絕缺少版本的舊 checkpoint，或拒絕與 runtime config 不一致的 checkpoint。
匯出的 ONNX model metadata 也會帶上相同 feature contract，供部署端檢查。

> 只有 `feature_version` 等於上述 `..._v8` 的 checkpoint/ONNX 可用；其他版本
> 即使 input shape 相同，normalization semantics 或 erb_fwd/erb_inv 矩陣也不同，
> 必須重訓後重新匯出。

## 訓練

目前訓練只接受共用 `../dataset_gen/pack_dataset.py` 產生的 raw
noisy/clean WAV tensor；STFT、log-ERB 與 runtime normalization 會在訓練時
即時計算。資料增強、指定 sample-rate 的 WAV generation、resample 與
packing 全部由 `../dataset_gen/` 維護，RNNoise-ERB 不保存重複版本。

訓練 objective 是直接 ERB IRM：對 22 個 band gain 與 ideal ratio
mask 在 `gamma=0.5` 的壓縮域做 MSE，對齊原始 RNNoise 論文採用的 perceptual
compromise。MRSL 的兩個 factor 為 0，沒有 VAD/activity 加權。低於
`energy_floor` 的 mixture band 無法定義 ratio，會被 mask 掉而不產生梯度。

```bash
# Step 0: 建立 dataset_gen 專用設定並填入 speech/noise/RIR paths
cp ../dataset_gen/config.example.ini ../dataset_gen/config.ini

# Step 1: 產生 RNNoise-ERB 專用的 16 kHz dataset
python3 ../dataset_gen/gen_dataset.py \
    --config ../dataset_gen/config.ini --output data_16k --hours 25 \
    --sample-rate 16000 --workers 4

# Step 2: 打包
python3 ../dataset_gen/pack_dataset.py \
    --input data_16k/pairs --output data_16k/packed \
    --shard-clips 512 --dtype float16

# Step 3: 訓練
python3 train.py --config config.ini --packed-dir data_16k/packed

# 可選：指定 GPU 並用 mmap 降低 RAM
python3 train.py --config config.ini \
    --packed-dir data_16k/packed --gpu 0 --mmap
```

`--mmap` 模式不會在 `__getitem__` 逐筆展開成 float32；FP16 batch 送到 GPU
後才轉 float32。訓練會隨機化資料區塊與區塊內順序，避免大型 shard 的全域
random access page fault。`config.ini` 的 `mmap_block_size`、
`mmap_num_workers`、`prefetch_factor` 可依磁碟速度與共用 RAM 調整；預設
256 / 2 / 2 是偏向低 RAM 的起點。

`../dataset_gen/gen_dataset.py` 常用參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--hours` | 8.3 | 目標音檔總時數；最多只向上補一個完整 segment |
| `--output` | `data` | 輸出目錄 |
| `--sample-rate` | config | 此次 generation rate；RNNoise 使用 16000 |
| `--workers` | 4 | DataLoader workers；0 表示單程序 |
| `--resume` | false | 從 `pairs/` 最大編號後繼續寫入 |
| `--seed` | config | 負數為每次 OS-random；非負數可重現 |

### 從斷點續訓

訓練中途中斷後，可以從最後的 checkpoint 繼續：

```bash
python3 train.py --config config.ini --packed-data data_16k/packed.pt \
    --resume output/rnnoise_epoch5.pth
```

續訓會恢復：model weights、optimizer 狀態、global_step、epoch 計數、best_val_loss。
⚠ **scheduler 狀態刻意不恢復**：checkpoint 裡沒有存它，因為 restore 會把寫入當時
那一輪的 `T_max` 帶回來蓋掉為現在這輪重建的值，續訓到較長的 epochs 時 lr 無法降到
min_lr。做法是照新的 epochs 重建，再用 `global_step` 快轉，這也是
`training_common.fast_forward_scheduler()` 做的事。Checkpoint 同時檢查 loss
contract，不一致的 checkpoint 不能 resume。

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
python3 inference.py --config config.ini --model output/rnnoise_best.pth \
                   --input noisy.wav --output clean.wav
```

`--pf-beta` 預設為 0（關閉），避免在還沒驗證 raw gain 前再加劇抑制。

`--atten-lim <dB>` 限制最大抑制量，做法對齊 Rikorose/DeepFilterNet
`enhance.py` 的 `--atten-lim`/`-a`：把 gain 往 1.0（不抑制）方向線性混合，
`lim = 10^(-|atten_lim_db|/20)`，`gain' = lim + gain*(1-lim)`——例如
`--atten-lim 12` 保證輸出最多只壓 12dB，其餘 noise floor 會保留。這在 ERB
band 層級套用（`rnnoise_apply_atten_lim`，C 端同名函式對應),
數學上等價於直接對 spectrum 做同樣的線性混合(因為 mode=1 反向矩陣是
partition of unity),但成本只需 22 次而非 257 次。預設不啟用(`None`,
維持目前最大抑制行為);純推論後處理,不影響 checkpoint/feature contract。
若要定位「全壓成 0」是 feature、model 或 post-filter 造成：

```bash
python3 inference.py --config config.ini --model output/rnnoise_best.pth \
  --input failing.wav --output debug.wav --dump-debug debug/failing.npz
```

dump 含 `erb_db`、兩路 features、`raw_gains`、`post_gains`
與輸入/輸出 waveform。這是 debug 用；checkpoint version gate 則是部署防呆。

[返回本主題](../README.md)
