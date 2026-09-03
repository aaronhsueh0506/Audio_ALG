# AIAEC streaming 與硬體匯出現況

本文件只描述目前保留的四個模型。已移除的 GTCRN/DFN2 AENR 衍生候選不
再是 release surface，也不應出現在 checkpoint、export 或文件清單。

## 共通原則

- 模型 accelerator graph 本身不保留狀態。
- GRU hidden、convolution cache、attention history、K/V ring 都是明確
  input/output，由 CPU 或 accelerator driver 在下一次呼叫送回。
- 一次呼叫處理一個新 STFT frame。
- STFT、WOLA、linear AEC、matched delay 與輸出合成都在模型 graph 外。
- ONNX 使用 real/imaginary pair，不使用 ONNX complex tensor。
- 每個模型的使用者入口都在其自己的目錄：
  - inference.py：逐 hop WAV 推論，以及 inference.py calib
  - export_onnx.py：stateless streaming graph

## 候選與邊界

| Model | Route | Signal input | Explicit state | Accelerator output |
|---|---|---|---|---|
| Align-ULCNet | PBFDKF -> RES+NR | linear error + far | K/V history、logit history、兩層 GRU hidden | enhanced RI + state delta |
| Align-CRUSE | E2E AEC+RES+NR | microphone + far | convolution/GRU/alignment state、score sum、frame index | real mask + next state |
| DeepVQE-S | E2E AEC+RES+NR | microphone + far | convolution/GRU/delay state | 3x3 complex CCM taps + next state |
| CAGCRN | E2E AEC+RES+NR | microphone + far | convolution/GRU/delay state | complex mask + next state |

所有模型使用目前的 16 kHz、FFT/window/hop 512/512/256 contract。

## Align-ULCNet

CPU 每 hop 執行 PBFDKF，取得 formed linear error 與該 hop 實際消費的
aligned far，再推進 STFT。
accelerator 每次收到當前 frame，以及 CPU 保存的 K/V、logit 與 GRU
state。graph 回傳 enhanced spectrum、當前 K/V/logit 與下一組 GRU
state。CPU 更新 ring 後以 WOLA 合成。

D（max_delay_frames）是 export-time state shape。它不改變 learned weight
shape，但 ONNX、calibration manifest 與 ulcnet_model_io 的記憶體配置必須
使用同一個 D。

## Align-CRUSE

state_align_score_sum 是不衰減的 float32 累加器，frame index 是 int64。
兩者必須保持原精度，不得做 integer PTQ。calibration 必須來自單一連續
錄音，避免把多個 reset stream 的 state range 拼接成假分布。

## DeepVQE-S

model graph 輸出 3x3 complex convolving-mask taps；CPU 端
deepvqe_process.c/.h 負責 taps 合成、WOLA 與 persistent spectral history。

## CAGCRN

model graph 輸出 complex mask。ERB forward/inverse table 必須從同一個
checkpoint 透過 CAGCRN/export_erb_matrix.py 匯出，不可混用預設表。

## 匯出

每個模型目錄執行：

    python3 export_onnx.py \
      --checkpoint checkpoint.pth \
      --output output/model.onnx --verify

calibration 也從同一個模型目錄執行：

    python3 inference.py calib \
      --checkpoint checkpoint.pth \
      --primary-dir /path/to/primary \
      --far-dir /path/to/far \
      --frames 8192 --format bin \
      --output calib/model

BIN layout 是 tensor/tensor_1.bin、tensor/tensor_2.bin 等，一個 invocation
一個檔案，manifest.json 記錄 dtype、shape、grid 與 state contract。NPZ
則使用 --format npz 與 .npz output。

Align-ULCNet calibration 的 far WAV 是現有 raw far；manifest 會把
calibration_far_input_mode=raw_far 與
deployment_far_input_mode=aligned_far 分開記錄，避免誤認兩個 seam 相同。

## C 前後處理

> 2026-09-03 已被取代：archive 改為依 configuration 分目錄（`make -C AIAEC
> print-lib-path` 印出絕對路徑），成員另增 `ulcnet_prepost`、`deepvqe_prepost`
> 與 `ulcnet_accelerator_adapter`；現況見 AIAEC/README.md「C pre/post-processing」。
> 以下保留當時的盤點。

AIAEC/build/libaiaec_prepost.a 提供：

- aiaec_process.c/.h：Align-CRUSE 與 CAGCRN mask 合成及共通 STFT/WOLA
- DeepVQE_S/deepvqe_process.c/.h：DeepVQE CCM 合成
- Align_ULCNet/ulcnet_process.c/.h：ULCNet analysis/synthesis
- Align_ULCNet/ulcnet_model_io.c/.h：外部 state/ring descriptor 與更新

## Release gate

1. model-local inference、calibration、ONNX export help 都可執行。
2. PyTorch 與 ONNX Runtime 至少連續重放三個 state step，數值在容許範圍。
3. BIN calibration 每個 input 的檔案數、dtype 與 per-frame shape 和 ONNX
   descriptor 一致。
4. reset 後所有 host state 回到初始值；失敗時不提交部分更新。
5. C scalar/SIMD 與 KISS/NE10 測試全綠。
6. 使用真實 checkpoint 做短音檔 spot check 後才開啟產品路徑。
