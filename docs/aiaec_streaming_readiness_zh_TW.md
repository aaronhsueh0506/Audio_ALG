# AIAEC streaming 與硬體匯出現況

本文件描述目前保留的四個模型。

## 共通原則

- 模型 accelerator graph 本身不保留狀態。
- GRU hidden、convolution cache、attention history、K/V ring 都是明確
  input/output，由 CPU 或 accelerator driver 在下一次呼叫送回。
- 一次呼叫處理一個新 STFT frame。
- STFT、WOLA、linear AEC、matched delay 與輸出合成都在模型 graph 外。C 端已有 pre/post class 的
  Align-ULCNet 與 DeepVQE-S 只收發頻譜：成幀由應用端用 audiolib 的 `AcStft`／`AcIstft` 做，
  遞迴 state 屬於應用端（float host 註冊 buffer，函式庫在 init／reset 歸零、`frame_finish()`
  整塊檢查；板端由 accelerator driver 保存，state generation 改變時以量化後的 0 重載）。
- ONNX 使用 real/imaginary pair，不使用 ONNX complex tensor。
- 每個模型的使用者入口都在其自己的目錄：
  - inference.py：逐 hop WAV 推論，以及 inference.py calib
  - export_onnx.py：stateless streaming graph

## 候選與邊界

| Model | Route | Signal input | Explicit state | Accelerator output |
|---|---|---|---|---|
| Align-ULCNet | PBFDKF -> RES+NR | linear error + far | K/V window、logit window、單一 h_gru（兩顆 subband GRU 的 hidden） | enhanced RI + 完整下一組 state window（就地綁定到輸入位址） |
| Align-CRUSE | E2E AEC+RES+NR | microphone + far | convolution/GRU/alignment state、score sum、frame index | real mask + next state |
| DeepVQE-S | E2E AEC+RES+NR | microphone + far | convolution/GRU/delay state | 3x3 complex CCM taps + next state |
| CAGCRN | E2E AEC+RES+NR | microphone + far | convolution/GRU/delay state | complex mask + next state |

出貨的 checkpoint 都是 16 kHz、FFT/window/hop 512/512/256 contract。Align-ULCNet 與 DeepVQE-S 的
C class 以執行期 grid（`UlcnetGrid`／`DeepVqeGrid`）運作：16000/512 是訓練過的 grid，48000/1024 只是 C 端
接受的 grid（沒有訓練過的模型）。

## Align-ULCNet

CPU 每 hop 執行 PBFDKF，取得 formed linear error，連同原始 raw far
（PBFDKF 的 aligned 副本只用來形成 linear error，對齊由模型的 TA block 負責）再推進 STFT。
accelerator 每次收到當前 frame，以及 CPU 保存的 K/V、logit 與 GRU
state。graph 回傳複數 mask（planar `[1,2,1,BINS]`）與每個 state 的完整下一個 window（輸入是完整 window，位移在 graph 內：
輸入端 Slice 加 Concat；runtime 把每個 `*_out` 綁到同名輸入的位址）。CPU 的 `frame_finish()`（底層 `ulcnet_model_io_enhance_spectrum()`）驗證後以壓縮域 error 乘上 mask、做逆冪，再由應用端的 `AcIstft` 做 WOLA 合成。

D（max_delay_frames）是 export-time state shape。它不改變 learned weight
shape，但 ONNX、calibration manifest、應用端的 state buffer 與 `UlcnetGridSpec.delay_depth`
必須使用同一個 D（函式庫只做範圍檢查；pool 大小不隨 D，D 進 `build_flags_hash`）。

## Align-CRUSE

state_align_score_sum 是不衰減的 float32 累加器，frame index 是 int64。
兩者必須保持原精度，不得做 integer PTQ。calibration 必須來自單一連續
錄音，避免把多個 reset stream 的 state range 拼接成假分布。

## DeepVQE-S

model graph 輸出 3x3 complex convolving-mask taps；CPU 端
`deepvqe_process.c/.h` 的 `deepvqe_ccm_process()` 把 taps 套在最近 `DEEPVQE_TIME_ORDER` 個 mic
頻譜上，兩條頻譜 ring 由 `deepvqe_prepost` class 從自己的 pool 借給它（`deepvqe_ccm_init()`），WOLA 由應用端的 `AcIstft` 做。

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

BIN layout 是 tensor/tensor_0000.bin、tensor/tensor_0001.bin 等，一個 invocation
一個檔案，manifest.json 記錄 dtype、shape、grid 與 state contract。NPZ
則使用 --format npz 與 .npz output。

Align-ULCNet calibration 的 far WAV 是 raw far；manifest 把
calibration_far_input_mode 與 deployment_far_input_mode（兩者皆為 raw_far）
分開記錄，讓 checkpoint 與板端契約不一致時可被偵測。

## C 前後處理

archive 依 configuration 分目錄（`make -C AIAEC print-lib-path` 印出
`libaiaec_prepost.a` 的絕對路徑）。成員：

- aiaec_process.c/.h：Align-CRUSE、CAGCRN 與 ULCNet 的 mask 套用（三個函式，不含 STFT/WOLA）
- DeepVQE_S/deepvqe_process.c/.h：DeepVQE CCM 合成
- DeepVQE_S/deepvqe_prepost.c/.h：DeepVQE-S pre/post 類別
- Align_ULCNet/ulcnet_process.c/.h：`UlcnetModel`／`UlcnetNnJob` 型別、`ulcnet_frame_is_finite()` 與應用端成幀的契約
- Align_ULCNet/ulcnet_model_io.c/.h：grid 與 descriptor、特徵前處理、mask 套用與 state 註冊檢查（state 本身屬於應用端）
- Align_ULCNet/ulcnet_prepost.c/.h、ulcnet_accelerator_adapter.c/.h：ULCNet pre/post 類別與 accelerator adapter

細節見 AIAEC/README.md「C pre/post-processing」。

## Release gate

1. model-local inference、calibration、ONNX export help 都可執行。
2. PyTorch 與 ONNX Runtime 至少連續重放三個 state step，數值在容許範圍。
3. BIN calibration 每個 input 的檔案數、dtype 與 per-frame shape 和 ONNX
   descriptor 一致。
4. reset 後所有註冊的 host state 回到初始值；失敗的 frame 不寫出輸出，註冊的 state 歸零且 state generation 加一。
5. C scalar/SIMD 測試全綠。
6. 使用真實 checkpoint 做短音檔 spot check 後才開啟產品路徑。
