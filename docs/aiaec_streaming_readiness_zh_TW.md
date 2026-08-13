# AIAEC 六候選模型 Streaming Readiness 評估

狀態：2026-08-13 逐行讀碼 + 實測（autograd receptive field、prefix 一致性、
參數量、MAC 統計）的查證紀錄。與程式碼衝突時以程式碼為準。

本文件回答：每個候選模型能不能改成「每 hop 餵 1 個新 frame、狀態跨呼叫
保留」的 streaming 推論；需要保存哪些 state；有沒有結構性阻礙。姊妹文件
[`align_ulcnet_embedded_streaming_design_zh_TW.md`](align_ulcnet_embedded_streaming_design_zh_TW.md)
描述 Align_ULCNet 的完整 embedded 部署提案，本文件是其第 6 節「streaming
model state」分析對其餘五個候選的推廣。

## 1. 六 model 共同現況

- （審計時點現況）六個 `denoise.py` 全部是 whole-wav offline：整檔 STFT
  → 單次 full-T forward → 整檔 ISTFT，當時沒有任何 streaming/stateful
  入口。**同批工作已補上**：每個 model 新增
  `create_stream_state()`/`forward_stream()` 與 `streaming.py` CLI——定位
  是 **NN frame-by-frame Python reference**（驗證等價與定義 state/I-O
  contract），不是完整產品 streaming flow（PBFDKF frontend 仍整檔
  offline、delay event 未接線、無 NPU export——見 §5）。
- 六個 `forward()` 全部丟棄 GRU hidden（`_ = self.gru(x)` 形式）；
  streaming 版由 `aiaec_streaming.StreamGRUCell` 在 forward_stream 中
  外部攜帶 hidden，offline forward 未動。
- STFT 全部走 `dataset_gen/aec_features.py` 的 `center=True`（DFN_AENR 另
  用自己的 `normalized=True` STFT）——framing 層面即帶 n_fft/2 = 1 hop 的
  lookahead。部署決策見設計文件第 7 節（第一版重現 center timing）。
- BatchNorm2d 遍佈各 model（6~22 個）：train 模式對 (B,T,F) 取統計，是
  utterance 級操作；`eval()` 用 running stats 後無跨時間行為，可 fold 進
  conv。**streaming 前必須確保 eval/fold。**
- delay ring 長度：`SignalGrid.delay_frames(1.0s)` = 63 @16k/hop256
  （`aiaec_common.py:46-50`）；只有 Align_ULCNet 對 (16000,512) 硬寫 64
  （`Align_ULCNet/model.py:168-172`）。

## 2. 總覽矩陣

| 項目 | Align_ULCNet | Align_CRUSE | DeepVQE_S | CAGCRN | GTCRN_AENR | DFN_AENR |
|---|---|---|---|---|---|---|
| 參數量（trainable, 16k） | 672,441 | 710,513 | 632,069 | 54,963（+12,929 buffer） | 24,389（+24,576 frozen ERB） | 2,113,480 |
| 演算法 lookahead | **0** | **0**（`causal_running`）；`paper_global` = 整段非因果 | **0** | **0** | **0** | **2 hops**（mask 1 + df 1 串聯；16k = 32 ms） |
| time kernel > 1 層數 | 1（score conv k5） | 6（(4,3)） | 13 | 17（含 dilated） | 6（(3,3) dilated） | 3 conv + 5-tap DF FIR |
| 時間軸雙向 RNN | 無（FGRU 在頻率軸） | 無 | 無 | 無（BiGRU 在頻率軸） | 無（intra 在頻率軸） | 無 |
| utterance 級操作（eval 後） | 無 | **cumsum 積分器** | 無 | 無（LN 在 channel 軸） | 無 | 無（EMA 是逐幀遞迴） |
| prefix/chunk 一致性（eval 實測） | ≤5.6e-9 | **前 63 幀不一致**（T=10 差 1.3e-2），之後收斂 | ≤1.1e-6 | ≤1.9e-7 | —（上游 stream 版逐項對應） | —（平移 2 幀後可比） |
| 最小 N | 1 | 1 | 1 | 1 | 1 | 1 out（3 frames in flight） |
| 跨時間 state 主項 | K/V ring ×D、logit 史 4 幀、GRU 2×2×128 | GRU 192、K/V ring 63、**cumsum acc + 絕對幀計數** | GRU 192、K/V ring 63、score 史 4 幀、CCM 2 幀譜 | time-GRU per-bin 25×24×2、CATA ring 63 | conv_cache/tra_cache/inter_cache ≈70 KB | GRU 5×256、DF ring 5 幀、df_convp 4 幀 c0（96 KB）、EMA 256 floats ×2 源 |
| 實測 MAC | — | — | — | — | ≈33.5 MMAC/s @16k | ≈310.6 MMAC/s @48k（GRU 佔 59%） |
| 既有 streaming 資產 | 無 | 無 | 無 | 無 | 上游 Python N=1 參考 `AINR/gtcrn_github/stream/gtcrn_stream.py` | C 骨架 `AINR/DeepFilterNet2/dfn2_process.c`（單輸入版） |
| 阻礙等級 | 🟢 無 | ⚠ 語意風險（見 §3.2） | 🟢 無（最乾淨） | 🟢 無；NPU op 面最重 | 🟢 無；鎖死 16 kHz | 🟡 接受 2-hop 可直接串流；要 1-hop 須 `df_lookahead=0` 重訓 |

## 3. 各 model 要點

### 3.1 Align_ULCNet

見設計文件第 6 節。唯一 time kernel > 1 = attention score conv（k5 causal，
外部保存 4 幀 logit 史）；far K/V ring 各 `[D,32,26]`；兩個 2 層 subband
GRU hidden 共 2.0 KB；BN ×6 必須 fold。實測 N=1 等價 ≤5.6e-9。

### 3.2 Align_CRUSE — 兩個語意風險

1. **cumsum 無衰減積分器**（`aiaec_common.py:314`，`causal_running` mode）：
   delay score 沿時間 `cumsum`，等於無限記憶——mic 在第 t 幀的擾動永久影
   響之後所有幀的 delay 分佈（實測影響到檔尾）。長通話下 delay 分佈會凍
   結、追不上 echo path 變化。streaming 化本身可行（外部保存 `[B,63]`
   accumulator + 絕對幀計數器，`observable` mask 依 `aiaec_common.py:302-304`
   由幀計數器重建），但**長時行為與 offline 訓練分佈不同屬產品風險**，
   上板前需決策。
2. **短檔（< D 幀）與 offline 不一致；長檔全幀精確等價**（2026-08-13
   實作後修正——先前「前 63 幀必然不一致、測試須排除 warm-up」的說法
   作廢）：offline 的 `observable` mask 用**最終 utterance 長度 T**；
   streaming 版假設 T >= D，因此對任何 >= D 幀（1 s @16k）的 utterance
   **每一幀（含前 63 幀）都精確等價**（實測 90 幀全幀 <= 5e-6，測試未排
   除任何幀）；只有整段 < D 幀的短檔才發散（實測 T=30 差 1.18e-2，有專
   屬測試釘住）。另一個要正式記錄的模型語意：startup 前 D 幀的 softmax
   含有尚未觀測（score=0）的 delay 候選——這是 offline 訓練語意本身，
   不是 streaming 造成。要不要改成逐幀 `delay > frame_index` mask（行為
   改變，需重訓/A-B）是待決策。
3. `paper_global` mode（`aiaec_common.py:305-311`）整段時間 `sum`，完全非
   因果；該 mode 的 checkpoint 不可 streaming，必須重訓。

### 3.3 DeepVQE_S — 六個裡時間軸最乾淨

13 個 causal conv（含 `FreqUpsample` 內 (4,3)）+ FrameDelayAttention（每幀
獨立 softmax、無累積 state）+ CCM 需 2 幀過去 raw complex 譜
（`aiaec_common.py:338-339`）。實測 conv 記憶 mic 34 幀 / far 96 幀、prefix
一致 ≤1.1e-6。無 deconv（上採樣是頻率 sub-pixel reshape）。NPU 注意：
`FrameDelayAttention` 的 `[B,C,T,D,F]` broadcast 必須逐幀化；輸出是 3×3
complex convolving mask（9 complex MAC/bin），不是 ratio mask。

### 3.4 CAGCRN

17 個 causal conv（depthwise dilated 至 dilation 4）+ CATA delay ring 63 +
TFGRU ×2（**頻率軸 BiGRU 每幀獨立；時間軸單向 GRU 的 state 是
per-frequency-bin**，25×24×2 = 1,200 floats）。LayerNorm ×4 只在 channel
軸，安全。可學 delay window 是純量參數，推論期為常數。實測 prefix 一致
≤1.9e-7。NPU 風險最大：CATA `q.unsqueeze(3)*delayed` 產生
`[B,24,T,63,49]`（`CAGCRN/model.py:135`），逐幀化後單幀仍 74k 元素；
ConvTranspose2d 用動態 `output_size`（`:249-251`）須固定。

### 3.5 GTCRN_AENR

骨幹繼承 `AINR/GTCRN/model.py`；AENR 只改第一層 in_channels 9→18。6 個
depthwise dilated (3,3) 全 causal（`F.pad` 左補，`AINR/GTCRN/model.py:163`）；
DPGRNN intra 雙向在**頻率軸**（`:220`）、inter 單向在時間軸（`:227-228`）。
**上游已有 N=1 streaming 參考**：`AINR/gtcrn_github/stream/gtcrn_stream.py:321-326`
的 cache layout（`conv_cache (2,1,16,16,33)` / `tra_cache (2,3,1,1,16)` /
`inter_cache (2,1,33,16)`，合計 ≈70 KB fp32）；AENR 變體 layout 完全相同，
可直接對應。限制：grid 鎖死 16 kHz/512（`GTCRN_AENR/model.py:24-28`），
48 kHz 直接 raise。實測 ≈0.535 MMAC/frame ≈ 33.5 MMAC/s。

### 3.6 DeepFilterNet_AENR

- **lookahead = 2 hops，不是 1**：erb_conv0/df_conv0 的 `LookaheadConv2d`
  取未來 1 幀 + deep filtering `df_lookahead=1`，cascade 串聯（compose 先
  mask 後 DF，`AINR/DeepFilterNet2/model.py:793-802`）→ 實測 `out[6]` 吃
  feature `t=8`。16k = 32 ms、48k = 21.3 ms。要 1 hop 須
  `df_lookahead=0` 並**重訓**（`AINR/DeepFilterNet2/config.ini:42`）。
- **EMA normalization 是逐幀一階遞迴**（`AINR/DeepFilterNet2/train.py:833-837,
  867-871`），初值是固定常數、非全段統計——streaming-safe。error/far 兩條
  state 必須獨立（`dataset_gen/model_views.py:134-138` 拒絕共用），但目前
  所有 AIAEC call site 都傳 `ema_state=None`（per-utterance cold start）；
  streaming 化只需把 state 接起來，公式不變。3τ 暖機約 273 幀。
- **全 repo 唯一 `normalized=True` STFT**（`model_views.py:140-146`），且
  輸入 5 個 tensor；`error_spec`/`far_spec` 是 DF feature 不是頻譜，真正
  頻譜叫 `linear_error`（命名陷阱）。
- DF ring 存的是**已套 ERB mask 的譜**；高頻段須同步延遲（
  `AINR/DeepFilterNet2/dfn2_process.h:148-168` 明文警告）。
- C streaming 骨架已存在（單輸入 DFN2 版：`dfn2_compose_stream` +
  `DFN2State`）；AENR 化需再開 error/far 前處理 state + 兩個 1×1
  conditioner。
- `_init_error_passthrough`（`DeepFilterNet_AENR/model.py:38-47`）把 far
  分支初始化為零貢獻；訓練後應檢查 far 權重是否仍近零（= far conditioning
  沒學起來）。
- 無 lsnr head → 不能複製上游的 per-frame stage skipping，每幀跑滿兩級。

## 4. 移植注意（跨 model 的坑）

1. `AecOutput.mask` 語意不一致：GTCRN 是全頻 complex CRM `[B,T,F]`、DFN
   是 ERB 實數 mask `[B,1,T,32]`、DeepVQE_S 是 `None`（CCM taps）、CAGCRN
   是 `[B,2,T,257]` bounded CRM——比較工具必須分支。
2. `AINR/GTCRN/model.py:189-190` `GRNN` 的 zeros 初始 hidden 未帶
   `dtype`：`model.double()`/fp16 路徑會 dtype mismatch 直接炸。
3. Align_ULCNet 的相位是**壓縮域 atan2**（對 0.3 次方後分量取角，
   `Align_ULCNet/model.py:34-35`），板端不可拿原始相位替代；
   `stage2_act = PReLU(32)` 被兩層共用（`model.py:240-241`），移植要照抄。
4. `causal_delay_stack` 的 `.flip(-1)` 會完整具現化 `[B,C,T,D,F]`
   （`aiaec_common.py:208-211`）：offline 長檔評測 D=64 時 key/value 各約
   200 MB；streaming 逐幀不受影響。
5. DFN `compose()` 兩次 `[B,513,T]` complex `.clone()`
   （`AINR/DeepFilterNet2/model.py:655, 804`），C/NPU 端應消除。

## 5. 交付狀態與範圍（2026-08-13）

已交付：`AIAEC/aiaec_streaming.py`（共用 stateful 元件，全部對照 offline
自證；StreamSTFT/ISTFT 對 torch center=True bit-exact）+ 六個 model 的
`create_stream_state()`/`forward_stream()` + `streaming.py` CLI（逐幀
N=1、打印 per-invocation I/O 與 state 清單、`--verify` 對照 whole-wav）
+ 六個 `AIAEC/tests/test_streaming_*.py`（等價/can-fail/fresh-state，全
部 mutation 驗證）。

另已存在（本節先前版本列為缺項，現已交付）：D 的 CLI/deployment
override（`denoise.py`/`streaming.py` 的 `--max-delay-frames`：checkpoint
contract 仍是 source of truth，override 只重建 alignment depth，權重
D-agnostic 但輸出跨 D 不嚴格相同）與 `denoise.py --stream`（逐幀
`create_stream_state()`/`forward_stream()` 路徑，與 offline graph 的
streaming 等價自證）。

**範圍界定：這是 NN frame-by-frame Python reference，不是產品
end-to-end streaming。** 尚未包含（對應設計文件的 Phase）：PBFDKF
frontend streaming 與 aligned-far 餵入（Phase 1 seam 已在 AEC C 端就
緒，餵入屬 Phase 2 A/B 的輸入分佈變更，不可先斬——C pipeline 變體的
`far_input_mode` 契約以 RAW_FAR 為預設、ALIGNED_FAR 為實驗性候選，即為
此決策的落地）、delay generation/reset/fallback 接線（Phase 5）、
explicit-state NPU export 與 C frontend/postprocess（Phase 4）。
