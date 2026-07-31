# AIAEC、4-ch AEC 與 signal-grid 完成審查

日期：2026-07-30

> **Dated structural audit.** Current model selection and checkpoint rules are
> maintained in [`../AIAEC/README.md`](../AIAEC/README.md),
> [`../AINR/README.md`](../AINR/README.md), and
> [`ai_aec_candidate_matrix.md`](ai_aec_candidate_matrix.md). The DFN-AENR
> description below was superseded when the local DFN2 cascade/alpha graph was
> restored.

## 結論

本輪要求的三個交付面已完成到下列邊界：

1. `AIAEC/` 已由原先三個泛稱 prototype 重構為六個具名候選架構，並有一套 AEC 專用 dataset generator。模型 forward、shape、有限值、因果路徑、參數量級與 dataset-to-model contract 都有自動測試。這代表**架構可進入訓練**，不代表 paper-only reconstruction 可載入未公開的作者 checkpoint。
2. `pipelines/4ch_pipelines/` 已完成「一個 shared matched/delay estimator、四個獨立 linear adaptive filters、外部 beamforming、一次 post-BF RES+NR」的結構，並通過兩組非同步真實錄音 acceptance。**RES 只跑一次，但必須吃 beamforming 後的 mono；不能任取其中一路。**
3. 傳統 AEC、NR 與整合 pipeline 已支援 power-of-two、`frame == FFT`、`hop == frame/2`、無隱藏補零的格點。C 與 Python 的 structural tests 全過；NR 另有 Python/C numeric parity。這是 structural sign-off，48 kHz 與 16 kHz/256 的最終音質仍應另外做 cohort/主觀驗證。

## 1. AIAEC 模型

### 1.1 現行候選與責任邊界

| 分類 | 模型 | 前級 | 公開輸入 | target | 16 kHz trainable parameters | 判定 |
|---|---|---|---|---|---:|---|
| AEC（含 RES） | `Align_CRUSE` | 無 matched/linear AEC | mic + unaligned far | `near_speech + local_noise` | 710,513 | 保留 |
| RES+NR | `Align_ULCNet` | frozen production linear AEC | linear error + far | clean reverberant near | 672,441 | paper reference |
| RES+NR | `GTCRN_AENR` | frozen production linear AEC | linear error + far | clean reverberant near | 48,965 | project variant |
| RES+NR | `DeepFilterNet_AENR` | frozen production linear AEC | linear error + independent error/far DFN features | clean reverberant near | 2,113,480 | project variant |
| E2E AEC+RES+NR | `DeepVQE_S` | 無 matched/linear AEC | mic + unaligned far | early near | 632,069 | primary |
| E2E AEC+RES+NR | `CAGCRN` | 無 matched/linear AEC | mic + unaligned far | clean reverberant near | 54,963 | backup |

舊的 `AECNet/`、`PostFilter/`、`JointAECNR/` 已移除。模型統一使用 complex STFT `[batch,time,frequency]`，輸出 `AecOutput`。

### 1.2 論文／source 對齊程度

- `Align_CRUSE`：保留論文的雙 encoder、soft delay distribution、GRU、skip projection、transpose-convolution decoder 與 mic magnitude mask。論文原格點為 16 kHz、320/160；專案改為 power-of-two grid 並維持約一秒的 delay span。作者未公開 code/checkpoint，因此 padding 與 projection width 是明示的 reconstruction。預設 `paper_global` 會使用完整序列證據；即時串流必須選 `causal_running` 並以該模式訓練。
- `Align_ULCNet`：實作 C-SamFR 的「兩個 FFT bins 一個 subband」取樣、雙 stream、latent cross-attention、FGRU、兩路 temporal GRU 與第二段 complex mask。作者未公開 code/checkpoint；FC activation 等論文未定義處是 reconstruction。
- `GTCRN_AENR`：GTCRN 主體沿用 audited local port，只把第一層由單譜 9 channels 擴成 error+far 的 18 channels。它是專案 variant，不可宣稱為作者發表的 AEC 模型；目前鎖定 upstream 16 kHz/512/256 ERB grid。
- `DeepFilterNet_AENR`：目前沿用 local DeepFilterNet2 cascade/alpha graph，只在 feature boundary 加兩個 1x1 conditioner。conditioner 初始化為 error-only exact pass-through；error/far 必須使用互相獨立的 EMA feature state。這也是專案 variant。審查當日的 DFN3-style 描述已被後續架構分流取代。
- `DeepVQE_S`：完成雙 encoder、約一秒 causal alignment attention、GRU bottleneck、sub-pixel decoder、指定 residual blocks 與 3x3 complex convolving mask。論文原始為 24 kHz/480/240；16/48 kHz power-of-two 版本是 project adaptation。作者未公開 code、loss、GRU width、head count 或 checkpoint。
- `CAGCRN`：完成雙 residual encoder、CATA、兩組獨立 TF-GRU、TFAG、mirrored decoder 與 CRM。論文的整數 `floor(D)` window 無法由一般 autograd 有效學習，本版改為 differentiable soft delay-window gate；因此可訓練但 checkpoint-incompatible。

主要來源：

- Align-CRUSE: <https://arxiv.org/abs/2208.11308>
- Align-ULCNet: <https://arxiv.org/abs/2410.13620>
- DeepVQE: <https://www.isca-archive.org/interspeech_2023/ristea23_interspeech.pdf>
- CAGCRN: <https://www.isca-archive.org/interspeech_2025/wang25d_interspeech.html>
- GTCRN official source: <https://github.com/Xiaobin-Rong/gtcrn>
- DeepFilterNet official source: <https://github.com/Rikorose/DeepFilterNet>

### 1.3 AEC dataset generator

唯一的公開入口與實作目錄都是 `AIAEC/dataset_gen/`。每個 3 秒 chunk 儲存七個 lossless stems：

1. `far_render`：AEC 真正可見的 clean digital reference。
2. `echo`：reference 經 loudspeaker nonlinearity、bulk delay 與 room path 後到 mic 的 echo。
3. `near_speech`：full-RIR、應保留的 reverberant near speech。
4. `near_target`：early near target，供 DeepVQE dereverberation task。
5. `local_noise`。
6. `mic_preclip = near_speech + local_noise + echo`。
7. `mic_postclip`：加入 capture clipping/AGC 後的實際 model input。

重要 contract：

- Align-CRUSE target 為 `S+N`，不會讓 AEC-only 路徑偷做 local NR。
- Align-ULCNet、GTCRN-AENR、DFN-AENR 必須由 frozen production linear AEC 產生 `E=Y-D_hat`；adapter 會拒絕用 oracle echo 做假 residual。
- DeepVQE-S target 為 early near；CAGCRN target 為 full-RIR clean near。
- parent sequence 為 20–60 秒，再切連續 3 秒 chunks；`SequenceChunkSampler`
  保證各 batch lane 的 chunk 次序與 `chunk_index==0` reset signal。它不會自動
  改寫 model state：外部 trainer 必須攜帶 per-lane recurrent/cache state，或在
  forward 前串接相鄰 chunks。
- manifest 在 render 前做 speaker、noise、room/RIR、device source-disjoint split，避免同一來源洩漏到 train/validation。
- 場景含 far/near only、double talk、reference dropout、echo-path change、nonlinear loudspeaker、clipping/AGC、delay jitter、SRO 與 codec-mismatch approximation。

### 1.4 AINR 邊界

對 `Audio_ALG/AINR/` 重新搜尋 `AIAEC`、`AEC`、acoustic echo、far-end、echo path、residual echo，結果為零。AEC model、AEC dataset contract 與 echo conditioning 都只存在 `AIAEC/`，沒有再污染 standalone AINR。

## 2. 4-ch linear AEC

### 2.1 最終資料流

```text
common raw far reference
        |
one shared matched/delay estimator
        |
one common aligned reference
        |
  +-----+-----+-----+
  |     |     |     |
linear linear linear linear AEC     (four independent echo paths)
  |     |     |     |
  +-----+-----+-----+
        |
external beamformer -> one mono
        | + coherently fused residual context
one post-BF RES state + one mono NR
        |
one gain fusion / synthesis output
```

四路 mic 的 acoustic echo paths 不同，所以 linear adaptive filter 不可共用。delay/matched estimator 看的是同一個 digital far reference，所以只建立一份，並讓四個 filters 在 shared delay 改變時一起 reset。

外部 beamformer 必須回傳 mono hop 與 `[4,n_freqs]` complex weights。error、mic、echo estimate 與 R2 context 使用同一組 spatial weights 合成；far spectrum/power 是共同 digital reference，先驗證各 lane 相同後保留一份，**不可把 far reference 乘四路權重相加**，否則 zero-sum beamformer 會把 reference 錯誤消掉。

2026-07-31 起 production API 明確切成
`process_pre_beamformer()` → 外部 SRP-PHAT/GSC →
`process_post_beamformer()`。本 repository 不實作外部演算法，也不再預設建立
equal-weight adapter；後者只有 evaluator/test 明確要求時才使用。pre handoff
保存四路 context snapshots 與 frame/generation identity，post resume 會拒絕
out-of-order、其他 pipeline instance 或 reset 前仍在 flight 的結果。

### 2.2 RES 是否可以取一路

不可以作為正式路徑。RES 的確只需跑一套，但它的 input 必須是 beamforming 後的 mono 和相同 beamforming geometry 下的 residual context。任取一路會：

- 捨棄 beamforming 的空間輸出；
- 讓 RES 所見 echo path 與真正輸出不一致；
- 讓一支 mic 的 residual estimate 代替四路複合 echo path。

任取一路只適合 bring-up/debug baseline，不是 production topology。

### 2.3 非同步實錄測試

資料來自 `datasets/aec_take_turn/` 與 `datasets/aec_together/`。實際檔名是 `unprocessed_4ch.wav`、`woman(ref).wav`、`man.wav`；evaluator 也接受 `women(ref).wav` fallback。

`man.wav` 與 `woman(ref).wav` 位於同一 source timeline，但 source timeline 與 4-ch capture 的開始時間不同。測試先用 near stem 推得一次 timeline offset，再把同一 offset 套到 far stem；不直接把 far 對齊 mic echo，因為那會把 live shared estimator 應估的 acoustic/system delay 一起消掉。

| case | capture | source timeline offset | independent expected echo delay | final shared delay | final error | first solid | resources |
|---|---:|---:|---:|---:|---:|---:|---|
| `aec_take_turn` | 645,120 samples / 40.32 s | 51,925 / 3245.31 ms | 4,233 / 264.56 ms | 4,160 / 260.00 ms | −73 / −4.56 ms | 12.384 s | 1 matcher / 4 linear / 1 RES |
| `aec_together` | 970,240 samples / 60.64 s | 52,540 / 3283.75 ms | 587 / 36.69 ms | 512 / 32.00 ms | −75 / −4.69 ms | 6.880 s | 1 matcher / 4 linear / 1 RES |

兩案均為 16 kHz、FFT/frame 512、hop 256，全程 finite，final delay error 小於半個 hop。使用 deterministic equal-weight test beamformer 時：

| case/cohort | linear attenuation | NR+RES attenuation |
|---|---:|---:|
| take-turn / far-only | 4.52 dB | 16.08 dB |
| take-turn / near-only | 3.36 dB | 9.80 dB |
| together / far-only | 4.66 dB | 16.77 dB |
| together / near-only | 2.47 dB | 6.73 dB |
| together / double-talk | 3.30 dB | 11.26 dB |

這些 attenuation 是 fixture 的能量變化，不是 near-speech quality score。equal-weight adapter 不是實際 beamformer，near-only attenuation 也不能被解讀為改善。因此可下的結論是：**資源拓樸、非同步對齊與資料流已通過；production beamformer 音質尚未簽核。**

另一項明示限制是 `AecResContext` 尚未輸出完整 unbounded R2、stationarity 與 AecState。現行 post-BF RES 使用 bounded R2 並省略 stationary mask，結構正確但不與完整 AEC3 internal RES bit-exact；實際 beamformer 接入後仍需 cohort tuning。

重現：

```bash
cd Audio_ALG
../.venv/bin/python -m pipelines.4ch_pipelines.evaluate_recordings \
  --datasets-root ../datasets
```

## 3. Frame、hop、FFT 與 sample rate

### 3.1 已支援格點

| sample rate | frame = FFT | hop | bins | frame duration | hop duration | default |
|---:|---:|---:|---:|---:|---:|---|
| 8 kHz | 256 | 128 | 129 | 32.000 ms | 16.000 ms | traditional compatibility |
| 16 kHz | 256 | 128 | 129 | 16.000 ms | 8.000 ms | selectable low-latency |
| 16 kHz | 512 | 256 | 257 | 32.000 ms | 16.000 ms | yes |
| 48 kHz | 1024 | 512 | 513 | 21.333 ms | 10.667 ms | yes |

所有列均使用 periodic sqrt-Hann、50% overlap、`frame == FFT`，沒有把 20 ms frame 補零到下一個 FFT size。16 kHz/256 與 16 kHz/512 都是合法格點；48 kHz 只接受 1024。

Python 整合 CLI 現在公開：

```bash
python -m pipelines.aec_nr_pipeline ... --fft-size 256   # 16 kHz only
python -m pipelines.aec_nr_pipeline ...                  # 16k -> 512, 48k -> 1024
```

錯誤的 cross-rate grid、非 power-of-two、frame/FFT 不等、非 50% overlap 都會在 init 前被拒絕。

### 3.2 時間常數

NR 的 EMA、MCRA minima window、init frames、scene-change frames、SPP/gain smoothing 都以原 10 ms reference retime 到實際 hop duration；整合 Python pipeline 的 near-activity hangover 也做相同轉換。因此例如 `L=32` 保留約 320 ms，而不是在不同 grid 盲目固定 32 frames。

AEC3-derived 的大量 constants 已透過 `aec3_scale` 依 grid 轉換；但部分 project-level/tuning fields 仍以每-hop 值表示，例如 shadow error alpha、warmup、EPC hangover、near recent hold/sustain、misadjust counters。這不妨礙建置與執行，但代表不同 grid 的收斂／hangover wall-clock behavior 未全部等價。**不可把本輪 structural pass 寫成跨 grid 音質等價或 bit-exact。**

### 3.3 記憶體與測試結果

Standalone AEC `aec_get_mem_size`：

| grid | bytes |
|---|---:|
| 8k/256 | 292,992 |
| 16k/256 | 397,072 |
| 16k/512 | 543,040 |
| 48k/1024 | 1,233,680 |

整合 `AudioPipelineMemRequirements` descriptor：

| grid | bytes |
|---|---:|
| 8k/256 | 377,120 |
| 16k/256 | 535,936 |
| 16k/512 | 709,088 |
| 48k/1024 | 1,672,336 |

驗收結果：

- Python：AIAEC、AINR、signal-grid、4-ch 合計 `192 passed`；六個 AIAEC
  candidates 均通過 finite forward 與 finite parameter-gradient backward；AIAEC
  另以真 renderer 產生一個 finite、可重組的 3 秒 48 kHz/1024 七-stem
  sample，不只測 STFT helper。
- AIAEC public dataset CLI 與 `compileall` 通過。
- Standalone 與 Audio_ALG 內嵌 AEC：各 `43 passed, 0 failed`；四格點 COLA、1500-hop finite、linear echo convergence、memory consistency 通過。
- Standalone 與 Audio_ALG 內嵌 NR：config validation、四格點 retiming、alignment guard、finite end-to-end 全部通過。
- 48 kHz NR Python/C standard-math parity：3,480 frames × 513 bins，共 1,785,240 個 gains；worst `|Δgain|=2.992e-5`、median `1.490e-8`、mean `4.966e-8`。
- Audio_ALG C pipeline：四格點 heap/pool byte-equal、finite、memory descriptor、negative validation 全部通過；reference board adapter 的兩次 lifecycle 與七個 negative cases 全過。
- 4-ch Python：包含 common-far/zero-sum-beamformer regression 在內全過，兩個實錄 acceptance exit 0。
- `git diff --check`：Audio_ALG、AEC、NR、audio_common 全部通過。

## 4. 最終判定與下一步

可立即開始的是模型架構訓練與 dataset smoke generation。建議優先順序維持：

1. E2E：`DeepVQE_S` primary，`CAGCRN` backup。
2. AEC-only（含 RES）：`Align_CRUSE`；若產品必須 packet streaming，直接用 `causal_running` 訓練，不要先訓 `paper_global` 再期待部署時等價切換。
3. Hybrid RES+NR：`Align_ULCNet` 做 paper reference；`GTCRN_AENR` 做小模型；`DeepFilterNet_AENR` 做較大但已有 DFN base 的比較。

目前六個模型的共同 public forward 是 clip-level API，尚未統一輸出每層
GRU/convolution/alignment cache。第一輪 3 秒訓練可直接使用；若要利用長 sequence
測 cold start、path change 與 drift，先在 trainer 端串接相鄰 chunks。真正逐 frame
部署前，還需要為入選模型定義並驗證 streaming-state export contract。

仍需在取得訓練權重／實際 beamformer 後完成的 quality gates：AECMOS/PESQ/STOI、far-only ERLE、double-talk near preservation、echo-path change recovery、寬頻 48 kHz 真實語料、實測 streaming latency，以及 post-BF 完整 RES context 的 cohort tuning。這些是後續模型／產品驗證，不是本輪架構與介面尚缺檔案。
