# AI-AEC 候選矩陣：架構設計與待審議事項

> **2026-07-30 狀態更新：本文 rev.2–rev.5 是歷史設計／論證記錄，不再是目前實作清單。**
> 原先的「零實作」、`AINR/AECNet`、`AINR/PostFilter`、
> `AINR/JointAECNR` 與 `AINR/dataset_gen/aec` 敘述均已失效；請勿依那些段落建立新工作。
>
> 現行實作位於 `AIAEC/`：`Align_CRUSE`、`Align_ULCNet`、
> `GTCRN_AENR`、`DeepFilterNet_AENR`、`DeepVQE_S`、`CAGCRN`，AEC dataset
> 的穩定入口為 `AIAEC/dataset_gen/`。完整完成審查、4-ch 實錄結果與
> signal-grid 驗證請以
> [`aiaec_4ch_signal_grid_review_2026_07_30.md`](aiaec_4ch_signal_grid_review_2026_07_30.md)
> 為準。
>
> 撰寫 2026-07-29 · rev.2 依深度審查重寫 · **rev.3 收斂為兩級可組合結構**
>
> **基準**：standalone `AEC` `eecddff` / `NR` `8cb9597` / `Audio_ALG` `37db9df` / `audio_common` `c68a50b`
> ⚠ `Audio_ALG/lib/aec` 子模組指標停在 `2049014`（落後 standalone 兩個 commit，但那兩個是 `05c9b7d` 與其 revert `eecddff`，**功能等價**）。本文引用的 `delay_aec3.h` / `aec3_balanced_config.h` / `test_rate_structural.c` 三檔在兩棵樹之間**位元相同**（`cmp` 驗證），故所有引用事實不受指標落後影響。

---

## rev.5 修訂記錄（第二輪文獻查證 + 實作落地）

⚠ **rev.5 收回了 rev.4 的兩項主張，並修正了一個記錯的數字。** 全部來自針對「單純 AI-AEC 的輸出形式」與「改 NR model 做 joint RES」兩題的專項查證（67 個代理、雙視角對抗式駁斥、0 錯誤）。

| # | rev.4 的說法 | 更正 |
|---|---|---|
| **W1** | 「AI AEC **必須**輸出 D_hat，**不得**對 mic 出 mask」，理由是「對 mic 出 mask 等於同時做了 AEC+RES+NR」 | ⚠ **理由錯，禁令過強。** 那是 **target** 的性質，不是乘法的性質——Seidel et al.（IWAENC 2022）就在 mic 上跑複數 mask 而 loss 是明確 echo-only（target = 含噪近端）。更嚴重的是**禁令被本文引用的同一批作者推翻**：他們隔年把自己的減法級換成 mic mask，理由是「direct echo estimation **is at a disadvantage** … it requires reconstruction of the echo's absolute spectral amplitudes」，四項指標全勝（ERLE_BB 30.40 vs 23.93 dB）。→ **降為「可辯護的預設，有已知代價」；該禁的是混合 target。** 見 §5.7 |
| **W2** | 隱含「減法 vs mask」是二分 | ⚠ **漏掉第三類**：**direct complex spectral mapping**（既非減法也非 mask）——LCSM、ISCRN、Zhang/Tan/Wang CRN、NetShell 的 OutE 變體。見 §5.7 |
| **W3** | 「尺度無關比值特徵」寫成設計原則 | ⚠ **那是本專案的假設，不是既有實踐。** 十二個系統的特徵抽取實作調查：主流是**帶絕對尺度**的原始壓縮譜。唯一的比值先例（Pfeifenberger & Pernkopf, Interspeech 2020）從未說明理由、從未測 frontend 可攜性。**最強反證：唯一出貨的生產級神經 RES 寫死 `kScale = 1/32768`**（註解 "Trained model expects [-1,1]-scaled signals"）。→ **改為比值當附加通道，與絕對譜並存。** 見 §5.9 |
| **W4** | 「GFTNN：只換線性級 → ERLE 擺盪 18.5 dB」 | ⚠ **數字與框架都錯。** 作者原話是 "almost 20 dB"；而且那是**兩條各自配置、訓練測試都匹配的管線比較**（postfilter 各自重訓），**不是固定權重換 frontend**。論文自己說 ERLE > 60 dB 後主觀無差異，**而擺盪兩端都在 60 dB 以上**。→ **它不是 postfilter 被打壞的證據，不可再當論據。** |
| **W5** | 「凍結第一級，用它的實際輸出訓第二級」 | ⚠ **文獻反對。** Braun & Valero：「**freezing the DAEC weights led to significantly worse results**」；Zhang & Wang（ICASSP 2022）聯合 53.43 dB ERLE / 2.68 PESQ vs 序列 49.47 / 2.59。見 §8.5 |
| **W6** | 「每個 frontend 訓一個 postfilter，用 `frontend_id` 鎖住」 | ⚠ **解法方向錯。** Panchapagesan（Google, Interspeech 2023）固定部署端、只換產訓練資料的 canceller → WER 相對擺盪 17–30%；但結論是**匹配強濾波器會過擬合**，訓練時用**更弱、殘餘更多、對齊更差**的 frontend 泛化更好。Seidel（ICASSP 2024）獨立採用同招。→ **改為 front-end domain randomisation。** 見 §8.9 |
| **W7** | EchoFree 當 standalone AI AEC 候選 | ⚠ 已於 rev.4 標為 #3，rev.5 補上它用的是**我們同款 PBFDKF**，是 #3 最直接的參照 |

### ✅ rev.5 新增的正面證據

| 主張 | 證據 |
|---|---|
| **echo-target + 減法確實有一份乾淨的 target 消融支持** | `AEC in a NetShell`（ICASSP 2021）：三種 target × 三種 fusion × 三種 skip，同一架構家族。誠實的比較是 echo vs **含噪近端** target（兩者都只做回音消除），在近乎匹配的 echo 抑制下 echo-target Pareto 勝：ERLE 16.65 vs 15.93 dB、近端單講 PESQ 4.50 vs 4.42、**全混合 PESQ 1.86 vs 1.46** |
| **格點** | NetShell 是 **16 kHz / K=512 / shift 256**——與目標格點一致 |
| **NR → joint RES 的結構有鐵板共識** | 八篇全部：**輸出頭不動**（band gain / 複數 mask / 幅度 mask）、**loss 形狀不動**、**乘在 post-AEC 的 E 上**。**要改的是輸入。** → 三個 AINR 模型的頭與 loss 都可保留 |
| **麥克風 Y 是跨陣營唯一穩定的贏家** | Franzen & Fingscheidt（ICASSP 2022）：`Y+E` 把噪音 dSNR 從 14.74 拉到 **22.38 dB**，理由是 **Y 未經 AEC 處理** |

### ⚠ rev.5 必須標明的兩個「證據其實很薄」

1. **echo-target 的近端優勢量級**：近端**單講**只 +0.08 PESQ；那個 +0.40 是**全混合（雙講）**。而且 NetShell 是 TIMIT/VCTK + 影像法 IR + 280 個測試混合、只有 PESQ/ERLE，**無 AECMOS、無真實裝置錄音、無聽測、無第三方複現**。
2. 🔴 **最該注意的一項，它直接打在 #4 → #3 的串接上**：NetShell 自己的最後幾列顯示，**把獨立訓練的 NR 級接到最好的 echo-target AEC 之後，近端 PESQ 從 4.50 掉到 3.55**。作者：「they show improved noise and residual echo suppression, but interestingly reveal a **degradation of near-end speech again** — whereas only our proposed … echo target AEC was able to maintain the near-end speech quality.」
   → **echo-target 的近端優勢只活在 AEC 輸出端。近端風險在第二級，不在第一級的 target 選擇。**

### 📦 rev.5 的實作狀態

三個模型專案與 AEC 資料生成器**已實作並測試**（271 測試全過）：

| 位置 | 對應 | 參數量 | 測試 |
|---|---|---|---|
| `AINR/dataset_gen/aec/` | 語料 + 共用 `AecGrid`/STFT/sampler | — | 25 |
| `AINR/AECNet/` | **#4** `(Y,X) → D_hat` | 1,944,322 | 54 |
| `AINR/PostFilter/` | **#3** `(E,D_hat[,X,Y]) → G` | 1,582,624 / **111.9 M MACs/s** @ 62.5 fps | 72 |
| `AINR/JointAECNR/` | **#5** `(Y,X) → S_hat` | 1,341,483 | 57 |

⚠ 交叉檢查抓到兩個**會靜默毀掉訓練**的缺陷，皆已修並加守衛：① 48 kHz 下以「taps」表示的濾波器跨度只涵蓋 171 ms，**低於語料的 120 ms bulk delay + RIR**，整個語料會變成「打不到的回音」而非「沒除乾淨的回音」，且不會有任何東西報錯；② 三個專案的 `model.py`/`train.py` 同名互相遮蔽，PostFilter 的測試實際在跑 RNNoise-ERB 的 `train()`。

---

## rev.4 修訂記錄（第二輪深度審查，全部已在 repo 驗證）

⚠ **rev.4 修正了 rev.3 的兩處內部自相矛盾。**

| # | rev.3 的錯誤 | 更正 |
|---|---|---|
| **V1** | AI AEC 的 target 寫成「near speech + local noise，**且允許殘餘回音**」 | ❌ **兩句不能同時當 supervised target。** 若 `T = S + N`，loss 就要求模型移除**全部**已標註的回音。殘餘回音必須是估計不完美**自然產生**的結果，不可人工混回 target。→ 改為 **D_hat 契約**，見 §5.5 |
| **V2** | AI AEC 畫成 mask-based（`E' = M ⊙ Y`） | ❌ 直接對 mic 出 mask 等於同時做了 AEC+RES+NR，**RES 邊界完全模糊**。→ 標準輸出改為**複數回音頻譜 D_hat**，見 §5.7 |
| **V3** | 「2 × 4 Cartesian product」 | ❌ 傳統 RES 需要 echo estimate / error / ERLE / 濾波器狀態。改為**有介面前提的相容圖**，見 §6.5 |
| **V4** | `4 + 3` 標為「零額外訓練」 | ❌ 與本文 §8.5 自相矛盾。在傳統 AEC 輸出上訓練的第二級，看到的殘餘分佈與 AI AEC 完全不同 → **直接接上只能當 smoke/OOD test，正式結果必須在 frozen AI AEC 的實際輸出上重訓/fine-tune**，見 §6.4 |
| **V5** | EchoFree 當成 standalone AI AEC 的候選 | ❌ EchoFree 是 **PBFDKF + Bark 神經後濾波器** = 本文的 **#3 拓樸**。⚠ 而且它用的就是**我們同款 PBFDKF**，是 #3 最直接的參照，見 §13.7 |
| **V6** | 「AINR 資料集必須加入殘餘回音」寫得無條件 | ⚠ 混淆了兩個範圍。**standalone AINR 產品維持 noise-only 是正確的**；只有**進 AEC 鏈的第二級**才需要，而那要用另一套 postfilter dataset，見 §8.6 |
| **V7** | 未提 split 資料洩漏 | ✅ 新增。`loader.py` 對**已生成的 item** 隨機切分，同一 speaker/noise/RIR 會橫跨 train/val，見 §8.7 |
| **V8** | 三份重複的格點推導未收斂 | ✅ 新增 `AudioGridSpec` 單一來源要求，見 §1.6 |

---

## rev.3 收斂（兩級可組合結構）

rev.2 的 10 格矩陣**是枚舉不是決策**。rev.3 收斂為**兩級可組合結構**：

| # | 決定 | 位置 |
|---|---|---|
| **C1** | 拓樸改述為 **第一級（2 選 1）× 第二級（4 選 1）** | §6（⚠ rev.4 修正為相容圖，見 §6.5） |
| **C2** | AI AEC 不含 RES | §5.5（⚠ rev.4 以 D_hat 契約重述） |
| **C3** | ⚠ **AI NR 進 `min()` 需要實數 gain，而三個 AINR 模型輸出形式不同** —— GTCRN 是 Complex Ratio Mask，**進不了 `min()`** | §5.6 |
| **C4** | 補上 **AI RES + 傳統 NR**（#2b）—— 針對本 repo 已知的近端殺手 | §6.2 |
| **C5** | 砍出第一輪：S1（NN R² 估計）、NN 濾波器控制（S3）。⚠ 但資料集的收斂場景不得跟著砍 | §12.2 |

---

## rev.2 修訂記錄（全部來自外部深度審查，全部已在 repo 上驗證）

| # | 原稿的錯誤 | 更正 |
|---|---|---|
| **R1** | 把「訓練資料含殘餘回音」與「輸入 R² 特徵」寫成互斥，並把前者標為❌禁止 | **推翻。** 兩者是獨立維度；下游 NN 若要處理殘餘回音，**必須在真實殘餘回音上訓練**，餵 R² 不會讓 noise-only 訓練的模型自動學會用它。真正該禁止的是**抑制歸屬不明**。見 §5 |
| **R2** | 四條互斥路徑 | **推翻。** AEC / RES / NR 是三個獨立軸，四條路壓縮掉了必要組合。改為 9 格矩陣，見 §6 |
| **R3** | 把 hybrid postfilter 的文獻支持套到舊 S2 | **推翻。** 文獻的 hybrid 是「線性 AEC → **一個** NN postfilter 同時處理殘餘回音+噪音」，不是「傳統 RES + 另一個 NN NR + min()」。支持最強的是原稿**沒有的「單一 AI joint RES+NS」（rev.3 的 #3）** |
| **R4** | 缺 isolation baseline | 補上 **`4`**（AI AEC 單獨）與 **`4 + 1`**（+ 傳統 NR）。沒有它們無法歸因改善來自 NNAEC 本體還是第二級 |
| **R5** | 「48 kHz 尚未支援」 | **事實錯誤。** 已 structural 支援 8/16/48 kHz，見 §2 |
| **R6** | 193-tap 無混疊「保證」 | 降級為**待驗證假設** + 測試清單，見 §1.4 |
| **R7** | 延遲只是表格欄位 | 升為**正式 gate**，見 §1.5 |
| **R8** | 波形式 E2E 的回音成分「數學上無定義、不可訓」 | 過度絕對。正確說法見 §7.3 |
| **R9** | joint model 後接任何 block = 編譯錯誤 | 研究階段過早。改為預設 + 允許清單，見 §7.4 |
| **R10** | §2 說共用 delay，§6 說路 3 的 delay 消失 | **內部矛盾。** 改為三種明確的對齊歸屬，見 §3.2 |
| **R11** | 資料集規格不足 | 大幅擴充，含 stems / 場景覆蓋 / **不可每 3 秒 reset state** / target ownership，見 §8 |
| **R12** | 多通道只列為「未查證」 | 升為**資料生成前必須決定**，見 §9 |

⚠ 審查中唯一的行號小誤：MCRA `L = 32` 在 `NR/c_impl/include/mmse_lsa_types.h:126`，非 `:34`。實質內容完全正確。

---

## 0. 給 reviewer 的閱讀指引

| 標記 | 意義 |
|---|---|
| ✅ | 已在本 repo 逐行讀過原始碼確認，附 `file:line` |
| 📄 | 引用外部已出貨原始碼（本 repo 內有快照） |
| 📊 | 本專案先前的實測數字 |
| ❓ | **未查證**或推估 |
| ⚠ | 已知風險或缺陷 |

---

## 1. 格點

### 1.1 目標格點與 ⚠ 跨取樣率時間幾何的取捨

| | 取樣率 | 分析窗 | hop | FFT | bins | 幀率 | 窗時長 | hop 時長 |
|---|---|---|---|---|---|---|---|---|
| **現況 16k** ✅ | 16 kHz | 320 | 160 | 512（補零） | 257 | 100 fps | 20 ms | 10 ms |
| **現況 48k** ✅ | 48 kHz | 960 | 480 | 1024（補零） | 513 | 100 fps | 20 ms | 10 ms |
| **目標 16k** | 16 kHz | 512 | 256 | 512（不補零） | 257 | 62.5 fps | **32 ms** | **16 ms** |
| **目標 48k** | 48 kHz | 1024 | 512 | 1024（不補零） | 513 | 93.75 fps | **21.333 ms** | **10.667 ms** |

現況來源：`pipelines/pipeline_dims.h:27-35`（`hop = 0.01×sr`，`frame = 2×hop`，FFT 補到次方）；48k 常數 `AEC3B_R48K_{N_BINS,FFT_SIZE,BLOCK_SIZE,HOP_SIZE} = 513/1024/960/480` ✅ `aec3_balanced_config.h:207-210`。

> ### ⚠ 這是一個原稿與審查都未點破的取捨
>
> **現況的兩個取樣率時間幾何一致**（都是 20 ms 窗 / 10 ms hop）；**新格點會打破這個一致性**（32/16 ms vs 21.333/10.667 ms）。
>
> 原因是算術上的硬限制：48 kHz 要拿到 32 ms 窗需要 **1536 點**，非 2 冪次。
> **「2 冪次且不補零」與「跨取樣率時間幾何一致」在 48 kHz 上不可兼得。**
>
> 三個選項，**這是 D1 的真正內容**：
> | | 16k | 48k | 代價 |
> |---|---|---|---|
> | (i) 兩率都 2 冪次不補零 | 512/256 | 1024/512 | 時間幾何分歧；所有時間相關常數必須分率推導 |
> | (ii) 保持時間幾何一致 | 512/256 | 1536/768 | 48k 需非 2 冪次 FFT（違反嵌入式約束） |
> | (iii) 維持現況 | 320/160 | 960/480 | 兩率一致且已驗證，但 AINR 模型格點對不上 |

### 1.2 格點變更的連鎖影響

**有利：**

1. ✅ **AINR 兩個 16 kHz 模型的格點與目標 16k 直接對上**：`AINR/GTCRN/config.ini:2-5` 顯式 `n_fft=512/win_len=512/hop_len=256`；`AINR/RNNoise-ERB/config.ini:2-3` 只設 `sr`/`n_fft`，`win_len`/`hop_len` 走預設。⚠ `AINR/DeepFilterNet2/config.ini` 是 48 kHz/1024/512，正好對上目標 48k。
2. **16k 幀率 100 → 62.5 fps**：AI 模組推論次數與加速器往返 ×0.625。⚠ **48k 只有 100 → 93.75 fps（×0.9375）**，幾乎沒有節省——原稿把 ×0.625 當成全域結論是錯的。
3. 文獻支持：16 kHz / 512 / 256 是該取樣率下最常見的配置（§13.2）。

**不利：**

4. **16k 演算法延遲 20 → 32 ms**（+12 ms）。⚠ 見 §1.5，這是 gate 不是欄位。
5. ⚠ 無混疊餘裕假設失效 —— 見 §1.4。
6. **所有時間相關常數必須重新推導** —— 見 §1.3。
7. **48 kHz delay estimator 的固定幾何** —— 見 §3.2。

### 1.3 ⚠ 常數必須由「秒 / Hz」推導，不可共用固定幀數

新舊格點的 hop 時長不同，且兩個取樣率之間也不同（§1.1）。因此**所有 frame-count、EMA、hangover、lookahead、filter partitions 都不能共用固定幀數**。

**EMA：** 應寫成 `alpha = exp(-hop / (sr × tau))`，由目標秒數推導。

| α | 16k / hop 256 | 48k / hop 512 |
|---|---|---|
| 固定 α = 0.99 → 實際 tau | **1.59 s** | **1.06 s** |

同一個 α 在兩率上是**不同的時間常數**。

**具體實例 —— NR 的 MCRA 最小值追蹤窗** ✅ `NR/c_impl/include/mmse_lsa_types.h:126`：

```c
config.L = 32;               // 32 × 10ms = 320ms (sync with Python v3_2)
```

| | hop 時長 | L=32 代表 | 保 320 ms 需要 |
|---|---|---|---|
| 現況（兩率） | 10 ms | 320 ms ✅ | — |
| 目標 16k | 16 ms | **512 ms** ❌ | **L = 20** |
| 目標 48k | 10.667 ms | **341 ms** ❌ | **L = 30** |

**AEC 側同類常數** ✅ `AEC/c_impl/include/aec3_scale.h`：
- `:54-57` 四個 leakage 係數含 `×2.5 = (160/16000)/(64/16000)`
- `:60` `AEC3_POOR_EXC_COUNTER_INITIAL_HOPS 400` = `blocks_to_hops(1000,160,16k)`

轉換機制**已參數化**（`:31-40`：`aec3_blocks_to_hops` / `aec3_per_block_rate_to_per_hop` / `aec3_per_bin_psd_threshold` / `aec3_nl_r2_norm_power` / `aec3_fft_density_scale` / `aec3_block_energy_scale`），`#define` 只是為 bit-exact 預折。**是「重折常數」不是「重寫機制」。**

⚠ **但三個常數繞過 `fft_density_scale`**（`:80,81,83`）：`AEC3_FILTER_NOISE_GATE_POWER_FLOAT`、`AEC3_STATIONARITY_MIN_NOISE_POWER_FLOAT`、`AEC3_RESIDUAL_NOISE_GATE_POWER`。原先歸類為「fullband 才處理」，**格點變更後 #1（現況鏈）就會撞上**。

### 1.4 ⚠ 193-tap 無混疊：降級為待驗證假設

**原稿主張**（已撤回其確定性）：現況 `win=320` 補零到 512 給了 `512−320+1 = 193` 點的線性卷積空間，逐 bin 複數 mask 數學上無混疊；改成 `win = n_fft` 後餘裕歸零。

**為何過度確定**：該論證只在「等效 FIR 確實被限制在 193 taps」**且**「crop / overlap-save 定義吻合」時成立。任意 time-varying mask 的 IFFT 一般**不是** 193-tap FIR；WOLA 也**不等於**自動取得 overlap-save 的線性卷積保證。

**改為假設 + 必須跑的測試：**

| 測試 | 內容 |
|---|---|
| impulse / time-alias test | 量測實際的時域環繞能量 |
| complex-mask reconstruction test | 已知 mask 下的重建誤差 |
| deep-filter order 上限測試 | 多幀複數 FIR 的階數上限 |
| measured streaming latency | 見 §1.5 |

⚠ **在這些測試跑完之前，不得以「有 193 taps 餘裕」或「餘裕歸零」為由做架構決策。**

### 1.5 ⚠ 延遲是正式 gate

**32 ms 窗 ≠ 32 ms 系統延遲。** 實際延遲由 buffering contract + impulse test 定義，必須量測。

參考點：**ICASSP AEC Challenge 將 algorithmic + buffering latency 上限訂在 20 ms**（2023 年起）。目標 16k 格點的窗長就是 32 ms。

⚠ 這表示新格點下的系統**可能不符合該類規範**。延遲必須列為**通過/不通過的 gate**，在格點定案前量測，而不是事後才發現。

---

### 1.6 ⚠ V8：三份重複的格點推導必須收斂成單一 `AudioGridSpec`

同一條公式（`hop = 0.01×sr`，`frame = 2×hop`，`fft = next_pow2(frame)`）目前在**三個地方各寫一次** ✅：

| 位置 | 內容 |
|---|---|
| `pipelines/pipeline_dims.h:27-35` | `compute_frame_dims()` |
| `AEC/c_impl/src/aec.c:589` | `aec_derive_dims()` |
| `NR/c_impl/include/mmse_lsa_types.h:79` | `mmse_lsa_default_config()`（註解：「20ms frame, 10ms hop — unified with AEC pipeline」） |

⚠ **格點變更要同時改三處，任何一處漏改就是 §3.1 那個 init 期 FATAL guard 才會抓到的錯**——而那個 guard 只比對 `n_freqs`/`hop`，抓不到窗長或補零與否的分歧。

**→ 建立單一 `AudioGridSpec`（sr → win / hop / fft / n_freqs / frame_rate），三方共用。** 這是格點遷移的前置，不是可選的整理。

## 2. ⚠ 事實更正：48 kHz 已經 structural 支援

原稿把 48 kHz 寫成未支援。**這是錯的。**

| 事實 | 位置 |
|---|---|
| AEC 取樣率白名單已從 `{16000}` 擴為 **`{8000, 16000, 48000}`** ✅ | `AEC/c_impl/src/aec.c:115-123` |
| 每率係數/門檻表 `AEC3B_RATE_TABLE` 已存在（含 R8K-/R48K- 區塊）✅ | `aec3_balanced_config.h:302+` |
| 48k 維度 `513 / 1024 / 960 / 480` ✅ | `aec3_balanced_config.h:207-210` |
| NR 亦接受 48 kHz ✅ | `NR/c_impl/include/mmse_lsa_types.h:34` |
| **8/16/48 structural test 已存在** ✅ | `AEC/c_impl/test/test_rate_structural.c` |
| ⚠ **但該測試未接進任何 Makefile** ✅（`grep` 兩棵樹的 Makefile 皆無） | — |

原始碼註解自述該多率活動「已 landed 並經 16 kHz byte-identical 驗證」。

**因此正確的工作描述是：**

> 把既有的 8/16/48 kHz 支援**重新格點化**，重新產生 rate tables（所有常數由秒/Hz 推導），**把 `test_rate_structural` 接進 CI**，並完成 48 kHz 的**品質認證**（目前所有 48k 測試訊號都是 16k 上採樣，❓ 從未以真寬頻訊號測過）。

---

## 3. 共用前端與對齊歸屬

### 3.1 `se_frontend`

```
  mic ──→ HPF → framing → sqrt-Hann 分析窗 → FFT ──→ Y(f)
  ref ──→ far-end ring buffer (2048 ms) ──────────→ X(f)
```

### 3.2 ⚠ 對齊歸屬：三種模式（取代原稿的內部矛盾）

原稿 §2 說四條路共用 matched-filter delay，§6 又說 E2E 路徑的 delay estimator 消失。**互相矛盾。** 明確拆為三種：

| 模式 | 說明 | 第一版建議 |
|---|---|---|
| **A1 externally aligned** | 前端 DSP delay estimator 對齊後才餵給模型 | ✅ **建議** |
| **A2 alignment-robust** | 模型自帶對齊（cross-attention 等），不需外部估計 | 第二輪 |
| **A3 hybrid** | 外部粗對齊 + 模型內細對齊 | 第二輪 |

**每一個矩陣格子都必須標明它用哪一種。** 第一版全部用 A1——這樣 delay estimator 對所有格子都是共用前端，矛盾消失。

### 3.3 ⚠ 48 kHz delay estimator 是實質風險

delay chain 的幾何是**固定樣本數，無 `sample_rate`** ✅ `AEC/c_impl/include/delay_aec3.h:95-105`：

```c
#define DA_DOWN_SAMPLING_FACTOR 4
#define DA_AEC3_BLOCK_SIZE      64
#define DA_SUB_BLOCK_SIZE       16          /* 64/4 */
#define DA_NUM_FILTERS          5
#define DA_WINDOW_SIZE_SB       32
#define DA_ALIGNMENT_SHIFT_SB   24
#define DA_FILTER_SIZE          512         /* 32×16 */
#define DA_FILTER_INTRA_SHIFT   384         /* 24×16 */
```

block rate `250 = 16000/64`。48 kHz 直接餵入會讓**所有時間範圍縮短三倍**（可測延遲、histogram、counter）。

**兩個方案：**

| | 做法 | 風險 |
|---|---|---|
| **✅ 建議** | 48k mic/ref 經 anti-alias decimator 降到 **16 kHz 固定 sidechain** 做 delay/SRO 估計，再把 delay 映回 48k 樣本域 | 低。delay 幾何完全不動 |
| 替代 | 全面重推 delay 常數 | 高。等於重做已 bit-exact 驗證過的模組 |

---

## 4. `AecResContext` 接縫與 R²

### 4.1 R² 的定義

✅ `AEC/c_impl/src/residual_echo_estimator.c:245`（`ree_estimate`），核心公式 `:292`：

```c
r2[k] = s2_linear[k] / max(erle[k], 1e-30);   /* 再加 reverb tail 與非線性路徑 */
```

`s2_linear` = Ŷ²（線性回音估計功率），`erle` = 線性濾波器實際達成的 ERLE。**語意：線性濾波器沒有除掉的回音還剩多少功率。** ⚠ 兩者都是線性濾波器的內部狀態 → 沒有線性濾波器的格子無法用此公式。

### 4.2 R² 的三個消費者 ✅

`pipelines/audio_pipeline.c:655-730`：

| # | 消費者 | 位置 | 作用 |
|---|---|---|---|
| 1 | OM-LSA NR | `:663-668` | `extra[k] = r2[k]/PSD_SCALE` → `ξ = S²/(N²+R²)` → `g_nr` |
| 2 | AEC3 `suppression_gain` | AEC 內部 | → `ctx.res_gain` = `g_res` |
| 3 | near-end lift | `:704-714` | `echo_frac = r2/e2` |

融合 `:678`：`sk_min_f32(p->g_total, p->g_nr, ctx.res_gain, n_freqs);`
CNG（`:722-730`）以 `res_gain` 決定 `sqrt(1 - G_res²)` → 間接依賴 R²。

📊 開關 `:666`：`nr_extra = p->legacy_amin ? NULL : p->extra;`
`legacy_amin=1`（NR 拿不到 R²）vs `=0`（現行預設）實測差距：**DT echo +0.23 / FS +0.14 / BAK +0.15**（A_min_pl，2026-06-09 上 main）。
⚠ 該數字在**古典線性 AEC 的殘餘水準**下測得，不可外插到其他殘餘水準。

### 4.3 ⚠ 已知文件缺陷（兩份文件皆有）

| 來源 | 說法 | |
|---|---|---|
| `AEC/docs/nn_integration_interface.md:20` | `near_spec` = **E(f)** 誤差頻譜 | ❌ 錯 |
| `Audio_ALG/docs/freq_domain_pipeline_design.md:19` | 同上 | ❌ 錯 |
| `AEC/c_impl/src/aec.c:2477` ✅ | `ctx->near_spec = a->main_filter.base.near_spec` | |
| `AEC/c_impl/src/pbfdkf.c:403` ✅ | `rfft_padded(p, p->near_buffer, blk, p->near_spec)` → **mic 頻譜** | |
| `AEC/c_impl/include/aec.h:506` ✅ | `error_spec = windowed linear error E(f)` | ✅ 正確 |

第二個缺陷：`r2`/`comfort_noise` 的 int16² 縮放只在 `aec.h:507-510` 註解，型別是裸 `const float*` → 保證 training-serving skew。

**兩者與架構決策無關，零風險清障。**

---

## 5. ⚠ 抑制決策歸屬（rev.2 取代原稿 §5.3）

### 5.1 原稿的錯誤

原稿把兩件事寫成互斥的三選一，並把「訓練資料含殘餘回音」標為 ❌ 禁止。**這在訓練邏輯上不成立。**

**它們是兩個獨立維度：**

| 維度 | 選項 |
|---|---|
| **D-train**：訓練分佈是否含殘餘回音 | 含 / 不含 |
| **D-feat**：模型是否有顯式的 R² / reference / linear echo estimate 特徵 | 有 / 無 |

**關鍵**：只要下游 NN 要處理殘餘回音，就**必須在真實殘餘回音上訓練**。即使輸入 R²，**noise-only 資料集也不會讓模型自動學會怎麼用那個通道**——它從未見過該通道與任何有意義的東西共變。

⚠ 這直接影響現況：✅ `AINR/dataset_gen/dataset.py` 中 `"echo"` 出現 **0 次** → 三個 AINR 模型**構造上沒見過殘餘回音**。要把任何一個放進 AEC 鏈，**資料集必須先加入殘餘回音，這是不可繞過的**。

### 5.2 真正該禁止的是什麼

> **兩個未校準的 aggressive gain 同時作用，再用 `min(g_res, g_nn)` 疊加，卻沒有明確的 suppression ownership。**

現行出貨設計之所以安全，不是因為「NR 不 echo-aware」，而是因為 `g_nr` 與 `g_res` 的 echo-awareness 來自**同一個 R²**——一個估計、兩個消費者、彼此校準。

### 5.3 三種合法介面（依抑制歸屬分類）

| 介面 | 網路輸出 | 誰擁有最終 gain | 對應矩陣格 |
|---|---|---|---|
| **I1** | 最終 **joint RES+NS gain** | **網路**（繞過 `g_res` / `min()`） | #3 · `4+3` · #5 |
| **I2** | **echo / noise 兩個 head** | **fuser**（帶 attenuation cap） | 待設計 |
| **I3** | 只輸出 **echo mask / R²** | **傳統 suppressor** | S1（探索項，§12.2） |

**⚠ 原稿的 S2（NN 取代 OM-LSA、保留傳統 RES、最後仍 `min()`）不屬於上述任何一種標準 hybrid。** 它是一個第四種：*NN NR inside legacy fusion*。它是合法的，但**必須改名，且不得援引 hybrid postfilter 的文獻支持**（見 §13.4）。

### 5.4 ⚠ 命名更正

| 原稿 | rev.2 |
|---|---|
| 「S2 = NN NR」，並稱其為 hybrid postfilter | **NN NR inside legacy fusion**（傳統 RES 仍在，仍走 `min()`） |
| （不存在） | **classic LEC + 單一 NN joint RES+NS**（文獻標準 hybrid，⚠ **需要新接縫 S4**，見 §6.3） |

### 5.5 ⚠ V1：AI AEC 的訊號模型與 target 契約

#### rev.3 的錯誤

rev.3 寫「AI AEC 的 target = near speech + local noise，**且允許殘餘回音**」。**這兩句不能同時作為 supervised target。**

若 `T = S + N`，loss 就要求模型移除**全部**已標註的回音。**殘餘回音必須是回音估計不完美所自然產生的結果，不可人工混回 target。**

#### ✅ 正確的訊號模型與契約

```
Y      = S + N + D            麥克風訊號（S 近端語音 / N 本地噪音 / D 回音）
X      = far-end reference
D_hat  = AI_AEC(Y, X)         ← 第一級的輸出：估計的回音
E      = Y − D_hat            ← AEC 輸出（減法產生）
R      = D − D_hat            ← 殘餘回音，結構上自然產生，不是 target 的一部分
```

| 級 | 輸入 | 輸出 | target |
|---|---|---|---|
| **第一級 AI AEC** | `(Y, X)` | **`D_hat`** | **`D`（回音本身）** |
| **第二級 RES/NR** | `(E, D_hat[, X])` | `S_hat` | `S`（或 early-reflection 版本） |

⚠ **`R` 不需要被「允許」——它是 `D − D_hat` 的定義，估計不完美就自然存在。** rev.3 那句話是對一個不需要建構的東西做了建構。

#### 建議的 loss 形式

```
L = L_echo(D_hat, D)                                  ← 主項：估計回音本身
  + λ_out  · L_output(Y − D_hat, S + N)               ← 減法後的輸出保真
  + λ_near · near_end_preservation                    ← 近端不被誤消
  + λ_idle · ‖D_hat‖   當 far-end / reference 靜默時   ← 遠端無訊號時不得憑空產生回音估計
```

⚠ `λ_idle` 項與 §12.3 的硬性 gate（`ref = 0 → output ≈ mic`）是同一件事的訓練側與驗證側。

#### 為何選「估計回音再相減」而非「直接輸出增強語音」

1. **文獻一致**：本文 §13.3 已引用的兩個 echo-only 模型正是此形式——`NKF-AEC` 的 `Ŝ = D − D̂`、`NLAEC` 的 `ŝ = s + (y − ŷ) + w`。⚠ **rev.3 引用了這些證據卻沒有把它們的形式帶進 §8.4，這是 rev.3 的內部不一致。**
2. **近端保真**：直接預測回音、再由麥克風相減，對 near-end 的保真通常優於直接回歸增強語音（`AEC in a NetShell` 對 target 的比較）。
3. **可歸因**：見 §5.7。

#### ⚠ 仍然成立：「見過」≠「負責」

第二級在推論時**必然**看到 `R`（第一級不可能完美）。

| | |
|---|---|
| 第二級**見過** `R`（訓練分佈） | ✅ **必須** —— 否則推論時是 OOD |
| 第二級**負責**抑制 `R` | ✅ 是 |

⚠ 但這只適用於**進 AEC 鏈的第二級**。**standalone AINR 產品維持 noise-only 資料集是正確的**（V6，見 §8.6）。

### 5.7 ⚠ V2：AI AEC 的輸出契約 —— 不得是 mic mask

若第一級直接做 `M(Y, X) · Y → enhanced speech`，它**已經同時做了 AEC + RES + NR，RES 的邊界完全模糊**，本文區分第一級/第二級的整個前提失效。

| 輸出形式 | 判定 |
|---|---|
| **複數回音頻譜 `D_hat`** | ✅ **建議** |
| 自適應濾波器係數 | ✅ 可接受（等價於 `D_hat = W ⊛ X`） |
| enhanced speech / mic mask | ❌ **不建議作為第一版** |

**`D_hat` 契約讓下列四項各自可量測：**

1. 回音估計品質 —— `L_echo(D_hat, D)`
2. 相減後的殘餘 —— `R = D − D_hat`
3. 第二級的貢獻 —— 比較 `E` 與 `S_hat`
4. **近端是否被 AEC 誤消除** —— 近端單講時 `D_hat` 應趨近 0

⚠ 這也**推翻了 rev.3 §6.3 / §7.4「AI AEC 必須是 mask-based 才能有 R² 輔出頭」的論證前提**——輸出 `D_hat` 之後根本不需要 R² 輔出頭（見 §5.8）。

### 5.8 🆕 `D_hat` 契約意外修復了傳統 RES 的介面

`ree_estimate()` 的兩個主要輸入是 `s2_linear`（Ŷ² 回音估計功率）與 `erle`（`residual_echo_estimator.c:245-256` ✅）。

**若第一級輸出 `D_hat`：**

| `ree` 需要 | 從 `D_hat` 契約取得 |
|---|---|
| `s2_linear` = Ŷ² | ✅ `|D_hat|²` |
| `erle` | ✅ 可由 `E[|Y|²] / E[|E|²]` 逐 bin 量測，不需濾波器內部狀態 |
| `render_psd`、`capture_psd` | ✅ 前端已有 |

**→ 傳統 `ree` → `suppression_gain` → `g_res` 這條路在 AI AEC 之後仍可運作**，`4 + 1` 就不是只能 echo-blind NR。

⚠ **但不是免費的。** `ree_estimate` 另外需要 `filter_delay_blocks`、`filter_length_blocks`、`usable`、`saturated`、`transparent_mode`、`force_nonlinear_path` 等**濾波器狀態旗標**，這些在 AI AEC 下沒有直接對應，必須：

- 提供替代估計（例如由 `D_hat` 的能量比推 `usable`），或
- 給保守預設值並記錄其影響

**這是 D12：`D_hat` → `ree` 的狀態旗標對應表，必須逐項決定。**

### 5.6 ⚠ C3：AI NR 進 `min()` 的相容性 —— 三個模型輸出形式不同 ✅

融合點 `sk_min_f32(g_total, g_nr, res_gain, n_freqs)`（`audio_pipeline.c:678`）要求 `g_nr` 是**實數逐 bin gain**。三個 AINR 模型的輸出**不一樣**：

| 模型 | 輸出 | 位置 ✅ | 能進 `min()`？ |
|---|---|---|---|
| **RNNoise-ERB** | 實數 band gains | `RNNoise-ERB/train.py:514,567`（`回傳: gains`） | ✅ 展開到 257 bin |
| **GTCRN** | **Complex Ratio Mask** | `GTCRN/model.py:276-281` | ❌ **不行** |
| **DeepFilterNet2** | 全頻實數 ERB mask，再於低頻 cascade deep filter + alpha（複數） | `AINR/DeepFilterNet2/model.py::DeepFilterNet2.compose` | ⚠ 最終低頻不行 |
| **DeepFilterNet3** | 高頻實數 ERB mask + 低頻 deep filter（複數） | `AINR/DeepFilterNet3/model.py::DeepFilterNet3.compose` | ⚠ 只有高頻可以 |

GTCRN 的 mask 是完整複數乘法，**會旋轉相位**：

```python
# GTCRN/model.py:276-281
class Mask(nn.Module):
    """Complex Ratio Mask"""
    def forward(self, mask, spec):
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
```

DFN2 現在恢復為**串聯 cascade + learned alpha**：全頻先套 ERB mask，
低頻再由 deep filter（複數多幀 FIR）處理並與 masked residual 混合。原本的
**平行 band split** 已獨立保留在 DFN3：

```python
# AINR/DeepFilterNet2/model.py::DeepFilterNet2.compose
spec_m = spec * full_band_bin_mask
spec_df = deep_filter_apply(spec_m, ...)
spec_e[:, :df_bins] = alpha * spec_df[:, :df_bins] + (1-alpha) * spec_m[:, :df_bins]

# AINR/DeepFilterNet3/model.py::DeepFilterNet3.compose
spec_e = deep_filter_apply(spec, ...)        # raw low-frequency spectrum
spec_e[:, df_bins:] = masked_high_spectrum  # parallel band split
```

#### 三種串接方式

| | 做法 | 適用 | 代價 |
|---|---|---|---|
| **串法 A** | `E(f) → AI NR → g_nr(實數) → min(g_nr, g_res)` | **僅 RNNoise-ERB** | ✅ preset / CNG / 融合 / near-end lift 全部不動 |
| **串法 B** | `E(f) → 先乘 g_res → E'(f) → AI NR(複數 mask) → out` | GTCRN / DFN2 | ⚠ 串聯而非融合；preset floor 移到 AI NR **之後**當 limiter；⚠ **AI NR 的訓練輸入必須是已過 g_res 的訊號**；⚠ CNG 位置需重定 |
| **串法 C** | 複數 mask 拆 `|M|` 與相位，`min(|M|, g_res)` 後再施加相位 | GTCRN / DFN2 | ⚠ 模型不是這樣訓的，需驗證退化程度 |

> ### ⚠ 對 bake-off 的直接影響
>
> **如果 bake-off 贏家是 GTCRN 或 DFN2，串法 A 不可用。**
> 而**串法 B 一旦成立，「傳統 RES + AI NR」就已經半隻腳踏進「AI joint RES+NS」的拓樸**——串聯而非融合、AI 擁有相位。兩者的界線會自己塌掉。
>
> **→ bake-off 的評分標準應把「能否進 `min()`」列為考量項，而不是只看 DNSMOS。**

---

## 6. 收斂後的結構：兩級可組合（rev.3 C1）

### 6.1 兩級組合表

rev.2 的 10 格是**枚舉**。實際上只有兩個選擇點：

```
第一級（AEC）    ：傳統 linear AEC  ／  AI AEC                    ← 2 選 1
第二級（後處理）：傳統 RES+NR ／ AI NR ／ AI RES ／ AI joint RES+NS  ← 4 選 1
```

| # | 第一級 | 第二級 | 抑制歸屬 | 訓練 | 狀態 |
|---|---|---|---|---|---|
| **1** | 傳統 linear AEC | 傳統 RES + NR | 傳統 | 0 | ✅ **已有（基準）** |
| **2** | 傳統 linear AEC | 傳統 RES + **AI NR** | 混合（`min()`） | 1 | 🔄 訓練中 ⚠ **需選串法 A/B/C** |
| **2b** | 傳統 linear AEC | **AI RES** + 傳統 NR | 混合（`min()`） | 1 | 🆕 **見 §6.2** |
| **3** | 傳統 linear AEC | **AI joint RES+NS**（S4） | 網路（I1） | 1 | 文獻支持最強 |
| **4** | **AI AEC**（`(Y,X) → D_hat`） | — / 傳統 RES+NR / AI joint | 依第二級 | 1（+1 若走 `4+3`） | 見 §6.4 · §6.5 |
| **5** | **AI joint AEC+RES+NS** | — | 網路（I1） | 1 | 天花板最高 |

**訓練 6 次，涵蓋全部組合**（⚠ rev.3 誤記為 5 次——`4+3` 需 fine-tune，見 §6.4）。

### 6.2 🆕 C4：#2b —— AI RES + 傳統 NR

⚠ **原稿沒有任何一格是「換掉 RES 卻保留 NR」。** 補上，理由是它針對性最強：

> 本 repo 自己的 **DT over-suppression audit 結論是：RES 的 Wiener gain 才是近端殺手，不是線性級。**

因此 #2b 直接打在已知瓶頸上，而且：

- 它是 #2 的**鏡像** —— 兩者合起來回答「**RES 和 NR 誰是弱點**」
- OM-LSA 連同四個 preset、A_min_pl 耦合**全部保留**
- 仍在 `min()` 之內，架構風險與 #2 同級
- ⚠ 與 §5.3 的 **I3（S1，NN 只出 R²）不同**：#2b 的網路**直接輸出 `g_res`**，傳統 `suppression_gain` 被取代，但 OM-LSA 與 `min()` 保留

### 6.3 ⚠ #3 / #4+3 / #5 需要新接縫 S4

現有接縫：

| 接縫 | 簽章 | 取代 |
|---|---|---|
| **S1** `residual_echo_estimator_t` | `(AecResCtx*) → R²(f)` ＋ `adjust_suppressor(cfg*)` | `ree_estimate()`（介面 I3） |
| **S2** `gain_provider_t` | `(spectrum, extra_noise_psd) → G(f)` | `mmse_lsa_process_gain()` |
| **S3** `filter_control_t` | per-bin step size / Kalman gain | PBFDKF 自適應控制 |

**#3 的網路同時擁有殘餘回音與噪音的抑制決策，必須繞過 `suppression_gain` 與 `min()`。現有 S1/S2 都做不到**——S1 只出估計量，S2 出的 gain 仍要進 `min()`。

```
S4  joint_suppressor_t : (AecResCtx*, spectrum) → 最終 G(f)
    ⚠ 擁有最終 gain。preset floor 與 attenuation cap 仍由 fuser 施加，
      但 min(g_nr, g_res) 這一步不存在。
    ⚠ CNG 需要替代參考量（不再有 g_res）。
```

**這是實質的架構新增，不是重新命名。** #2b 亦可視為 S1' —— 一個直接輸出 `g_res` 的接縫（與 S1 的 I3 介面不同）。

### 6.4 ⚠ V4：#4 的三個評估組態 —— **只有兩個是零訓練**

```
4        AI AEC 單獨                  ← 零訓練。元件本身的品質
4 + 1    AI AEC + 傳統 RES + OM-LSA   ← 零訓練（⚠ 需 D12 的狀態旗標對應，見 §5.8）
4 + 3    AI AEC + AI joint RES+NS     ← ⚠ 需要一次 fine-tune / 重訓，不是零成本
```

#### ⚠ `4 + 3` 為何不是零成本（rev.3 的錯誤）

rev.3 寫「`4 + 3` 直接複用 #3 的第二級模型 → 零額外訓練」。**這與本文 §8.5 自相矛盾**，而 §8.5 是對的。

**#3 的第二級是在傳統 AEC 的輸出上訓練的。** AI AEC 的輸出殘餘分佈與它**完全不同**：

| 差異軸 | 傳統 AEC | AI AEC |
|---|---|---|
| 收斂特性 | 自適應濾波器的收斂軌跡 | 無收斂概念（前饋推論） |
| musical noise | 有其特徵樣態 | 不同樣態 |
| 非線性回音 | 線性級模不到 → 大量殘留 | 模型可部分建模 |
| 相位誤差 | 線性濾波器的相位行為 | 依模型而異 |
| 近端洩漏 | 由 DT 偵測與 step-size 決定 | 由訓練目標決定 |
| path-change transient | 重收斂尖峰 | 無 |

**因此：**

| 用途 | 可否直接接上 |
|---|---|
| smoke test / OOD test | ✅ 可以，而且**應該做**——它直接量出分佈差多遠 |
| **正式結果** | ❌ **必須在 frozen AI AEC 的實際輸出上重訓或 fine-tune** |

**→ `4 + 3` 計為 1 次 fine-tune。訓練總數從 rev.3 的 5 次修正為 6 次。**

#### `4` 與 `4 + 1` 仍是零訓練

砍掉它們省不到訓練，只失去歸因能力：

| 缺了 | 後果 |
|---|---|
| `4` 單獨 | #4 贏了，不知道是第一級還是第二級贏的 |
| `4 + 1` | 缺 2×2 的第四角，測不出交互作用；且無法回答「能否保留 OM-LSA」 |

|  | 傳統後處理 | AI joint 後處理 |
|---|---|---|
| **傳統 AEC** | #1 | **#3** |
| **AI AEC** | **4 + 1** | **4 + 3**（⚠ 需 fine-tune） |

⚠ `4 + 1` 沒有文獻先例（沒人發表過「神經 echo-only AEC → 手寫 OM-LSA」）——正因為沒人做過，自己量才有價值。而 §5.8 顯示 **`D_hat` 契約讓這條路在介面上是可行的**。

### 6.5 ⚠ V3：不是 Cartesian product，是有介面前提的相容圖

rev.3 把「第一級 2 選 1 × 第二級 4 選 1」寫成自由組合。**錯。** 傳統 RES 依賴線性 AEC 提供的：回音估計、error 訊號、ERLE、濾波器狀態統計。**第一級若只提供 enhanced signal，就接不上傳統 RES。**

**改述為介面前提：**

| 第二級 | 需要第一級提供 | 傳統 linear AEC | AI AEC（`D_hat` 契約） | AI AEC（僅出 enhanced） |
|---|---|---|---|---|
| 傳統 RES + NR | `Ŷ`、`E`、ERLE、濾波器狀態旗標 | ✅ 全有 | ⚠ **主要輸入有**（§5.8），狀態旗標需 D12 對應 | ❌ 接不上 |
| AI NR（#2） | `E` | ✅ | ✅ | ✅ |
| AI RES（#2b） | `Ŷ`、`E`、ERLE | ✅ | ⚠ 同上 | ❌ |
| AI joint RES+NS（#3） | `E`、`Ŷ`（可選 `X`） | ✅ | ✅ | ⚠ 只有 `E` |

**⚠ 這張表就是選擇 `D_hat` 契約（§5.7）的第二個理由**：它讓第一級的兩個選項在介面上**近乎等價**，相容圖才會接近完整，可抽換性才是真的。

若第一級只出 enhanced speech，能接的只剩「AI NR」一格——**可抽換性名存實亡**。

---

## 7. 各組態的結構

### 7.1 #1 — 全古典（現況）

```
  mic ─┐                ┌──────────────┐  E(f) ──────────────────────┐
       ├→ se_frontend ─→│ 線性 AEC     │                             │
  ref ─┘   (A1 對齊)    │ PBFDKF+shadow│  Ŷ², ERLE ──→ R² 估計 (ree) │
                        └──────────────┘                │            │
                                    ┌───────────────────┼────────┐   │
                                    ▼                   ▼        ▼   │
                            suppression_gain      OM-LSA NR   near-end
                                → g_res           ξ=S²/(N²+R²)   lift
                                    │                → g_nr        │
                                    └──── min() ────────┘          │
                                              │ g_total ───────────┘
                                    S(f)=E(f)·g_total → CNG(用 g_res) → out
```
保留全部；插槽 0；風險無。

### 7.2 #2 / #2b — 傳統鏈 + 單一插槽

拓樸與 #1 **完全相同**，只有一個方塊換成 NN：

| | 換掉的方塊 | 抑制歸屬 |
|---|---|---|
| **#2** | `OM-LSA` → **AI NR** | 混合（*inside legacy fusion*，仍走 `min()`） |
| **#2b** | `suppression_gain` → **AI RES**（直接輸出 `g_res`） | 混合（OM-LSA 與 `min()` 保留） |
| （探索）**S1** | `ree` → NN R² 估計器 | 傳統（介面 I3，suppressor 仍擁有最終 gain） |

⚠ **#2 的 AI NR 必須在含殘餘回音的資料上訓練**（§5.1）。是否再加 R² 輸入特徵是**獨立**的第二個選擇（D2）。
⚠ **#2 的串接方式取決於模型輸出形式**（§5.6）——串法 A 僅 RNNoise-ERB 適用。

### 7.3 #3 — 線性 AEC + 單一 AI joint RES+NS

```
  mic ─┐                ┌──────────────┐  E(f) ──┐
       ├→ se_frontend ─→│ 線性 AEC     │         │
  ref ─┘   (A1)         │              │  Ŷ², X ─┤
                        └──────────────┘         │
                                    ╔════════════▼══════════════╗
                                    ║ S4 joint AI RES+NS        ║
                                    ║  out: 最終 G(f)（介面 I1）║
                                    ╚════════════╤══════════════╝
                                    fuser: preset floor + attenuation cap
                                                 │
                                       S(f)=E(f)·G → CNG(⚠ 需替代參考量) → out
```

❌ 消失：`suppression_gain`、`min()`、OM-LSA、near-end lift
⚠ CNG 需要新的參考量（不再有 `g_res`）

### 7.4 #4 — AI AEC（`D_hat` 契約）＋ 三個評估組態

```
  Y = S+N+D ─┐              ╔═══════════════════════════╗
             ├→ se_frontend →║ AI AEC                    ║──→ D_hat  估計的回音
  X (ref) ───┘    (A1)       ║  in : (Y, X)              ║      │      （複數頻譜）
                             ║  out: D_hat               ║      │
                             ║  target: D  ⚠ 不是 S+N    ║      │
                             ╚═══════════════════════════╝      │
                                          E = Y − D_hat  ◄──────┘  減法（不是 mask）
                                              │  R = D − D_hat 自然殘留
              ┌───────────────────────────────┤
              │                               │
   4          └→ 直接輸出 E ─────────────────────────────────────→ out
   4 + 1      → ree(|D_hat|², ERLE) → g_res + OM-LSA → min() ────→ out   ⚠ 需 D12
   4 + 3      → #3 的 AI joint RES+NS (S4) ⚠ 需 fine-tune ───────→ out
```

⚠ **輸出是 `D_hat` 不是 mask**（§5.7）——直接對 mic 出 mask 會讓 RES 邊界消失。
⚠ **`R` 不是 target 的一部分**，是 `D − D_hat` 的自然結果（§5.5）。
⚠ **`4 + 3` 需要一次 fine-tune**，不是零成本（§6.4）。
⚠ **`4 + 1` 在 `D_hat` 契約下是可行的**——`ree` 的主要輸入 `s2_linear = |D_hat|²`、`erle` 可外部量測（§5.8），但狀態旗標需 D12 對應。

**⚠ R8 更正 —— 原稿說波形式 E2E 的回音成分「數學上無定義、不可訓」，過度絕對。準確說法：**

1. **僅由最終非線性輸出，echo attribution 不唯一。** 這部分成立。
2. **但模型可以增加 echo waveform / echo mask / echo-power 的 auxiliary head**，用**獨立的 echo stem** 監督。這對波形式模型同樣可行。
3. **mask / deep filter 的 component attribution 也是一個「定義好的 counterfactual convention」，不是物理上唯一的分解。** 原稿把它說成「精確可算」，抬高了它相對於波形式的地位。

**結論修正**：mask-based **不是** R² 輔出頭的必要條件；它只是讓 attribution 有一個現成的慣例。波形式模型走 aux head + 獨立 stem 監督即可。

### 7.5 #5 — 單一 joint AI（AEC + RES + NS）

```
  mic ─┐                ┌─────────────────────────────┐
       ├→ se_frontend ─→│ joint AI: AEC+RES+NS(+derev)│──→ S(f) → out
  ref ─┘   (A1/A2)      └─────────────────────────────┘
```

**⚠ R9 更正 —— 原稿把「#5 後接任何 block」定為編譯錯誤，研究階段過早。**

joint 輸出**設為預設最終輸出**，但下列後處理應保留可能性：

| 允許的後接 | 理由 |
|---|---|
| safety attenuation limiter | 防止模型過度抑制 |
| CNG（由**預測的 noise PSD** 驅動） | joint 模型仍可輸出噪音 PSD |
| classical fallback | 模型失效時的退路 |
| optional refinement ablation | 研究需要 |

**❌ 仍應禁止的**：第二個完整的 NS 級（雙重降噪）。這應是**執行期檢查的預設**，不是型別層的全面封鎖。

---

## 8. 資料集與訓練計畫

### 8.1 必須保存的 stems

| stem | 符號 | 說明 |
|---|---|---|
| `far_render` | `X` | 送到喇叭的參考訊號 |
| **`echo`** | **`D`** | 純回音成分 —— ⚠ **這就是第一級的 target**（§5.5） |
| `near_speech` | `S` | 近端語音（或 early-reflection 版本） |
| **`local_noise`** | `N` | 本地噪音 |
| **`mic_preclip`** | — | 未經削波/AGC 的麥克風混合 |
| **`mic_postclip`** | `Y` | 實際麥克風訊號（削波/AGC 後） |
| **classic AEC 的 linear error / echo estimate** | `E`, `Ŷ` | ⚠ 必須用**出貨 C build**（fast-math 已編進去，無 flag），非 Python |

⚠ **`mic_preclip` / `mic_postclip` 分開存**，才能把削波/AGC 的影響與回音路徑本身的非線性分離。

**metadata（逐 sample，不只全域 `meta.json`）：**

```
sequence_id / chunk_index          ← 跨 chunk 保留 RNN state 用（§8.3）
sample_rate
speaker / noise / RIR IDs          ← source-disjoint split 的依據（§8.7）
SER / SNR / ERL
bulk delay / delay jitter / SRO
speaker nonlinearity
clipping / AGC / codec
generator commit + config hash
frontend commit + config hash      ← postfilter dataset 專用
```

⚠ **殘差分佈是有版本的介面** —— 沒有 frontend 的 commit hash，postfilter dataset 無法重現。

### 8.2 必須覆蓋的場景

far-end only · near-end only · **double-talk** · **reference inactive / dropout** · **echo-path change** · **nonlinear loudspeaker** · **clipping / AGC** · **delay jitter** · **SRO** · **codec / resampling mismatch**

⚠ **`reference inactive / dropout` 是硬性需求**——它同時是 §5.5 的 `λ_idle` loss 項與 §12.3 硬 gate 的訓練/驗證資料。

### 8.3 ⚠ 序列長度與狀態

| 用途 | 長度 |
|---|---|
| standalone AINR | 現有 3 秒可保留 |
| **AEC** | **生成 20–60 秒 parent sequence，再切成連續 3 秒 chunk** |

⚠ **切成 chunk 後必須靠 `sequence_id` / `chunk_index` 保持 RNN state 連續**。
⚠ **若每 3 秒 reset，會低估 convergence、path change 與長時 drift** —— 量到的是冷啟動而非穩態。

AEC convergence、path change、clock drift 的評測需要：
- **更長的連續序列**（20–60 秒），或
- **跨 chunk 保留 streaming state**

⚠ 這同時是評測要求：以 3 秒獨立片段評 AEC 收斂，量到的是冷啟動而非穩態。

### 8.4 ⚠ Target ownership（V1 修正後）

| 組態 | 輸入 | 輸出 | **target** |
|---|---|---|---|
| **AI AEC**（#4 第一級） | `(Y, X)` | **`D_hat`** | **`D`（回音本身）** ⚠ 不是 `S+N` |
| **AI joint RES+NS**（#3、`4+3` 第二級） | `(E, D_hat[, X])` | `S_hat` | `S`（或 early-reflection） |
| **AI NR**（#2 第二級） | `E`（已過 `g_res`，視串法而定 §5.6） | `g_nr` 或 `S_hat` | `S` |
| **AI RES**（#2b） | `(E, Ŷ, ERLE)` | `g_res` | `g_res` 的行為目標 |
| **joint AI AEC+RES+NS**（#5） | `(Y, X)` | `S_hat` | `S` |

⚠ **`R = D − D_hat` 不是任何 target 的一部分** —— 它是估計不完美的結果，不可人工混入（§5.5）。

⚠ 「**見過**」與「**負責**」是兩件事：所有**進 AEC 鏈的第二級**，訓練分佈都必須含殘餘回音，不論它是否負責抑制。⚠ 但這**不適用於 standalone AINR 產品**（§8.6）。

### 8.5 #4 系列的訓練順序

1. 訓 **AI AEC**：`(Y, X) → D_hat`，target = `D`
2. **凍結**第一級，用它在**真實資料上的實際輸出** `(E, D_hat)` 訓練第二級
3. 最後才考慮是否 joint fine-tune

⚠ 第 2 步用「實際輸出」而非理想線性殘差，是避免第二級訓練在推論時不存在的分佈上。
⚠ `4` 與 `4 + 1` 在第 1 步結束後即可評估，**不需要第 2 步**。
⚠ **`4 + 3` 需要第 2 步**——不能直接套用 #3 在傳統 AEC 上訓出來的第二級（§6.4）。

### 8.6 ⚠ V6：資料集範圍 —— standalone AINR 不需要回音

rev.3 寫「AINR 資料集必須加入殘餘回音，不可繞過」，**混淆了兩個範圍**：

| 用途 | 資料集 | 是否需要回音 |
|---|---|---|
| **standalone AINR 產品**（現在正在訓的三個模型） | 現有 `dataset_gen`（speech + noise + RIR） | ❌ **不需要，維持現狀是正確的** |
| **進 AEC 鏈的第二級**（#2 / #3 / `4+3`） | **另一套 postfilter dataset** | ✅ 必須 |

**postfilter dataset 的產生方式**：等 frontend 穩定後，用**特定版本**的 classic AEC 或 frozen AI AEC 的**實際輸出**生成，並保存 frontend 的 commit / config hash。

⚠ 格點變更（320/160 → 512/256）**不需要重新生成 waveform 資料集**（`dataset_gen` 存的是波形），但需要：重新訓練、重算 feature normalization / init、重新校準 loss 與 band mapping、把舊 checkpoint 判定為不相容。
✅ 現有 `dataset_gen/config.example.ini:13` 已支援 `sr` 選擇（一次一種取樣率）。

### 8.7 ⚠ V7：現有 split 有資料洩漏風險

✅ `AINR/dataset_gen/loader.py` 的 `locality_preserving_random_split` 是對**已生成的 item** 做隨機排列。由於生成過程是 speech × noise × RIR 的組合，**同一個 speaker / speech source / noise / RIR 可能同時出現在 train 與 validation**。

**建議改為生成前先固定 manifest，並保證：**

- speaker-disjoint
- speech-file-disjoint
- noise-disjoint
- RIR / room-disjoint
- device / nonlinearity-disjoint

**且每個 sample 保存 provenance**（不只是全域 `meta.json`）。

⚠ 此問題**現在就存在於 standalone AINR 的 bake-off**——RNNoise-ERB vs GTCRN 的比較若有洩漏，兩邊都被高估，相對排名未必受影響但絕對數字不可信。

### 8.8 ⚠ 建議的 generator 拆分

不要在現有 NR generator 上加 `task=aec` 分支：

```
dataset_gen/common/       共用的 RIR / SNR / resample / manifest
dataset_gen/nr/           現有（speech + noise），standalone AINR 用
dataset_gen/aec/          新增：stems + 長序列 + 場景
dataset_gen/postfilter/   新增：由特定版本 frontend 的實際輸出生成
```

---

## 9. ⚠ 多通道：資料生成前必須決定

原稿只把四通道列為「未查證」。**不夠。** 若最終是 mic array，下列決定會直接改變 model input shape、RIR 數量、compute 與 spatial coherence，**必須在資料生成前定案**：

| 決策 | 選項 |
|---|---|
| 拓樸 | 每支 mic 先 AEC 再 beamforming ／ 先 beamforming 再單路 AEC ／ multichannel NN AEC |
| Echo path | 單喇叭（SISO）／ 多喇叭（MIMO） |

**若第一階段只做 mono，文件必須明寫**：

> 第一階段**排除 multichannel AEC**，但保留介面（stems 與 metadata 以可擴充為多通道的形式儲存）。

---

## 10. 待決策清單

**⚠ rev.3 的三個當務之急**（reviewer 請優先檢視）：

| # | 決策 | 現況 | 為何是當務之急 |
|---|---|---|---|
| **D-a** | **#2 的串接方式 A / B / C**（§5.6） | **未定** | 取決於 bake-off 贏家。GTCRN 是 Complex Ratio Mask，**進不了 `min()`** ⚠ 這也表示 **bake-off 的評分標準應把「能否進 `min()`」列為考量項** |
| **D-b** | **`dataset_gen/aec/` 規格凍結** | **未定** | #2~#5 全部卡在這 |
| **D-c** | **AI AEC 的輸入/輸出/target 契約**（§5.5、§5.7） | 建議 `(Y,X) → D_hat`，target = `D` | ⚠ **rev.4 的核心修正**。決定相容圖是否完整（§6.5）、`4+1` 是否可行（§5.8）、能否分項歸因（§5.7） |

**其餘決策：**

| # | 決策 | 現況 | 影響 |
|---|---|---|---|
| **D1** | 格點：§1.1 的 (i)/(ii)/(iii) | 傾向 (i) | ⚠ (i) 打破現有的跨率時間幾何一致性 |
| **D2** | #2 的 AI NR 是否加 R² 輸入特徵 | 未定 | ⚠ **與「資料集含殘餘回音」是獨立決策**，後者不可省 |
| **D3** | 對齊模式 A1/A2/A3 | 建議 A1 | 決定 delay estimator 是否共用 |
| **D4** | 48k delay：16k sidechain vs 全面重推常數 | 建議 sidechain | 見 §3.3 |
| **D5** | 模組分開訓練 vs 共用 trunk | 傾向分開 | 代價是 N 次加速器往返（❓未量測） |
| **D6** | 三個繞過 `fft_density_scale` 的常數 | 未定 | #1 就會撞上 |
| **D7** | 第一輪跑哪些 | **已收斂**：#1 / #2 / #2b / #3 / #4(+1,+3) / #5 | 見 §12.1 |
| **D8** | 是否新增接縫 S4 | #3 / `4+3` / #5 皆需要 | §6.3，架構實質新增 |
| **D9** | 多通道拓樸，或明確排除 | 未定 | ⚠ 資料生成前必須定案 |
| **D10** | 延遲 gate 的門檻值 | 未定 | §1.5，可能與 20 ms 業界慣例衝突 |
| **D11** | 失去 `g_res` 後 CNG 的替代參考量 | 未定 | #3 / `4+3` / #5 皆需要 |
| **D12** | 🆕 **`D_hat` → `ree` 的狀態旗標對應表** | 未定 | §5.8。`s2_linear`/`erle` 可得，但 `usable`/`saturated`/`filter_delay_blocks` 等需替代估計或保守預設。**決定 `4+1` 能否成立** |
| **D13** | 🆕 **source-disjoint split** | 未定 | §8.7。⚠ **現在就影響 standalone AINR 的 bake-off 絕對數字** |
| **D14** | 🆕 單一 `AudioGridSpec` | 未定 | §1.6。三份重複推導，格點遷移的前置 |

---

## 11. 未查證項

1. ❓ 古典鏈成本 X（`make bench` 可得 —— `AEC/c_impl/Makefile:658` 有 target，但 15 個 config 目錄皆無 binary ✅）
2. ❓ **加速器算子支援清單**，尤其 GRU 與複數表達。**不支援會無聲退回 CPU 並吃掉整個預算**
3. ❓ 加速器單次呼叫開銷（16k @ 62.5 Hz / 48k @ 93.75 Hz）
4. ❓ **實測串流延遲**（§1.5）
5. ❓ **193-tap 假設的四項測試**（§1.4）
6. ❓ 48 kHz 真寬頻品質認證（現有 48k 測試訊號皆為 16k 上採樣）
7. ❓ 多通道硬體前提：取樣同步、SRO、逐通道校正、麥克風匹配
8. ❓ **S1（I3 介面）零同儕審查先例**（§13.4）

---

## 12. 建議執行順序

```
① 修兩個接縫 bug（near_spec 文件 ×2、int16² 縮放進型別）      ← 零風險
② make bench 量 X；查加速器算子支援清單                        ← 一天，砍掉一整欄
③ 古典鏈重新格點化：16k 512/256、48k 1024/512
   ⚠ 所有常數由秒/Hz 推導（MCRA L、leakage、POOR_EXC、D6）
④ 48k delay 改固定 16k sidechain；test_rate_structural 接進 CI
   ⚠ 再做真實 fullband 品質認證
⑤ §1.4 的四項混疊測試 + §1.5 的延遲量測 → 格點正式定案
⑥ dataset_gen/aec/：stems + 長時 streaming + 場景覆蓋 + target ownership
⑦ 新增接縫 S4 + tagged union
⑧ 第一輪模型
```

### 12.1 第一輪範圍（rev.3 收斂）

| # | 組態 | 訓練次數 | 角色 |
|---|---|---|---|
| **1** | 傳統 AEC/RES + 傳統 NR | **0** | ✅ 基準 |
| **2** | 傳統 AEC/RES + AI NR | 1 | 🔄 訓練中 ⚠ 需選串法（D-a） |
| **2b** | 傳統 AEC + **AI RES** + 傳統 NR | 1 | 🆕 打在已知的近端殺手上；#2 的鏡像 |
| **3** | 傳統 AEC + AI joint RES+NS | 1 | 文獻支持最強（§13.4） |
| **4** | AI AEC（`(Y,X) → D_hat`） | 1 | 元件本身 |
| **4 + 1** | AI AEC + 傳統 RES + OM-LSA | **0** | 免費 baseline ⚠ 需 D12 |
| **4 + 3** | AI AEC + AI joint RES+NS | **1（fine-tune）** | ⚠ **V4：不是零成本**（§6.4） |
| **5** | joint AI AEC+RES+NS | 1 | 天花板最高 |

**訓練 6 次，涵蓋 8 個組態。**（rev.3 誤記為 5 次——`4+3` 需 fine-tune）

⚠ **只有 `4` 與 `4+1` 是零訓練**（第一級 checkpoint 的評估組態）。砍掉省不到訓練，只失去歸因能力（§6.4）。

⚠ **#2b 不可省**：本 repo 的 DT over-suppression audit 指出 **RES 才是近端殺手**；沒有 #2b 就無法區分「RES 是弱點」與「NR 是弱點」。

### 12.3 🆕 standalone AI AEC 的驗證 gate

**硬性 gate（不通過就不算完成）：**

> **`ref = 0` → `output ≈ mic`**

遠端靜默時模型不得憑空產生回音估計。⚠ 這與 §5.5 的 `λ_idle · ‖D_hat‖` loss 項是同一件事的訓練側與驗證側，且需要 §8.2 的 `reference inactive / dropout` 場景資料。

**其餘量測項：**

| 項目 | 條件 |
|---|---|
| ERLE | far-end only |
| near-end preservation | near-end only |
| DT 近端劣化 | double-talk ⚠ **本專案的束縛點** |
| 重收斂行為 | echo-path change |
| 穩健性 | delay jitter / SRO |
| RTF / 實測延遲 | §1.5 |

⚠ **`4 + 3` 的第一步應該是「直接接上 #3 的第二級跑 OOD test」**——不是為了當正式結果，而是**直接量出兩種 frontend 的殘餘分佈差多遠**。這個數字決定 fine-tune 的必要規模。

### 12.2 rev.3 C5：砍出第一輪的項目

| 項目 | 決定 | ⚠ 砍掉會失去什麼 |
|---|---|---|
| **S1**（NN R² 估計 → 傳統 suppressor，介面 I3） | **砍出路線圖**，保留為文件記錄 | 唯一「DSP 決策全保留」的選項，是最小變動的 fallback。⚠ 但零同儕審查先例、無可對標基準（§13.4），且 §13.5 那兩個沒有參照可抄的消費者重調成本高 |
| **NN 濾波器控制（S3）** —— 單獨或搭 AI 後處理 | **砍出第一輪** | **後濾波器修不了收斂問題**。S3 針對的是 path change、delay 重取得、冷啟動收斂速度——#2~#5 全部碰不到的軸。⚠ 本專案目前瓶頸是 DT 近端品質不是收斂（delay arc 已結案為「tracker 沒壞」），故可延後 |

> ### ⚠ 砍掉 S3 不表示可以砍掉對應的資料
>
> 資料集規格中的 **echo-path change / delay jitter / SRO** 場景**必須保留**（§8.2）。
> 理由有二：① 資料重生很貴；② 那些場景是 #2~#5 **穩健性評測**的共同需求，不是只有 S3 用。

第二輪：neural adaptive-filter control（S3）。NKF（5.3K params, RTF 0.09）、Fraunhofer adaptation-control 都是獨立且有研究支持的家族。

---

## 13. 文獻查證（2026-07-29）

### 13.1 方法

53 個代理：4 家族平行搜索 → 每個模型由**兩個不同視角**嘗試駁斥（來源/拓樸、數字/歸屬）→ 完整性批判。
**CONFIRMED 28 · PARTIALLY_WRONG 20 · REFUTED 0。** 20 筆多為書目與**歸屬**層級（第三方數字被當成作者數字）。

### 13.2 格點的文獻對照

16 kHz / 512 / 256 是該取樣率下最常見的配置：

| 模型 | 格點 |
|---|---|
| EchoFree | 16 kHz, window 512, hop 256, FFT 512 |
| ULCNet-AENR / Align-ULCNet | 16 kHz, NFFT 512, window 32 ms, hop 16 ms |
| FCRN DNN AEC（echo-only 變體） | 16 kHz, K=512, shift 256 |
| Multi-Input FCRN | 16 kHz, K=512, R=256 |

⚠ 對照：`Seidel et al. ICASSP 2024` 用 window 1024 / DFT K=512 / **shift 128**；`NKF` 用 FFT 1024 / hop 256。hop 256 是主流，窗長有 512 與 1024 兩派。

### 13.3 ⚠ echo-only E2E 存在，且 task-split 有直接量測支持

原稿假設「純回音 E2E 幾乎不存在」，**被推翻**。該家族 16 個成員，多個有明確 echo-only 訓練目標：

| 模型 | 證據 | 規模 |
|---|---|---|
| **DAEC**（Braun & Valero, Microsoft, IWAENC 2022）"Task Splitting" | **實測**單獨評估 DAEC 級：不移除噪音、無近端訊號劣化 | DAEC-64 ≈ 2.7M MACs |
| **gGCRN16**（WASPAA 2023） | loss 為 logMSE 對 `s_b(n)+n_b(n)`（**含噪近端**） | 1.3M / 583 MFLOPS |
| **FCRN DNN AEC**（ICASSP 2021） | echo-target 與 noisy-speech-target 兩變體**分開回報** | 5.2M（EarlyF） |
| **ICASSP 2021 AEC Challenge baseline** | 主辦方自述 echo-only | 未載明 |
| **NKF-AEC**（ICASSP 2023） | `Ŝ = D − D̂`，近端噪音依構造通過 | **5.3K params**, RTF 0.09 |
| **NLAEC**（Interspeech 2021） | `ŝ = s + (y−ŷ) + w`，噪音項明確保留 | 17K / 500 MFLOPS |

**`4 + 3` 的直接證據** —— Braun & Valero 量測「拆分 vs 聯合」，**拆分勝出**：

> "using a separate echo cancellation module and a module for noise and residual echo removal results in **less near-end speech distortion and better performance during double-talk at the same complexity**"

> "the AEC module is removing only echo, which creates **no significant signal distortion** in contrast to echo and noise suppressors"

⚠ 雙講近端劣化正是本 repo 已知的硬牆。
⚠ **他們的第二級是訓練出來的網路，不是古典 NR。`4 + 1`（AI AEC + 古典 OM-LSA）沒有先例**——這正是 `4 + 1` 必須作為 baseline 跑的原因。
⚠ 同時，Braun & Valero 的拓樸正是本文採用的分工（第二級負責 noise **and residual echo**），這是 §5.5 契約的主要依據。

### 13.4 ⚠ 文獻支持的正確歸屬（R3 更正）

**原稿把 hybrid postfilter 的文獻支持套到 #2（AI NR inside legacy fusion）。錯誤。**

| | 文獻的 hybrid | rev.3 的 #2 |
|---|---|---|
| 拓樸 | 線性 AEC → **一個** NN postfilter 同時處理殘餘回音 + 噪音 | 線性 AEC → 傳統 RES → **另一個** AI NR → `min()` |
| 抑制歸屬 | 網路（I1） | 混合 |
| 對應組態 | **#3** | #2 |

**所以文獻支持最強的是 #3，不是 #2。**

⚠ 但注意 §5.6 的推論：**若 bake-off 贏家是複數 mask 模型（GTCRN / DFN2），#2 只能走串法 B（串聯），而串法 B 的拓樸已經接近 #3。** 兩者的界線在實作上會自己塌掉——這反而降低了「#2 用錯文獻支持」的實務影響，但**不改變兩者是不同拓樸的事實**。

**S1（I3：NN 估計量 → 不變的傳統 suppressor）零同儕審查成員：**

1. 唯一成員（本 repo 快照的出貨原始碼）**未發表**：無論文、無參數量、無 MAC、tree 內**無 `.tflite`**——沒有可對標的東西
2. **已發表的 hybrid 分類法沒有這個格子。** `Seidel et al., ICASSP 2024` §1 逐字：
   > "(2) Hybrid methods are themselves grouped into two categories: (i) a combined linear echo canceller (LEC) followed by an NN postfilter (PF) as residual echo and/or noise suppressor, or (ii) an NN-aided step-size control or state estimation for the LEC."
3. 四次獨立搜索「NN → 殘餘回音 PSD → 傳統抑制器」只找回**非神經**的 PSD 估計器

⚠ 批判進一步主張 S1「只是後濾波器家族的重新參數化」。**本文評估：分類論證只對一半。** 形式上網路確實輸出 mask（`R² = E²×mask`），但**最終施加的 gain 是 DSP 用自己的 ENR/EMR、nearend-dominance、min-gain floor 算的**——preset、CNG、`min()` 都活在那裡。「輸出 mask 還是 PSD」是參數化細節；「誰擁有決策」才是架構問題。**但實務結論不受影響：我們會是第一個，沒有基準。**

### 13.5 參照實作（S1 用） 📄

`AEC/docs/aec3_extracts/src/aec3/neural_residual_echo_estimator/`（含 `.proto` 與兩份 unittest）

**輸入 / 輸出**（`neural_feature_extractor.h:26-40`）：
```cpp
enum class ModelInputEnum  { kMic=0, kLinearAecOutput=1, kAecRef=2, kModelState=3 };
enum class ModelOutputEnum { kEchoMask=0, kUnboundedEchoMask=1, kModelState=2 };
```

**推論速率** ✅：`kSupportedFrameSizeSamples = {256}`（`_impl.cc:49`）、`step_size_(frame_size_/2)`（`_impl.cc:279`）→ **frame 256 / step 128**。

**輸出轉換**（`_impl.cc:85-106`）：
```cpp
downsampled_mask[i] = max_element(&mask[factor*(i-1)+1], factor);   // 逐帶取 max
m = 1.0f - (1.0f - m) * (1.0f - m);                                 // nearend → echo 功率佔比
```

**R²**（`_impl.cc:560`）：`R2[k] = E2[k] * mask[k];` ⚠ **尺度無關比值** —— 上游 gain 重調、int16² 縮放、preset 換檔皆不移動模型輸出分佈。

**`AdjustConfig`**（`_impl.cc:569-589`）⚠ 簽章是 **Suppressor-only**（`_impl.h:83` 回傳 `EchoCanceller3Config::Suppressor`）：
```cpp
enr_transparent=0.0f; enr_suppress=1.0f; emr_transparent=0.3f;
max_inc_factor=100.0f; max_dec_factor_lf=0.0f; nearend_average_blocks=1;
dominant_nearend_detection.{enr_threshold=0.5f, trigger_threshold=2};
high_frequency_suppression.{limiting_gain_band=24, bands_in_limiting_gain=3};
```

⚠ **參照的上游沒有獨立 NR**（`aec3_extracts/src/aec3/` 下只有 `suppression_gain`/`suppression_filter`/`comfort_noise_generator` ✅）。噪音在其中是**遮蔽門檻**（`suppression_gain.cc:215-233`：`emr = echo[k]/(masker[k]+1.f)`，`masker` = comfort_noise），不是被抑制的對象。
**所以 `AdjustConfig` 只重調 §4.2 三個消費者中的一個。** 另外兩支需自行推導。

### 13.6 ⚠ 第三方重現問題

**DeepVQE-S 有四個互不相同的參數量：**

| 來源 | 參數量 | MAC | 性質 |
|---|---|---|---|
| Microsoft（原作者） | **0.59M** | 未載明（0.14 ms/frame） | 原始 |
| Seidel et al. ICASSP 2024 | 0.72M | 2170 MMACs/s | 重訓，**自述拿掉對齊 self-attention** |
| EchoFree | 0.82M（+39%） | 315 MMACs/s | 重訓，506 h 訓練資料 |
| DiffVQE（full DeepVQE） | 5.29M（原 7.5M） | 42.24 GFLOPS | 重訓 |

其他歸屬更正：**DeepVQE 未參加 ICASSP 2023 挑戰賽**（論文腳註 1）；DeepVQE 的 4.02 GMACs 是 Fraunhofer 的；**MTFAA 作者自述參數量無法取得**（付費牆），流傳的 1.5M/RTF 0.60 是主辦方回報；**GFTNN 提交用 wRLS 不是 MDF**；`DNN-FDAF` 是 **ICASSP 2022**；`Higher-Order Meta-AF` 是 **IWAENC 2022**。

### 13.7 候選（依相關度）

| 模型 | 為何相關 | 作者自述數字 |
|---|---|---|
| **Seidel et al., ICASSP 2024** | **#3 的最佳參照**：NLMS + Bark 後濾波器，與我們的 ERB/Bark 管線同構；唯一發表 hybrid vs E2E 頭對頭 MACs | 1.58M / **235 MMACs/s** / RTF 0.22%；16 kHz, window 1024, DFT K=512, shift 128 |
| **EchoFree** | ⚠ **V5 更正：它是 #3 的參照，不是 standalone AI AEC。** 拓樸 = **PBFDKF + Bark 神經後濾波器**——**用的就是我們同款 PBFDKF**，因此是 #3 最直接的參照。"echo-only" 指的是它不做降噪（任務範圍），不是指它取代線性濾波器。格點完全吻合目標 16k | 278K / **30 MMACs/s**；16 kHz/512/256；GRU 192 |
| **FCRN echo estimator** | 🆕 **#4（standalone AI AEC）的首選參照**：輸入 mic/reference 複數頻譜，**輸出複數回音估計**，正是 §5.7 的 `D_hat` 契約。格點 16 kHz / 512 / 256 **與目標完全一致** | 5.2M（EarlyF）；ConvLSTM |
| **CRUSE-DAEC-32** | 🆕 #4 的產品化候選，小模型。`in = [Re(Y),Im(Y),Re(X),Im(X)] → out = [Re(D_hat),Im(D_hat)]`，`E = Y − D_hat` | DAEC-64 ≈ 2.7M MACs；grouped GRU |
| **Align-ULCNet / ULCNet-AENR** | 格點吻合；聯合 AEC+NS | 0.69M / **100 MMACs/s**；16 kHz/NFFT 512/hop 16 ms |
| **DAEC + NRES**（Braun & Valero） | **`4 + 3` 的直接參照** | DAEC-64 ≈ 2.7M MACs；16 kHz, 20 ms sqrt-Hann |
| **NKF-AEC** | **S3（第二輪）的參照**；極小 | **5.3K params**, RTF 0.09；16 kHz, FFT 1024, hop 256 |
| **Braun & Valero, TASLP 2024** + `EC-Evaluation-Toolbox` | **唯一針對這幾條路的同儕審查頭對頭**，測收斂、RIR 切換、延遲回音 | ⚠ 付費牆未開啟，confidence medium-low |

⚠ **未涵蓋的家族**：Ohio State（Hao Zhang / DeLiang Wang）系、network 內建對齊（cross-attention）系、個人化 AEC、擴散式 AEC。前兩者與本專案相關。

---

## 附錄 A：`file:line` 索引

| 主張 | 位置 |
|---|---|
| 現況格點計算 | `pipelines/pipeline_dims.h:27-35` |
| 48k rate 常數 513/1024/960/480 | `AEC/c_impl/include/aec3_balanced_config.h:207-210` |
| rate table | `AEC/c_impl/include/aec3_balanced_config.h:302+` |
| **48 kHz 白名單已開** | `AEC/c_impl/src/aec.c:115-123` |
| **structural test 存在但未接 Makefile** | `AEC/c_impl/test/test_rate_structural.c` |
| **NR MCRA `L = 32` = 320 ms** | `NR/c_impl/include/mmse_lsa_types.h:126` |
| NR 接受 8/16/48 kHz | `NR/c_impl/include/mmse_lsa_types.h:34` |
| **delay 固定幾何（無 sample_rate）** | `AEC/c_impl/include/delay_aec3.h:95-105` |
| hop 轉換函式 | `AEC/c_impl/include/aec3_scale.h:31-40` |
| leakage 係數（×2.5） | `AEC/c_impl/include/aec3_scale.h:54-57` |
| `POOR_EXC_COUNTER_INITIAL_HOPS` | `AEC/c_impl/include/aec3_scale.h:60` |
| 繞過 `fft_density_scale` 的三常數 | `AEC/c_impl/include/aec3_scale.h:80,81,83` |
| R² 公式 | `AEC/c_impl/src/residual_echo_estimator.c:292` |
| `ree_estimate` 輸入 | `AEC/c_impl/src/residual_echo_estimator.c:245-256` |
| R² → NR 的 `extra` / `legacy_amin` | `pipelines/audio_pipeline.c:663-668` |
| `min()` 融合 | `pipelines/audio_pipeline.c:678` |
| near-end lift | `pipelines/audio_pipeline.c:704-714` |
| CNG 用 `res_gain` | `pipelines/audio_pipeline.c:722-730` |
| `AecResContext` | `AEC/c_impl/include/aec.h:490-521` |
| `near_spec` 實際來源 | `AEC/c_impl/src/aec.c:2477` + `src/pbfdkf.c:403` |
| `error_spec` 才是 E(f) | `AEC/c_impl/src/aec.c:2485` + `include/aec.h:506` |
| 文件錯誤 ×2 | `AEC/docs/nn_integration_interface.md:20`、`Audio_ALG/docs/freq_domain_pipeline_design.md:19` |
| `bench` target | `AEC/c_impl/Makefile:658` |
| 參照實作 I/O | `AEC/docs/aec3_extracts/.../neural_feature_extractor.h:26-40` |
| 參照 frame/step | `AEC/docs/aec3_extracts/.../neural_residual_echo_estimator_impl.cc:49,279` |
| 參照 mask 轉換 | `AEC/docs/aec3_extracts/.../neural_residual_echo_estimator_impl.cc:85-106` |
| 參照 `R²=E²×mask` | `AEC/docs/aec3_extracts/.../neural_residual_echo_estimator_impl.cc:560` |
| 參照 `AdjustConfig`（Suppressor-only） | `..._impl.cc:569-589` + `..._impl.h:83` |
| 參照上游無 NR；噪音為 masker | `AEC/docs/aec3_extracts/src/aec3/suppression_gain.cc:215-233` |
| AINR 資料集無回音 | `AINR/dataset_gen/dataset.py`（`"echo"` 0 次） |
| AINR 16k 格點 | `AINR/GTCRN/config.ini:2-5`、`AINR/RNNoise-ERB/config.ini:2-3` |
| AINR 48k 格點 | `AINR/DeepFilterNet2/config.ini:2-5` |
| **RNNoise-ERB 輸出實數 gains** | `AINR/RNNoise-ERB/train.py:514,567` |
| **GTCRN 輸出 Complex Ratio Mask** | `AINR/GTCRN/model.py:276-281`（`forward` 於 `:305`，回傳於 `:330`） |
| **DFN2 cascade + alpha（全頻 mask / 低頻複數 DF residual mix）** | `AINR/DeepFilterNet2/model.py::DeepFilterNet2.compose` |
| **DFN3 平行 band split（高頻實數 mask / 低頻複數 deep filter）** | `AINR/DeepFilterNet3/model.py::DeepFilterNet3.compose` |
