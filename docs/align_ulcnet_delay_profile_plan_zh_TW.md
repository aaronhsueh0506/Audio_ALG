# Align-ULCNet delay profile 實作與驗證計畫

## 0. 最終範圍裁定

使用者已用既有 checkpoint 完成 raw/aligned far 與 small-D sweep，並判定
輸出無實質差異。這份證據在本專案中視為接受：D 可於 export 時直接調整，
aligned far 可作為部署輸入；兩者不再重跑泛用分數，也不再改訓練 contract。

本輪唯一新增功能是 **matched-filter n 的產品化調整**：依產品 bulk-delay
範圍選 n，讓 AEC init/pool、mono/4ch wrapper 與診斷工具真正使用該值。
dataset generation 固定維持既有 n=5 contract，既有 WAV、packed data、
checkpoint 與 weights 全部不動。

## 1. 決策與適用範圍

本計畫把產品的 delay 能力拆成三個彼此獨立的 init/export 預算：

| 參數 | 決定時機 | 責任 | 改動後是否要重新配置 |
|---|---|---|---|
| matched-filter `n` (`delay_num_filters`) | AEC init | 搜尋並鎖定 bulk far-to-mic delay | 要重新查詢 AEC pool、重新 init；不改 NN 權重 |
| PBFDKF `filter_length` | AEC init | reference 已對齊後，建模 causal echo path / RIR tail | 要重新查詢 AEC pool、重新 init；不改 NN tensor shape |
| Align-ULCNet `D` (`max_delay_frames`) | ONNX export | TA 在送入 NN 的 far 上搜尋 current/past frame | 要輸出另一份 ONNX/descriptor 並重建 state pool；權重可沿用 |

產品化方向成立：在產品 bulk delay 落於 matched-filter 可可靠搜尋的範圍時，
可以縮小 `n`；再將 NN far branch 從 `raw_far` 改成 AEC 實際使用的
`aligned_far`，使 TA 只需處理對齊後的 residual frame lag，因而有機會縮小
`D`。

但需注意：

1. `D` 的格點是 model hop。固定 grid 為 16 kHz、FFT/frame 512、hop 256，
   因此一格是 16 ms；它不是 sample-level fractional/fine delay aligner。
2. matched estimator 與 AEC reference ring 負責 bulk/sample-level reference
   alignment；PBFDKF 責責 linear echo-path modelling；TA 處理 NN far feature
   尚存的整幀差異與短期變動。
3. 三個範圍不能相加成一個「總 delay coverage」。每一層都必須各自滿足
   前一層交付的輸入條件。
4. checkpoint 的 `raw_far` 是 training provenance。使用者完成 sweep 後已
   決定 production 固定讀 aligned-far seam；export descriptor 因此固定
   `aligned_far`，pipeline 不提供 runtime mode；不需為此重訓。
5. `n` 與 `D` 都是 init/export profile，不允許逐 hop 熱切換。切換 profile
   必須停止串流、重建 AEC/model state，並清除 STFT/WOLA、K/V、logit、GRU
   與 delay-generation 相關狀態。
6. `n` 與 `D` 都不要求重新訓練：n 不屬於 NN graph；Align-ULCNet 的
   trainable weight shapes 也不依賴 D。只要使用者明確選定 deployment
   profile，相同 checkpoint weights 可以建立不同 n 的 frontend，並 export
   成不同 D 的 ONNX。這是允許的行為變更，不是 contract error。

## 2. 正確訊號關係

以下兩條 far branch 只用於說明 sweep 對照；production 固定走
`aligned_far` seam，沒有 runtime mode switch：

```text
raw far ───────────────┬───────────────────────────────┐
                       │                               │
mic ── matched delay estimator (n)                    │ raw_far mode
                       │                               │
                       v                               v
              applied delay/reference ring       NN far STFT
                       │
                       ├── aligned_far ────────────────┐ aligned_far mode
                       │                               v
                       v                         NN far STFT
                  PBFDKF(filter_length)
                       │
                       v
                  linear_error STFT ──────────> Align-ULCNet TA (D)
```

必要條件：

```text
MATCHED:
  product bulk delay 必須落在 n 的可靠搜尋範圍內

PBFDKF:
  applied reference 不得晚於真正 echo（不可形成負 residual）
  positive residual + 必須建模的 echo tail 必須落在 filter span 內

raw_far model input:
  model 看見的總 far/error frame lag <= (D - 1) * 16 ms

aligned_far model input:
  對齊後 residual frame lag <= (D - 1) * 16 ms
```

`raw_far` 下，matched filter 雖然能讓 PBFDKF 正常消回音，但不會縮短 NN
far branch 看見的 raw far-to-error lag，因此不能用 `n` 的範圍補足小 `D`。
`aligned_far` 下，NN 才能直接享有 matched/reference alignment 的結果。

## 3. 建議的驗證順序

不得同時先改 `n`、far mode 與 `D`。依下列順序建立可歸因證據：

1. **基準**：`n=5 + raw_far + D=64`，固定 16 kHz/512/256。
2. **只縮 n**：維持 `raw_far + D=64`，逐一驗證 `n=4/3/2/1` 的 delay
   acquisition、steady-state linear output、path-change 與 memory/CPU。
3. **既有結論（不重跑）**：raw/aligned far 與 small-D sweep 已完成並由
   使用者接受；本輪只保留 C descriptor、時間戳、lock gate、reset 與輸入
   接線的結構驗證。
4. **只縮 D**：已完成，不再以泛用分數重跑；export 時建立選定 D 的 graph、
   descriptor 與 state pool。
5. **組合 profile**：只在前四步各自通過後，組合選定的 `n + D`，執行
   route/clock/path-change 壓力測試。

這個順序可區分「matched search 不足」、「aligned far 時間軸錯誤」與
「TA history 太短」，避免用音質結果反推錯誤根因。

### 3.1 既有證據與尚缺證據

先前的 `sweep_delay_depth.py` 已經用同一份 checkpoint 測過
`D=64/32/16/8/4/2/1`，也跑過 `raw_far` 與 `aligned_far` 路徑；這可作為
「相同權重能以較小 D 執行」及 D-only 波形差異的既有證據，不需在本輪
重複執行。

工具現已支援 `--delay-num-filters`，並從 live AEC instance 讀回 resolved n；
checkpoint linear contract 維持不記錄 n，dataset generation 固定 n=5。

因此目前狀態是：

- **已驗證**：`n=5 + small D`；`n=5` 下的 raw/aligned diagnostic；n 本身的
  AEC init、搜尋與 RAM scaling。
- **產品端仍須量測**：small n 是否涵蓋各 SKU/route 的 bulk-delay 分佈；這是
  delay/liveness gate，不是 checkpoint 品質或重訓 gate。
- **實作端已完成**：production 移除 runtime far mode、ALIGNED timestamp、
  UNLOCKED 仍套用模型、MATCHED/FIXED alignment-transition reset 與 C
  descriptor validation。Exporter 目前輸出 ONNX + JSON descriptor；若板端
  build 要直接 include generated C header，仍需再加 header generator。

## 4. 無 checkpoint 即可完成的實作

### 4.1 固定並驗證 signal grid

- `mono_alignulcnet` 與 `4ch_alignulcnet` 預設必須明確寫入：
  `sample_rate=16000`、`fft_size=512`、`frame/window=512`、`hop=256`。
- 不得只給 sample rate 再由 library 猜 FFT；init 必須收到或解析出完整
  grid，且僅接受已支援的組合。
- config、C descriptor、ONNX metadata、C pre/post 與 README 必須使用相同
  名稱與數值。
- 新增負向測試：16 kHz/256 FFT、48 kHz/1024 FFT、錯誤 hop、descriptor
  grid mismatch 均應 fail-fast，不可 silent fallback。

### 4.2 將 n 做成真正的產品 init 參數

- `delay_num_filters` 僅在 `MATCHED` 合法，範圍 `1..5`；`0` 不代表關閉。
- `FIXED` 與 `EXTERNAL_ALIGNED` 走各自 mode，不建立 matched bank。
- `get_mem_size()` 與 `init()` 必須使用同一個 resolved
  `(sample_rate, hop_size, n)`，carve 後做 lockstep/end-cursor 檢查。
- 修改 `n` 後必須重新查詢 pool 並 init；API/README 不得暗示 runtime
  setter。
- mono 與 4ch wrapper 都要傳遞產品指定的 n。4ch 只建立一個 shared
  estimator，不能退回每 lane 一份。
- 應用範例不得把 n 藏在 source literal；從單一 product config/descriptor
  載入並打印 resolved profile。

### 4.3 釐清 model ABI 與產品 tuning provenance

嚴格 runtime ABI、訓練 provenance 與 deployment tuning 要分開：

- **同一份 ONNX/runtime 必須完全相等才可 init**：sample rate、
  FFT/frame/hop、exported `D`、state-layout version、tensor
  shapes/dtypes/order、`far_input_mode`。這是 graph/state ABI，不是要求
  exported D 必須等於 checkpoint 訓練時的 D。
- **產品 init/profile 欄位**：delay mode、n、fixed delay、max delay、PBFDKF
  filter length。它們需記錄並打印，但不應假裝是 ONNX tensor shape。
- **訓練 provenance**：訓練資料使用的 AEC behavior hash、n、filter length、
  training D 與 far-input semantic。
- **相容性規則**：明確的 deployment override 可以讓 runtime n/D 與訓練
  provenance 不同；loader/exporter 應記錄 `training_n/training_D` 與
  `runtime_n/export_D`，但不得因此拒絕 checkpoint。沒有明確 override 時
  才沿用 checkpoint provenance。far branch 不是產品 tuning knob：export
  metadata 分開記錄 `training_far_input_mode`，部署 descriptor 固定寫
  `far_input_mode=aligned_far`；不得默默把 raw far 接到 production graph。

- **calibration artifact 也在同一條 ABI 鏈上**：
  `AIAEC/Align_ULCNet/inference.py calib` 產生的 report 記錄
  `max_delay_frames` 與 `state_layout_version`，必須與 export artifact 及 C
  descriptor 三方一致（D 決定 state tensor shape 與 host state RAM）；現由
  `AIAEC/tests/test_export_streaming_calibration.py` 交叉比對。同一份 report
  另以 `calibration_far_input_mode` / `deployment_far_input_mode` 兩個欄位
  分開記錄「這組 range 是在哪條 seam 上錄的」與「部署會餵哪條 seam」——
  Align-ULCNet 刻意以訓練域 `linear_error + raw_far` 做 calibration（該階段
  不跑 matched filter），不代表兩條 seam 相同。

`AIAEC/Align_ULCNet/export_onnx.py` 目前同時產出：

1. `.onnx`。
2. JSON metadata（供工具/測試）。

板端若不讀 JSON，後續應再由同一份 metadata 產 generated C descriptor
header，至少含 grid、D、固定 aligned-far ABI、layout version 與 tensor sizes；
不能在兩個 `main.c` 各自維護另一份數字。

兩個 application `main.c` 最終不得手寫 `D=8`；必須載入 exporter 的
descriptor（目前 example main 仍以 `descriptor_default(8)` 示範 board TODO）。
descriptor 與 pipeline config 不一致時 init 失敗，不可 clamp 或 fallback。

### 4.4 aligned_far 的時間軸與狀態契約

- `AecLinearContext.aligned_far_hop` 必須是「本 hop 的 PBFDKF 實際消費的
  reference」，不可重新估計或用 raw far 近似。
- 每 hop 一次性讀取 `{aligned_far_hop, delay_samples, confidence,
  delay_state, generation}`，避免跨呼叫借用失效的 pointer。
- `UNLOCKED`：seam 暫時承載 raw far；model 照常 step，成功且 finite 的輸出
  照常套用。D 決定它能否解掉剩餘 frame offset。
- `CHANGED`：在本 hop inference 前清除 K/V、logit、GRU 等 model state；
  之後才允許新 aligned far 建立狀態。
- `LOCKED`：seam 承載 aligned far，正常套用 model output。
- FIXED 沒有 estimator CHANGED event；wrapper 必須在首次
  `UNLOCKED→LOCKED` 時做等價 reset。
- mono/4ch 的 WOLA 與 far branch 必須同 frame timestamp。4ch beamforming
  引入的 one-hop compensation 要由 impulse test 明確鎖定，不能靠註解。

### 4.5 D profile 與外部 state

- 支援 `D=2..64`；`D=1` 只允許 Python 評估，不列入 portable ONNX ABI。
- 每個 D 是獨立 ONNX/descriptor/profile；不可在同一 instance 中熱改。
- CPU/DSP 擁有 K/V ring、logit history、兩組 GRU hidden；accelerator 每次
  只處理 T=1 並回傳 delta state。
- state pool size 必須隨 D 下降；query/init/carve 必須 lockstep。
- prepare 後 accelerator fail、NaN/Inf 或 partial write 時不得 commit state；
  output 必須 identity/fail-open，下一幀可以恢復。
- delay generation 改變時必須 reset state，不能保留前一個 alignment 的
  K/V history。

### 4.6 static memory 與 backend

- release example 預設展示 caller-owned static pool；heap `create()` 只能是
  可選 convenience API，不能是唯一示範路徑。
- AEC pool 隨 n/filter length 變化；model state pool隨 D 變化。文件列出的
  bytes 必須由 `--print-mem-size` 或測試即時計算，不能手抄常數。
- KISS/NE10 與 SIMD=0/1 共用相同 config/descriptor；所有組合以
  `WERROR=1` clean rebuild。

## 5. 無 checkpoint 的驗證矩陣

### 5.1 AEC matched search 與 pool

對 `n=1..5`：

- 用 broadband seeded synthetic far，建立已知整數與非 hop 整數 delay。
- 測範圍內 acquisition、邊界前後；另以「超範圍主路徑 + 範圍內早期反射」
  釘住可能發生的 confident mis-lock，證明產品選 n 不能只依 seam confidence。
- 測 cold boot、path change、mid-stream reset、saturation/NaN guard。
- 比較 C/Python 的 applied delay、state transition、generation 與 aligned
  far；容許的量化誤差需明訂。
- 驗證 pool 隨 n 單調縮小，且每少一個 bank 的 byte delta 符合實際
  `get_mem_size()`；mutation 掉 n 傳遞或固定成 5 時測試必須失敗。

可靠搜尋上限採 AEC 目前合約值，而非幾何 span：

| n | 可靠 bulk-delay 上限 |
|---:|---:|
| 1 | 約 125 ms |
| 2 | 約 221 ms |
| 3 | 約 317 ms |
| 4 | 約 413 ms |
| 5 | 約 509 ms |

### 5.2 aligned_far seam

- impulse：`aligned_far` 的 impulse timestamp 必須等於 PBFDKF 當下讀取的
  reference timestamp。
- broadband correlation：排除 silence/periodic ambiguity，估測 residual
  lag；不得重現把 corpus 尾長差誤判成 3744 ms delay 的問題。
- delay change：確認 generation 只在 applied alignment 真正改變時增加，
  model reset 發生在新 frame inference 前。
- mono 與 4ch 分別做 far timestamp 測試；故意移除 4ch one-hop buffer 的
  mutation 必須精確失敗一個 hop。
- EXTERNAL_ALIGNED/FIXED/MATCHED 三種 mode 分別驗證 context 語意。

### 5.3 ONNX/state ABI（可使用隨機權重）

對 `D=2,4,8,16,32,64`：

- export、ONNX checker、shape inference 與一幀 inference。
- offline PyTorch、`forward_stream()` 與 explicit-state exported graph 在
  warm-up 後數值 parity。
- generated JSON/C header/C descriptor 三方逐欄 parity。
- C state-pool query/init/reset/prepare/commit 測試；pool bytes 隨 D 單調。
- mutation D、far mode、layout version、任一 tensor dimension/order 時 init
  必須 fail。
- repeated reset、accelerator failure、NaN/Inf、partial write 後 state 不得
  前進，下一個成功 frame 可恢復。

### 5.4 application flow

用 identity/deterministic fake model callback，不需 checkpoint：

- mono/4ch、`n=1/2/3/5`、descriptor `D=4/8/16/64`；production 固定
  aligned-far seam，raw/aligned A/B 僅由 sweep 覆蓋。
- UNLOCKED 仍必須 invoke/apply model；MATCHED CHANGED 與 FIXED 首次可用
  都必須先 reset model state。
- model callback 的 error/far frame timestamp 必須匹配。
- KISS/NE10 × SIMD 0/1 × WERROR。
- static pool、zero unexpected heap、ASan/UBSan（可用平台）、leak check。
- 所有 public init 的 invalid-argument、undersized/misaligned pool 與重复 init
  負向測試。

## 6. 產品 delay gate（不是泛用音質跑分）

`n`/`D` 的選擇依據是應用場景的 delay，而不是 VCTK、AECMOS 或任意 corpus
的平均分數。縮小 n/D 不要求重訓，也不要求重跑 800-case benchmark。

產品端應量測：

1. raw far→mic bulk delay：cold/warm boot、route/codec/buffer switch、
   suspend/resume、CPU underrun recovery、clock drift 與 path change；用此分佈
   加 acquisition margin 選 n。
2. matched/reference alignment 後，送入 NN 的 aligned-far/error residual
   frame lag；用此分佈加至少兩個 hop margin 選 D。
3. acquisition time、mislock、unlock duration、same-delay relock、delay-change
   reset 次數與 oldest-boundary hit，確認 profile 仍有 liveness。
4. static pool bytes、實機平均/峰值 CPU 與 target NPU latency，確認縮小 profile
   確實換到所需資源收益。

品質驗證的角色只剩：

- **單純改 n/D**：少量真實錄音做 finite、fail-open、無明顯爆音/斷裂與
  waveform sanity；不是用平均 MOS 決定 coverage。
- **`raw_far -> aligned_far`**：既有 sweep 已完成且使用者接受結果，本輪
  不再重跑；只保留 descriptor/pipeline mode 一致性測試。
- **4ch BF input**：另做少量真實 SRP/GSC flow sanity，因為它與 mono
  training input 分佈不同；同樣不以泛用 NR corpus 代替 delay 驗證。

建議先輸出候選而不覆寫正式 artifact：

```text
align_ulcnet_raw_d64.onnx        # 舊訓練契約基準
align_ulcnet_aligned_d16.onnx    # candidate
align_ulcnet_aligned_d8.onnx     # candidate
align_ulcnet_aligned_d4.onnx     # candidate
```

aligned far 已接受沿用既有權重；正式 artifact 仍須準確標記實際
far-input mode，避免 board 把 raw/aligned descriptor 接反。

## 7. 產品 profile 選擇規則

- `n` 由各 SKU/route 的 bulk-delay 分佈決定，不由 D 推導。至少用觀測
  p99.9 加 acquisition margin，並涵蓋 boot、route change 與 jitter。
- `D` 由 **對齊後** NN far/error residual-frame lag 分佈決定；至少保留
  兩格變動 margin，並觀察 attention oldest-boundary hit。
- PBFDKF filter length 由 residual delay + acoustic echo tail 決定，不因 n
  或 D 變小就自動縮短。
- 未達 matched coverage 的產品不要硬用小 n：改用較大 n、FIXED（已知固定
  delay）或 EXTERNAL_ALIGNED（上游保證對齊）。

建議的實驗起點，而非 release 結論：

| profile | AEC delay | NN far | D | 用途 |
|---|---|---|---:|---|
| baseline | MATCHED n=5 | raw | 64 | 舊 checkpoint 對照 |
| transition | MATCHED n=5 | aligned | 64 | 單獨驗證 far semantic |
| candidate A | MATCHED n=5 | aligned | 8 | 先只縮 TA RAM/算力 |
| candidate B | MATCHED n=3 | aligned | 8 | bulk delay 確認小於約 317 ms 後再測 |
| short-route | MATCHED n=1/2 | aligned | 4/8 | 僅限已量測的短延遲 SKU |

完成 §4–§5 表示實作、ABI、時間軸與 memory scaling 正確；正式 n/D 由
§6 的產品 delay/liveness 分佈決定。4ch input distribution 的少量 flow
sanity 仍獨立保留。

## 8. 現行程式盤點後的精確工作項目

本節以目前 repository 實際 API/測試為基準，供實作者直接依序執行。

### 8.1 不需重寫的現有元件

- `ulcnet_model_io_get_mem_requirements()` 已依 descriptor `delay_depth`
  精確配置 K/V、logit 與 delta state；D=4/8 不會保留 D=64 最大陣列。
- `ulcnet_model_io_reset/prepare/commit()` 已具備 external-state reset、NaN
  prefill、one-prepare/one-commit 與失敗不推進 persistent state 的交易語意。
- mono 已有 ALIGNED unlock fail-open、RAW unlocked apply、delay-change reset、
  NaN/partial-write 與 raw far timestamp 測試。
- 4ch 已是單一 shared delay estimator；四個 lane AEC 都以
  `EXTERNAL_ALIGNED` 建構，不會各自再配置 matched bank。
- AEC standalone 已驗證 n=1..5 的 estimator pool/search/parity；不要重做
  estimator core。

### 8.2 保持 dataset/checkpoint contract 不變

- `LinearAecContract` 維持 v3；dataset generator 與 training materializer
  仍固定使用 `MATCHED n=5`。
- 不把 n/D 納入 checkpoint compatibility，也不新增 v3→v4 migration。
- `aec_behavior_hash` 會因為 `lib/aec` 動到 signal scope（`aec.py` +
  `modules/`）而改變，即使該改動在預設路徑上完全不動訊號。既有 shard 與
  已訓練 checkpoint 記的是舊 hash，inference 入口會擋下來。處理方式是
  `linear_aec.py` 的 `ACCEPTED_BEHAVIOR_HASH_MIGRATIONS`：明列
  `recorded → current` 一對 hash，且必須附上「凍結 frontend 前後
  byte-identical」的實測證據，以及「同一套 harness 把該機制打開後 bytes
  真的會變」的對照組。命中時只發 `RuntimeWarning` 後放行。
  - 只在 `aec_behavior_hash` 是唯一相異欄位時成立；其他欄位有差異照樣拒絕。
  - 單向、單跳、逐對列舉；未列出的 hash 一律拒絕。單跳＝表中的 value
    不會再被當成 key 讀一次，兩段連續 migration 必須補上「合成後的那一對」
    並重新逐端驗證。
  - **既有 shard 不需重生、不需 re-stamp，已訓練的 checkpoint 不需重訓**，
    `behavior_hash_schema` 也不 bump（canonicalizer 沒變）。命中時只發
    `RuntimeWarning` 後放行。
  - **目前是兩筆**（`ACCEPTED_BEHAVIOR_HASH_MIGRATIONS`，
    `AIAEC/dataset_gen/linear_aec.py:495-528`），兩筆都指向同一個 current
    hash：第一筆對應 delay-mode 產品化（frozen 路徑仍解析成 MATCHED n=5），
    第二筆對應 delay backward-quarantine（預設 OFF，AIAEC 從不設定）。
    每一筆旁邊都記著 byte-identical 的實測證據與「同一套 harness 打開該機制
    後 bytes 真的會變」的對照組。
- 撤回目前 WIP 中 `linear_aec.py` 的 contract-v4、
  `accepted_fingerprints()` 及 pack/training contract 連鎖修改。
- deployment n 是 runtime AEC init override；D 是 ONNX export/descriptor
  override。兩者都可和訓練 provenance 不同，不可因此拒絕載入權重。

### 8.3 讓既有 sweep 額外測 runtime n

只新增：

```text
--delay-num-filters 1..5
```

- 沒給參數時維持現行 `n=5`，既有 D/raw/aligned 結果不變。
- override 只作用於本次診斷建立的 AEC instance，不寫回 dataset/checkpoint
  contract。
- `--input-is-linear-error` 已繞過 PBFDKF，若同時傳 n 必須 fail-fast。
- summary/trace 記錄 requested/resolved n、D、far mode 與 grid。
- `run_linear_aec_with_taps()` 必須從 runtime AEC instance 讀回實際 n 核對。
- mutation：忽略 n、固定成 5、或只寫 requested 未驗 resolved 時失敗。

### 8.4 修正 4ch unlock→same-delay relock reset

此問題已修正。舊版 4ch 的 `changed` 僅由
`eligible && estimated != accepted_delay` 產生，會漏掉首次鎖在 0 及失鎖後
同值重鎖；現行實作改以「新的可用 alignment generation」判斷。

修正語意：

```text
was_usable = previous solid && previous delay_samples >= 0
now_usable = eligible
changed = now_usable && (!was_usable || estimated != accepted_delay)
```

- `changed` 表示「本 hop 開始一個新的可用 alignment generation」，包含
  首次鎖定、失鎖後重鎖同值、以及鎖定中改 delay。
- 如新增 `alignment_generation`，不得重用
  `FourAecNrResFrameToken.generation`；後者是 pre/post ownership token，與
  delay generation 無關。
- 新增 deterministic 測試：LOCKED(A) → UNLOCKED 多 hop → LOCKED(A)，確認
  model 在 unlock 期間只 step 不 apply，重鎖 hop inference 前 reset 恰一次。
- mutation 回舊的 `estimated != accepted_delay` 判斷必須失敗。

### 8.5 補 aligned-far timestamp coverage

現有 mono/4ch wrapper 的 timestamp 測試已覆蓋 production seam；4ch core
另驗證 `pre.aligned_ref` 並穿過 ULCNet wrapper 的 one-hop far compensation。

需新增：

- mono production seam：model 收到的 error/far STFT frame timestamp
  一致，far samples 等於該 hop PBFDKF 實際消費的 aligned far。
- 4ch production seam：穿過 core + BF WOLA + wrapper far-delay buffer，
  model 兩個 branch timestamp 一致。
- delay change 前後各放不同 impulse，驗證 far-delay buffer 與 beam OLA
  同時清除，不會混用兩個 alignment generation。
- 拔掉 4ch one-hop buffer 的 mutation 應固定差 256 samples；將 ALIGNED
  source 誤接 raw far 的 mutation也必須失敗。

### 8.6 4ch RAM 報告必須量實際 wrapper，不套用 mono 常數

AEC standalone 的每少一個 matched filter 省 5,728 B，是「每個完整
MATCHED AEC instance」的合約。4ch 四個 lane 是 `EXTERNAL_ALIGNED`，n 只
作用在 wrapper 的單一 shared `DelayAec3`；delay ring 又由
`max_delay_ms`/hop 決定，不能宣稱省 `4 * 5,728 B`，也不能直接套用 mono
差值。

- 對 4ch 呼叫實際 `four_aec_nr_res_get_mem_breakdown()`/總 size query，列出
  n=1..5 的 shared-estimator bytes、delay-ring bytes、lane AEC bytes 與總數。
- 斷言 lane AEC bytes 對 n 不變、shared estimator bytes 隨 n 下降、總 pool
  與 init carve 完全一致。
- mono 則繼續用 `aec_get_mem_size()` 的 5,728 B/filter 合約。

### 8.7 建立單一 AIAEC test 入口

`AIAEC/Makefile` 目前只有 `lib/clean`。新增不改演算法的測試入口：

```text
make test-c       # model_io/process/adapter C contract
make test-python  # AIAEC pytest（有 torch 的環境）
make test         # 可用環境下合併執行
```

- adapter standalone binary 的 build/run 不再只能靠頂層 pipelines Makefile
  的隱含 target；但仍重用相同 source/object，不能複製 driver 實作。
- 無 torch 的 release/board 環境至少可跑 `test-c`，並清楚回報 Python gate
  為未執行，而不是假裝全綠。

### 8.8 執行順序與停止條件

1. 撤回 linear-AEC contract-v4 WIP，確認既有 v3 dataset/checkpoint 路徑不變。
2. sweep runtime n override + report/mutation。
3. 4ch relock reset blocker。
4. 4ch/mono n memory scaling report與 static-pool init parity。
5. AIAEC Makefile test entry與 n 使用文件同步。

任一步若改變 legacy `n=5 + raw_far + D=64` 的 linear-error WAV，立即停止；
不得靠重生 dataset 或更新 golden 掩蓋。上述全部通過後可放行結構與部署
ABI；產品 n/D 由 §6 的應用 delay 量測裁決，不以泛用音質分數代替。

本節的執行順序現已完成；legacy raw-far 只保留為 checkpoint provenance 與
`sweep_delay_depth.py` 的診斷基準，不再是 production pipeline 選項。
