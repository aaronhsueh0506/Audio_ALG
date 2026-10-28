[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md) ｜ [下一份](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md) <!-- split-nav -->
## 6. AI 模型 pre/post 類別

四個模型各有一個 pre/post 類別，把「呼叫端音訊 ↔ 加速器張量」之間的所有東西
（framing、window、FFT/IFFT/OLA、特徵前端、遞迴 ring 與狀態交易）收成一個物件：

| 模型 | Public header | 訊號輸入 |
|---|---|---|
| Align-ULCNet | `AIAEC/Align_ULCNet/ulcnet_prepost.h` | linear-AEC error + aligned far |
| DeepVQE-S | `AIAEC/DeepVQE_S/deepvqe_prepost.h` | **raw microphone** + far reference |
| DeepFilterNet2 | `AINR/DeepFilterNet2/dfn2_prepost.h` | 單路 noisy |
| DeepFilterNet3 | `AINR/DeepFilterNet3/dfn3_prepost.h` | 單路 noisy；raw 低頻 DF / 高頻 ERB |

每個類別都是**第三個 TU**：它組合的兩個檔案（`dfn3_process.c` + `dfn3_model_io.c`；`ulcnet_process.c` + `ulcnet_model_io.c`；
`aiaec_process.c` + `deepvqe_process.c`；`dfn2_process.c` + `dfn2_model_io.c`）必須維持各自
可獨立連結給自己的 parity 測試用，所以類別不能放進其中任何一個。

### 6.1 共通生命週期

四個類別的生命週期形狀相同，前綴為 `ulcnet_` / `deepvqe_` / `dfn2_` / `dfn3_`。
DFN3 的兩個 heads、layout 9、state 綁定、lookahead 與 API 範例見
[DFN3 模型頁](html/ainr_dfn3.html)；下面原有的 DFN2 專節仍描述其 cascade/alpha。

DFN2／DFN3 的 lookahead 是 M（`mask_lookahead`，0..2）與 D（`df_lookahead`，0..4）兩個獨立設定，共 15 個 pair，出貨為 1/1。
pair 是模型合約，C 端在建置時以 `-DDFN2_MASK_LOOKAHEAD=<M> -DDFN2_DF_LOOKAHEAD=<D>`（DFN3 為 `DFN3_*`）加在每個編譯單元上，
須與 checkpoint 一致；不同 pair 的目標檔連結失敗。模型延遲 DFN2（cascade）為 M+D hops，DFN3（平行）為 max(M, D) hops；
`pre_process` 在前 M 拍回 0。DFN2 另可每次 invocation 跑 N 個 frame（`export_onnx.py --frames-per-call N` 加 `-DDFN2_FRAMES_PER_CALL=N`，預設 1；`pre_process` 每 N 拍回 1 一次，延遲多 N-1 hop，pipelines 固定 N=1）。規則見 [DFN2 細節頁](html/ainr_dfn2_detail.html)。

```text
_config_defaults(&cfg, io_mode[, delay_depth])
  -> _get_mem_size(&cfg, &req)
  -> _init(pool, req.bytes, &cfg)  或  _init_ex(pool, bytes, &cfg, &expected)
  -> 每-hop 序列（6.2）
  -> _reset(p)（stream boundary）
  -> _destroy(p)
```

- `io_mode` **init-time 固定**，因為它決定 pool 大小：`*_IO_TIME` 是 hop 進 hop 出
  （類別做分析端 framing，transform 呼叫**呼叫端的** `FftHandle`；合成是**應用端持有的 audiolib `AcIstft`**，
  經 `cfg.istft` 借給類別，`frame_commit()` 把每個增強頻譜 push 進去，`_reset()` 一併清它），
  `*_IO_FREQ` 是頻譜進頻譜出、完全不做 transform，用來接在 AEC/BF seam 後面串接：
  `_post_process_freq()` 交出頻譜，由應用端自己呼叫 `ac_istft_push_planar()`。
  inverse FFT 與 overlap-add 一律是 `AcIstft`，模型目錄裡沒有（`make audit-no-transform` 把關）。
- `_get_mem_size()` 是 reject-first：NULL 引數、未知 `io_mode`、`delay_depth` 超出範圍、
  TIME 模式沒有可用的 `fft`/`window`/`istft`（`istft` 必須是該模型 `*_framing_config()` 描述的物件：
  fft 大小、overlap-add 模式、合成尺度相符，窗表內容與分析窗相同；DFN2 另加：缺 ERB 矩陣、非有限的 `atten_lim_db`），
  一律回 `-1` 且 `*req` 一個 byte 都不寫。
- `_init_ex()` 是 **stale-pool gate**：固定 32-byte 的 `*PrepostMemReq`
  （`descriptor_version`、`layout_version`、`io_mode`、`build_flags_hash`（FNV-1a-32）、
  `alignment`、`reserved`、`bytes`）與本 build 不符就拒絕，而不是重新解讀一塊為別的組態
  配出來的 pool。`expected` 傳 NULL 時等同 `_init()`。
- `_create()` 是 get_mem_size + 對齊配置 + init，給 host 工具與測試；板端走 `_init()`。
  `_destroy()` 只釋放 `_create()` 配置的東西：對 `_init()` 實例它是**可重複呼叫的 no-op**
  （pool 永遠是 caller 的），對 `_create()` 實例呼叫第二次就是 double free。
- 借用而不擁有：`FftHandle`、window 表、DFN2 的 ERB 矩陣都是 caller-owned，類別從不建立或銷毀。
  分析用的 `AcStft` 是 `*_IO_TIME` 的 pool 內物件，借用這些表；DFN2／DFN3 的 `cfg.window` 為 NULL 時，
  由分析物件在 pool 內自帶 sqrt-Hann 表（應用端的 `AcIstft` 用同一個 `ac_window_build()` 產生的表，內容相同即可）。
  合成用的 `AcIstft` 不在 pool 內：應用端用 `dfn2_framing_config()`／`dfn3_framing_config()`／`ulcnet_framing_config()`／
  `deepvqe_framing_config()` 取得設定、`ac_istft_get_mem_size()` 配記憶體、`ac_istft_init()` 建好後填進 `cfg.istft`。
- `_reset()` 清空所有遞迴／ring／狀態、丟掉未結束的 frame、framing 與 clock 回 init 值；
  config、borrowed 指標與 pool 都不動。

### 6.2 每-hop 序列與回傳值合約

```c
int n = ulcnet_prepost_pre_process(p, err_hop, far_hop);  /* 需要幾次推論 */
if (n == 1) {
    UlcnetModelIoInputs  in;
    UlcnetModelIoOutputs out;
    ulcnet_prepost_frame_inputs(p, &in, &out);   /* 指向本實例 pool 的 view */
    /* 每個 out.*_out 綁到 in 的同名位址；先讀完所有 state 輸入、才寫輸出。
     * （DFN2、DFN3、GTCRN 同樣就地綁定。） */
    if (accelerator_run(&in, &out) != 0 ||
        ulcnet_prepost_frame_commit(p) != 0)
        ulcnet_prepost_frame_skip(p);            /* 保住 framing 節奏 */
}
int written = 0;
ulcnet_prepost_post_process(p, out_hop, &written);
```

- **`_pre_process()` 回的是「這一拍需要的加速器呼叫次數」，不是布林。**
  Align-ULCNet 與 DeepVQE-S **從第 0 拍起恆為 1**（一 hop 進、一次推論、一 hop 出，兩個 io_mode 皆然）；
  DeepFilterNet2／3 **前 M 拍回 0**（M = `mask_lookahead`；出貨 M = 1 即第 0 拍）、之後恆 1，因為它們的圖吃
  `[t-(2-M), …, t+M]`，前 M 拍右鄰居還沒到，那幾拍**不可以**呼叫加速器。
- `_frame_inputs()` 只把**非 state 的輸出**（heads：`output`／`taps`／`erb_mask`／`coefs`／`alpha`，
  Align-ULCNet 的 `output`）**預填 NaN**，所以加速器只寫一半會在 commit
  被抓到，而不是把上一框的值當成本框結果。**state 不預填**（它同時是下一框的輸入）。
  回 `0`，沒有開啟中的 frame 回 `-1`。
- `_frame_commit()` 是**純驗證**：heads 與 state 都是有限值就成功（CCM ring／compose／synthesis 照常前進，
  commit 本身不複製 state）。就地路徑遇到任何一處非有限值時 state 已被加速器覆寫、**無法回滾**，
  所以**所有遞迴 state 歸零**（冷啟動）、state generation 加一，framing／CCM ring 不前進、frame 維持開啟、回 `-1`（Align-ULCNet：key／value／logit 三個
  window 與 h_gru 共四個 state 全歸零；DeepFilterNet2：`h_gru` 與 pathway window 兩個遞迴張量；DeepVQE-S：16 個 state）。
  接著只有兩條路：`_frame_skip()`，或重新 `_frame_inputs()` 再從冷 state 跑一次加速器。
- **commit 前面一定要有一次 `_frame_inputs()`**（`prepared` 閂鎖，四個類別皆然）：
  沒跑過的加速器不能把沒被碰過的 buffer 當成結果送出。
- **遞迴 state 的綁定。** 每個 state 張量在 pool 裡只有一個 buffer，位址在實例壽命內固定（沒有交替的 bank、沒有指標交換）。
  - **Align-ULCNet、DeepVQE-S、DFN2、DFN3、GTCRN：就地綁定，沒有複製路徑。** runtime 把每個 `*_out`（C 端欄位名 `*_next`）綁到
    `_frame_inputs()` 回傳的同名輸入位址，**先讀完**所有 state 輸入、**才寫**輸出；graph 回完整的下一個 window
    （位移在 graph 內），CPU 不搬、不複製。commit／validate 被拒時無法回滾：遞迴 state 從零重來、state generation 加一；
    回報失敗的 runtime 視為沒有寫入。
  - **RNNoise-ERB（複製路徑）**：runtime 把 `h1_out`／`h2_out`／`h3_out` 寫進自己擁有的 tensor，host 呼叫
    `rnnoise_model_state_inherit()` 複製進 state。inherit 是**先檢查再複製**：任何一處非有限值就一個都不複製、回 `-1`、
    state 與原來逐位元相同，這一幀視為失敗。指標等於目的位址的 tensor 視為已就地寫入，不複製也不由 inherit 檢查，
    所以同一個呼叫兩種綁定都適用。
  非有限值以外的「寫一半／混寫」都偵測不到。
- **state generation（Align-ULCNet、DeepVQE-S、DFN2、DFN3、GTCRN）**：`ulcnet_model_io_state_generation()`、
  `ulcnet_prepost_state_generation()`、`ulcnet_accelerator_adapter_state_generation()`、
  `deepvqe_prepost_state_generation()`、`dfn2_prepost_state_generation()`、`dfn2_model_io_state_generation()`、
  `dfn3_prepost_state_generation()`、`dfn3_model_io_state_generation()`、`gtcrn_model_state_generation()` 回傳 C 端把 state 清零的次數：`_reset()` 與被拒絕的 `_frame_commit()` 各加一，
  新建實例的起始值：Align-ULCNet、DeepVQE-S 為 0，DFN2、DFN3、GTCRN 為 1；成功的 commit、`_frame_skip()` 與再次呼叫被拒的 commit 都不動它；`NULL` 回 0，數值回繞於 2^32。
  定點 accelerator 的 state 緩衝在 C 清零時不會跟著變，板端 sample 比較這個值，變了就用量化後的 0 重填 state 輸入。
- **Align-ULCNet 的四個 state**（`key_history`／`value_history`／`logit_history`／`h_gru`
  ↔ 對應的 `*_out`）：輸入是完整 window、graph 回每個 state 的**完整下一個 window**，shape 與輸入完全相同：
  K/V `[1,32,D,TA]`（newest-first）取輸入 slot `0..D-2` 再把新一幀接在前面，logit `[1,32,5,D]`（oldest-first）
  取輸入 frame `1..4` 再把新的一列接在後面（輸入端 Slice 加 Concat，沒有 state 輸出由 Slice 產生），`h_gru`
  `[1,4,1,128]` 是兩顆 subband GRU 的 hidden 疊在一起；沒有 CPU 端的 ring helper。`output` 是學到的複數 mask
  （planar `[1,2,1,BINS]`），commit 驗證後以壓縮域 error 乘上它並逆冪。
- **Align-ULCNet 的 commit 有限性檢查只看「這一幀 graph 新寫的那一格」**：`output`、每個 channel 的 K/V slot 0
  與 logit 最後一幀，以及 `h_gru` 全部；較舊的格子與被丟掉的那一格不重查，留在那裡的非有限值會在下一幀進到 `output` 被擋下。
  DeepVQE-S 同理：每個 tap、`h_gru` 全部，以及每個 window 的最新一幀（attention ring 的 slot 0、conv／score window 的最後一幀）。
  `_frame_skip()` 不動任何 state（維持它當下的樣子；就地路徑視同回報失敗的 runtime 沒寫）。

拒絕情境整理：

| 呼叫 | 情境 | 回傳 |
|---|---|---|
| `_pre_process` / `_pre_process_freq` | 上一個 frame 還開著（既未 commit 也未 skip） | `-1`（一拍被拒絕，不會默默疊上去） |
| `_pre_process` / `_pre_process_freq` | 與實例的 `io_mode` 不符 | `-1` |
| `_frame_inputs` / `_frame_commit` / `_frame_skip` | 沒有開啟中的 frame | `-1` |
| `_frame_commit` | 沒有 `_frame_inputs()` 在前面 | `-1`，狀態不動 |
| `rnnoise_model_state_inherit` | 任何將被複製的 tensor 非有限、NULL、缺 tensor 或元素數不符 | `-1`，不複製任何東西，state 逐位元不變 |
| `dfn2_prepost_set_erb_matrices` / `_set_atten_lim` | frame 開著 | `-1` |

### 6.3 skip 政策：三個模型不一樣

`_frame_skip()` 是加速器跑失敗（或對齊邊界 reprime）時唯一正確的出口——它讓 framing
節奏繼續走，ring／CCM 不步進，遞迴 state **維持原樣**（複製路徑是 inherit 沒有動它；就地路徑則是加速器寫完的樣子，回報失敗的 runtime 視為沒有寫入，
所以 state 停在原處；寫了一半的 state 只有在非有限值時才偵測得到）。但「這一框輸出什麼」每個模型不同：

| 模型 | skip 輸出 | 理由 |
|---|---|---|
| Align-ULCNet | error 頻譜原樣穿過（identity） | 它的 stream 0 已經是 linear AEC 之後的殘差 |
| DeepFilterNet2 | unit ERB mask + alpha 0（identity） | `erb_inv` 是 partition of unity（每個 bin 的 band 權重和為 1），unit band mask 展開就是 unit bin gain——**精確**而非近似 |
| DeepVQE-S | **靜音（fail closed）** | 它的 stream 0 是**原始麥克風**，pass-through 會送出**完全未消除的回音**；`deepvqe_prepost_skip_policy_name()` 回 `"mute_fail_closed"` |

DeepVQE-S 的代價是 near-end 一框的缺口，上限就是一框；另一個選項的代價則是無上限的回音進遠端。
要 graceful degradation 的呼叫端得自己跑一個 canceller 並在 pipeline 層 cross-fade——
那不是這個類別的決定，它看不到有沒有那個 canceller。

### 6.4 Align-ULCNet：`ulcnet_prepost.h`

```c
int  ulcnet_prepost_config_defaults(UlcnetPrepostConfig *cfg, int io_mode, int delay_depth);
int  ulcnet_prepost_get_mem_size(const UlcnetPrepostConfig *cfg, UlcnetPrepostMemReq *req);
UlcnetPrepost *ulcnet_prepost_init(void *pool, size_t bytes, const UlcnetPrepostConfig *cfg);
UlcnetPrepost *ulcnet_prepost_init_ex(void *pool, size_t bytes, const UlcnetPrepostConfig *cfg,
                                      const UlcnetPrepostMemReq *expected);
UlcnetPrepost *ulcnet_prepost_create(const UlcnetPrepostConfig *cfg);
void ulcnet_prepost_destroy(UlcnetPrepost *p);
void ulcnet_prepost_reset(UlcnetPrepost *p);
uint32_t ulcnet_prepost_state_generation(const UlcnetPrepost *p);
int  ulcnet_prepost_hop_size(const UlcnetPrepost *p);
int  ulcnet_prepost_num_bins(const UlcnetPrepost *p);
int  ulcnet_prepost_io_mode(const UlcnetPrepost *p);
const UlcnetModelIoDescriptor *ulcnet_prepost_descriptor(const UlcnetPrepost *p);

int  ulcnet_prepost_pre_process(UlcnetPrepost *p, const float err_hop[ULCNET_HOP],
                                const float far_hop[ULCNET_HOP]);
int  ulcnet_prepost_pre_process_freq(UlcnetPrepost *p,
                                     const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                                     const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS]);
int  ulcnet_prepost_frame_inputs(UlcnetPrepost *p, UlcnetModelIoInputs *inputs,
                                 UlcnetModelIoOutputs *outputs);
int  ulcnet_prepost_frame_commit(UlcnetPrepost *p);
int  ulcnet_prepost_frame_skip(UlcnetPrepost *p);
int  ulcnet_prepost_post_process(UlcnetPrepost *p, float out_hop[ULCNET_HOP], int *written);
int  ulcnet_prepost_post_process_freq(UlcnetPrepost *p, float re[ULCNET_BINS], float im[ULCNET_BINS]);
```

- `cfg` 欄位：`io_mode`、`delay_depth`（D，範圍 `ULCNET_MODEL_IO_MIN_D`=2 到
  `ULCNET_MODEL_IO_MAX_D`=64）、`fft`、`window`、`istft`（後三者 `ULCNET_IO_TIME` 必填、borrowed，`istft` 為應用端用
  `ulcnet_framing_config(&sc, fft, window, 0)` 建的 `AcIstft`；
  window 為 `ac_window_build()` 產出的 `ULCNET_N_FFT` 項；`ULCNET_IO_FREQ` 不用）。
- **error 與 far 必須是同一個輸入 hop**：類別內部不做任何 skew。上游 WOLA 造成的落差由呼叫端先對齊。
- `ULCNET_IO_FREQ` 收的是模型自己 framing 的未正規化 rfft（center=False，一 hop 一框）——
  AEC 的 `AecResContext.error_spec` 已經正好是這個 framing 慣例（但它是交錯的 `Complex`，
  要拆成 re/im 兩個陣列再傳）。
- 加速器邊界的 `UlcnetModelIoInputs`／`UlcnetModelIoOutputs` 定義在 `ulcnet_model_io.h`。
- 一個 build 只服務一個 grid：預設 16000/512，用 `-DULCNET_MODEL_IO_SR=48000
  -DULCNET_MODEL_IO_N_FFT=1024` 建另一個產品 grid；其餘組合由 `#error` 擋掉。

### 6.5 DeepVQE-S：`deepvqe_prepost.h`

生命週期與每-hop 函式和 6.4 逐一對應（前綴 `deepvqe_prepost_`，訊號引數為
`mic_hop`/`far_hop`、`mic_re/mic_im/far_re/far_im`，長度 `AIAEC_HOP`／`AIAEC_N_BINS`），
`frame_inputs()` 用的是本檔自己的 `DeepVqePrepostInputs`／`DeepVqePrepostOutputs`。額外 API：

```c
const char *deepvqe_prepost_state_name(int state_id);       /* graph input 名 */
const char *deepvqe_prepost_state_name_out(int state_id);   /* graph output 名 */
int    deepvqe_prepost_state_shape(int state_id, int delay_depth,
                                   int dims[DEEPVQE_STATE_MAX_RANK]);
size_t deepvqe_prepost_state_elements(int state_id, int delay_depth);
const char *deepvqe_prepost_skip_policy_name(void);
int  deepvqe_prepost_descriptor_default(int delay_depth, DeepVqePrepostDescriptor *descriptor);
int  deepvqe_prepost_descriptor_validate(const DeepVqePrepostDescriptor *descriptor);
const DeepVqePrepostDescriptor *deepvqe_prepost_descriptor(const DeepVqePrepost *p);
```

- 圖邊界：兩個 power-law 壓縮後的 planar 訊號 input（`mic`/`far`，`[1,2,1,AIAEC_N_BINS]`，
  先 real 平面、再 imaginary 平面；壓縮 `x·|x|^(DEEPVQE_COMPRESSION_EXP−1)` 由
  `frame_inputs()` 在 host 端做，圖自第一個 conv 起；`DeepVqePrepostInputs.mic`/`.far` 即這兩個 tensor，
  CCM 用的 raw mic 頻譜另存於類別內）＋
  **16 個顯式 state**，順序就是 `DeepVqeStateId` 列舉（= exporter 的 `input_names[2:]` 逐字），
  輸出是 packed CCM taps
  `[1,1,BINS,DEEPVQE_TIME_ORDER*DEEPVQE_FREQ_TAPS*2]`（16 kHz grid 為
  `[1,1,257,18]`，末軸仍按 `[time][frequency][RI]` 排列）＋ 16 個 next state。
  按 index 綁的 adapter 必須用這個列舉，按名字綁的用 `_state_name()`。
- DeepVQE-S 回的是**每個 state 的完整下一個 window**（不是差量；shape 與輸入相同，全部 rank 4，`h_gru` 為
  `[1,1,1,192]`，conv window 4 幀、score window 5 幀），pool 裡每個 state 只有一份 buffer，runtime 把 `*_out` 就地綁定到輸入位址；
  commit 的有限性檢查只走每個 tap、`h_gru` 與各 window 的最新一幀，遇非有限值時整組 state 歸零、
  `deepvqe_prepost_state_generation()` 加一（`uint32_t deepvqe_prepost_state_generation(const DeepVqePrepost *p)`）。
- `descriptor_validate()` 拿 ONNX/JSON metadata 裡的 13 欄 `c_descriptor` 對本 build 的 ABI 逐欄比對，
  只有 `delay_depth` 是 export-time 部署參數、僅做範圍檢查；`DEEPVQE_PREPOST_LAYOUT_VERSION` = 4。
- D 範圍 `DEEPVQE_PREPOST_MIN_D`=1 到 `DEEPVQE_PREPOST_MAX_D`=256，出貨值
  `DEEPVQE_PREPOST_DEFAULT_D`=32（16 kHz 上的半秒搜尋範圍；D 是 export 期參數，圖 shape 與 C 傳入的 `delay_depth` 必須相同，descriptor 只檢查範圍，抓不到不一致）。
- grid 只有 16 kHz/512/256：`aiaec_process.h` 帶 `#error` 守衛，grid 換掉是編譯失敗而不是默默重新解讀張量。

[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md) ｜ [下一份](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md) <!-- split-nav -->
