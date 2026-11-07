[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md) ｜ [下一份](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md) <!-- split-nav -->
### 6.6 DeepFilterNet2、DeepFilterNet3、GTCRN、RNNoise-ERB

#### DeepFilterNet2：`dfn2_prepost.h`

生命週期與每-hop 函式同樣對應（前綴 `dfn2_prepost_`，單路訊號 `spec_re,spec_im`，長度是 grid 的 `n_bins`；grid 用 `dfn2_grid_make(&grid, &spec)`，見 §6.1；**沒有** `delay_depth`）。額外 API：

```c
const DFN2Grid *dfn2_prepost_grid(const DFN2Prepost *p);          /* 實例跑的 grid；NULL = NULL */
int  dfn2_prepost_model_lookahead_frames(const DFN2Prepost *p);  /* = mask_lookahead + df_lookahead + frames_per_call - 1（出貨 1/1、每次 1 個 frame 為 2） */
int  dfn2_prepost_layout_version(const DFN2Prepost *p);
unsigned int dfn2_prepost_state_generation(const DFN2Prepost *p);  /* init 後 1；每次 _reset、每個暖機幀與被拒的 _frame_finish 加 1 */
int  dfn2_prepost_set_erb_matrices(DFN2Prepost *p, const float *erb_fwd, const float *erb_inv);
int  dfn2_prepost_set_atten_lim(DFN2Prepost *p, float atten_lim_db);
int  dfn2_prepost_pre_process_freq_dual(DFN2Prepost *p, const float *est_re, const float *est_im,
                                        const float *app_re, const float *app_im);
int  dfn2_prepost_post_process_freq(DFN2Prepost *p, float *re, float *im, int *valid);  /* 各 grid.n_bins 個 float */
int  dfn2_prepost_output_frame_index(const DFN2Prepost *p, long long *frame);
```

- `cfg` 欄位：`grid`（`_config_defaults` 填入，按值複製）、`erb_fwd`／`erb_inv`（必填，borrowed）、`atten_lim_db`（0 = 關閉）、
  `gru_hidden`／`df_convp_history`（各配 `*_elements`，float host 才登錄，規則見 §6.2）。
- **Grid。** spec 欄位與出貨值見 §6.1；hop、n_bins、兩個 EMA alpha 與 `stft_scale`／`stft_unscale` 由公式推導，同一個 `&grid` 也交給 `dfn2_model_io_descriptor_default()`／`_validate()`。
  出貨 checkpoint 是 48 kHz／1024／512／32 個 ERB band／96 個 DF bin／階數 5／1/1，另有 16 kHz／256（24 個 ERB band、0/0）一列。
  (sample rate, FFT size) 對應 n_erb、df_bins 與 pair 的表是模型的性質，由應用端持有（板端 sample 與 pipelines 的兩個範例 main 各有自己的表；AEC 沒有 32 kHz 格點，所以接在 AEC 後面的板端 sample 與兩個範例 main 都沒有 32000／512 那一列）。
  只有測試的兩個標頭 `AINR/tests/dfn2_test_grid.h` 與 `pipelines/dfn/tests/dfn_test_grid.h` 以 `-DDFN2_GRID_KHZ=16|32|48` 等在建置時選 grid，函式庫與 pipeline 的單元都不讀它們；pipeline 的 grid 放在 `DfnResStageConfig.grid`。
  沒有編譯期容量：換 grid 就重新查大小並重新 init，且需搭配該 grid 訓練的模型與 ERB 矩陣。規則與各 grid 的大小見 [ainr_dfn2_grids](html/ainr_dfn2_grids.html)。
- **Lookahead pair。** mask lookahead `M`（`mask_lookahead`，0..2）與 deep-filter lookahead `D`（`df_lookahead`，0..`df_order`−1，`df_order` 為 1..16，出貨 5）是 `DFN2GridSpec` 的兩個欄位，由應用端在**執行期**給定
  （階數 5 時共 15 個合法組合，0/0 到 2/4，兩者之間沒有大小限制）；函式庫沒有預設值，也沒有建置期的 pair 旗標或連結期的 pair 符號。匯出的 ONNX graph 不隨這一對改變，隨之改變的是 host 端
  的視窗（`heads[t]` 用 source frame `[t-(2-M) .. t+M]`）與就緒條件（看過 `M+1` 個 frame 才開始呼叫加速器）；checkpoint 與它訓練時的那一對綁定，
  與 grid 不符的 graph 或 pool 由 descriptor（`dfn2_model_run_frame()`）與 pool hash 在執行期擋下。目前只有出貨的 1/1 有 checkpoint，其他組合的音質沒有量測。
  ring 深度 `max(df_order, M+D+1)`：`M+D` ≤ 4 的 pair 都是 5，1/4、2/3 為 6，2/4 為 7；ring 越深 pool 越大（§6.7）。`pipelines/Makefile` 的 `DFN2_MASK_LOOKAHEAD`／`DFN2_DF_LOOKAHEAD` 變數只決定**測試程式**跑哪一對
  （經 `pipelines/dfn/tests/dfn_test_grid.h`），pipeline 的單元與範例 main 不含 pair，不是函式庫的選擇器。細節見 `AINR/README.md` 的 "Lookahead pairs"。
- **每次 invocation 跑 N 個 frame。** `frames_per_call`（N，1..64）是 `DFN2GridSpec` 的欄位，需要 `export_onnx.py --frames-per-call N` 匯出的 graph（預設 1）；N > 1 時 `_pre_process_freq()` 每 N 拍回 1 一次（N = 1 則每拍回 1），延遲多 N−1 hop；
  pipelines 的 `DfnResStage`／`DfnRateBridge` 固定 N = 1（`frames_per_call` 不為 1 即拒絕）。規則見 [DFN2 細節頁](html/ainr_dfn2_detail.html)。
- **`-DDFN2_MASK_ONLY`（每個編譯單元都要加）** 只建 stage 1：沒有 deep filter，`df_order` 存成 0、`df_lookahead` 必須為 0，沒有 `df_convp_history`、`coefs`、`alpha`，graph layout 是另一個版本號；
  混用的單元連結失敗（`DFN2_REQUIRE_VARIANT_LINK()` 引用 `dfn2_variant_full` 或 `dfn2_variant_mask_only`，連結符號只剩這兩個，grid 與 pair 不在其中）。
- 兩個 setter **只能在 hop 之間**呼叫，frame 開著時回 `-1`。要精確理解它買到什麼：
  這是 **per-hop atomicity，不是 per-source-frame consistency**——hop *t* 展開的 mask 屬於 source frame *t*−*M*，
  而圖的特徵視窗橫跨三拍（出貨 1/1：`[t-1, t, t+1]`），
  所以一次被接受的 between-hop 換手仍會有幾拍同時踩到新舊矩陣。要 per-source-frame 一致，
  那就是一個 stream boundary：先 `_reset()`，再換。
- `dfn2_prepost_pre_process_freq_dual()` 把估計分支（特徵、graph 視窗、heads）與套用分支（mask、deep filter、attenuation limit 作用的對象，逐框保存）分開餵；兩者相同就是 `_pre_process_freq()`。
  pipelines 的 RES 串接用它：在 RES 前的頻譜上估計、在 RES 後的頻譜上套用（§6.9）。
- `_output_frame_index()` 給出最近一次輸出屬於哪一個 source frame（自 reset 起 0 起算），
  用來把輸出串流與其他同框時鐘的東西配對；還沒有輸出時回 `-1`。
- 暖機（N = 1、M ≥ 1）：每拍的流程相同，沒有 ready 變數；前 `M` 拍是暖機幀，`_pre_process_freq()` 照回 1、加速器照跑，heads 被丟棄、登錄的 state 歸零、generation 加一（板端每個暖機幀之後重填一次量化零，串流開頭多 M 次 NPU 呼叫），第一個真幀從零 state 開始、音訊逐位元不變。
  輸出前 `M + D` 拍（出貨 1/1 為兩拍）`_post_process_freq()` 的 `*valid` = 0 且輸出頻譜為零；`valid` 只是資訊，合成端在 `AC_OLA_ACCUMULATE`（預設）下可每拍無條件 push（`AC_OLA_ENV_NORM` 不適用）。
  收尾 flush 要再推 `M + D + 1` 個全零 hop（`M + D` 個頻譜加 1 個重疊相加），**加速器那幾拍也要照跑**。限制：`frames_per_call` N > 1 維持舊流程（前面的拍回 0、不可呼叫 `_frame_open()`，暖機與 flush 都多 `N - 1` 拍）；M = 0 沒有暖機幀。
- **ERB 矩陣是應用端讀入的資料。** 函式庫不碰檔案系統、不內建表，只以 `erb_fwd`／`erb_inv` 兩個 `const float *` 借用，不複製、不轉置：正向 bin-major `[n_bins][n_erb]`，反向 band-major `[n_erb][n_bins]`
  （GTCRN：`[erb_high_bins][erb_high_bands]` 與 `[erb_high_bands][erb_high_bins]`）。每個矩陣是 `n_bins × n_erb` 個 float32（GTCRN 是 `erb_high_bins × erb_high_bands` 個；DFN2 的 48 kHz／1024／32 band：513 × 32 × 4 = 65,664 bytes）。
  檔案由應用端載入並持有，指標指到的記憶體必須活得比借用它的所有物件久；DFN2 的反向矩陣必須是 partition of unity（`dfn2_process.h` 標頭註明，`DfnResStage` 會拒絕不符者）。
  樹內的產生者：
  - DFN2、DFN3 的 `export_erb_matrix.py --runtime-bins DIR` 寫 `erb_fwd.bin`（bin-major）與 `erb_inv.bin`（band-major），皆為 raw little-endian float32，另有 `erb_matrices.json`，正好是上列的兩個指標。
    `--output-dir` 寫的 `erb_matrix.bin` 是 8-byte header（int32 `n_bins`、int32 `n_bands`）接 float32 正向 `[n_bins][n_bands]`、float32 反向 `[n_bins][n_bands]`；反向在這個檔裡也是 bin-major，交給函式庫前要先轉成 band-major。
  - GTCRN 的 `--runtime-bins DIR` 寫 `erb_fwd.bin`（`[erb_high_bins][erb_high_bands]`）與 `erb_inv.bin`（`[erb_high_bands][erb_high_bins]`）加 json，正是函式庫要的那一對。它的 `--output-dir` `erb_matrix.bin`（需 `--model`）是另一個檔：
    16-byte header（int32 `n_bins`、`n_erb`、`sub1`、`sub2`）接含低頻 identity 區塊的整張正向 `[n_bins][n_erb]` 與反向 `[n_erb][n_bins]`，不是函式庫要的那一對。
  - RNNoise-ERB 的匯出器沒有 `--runtime-bins`。`--format bin` 寫 `erb_matrix.bin`：int32 `n_bins`、int32 `n_bands`、`W_fwd` `[n_bins][n_bands]`、`W_inv` `[n_bins][n_bands]`（bin-major）、int32 `nfftborder[n_bands]`；`--format npy` 寫 `erb_fwd.npy`、`erb_inv.npy`，`--format c` 寫 `erb_matrix.h`，`--format all` 全寫，矩陣一律是 bin-major `(n_bins, n_bands)`。
    band-major 的反向矩陣（`RNNoiseConfig.erb_inv`）樹內匯出器沒有提供，要由應用端自己從 `W_inv` 轉置。
  - pipelines 的範例 main（`mono_aec_dfn_res`、`4ch_aec_bf_dfn_res`）以 `--erb-fwd FILE`、`--erb-inv FILE`（兩者一起給或都不給）讀 DFN2 的 `erb_fwd.bin`／`erb_inv.bin`；都不給就用 unit partition（fail-open，會印出提示）。
  - 板端 sample 在本 repo 之外：讀 DFN2／DFN3 匯出器的 `erb_matrix.bin`（檔名 `fb_<sample_rate>_<n_erb>.bin`），在 init 自己把反向轉成 band-major，再交出兩個指標。
- **頻譜尺度。** DFN2、DFN3、RNNoise-ERB 的頻譜是 `torch.stft(normalized=True)` 尺度（乘了 1/√n_fft；出貨 grid 為 1/32，換算常數是 grid 的 `stft_scale`／`stft_unscale`，由應用端的 `AcStft`／`AcIstft` 套，§6.8），
  AIAEC 兩個模型與 GTCRN 交出的則是**未正規化**的 rfft。直接串接會剛好差 √n_fft 倍（出貨 DFN2 grid 為 32 倍；過驅動或欠驅動），
  而且兩邊都偵測不到——都是量綱合法的 float 頻譜，ERB 特徵又是 EMA 正規化的。
  兩者也不同 grid（出貨的 48 kHz／1024／512 對 16 kHz／512），中間本來就有 resampler，換算就放在那一層。
  pipelines 的 `DfnResStage`／`DfnRateBridge` 收 audio-scale 頻譜、自己套 `stft_scale`／`stft_unscale`，呼叫端的 `AcStft`／`AcIstft` 維持 scale 1（§6.9）。
  特徵分支內部另有 train.py 的 analysis scale，數值相同但**只**作用在餵給正規化器的拷貝上，不是頻譜進出時要乘的換算。

#### DeepFilterNet3：`dfn3_prepost.h`

與 DFN2 同一組函式（`dfn3_prepost_*`，含 `_pre_process_freq_dual`、`dfn3_model_run_frame()`），grid 用 `DFN3GridSpec`／`dfn3_grid_make()`（`lib/ainr/dfn3/dfn3_process.h`；欄位同 DFN2，沒有 `frames_per_call`）。差異：

- 兩個 heads 平行（不是 cascade）：模型延遲是 `grid.output_delay` = max(M, D) 個 frame（`dfn3_prepost_model_lookahead_frames()`）；ring 深度 `df_order + output_delay − D`（出貨 1/1 與 0/0、2/4 為 5，1/0 與 2/1 為 6，2/0 為 7）。
  lookahead pair 是 `DFN3GridSpec` 的欄位（M 0..2、D 0..`df_order`−1），在執行期給定；pair 不符由 descriptor、pool hash 與 grid 檢查在執行期擋下。出貨只有 1/1。
- graph 視窗固定三個 frame（`DFN3_MODEL_INPUT_FRAMES`）；heads 是 `erb_mask [n_erb]` 與 `coefs [df_bins][df_order][2]`，沒有 alpha。
- `_pre_process_freq()` 每拍回 1，前 M 拍是暖機幀（行為同 DFN2：加速器照跑、heads 丟棄、state 歸零、generation 加一）；`D > M` 時加速器也會在還沒有輸出的拍上跑。`_post_process_freq()` 的 `*valid` 在 `output_delay` 拍之前為 0（資訊用，頻譜為零，合成可每拍 push）。
- `_frame_bypass()`：unit ERB 增益加原始低頻 bin 上的一個 unit tap（identity）。沒有 mask-only 形態、沒有 `frames_per_call`。

#### GTCRN：`gtcrn_process.h`

沒有 `frame_*` 與 `create`／`destroy`：`GTCRNGridSpec { n_fft, erb_low_bins, erb_high_bands }` → `gtcrn_grid_make()` → `gtcrn_process_config_defaults(&cfg, &grid)` → `cfg.erb_fwd`／`cfg.erb_inv`（與可選的 `cfg.state[8]`）→
`gtcrn_process_get_mem_size(&cfg, &req)` → `gtcrn_process_init()`／`_init_ex()`；另有 `gtcrn_process_reset()`、`gtcrn_process_state_generation()`、`gtcrn_process_grid()`、`gtcrn_process_set_erb_matrices()`。
一次圖呼叫只吃一個 frame（歷史在 state tensor 裡），不要補成多個 frame。頻譜是 interleaved `[n_bins][2]`（audiolib 的 `Complex` 陣列），未正規化。

```c
Complex spec[257], enh[257];                       /* grid.n_bins = 257（n_fft 512）*/
ac_stft_push(stft, in_hop, spec);
gtcrn_model_input(p, (const float *)spec, mag, real_part, imag_part);   /* 各 grid.erb_bands 個 float */
run_gtcrn(mag, real_part, imag_part, mask_erb);    /* 加速器：mask_erb 為 interleaved [erb_bands][2]；8 個 state 就地更新 */
if (gtcrn_process_mask_validate(p, mask_erb) != 0 || gtcrn_process_state_validate(p) != 0)
    memcpy(enh, spec, sizeof enh);                 /* -1：頻譜直通 */
else
    gtcrn_model_output(p, mask_erb, (const float *)spec, (float *)enh);   /* enh 可以就是 spec */
ac_istft_push(istft, enh, out_hop);
```

要偵測沒寫的 mask，先把它填 NaN。state 以 `GTCRNStateId` 順序登錄或不登錄（§6.2）。

#### RNNoise-ERB：`process.h`

同樣沒有 `frame_*` 與 `create`／`destroy`：`RNNoiseGridSpec` → `rnnoise_grid_make()` → `rnnoise_config_defaults(&cfg, &grid)` → `cfg.erb_fwd`／`cfg.erb_inv`（與可選的 `h1`／`h2`／`h3`）→
`rnnoise_get_mem_size(&cfg, &req)` → `rnnoise_init()`／`_init_ex()`；另有 `rnnoise_reset()`、`rnnoise_state_generation()`。lookahead 是 `lookahead_frames`（0..2，出貨 1），`grid.conv_delay` = 2 − `lookahead_frames`。

```c
ac_stft_push_planar(stft, in_hop, re, im);
spectrum_ring_push(re, im);                                   /* 應用端保存最近 3 拍頻譜 */
if (rnnoise_compute_features(st, re, im, erb, spec) == 1) {   /* 3 拍窗滿了才回 1 */
    run_rnnoise(erb, spec, band_gains);                       /* 加速器：h1..h3 就地更新 */
    if (rnnoise_frame_finish(st, band_gains) != 0)            /* -1：用 unit 增益 */
        for (b = 0; b < grid.n_bands; ++b) band_gains[b] = 1.0f;
    rnnoise_apply_atten_lim(st, band_gains, atten_lim_db);    /* 可選 */
    rnnoise_expand_gains(st, band_gains, bin_gains);
    spectrum_ring_get(grid.conv_delay, dre, dim);             /* 這個增益屬於 t − lookahead_frames 那一拍 */
    rnnoise_apply_gains(st, dre, dim, bin_gains);             /* 原地；已含 stft_unscale，AcIstft 不再乘 */
    ac_istft_push_planar(istft, dre, dim, out_hop);
} else {
    memset(out_hop, 0, hop_size * sizeof(float));             /* 窗還沒滿：沒有輸出 */
}
```

`rnnoise_compute_features()` 的輸出是扁平陣列：`erb` 有 `rnnoise_erb_window_elements(&grid)` = 3·`n_bands` 個 float，`spec` 有 `rnnoise_spec_window_elements(&grid)` = 3·2·`spec_bins` 個 float。
最新窗 `[t−2, t−1, t]` 的增益屬於第 `t − lookahead_frames` 拍的頻譜，所以應用端要自己保存並延遲那一拍頻譜，取窗內 slot `grid.conv_delay`（0 = 最舊）。

### 6.7 Pool 大小

以現行 build 在 64 位元主機（`BACKEND=ne10`，NEON 開關不影響數字）量得，供規劃用；產品一律以 resolved config 呼叫 `_get_mem_size()`，
**不要把這裡的數字寫死**（[§7](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md#7-static-pool-共通規則) 規則同樣適用）。單位 bytes。32 位元目標的位元組數待量測（指標縮小，數字會不同）。

| 類別 / grid | pool | 備註 |
|---|---:|---|
| DeepFilterNet2 48 kHz／1024（出貨 grid，32 個 ERB band）1/1 | 92,608 | ring 5 |
| DeepFilterNet2 48 kHz／1024，2/4 | 116,672 | ring 7（2/3 與 1/4 為 ring 6） |
| DeepFilterNet2 48 kHz／1024，1/1，`frames_per_call` 4 | 144,112 | |
| DeepFilterNet2 16 kHz／256（24 個 ERB band）0/0 | 47,744 | 1/1 的 bytes 相同 |
| DeepFilterNet2 32 kHz／512（28 個 ERB band，規劃中、尚無訓練模型）1/1 | 62,784 | |
| DeepFilterNet2 `-DDFN2_MASK_ONLY` 48 kHz／1024，mask 1／df 0 | 32,224 | 16 kHz／256 為 11,936 |
| DeepFilterNet3 48 kHz／1024（出貨）1/1 | 90,384 | ring 5（0/0、2/4 同） |
| DeepFilterNet3 48 kHz／1024，2/0 | 114,448 | ring 7（1/0 與 2/1 為 ring 6，102,416） |
| GTCRN `n_fft` 512（65 個低頻 bin、64 個 band） | 3,328 | |
| RNNoise-ERB 16 kHz／512，22 個 band | 5,600 | `RNNoiseMemReq` 24 bytes |
| Align-ULCNet 16 kHz／512 | 14,784 | 與 D 無關（D=8 與 D=64 相同；hash 不同）；48 kHz／1024 為 29,120 |
| DeepVQE-S 16 kHz／512，D=32 | 35,504 | 與 D 無關；48 kHz／1024 為 70,320 |

pool 只裝特徵緩衝、ring、normaliser 與 staging：**不含遞迴 state**（應用端的緩衝，float host 登錄、板端在加速器裡；出貨 DFN2 圖的 state 是 `h_gru` 5×256 個 float 加
`df_convp_history` 64×5×96 個 float，共 128,000 bytes）、**不含 ERB 矩陣**（應用端讀入的區塊），**也不含 `AcStft`／`AcIstft`**（用 `ac_stft_get_mem_size()`／`ac_istft_get_mem_size()` 現查）。
DFN2／DFN3 的 pool 隨 grid 與 ring 深度變，與 lookahead pair 的關係只經 ring；Align-ULCNet 與 DeepVQE-S 的 pool 只隨 `n_bins` 變。

### 6.8 Framing：應用端用 audiolib 分析與合成

函式庫不持有 window、FFT 或 STFT 物件，也沒有 framing 設定函式：應用端用 audiolib 的 `AcStft`／`AcIstft`，設成各模型訓練時的 framing。共通設定是 `ac_stft_config_default(&sc, n_fft)`：
periodic sqrt-Hann 窗、center 0（一次 push 一框，起始之前補零歷史）、`hop = n_fft / 2`（50% overlap）、正向 FFT 不正規化、反向含 1/N、兩個 scale 預設 1.0、`AC_OLA_ACCUMULATE`；identity 鏈延遲一個 hop。

```c
AcStftConfig sc;
ac_stft_config_default(&sc, grid.n_fft);
sc.spec_scale  = grid.stft_scale;      /* DFN2、DFN3、RNNoise-ERB；其餘維持 1.0f */
sc.ispec_scale = grid.stft_unscale;    /* DFN2、DFN3；其餘維持 1.0f */
/* Align-ULCNet、DeepVQE-S 另設 sc.ola = AC_OLA_ENV_NORM（分析忽略 ola） */
size_t sb = ac_stft_get_mem_size(&sc);
size_t ib = ac_istft_get_mem_size(&sc);
AcStft  *stft  = ac_stft_init(alloc16(sb), sb, &sc);
AcIstft *istft = ac_istft_init(alloc16(ib), ib, &sc);
```

| 函式庫 | `spec_scale`（AcStft） | `ispec_scale`（AcIstft） | 合成 `ola` | 備註 |
|---|---|---|---|---|
| DFN2、DFN3 | `grid.stft_scale`（1/√n_fft） | `grid.stft_unscale`（√n_fft） | `AC_OLA_ACCUMULATE` | normalized=True 分析；center 0 與訓練圖的 center=True 只差第一框 |
| GTCRN | 1.0 | 1.0 | `AC_OLA_ACCUMULATE` | 未正規化；頻譜以 interleaved `Complex` 傳 |
| RNNoise-ERB | `grid.stft_scale` | 1.0 | `AC_OLA_ACCUMULATE` | `rnnoise_apply_gains()` 已乘 `grid.stft_unscale`，`AcIstft` 不可再乘 |
| Align-ULCNet | 1.0 | 1.0 | `AC_OLA_ENV_NORM`（第一框不輸出） | center 0 從第一次 push 起一 push 一框；Python 參考是 centered（0、2、之後每拍 1 框）；合成延遲 `grid.hop` |
| DeepVQE-S | 1.0 | 1.0 | `AC_OLA_ENV_NORM` | 從第一個 hop 起每拍一框，除 framing 本身的一個 hop 外沒有額外延遲 |

center = 0 與 AEC／BF seam 的頻譜是同一個慣例，所以只有一路頻譜在手的呼叫端可以用它補另一路而兩路保持 hop 對齊。

> **同一個 `AcStft` 不可混用兩種 `center`**：`center` 在建立時固定，兩種排程不共用物件。

`AcStft`／`AcIstft` 可以借用同一個 `FftHandle` 與窗表（`sc.fft`、`sc.window`；借用的窗表須活得比物件久），用法見 `audio_stft.h`。
錯誤的窗或 scale 在函式庫這一側偵測不到——兩邊都是量綱合法的 float 頻譜——所以 scale 要照上表設。

### 6.9 Pipeline 層：頻域 stage 串接與 NN 的 begin / nn_begin / nn_end / end

四個 pipeline 範例把三個 library 與這些類別串成一條鏈，範例碼在 `pipelines/mono_aec_nr_res/`、`pipelines/4ch_aec_bf_nr_res/`、`pipelines/dfn/`、`pipelines/ulcnet/`；接縫是借用視圖（mono 為 `aec.h` 的 `AecResContext`，4ch 為 `mcab.h` 的 `McabFrame`；有效至其產生者下一次執行；任何 stage 都不得寫入）。
增益類 stage 回傳 `const float*`（`aec_res_process()`、`anr_get_gain()`；mono 的 RES gain 也可直接讀
`ctx.res_gain`），各範例 `main` 在每個 hop 明寫 `sk_min_f32 → sk_capply_gain_f32 → aec_cng_fill`，再 `ac_istft_push`；
沒有函式指標表，也沒有把這幾步包起來的共用物件。

初始化也是各自為政：每個函式庫（`Aec`、`DelayAec`、`AecRes`（4ch 的範例自己建，`Mcab` 不含）、`AnrDenoiser`、`AcIstft`、`UlcnetNnStage`、`DfnResStage`／`DfnRateBridge`、
4ch 的 `Mcab`）有自己的大小查詢與 `init`，整合者加總大小並切塊；記憶體和（每個函式庫一行）見
[`pipelines/README_01_latency_call_flow.md`](../pipelines/README_01_latency_call_flow.md)「Memory Budget」與 [`integration_example.html`](html/integration_example.html)。

跑 NN 的應用把一個 hop 拆成四個呼叫，加速器夾在中間，模型 config 的 `infer` 回呼因此可以不設：

```c
/* DFN2：dfn/dfn_hop.h；ULCNet：ulcnet/ulcnet_nn_stage.h（begin 的引數不同） */
dfn_hop_begin(&hop, frame.e, P);                         /* P = B·G_res（B = frame.b）已由範例 main 先形成 */
while (dfn_hop_nn_begin(&hop, &job) > 0) {               /* 1 = 有一幀（暖機幀也是 1）；0 = 沒有（reprime／rate bridge 的 FIFO 未滿） */
    dfn_hop_nn_end(&hop, run_npu(&job));                 /* rc != 0 或輸出非有限 ⇒ identity（fail-open） */
}
dfn_hop_end(&hop, out);                                  /* 合成剛好一個 hop */
```

`job` 是 `UlcnetNnJob`（ULCNet）或 `DfnResStageJob`（DFN2，內含本章 `DFN2PrepostInputs`／`Outputs`）；
輸出預填 NaN，必須寫滿。DFN 應用經 rate bridge 時一個 native hop 可能開 0..2 個 48 kHz 幀，所以 `nn_begin` 放在迴圈裡。
`DfnResStage`／`DfnRateBridge` 的 config 收模型的 `DFN2Grid`、`erb_fwd`／`erb_inv`（partition of unity）與 `DFN2Model`；它們**不登錄 state**（state 歸 model callback），
callback 依 `dfn_res_stage_state_generation()`／`dfn_rate_bridge_state_generation()` 在值改變時重填量化後的 0（reset、每個暖機幀、被拒的 commit 各加一；`dfn_res_stage_begin()` 每拍回 1，暖機幀照呼叫 callback、`end()` 丟棄 heads）。它們收 audio-scale 頻譜、自己乘 `stft_scale`／`stft_unscale`，呼叫端的 `AcStft`／`AcIstft` 維持 scale 1。
`ulcnet_nn_stage_mem_size(&grid, err_analysis)` 與 `ulcnet_nn_stage_init(mem, bytes, &grid, err_analysis)` 收應用端的 `UlcnetGrid`。
4 聲道 Align-ULCNet 若要用同一 hop 的 lane 頻譜算出的 mask 與 VAD（例如 masker 讀 lane 0），把 sample helper 的 hop 拆成
`mcab_sample_beam_process_front()`（回傳唯讀 lane 頻譜）與 `mcab_sample_beam_process_beam(b, &front, vad_raw, vad_out, mask, &beam, &info)`
（`mask` 可為 NULL）；`info` 回 DOA（rad，NaN = 無）、BF 適應旗標與 `info.delay`。`mcab_sample_beam_process_with_activity(b, mics, far, v, v, NULL, &beam, &info)` 等於
`process_front` 加 `process_beam(front, v, v, NULL)`；`ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, {info.delay.changed, info.delay.solid})` 接在後面。任何一步被拒，整合者把 helper（連同 flow）與 stage 一起 reset。
沒被取走或沒被關閉的幀走 identity，輸出時序不隨模型健康度改變。範例與記憶體規則見
`pipelines/README_01_latency_call_flow.md` 的 Call flow 與 [`integration_example.html`](html/integration_example.html)。

### 6.10 各函式庫各自初始化：記憶體加總與 flow 的公開面

沒有「整條 pipeline 的 `get_mem_requirements()`」：每個函式庫查自己的大小，整合者加總（每塊 16-byte 對齊）。
下表是本 checkout、`BACKEND=ne10`、`SIMD=1`、16 kHz／256（預設格點）的實測值；換格點、backend、編譯選項都會變，
一律用各自的查詢重量。

| 函式庫物件 | 大小查詢 | 16 kHz／256 | 備註 |
|---|---|---:|---|
| `Aec`（mono，MATCHED n=5） | `aec_get_mem_size()` | 366,000 | 48 kHz／1024：1,162,736 |
| `AnrDenoiser`（函式庫 level） | `anr_get_mem_size()` | 45,344 | 範例程式有 NR 時 |
| `AcIstft`（合成） | `ac_istft_get_mem_size()` | 11,344 | 範例程式有合成時；此欄 11,344 是 4ch 範例 `mcab_static` 的合成（帶自己的窗表）；mono 範例的合成借用 AEC 的合成窗（`aec_synthesis_window()`），不另帶窗表，為 10,320 |
| `AecRes`（4ch 後波束 RES） | `aec_res_get_mem_size()` | 7,904 | `Mcab` 的 pool 不含；4ch 的範例自己建一個（48 kHz／1024：29,408） |
| 增益／頻譜暫存 | 範例程式自己算 | 2,624 | mono 與 `mcab_external_static` 為三個 `n_freqs` float 向量 + 一個複數向量，各自 `ALIGN16`；`mcab_static` 為兩個向量 + 一個複數向量（2,096） |
| 4ch lane（`Aec` × 4，`mcab_lane_config()`） | `aec_get_mem_size()` 每條 | 803,968（4 × 200,992） | 48 kHz／1024：2,936,832；由應用建立，不在 `Mcab` 的 pool 內 |
| 4ch flow `Mcab`（含共用 estimator 的 front end） | `mcab_get_mem_requirements()`；estimator 另用 `delay_aec_get_mem_size()` | `Mcab` pool 74,624；front end（estimator + `Mcab` pool）110,048 | `enable_post = 0` 時 `Mcab` pool 為 68,368；front end 48 kHz／1024：267,072（`mcab_external_static --print-mem-size`） |
| 4ch sample 的 `Unit`（lane + estimator + `Mcab` pool + SRP-PHAT／BF helper） | `mcab_sample_unit_memory()` | 1,706,240 | 48 kHz／1024：6,329,312（`mcab_static --print-mem-size`） |
| ULCNet stage | `ulcnet_nn_stage_mem_size(&grid, 0)` | 36,464 | grid 是執行期值；48 kHz／1024：71,792；`err_analysis = 1` 另加（16 kHz／512：42,704） |
| DFN2 stage／rate bridge | `dfn_res_stage_get_mem_requirements()`／`dfn_rate_bridge_get_mem_requirements()` | 105,056／222,960 | bridge 內含 stage；16 kHz 走 bridge，48 kHz 直接用 stage 加呼叫端的 `AcIstft`；數字是出貨的 48 kHz／1024 grid、lookahead 1/1（兩支範例 main 的預設）、host 16 kHz／256，其他 grid 或 pair 的 pool 大小要現查 |

mono 的 16 kHz／256 合計 = `ALIGN16(366,000) + 45,344 + 10,320 + 2,624 = 424,288`（AEC + NR + 合成 + 暫存）；4ch sample 的 `Unit` 加 RES／NR／合成／暫存 = `1,706,240 + (7,904 + 45,344 + 11,344 + 2,096) = 1,772,928`（`mcab_static --sample-rate 16000` 印的 `Total:`）。

**flow 的公開面**（`mcab.h`）：`mcab_default_config`、`mcab_apply_lane_contract`／`mcab_lane_config`／`mcab_delay_estimator_config`、
`mcab_get_mem_requirements`、`mcab_init`／`mcab_init_ex`／`mcab_create`、`mcab_process_front`、`mcab_fuse`（可選）、`mcab_reset`／`mcab_destroy`、
形狀與生效值 accessor（`mcab_hop_size()`、`mcab_fft_size()`、`mcab_n_freqs()`、`mcab_sample_rate()`、`mcab_enable_post()`）、`mcab_get_last_timing`。
輸出是借用的 `McabFront`（lane 頻譜與對齊後的 far），`mcab_fuse()` 之後另有 `McabFrame`（`g_res_ctx` 為 NULL）；beamformer、RES、NR、comfort noise、合成是你的後級
（範例的 beamformer 是 sample 層 helper `mcab_sample_beam_*`，不在 archive 內）。

**串接順序**（4ch，一 hop）：`mcab_process_front()` → beamformer → `mcab_fuse()`（`frame.g_res_ctx` 為 NULL）→ `mcab_sample_res_input()` + `aec_res_process()` → `mcab_sample_anr_gain()` → `sk_min_f32()` →
`sk_capply_gain_f32()` → `aec_cng_fill()` → `ac_istft_push()`。**mono**：`aec_process_context()` → `aec_get_res_context()`（`ctx.res_gain` 是 `Aec` 自己的 RES gain）→ `aec_get_output_base()`（渲染底座，NR 估計與增益都乘在它上面）→ `anr_process_gain()`／`anr_get_gain()` → 同上。
失敗時重置 flow（`mcab_reset()` 連 lane 與 estimator 一起重置；或 mono 的 `Aec`）、4ch 的 `AecRes`、降噪器、合成與 comfort-noise RNG（以及 NN stage）。

---

[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md) ｜ [下一份](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md) <!-- split-nav -->
