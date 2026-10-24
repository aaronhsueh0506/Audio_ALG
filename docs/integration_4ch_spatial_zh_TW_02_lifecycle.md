> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_01_quick_start.md) ｜ [下一份](integration_4ch_spatial_zh_TW_03_config.md) <!-- split-nav -->
## 3. Lifecycle

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | query | `audio_pipeline_4ch_get_mem_requirements(&cfg, &req)` | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | 回傳 `-1`：`out` 為 NULL；`cfg` 未通過本層驗證（[§4.2](integration_4ch_spatial_zh_TW_03_config.md#42-陣列幾何4--2-欄)）或核心層驗證（[§4.1](integration_4ch_spatial_zh_TW_03_config.md#41-cfgcore核心層欄位)）；SRP 或 BF 回報 size 0；此 TU 編譯時沒帶 `-DAUDIO_PIPELINE_BACKEND_STR="ne10"` |
| 2 | allocate | 你自己的配置器，取得 `req.bytes` 且對齊 `req.alignment`（16）的記憶體 | 拿到 `req` 之後 | pool **不需要**預先清零（poison pattern 可接受）。`init_ex()` 自己會把整塊 memset 成 0 |
| 3 | init | `audio_pipeline_4ch_init_ex(pool, req.bytes, &cfg, &req)` | 拿到 pool 之後，處理第一個 hop 之前 | 回傳 `NULL`：8 項 descriptor 檢查任一不符；pool 未 16-byte 對齊；pool 太小；config 無效；core / SRP / BF 任一 init 失敗。**沒有任何診斷輸出**（見 [§5.4](integration_4ch_spatial_zh_TW_04_errors.md#54-診斷輸出去向)） |
| 4a | process | `audio_pipeline_4ch_process_front()` → `audio_pipeline_4ch_process_beam()`；或 `process_with_activity()`；或 `process()` | 每收到一個 hop 呼叫一次 | `-1` / `-2` / `-3`，見 [§5.2](integration_4ch_spatial_zh_TW_04_errors.md#52-process-各入口與-fuse-的完整拒絕清單) |
| 4b | fuse（可選） | `audio_pipeline_4ch_fuse(p, &frame)` | 同一 hop 的 beam 之後、下一次 `process_front()` 之前 | `-1` / `-2` / `-3`，見 [§5.2](integration_4ch_spatial_zh_TW_04_errors.md#52-process-各入口與-fuse-的完整拒絕清單) |
| 4c | 你的後級 | `se_res_stage_gain()` → `se_anr_stage_gain()` → `sk_min_f32()` → `sk_capply_gain_f32()` → `aec_cng_fill()` → `ac_istft_push()`，或 NN stage | fuse（或 beam）之後 | 由你負責；`frame.g_res_ctx` 為 NULL，RES 增益來自你自己的 `AecRes`。任何一步被拒，你必須把單元與你的後級（含 `aec_res_reset()`）**一起** reset（§3.1） |
| 5 | reset | `audio_pipeline_4ch_reset(p)`，加上你的後級的 reset | 換回聲路徑、換場景、切換到不相關的另一條串流時 | 不會失敗。`p == NULL` 或已 destroy 時是 no-op |
| 6 | destroy | `audio_pipeline_4ch_destroy(p)` | 不再處理任何 hop 之後、釋放 pool 之前 | 不會失敗。NULL 安全。pool instance 冪等；`create()` instance 只能呼叫一次 |
| 7 | release | 你自己的釋放器 | **一定要在第 6 步之後** | 順序顛倒 = use-after-free |

heap 便利路徑：`audio_pipeline_4ch_create(&cfg)` 把第 1~3 步合成一步。

`audio_pipeline_4ch_init(pool, bytes, &cfg)` 等同
`audio_pipeline_4ch_init_ex(pool, bytes, &cfg, NULL)` —— 不做 descriptor 檢查。

### 3.1 `reset()` 到底重置了什麼，以及失敗時整合者要做什麼

單元的 `reset()` 重置：

* 前端（delay matcher、四路 AEC、front 有效旗標）
* SRP-PHAT 狀態
* BF 狀態
* 本層的內建 VAD 狀態（`noise_power` 回到門檻值、hangover 歸零）
* `frame_index` 與 per-hop phase 歸零

它**不會**重置你的後級（`AecRes`、NR、CNG rng、合成、NN stage），也**不會**重新驗證 config，
更不會重新計算 [§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值) 那三個「生效值」——那三個是在 init 時算好的，`reset()` 不動它們。

**失敗語意是整合者的責任。** 單元的任何一個呼叫回非 `OK`，或你的 `se_res_stage_gain()`／`se_anr_stage_gain()` 回 `NULL`，
或合成輸出不是有限值，你要**同時**呼叫 `audio_pipeline_4ch_reset()`、`aec_res_reset()`、`anr_reset()`、`ac_istft_reset()`，
並把 comfort-noise RNG 設回 `SE_CNG_RNG_SEED`，以及用到 NN stage 時的 `ulcnet_nn_stage_reset()`。
漏掉其中一個，系統就不等於新建的實例。DFN2 的 stage 不因為 AEC 的 delay change 而 reset（見
`pipelines/dfn_res_stage.h`）。

### 3.2 執行期唯讀存取

形狀 accessor（`p == NULL` 或已 destroy 時全部回 `-1`）：

| 呼叫 | 回傳 |
|---|---|
| `audio_pipeline_4ch_hop_size(p)` | 每次 process 要餵幾個 sample（per channel） |
| `audio_pipeline_4ch_frame_size(p)` | frame 長度（等於 FFT 長度） |
| `audio_pipeline_4ch_fft_size(p)` | FFT 長度 |
| `audio_pipeline_4ch_n_freqs(p)` | 頻域 bin 數 |
| `audio_pipeline_4ch_sample_rate(p)` | 實際生效的取樣率 |
| `audio_pipeline_4ch_doa_sample_rate/frame_size/hop_size/fft_size(p)` | DOA 階段的格點 |
| `audio_pipeline_4ch_bf_sample_rate/frame_size/hop_size/fft_size(p)` | BF 階段的格點 |

AEC、DOA、BF，以及你串的 RES 與 NR，全部**共用同一組格點**，`frame == fft_size`、`hop == fft_size / 2`。
你不能分別指定互相矛盾的 frame / hop / FFT。上面三組 accessor 之所以分開，
是讓整合測試能證明三者確實一致（實測 16 kHz 預設下三組皆為 hop 128 / fft 256 / n_freqs 129）。

結構稽核 accessor（`p == NULL` 或已 destroy 時回 `0`）：

| 呼叫 | 有效 handle 的回傳 |
|---|---|
| `audio_pipeline_4ch_matched_filter_count(p)` | `1`（`MATCHED`）；`FIXED`／`EXTERNAL_ALIGNED` 時 `0` |
| `audio_pipeline_4ch_linear_aec_count(p)` | `4` |

單元的結構稽核計數只有上面兩個；RES、NR 與合成由你自己決定，單元不持有它們。

生效值 accessor（見 [§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值)）：`audio_pipeline_4ch_bf_effective_adapt_interval(p)`、
`audio_pipeline_4ch_bf_lambda(p)`。

診斷：`audio_pipeline_4ch_get_core(p)`（唯讀的 `const FourAecNrRes*`，給核心層的稽核 accessor 與
`four_aec_nr_res_get_last_timing()` 用；不要拿它去呼叫會改狀態的函式）與
`audio_pipeline_4ch_last_core_timing(p, &t)`（核心最近一個 hop 的各階段耗時，`FourAecNrResLastTiming`；
`p` 為 NULL 時把 `t` 清零，`t` 不得為 NULL）。

建置資訊：`audio_pipeline_4ch_spatial_backend()` —— **不吃 handle**，回傳一個
`const char*`（永不為 NULL），表示 spatial kernel 實際編進去的是哪一種實作
（NEON 或 scalar）。純資訊用途，可寫進你的開機 log。

### 3.3 執行期強度控制

單元**沒有**強度 setter 函式，也不持有 `AecRes`：四條 lane 只算線性殘留，RES 與 NR 都是你串的後級。
你對自己 init 的物件直接呼叫函式庫的 setter（完整契約見核心手冊 [`integration_4ch_core_zh_TW.md` §3.2](integration_4ch_core_zh_TW_02_lifecycle.md#32-執行期強度控制)）：

```c
aec_res_set_preset(res, AEC_PRESET_AGGRESSIVE, 100.0f);   /* res 是你自己的 AecRes；回 0 成功 */
anr_set_mode(nr, ANR_MODE_MILD);                  /* 函式庫的 setter；nr 是你自己的降噪器 */
float live, target;                               /* 線性功率 */
if (aec_res_split_floor(res, &live, &target) == 0 && live == target) {
    /* ramp 已經走完 */
}
```

這裡只重述整合時最容易踩到的一點：

> **不要自己對四條 lane 迴圈呼叫 `aec_set_preset()`。** 四條 lane 都以
> `spatial_linear_context` 建立，永遠不會走到自己的 suppression-gain 路徑，
> 它們的地板什麼都不塑形。真正乘上輸出的 gain 來自你串的**唯一一個**
> post-beam `AecRes`；同樣地，NR 也只有一個共用實例，不是每條 lane 一個。
> 上面兩個 setter 針對的就是那兩個共用實例。單元的 `reset()` 不碰它們，
> 你自己呼叫的 `aec_res_reset()` 不會丟掉用 `aec_res_set_preset()` 設的 preset。

`ramp_ms == 0` 代表下一個 hop 就套用（**不是錯誤**），`> 0` 則以 dB 為單位線性走
過去，上限 60 秒。兩個 setter 都不是重啟：AEC、SRP、BF、NR 的狀態全部繼續跑；
preset 的改動撐得過該 stage 的 reset，ramp 進度不會。**在兩個 hop 之間
呼叫、與 process 序列化；非 thread-safe。**

A/B 量測時：far-active 地板只在 **far-active 且非 double-talk** 的 hop 上生效
（DT 地板：mild/balanced 共用 -20 dB，aggressive 為 -30 dB；latch 觸發前用的是 far-silent 地板），而同一個 gain
還決定 comfort noise 量，所以整段錄音的平均值移動幅度會小於 dB 落差所暗示的量。
細節與建議的比較方式見核心手冊 §3.2。

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_01_quick_start.md) ｜ [下一份](integration_4ch_spatial_zh_TW_03_config.md) <!-- split-nav -->
