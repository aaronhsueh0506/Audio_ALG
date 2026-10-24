導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW_01_library_quickstart.md) ｜ [下一部分](integration_4ch_core_zh_TW_03_config.md) <!-- split-nav -->
## 3. Lifecycle

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | query | `four_aec_nr_res_get_mem_requirements(&cfg, &req)` | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | 回傳 `-1`：`out` 為 NULL；`cfg` 未通過驗證（[§4](integration_4ch_core_zh_TW_03_config.md#4-config-完整參考)）；此 TU 編譯時沒帶 `-DAUDIO_PIPELINE_BACKEND_STR="ne10"` |
| 2 | allocate | 你自己的配置器，取得 `req.bytes` 且對齊 `req.alignment`（16）的記憶體 | 拿到 `req` 之後 | pool **不需要**預先清零（poison pattern 可接受，已實測）。`init_ex()` 自己會把整塊 memset 成 0 |
| 3 | init | `four_aec_nr_res_init_ex(pool, req.bytes, &cfg, &req)` | 拿到 pool 之後，處理第一個 hop 之前 | 回傳 `NULL`：8 項 descriptor 檢查任一不符；pool 未 16-byte 對齊；pool 太小；config 無效；子模組 init 失敗。**沒有任何診斷輸出**（見 [§5.5](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#55-診斷輸出去向)） |
| 3b | 後級 stage | 各 stage 自己的 `aec_res_get_mem_size()`／`anr_get_mem_size()`／`ac_istft_get_mem_size()` → `aec_res_init()`／`anr_init()`／`ac_istft_init()` | 你的後級 stage 要用時 | `mem_size` 回 `0`、`init` 回 `NULL`：config 無效或 pool 太小／未對齊。stage 的記憶體由你自己查大小、自己 carve，與前端的 pool 分開（[§2](integration_4ch_core_zh_TW_01_library_quickstart.md#2-quick-start) 的範例） |
| 4a | front_process | `four_aec_front_process(p, mics, ref, &front)` | 每收到一個 hop 呼叫一次 | `-1` / `-3`，見 [§5.2](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#52-four_aec_front_process-的完整拒絕清單依實際檢查順序) |
| 4b | 你的 beamformer | — | 在 front 與 fuse 之間 | 由你負責 |
| 4c | fuse | `four_aec_fuse(p, weights, beamformed_error, &frame)` | 緊接著同一幀的 `front_process()` | `-1` / `-2` / `-3`，見 [§5.3](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#53-four_aec_fuse-的完整拒絕清單依實際檢查順序) |
| 4d | 你的後級 stage | `se_res_stage_gain` … `ac_istft_push` | fuse 之後、下一次 `front_process()` 之前 | 由你負責 |
| 5 | reset | `four_aec_nr_res_reset(p)` | 換回聲路徑、切換串流時 | 不會失敗。`p == NULL` 或已 destroy 時是 no-op。**會作廢目前的 front 視圖與 `SeSpecFrame`**；你的 stage 要各自 reset（`aec_res_reset`、`anr_reset`、`ac_istft_reset`，並把 CNG 的 rng 設回 `SE_CNG_RNG_SEED`） |
| 6 | destroy | `four_aec_nr_res_destroy(p)` | 不再處理任何 hop 之後、釋放 pool 之前 | 不會失敗。NULL 安全。pool instance 冪等；`create()` instance 只能呼叫一次 |
| 7 | release | 你自己的釋放器 | **一定要在第 6 步之後** | 順序顛倒 = use-after-free |

heap 便利路徑：`four_aec_nr_res_create(&cfg)` 把第 1~3 步合成一步，
`four_aec_nr_res_destroy()` 會釋放它自己配置的那一塊。其餘完全相同。

### 3.1 執行期唯讀存取

| 呼叫 | 回傳 | NULL / 已 destroy 時 |
|---|---|---|
| `four_aec_nr_res_hop_size(p)` | 每次 front_process 要餵幾個 sample（per channel） | `-1` |
| `four_aec_nr_res_fft_size(p)` | FFT 長度 | `-1` |
| `four_aec_nr_res_n_freqs(p)` | 頻域 bin 數 | `-1` |
| `four_aec_nr_res_sample_rate(p)` | 實際生效的取樣率 | `-1` |
| `four_aec_nr_res_matched_filter_count(p)` | `MATCHED` 時為 `1`，`FIXED`／`EXTERNAL_ALIGNED` 時為 `0` | `0` |
| `four_aec_nr_res_duty_hops_run(p)` | 共用 estimator 實跑 matched-filter 分析的 hop 數（duty-cycle census 分子；分母用 `estimator_calls`，`duty_hops_total()` 是其別名） | `0` |
| `four_aec_nr_res_linear_aec_count(p)` | 恆為 `4` | `0` |
| `four_aec_nr_res_far_fft_real_compute_count(p)` | 累計計數（`long`），每次 `front_process()` **+1**（不是 +4）；`reset()` 會歸零 | `0` |

後面的計數是**結構稽核用**的：前兩個對有效 handle 回傳固定常數（`linear_aec_count` 恆 4），
用來在整合測試裡證明拓撲沒有被改掉；`far_fft_real_compute_count` 用來證明跨聲道的 far-end 共用真的生效。
它們對處理結果沒有任何影響。後級 stage 是否存在由你自己的串接決定，前端沒有 `nr`／`post_res` 的稽核計數。

### 3.1b 逐階段耗時（診斷用）

```c
typedef struct FourAecNrResLastTiming {
    uint32_t delay_us, frontend_us, linear_us, lane_res_us;   /* four_aec_front_process */
    uint32_t fuse_us;                                         /* four_aec_fuse          */
} FourAecNrResLastTiming;

void four_aec_nr_res_get_last_timing(const FourAecNrRes* p,
                                     FourAecNrResLastTiming* out);
```

最近一個 hop 前端每個階段的 wall-clock 成本（微秒，`CLOCK_MONOTONIC`）。純診斷：
pipeline 自己不回讀，也不影響處理。`p` 為 NULL 或已 destroy 時把 `out` 清零而
不是失敗；`out` 不得為 NULL。RES／NR／CNG／逆轉換由你的應用串接，耗時由你自己量。

本層自己量的是 `fuse_us` 與 `delay_us` 的共用延遲估計部分（不含 `align_render()` 的 ring-buffer 複製與
realign 掃描）。`frontend_us`、`linear_us`、`lane_res_us` 來自 `aec_get_last_timing()`
（`aec.h`）：四條 lane 各讀一次再相加，所以得到的是該階段在整個 lane 迴圈上的
wall-clock 成本。lane 內部的 `delay_us` 也一併加進上面的 `delay_us`——它是同一
個量（對齊遠端所花的時間），且**結構上恆為 0**，因為四條 lane 都是
`AEC_DELAY_EXTERNAL_ALIGNED`，沒有自己的 ring 也沒有估計器；相加而不是丟棄，是
為了讓日後改變 lane 模式時不會靜默漏掉那筆成本。

這三欄由**另一半旗標** `-DAEC_STAGE_TIMING=1` 管轄。只建本層那半，這三欄讀 0
而其餘照常量測，是可讀的狀態而非壞掉；`make PROFILE=1` 兩半一起開。

各階段**加起來不等於**整個呼叫的時間，呼叫端要自己用減法補餘額：front 側餘額涵蓋
輸入去交錯／有限值檢查、`align_render()` 與 realign 掃描。

前半在 `four_aec_front_process()` 接受引數**之後**清零，`fuse_us` 在 `four_aec_fuse()` 接受之後清零，所以中途 bail-out 的 hop 會回報 0 而不是
上一個 hop 的數字；被引數檢查直接拒絕的呼叫則整份記錄不動。

解析度是微秒，所以低於 1 µs 的階段會讀到 0。

**目標平台沒有 `CLOCK_MONOTONIC` 時**：`clock_gettime` 是 POSIX 而非 C99。用
`make PROFILE=1 EXTRA_CFLAGS='-DFOUR_AEC_NR_RES_NOW_US=board_timer_us
-DAEC_NOW_US=board_timer_us -include my_timer.h'` 換掉時鐘；巨集要是**純識別字**
（Makefile 的 FP policy 不允許 `EXTRA_CFLAGS` 出現括號），指向不收參數、回傳
`uint32_t` 微秒的函式。替代時鐘必須**單調**，否則無號減法會產生接近 32-bit 全距
的荒謬值。本層與 `lib/aec` 各有自己的覆寫點，刻意不共用——一個元件的時鐘屬於該
元件自己的建置契約。開著旗標但指向常數函式是合法用法，所有欄位讀 0。

### 3.2 執行期強度控制

強度面屬於擁有 RES 與 NR 的那一層。前端 `FourAecNrRes` **沒有** preset／mode setter；
你自己串後級時，直接對你 carve 的 stage 呼叫：

```c
aec_res_set_preset(res, AEC_PRESET_AGGRESSIVE, 100.0f);          /* 回 0 成功 */
anr_set_mode(nr, ANR_MODE_MILD);
float live, target;
aec_res_split_floor(res, &live, &target);                        /* 唯讀，線性功率 */
```

上式的 `res`、`nr` 就是你自己 `aec_res_init()`／`anr_init()` 回來的那兩個 handle
（`4aec_nr_res_static.c` 的 `Post` 結構）。線性 AEC + BF 單元同樣不持有 RES，強度也是對你自己建的 `AecRes`
呼叫 `aec_res_set_preset()`（見 `docs/integration_4ch_spatial_zh_TW_02_lifecycle.md`）。

> **本節最重要的一件事：對四條 lane 重新指定 preset 是無效操作。**
>
> 四條 AEC lane 都以 `spatial_linear_context` 建立，因此**沒有任何一條**會走到
> 自己的 suppression-gain 路徑；它們各自的地板什麼都不塑形。真正乘上輸出、
> 也真正決定 comfort noise 量的那個 gain，來自**唯一一個**共用的 post 級抑制器
> （`AecRes`；核心前端與單元都由你的應用擁有）。這也是為什麼 preset setter 要作用在它上面，而不是要你自己
> 對四條 lane 迴圈呼叫 `aec_set_preset()`。這一點由測試釘住
> （`pipelines/4ch_aec_bf_nr_res/tests/test_audio_pipeline_4ch.c` 的 `test_caller_res_reset_and_preset`），不是推測。

| 呼叫 | 改／讀什麼 | 失敗（回 `-1`，**什麼都不寫**） |
|---|---|---|
| `aec_res_set_preset(res, preset, ramp_ms)` | **共用 post 級抑制器**的 far-active split floor | `res` 為 NULL；`preset` 超出 enum；`ramp_ms` 非有限值或超出 `[0, 60000]` |
| `anr_set_mode(nr, mode)` | 你擁有的**那一個**降噪器（NR 是對 beamform 後的訊號跑，不是每條 lane 一個） | `nr` 為 NULL；`mode` 超出 enum |
| `aec_res_split_floor(res, &live, &target)` | 唯讀。`live` = 抑制器**當下**套用的值，`target` = 撐得過 reset 的已設定值；兩者只有在 ramp 走到一半時才不同。任一指標可為 NULL | `res` 為 NULL |

`live` / `target` 這一對是**板端唯一值得記錄的強度量**：它是真正塑形輸出的東西
（lane 自己的地板什麼都不塑形），而兩個值合在一起就回答了「我要求的改動落地了沒有」。

`ramp_ms == 0` 代表下一個 hop 就套用，**不是錯誤**，落點與「用該 preset 從頭建一個
新實例」完全相同；`> 0` 則以 dB 為單位線性走過去，上限 60 秒。mild ↔ aggressive 是
18 dB 落差、地板又是硬性 clamp，所以互動式旋鈕應該給一個 ramp。ramp 進行中再呼叫
一次，會從當前的 live 值重新起走。

兩個 setter 都**不是重啟**：四條 lane 的濾波器、共用延遲鎖定、NR 的噪聲底與增益
平滑歷史全部繼續跑。preset 的改動**撐得過**該 stage 的 reset（目標值另存一份，reset 會重新套用），但 ramp 進度會被丟掉。
**在兩個 hop 之間呼叫、與 `front_process()`／`fuse()` 序列化；非 thread-safe。**

> 範例的降噪器以 `anr_config_for_mode_grid()` 建立、沒有覆寫任何欄位（最小值追蹤視窗 `L` 用函式庫預設），所以 `anr_set_mode()`
> 直接換強度。若你在 preset 之上改過欄位，`anr_set_mode()` 會把它們蓋回 preset 值：
> 那種實例請自己組好完整 target 再呼叫 `anr_reconfigure()`。

**A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
hop 上生效：double-talk 期間強制套用 DT 地板（mild/balanced 共用 -20 dB，aggressive 為 -30 dB）；far-active latch 觸發之前套用的是 far-silent 地板。同一個 gain 還
決定注入的 comfort noise 量（振幅正比於 `sqrt(1 − G_res²)`，再乘該 bin 的 NR gain（下限 −10 dB）——地板壓得越深、CNG
反而越多）。因此**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且一個只量
echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制頭上。請在 echo 對齊或
degradation 對齊的條件下比較，並實際試聽。

---

導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW_01_library_quickstart.md) ｜ [下一部分](integration_4ch_core_zh_TW_03_config.md) <!-- split-nav -->
