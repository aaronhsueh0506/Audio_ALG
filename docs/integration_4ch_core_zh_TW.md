# 四麥克風核心 Pipeline 整合手冊（`libaudio_pipeline_4ch.a` / `4aec_nr_res.h`）

適用對象：**已經擁有自己的 beamformer（SRP-PHAT / GSC 或任何等效方案）**，
只想要四路 linear AEC 前端 + 單路 post-beam NR/RES 後端的函式庫使用者。

如果你**沒有**自己的 beamformer、想要含 direction finding 與 beamformer 的完整
pipeline，請看 `docs/integration_4ch_spatial_zh_TW.md`。

本文只講「什麼時候呼叫哪個 function」「每個參數怎麼設」。演算法內部推導不在本文範圍。

> **路徑慣例**：本文所有路徑都相對於 Audio_ALG repository 根目錄。
> `audio_common` 是與 Audio_ALG 平行的另一個 checkout，本文寫成 `<audio_common>/`；
> 從 `pipelines/4ch_aec_bf_nr_res/` 出發的相對路徑是 `../../../audio_common`。
>
> **數值來源**：本文所有數字都是從當前 source 讀出，或以當前 checkout 實際建置後量測。
> 記憶體數字**必須**用 API 重新查詢，不要抄本文（見 §4.6）。

---

## 1. 這個函式庫是什麼、你連結什麼、你 `#include` 什麼

### 1.1 這是什麼

`4aec_nr_res.h` 提供的是一條**中間被切開**的處理鏈：

```
mic hop [hop][4] + ref hop [hop]
   │
   ├─► process_pre()   1 個共用 delay matcher → 1 條共用對齊後 reference → 4 條獨立 linear AEC
   │                    回傳：4 路 linear 頻譜 + 1 份 interleaved 時域輸出 + 1 個 token
   │
   ▼
 【你的 beamformer】    讀那 4 路頻譜，算出這一幀的「有效權重」Complex[4][n_freqs]
   │
   ▼
   └─► process_post()  以那組權重同調投影 → 1 次 post-beam RES gain → 1 次 NR → 1 次 iFFT/OLA
                        回傳：1 個 hop 的 mono 輸出
```

這個模組**明確擁有**：1 個共用 delay matcher、4 個獨立 linear AEC adaptive filter、
1 個 post-beam residual suppressor、1 個 mono MMSE-LSA denoiser、1 條最終 iFFT/OLA。

它**明確不做**：beamformer、DOA、VAD。它也不會挑某一支麥克風的 RES gain，
更不會每聲道各跑一份 NR/RES。

### 1.2 Header closure

```c
#include "4aec_nr_res.h"
```

一行就夠。它自己會拉進 `aec.h` 與 `mmse_lsa_denoiser.h`，後者再拉進 `fft_wrapper.h`。

### 1.3 `Complex` type 從哪來

`Complex` 定義在 `<audio_common>/include/fft_wrapper.h`，
透過 `4aec_nr_res.h` → `aec.h` → `fft_wrapper.h` 傳遞進來。
**你會直接用到它**：

* `FourAecNrResPreFrame.linear_spectra[ch]` 是 `const Complex*`
* `four_aec_nr_res_process_post()` 的 `weights` 是 `const Complex*`

不需要另外 include `fft_wrapper.h`。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I pipelines/4ch_aec_bf_nr_res
-I lib/aec/c_impl/include
-I lib/nr/c_impl/include
-I <audio_common>/include
```

四個就夠（已實測）。**注意：不需要 `third_party/doa`、`third_party/GSC`、`third_party/utility`**
—— 那三個是完整 spatial wrapper 才需要的（見另一份手冊）。

語言標準：`-std=gnu99`（或以上的 GNU 方言）。

### 1.5 Archive link order

```
你的 .o
libaudio_pipeline_4ch.a
libaec.a
libmmse_lsa.a
libaudio_common.a
-lm
```

* `libaudio_pipeline_4ch.a` 只含**一個** object：`4aec_nr_res.o`（Makefile
  `PIPELINE_OBJS` 只列這一個 `.o`）。`audio_pipeline_4ch.o`（完整 spatial
  wrapper）**不在這個 archive 裡**——需要它的每個執行檔（`audio_pipeline_4ch_raw`
  等）各自把 `audio_pipeline_4ch.o` 當獨立 object 加進自己的 link 命令，不是
  從 archive 帶出來的。只用本手冊這套 API 時，你的 link 命令根本不提
  `audio_pipeline_4ch.o`，所以你**不需要**連 `libdoa.a` / `libgsc.a` /
  `libspatial_common.a`。
  （已實測：上面五項就能把只用 `4aec_nr_res.h` 的程式連成執行檔。）
* archive 路徑是 config-keyed 的（`bin/<backend>-<hash>/`）。用各 Makefile 的
  `make -s print-lib-path` 取得，不要硬寫。
* `BACKEND=ne10` 時最終 link 要用 C++ driver（`c++`）。
* 四個 archive 必須用同一組 backend / 編譯選項建置，否則會在
  `four_aec_nr_res_init_ex()` 的 descriptor 檢查被擋下。

---

## 2. Quick start

完整、可編譯、已實際跑過的最小整合（heap 路徑，權重先用固定值佔位）：

```c
#include <stdio.h>
#include "4aec_nr_res.h"

int main(void) {
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);

    FourAecNrRes* p = four_aec_nr_res_create(&cfg);   /* heap path */
    if (!p) { fprintf(stderr, "create failed\n"); return 1; }

    const int hop     = four_aec_nr_res_hop_size(p);
    const int n_freqs = four_aec_nr_res_n_freqs(p);
    float mics[4 * 1024], ref[1024], out[1024];
    Complex weights[FOUR_AEC_NR_RES_CHANNELS * 1024];

    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < hop; i++) {                    /* 換成你自己的擷取來源 */
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ch++)
                mics[i * FOUR_AEC_NR_RES_CHANNELS + ch] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }

        FourAecNrResPreFrame pre;
        if (four_aec_nr_res_process_pre(p, mics, ref, &pre) != FOUR_AEC_NR_RES_OK) break;

        /* 你的 beamformer 在這裡讀 pre.linear_spectra[ch][bin] / pre.linear_interleaved，
         * 並寫出它這一幀的有效權重，channel-major：[ch * n_freqs + bin] */
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ch++)
            for (int k = 0; k < n_freqs; k++) {
                weights[ch * n_freqs + k].r = 0.25f;
                weights[ch * n_freqs + k].i = 0.0f;
            }

        if (four_aec_nr_res_process_post(p, &pre.token, weights, out) != FOUR_AEC_NR_RES_OK) break;
        /* 消費 out：hop 個 float，已增強的 mono 音訊 */
    }

    four_aec_nr_res_destroy(p);
    printf("ok hop=%d n_freqs=%d\n", hop, n_freqs);
    return 0;
}
```

編譯：

```sh
cc -std=gnu99 -O2 -Wall -Wextra -o app app.c \
  -I pipelines/4ch_aec_bf_nr_res \
  -I lib/aec/c_impl/include \
  -I lib/nr/c_impl/include \
  -I <audio_common>/include \
  <4ch-bin>/libaudio_pipeline_4ch.a \
  <aec-bin>/libaec.a \
  <nr-bin>/libmmse_lsa.a \
  <audio_common-bin>/libaudio_common.a \
  -lm
```

---

## 3. Lifecycle

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | query | `four_aec_nr_res_get_mem_requirements(&cfg, &req)` | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | 回傳 `-1`：`out` 為 NULL；`cfg` 未通過驗證（§4）；此 TU 編譯時沒帶 `-DAUDIO_PIPELINE_BACKEND_STR="kiss"` 或 `"ne10"` |
| 2 | allocate | 你自己的配置器，取得 `req.bytes` 且對齊 `req.alignment`（16）的記憶體 | 拿到 `req` 之後 | pool **不需要**預先清零（poison pattern 可接受，已實測）。`init_ex()` 自己會把整塊 memset 成 0 |
| 3 | init | `four_aec_nr_res_init_ex(pool, req.bytes, &cfg, &req)` | 拿到 pool 之後，處理第一個 hop 之前 | 回傳 `NULL`：8 項 descriptor 檢查任一不符；pool 未 16-byte 對齊；pool 太小；config 無效；子模組 init 失敗。**沒有任何診斷輸出**（見 §5.5） |
| 4a | process_pre | `four_aec_nr_res_process_pre(p, mics, ref, &pre)` | 每收到一個 hop 呼叫一次。**必須**在對應的 `process_post()` 之後才能再呼叫 | `-1` / `-2` / `-3`，見 §5.2 |
| 4b | 你的 beamformer | — | 在 pre 與 post 之間 | 由你負責 |
| 4c | process_post | `four_aec_nr_res_process_post(p, &pre.token, weights, out)` | 緊接著同一幀的 `process_pre()` | `-1` / `-2` / `-3`，見 §5.3 |
| 5 | reset | `four_aec_nr_res_reset(p)` | 換回聲路徑、切換串流時 | 不會失敗。`p == NULL` 或已 destroy 時是 no-op。**會作廢尚未完成的 pre-frame token** |
| 6 | destroy | `four_aec_nr_res_destroy(p)` | 不再處理任何 hop 之後、釋放 pool 之前 | 不會失敗。NULL 安全。pool instance 冪等；`create()` instance 只能呼叫一次 |
| 7 | release | 你自己的釋放器 | **一定要在第 6 步之後** | 順序顛倒 = use-after-free |

heap 便利路徑：`four_aec_nr_res_create(&cfg)` 把第 1~3 步合成一步，
`four_aec_nr_res_destroy()` 會釋放它自己配置的那一塊。其餘完全相同。

### 3.1 執行期唯讀存取

| 呼叫 | 回傳 | NULL / 已 destroy 時 |
|---|---|---|
| `four_aec_nr_res_hop_size(p)` | 每次 pre 要餵幾個 sample（per channel） | `-1` |
| `four_aec_nr_res_fft_size(p)` | FFT 長度 | `-1` |
| `four_aec_nr_res_n_freqs(p)` | 頻域 bin 數 | `-1` |
| `four_aec_nr_res_sample_rate(p)` | 實際生效的取樣率 | `-1` |
| `four_aec_nr_res_matched_filter_count(p)` | 恆為 `1` | `0` |
| `four_aec_nr_res_duty_hops_run(p)` | 共用 estimator 實跑 matched-filter 分析的 hop 數（duty-cycle census 分子；分母用 `estimator_calls`，`duty_hops_total()` 是其別名） | `0` |
| `four_aec_nr_res_linear_aec_count(p)` | 恆為 `4` | `0` |
| `four_aec_nr_res_nr_count(p)` | 恆為 `1` | `0` |
| `four_aec_nr_res_post_res_count(p)` | 恆為 `1` | `0` |
| `four_aec_nr_res_far_fft_real_compute_count(p)` | 累計計數（`long`），每次 `process_pre()` **+1**（不是 +4）；`reset()` 會歸零 | `0` |

後面五個是**結構稽核用**的：前四個對有效 handle 永遠回傳固定常數（1 / 4 / 1 / 1），
用來在整合測試裡證明拓撲沒有被改掉；最後一個用來證明跨聲道的 far-end 共用真的生效。
它們對處理結果沒有任何影響。

### 3.1b 逐階段耗時（診斷用）

```c
typedef struct FourAecNrResLastTiming {
    uint32_t delay_us, frontend_us, linear_us, lane_res_us;   /* process_pre  */
    uint32_t fuse_us,  res_us,      nr_us,     synth_us;      /* process_post */
} FourAecNrResLastTiming;

void four_aec_nr_res_get_last_timing(const FourAecNrRes* p,
                                     FourAecNrResLastTiming* out);
```

最近一個 hop 每個階段的 wall-clock 成本（微秒，`CLOCK_MONOTONIC`）。純診斷：
pipeline 自己不回讀，也不影響處理。`p` 為 NULL 或已 destroy 時把 `out` 清零而
不是失敗；`out` 不得為 NULL。

**八個欄位分成兩個來源。** 本層自己量的是 `fuse_us`、`res_us`（功率準備 ＋
抑制增益）、`nr_us`、`synth_us`（逆轉換 ＋ 加窗 overlap-add ＋ hop 送出），以及
`delay_us` 的共用延遲估計部分（不含 `align_render()` 的 ring-buffer 複製與
realign 掃描）。

`frontend_us`、`linear_us`、`lane_res_us` 來自 `aec_get_last_timing()`
（`aec.h`）：四條 lane 各讀一次再相加，所以得到的是該階段在整個 lane 迴圈上的
wall-clock 成本。lane 內部的 `delay_us` 也一併加進上面的 `delay_us`——它是同一
個量（對齊遠端所花的時間），且**結構上恆為 0**，因為四條 lane 都是
`AEC_DELAY_EXTERNAL_ALIGNED`，沒有自己的 ring 也沒有估計器；相加而不是丟棄，是
為了讓日後改變 lane 模式時不會靜默漏掉那筆成本。

這三欄由**另一半旗標** `-DAEC_STAGE_TIMING=1` 管轄。只建本層那半，這三欄讀 0
而其餘照常量測，是可讀的狀態而非壞掉；`make PROFILE=1` 兩半一起開。

各階段**加起來不等於**整個呼叫的時間，呼叫端要自己用減法補餘額：pre 側餘額涵蓋
輸入去交錯／有限值檢查、`align_render()` 與 realign 掃描；post 側餘額涵蓋 gain
融合、near-floor 閘與 comfort-noise 迴圈。

兩半各自在自己的呼叫被**接受之後**清零，所以中途 bail-out 的 hop 會回報 0 而不是
上一個 hop 的數字；被引數／token 檢查直接拒絕的呼叫則整份記錄不動。在
`process_post*()` 回傳後讀才是完整的。

解析度是微秒，所以低於 1 µs 的階段會讀到 0。

**目標平台沒有 `CLOCK_MONOTONIC` 時**：`clock_gettime` 是 POSIX 而非 C99。用
`make PROFILE=1 EXTRA_CFLAGS='-DFOUR_AEC_NR_RES_NOW_US=board_timer_us
-DAEC_NOW_US=board_timer_us -include my_timer.h'` 換掉時鐘；巨集要是**純識別字**
（Makefile 的 FP policy 不允許 `EXTRA_CFLAGS` 出現括號），指向不收參數、回傳
`uint32_t` 微秒的函式。替代時鐘必須**單調**，否則無號減法會產生接近 32-bit 全距
的荒謬值。本層與 `lib/aec` 各有自己的覆寫點，刻意不共用——一個元件的時鐘屬於該
元件自己的建置契約。開著旗標但指向常數函式是合法用法，所有欄位讀 0。

### 3.2 執行期強度控制

```c
int four_aec_nr_res_set_aec_preset(FourAecNrRes* p, AecPreset preset, float ramp_ms);
int four_aec_nr_res_set_nr_mode(FourAecNrRes* p, MmseLsaNrMode mode);
int four_aec_nr_res_post_split_floor(const FourAecNrRes* p, float* live, float* target);
```

> **本節最重要的一件事：對四條 lane 重新指定 preset 是無效操作。**
>
> 四條 AEC lane 都以 `spatial_linear_context` 建立，因此**沒有任何一條**會走到
> 自己的 suppression-gain 路徑；它們各自的地板什麼都不塑形。真正乘上本核心輸出、
> 也真正決定 comfort noise 量的那個 gain，來自**唯一一個**共用的 post 級抑制器。
> `four_aec_nr_res_set_aec_preset()` 針對的就是它——這也是它存在、而不是要你自己
> 對四條 lane 迴圈呼叫 `aec_set_preset()` 的原因。這一點由測試釘住
> （`tests/test_4aec_nr_res.c` 的 `test_runtime_strength()`），不是推測。

```c
/* 使用者把回音抑制轉到 aggressive、降噪轉到 mild */
if (four_aec_nr_res_set_aec_preset(p, AEC_PRESET_AGGRESSIVE, 100.0f) != 0) { /* 引數不合法 */ }
if (four_aec_nr_res_set_nr_mode(p, MMSE_LSA_NR_MILD) != 0)                  { /* 同上 */ }

/* 想確認要求有沒有落地，就讀這一對（單位是線性功率，不是 dB） */
float live, target;
if (four_aec_nr_res_post_split_floor(p, &live, &target) == 0 && live == target) {
    /* ramp 已經走完 */
}
```

| 呼叫 | 改／讀什麼 | 回 `-1` 的情況（`-1` 時**什麼都不寫**） |
|---|---|---|
| `four_aec_nr_res_set_aec_preset()` | **共用 post 級抑制器**的 far-active split floor | `p` 為 NULL 或已 destroy；`preset` 超出 enum；`ramp_ms` 非有限值或超出 `[0, 60000]`；`cfg.enable_post == 0`（pre-only 核心根本沒有抑制器） |
| `four_aec_nr_res_set_nr_mode()` | 本核心擁有的**那一個**降噪器（NR 是對 beamform 後的訊號跑，不是每條 lane 一個） | `p` 為 NULL 或已 destroy；`cfg.enable_post == 0`；`mode` 超出 enum；重組出的 target 被拒 |
| `four_aec_nr_res_post_split_floor()` | 唯讀。`live` = 抑制器**當下**套用的值，`target` = 撐得過 reset 的已設定值；兩者只有在 ramp 走到一半時才不同。任一指標可為 NULL | `p` 為 NULL 或已 destroy；`cfg.enable_post == 0` |

`live` / `target` 這一對是**板端唯一值得記錄的強度量**：它是真正塑形輸出的東西
（lane 自己的地板什麼都不塑形），而兩個值合在一起就回答了「我要求的改動落地了沒有」。

`ramp_ms == 0` 代表下一個 hop 就套用，**不是錯誤**，落點與「用該 preset 從頭建一個
新實例」完全相同；`> 0` 則以 dB 為單位線性走過去，上限 60 秒。mild ↔ aggressive 是
18 dB 落差、地板又是硬性 clamp，所以互動式旋鈕應該給一個 ramp。ramp 進行中再呼叫
一次，會從當前的 live 值重新起走。

兩個 setter 都**不是重啟**：四條 lane 的濾波器、共用延遲鎖定、NR 的噪聲底與增益
平滑歷史全部繼續跑。要重啟請用 `four_aec_nr_res_reset()`——注意 preset 的改動
**撐得過 reset**（目標值另存一份，reset 會重新套用），但 ramp 進度會被丟掉。
**在兩個 hop 之間呼叫、與 `process_pre()`／`process_post()` 序列化；非 thread-safe。**

> **不要繞過 `set_nr_mode()` 去呼叫 `mmse_lsa_set_mode()`。** 本核心的 NR 組態是
> 「canonical 強度 preset **加上**自己的覆寫」（`broadband_threshold`、`L`、
> `alpha_decay`）。`mmse_lsa_set_mode()` 組的是裸的 canonical preset，在本核心的
> 實例上會被**拒絕**（它的 `L` 不同）——所以 `four_aec_nr_res_set_nr_mode()` 做的
> 事是重組本核心的完整組態，再交給 `mmse_lsa_reconfigure()`。

**A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
hop 上生效：double-talk 期間本核心強制套用 DT 地板，而 DT 地板在三個 preset 之間
**完全相同**；far-active latch 觸發之前套用的是 far-silent 地板。同一個 gain 還
決定注入的 comfort noise 量（振幅正比於 `sqrt(1 − G_res²)`——地板壓得越深、CNG
反而越多）。因此**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且一個只量
echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制頭上。請在 echo 對齊或
degradation 對齊的條件下比較，並實際試聽。

---

## 4. Config 完整參考

`FourAecNrResConfig` 共 9 個欄位。取得預設值的唯一正確方式：

```c
FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
/* 之後只覆寫你真正要改的欄位 */
```

`four_aec_nr_res_default_config()` **不做驗證**，只是填值。
驗證發生在 `get_mem_requirements()` / `init()` / `init_ex()` / `get_mem_breakdown()`
共用的同一個 reject-first 閘。

### 4.1 逐欄位

| 欄位 | 型別 | `default_config()` 給的值 | 合法值（驗證器實際檢查的） | 你該怎麼設 |
|---|---|---|---|---|
| `sample_rate` | `int` | 你傳給 `default_config()` 的值 | **只有 `16000` 或 `48000`**。`8000` 會被拒絕（這點和單聲道 pipeline 不同） | 設成你音訊實際的取樣率 |
| `fft_size` | `int` | `0` | `0` = 依 rate 取預設（16 kHz → 256，48 kHz → 1024）。明確指定時：16 kHz 接受 `256` 或 `512`；48 kHz 只接受 `1024` | 除非有明確理由要 16 kHz 的 512 grid，否則留 `0` |
| `filter_length` | `int` | `0` | `[0, 4096]`。負數或 > 4096 拒絕 | `0` = 沿用 AEC 依取樣率算出的預設（16 kHz → 832 taps，48 kHz → 3072 taps）。回聲尾巴短、記憶體吃緊時可調小；這是四路 AEC 記憶體最主要的旋鈕 |
| `capture_proxy_channel` | `int` | `0` | `[0, 3]`。其他值拒絕 | 共用 delay matcher 要用哪一支麥克風當輸入。挑訊噪比最穩定、離喇叭關係最典型的那一支 |
| `max_delay_ms` | `float` | `1024.0f` | 必須是有限值，且落在 `[0.0, 4096.0]` | 共用 reference delay line 的容量上限（毫秒）。實際的 ring 大小 = `ceil(max_delay_ms * sample_rate / 1000) + 2 * hop + 1` 個 sample。設得比你系統真實最大 mic↔ref 延遲**略大**即可；設太大只是白白吃記憶體 |
| `aec_preset` | `AecPreset` | `AEC_PRESET_BALANCED`（= 1） | `MILD`(0) / `BALANCED`(1) / `AGGRESSIVE`(2)。列舉以外拒絕（**不會**默默 fallback） | 近端保留優先 → `MILD`；回聲抑制優先 → `AGGRESSIVE` |
| `nr_mode` | `MmseLsaNrMode` | `MMSE_LSA_NR_BALANCED`（= 2） | `MILD`(0) / `MODERATE`(1) / `BALANCED`(2) / `AGGRESSIVE`(3)。列舉以外拒絕 | 降噪強度，四級都可用 |
| `enable_cng` | `int`（bool） | `1` | 只接受 `0` 或 `1` | `1` = 在被抑制的 bin 填舒適噪音 |
| `legacy_amin` | `int`（bool） | `0` | 只接受 `0` 或 `1` | `1` = NR 的 noise prior 不摺入 R²。只用於比對舊行為，新整合請保持 `0` |

### 4.2 Grid（由 `sample_rate` + `fft_size` 唯一決定）

`hop = fft_size / 2`、`n_freqs = fft_size / 2 + 1`。合法組合只有三組：

| `sample_rate` | `fft_size` | `hop` | `n_freqs` | 備註 |
|---:|---:|---:|---:|---|
| 16000 | 256 | 128 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 512 | 256 | 257 | 需明確指定 |
| 48000 | 1024 | 512 | 513 | `fft_size = 0` 的 rate 預設 |

**不要把 hop / n_freqs 寫死。** 用 `four_aec_nr_res_hop_size()` / `four_aec_nr_res_n_freqs()` 查。

### 4.3 Buffer 形狀（每次 `process_pre` / `process_post`）

| Buffer | 誰擁有 | 形狀 | 說明 |
|---|---|---|---|
| `microphones_interleaved` | 你 | `float[hop * 4]`，`[sample * 4 + channel]` | 唯讀，只在該次呼叫期間被讀取。所有元素必須是有限值 |
| `ref` | 你 | `float[hop]` | 唯讀。所有元素必須是有限值 |
| `weights` | 你 | `Complex[4 * n_freqs]`，**channel-major**：`[channel * n_freqs + bin]` | 唯讀 |
| `out` | 你 | `float[hop]` | 由 `process_post()` 完整覆寫 |
| `pre.linear_interleaved` | 函式庫 | `const float[hop * 4]`，`[sample * 4 + channel]` | 見 §4.4 |
| `pre.linear_spectra[ch]` | 函式庫 | `const Complex[n_freqs]` | 見 §4.4 |

**權重約定**：`output[bin] = Σ_channel weights[channel][bin] * input[channel][bin]`，
**不做共軛**。多幀 / 時域型 GSC 請把它「這一幀的有效頻率響應」填進來。

**一條在每個 bin 權重都是 0 的 lane 不會投票。** `process_post()` 除了投影頻譜，還會把四條
lane 的 `filter_converged` / `dt_indicator` / `saturation_level` 縮成三個純量去操縱那個共用的
post 級抑制器（其中 `dt_indicator` 決定它套哪一個地板）。一條在每個 bin 權重都是 0 的 lane 對
融合後的頻譜貢獻恰好為零，所以也不參與這三個縮減——縮減從中性值起算，lane 0 一樣受檢。整組
全零權重會先被 `-1` 擋掉（見 §5.3），所以至少有一條 lane 會通過這個檢查；真正在 beamform 的
權重每條 lane 都有非零 bin，這條規則不會跳過任何東西。

外部 beamformer**不需要**再回傳一份 mono hop —— 合成是在 `process_post()` 內部
以那組權重加權後的頻譜完成的。

### 4.4 `FourAecNrResPreFrame` 各欄位

| 欄位 | 型別 | 內容 |
|---|---|---|
| `token` | `FourAecNrResFrameToken` | 交還給 `process_post()` 的排序權杖。見 §5.1 |
| `delay` | `FourAecNrResDelayState` | 共用 matcher 這一幀的狀態，見下 |
| `hop_size` | `int` | 這一幀的 hop |
| `n_channels` | `int` | 恆為 `FOUR_AEC_NR_RES_CHANNELS`（4） |
| `n_freqs` | `int` | 這一幀的 bin 數 |
| `linear_interleaved` | `const float*` | 四路 linear AEC 的**時域** hop，interleaved。做自己的 sqrt-Hann 分析的時域 beamformer 要吃這個。這是 pipeline 自己的 buffer（有拷貝，不是別名） |
| `linear_spectra[ch]` | `const Complex*` | 四路 linear AEC 的**頻譜**，就是 `process_post()` 稍後會用的同一份。讓外部 beamformer 直接沿用既有的分析轉換，不必重做一次 STFT。**這是別名**，直接指向各 lane 自己的 buffer |

`FourAecNrResDelayState` 欄位：

| 欄位 | 型別 | 意義 |
|---|---|---|
| `delay_samples` | `int` | 目前採用的延遲（sample） |
| `confidence` | `float` | 估計信心值 |
| `solid` | `int` | 估計是否已穩定 |
| `changed` | `int` | 非 0 表示這一幀延遲值改變了 —— **你外部的 STFT / OLA 歷史必須跟著清掉** |
| `estimator_calls` | `uint64_t` | 累計呼叫次數 |
| `estimator_updates` | `int` | 累計更新次數 |

**兩個 pre-frame 指標的生命週期（重要）**：
`linear_interleaved` 與 `linear_spectra[ch]` **只在下列任一事件發生之前有效**：
對應的 `process_post()`、`four_aec_nr_res_reset()`、`four_aec_nr_res_destroy()`。

這是 **API 約定，不是記憶體安全保證**。
`reset()` 不保證那些記憶體被清掉或覆寫 —— lane 的舊頻譜可能就原地留著。
也就是說：你違反這條規則時，程式**不會**壞給你看，只會安靜地讀到未定義內容。
不要用「讀到垃圾」當作偵測手段。

### 4.5 哪些欄位會改變記憶體用量

| 欄位 | 影響 `req.bytes` 嗎 |
|---|---|
| `sample_rate` | 會（改 grid、改 AEC 預設 filter 長度） |
| `fft_size` | 會（改 grid） |
| `filter_length` | 會（四路 AEC 等比縮放） |
| `delay_mode` | 會（僅 `MATCHED` 建共用 estimator；三態差異見 4.6a） |
| `delay_num_filters` | 僅 `MATCHED` 有效，每少一個固定省 5,728 B |
| `fixed_delay_samples` | 僅 `FIXED` 有效，決定共用 delay ring 大小 |
| `max_delay_ms` | 會（僅 `MATCHED` 用於 sizing delay ring） |
| `enable_post` | 會（`0` 時不配置 NR/RES/iFFT，見 4.6b；ULCNet wrapper 用這個省 post 級） |
| `capture_proxy_channel` | 不變 |
| `aec_preset` / `nr_mode` / `enable_cng` / `legacy_amin` | 不變 |

### 4.6 實測記憶體（僅供量級參考，務必自己重查）

以下是本次 checkout（`layout_version=15`）、`BACKEND=kiss`、`SIMD=1`、
`delay_mode=MATCHED`（預設,n=5）、`enable_post=1`（預設）下直接呼叫 API
量到的值。換 backend、換編譯選項、更新 submodule 都會變。本輪 `sizeof(Aec)`
由 5832 變 5848 B，每個 AEC 實例的 pool 依 grid 各長一個常數
（16 kHz/256 +5,664 B、16 kHz/512 +5,120 B、48 kHz +18,464 B），四路即四倍，
所以 `aec_bytes` 與 `req.bytes` 兩欄一起移動；`nr_bytes`／`fft_bytes`／
`wrapper_bytes` 不動，所有差額不變。

| Config | `req.bytes` | `aec_bytes`（四路合計） | `nr_bytes` | `fft_bytes` | `wrapper_bytes` |
|---|---:|---:|---:|---:|---:|
| 16000，預設（256/128，`max_delay_ms=1024`） | 1,136,288 | 881,728 | 122,160 | 8,784 | 123,616 |
| 16000，`fft_size=512` | 1,689,872 | 1,395,840 | 133,472 | 16,976 | 143,584 |
| 16000，`max_delay_ms=100` | 1,077,152 | 881,728 | 122,160 | 8,784 | 64,480 |
| 16000，`filter_length=512` | 1,037,280 | 782,720 | 122,160 | 8,784 | 123,616 |
| 48000，預設（1024/512） | 3,754,768 | 3,028,032 | 374,336 | 33,360 | 319,040 |

`four_aec_nr_res_get_mem_breakdown()` 的 `total_bytes` 與 `get_mem_requirements()` 的
`req.bytes` 在上述每一組都相等；`wrapper_bytes` 已包含控制區塊。四路合計的
`aec_bytes` 除以 4 得單路 220,432 B（@16k/256）——與 `lib/aec` 的
`AEC_DELAY_EXTERNAL_ALIGNED` 單體大小完全相同,因為每路內部本來就是
`EXTERNAL_ALIGNED`（delay 由本層共用估計器提供,不建自己的 estimator/ring）。

#### 4.6a delay_mode 對 16000/預設 grid 的影響（`enable_post=1`，`MATCHED n=5` 為 baseline）

| `delay_mode` | `req.bytes` | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 1,136,288 | — |
| `FIXED`，`fixed_delay_samples=1600`（100 ms） | 1,042,688 | −93,600 |
| `EXTERNAL_ALIGNED` | 1,035,776 | −100,512 |

省下的量比單聲道版本小，因為這裡只省**一份共用**的 estimator/ring（四路
共用一個 aligner），不是四份各自的——與本頁「單一共用 aligner」的結構
一致。

#### 4.6b `enable_post=0`（ULCNet wrapper 用的 pre-only 核心）

| Config | `req.bytes` |
|---|---:|
| 16000，`fft_size=512`，`enable_post=0` | 1,506,448 |

即 [`pipeline_ulcnet_4ch.html`](html/pipeline_ulcnet_4ch.html) 記載的 ULCNet
4ch wrapper私有核心大小；比同格點 `enable_post=1` 少 183,424 B（NR/RES/iFFT
的 `nr_bytes+fft_bytes` 加上一部分 `wrapper_bytes`）。本輪同樣是四路各長
5,120 B（@16 kHz/512；`enable_post=0` 的 wrapper 為 110,608 B，與後端無關），
兩側同幅移動，所以差額維持實測的 183,424 B。
**實際配置一律以 `req.bytes` 為準。**

---

## 5. 錯誤處理

### 5.1 pre/post 交接是本 API 面上**唯一**的有狀態協定

把它當成一台狀態機來實作，不要當成兩個獨立的 function call。

```
                     ┌──────────────────────────────────────────────┐
                     │                                              │
                     ▼                                              │
              ┌─────────────┐                                       │
   init ────► │    IDLE     │                                       │
              │ (無 pending)│                                       │
              └──────┬──────┘                                       │
                     │  process_pre()  → OK                         │
                     ▼                                              │
              ┌─────────────┐                                       │
              │   PENDING   │  持有 1 個有效 token                  │
              │  (剛好 1 幀)│  linear_* 指標在此期間有效            │
              └──┬───┬───┬──┘                                       │
                 │   │   │                                          │
   process_pre() │   │   │ process_post(正確 token, 合法 weights)   │
   → SEQUENCE_   │   │   └──────────► OK，token 消耗 ───────────────┘
     ERROR (-2)  │   │
     狀態不變 ───┘   │ reset()  → 回到 IDLE，token 立刻作廢
                     │           （你必須同時停止讀 linear_* 指標）
                     └──────────► IDLE
```

**規則（全部已實測驗證）**：

| 情境 | 回傳 | 之後的狀態 |
|---|---|---|
| 正常 `pre()` → `post()` | 兩者皆 `OK`(0) | token 被消耗，回到 IDLE |
| PENDING 中再次 `pre()` | `SEQUENCE_ERROR`(-2) | **狀態完全不變**，原本那一幀仍然 pending。你可以繼續用原 token 做 `post()` |
| 同一個 token 重播（`post()` 兩次） | 第二次 `SEQUENCE_ERROR`(-2) | 不變 |
| 把 A instance 的 token 拿去 B instance 的 `post()` | `SEQUENCE_ERROR`(-2) | 不變 |
| PENDING 中呼叫 `reset()`，之後再 `post()` | `SEQUENCE_ERROR`(-2) | reset 已把狀態清乾淨 |
| `post()` 傳入全零 / 非有限的 weights | `INVALID_ARGUMENT`(-1) | **pending 幀保留**，你可以修正 weights 後用同一個 token 重試（已實測重試成功） |
| `pre()` 或 `post()` 內部一致性檢查失敗 | `DSP_ERROR`(-3) | **instance 已被自動 `reset()`**，pending 幀消失，token 作廢 |

**Token 怎麼傳**：`process_post()` 的第二個參數型別是 `const FourAecNrResFrameToken*`，
但比對是**逐欄位按值**做的。你要做的是：把 `pre.token` **原封不動**保存下來
（整個 struct 拷貝即可，32 bytes），然後把指向那份拷貝的指標傳進去。
**不要修改任何欄位。**

`FourAecNrResFrameToken` 有 4 個欄位，`process_post()` 全部都會比對：

| 欄位 | 型別 | 作用 |
|---|---|---|
| `frame_index` | `uint64_t` | 擋重播（同一幀不能 post 兩次） |
| `generation` | `uint64_t` | 擋 `reset()` 之後的舊 token（每次 reset 遞增） |
| `owner_cookie` | `uintptr_t` | 擋跨 instance 使用（同時比對「是否等於這個 instance 的位址」） |
| `instance_epoch` | `uint64_t` | 擋 caller-owned pool 的 ABA：`destroy()` 不釋放你的 pool，你可以把新 instance init 在**完全相同的位址**上，此時前三個欄位可能完全重合。`instance_epoch` 由一個 process 內的單調計數器在建構時蓋章（從 1 開始，永不從 pool bytes 讀回），所以任兩次建構必不相同 |

> 這個 epoch 計數器**不是 atomic**：建構被假設為單執行緒。多執行緒同時建構 instance
> 不在支援範圍。

**同時只能有一幀在飛。** 想要 pipeline 化（beamformer 在另一顆核心上跑、
排隊多幀）在這個 API 上是做不到的。

### 5.2 `four_aec_nr_res_process_pre()` 的完整拒絕清單（依實際檢查順序）

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、`p` 已 destroy、`microphones_interleaved` / `ref` / `out` 任一為 NULL | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | 已有 pending 幀 | `SEQUENCE_ERROR`(-2) | 無 |
| 3 | `microphones_interleaved`（`hop*4` 個）或 `ref`（`hop` 個）含非有限值 | `INVALID_ARGUMENT`(-1) | 無 |
| 4 | 內部 render 對齊失敗 | `DSP_ERROR`(-3) | **instance 已被自動 reset** |
| 5 | 某條 lane 的輸出 context 不完整 | `DSP_ERROR`(-3) | **instance 已被自動 reset** |

注意順序：NULL 檢查在 pending 檢查**之前**。所以在 PENDING 狀態下傳 NULL `out`
會拿到 `-1`（不是 `-2`），且不改變狀態。

`process_pre()` **不會**檢查 buffer 長度 —— `hop*4` 與 `hop` 是你的責任。

### 5.3 `four_aec_nr_res_process_post()` 的完整拒絕清單（依實際檢查順序）

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、`p` 已 destroy、`token` / `weights` / `out` 任一為 NULL | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | token 不匹配（重播 / 跨 instance / 被 `reset()` 作廢 / 根本沒有 pending 幀） | `SEQUENCE_ERROR`(-2) | 無 |
| 3 | `weights` 含非有限值，或 `Σ(|re| + |im|)` ≤ `1e-12`（等同全零） | `INVALID_ARGUMENT`(-1) | **pending 幀保留**，可修正後重試 |
| 4 | 內部融合/一致性檢查失敗 | `DSP_ERROR`(-3) | **instance 已被自動 reset** |
| 5 | 內部 RES/NR 階段失敗 | `DSP_ERROR`(-3) | **instance 已被自動 reset** |

**token 檢查排在 weights 檢查之前**：token 壞掉 + weights 也壞掉 → 你拿到的是 `-2`。

### 5.4 錯誤語意總表

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `four_aec_nr_res_default_config()` | 回傳填好的 struct | 不會失敗，**也不驗證** | 傳入非法 `sample_rate` 也照填，之後才會被拒 |
| `four_aec_nr_res_get_mem_requirements()` | `0`，`*out` 填妥 | `-1` | — |
| `four_aec_nr_res_get_mem_breakdown()` | `0`，`*out` 填妥 | `-1` | — |
| `four_aec_nr_res_init()` | 非 NULL handle | `NULL` | 等同 `init_ex(..., NULL)` |
| `four_aec_nr_res_init_ex()` | 非 NULL handle | `NULL` | 8 項檢查任一不符即 NULL |
| `four_aec_nr_res_create()` | 非 NULL handle | `NULL` | config 無效或配置失敗 |
| `four_aec_nr_res_process_pre()` | `FOUR_AEC_NR_RES_OK`(0) | `-1` / `-2` / `-3` | 見 §5.2 |
| `four_aec_nr_res_process_post()` | `FOUR_AEC_NR_RES_OK`(0) | `-1` / `-2` / `-3` | 見 §5.3 |
| `four_aec_nr_res_reset()` | `void` | 不會失敗 | `p == NULL` 或已 destroy → 靜默 no-op |
| `four_aec_nr_res_destroy()` | `void` | 不會失敗 | 見 §5.6 |
| `*_hop_size` / `*_fft_size` / `*_n_freqs` / `*_sample_rate` | 正值 | `-1` | `p == NULL` 或已 destroy |
| `*_matched_filter_count` / `*_linear_aec_count` / `*_nr_count` / `*_post_res_count` | `1` / `4` / `1` / `1` | `0` | `p == NULL` 或已 destroy |
| `four_aec_nr_res_far_fft_real_compute_count()` | 累計值（`long`） | `0` | `p == NULL` 或已 destroy |

> **注意慣例不一致**：本層的 `get_mem_requirements()` / `get_mem_breakdown()` 用
> **0 = 成功 / −1 = 失敗**；但下層模組的 `aec_get_mem_size()` /
> `mmse_lsa_get_mem_size()` / `fft_get_mem_size()` 是**回傳 0 代表失敗**。
> 另外，形狀 accessor 用 `-1` 表示無效 handle，而結構稽核 accessor 用 `0`。

### 5.5 診斷輸出去向

**這個模組在任何拒絕路徑上都不輸出任何訊息，也不連結任何 stdio 符號。**
（`4aec_nr_res.c` 完全沒有 `printf` 家族的呼叫；已用 `nm` 確認 archive 內無相關符號。）

也就是說：`init_ex()` 回 `NULL` 時你**不會**知道是 8 項裡的哪一項不符。
板端 bring-up 時請自行在呼叫端逐欄位比對 `expected` 與剛查到的 `req`，
把差異印出來。

`NO_STDIO=1` 對這個模組沒有任何影響（本來就沒有 stdio）。

### 5.6 `destroy()` 的兩種語意

| Instance 來源 | `destroy()` 行為 |
|---|---|
| `four_aec_nr_res_init()` / `init_ex()`（caller-owned pool） | **不釋放**你的 pool。NULL 安全、**冪等**，重複呼叫安全（已實測）。之後由你自己交還或重用 pool |
| `four_aec_nr_res_create()`（heap） | 釋放 `create()` 配置的那一塊。遵循一般 `free()` 語意：**只能呼叫一次** |

destroy 之後不要再呼叫任何 accessor。

---

## 6. 出貨內容 vs 範例

`pipelines/4ch_aec_bf_nr_res/` 目錄不是自解釋的。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c` | `libaudio_pipeline_4ch.a`（object 之一） | **本手冊的主體** |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c` | `libaudio_pipeline_4ch.a`（object 之二） | 完整 spatial wrapper。只用本手冊 API 時 linker 不會拉進來 |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.h` | — | 本手冊的公開 API |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h` | — | 另一份手冊的公開 API |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_internal.h`、`4aec_projection_kernels.h` | — | 內部 header，**不是**公開 API，不要 include |

### 6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_static.c` | caller-owned pool 路徑的主機參考程式 | 示範 query → allocate → init → process_pre/post → destroy → release。它在 pre/post 之間塞的是**固定等權重**，那是決定性的 smoke adapter，**不是** production beamformer |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_static.c` | 完整 spatial pipeline 的 pool 路徑主機參考程式 | 屬於另一份手冊；列在這裡是為了讓目錄內容一目了然 |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c` | raw-float 錄音驗證 runner | 主機一次性工具，走 heap `create()` 路徑 |

單聲道的 library 與參考程式（`pipelines/mono_aec_nr_res/main.c`、
`static_main.c`、`example_board_adapter.c`）見 `docs/integration_mono_zh_TW.md`。

### 6.3 其他（不出貨）

* `pipelines/4ch_aec_bf_nr_res/tests/` —— 測試。
* `pipelines/4ch_aec_bf_nr_res/third_party/` —— DOA / GSC / spatial kernel 函式庫。
  **只用本手冊這套 API 的話完全不需要它們。**
* `.py` 檔 —— 主機端評估與參考實作，不是 C 產品的一部分。

---

## 7. 整合檢查清單

建置與連結：

- [ ] 四個 archive（`libaudio_pipeline_4ch.a` / `libaec.a` / `libmmse_lsa.a` /
      `libaudio_common.a`）是用**同一組** backend 與編譯選項建出來的
- [ ] link 順序照 §1.5；沒有多連 `libdoa.a` / `libgsc.a` / `libspatial_common.a`
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用 `print-lib-path` 查的
- [ ] 沒有 include `4aec_nr_res_internal.h` 或 `4aec_projection_kernels.h`

初始化：

- [ ] 每次 init 之前都重新呼叫 `four_aec_nr_res_get_mem_requirements()`
- [ ] pool 對齊 16、大小 ≥ `req.bytes`
- [ ] 用 `init_ex()` 並把剛查到的 `req` 當 `expected` 傳回去
- [ ] `init_ex()` 回 `NULL` 時你的程式會停下來 —— 而且你**自己**在呼叫端印出了
      是哪一個欄位不符（函式庫不會告訴你）
- [ ] `sample_rate` 沒有設成 8000

pre/post 協定：

- [ ] 你的程式結構保證「一個 `pre()` 對應一個 `post()`」，中間不會插入第二個 `pre()`
- [ ] `pre.token` 是整份拷貝保存的，沒有被修改
- [ ] `weights` 是 channel-major `[ch * n_freqs + bin]`，不是 bin-major
- [ ] `weights` 不會全零（會被 `-1` 拒絕）
- [ ] 有處理 `-2`（順序錯）與 `-3`（instance 已被自動 reset，你的外部狀態也要跟著清）
- [ ] `pre.delay.changed` 非 0 時，你外部 beamformer 的 STFT / OLA 歷史有跟著清掉
- [ ] `linear_interleaved` / `linear_spectra` 只在 `post()` / `reset()` / `destroy()`
      之前讀取，沒有跨幀保存指標
- [ ] `reset()` 之後你的 beamformer 立刻停止讀那些指標（不要等 token 被拒才反應）

執行期：

- [ ] `microphones_interleaved` 是 `[sample * 4 + channel]` interleaved，長度 `hop * 4`
- [ ] mic / ref 內容都是有限值（NaN/Inf 會被 `-1` 拒絕）
- [ ] hop / n_freqs 是查來的，不是寫死的
- [ ] 同一個 instance 沒有被多 thread 同時處理

收尾：

- [ ] `destroy()` 在釋放 pool **之前**
- [ ] `create()` 建的 instance 只 `destroy()` 一次
- [ ] destroy 之後沒有再呼叫任何 accessor

---

## 8. 版本與相容性

### 8.1 `FourAecNrResMemReq`

固定 32 bytes、每個欄位固定 byte offset（由 header 內的 `_Static_assert` 釘死），
全部是定寬整數。與單聲道 `AudioPipelineMemReq` 同形狀、同語意，但**版本號各自獨立**。

| Offset | 欄位 | 型別 | 目前值 |
|---:|---|---|---|
| 0 | `descriptor_version` | `uint32_t` | `1` |
| 4 | `layout_version` | `uint32_t` | `15` |
| 8 | `backend_id` | `uint32_t` | `1` = KISS，`2` = NE10（永遠不會是 0） |
| 12 | `build_flags_hash` | `uint32_t` | FNV-1a-32，隨 build 變動 |
| 16 | `alignment` | `uint32_t` | `16` |
| 20 | `reserved` | `uint32_t` | `0`（必須為 0，`init_ex()` 會驗證） |
| 24 | `bytes` | `uint64_t` | pool 總需求 |

### 8.2 `init_ex()` 的 8 項檢查

`expected == NULL` 時完全等同 `four_aec_nr_res_init()`。
`expected != NULL` 時，下列 8 項全部成立才會開始 carve；任一不符即回 `NULL`
（**不輸出任何訊息**）：

1. `descriptor_version` 相符
2. `layout_version` 相符
3. `backend_id` 相符
4. `build_flags_hash` 相符
5. `alignment` 相符
6. `reserved == 0`
7. `expected->bytes >= 目前 build 的需求`
8. `bytes`（實際交進來的 pool 大小）`>= 目前 build 的需求`

### 8.3 序列化

可以逐 byte 拷貝到檔案 / flash / 訊息 buffer 再讀回來，**但僅限相同 endianness**。
不提供 byte-swap 輔助，跨 endianness 交換不在支援範圍。

### 8.4 相容性規則（給整合者的三條）

1. **descriptor 永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用。
2. **兩個 backend 之間永不互通**，即使 `bytes` 相同。
3. **升級 library 之後一定要重跑一次 `get_mem_requirements()`**，並確認 pool 預算仍足夠。
