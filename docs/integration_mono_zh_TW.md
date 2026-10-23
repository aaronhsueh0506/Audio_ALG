# 單聲道整合手冊（`Aec` + 範例鏈，沒有 mono 函式庫）

適用對象：要把單聲道 AEC → NR → RES 連進自己產品的**函式庫使用者**。

本文只講「什麼時候呼叫哪個 function」「每個參數怎麼設」。演算法內部推導不在本文範圍。

> **路徑慣例**：本文所有路徑都相對於 Audio_ALG repository 根目錄。
> `audiolib` 是與 Audio_ALG 平行的另一個 checkout，本文寫成 `<audiolib>/`；
> 從 `pipelines/` 出發的相對路徑是 `../../audiolib/common`。
>
> **數值來源**：本文所有數字都是從當前 source 讀出，或以當前 checkout 實際建置後量測。
> 記憶體數字**必須**用 API 重新查詢，不要抄本文（見 §4.5）。

---

## 1. 你連結什麼、你 `#include` 什麼

### 1.1 構成：每個函式庫各自初始化，鏈由你接

mono 沒有 unit、也沒有 pipeline 函式庫。處理鏈由下列獨立的函式庫物件組成，**每一個都有自己的
`mem_size` 與 `init`**，初始化與串接是整合者的事：

```
mic hop + ref hop ──► Aec (aec_process_context) ──► se_frame_from_aec ──► SeSpecFrame
                                                                              │
        ┌─────────────────────────────────────────────────────────────────────┘
        ▼   AecRes（可選，mono 用 AEC 自己的 G_res）· AnrDenoiser · AcIstft
   se_sample_chain_form ─► se_sample_chain_synthesize ──► out hop
```

| 物件 | 來源 | 大小查詢 | 初始化 |
|---|---|---|---|
| `Aec`（線性 AEC，含 AEC3 殘留 gain 的 context） | `lib/aec`（`libaec.a`） | `aec_get_mem_size(&aec_cfg)` | `aec_init(mem, bytes, &aec_cfg)` |
| `AnrDenoiser` | `lib/nr`（`libanr.a`） | `anr_get_mem_size(&nr_cfg)`（`nr_cfg = anr_config_for_mode_grid(sr, fft_size, mode)`） | `anr_init()` |
| `AcIstft`（合成） | audiolib（`libaudio_common.a`） | `ac_istft_get_mem_size()` | `ac_istft_init()` |
| 範例鏈 `SeSampleChain`（把上面三者與增益向量放進**一塊** pool） | `pipelines/samples/se_sample_chain.h`（header-only） | `se_sample_chain_mem_size(&sc)` | `se_sample_chain_init(mem, bytes, &sc)` |
| 格點與 `AecConfig` 推導 | `pipelines/common/se_config.{c,h}`（`se_derive()`） | – | – |

範例鏈是**範例碼**：沒有 ABI、沒有 layout version、沒有 descriptor、不進任何 archive，可整份複製改寫。
你也可以只用 `Aec`，自己用 `aec_res.h`、`anr.h`、`audio_stft.h` 逐個初始化與串接；
`mono_aec_nr_res/main.c` 與 `static_main.c` 是現成的兩種寫法。

### 1.2 `--aec-only`

只要線性 AEC 的輸出，就不建鏈：`aec_process(aec, mic, ref, out)` 直接輸出時域 hop，
不需要 `SeSampleChain`（見 `static_main.c --print-mem-size --aec-only`：pool 只有 `Aec` 那一塊）。
`aec_process()` 與 `aec_process_context()` 推進相同的狀態，同一個實例上可以混用。

### 1.3 Header closure

```c
#include "aec.h"                       /* Aec, AecConfig, aec_process_context()               */
#include "common/se_config.h"          /* se_derive(): 格點 + AecConfig                         */
#include "common/se_stage.h"           /* SeSpecFrame, se_frame_from_aec(), se_res_stage_gain() */
#include "samples/se_sample_chain.h"   /* 範例鏈（會拉進 anr.h、aec_res.h、audio_stft.h）          */
#include "utility/mem_align.h"         /* ALIGN16                                             */
```

`Complex` 定義在 `<audiolib>/common/include/fft_wrapper.h`，已透過 `se_stage.h` 帶進來。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I pipelines
-I lib/aec/c_impl/include
-I lib/nr/c_impl/include
-I <audiolib>/common/include
-I <audiolib>/common
```

`pipelines/Makefile` 自己建置範例時的 `-I` 清單比這個長（多了 `lib/aec/c_impl/example`），
那是給 CLI 的 WAV I/O 用的，consumer 不需要。語言標準：以 `-std=gnu99` 建置；
NE10 後端的 header 需要 GNU 擴充，所以你的 TU 也請用 `gnu99`（或以上的 GNU 方言）。

### 1.5 Archive link order

link 順序有意義（單向相依，由上而下）：

```
你的 .o
se_config.o                 （pipelines/common/se_config.c 編出來的 object；你自己編或取 print-obj-dir 下的）
libaec.a
libanr.a
libaudio_common.a
-lm
```

* mono 沒有函式庫 TU。`se_config.o` 是唯一需要 pipelines 提供的 object，
  鏈本身是 header-only。
* 三個 archive 的實際路徑是 config-keyed 的（`bin/<backend>-<hash>/`），不是固定名稱。
  用各自 Makefile 的 `make -s print-lib-path` 取得，不要硬寫路徑。
* `BACKEND=ne10` 時 audiolib 內含一個 C++ TU，最終 link 要用 C++ driver（`c++`），不是 `cc`。
* 各 archive 必須用**同一組** backend / 編譯選項建置。

---

## 2. Quick start

以下是完整、可編譯、已實際跑過的最小整合（用 `posix_memalign` 當 pool 的主機版）：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "common/se_config.h"
#include "common/se_stage.h"
#include "samples/se_sample_chain.h"
#include "utility/mem_align.h"

int main(void) {
    static const SeRules rules = { .allow_8k = 1, .spatial_context = SE_CONTEXT_PRESET };
    SeRequest req;
    SeGrid grid;
    AecConfig aec_cfg;
    memset(&req, 0, sizeof(req));
    req.sample_rate = 16000;
    req.delay_mode = AEC_DELAY_MATCHED;
    req.delay_num_filters = 5;
    req.fixed_delay_samples = -1;
    req.aec_preset = AEC_PRESET_BALANCED;
    if (se_derive(&req, &rules, &grid, &aec_cfg) != 0) return 1;      /* grid + AecConfig */

    SeSampleChainConfig sc = se_sample_chain_default_config(16000, grid.fft_size);
    sc.res = SE_SAMPLE_RES_FRAME_GAIN;                                /* the AEC's own G_res */

    size_t aec_bytes = aec_get_mem_size(&aec_cfg);                    /* 0 = invalid */
    size_t chain_bytes = se_sample_chain_mem_size(&sc);               /* 0 = invalid */
    if (aec_bytes == 0 || chain_bytes == 0) return 1;
    void* pool = NULL;
    if (posix_memalign(&pool, 16, ALIGN16(aec_bytes) + chain_bytes) != 0) return 1;

    Aec* aec = aec_init(pool, aec_bytes, &aec_cfg);
    SeSampleChain* chain =
        se_sample_chain_init((uint8_t*)pool + ALIGN16(aec_bytes), chain_bytes, &sc);
    if (!aec || !chain) return 1;

    float mic[512], ref[512], out[512];
    for (int frame = 0; frame < 100; frame++) {
        SeSpecFrame f;
        const Complex* p;
        for (int i = 0; i < grid.hop_size; i++) {            /* your capture source */
            mic[i] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }
        aec_process_context(aec, mic, ref);                   /* linear AEC, no time-domain emit */
        if (se_frame_from_aec(aec, &f) != 0 ||
            !(p = se_sample_chain_form(chain, &f)) ||         /* RES gain, NR gain, min, CNG */
            se_sample_chain_synthesize(chain, p, out) != 0 ||
            !se_hop_is_finite(out, grid.hop_size)) {
            aec_reset(aec);                                   /* a refused hop: reset both */
            se_sample_chain_reset(chain);
            continue;
        }
        /* consume out: grid.hop_size floats */
    }

    se_sample_chain_destroy(chain);                           /* releases the denoiser, not the pool */
    aec_destroy(aec);
    free(pool);
    printf("ok hop=%d\n", grid.hop_size);
    return 0;
}
```

編譯（`<...>` 換成你 `print-lib-path` 查到的路徑；`se_config.c` 用同一組 `-I`）：

```sh
cc -std=gnu99 -O2 -Wall -Wextra -c app.c -o app.o \
  -I pipelines -I lib/aec/c_impl/include -I lib/nr/c_impl/include \
  -I <audiolib>/common/include -I <audiolib>/common
cc -std=gnu99 -O2 -c pipelines/common/se_config.c -o se_config.o \
  -I pipelines -I lib/aec/c_impl/include -I lib/nr/c_impl/include \
  -I <audiolib>/common/include -I <audiolib>/common
c++ -o app app.o se_config.o \
  <aec-bin>/libaec.a \
  <nr-bin>/libanr.a \
  <audiolib-bin>/libaudio_common.a \
  -lm
```

> `float mic[512]` 是因為本例已知 hop 最大為 512（48 kHz）。
> 產品程式請依 `grid.hop_size` 配置，不要寫死。

---

## 3. Lifecycle

板端（caller-owned pool）的完整順序如下。每個函式庫各自查大小、各自 init，整合者負責加總與切塊。

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | derive | `se_derive(&req, &rules, &grid, &aec_cfg)` | 每次 init 之前 | 回傳 `-1`：任何一條規則被違反（§4.1）；輸出內容未定義 |
| 2 | query | `aec_get_mem_size(&aec_cfg)`、`se_sample_chain_mem_size(&sc)` | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | 回傳 `0`：config 無效（**注意：這兩個是 0 = 失敗，與 4ch 單元的 `-1` 慣例不同**） |
| 3 | allocate | 你自己的配置器，取得 `ALIGN16(aec_bytes) + chain_bytes`（再加你自己的 stage）且 16-byte 對齊的記憶體 | 拿到大小之後 | 配置失敗由你自己處理。pool **不需要**預先清零（poison pattern 也可以）；但必須在 handle 生命週期內獨佔、不被其他人寫入 |
| 4 | init | `aec_init(pool, aec_bytes, &aec_cfg)`、`se_sample_chain_init(pool + ALIGN16(aec_bytes), chain_bytes, &sc)` | 拿到 pool 之後，處理第一個 hop 之前 | 回傳 `NULL`：pool 未 16-byte 對齊、太小、config 無效、子模組 init 失敗。已建好的那一個要由你 destroy |
| 5 | process | `aec_process_context(aec, mic, ref)` → `se_frame_from_aec(aec, &f)` → `se_sample_chain_form(chain, &f)` → `se_sample_chain_synthesize(chain, P, out)` | 每收到一個 hop 呼叫一次。`mic`/`ref`/`out` 都必須剛好 `grid.hop_size` 個 float | `aec_process_context()` 回傳 `void`；`se_frame_from_aec()` 回 `-1`（NULL 引數或 lane 沒有 seam）；`form()` 回 `NULL`（NULL 引數、另一個格點的 frame、或某個 stage 拒絕該 hop）；`synthesize()` 回 `-1`。**不會**檢查 buffer 長度 —— 長度錯是你的責任 |
| 6 | recover | `aec_reset(aec)` **加** `se_sample_chain_reset(chain)` | 任何一步被拒、輸出非有限、換回聲路徑（換喇叭、重新 seat）或切換到不相關的另一條串流時 | 不會失敗。兩個必須一起 reset：鏈的 reset 也把 comfort-noise RNG 設回 `SE_CNG_RNG_SEED`，漏掉其中一個就不等於新建的實例 |
| 7 | destroy | `se_sample_chain_destroy(chain)`、`aec_destroy(aec)` | 不再處理任何 hop 之後、釋放 pool 之前 | 不會失敗。`aec_destroy` NULL 安全、可重複；`se_sample_chain_destroy` 只釋放降噪器的資源 |
| 8 | release | 你自己的釋放器 | **一定要在第 7 步之後** | 順序顛倒 = use-after-free |

補充規則：

* **大小不可快取**。firmware image 若拿舊 build 算出的大小去餵新 library，就會在錯誤大小/佈局的 pool 上切記憶體。
  mono 沒有 descriptor 閘（那是 4ch 單元的 `AudioPipeline4Ch` 才有的契約，見
  [`integration_4ch_spatial_zh_TW.md`](integration_4ch_spatial_zh_TW.md)），所以每次 build 都重查大小是你的責任；
  `aec_init()` 與 `se_sample_chain_init()` 只會擋下過小或未對齊的 pool。
* pool 生命週期內 sub-module 都是指進 pool 的裸指標，不是拷貝。pool 不能被搬動、
  不能被別人寫、不能和另一個 `Aec`/NR/FFT instance 共用。
* 單一 instance 不可被多個 thread 同時 `process`。多條串流請各自建立 instance。
* `SeSpecFrame` 的指標都借自 `Aec`，到下一次 `aec_process_context()` 或 reset 為止有效；`P` 必須在同一個 hop 內形成並合成。

### 3.1 執行期唯讀存取

| 呼叫 | 用途 |
|---|---|
| `grid.hop_size` / `grid.n_freqs` / `grid.fft_size`（`se_derive()` 輸出） | 每次 process 要餵幾個 float、頻域 bin 數 |
| `se_sample_chain_res(chain)` | `AecRes*`（只有 `res = SE_SAMPLE_RES_AEC_RES` 時非 NULL；mono 範例用 `FRAME_GAIN`，回 NULL） |
| `se_sample_chain_nr(chain)` | 範例鏈那一個 `AnrDenoiser*`（`enable_nr = 0` 時 NULL），只用來做你自己的診斷讀取（`anr_debug_status()`）；**不要**直接 reset 或 destroy，一律走 `se_sample_chain_reset()` / `_destroy()` |
| `se_sample_chain_last_res_gain(chain)` / `_last_nr_gain(chain)` | 上一個 `form()` 用的 RES 增益（min 之前）與 NR 增益，借用到下一次呼叫 |
| `aec_debug_status(aec, &st)` | 診斷：delay、ERLE、收斂旗標（mono CLI 的 `--debug` 就是這個加 `anr_debug_status()`） |

### 3.1b 逐階段耗時（診斷用）

`aec_get_last_timing(aec, &t)` 回報 **AEC 上一個 hop** 的各階段牆鐘耗時（微秒），純診斷：

```c
AecStageTiming t;
aec_process_context(aec, mic, ref);
aec_get_last_timing(aec, &t);
/* t.delay_us t.frontend_us t.linear_us t.res_us ← lib/aec 量 */
```

鏈與合成的耗時是**範例程式自己**在呼叫前後讀時鐘量的（`mono_aec_nr_res/main.c` 的 `process_hop()`
與 `timing_now_us()`），不是任何函式庫的 API。

| 欄位 | 涵蓋範圍 |
|---|---|
| `delay_us` | **延遲估計與 ring 對齊**：matched-filter bank、其 duty-cycle 記帳、reference ring 寫入與延遲補償讀取。`EXTERNAL_ALIGNED` 下為 0，`FIXED` 下很小 |
| `frontend_us` | AEC 到主濾波器之前的**其餘**部分：mic HPF、飽和、render activity、`mu_scale`、mic-clip、RSA、shadow filter。**不含 `delay_us`**，兩者不重複計算 |
| `linear_us` | AEC 主自適應濾波器（`pbfdkf_process`），含 far-end FFT |
| `res_us` | AEC 自己的 AEC3 post/RES 區塊 |

**實測：`delay_us` 通常是整個 hop 最大的一塊。** 16 kHz/256 ne10 上量到
60.9%，`frontend_us`（mic HPF/RSA/shadow filter）只有 6.8%。這條線直接對應
`delay_num_filters`（1..5）這個可調旋鈕。

**各階段不會加總成整通呼叫。** 餘數包含引數檢查、`aec_get_res_context()`，以及
AEC 內部（因此從這層看不見）的 stationarity 更新、`e2_coarse`/ERL 發布、DT
analyzer、EPV、shadow_rise、失調估計、功率 EMA 與收斂偵測。要呈現完整分解的
呼叫端必須把餘數明確列出來，不能讓部分看起來像全部。

**預設關閉。** AEC 內部 7 次 clock 讀取，release build 一次都不做，欄位讀 0。
`make PROFILE=1` 會開啟（`-DAEC_STAGE_TIMING=1`）。呼叫端自己的顯示旗標（`--timing`）**不會**打開 AEC 的量測。

**記錄本身不是條件編譯的，只有取樣是。** 欄位無論旗標開關都在 struct 裡，所以
profile build 與 release build 切出來的 pool 逐位元相同，開旗標永遠不會移動任何 offset。

**目標平台沒有 `CLOCK_MONOTONIC` 時。** `clock_gettime` 是 POSIX 而非 C99。這種
平台用 `make PROFILE=1 EXTRA_CFLAGS='-DAEC_NOW_US=board_timer_us -include my_timer.h'`
換掉 AEC 的時鐘：巨集要是**純識別字**（Makefile 的 FP policy 不允許 `EXTRA_CFLAGS` 出現括號），
指向不收參數、回傳 `uint32_t` 微秒的函式。替代時鐘必須單調，否則無號減法會產生接近 32-bit 全距的
荒謬值。每個元件各有自己的覆寫點（`lib/aec`、四路核心），刻意不共用——一個元件的時鐘屬於該元件自己的建置契約。
開著旗標但指向常數函式是合法用法，所有欄位讀 0，`--timing` 的報表無法分辨這種情況與「根本沒開旗標」。

兩支 CLI 都有 `--timing`，在結束時印出整段 run 的每 hop 平均與佔比（含未歸屬餘數）。

### 3.2 執行期強度控制

兩條強度軸都可以在**運轉中**改，不需要重建、不動 pool：

```c
int aec_set_preset(Aec* a, AecPreset preset, float ramp_ms);
int anr_set_mode(AnrDenoiser* nr, AnrMode mode);   /* nr = se_sample_chain_nr(chain) */
```

```c
/* 使用者把回音抑制轉到 aggressive、降噪轉到 mild */
if (aec_set_preset(aec, AEC_PRESET_AGGRESSIVE, 100.0f) != 0)   { /* 引數不合法 */ }
if (anr_set_mode(se_sample_chain_nr(chain), ANR_MODE_MILD) != 0) { /* 同上 */ }

aec_process_context(aec, mic, ref);   /* 下一個 hop 起生效 */
```

| 呼叫 | 改什麼 | 回 `-1` 的情況（`-1` 時**什麼都不寫**） |
|---|---|---|
| `aec_set_preset()` | `Aec` 的 far-active 殘留回音地板（三個 preset 唯一的差異軸；mono 的 RES 增益 `g_res_ctx` 就是這個 lane 算的） | `a` 為 NULL；`preset` 超出 enum；`ramp_ms` 非有限值或超出 `[0, 60000]` |
| `anr_set_mode()` | 範例鏈降噪器的強度 preset（`q`、`xi_min_db`、`g_min_db`、`noise_over_subtraction`） | `nr` 為 NULL（含鏈沒有降噪器、`enable_nr = 0`）；`mode` 超出 enum |

`ramp_ms == 0` 代表下一個 hop 就套用，**不是錯誤**，落點與「用該 preset 從頭建一個
新實例」完全相同；`> 0` 則以 dB 為單位線性走過去，上限 60 秒。mild ↔ aggressive 是
18 dB 落差、而地板是硬性 clamp，所以互動式旋鈕應該給一個 ramp（100 ms 是合理起點）。
ramp 進行中再呼叫一次，會從當前的 live 值重新起走。

兩個 setter 都**不是重啟**：AEC 的濾波器與延遲鎖定、NR 追蹤中的噪聲底與增益平滑歷史
全部繼續跑。要重啟請用 `aec_reset()` 加 `se_sample_chain_reset()`。**在兩個 hop 之間呼叫、與
process 序列化；非 thread-safe。**

> 範例鏈的降噪器以 `anr_config_for_mode_grid()` 建立、沒有覆寫任何欄位，所以 `anr_set_mode()`
> 直接換強度。若你在 preset 之上改過欄位，`anr_set_mode()` 會把它們蓋回 preset 值：
> 那種實例請自己組好完整 target 再呼叫 `anr_reconfigure()`。

**A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
hop 上生效：double-talk 期間套用的是 DT 地板，而 DT 地板在三個 preset 之間**完全
相同**；far-active latch 觸發之前套用的是 far-silent 地板。同一個 `ctx.res_gain`
還決定注入的 comfort noise 量（振幅正比於 `sqrt(1 − G_res²)`，再乘該 bin 的 NR gain（下限 −10 dB），見
`samples/se_sample_chain.h` 的 `se_sample_chain_form()` 的 CNG 步驟——地板壓得越深、CNG 反而越多）。
因此**整段錄音的平均值移動
幅度會小於 dB 落差所暗示的量**，而且一個只量 echo／degradation 的 A/B 會把 CNG 的
變化錯記到別的機制頭上。請在 echo 對齊或 degradation 對齊的條件下比較，並實際試聽。

---

## 4. Config 完整參考

mono 的設定分成兩份：`SeRequest`（交給 `se_derive()`，決定格點與 `AecConfig`）與
`SeSampleChainConfig`（交給鏈的 `mem_size`/`init`，決定 RES／NR／CNG／合成）。
`se_derive()` 與 `se_sample_chain_default_config()` 都**不做**完整驗證，驗證發生在
`se_derive()` 本身（回 `-1`）與 `aec_get_mem_size()`／`se_sample_chain_mem_size()`（回 `0`）。

### 4.1 `SeRequest` → `se_derive()`

| 欄位 | 型別 | 範例的值 | 合法值（驗證器實際檢查的） | 你該怎麼設 |
|---|---|---|---|---|
| `sample_rate` | `int` | 你的取樣率 | `8000`（需 `rules.allow_8k`）/ `16000` / `48000`。其他一律拒絕（例如 `44100`） | 設成你音訊實際的取樣率。這個值同時決定預設 grid |
| `fft_size` | `int` | `0` | `0` = 依 rate 取預設。明確指定時：8 kHz 只接受 `256`；16 kHz 接受 `256` 或 `512`；48 kHz 只接受 `1024`。其他組合拒絕 | 除非你有明確理由要 16 kHz 的 512 grid，否則留 `0` |
| `filter_length` | `int` | `0` | `0` = 依 preset／rate 取預設；否則 `[1, SE_FILTER_LENGTH_MAX(4096)]` | PBFDKF taps，init 後不可變；沒有明確需求就留 `0` |
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `AEC_DELAY_MATCHED` / `AEC_DELAY_FIXED` / `AEC_DELAY_EXTERNAL_ALIGNED` | 語意同 `lib/aec`；記憶體差異見 §4.5a |
| `delay_num_filters` | `int` | `5` | 僅 `MATCHED` 有效：`[1, 5]`；其他模式要求維持 `5`（`rules.check_delay = 1` 時檢查） | matched-filter bank 大小，決定 bulk delay 搜尋範圍；init 參數，不是 runtime setter |
| `fixed_delay_samples` | `int` | `-1` | 僅 `FIXED` 有效（原生取樣點數，上限為該取樣率下的兩分鐘）；其他模式須為 `-1`（`rules.check_delay = 1` 時檢查） | 已知固定 delay 時使用 |
| `aec_preset` | `AecPreset` | `AEC_PRESET_BALANCED`（= 1） | `AEC_PRESET_MILD`(0) / `AEC_PRESET_BALANCED`(1) / `AEC_PRESET_AGGRESSIVE`(2) / `AEC_PRESET_NO_FLOOR`(3)。列舉以外的整數會被拒絕（**不會**默默 fallback 成 balanced） | 近端保留優先 → `MILD`；回聲抑制優先 → `AGGRESSIVE`；不確定就留預設 |

`SeRules` 是你的應用自己的規則：`allow_8k`、`check_delay`（是否拒絕與 `delay_mode` 矛盾的 delay 設定；
mono 範例沒開）、`spatial_context`（mono 用 `SE_CONTEXT_PRESET`：lane 保留 preset 自己的 context 層級，
所以 `g_res_ctx` 有值）。`se_derive()` 輸出的 `AecConfig` 已設好 `enable_res = 0`、`return_res_context = 1`：
lane 不自己輸出時域音訊，增益由鏈套用。

### 4.2 `SeSampleChainConfig` → 鏈

| 欄位 | 型別 | `se_sample_chain_default_config()` 給的值 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `sample_rate`、`fft_size` | `int` | 你傳進去的值 | 必須是合法格點（與 `se_derive()` 同一組） | 用 `grid.fft_size`，不要寫 0 |
| `res` | `SeSampleRes` | `SE_SAMPLE_RES_AEC_RES` | `OFF` / `FRAME_GAIN` / `AEC_RES` | **mono 設 `SE_SAMPLE_RES_FRAME_GAIN`**（取 `frame->g_res_ctx`，即 AEC 自己的殘留 gain）；`OFF` = 不套 RES（`G_res` 視為 1），comfort noise 因為沒有被 RES 挖掉的 bin 而不填；`AEC_RES` 是 4ch 用的獨立 `AecRes`（需要 `r2`、`cn`、`far_hop` 等輸入） |
| `res_preset` | `AecPreset` | `AEC_PRESET_BALANCED` | 僅 `AEC_RES` 有意義 | mono 用 `aec_set_preset()` 調強度，這個欄位不用 |
| `enable_nr` | `int`（bool） | `1` | 只接受 `0` 或 `1` | `0` = 不建 MMSE-LSA（state 不進 pool、`se_sample_chain_nr()` 回 `NULL`），`g_total` 只剩 `G_res` |
| `nr_mode` | `AnrMode` | `ANR_MODE_BALANCED`（= 2） | `ANR_MODE_MILD`(0) / `MODERATE`(1) / `BALANCED`(2) / `AGGRESSIVE`(3)。`enable_nr = 1` 時檢查 | 降噪強度。四級都可用 |
| `legacy_amin` | `int`（bool） | `0` | 只接受 `0` 或 `1` | `1` = noise-only NR：noise floor 不摺入 R²。只用於比對，新整合請保持 `0` |
| `enable_cng` | `int`（bool） | `1` | 只接受 `0` 或 `1` | `1` = 在 RES 抑制掉的 bin 填舒適噪音，位準再乘該 bin 的 NR gain、下限 −10 dB（NR 看不到 comfort noise，在此補套）。mono 範例把它與 `aec_cfg.enable_cng`（AEC preset 自己的值）取 AND |
| `with_synthesis` | `int`（bool） | `1` | 只接受 `0` 或 `1` | `0` = 鏈停在頻譜 `P`（DFN2 等後接的 stage 自己合成） |

CLI 旗標與欄位的對應：`--aec-only` = 不建鏈、用 `aec_process()`；`--no-nr` = `enable_nr = 0`；
`--no-res` = `res = OFF`；`--no-cng` = `enable_cng = 0`；`--legacy-amin` = `legacy_amin = 1`。

### 4.3 Grid（由 `sample_rate` + `fft_size` 唯一決定）

`hop = fft_size / 2`、`frame = fft_size`、`n_freqs = fft_size / 2 + 1`。
合法組合只有這四組：

| `sample_rate` | `fft_size` | `hop` | `frame` | `n_freqs` | 備註 |
|---:|---:|---:|---:|---:|---|
| 8000 | 256 | 128 | 256 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 256 | 128 | 256 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 512 | 256 | 512 | 257 | 需明確指定 |
| 48000 | 1024 | 512 | 1024 | 513 | `fft_size = 0` 的 rate 預設 |

**不要把 hop 寫死。** 16 kHz 有兩組合法 grid，寫死任一組對另一組就是錯的。
一律用 `se_derive()` 輸出的 `grid.hop_size` / `grid.n_freqs` 查。

### 4.4 哪些欄位會改變記憶體用量

| 欄位 | 影響 pool 大小嗎 |
|---|---|
| `sample_rate`、`fft_size` | 會（改 grid、改 AEC filter 長度） |
| `filter_length` | 會（非 0 時覆寫 AEC filter 長度／n_partitions） |
| `delay_mode` | 會（`MATCHED` 建 matched-filter estimator + 大 ring；`FIXED` 只建剛好夠用的小 ring；`EXTERNAL_ALIGNED` 兩者都不建，見 §4.5a） |
| `delay_num_filters` | 僅 `MATCHED` 有效：每少一個固定省 5,728 B（四格點皆同，與 `lib/aec` 一致） |
| `fixed_delay_samples` | 僅 `FIXED` 有效：ring 大小 = `ALIGN16((fixed_delay_samples+hop)×4)` B |
| `aec_preset`、`nr_mode`、`res_preset` | 實測不變 |
| `enable_nr` | 會（`0` 省下整份 MMSE-LSA state） |
| `res` | `AEC_RES` 多一份 `AecRes`（`aec_res_get_mem_size()`）；`FRAME_GAIN`／`OFF` 不加 |
| `with_synthesis` | 會（`0` 省下 `AcIstft`） |
| `enable_cng`、`legacy_amin` | 不變 |

### 4.4b 診斷用的分項

```c
SeSampleChainSizes cs;
if (se_sample_chain_sizes(&sc, &cs) == 0) { /* cs.res_bytes, cs.nr_bytes, cs.synth_bytes, cs.total */ }
```

`aec_get_mem_size()` 給 AEC 那一塊。`cs.total` 已含鏈的控制區塊與增益／頻譜暫存，各自 16-byte 對齊。
這只是給人看的表格；**真正要配置多少，永遠以兩個 `mem_size` 的回傳值為準**，加上各塊的對齊。

### 4.5 實測記憶體（僅供量級參考，務必自己重查）

以下是**本次 checkout、`BACKEND=ne10`、`pipelines/Makefile` 預設選項**下，以
`aec_nr_pipeline_static --print-mem-size` 量到的值（delay 列用 `aec_get_mem_size()` 與
`se_sample_chain_mem_size()` 直接呼叫）。換編譯選項、更新 submodule 都會變，請自己重查。

鏈的控制區塊與增益／頻譜暫存合計於「鏈其餘」欄（`ALIGN16(sizeof(SeSampleChain))` 加三個 `n_freqs` float 向量與一個 `n_freqs` 複數向量）。

| Config | 總計 | AEC（`aec_get_mem_size`） | 合成 `AcIstft` | `nr_bytes` | 鏈其餘 |
|---|---:|---:|---:|---:|---:|
| 8000，預設 | 327,232 | 284,352 | 11,344 | 28,784 | 2,752 |
| 16000，預設（256/128） | 457,184 | 397,744 | 11,344 | 45,344 | 2,752 |
| 16000，`fft_size=512` | 609,024 | 525,440 | 21,840 | 56,432 | 5,312 |
| 48000，預設（1024/512） | 1,422,016 | 1,224,176 | 42,832 | 144,576 | 10,432 |
| 16000，`enable_nr=0` | 411,840 | 397,744 | 11,344 | 0 | 2,752 |
| 16000，`--aec-only`（不建鏈） | 397,744 | 397,744 | – | – | – |

總計 = `ALIGN16(AEC)` + 鏈（各塊對齊後相加）。

#### 4.5a delay_mode 對 16000/預設 grid 的影響（`MATCHED` n=5 為 baseline）

| `delay_mode` | AEC 加鏈合計 | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 457,184 | — |
| `MATCHED` n=1 | 434,272 | −22,912 |
| `FIXED`，`fixed_delay_samples=1600`（100 ms） | 299,088 | −158,096 |
| `EXTERNAL_ALIGNED` | 292,176 | −165,008 |

這四列的差額全部落在 AEC 那一塊（`delay_mode`/`delay_num_filters` 不影響
鏈），數字與 `lib/aec` 自己的 `aec_get_mem_size()` 一致
（見 `lib/aec/docs/c_user_manual_zh_TW.md` §4）。**沒有 CLI 旗標**能測試
非預設 delay mode——上表用直接呼叫
`aec_get_mem_size()`／`se_sample_chain_mem_size()` 量得。

---

## 5. 錯誤處理

這個 codebase 混用了數種回傳慣例。下表是**每個 function 實際的**慣例。

### 5.1 錯誤語意總表

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `se_derive()` | `0`，輸出填妥 | `-1` | 輸出未定義；無任何 log |
| `aec_get_mem_size()` | pool bytes | `0`（config 無效） | 與 4ch 的 `get_mem_requirements()`（0 = 成功）相反 |
| `se_sample_chain_mem_size()` | pool bytes | `0`（config 無效） | 同上 |
| `se_sample_chain_sizes()` | `0`，`*out` 填妥 | `-1` | 診斷分項 |
| `aec_init()` | 非 NULL handle | `NULL` | pool 未對齊、太小、config 無效 |
| `se_sample_chain_init()` | 非 NULL handle | `NULL` | 同上，且任何子物件 init 失敗；失敗後 pool 仍可重新用於正確的 init |
| `aec_process_context()` / `aec_process()` | `void` | – | 不回傳錯誤；異常由後面的 `se_frame_from_aec()`／`form()` 回傳值看出 |
| `se_frame_from_aec()` | `0` | `-1` | NULL 引數，或 lane 沒有 seam（`enable_res=0`、`return_res_context=1`） |
| `se_sample_chain_form()` | 鏈擁有的 `P`（`n_freqs` bin） | `NULL` | NULL 引數、frame 的 `n_freqs` 不同、或 stage 拒絕該 hop |
| `se_sample_chain_synthesize()` | `0` | `-1` | NULL 引數或鏈沒有合成 |
| `se_hop_is_finite()` | `1` | `0` | 你在採用輸出之前要做的檢查 |
| `aec_reset()` / `se_sample_chain_reset()` | `void` | 不會失敗 | NULL 時是 no-op（鏈）；兩個要一起呼叫 |
| `aec_destroy()` / `se_sample_chain_destroy()` | `void` | 不會失敗 | NULL 安全 |
| `aec_set_preset()` / `anr_set_mode()` | `0` | `-1` | `-1` 時什麼都不寫（§3.2） |

> **注意慣例不一致**：`aec_get_mem_size()` / `se_sample_chain_mem_size()` / `anr_get_mem_size()` /
> `fft_get_mem_size()` 是 **回傳 0 代表失敗**；4ch 單元的 `audio_pipeline_4ch_get_mem_requirements()` 是
> **0 = 成功 / −1 = 失敗**。判斷式別混用。

### 5.2 init 的拒絕清單

`aec_init()`：`mem == NULL`、`mem` 未 16-byte 對齊、`mem_size` 小於 `aec_get_mem_size()`、config 無效、子模組 init 失敗 → `NULL`。
`se_sample_chain_init()`：`mem == NULL`、未 16-byte 對齊、`bytes` 小於 `se_sample_chain_mem_size()`、config 無效
（出界 enum、bool 欄位不是 0/1、不合法格點）、`AecRes`／降噪器／合成物件 init 失敗 → `NULL`。
兩者都**不會**動到 pool 以外的記憶體，被拒之後同一塊 pool 仍可用在正確的 init
（`example_board_adapter.c` 的 `run_negative_demonstrations()` 展示並檢查這一點）。

### 5.3 `destroy()` 的語意

兩個 `destroy()` 都**不釋放**你的 pool；`aec_destroy()` NULL 安全、可重複呼叫；`se_sample_chain_destroy()` 釋放降噪器的內部資源。
`posix_memalign` 之類由你配置的 pool，由你在 destroy 之後釋放。

### 5.4 診斷輸出去向

* mono 範例鏈本身（header-only）不輸出任何東西。lib/aec／lib/nr／audiolib 的 init 期診斷寫到 **stderr**，
  以 `NO_STDIO=1` 建置時編成 no-op，**回傳值語意完全不變**。
* 每 hop 的 `aec_process_context()`、`form()`、`synthesize()` **在任何 build 下都不會輸出任何東西**。

---

## 6. 出貨內容 vs 範例

`pipelines/` 目錄不是自解釋的。下表區分「函式庫」與「參考程式」。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `lib/aec`、`lib/nr`、audiolib | `libaec.a`、`libanr.a`、`libaudio_common.a` | 線性 AEC（含 AEC3 殘留 gain context）、降噪、FFT／STFT／合成 |
| `pipelines/common/se_config.c` / `.h` | `se_config.o`（4ch 單元的 archive 也含它） | 格點解析與 `AecConfig` 推導（`se_derive()`）；各應用的規則（rate 集合、delay 檢查、lane 對齊、seam context）以 `SeRules` 傳入 |
| `pipelines/common/se_stage.h` | header-only | `SeSpecFrame`、`se_res_stage_gain()`、`se_frame_from_aec()`、CNG 常數 |

### 6.2 範例碼與參考執行檔（**沒有任何一個會進產品 archive**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/samples/se_sample_chain.h`、`se_anr_stage.h` | header-only 範例鏈與 NR 級的 gain 呼叫 | 示範 RES 增益 → NR 增益 → min → apply → comfort noise → 合成；可整份複製改寫，沒有 ABI |
| `pipelines/mono_aec_nr_res/main.c` | heap 路徑 CLI（WAV in / WAV out） | 示範 `Aec` + 範例鏈的組合；離線比對用 |
| `pipelines/mono_aec_nr_res/static_main.c` | caller-owned pool 路徑 CLI | 示範 query → allocate → init → process → destroy → release；另提供 `--print-mem-size` |
| `pipelines/mono_aec_nr_res/example_board_adapter.c` | 板端 adapter 的**主機模擬** | 示範完整呼叫序列、pool 大小加總與拒絕路徑。裡面每個 `board_mem_*` 都是主機上的假實作，不是平台程式碼。標了 `// BOARD:` 的地方就是你要換成真實平台程式碼的位置 |

四麥克風的 library 與參考程式是**另外兩份**手冊的主題：
`docs/integration_4ch_core_zh_TW.md`、`docs/integration_4ch_spatial_zh_TW.md`。

### 6.3 測試（不出貨）

`pipelines/mono_aec_nr_res/tests/test_mono_chain.c`（鏈對手寫 oracle `conv_chain_mono.h` 逐位元）、
`pipelines/mono_aec_nr_res/tests/test_no_stdio_stack.c`。

---

## 7. 整合檢查清單

建置與連結：

- [ ] `libaec.a` / `libanr.a` / `libaudio_common.a` 與你的 `se_config.o` 是用**同一組** backend 與編譯選項建出來的
- [ ] link 順序照 §1.5
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用各 Makefile 的 `print-lib-path` 查的，不是硬寫的
- [ ] 你的 TU 用 `gnu99`（或以上的 GNU 方言）

初始化：

- [ ] 每次 init 之前都重新呼叫 `aec_get_mem_size()` 與 `se_sample_chain_mem_size()`，沒有快取大小
- [ ] 兩塊的大小都檢查過 `!= 0`
- [ ] pool 16-byte 對齊，大小 ≥ `ALIGN16(aec_bytes)` + `chain_bytes`（加上你自己的 stage）
- [ ] `aec_init()` 與 `se_sample_chain_init()` 回 `NULL` 時你的程式會停下來，不會繼續跑

執行期：

- [ ] `mic` / `ref` / `out` 三個 buffer 都剛好 `grid.hop_size` 個 float
- [ ] hop 大小是查來的，不是寫死的
- [ ] `se_frame_from_aec()`、`se_sample_chain_form()`、`se_sample_chain_synthesize()` 的回傳值與輸出有限性都有檢查
- [ ] 任何一步被拒，或換回聲路徑／換串流時，`aec_reset()` 與 `se_sample_chain_reset()` **一起**呼叫
- [ ] 同一個 instance 沒有被多 thread 同時 process
- [ ] 沒有直接對 `se_sample_chain_nr()` 拿到的降噪器呼叫 reset／destroy

收尾：

- [ ] `destroy()` 在釋放 pool **之前**
- [ ] pool 在 handle 存活期間沒有被搬動、覆寫或共用

板端額外：

- [ ] 若 image 不能連 stdio，函式庫是以 `NO_STDIO=1` 建的
- [ ] 已確認 pool 不需要預先清零（dirty pool 可接受）
- [ ] bring-up log 有保留 init 的拒絕訊息（NO_STDIO build 就沒有了，此時只能靠回傳值）

---

## 8. 版本與相容性

mono 沒有 descriptor，也沒有 layout version：`Aec`、`AnrDenoiser`、`AecRes`、`AcIstft` 只提供大小查詢，
範例鏈是範例碼。需要 descriptor 閘的出貨單元只有 4ch 的 `AudioPipeline4Ch`
（`AudioPipeline4ChMemReq`，`AUDIO_PIPELINE_4CH_LAYOUT_VERSION`）與 DFN2 的 stage／bridge，
見 [`integration_4ch_spatial_zh_TW.md`](integration_4ch_spatial_zh_TW.md) 與
[`pipeline_board_integration.md`](pipeline_board_integration.md)。

給整合者的兩條規則：

1. **大小永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用。
2. **升級 library 之後一定要重跑一次大小查詢**，並確認你的 pool 預算仍然足夠。
