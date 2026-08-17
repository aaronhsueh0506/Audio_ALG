# 單聲道 Pipeline 整合手冊（`libaudio_pipeline.a`）

適用對象：要把單聲道 AEC → NR → RES pipeline 連進自己產品的**函式庫使用者**。

本文只講「什麼時候呼叫哪個 function」「每個參數怎麼設」。演算法內部推導不在本文範圍。

> **路徑慣例**：本文所有路徑都相對於 Audio_ALG repository 根目錄。
> `audio_common` 是與 Audio_ALG 平行的另一個 checkout，本文寫成 `<audio_common>/`；
> 從 `pipelines/` 出發的相對路徑是 `../../audio_common`。
>
> **數值來源**：本文所有數字都是從當前 source 讀出，或以當前 checkout 實際建置後量測。
> 記憶體數字**必須**用 API 重新查詢，不要抄本文（見 §4.5）。

---

## 1. 這個函式庫是什麼、你連結什麼、你 `#include` 什麼

### 1.1 這是什麼

`libaudio_pipeline.a` 提供一條固定的單聲道處理鏈，一次吃一個 hop 的 `mic` / `ref`，吐一個 hop 的 `out`：

```
mic hop + ref hop  ──►  linear AEC  ──►  echo-aware NR  ──►  RES gain 融合  ──►  iFFT + OLA  ──►  out hop
```

* 它**不做** WAV I/O、不解析 argv、不寫 stdout。
* 它**不呼叫** `malloc`，除非你走 `audio_pipeline_create()` 這條 heap 便利路徑。
* 它一次只處理**剛好一個 hop**，hop 長度由 config 決定，用 `audio_pipeline_hop_size()` 查。

### 1.2 Header closure

你的 TU 只要寫一行：

```c
#include "audio_pipeline.h"
```

`audio_pipeline.h` 自己會拉進 `aec.h` 與 `mmse_lsa_denoiser.h`，後者再拉進 `fft_wrapper.h` 等。
你不需要、也不應該自己 include 那些下游 header。

### 1.3 `Complex` type 從哪來

`Complex` 定義在 `<audio_common>/include/fft_wrapper.h`。
單聲道 public API **不需要**你直接使用 `Complex`（`AudioPipelineConfig`、`AudioPipelineMemReq`、
`AudioPipelineMemBreakdown`、`audio_pipeline_process()` 都沒有 `Complex` 參數）。
只有當你想自己讀 `audio_pipeline_get_aec()` 回傳的 AEC 診斷資料時才會碰到它，
而那時 `Complex` 已經透過上面那一行 include 進來了。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I pipelines
-I lib/aec/c_impl/include
-I lib/nr/c_impl/include
-I <audio_common>/include
```

四個就夠（已實測：只用這四個即可編過只含 `#include "audio_pipeline.h"` 的 TU）。
`pipelines/Makefile` 自己建置 library 時的 `-I` 清單比這個長（多了 `lib/aec/c_impl/example`
與 `<audio_common>/lib/kiss_fft`），那是給 CLI 的 WAV I/O 與後端內部用的，consumer 不需要。

語言標準：library 以 `-std=gnu99` 建置；NE10 後端的 header 需要 GNU 擴充，
所以你的 TU 也請用 `gnu99`（或以上的 GNU 方言）。

### 1.5 Archive link order

link 順序有意義（單向相依，由上而下）：

```
你的 .o
libaudio_pipeline.a
libaec.a
libmmse_lsa.a
libaudio_common.a
-lm
```

* `libaudio_pipeline.a` **只含一個 object**（`audio_pipeline.o`），不是把 AEC/NR/audio_common
  打包進去的 fat archive。上面四個 archive 你都要自己連。
* 四個 archive 的實際路徑是 config-keyed 的（`bin/<backend>-<hash>/`），不是固定名稱。
  用各自 Makefile 的 `make -s print-lib-path` 取得，不要硬寫路徑。
* `BACKEND=ne10` 時 audio_common 內含一個 C++ TU，最終 link 要用 C++ driver（`c++`），
  不是 `cc`。`BACKEND=kiss` 用 `cc` 即可。
* 四個 archive 必須用**同一組** backend / 編譯選項建置。混用會在
  `audio_pipeline_init_ex()` 的 descriptor 檢查被擋下（見 §5.3）。

---

## 2. Quick start

以下是完整、可編譯、已實際跑過的最小整合（heap 路徑）：

```c
#include <stdio.h>
#include "audio_pipeline.h"

int main(void) {
    AudioPipelineConfig cfg = audio_pipeline_default_config(16000);

    AudioPipeline* p = audio_pipeline_create(&cfg);   /* heap path */
    if (!p) { fprintf(stderr, "create failed\n"); return 1; }

    const int hop = audio_pipeline_hop_size(p);
    float mic[1024], ref[1024], out[1024];

    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < hop; i++) {           /* 換成你自己的擷取來源 */
            mic[i] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }
        if (audio_pipeline_process(p, mic, ref, out) != 0) {
            fprintf(stderr, "process failed\n");
            audio_pipeline_destroy(p);
            return 1;
        }
        /* 消費 out：hop 個 float，已增強的近端音訊 */
    }

    audio_pipeline_destroy(p);   /* 釋放 create() 配置的 pool */
    printf("ok hop=%d\n", hop);
    return 0;
}
```

編譯（`<...>` 換成你 `print-lib-path` 查到的路徑）：

```sh
cc -std=gnu99 -O2 -Wall -Wextra -o app app.c \
  -I pipelines \
  -I lib/aec/c_impl/include \
  -I lib/nr/c_impl/include \
  -I <audio_common>/include \
  <pipelines-bin>/libaudio_pipeline.a \
  <aec-bin>/libaec.a \
  <nr-bin>/libmmse_lsa.a \
  <audio_common-bin>/libaudio_common.a \
  -lm
```

> `float mic[1024]` 是因為本例已知 hop 最大為 512（48 kHz）。
> 產品程式請依 `audio_pipeline_hop_size()` 回傳值配置，不要寫死。

---

## 3. Lifecycle

板端（caller-owned pool）的完整順序如下。heap 路徑（`audio_pipeline_create()`）
把第 1~3 步合成一步，其餘相同。

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | query | `audio_pipeline_get_mem_requirements(&cfg, &req)` | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | 回傳 `-1`：`cfg` 或 `out` 為 NULL；config 未通過驗證（§4）；此 TU 編譯時沒帶 `-DAUDIO_PIPELINE_BACKEND_STR="kiss"` 或 `"ne10"`（`backend_id` 會是 0，直接拒絕） |
| 2 | allocate | 你自己的配置器，取得 `req.bytes` 且對齊 `req.alignment` 的記憶體 | 拿到 `req` 之後 | 配置失敗由你自己處理。pool **不需要**預先清零（poison pattern 也可以，已實測）；但必須是 16-byte 對齊、且在 handle 生命週期內獨佔、不被其他人寫入 |
| 3 | init | `audio_pipeline_init_ex(pool, req.bytes, &cfg, &req)` | 拿到 pool 之後，處理第一個 hop 之前 | 回傳 `NULL`：8 項 descriptor 檢查任一不符（§5.3）；pool 未 16-byte 對齊；pool 太小；config 無效；子模組 init 失敗；grid 不一致 |
| 4 | process | `audio_pipeline_process(p, mic, ref, out)` | 每收到一個 hop 呼叫一次。三個 buffer 都必須剛好 `audio_pipeline_hop_size(p)` 個 float | 回傳 `-1`：`p`/`mic`/`ref`/`out` 任一為 NULL。**不會**檢查 buffer 長度 —— 長度錯是你的責任 |
| 5 | reset | `audio_pipeline_reset(p)` | 換回聲路徑（換喇叭、重新 seat）或切換到不相關的另一條串流時 | 不會失敗。`p == NULL` 時是 no-op。不重新驗證 config、不動 pool |
| 6 | destroy | `audio_pipeline_destroy(p)` | 不再處理任何 hop 之後、釋放 pool 之前 | 不會失敗。`NULL` 安全。pool instance 可重複呼叫；`create()` instance **只能呼叫一次**（見 §5.4） |
| 7 | release | 你自己的釋放器 | **一定要在第 6 步之後** | 順序顛倒 = use-after-free |

補充規則：

* **`req` 不可快取**。firmware image 若拿舊 build 算出的 `req` 去餵新 library，
  就會在錯誤大小/佈局的 pool 上切記憶體。第 3 步用 `init_ex()` 把剛查到的 `req`
  傳回去，就是為了在板端 bring-up 當場擋下這件事。
* `audio_pipeline_init(pool, bytes, &cfg)` 等同 `audio_pipeline_init_ex(pool, bytes, &cfg, NULL)`
  —— 不做 descriptor 檢查。只有在你每次都重新 derive `req` 的情況下才適合用它。
* pool 生命週期內 sub-module 都是指進 pool 的裸指標，不是拷貝。pool 不能被搬動、
  不能被別人寫、不能和另一個 `AudioPipeline`/AEC/NR/FFT instance 共用。
* 單一 instance 不可被多個 thread 同時 `process`。多條串流請各自建立 instance。

### 3.1 執行期唯讀存取

| 呼叫 | 用途 |
|---|---|
| `audio_pipeline_hop_size(p)` | 每次 `process()` 要餵幾個 float |
| `audio_pipeline_n_freqs(p)` | 頻域 bin 數（`fft_size/2 + 1`） |
| `audio_pipeline_sample_rate(p)` | 實際生效的取樣率 |
| `audio_pipeline_get_aec(p)` | 取得底層 `Aec*`，**只**用來做你自己的診斷讀取 |
| `audio_pipeline_get_nr(p)` | 取得底層 `MmseLsaDenoiser*`，同上；`cfg.aec_only` 時為 `NULL` |

`get_aec()` / `get_nr()` 拿到的 handle **不可**直接呼叫任何 `_reset` / `_destroy` /
會改狀態的 entry point —— 一定要走 `audio_pipeline_reset()` / `audio_pipeline_destroy()`，
否則 pipeline 自己持有的狀態（OLA、CNG RNG、near-end hangover counter）會和子模組脫節。

---

## 4. Config 完整參考

`AudioPipelineConfig` 共 7 個欄位。取得預設值的唯一正確方式：

```c
AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
/* 之後只覆寫你真正要改的欄位 */
```

`audio_pipeline_default_config()` **不做驗證**，它只是填值；驗證發生在
`get_mem_requirements()` / `init()` / `init_ex()` / `get_mem_breakdown()`
共用的同一個 reject-first 閘。

### 4.1 逐欄位

| 欄位 | 型別 | `default_config()` 給的值 | 合法值（驗證器實際檢查的） | 你該怎麼設 |
|---|---|---|---|---|
| `sample_rate` | `int` | 你傳給 `default_config()` 的值 | `8000` / `16000` / `48000`。其他一律拒絕（例如 `44100` 會在任何 size 運算之前就被擋掉） | 設成你音訊實際的取樣率。這個值同時決定預設 grid |
| `fft_size` | `int` | `0` | `0` = 依 rate 取預設。明確指定時：8 kHz 只接受 `256`；16 kHz 接受 `256` 或 `512`；48 kHz 只接受 `1024`。其他組合拒絕 | 除非你有明確理由要 16 kHz 的 512 grid，否則留 `0` |
| `aec_preset` | `AecPreset` | `AEC_PRESET_BALANCED`（= 1） | `AEC_PRESET_MILD`(0) / `AEC_PRESET_BALANCED`(1) / `AEC_PRESET_AGGRESSIVE`(2)。列舉以外的整數會被拒絕（**不會**默默 fallback 成 balanced） | 近端保留優先 → `MILD`；回聲抑制優先 → `AGGRESSIVE`；不確定就留預設 |
| `nr_mode` | `MmseLsaNrMode` | `MMSE_LSA_NR_BALANCED`（= 2） | `MMSE_LSA_NR_MILD`(0) / `MMSE_LSA_NR_MODERATE`(1) / `MMSE_LSA_NR_BALANCED`(2) / `MMSE_LSA_NR_AGGRESSIVE`(3)。列舉以外拒絕 | 降噪強度。四級都可用（CLI 只認得其中三個，library API 四個都能設） |
| `aec_only` | `int`（bool） | `0` | 只接受 `0` 或 `1`。`2` 之類的「truthy」值會被拒絕 | `1` = 只跑 linear AEC，完全跳過 NR/RES/最終 OLA。用來隔離問題，或你自己接後級。此時 `get_nr()` 回 `NULL`，且 FFT/NR/pipeline buffer 都不配置 |
| `enable_cng` | `int`（bool） | `1` | 只接受 `0` 或 `1` | `1` = 在 AEC 抑制掉的 bin 填舒適噪音。實際生效值是「AEC preset 自己的 `enable_cng`」與這個欄位的 AND |
| `legacy_amin` | `int`（bool） | `0` | 只接受 `0` 或 `1` | `1` = 回到舊的 min-only 行為：NR 的 noise floor 不摺入 R²，且 near-end floor 強度固定。只用於比對舊行為，新整合請保持 `0` |

### 4.2 Grid（由 `sample_rate` + `fft_size` 唯一決定）

`hop = fft_size / 2`、`frame = fft_size`、`n_freqs = fft_size / 2 + 1`。
合法組合只有這四組：

| `sample_rate` | `fft_size` | `hop` | `frame` | `n_freqs` | 備註 |
|---:|---:|---:|---:|---:|---|
| 8000 | 256 | 128 | 256 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 256 | 128 | 256 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 512 | 256 | 512 | 257 | 需明確指定 |
| 48000 | 1024 | 512 | 1024 | 513 | `fft_size = 0` 的 rate 預設 |

**不要把 hop 寫死。** 16 kHz 有兩組合法 grid，寫死任一組對另一組就是錯的。
一律用 `audio_pipeline_hop_size()` / `audio_pipeline_n_freqs()` 查。

### 4.3 哪些欄位會改變記憶體用量

| 欄位 | 影響 `req.bytes` 嗎 |
|---|---|
| `sample_rate` | 會（改 grid、改 AEC filter 長度） |
| `fft_size` | 會（改 grid） |
| `aec_only` | 會（`1` 時不配置 FFT/NR/pipeline buffer） |
| `filter_length` | 會（非 0 時覆寫 AEC filter 長度／n_partitions） |
| `delay_mode` | 會（`MATCHED` 建 matched-filter estimator + 大 ring；`FIXED` 只建剛好夠用的小 ring；`EXTERNAL_ALIGNED` 兩者都不建，見下方 §4.5a） |
| `delay_num_filters` | 僅 `MATCHED` 有效：每少一個固定省 5,728 B（四格點皆同，與 `lib/aec` 一致） |
| `fixed_delay_samples` | 僅 `FIXED` 有效：ring 大小 = `ALIGN16((fixed_delay_samples+hop)×4)` B |
| `aec_preset` | 實測不變 |
| `nr_mode` | 實測不變 |
| `enable_cng` | 不變 |
| `legacy_amin` | 不變 |

### 4.4 診斷用的分項

```c
AudioPipelineMemBreakdown b;
if (audio_pipeline_get_mem_breakdown(&cfg, &b) == 0) { /* b.aec_bytes, ... */ }
```

`AudioPipelineMemBreakdown` 欄位：`aec_bytes`、`fft_bytes`、`nr_bytes`、`pipeline_bytes`、
`hop`、`frame_sz`、`fft_sz`、`n_freqs`。
`aec_only` 時 `fft_bytes` / `nr_bytes` / `pipeline_bytes` 為 0。

這只是給人看的表格。**真正要配置多少，永遠以 `AudioPipelineMemReq.bytes` 為準** ——
`req.bytes` 還額外含 `AudioPipeline` 控制區塊，分項表沒有把它獨立成一欄。

### 4.5 實測記憶體（僅供量級參考，務必自己重查）

以下是**本次 checkout（layout_version=6，delay-mode 產品化後）、`BACKEND=kiss`、
`pipelines/Makefile` 預設選項**下，2026-08-16 直接呼叫 API 量到的值。換
backend、換編譯選項、更新 submodule 都會變。

| Config | `req.bytes` | `aec_bytes` | `fft_bytes` | `nr_bytes` | `pipeline_bytes` |
|---|---:|---:|---:|---:|---:|
| 8000，預設 | 357,712 | 275,648 | 8,784 | 67,424 | 5,696 |
| 16000，預設（256/128） | 516,528 | 379,728 | 8,784 | 122,160 | 5,696 |
| 16000，`fft_size=512` | 670,736 | 508,800 | 16,976 | 133,472 | 11,328 |
| 48000，預設（1024/512） | 1,597,472 | 1,167,024 | 33,360 | 374,336 | 22,592 |
| 16000，`aec_only=1` | 379,888 | 379,728 | 0 | 0 | 0 |

`req.bytes` 減去四個分項（各自 16-byte 對齊後）的差額，就是 `AudioPipeline`
控制區塊，本次量測在每一組 config 都是 160 B。

#### 4.5a delay_mode 對 16000/預設 grid 的影響（`MATCHED` n=5 為 baseline）

| `delay_mode` | `req.bytes` | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 516,528 | — |
| `MATCHED` n=1 | 493,616 | −22,912 |
| `FIXED`，`fixed_delay_samples=1600`（100 ms） | 358,432 | −158,096 |
| `EXTERNAL_ALIGNED` | 351,520 | −165,008 |

這四列的差額全部落在 `aec_bytes`（`delay_mode`/`delay_num_filters` 不影響
FFT/NR/pipeline 分項），數字與 `lib/aec` 自己的 `aec_get_mem_size()` 一致
（見 `lib/aec/docs/c_user_manual_zh_TW.md` §4）。**沒有 CLI 旗標**能測試
非預設 delay mode——上表用直接呼叫
`audio_pipeline_get_mem_requirements()`/`_get_mem_breakdown()` 量得。

---

## 5. 錯誤處理

這個 codebase 混用了數種回傳慣例。下表是**每個 function 實際的**慣例。

### 5.1 錯誤語意總表

| Function | 成功 | 失敗 | 失敗時的診斷 |
|---|---|---|---|
| `audio_pipeline_default_config()` | 永遠回傳一個填好的 struct | 不會失敗，**也不驗證** | 無 |
| `audio_pipeline_get_mem_requirements()` | `0`，`*out` 填妥 | `-1` | 無（僅回傳碼） |
| `audio_pipeline_get_mem_breakdown()` | `0`，`*out` 填妥 | `-1` | 無 |
| `audio_pipeline_init()` | 非 NULL handle | `NULL` | 部分路徑寫 stderr（見 5.2） |
| `audio_pipeline_init_ex()` | 非 NULL handle | `NULL` | 每一項 descriptor 不符各自一行 stderr（見 5.3） |
| `audio_pipeline_create()` | 非 NULL handle | `NULL` | 同 `init()`；另含 `posix_memalign` 失敗 |
| `audio_pipeline_process()` | `0` | `-1` | 無。**per-hop 路徑完全不寫任何 log** |
| `audio_pipeline_reset()` | `void` | 不會失敗 | `p == NULL` 時靜默 no-op |
| `audio_pipeline_destroy()` | `void` | 不會失敗 | `p == NULL` 時靜默 no-op |
| `audio_pipeline_hop_size()` | ≥ 1 | `-1`（`p == NULL`） | 無 |
| `audio_pipeline_n_freqs()` | ≥ 1 | `-1`（`p == NULL`） | 無 |
| `audio_pipeline_sample_rate()` | > 0 | `-1`（`p == NULL`） | 無 |
| `audio_pipeline_get_aec()` | 非 NULL（handle 有效時必定非 NULL） | `NULL`（`p == NULL`） | 無 |
| `audio_pipeline_get_nr()` | 非 NULL | `NULL`（`p == NULL`，或 `cfg.aec_only == 1`） | 無 |

> **注意慣例不一致**：本層的 `get_mem_requirements()` / `get_mem_breakdown()` 用 **0 = 成功 / −1 = 失敗**；
> 但底下模組的 `aec_get_mem_size()` / `mmse_lsa_get_mem_size()` / `fft_get_mem_size()` 是
> **回傳 0 代表失敗**。如果你直接呼叫那些下層 function，判斷式要反過來寫。

### 5.2 `audio_pipeline_init()` 的完整拒絕清單

依實際檢查順序：

1. `mem == NULL` 或 `cfg == NULL` → `NULL`（無診斷）
2. `mem` 未 16-byte 對齊 → `NULL` + stderr
3. `cfg` 未通過驗證（§4.1 的任一條）→ `NULL`（無診斷）
4. `bytes` 連 `AudioPipeline` 控制區塊都放不下 → `NULL` + stderr
5. 扣掉控制區塊後的 sub-pool 太小 → `NULL` + stderr
6. `aec_init()` 失敗 → `NULL` + stderr
7. `fft_init()` 或 `mmse_lsa_init()` 失敗（非 `aec_only` 時）→ `NULL` + stderr
8. pipeline 與 AEC 的 `n_freqs`/`hop` 不一致，或與 FFT/NR 的 `n_freqs` 不一致 → `NULL` + stderr

### 5.3 `audio_pipeline_init_ex()` 額外的 8 項 descriptor 檢查

`expected == NULL` 時完全等同 `audio_pipeline_init()`。
`expected != NULL` 時，依序檢查下列 8 項，任一不符即回 `NULL`（**不會**動到 pool），
每一項各自輸出一行含欄位名與兩邊整數值的 stderr 訊息：

| # | 檢查 |
|---|---|
| 1 | `expected->descriptor_version == 目前 build 的值`（最先檢查：struct ABI 不同時其他欄位都沒有意義） |
| 2 | `expected->layout_version == 目前 build 的值` |
| 3 | `expected->backend_id == 目前 build 的值`（整數 `==` 比較，不是 `strcmp`） |
| 4 | `expected->build_flags_hash == 目前 build 的值` |
| 5 | `expected->alignment == 目前 build 的值` |
| 6 | `expected->reserved == 0` |
| 7 | `expected->bytes >= 目前 build 的需求` |
| 8 | `bytes`（這次實際交進來的 pool 大小）`>= 目前 build 的需求` |

第 7 與第 8 是兩件不同的事：呼叫端可能拿著一個 `bytes` 夠大的舊 descriptor，
但實際配置/傳入了更小的一塊記憶體。

### 5.4 `destroy()` 的兩種語意

| Instance 來源 | `destroy()` 行為 |
|---|---|
| `audio_pipeline_init()` / `init_ex()`（caller-owned pool） | **不釋放**你的 pool。NULL 安全、**冪等**，重複呼叫安全。之後由你自己交還 pool |
| `audio_pipeline_create()`（heap） | 釋放 `create()` 配置的那一塊。遵循一般 `free()` 語意：**只能呼叫一次**。第二次呼叫就是 double-free，本 function 無法偵測（它要檢查的那塊記憶體正是被釋放的那塊） |

### 5.5 診斷輸出去向

* 預設：所有 init/build 期診斷寫到 **stderr**。
* 以 `NO_STDIO=1` 建置 `libaudio_pipeline.a` 時：所有診斷編成 no-op，且該 object
  完全不會參照 `fprintf`/`printf`/`puts`/`fputs`/`stderr`。
  **回傳值語意完全不變** —— 每一種失敗都仍然透過 `NULL` / `-1` 通知你。
* per-hop 的 `audio_pipeline_process()` **在任何 build 下都不會輸出任何東西**。

---

## 6. 出貨內容 vs 範例

`pipelines/` 目錄不是自解釋的。下表區分「函式庫」與「參考程式」。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `pipelines/mono_aec_nr_res/audio_pipeline.c` | `libaudio_pipeline.a` | archive 內**只有這一個 object**。這是本手冊的主體 |
| `pipelines/mono_aec_nr_res/audio_pipeline.h` | — | 公開 API |
| `pipelines/mono_aec_nr_res/pipeline_dims.h` | — | grid 解析用的 `static inline` header，被 `audio_pipeline.c` 與兩個 CLI 共用 |

### 6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/mono_aec_nr_res/main.c` | heap 路徑 CLI（WAV in / WAV out） | 示範 `audio_pipeline_create()` 用法；離線比對用 |
| `pipelines/mono_aec_nr_res/static_main.c` | caller-owned pool 路徑 CLI | 示範 query → allocate → init → process → destroy → release；另提供 `--print-mem-size` |
| `pipelines/mono_aec_nr_res/example_board_adapter.c` | 板端 adapter 的**主機模擬** | 示範完整呼叫序列、錯誤處理與 descriptor contract。裡面每個 `board_mem_*` 都是主機上的假實作，不是平台程式碼。標了 `// BOARD:` 的地方就是你要換成真實平台程式碼的位置 |

四麥克風的 library 與參考程式是**另外兩份**手冊的主題：
`docs/integration_4ch_core_zh_TW.md`、`docs/integration_4ch_spatial_zh_TW.md`。

### 6.3 測試（不出貨）

`pipelines/mono_aec_nr_res/tests/test_audio_pipeline.c`、
`pipelines/mono_aec_nr_res/tests/test_no_stdio_stack.c`。

---

## 7. 整合檢查清單

建置與連結：

- [ ] 四個 archive（`libaudio_pipeline.a` / `libaec.a` / `libmmse_lsa.a` / `libaudio_common.a`）
      是用**同一組** backend 與編譯選項建出來的
- [ ] link 順序照 §1.5
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用各 Makefile 的 `print-lib-path` 查的，不是硬寫的
- [ ] 你的 TU 用 `gnu99`（或以上的 GNU 方言）

初始化：

- [ ] 每次 init 之前都重新呼叫 `audio_pipeline_get_mem_requirements()`，沒有快取 `req`
- [ ] pool 對齊 `req.alignment`（目前是 16），大小 ≥ `req.bytes`
- [ ] 用 `audio_pipeline_init_ex()` 並把剛查到的 `req` 當 `expected` 傳回去
- [ ] `init_ex()` 回 `NULL` 時你的程式會停下來，不會繼續跑

執行期：

- [ ] `mic` / `ref` / `out` 三個 buffer 都剛好 `audio_pipeline_hop_size(p)` 個 float
- [ ] hop 大小是查來的，不是寫死的
- [ ] `audio_pipeline_process()` 的回傳值有檢查
- [ ] 換回聲路徑或換串流時有呼叫 `audio_pipeline_reset()`
- [ ] 同一個 instance 沒有被多 thread 同時 process
- [ ] 沒有直接對 `get_aec()` / `get_nr()` 拿到的 handle 呼叫 `_reset` / `_destroy`

收尾：

- [ ] `destroy()` 在釋放 pool **之前**
- [ ] `create()` 建的 instance 只 `destroy()` 一次
- [ ] pool 在 handle 存活期間沒有被搬動、覆寫或共用

板端額外：

- [ ] 若 image 不能連 stdio，library 是以 `NO_STDIO=1` 建的
- [ ] 已確認 pool 不需要預先清零（dirty pool 可接受）
- [ ] bring-up log 有保留 `init_ex()` 的拒絕訊息（NO_STDIO build 就沒有了，
      此時只能靠回傳值）

---

## 8. 版本與相容性

### 8.1 `AudioPipelineMemReq`

固定 32 bytes、每個欄位固定 byte offset（由 header 內的 `_Static_assert` 釘死），
全部是定寬整數，沒有指標、沒有 `size_t`。

| Offset | 欄位 | 型別 | 目前值 |
|---:|---|---|---|
| 0 | `descriptor_version` | `uint32_t` | `2` |
| 4 | `layout_version` | `uint32_t` | `6` |
| 8 | `backend_id` | `uint32_t` | `1` = KISS，`2` = NE10（永遠不會是 0） |
| 12 | `build_flags_hash` | `uint32_t` | FNV-1a-32，隨 build 變動 |
| 16 | `alignment` | `uint32_t` | `16` |
| 20 | `reserved` | `uint32_t` | `0`（必須為 0，`init_ex()` 會驗證） |
| 24 | `bytes` | `uint64_t` | pool 總需求 |

### 8.2 序列化

這個 struct 可以逐 byte 拷貝到檔案 / flash / 訊息 buffer 再讀回來，
**但僅限相同 endianness**。本 library 不提供任何 byte-swap 輔助；
跨 endianness 交換明確不在支援範圍，需要的話請自己在序列化邊界處理。

### 8.3 各版本欄位什麼時候會變

* `descriptor_version`：只在 `AudioPipelineMemReq` 自己的欄位集合 / 順序 / 寬度改變時遞增。
* `layout_version`：在 `audio_pipeline.c` 自己的 carve 順序、buffer 集合或單一 buffer
  sizing 公式改變時遞增。AEC / NR / FFT backend 各自內部佈局的改變**不會**動這個版本 ——
  那些改變本來就會改到 `bytes`，會被 pool 過小的檢查抓到。
* `build_flags_hash`：涵蓋 backend 身分、本檔案自己 7 個 scratch buffer 的 carve 順序、
  以及對齊粒度。**不涵蓋** `AecConfig` / `MmseLsaConfig` 的 preset 與 tunable 數值
  （那些是 config 不是 layout，且已經反映在 `bytes` 上）、也不涵蓋 compiler / ABI / toolchain。

### 8.4 相容性規則（給整合者的三條）

1. **descriptor 永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用。
2. **兩個 backend 之間永不互通**。即使 `bytes` 剛好相同，KISS 與 NE10 也不是
   byte-identical 的，descriptor 不能互相代用。
3. **升級 library 之後一定要重跑一次 `get_mem_requirements()`**，
   並確認你的 pool 預算仍然足夠。
