Navigation: [Index](README.md) | [Previous part](README_02_build_debug_board.md) <!-- split-nav -->
## Tunable Parameters

External means a third party may tune it; internal means it exists and is tunable in the code but is
not recommended to open (where it lives and why is listed in section 0.3 of
[`../docs/integration_mono_zh_TW.md`](../docs/integration_mono_zh_TW_01_overview_link_quickstart.md#03-參數面外部與內部)). The tables below list
external parameters first and mark the internal constants as such.

### AEC (`AecConfig`, see `aec.h`)

**Presets**: `AEC_PRESET_MILD` / `AEC_PRESET_BALANCED`（default）/ `AEC_PRESET_AGGRESSIVE` / `AEC_PRESET_NO_FLOOR`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | 8000 / 16000 / 48000，自動計算 frame/fft/hop |
| `filter_length` | sr×52ms（sr≥44100 時 sr×64ms） | 自適應濾波器長度（416@8k, 832@16k／52ms, 3072@48k／64ms；`aec_config_defaults()` 的公式，不是固定 32ms） |
| `enable_highpass` | 1 | 高通濾波器（移除 DC + 低頻） |
| `highpass_cutoff_hz` | 80.0 | HPF 截止頻率 (Hz) |

**RES / preset strength axis**：RES has no standalone `ResConfig` — it lives inside `AecConfig`
and the AEC3 post-filter chain (`SuppressionGain`, `ResidualEchoEstimator`, etc.), surfaced
externally through the `AecResContext` seam (see [`## Modules`](README.md#modules)). The four AEC presets
differ in exactly three fields:

| Parameter | Mild | Balanced | Aggressive | No-floor | Description |
|-----------|------|----------|------------|----------|-------------|
| `min_gain_floor_far_active_db` | -20 | -28 | -38 | -28（inert） | AEC3 `SuppressionGain` 遠端活躍時的最低增益下限 dB（最大抑制量） |
| `min_gain_floor_dt_db` | -20 | -20 | -30 | -20（inert） | double-talk 期間套用的地板 dB；mild/balanced 停在 dataclass 共用預設 -20，只有 aggressive 額外下探到 -30（其餘欄位：filter length、Kalman Q、delay buffer、CNG…四個 preset 皆相同） |
| `min_gain_split_floor_enabled` | true | true | true | **false** | 地板機制本身的開關；`AEC_PRESET_NO_FLOOR` 把它關掉，此時上面兩個 dB 欄位不生效（AEC3 原生 min_gain 不被任何東西往上拉），抑制可以深到 AEC3 自己算出的極限 |

### Runtime strength control（不重建、不動 pool）

兩條強度軸都可以在**運轉中**改。呼叫時機一律是**兩個 hop 之間**、與 process 序列化；
**非 thread-safe**。全部回 `0` 或 `-1`，`-1` 時什麼都不寫。強度 setter 屬於各函式庫，
不屬於任何 pipeline：

| Entry point | 目標 |
|---|---|
| `aec_set_preset(aec, preset, ramp_ms)` | mono：`Aec` 的 far-active 地板（mono 的 RES 增益 `ctx.res_gain` 就在 AEC 裡，經 `se_frame_from_aec()` 成為 `g_res_ctx`） |
| `aec_res_set_preset(res, preset, ramp_ms)` | 4ch：應用自備的後波束 `AecRes`（單元與核心都不持有；`res` 為 `NULL` 時回 `-1`） |
| `aec_res_split_floor(res, &live, &target)` | 4ch：唯讀，線性功率。`live == target` 代表 ramp 已走完 |
| `anr_set_mode(nr, mode)` | 範例自己初始化的**那一個**降噪器（`anr_init()` 回傳的 `AnrDenoiser*`）；`NULL` 回 `-1` |

4ch 的初始 RES 強度由應用傳給 `aec_config_from_preset()` 的 preset 決定，執行中改強度走上面的
`aec_res_set_preset()`；`aec_res_reset()` 不會把它改回建構時的 preset。單元 `AudioPipeline4Ch` 與核心
`FourAecNrRes` 都沒有強度 setter：它們的 lane 只算線性殘留，RES 與 NR 不在裡面。

`ramp_ms == 0` 代表下一個 hop 套用（**不是錯誤**），落點與「用該 preset 從頭建一個新
實例」完全相同；`> 0` 則以 dB 為單位線性走過去，上限 60 秒（mild ↔ aggressive 是
18 dB 落差、地板又是硬性 clamp，互動式旋鈕應該給一個 ramp）。ramp 途中再呼叫會從當前
live 值重新起走。兩者都**不是重啟**：濾波器、延遲鎖定、噪聲底與增益平滑歷史全部繼續
跑；要重啟請用對應的 `_reset()`。

`AEC_PRESET_NO_FLOOR` 也可以在**已經在跑的** instance 上用同一個 setter 切換，不必重建：
`min_gain_split_floor_enabled` 這個 gate 會跟兩個 dB 欄位同一次呼叫一起 retarget（沒有
ramp，立即生效，跟 DT 地板一樣是硬切），所以不會有一個 hop 是「preset 標籤已經是
NO_FLOOR，但 gate 還沒跟上」的中間狀態。Python 對應：mono 的 `AEC.set_preset()`；4ch 的
`PostBeamResidualSuppressor.set_preset()`（Python 的 `FourChannelAecPipeline` 不持有它：應用自己建一個，
`PostBeamResidualSuppressor.from_aec_config(config)` 或 `PostBeamResidualSuppressor(sr, hop, n_freqs, preset)`，
每 hop 呼叫 `process(post.context, post.aligned_render)`，`reset()` 自己重置）。

> **4ch：對四條 lane 重新指定 preset 是無效操作。** 四條 lane 都以
> `spatial_linear_context` 建立，從不走到自己的殘留抑制器，它們的地板
> 什麼都不塑形。真正乘上輸出的 gain 來自應用自備的**唯一一個**後波束 `AecRes`，所以強度要設在它身上
> （由 `4ch_aec_bf_nr_res/tests/test_audio_pipeline_4ch.c` 的 `test_caller_res_reset_and_preset` 釘住）。

> **NR：範例的降噪器就是函式庫的 preset。** 它以 `anr_config_for_mode_grid()` 建立、
> 沒有任何覆寫，所以 `anr_set_mode()` 直接換強度（`tests/test_se_common.c` 的 `test_nr_set_mode` 釘住）。要改 preset 以外的欄位時，自己組好
> 完整 target 再呼叫 `anr_reconfigure()`（grid、`L`、`num_init_frames` 不可變）。

> **A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
> hop 生效：double-talk 期間套的是 DT 地板（mild/balanced 共用 -20 dB，aggressive 是
> -30 dB，見上表 `min_gain_floor_dt_db`）；far-active latch 觸發前套的是 far-silent 地板。同一個 `G_res` 還決定注入的 comfort
> noise 量（振幅正比於 `sqrt(1 - G_res^2)`，再乘該 bin 的 `G_nr`（下限 −10 dB），見兩處實作的 CNG
> 步驟——地板越深、CNG 反而越多）。所以**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且只量
> echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制。請在 echo 對齊或
> degradation 對齊的條件下比較，並實際試聽。

兩條 Align-ULCNet 範例（`mono_alignulcnet/`、`4ch_alignulcnet/`）**刻意不提供**
這些 setter：mono 變體沒有 NR 實例、且算出的 `G_res` 沒有任何消費者（輸出由 ULCNet
mask 塑形）；4ch 變體用的是 `core.enable_post = 0` 的單元，抑制器與降噪器
根本不存在。強度是模型的性質，不是 runtime 旋鈕。

### NR (`AnrConfig`, see `anr_types.h`)

**Modes**: `ANR_MODE_MILD` / `ANR_MODE_MODERATE` / `ANR_MODE_BALANCED`（default）/ `ANR_MODE_AGGRESSIVE`

> Both C pipeline CLIs' `parse_nr_mode()` — `mono_aec_nr_res/main.c` and
> `4ch_aec_bf_nr_res/4aec_nr_res_static.c` — accept
> `--nr-preset mild|moderate|balanced|aggressive`; anything else falls back to
> `ANR_MODE_BALANCED` with no error. The Python `aec_nr_pipeline.py` CLI accepts
> `mild|balanced|aggressive` only.

`g_min_db` is amplitude dB (`g_min = 10^(g_min_db/20)`).

> **Retime basis.** `anr_config_for_mode_grid()` retimes every `alpha_*`, `L`,
> `num_init_frames` and `scene_change_min_frames` from its authored reference
> duration to the actual hop of the grid (`anr_retime_alpha()`,
> `anr_retime_alpha_ref()`, `anr_retime_frames()`, `anr_retime_frames_ms()`);
> `q`, `g_min_db` and `xi_min_db` are plain constants. The reference is 16 ms/hop
> for `alpha_xi`, `alpha_attack`, `alpha_d_speech` and `L`, and 10 ms/hop for the
> other `alpha_*` and the frame counts. The tables list the authored constants,
> not the realized value at a grid; read that back from `anr_default_config()` /
> `anr_config_for_mode()` (or Python `core/nr_strength.py`) instead of scaling by hand.

The strength presets change only these four fields:

| Parameter | Mild | Moderate | Balanced | Aggressive | Description |
|-----------|------|----------|----------|------------|-------------|
| `g_min_db` | -20 | -23 | -25 | -28 | 最小增益 dB（最大抑制量，amplitude dB, /20） |
| `q` | 0.58 | 0.54 | 0.52 | 0.45 | 語音先驗機率（低→積極抑噪） |
| `xi_min_db` | -10 | -10 | -10 | -12 | 先驗 SNR 下限 dB |
| `noise_over_subtraction` | 1.2 | 1.3 | 1.4 | 1.3 | 噪聲過減係數 |

All modes share these tracker and gain-smoothing constants (internal: jointly tuned, not
recommended to open):

| Parameter | Authored value | Basis | Description |
|-----------|----------------|-------|-------------|
| `alpha_xi` | 0.92 | 16 ms | Decision Directed 先驗 SNR 平滑（SPP） |
| `alpha_attack` | 0.15 | 16 ms | Attack 平滑（語音起始追蹤） |
| `alpha_g` | 0.88 | 10 ms | 增益路徑上沒有作用（`anr_types.h` 註明 unused；實際作用的是 `alpha_attack` / `alpha_decay`） |
| `alpha_decay` | 0.88 | 10 ms | Decay 平滑（噪聲釋放） |
| `alpha_s` | 0.95 | 10 ms | 功率譜時間平滑（MCRA） |
| `alpha_d` | 0.903414 | 10 ms | 噪聲更新速率（= 0.85 @ 16 ms） |
| `L` | 32 frames | 16 ms | 最小值追蹤視窗；16 kHz 預設 256/128 grid（hop 8 ms）的實際值為 64 |
| `num_init_frames` | 20 frames | 10 ms | 初始化靜默幀數；同一 grid 的實際值為 25（約 200 ms） |
| `scene_change_threshold_db` | 10.0 | 純 dB 常數 | 場景轉換偵測閾值，不 retime |

> 各範例 `main` 直接用上表的函式庫預設（`anr_config_for_mode_grid()`，
> `broadband_threshold = 1.0`），不另外覆寫。

---

## Troubleshooting & Tuning Guide

Remedies are grouped by who may apply them. **External** remedies use the parameters and setters
listed under "Tunable Parameters"; **internal** remedies change constants that were tuned jointly and
validated on three grids (16k/256, 16k/512, 48k/1024), so apply one only together with that
re-validation.

### External remedies (AEC, NR, pipeline)

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **殘留回聲明顯** | RES 抑制不足 | `aec_set_preset()` 改用 aggressive（給 `ramp_ms`，例如 100） |
| **殘留回聲 + 遠端持續講話** | Filter 未完全收斂或尾巴不夠長 | 增加 `filter_length`（如 832→1536，init 參數，要重查 pool），確認 mic-ref delay 落在 `delay_mode` 的搜尋範圍 |
| **近端語音被壓制（DT degradation）** | RES 過度抑制 | `aec_set_preset()` 改用 mild |
| **輸出底噪不自然（突然靜音）** | CNG 未開啟 | 確認 `enable_cng=1`（preset 預設已開），範例沒有帶 `--no-cng` |
| **噪聲殘留太多** | 抑制量不夠 | `anr_set_mode()` 往 aggressive 走（`g_min_db` -28 dB、`q` 0.45） |
| **語音被吃掉** | NR 抑制太激進 | `anr_set_mode()` 往 mild 走（`g_min_db` -20 dB、`q` 0.58） |
| **回聲消了但底噪變大** | NR 沒開或太保守 | 確認 NR mode 非 MILD，且範例沒有帶 `--no-nr` |
| **整體語音品質差（悶、失真）** | 多階段過度處理 | AEC 與 NR 都改 MILD，只在必要時加強個別模組；用 `--no-nr` / `--no-res` 隔離是哪個增益來源 |
| **處理 48kHz 音訊記憶體不足** | 模組記憶體隨 fft_size 增長 | 縮短 `filter_length`（見 [Memory Budget](README_01_latency_call_flow.md#memory-budget) 的說明） |
| **NR 把回聲當噪聲學進去** | AEC 殘留回聲被 MCRA 當底噪 | 先確保 AEC 收斂良好再調 NR；範例預設把 R² 折入 NR 的噪聲底（不要帶 `--legacy-amin`） |

### Internal remedies (not recommended to open)

| 症狀 | 內部欄位（位置） | 風險 |
|------|------|------|
| 收斂太慢 | AEC `kalman_q_high`、`warmup_frames`（`aec.h`） | `warmup_frames` 以 10 ms 為單位並在建立時 retime；Q 決定收斂／發散邊界 |
| Filter 發散（輸出爆音） | AEC `kalman_q_high`、`mu`（`aec.h`） | 驗證器範圍很寬，範圍內多數值不收斂 |
| Echo path 變化後適應慢 | AEC `shadow_mu_nlms`、`shadow_err_alpha`（`aec.h`） | 與 `epc_*` 偵測器聯合調過 |
| Musical noise | NR `alpha_decay`、`alpha_xi`（`anr_types.h`） | 與增益平滑聯合調過；`alpha_g` 在增益路徑上沒有作用，調它無效 |
| 語音起始被截斷 | NR `alpha_attack`（`anr_types.h`） | 與 `alpha_decay` 成對 |
| 噪聲環境切換後適應慢 | NR `L`（pool 版面欄位）、`scene_change_*` | `L` 要重新 `anr_get_mem_size()` 並重建實例；`anr_set_mode()` 不能改它 |
| 初始化期語音被壓 | NR `num_init_frames` | 開頭那段時間 pass-through，且決定 pool 版面 |
| 穩態噪聲殘留（風扇聲） | NR `alpha_d`（噪聲更新速率） | MCRA 追蹤核心常數 |
| 語音段噪聲估計上升 | NR `alpha_xi`（SPP 平滑） | 同上 |

### Verification

Both versions build from the default `make`; `tests/test_mono_chain.c` checks both programs against the
hand-written oracle bit for bit (so the pooled and the heap outputs are equal for the tested option sets). The
static build derives the AEC, denoiser and synthesis configs from one `SeGrid`, so the three share
one grid by construction (`--print-mem-size` prints the grid and runs a real init of every stage).

Navigation: [Index](README.md) | [Previous part](README_02_build_debug_board.md) <!-- split-nav -->
