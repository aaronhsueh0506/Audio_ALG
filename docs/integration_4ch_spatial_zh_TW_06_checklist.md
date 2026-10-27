> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md) ｜ [下一份](integration_4ch_spatial_zh_TW_07_versioning.md) <!-- split-nav -->
## 7. 整合檢查清單

建置與連結：

- [ ] `libaudio_pipeline_mcab.a` / `libdoa.a` / `libbf.a` /
      `libspatial_common.a` / `libaec.a` / `libanr.a` / `libaudio_common.a` 與你編的 `mcab_sample_beam.o`、`mcab_sample_flow.o`
      全部是用**同一組** backend、`SIMD` 與編譯選項建出來的
- [ ] link 順序照 [§1.5](integration_4ch_spatial_zh_TW.md#15-archive-link-order)
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用 `print-lib-path` 查的
- [ ] 編譯 `mcab.c` 的 TU 有 `-DAUDIO_PIPELINE_BACKEND_STR="ne10"`（否則 `mcab_get_mem_requirements()` 回 `-1`）
- [ ] 編譯 helper 的 `-I` 清單含 `<audiolib>/doa`、`<audiolib>/bf`（`mcab_sample_beam.h` 會 include `srp.h`／`bf.h`）；
      你自己的程式碼沒有直接呼叫 `srp_*`／`bf_*`
- [ ] 沒有 include `mcab_internal.h`
- [ ] 開機 log 有記錄 `mcab_sample_beam_spatial_backend()`

Config：

- [ ] `sample_rate` 是 16000 或 48000（**不是** 8000）
- [ ] `geometry` 與對應的幾何欄位配對正確
      （UCA → `uca_radius_m`；ULA → `ula_spacing_m`；CUSTOM → 兩個座標陣列）
- [ ] 用 UCA/ULA 時，你知道 `microphone_x_m` / `microphone_y_m` **會被忽略**
- [ ] `num_angles` 是依實際需要的角度解析度設的，不是隨手加大的（記憶體影響很大）
- [ ] `doa_low_freq_hz < Nyquist`、`doa_high_freq_hz <= Nyquist`，且 `doa_low_freq_hz` 小於解析後的上限（`doa_high_freq_hz = 0` 時為 `min(7000, Nyquist)`）
- [ ] `doa_switch_consecutive`、`doa_update_interval` 都 `> 0`；`bf_adapt_interval` 在 `[1, MCAB_SAMPLE_BEAM_MAX_ADAPT_INTERVAL]`
- [ ] `bf_lambda` 落在 `(0, 1]`
- [ ] `auto_vad_snr_ratio >= 1.0`
- [ ] 四條 lane 的 `AecConfig` 都經過同一次 `mcab_apply_lane_contract()`（或直接用 `mcab_lane_config()`），lane 的 `sample_rate` 與 flow 相同

生效值（[§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值)）：

- [ ] 你知道 `bf_lambda` / `bf_adapt_interval` / `auto_vad_hangover_frames`
      三者的**設定值 ≠ 生效值**
- [ ] init 之後有讀 `mcab_sample_beam_bf_effective_adapt_interval()` 與
      `mcab_sample_beam_bf_lambda()`，並確認是你預期的值
- [ ] 換 grid（改 `sample_rate` 或 `fft_size`）之後，有重新確認上面兩個生效值
- [ ] 若使用內建 VAD，`auto_vad_hangover_frames` 的生效幀數是自己用公式算過的
      （**沒有 read-back accessor**）

初始化：

- [ ] 每次 init 之前都重新呼叫 `mcab_get_mem_requirements()`、`aec_get_mem_size()`、`delay_aec_get_mem_size()` 與 `mcab_sample_beam_get_mem_size()`
- [ ] 每塊 pool 對齊 16、大小 ≥ 查到的 bytes
- [ ] 四條 lane 是**四個不同**的 `aec_init()` 物件，且是新建的（或已 reset）；MATCHED 才傳 estimator，其他模式 `McabLibs.delay` 為 NULL
- [ ] `DelayAec` 結構本身與它的 pool 一樣活到 `mcab_destroy()` 之後
- [ ] 用 `mcab_init_ex()` 並把剛查到的 `req` 當 `expected` 傳回去
- [ ] `mcab_init_ex()` 或 `mcab_sample_beam_init()` 回 `NULL` 時你的程式會停下來 —— 而且你**自己**在呼叫端印出了
      是哪一個欄位不符（函式庫不會告訴你）
- [ ] flow 掛著的期間，沒有自己對這些 lane／estimator 呼叫處理或 reset 函式

執行期：

- [ ] 有自己 VAD 的話用的是 `mcab_sample_beam_process_with_activity()`，不是 `mcab_sample_beam_process()`
- [ ] `vad_raw` / `vad_out` 只傳 `0` 或 `1`
- [ ] `frequency_mask` 若非 NULL，長度至少 `n_freqs` 個 `int`
- [ ] `microphones_interleaved` 是 `[sample * 4 + channel]` interleaved，長度 `hop * 4`
- [ ] mic / ref 內容都是有限值
- [ ] hop / n_freqs 是查來的，不是寫死的
- [ ] 有處理 `-3`（flow 已被自動 reset，你的後級——RES、NR／合成的 reset 與 CNG rng、NN stage——與外部狀態也要跟著清）
- [ ] 任何呼叫被拒、`aec_res_process()` 或 `mcab_sample_anr_gain()` 回 `NULL` 或輸出非有限時，flow／helper 與後級（含 `aec_res_reset()`）是**一起** reset 的
- [ ] 每一個後級 stage 都各自查大小（`aec_res_get_mem_size()`、`anr_get_mem_size()`、`ac_istft_get_mem_size()` 等），與 flow、helper 的 pool 分開配置；`AecRes` 的 `fft_size` 取自 `mcab_fft_size(flow)`
- [ ] 用 `isnan()` 判斷 `mcab_sample_beam_bf_lambda()` 的失敗，不是拿它跟數值比較
- [ ] 同一個 instance 沒有被多 thread 同時處理

收尾：

- [ ] 拆除順序是 helper → `mcab_destroy()` → 四條 lane（`aec_destroy()`）→ 釋放記憶體
- [ ] `destroy()` 在釋放 pool **之前**
- [ ] `mcab_create()` 建的 instance 只 `destroy()` 一次
- [ ] destroy 之後沒有再呼叫任何 accessor

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md) ｜ [下一份](integration_4ch_spatial_zh_TW_07_versioning.md) <!-- split-nav -->
