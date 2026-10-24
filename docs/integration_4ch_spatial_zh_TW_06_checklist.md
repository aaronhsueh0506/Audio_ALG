> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md) ｜ [下一份](integration_4ch_spatial_zh_TW_07_versioning.md) <!-- split-nav -->
## 7. 整合檢查清單

建置與連結：

- [ ] 七個 archive（`libaudio_pipeline_4ch.a` / `libdoa.a` / `libbf.a` /
      `libspatial_common.a` / `libaec.a` / `libanr.a` / `libaudio_common.a`）
      全部是用**同一組** backend、`SIMD` 與編譯選項建出來的
- [ ] link 順序照 [§1.5](integration_4ch_spatial_zh_TW.md#15-archive-link-order)
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用 `print-lib-path` 查的
- [ ] 沒有 include `bf.h` / `srp.h` / 任何 `audiolib` header
- [ ] 沒有 include `4aec_nr_res_internal.h` 或 `4aec_projection_kernels.h`
- [ ] 開機 log 有記錄 `audio_pipeline_4ch_spatial_backend()`

Config：

- [ ] `core.sample_rate` 是 16000 或 48000（**不是** 8000）
- [ ] `geometry` 與對應的幾何欄位配對正確
      （UCA → `uca_radius_m`；ULA → `ula_spacing_m`；CUSTOM → 兩個座標陣列）
- [ ] 用 UCA/ULA 時，你知道 `microphone_x_m` / `microphone_y_m` **會被忽略**
- [ ] `num_angles` 是依實際需要的角度解析度設的，不是隨手加大的（記憶體影響很大）
- [ ] `doa_low_freq_hz < Nyquist`、`doa_high_freq_hz <= Nyquist`
- [ ] `doa_switch_consecutive`、`doa_update_interval`、`bf_adapt_interval` 都 `> 0`
- [ ] `bf_lambda` 落在 `(0, 1]`
- [ ] `auto_vad_snr_ratio >= 1.0`

生效值（[§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值)）：

- [ ] 你知道 `bf_lambda` / `bf_adapt_interval` / `auto_vad_hangover_frames`
      三者的**設定值 ≠ 生效值**
- [ ] init 之後有讀 `audio_pipeline_4ch_bf_effective_adapt_interval()` 與
      `audio_pipeline_4ch_bf_lambda()`，並確認是你預期的值
- [ ] 換 grid（改 `sample_rate` 或 `fft_size`）之後，有重新確認上面兩個生效值
- [ ] 若使用內建 VAD，`auto_vad_hangover_frames` 的生效幀數是自己用公式算過的
      （**沒有 read-back accessor**）

初始化：

- [ ] 每次 init 之前都重新呼叫 `audio_pipeline_4ch_get_mem_requirements()`
- [ ] pool 對齊 16、大小 ≥ `req.bytes`
- [ ] 用 `init_ex()` 並把剛查到的 `req` 當 `expected` 傳回去
- [ ] `init_ex()` 回 `NULL` 時你的程式會停下來 —— 而且你**自己**在呼叫端印出了
      是哪一個欄位不符（函式庫不會告訴你）

執行期：

- [ ] 有自己 VAD 的話用的是 `process_with_activity()`，不是 `process()`
- [ ] `vad_raw` / `vad_out` 只傳 `0` 或 `1`
- [ ] `frequency_mask` 若非 NULL，長度至少 `n_freqs` 個 `int`
- [ ] `microphones_interleaved` 是 `[sample * 4 + channel]` interleaved，長度 `hop * 4`
- [ ] mic / ref 內容都是有限值
- [ ] hop / n_freqs 是查來的，不是寫死的
- [ ] 有處理 `-3`（整個單元已被自動 reset，你的後級——RES、NR／合成的 reset 與 CNG rng、NN stage——與外部狀態也要跟著清）
- [ ] 任何呼叫被拒、`se_res_stage_gain()` 或 `se_anr_stage_gain()` 回 `NULL` 或輸出非有限時，單元與後級（含 `aec_res_reset()`）是**一起** reset 的
- [ ] 每一個後級 stage 都各自查大小（`aec_res_get_mem_size()`、`anr_get_mem_size()`、`ac_istft_get_mem_size()` 等），與單元 pool 分開配置；`AecRes` 的 `fft_size` 取自 `audio_pipeline_4ch_fft_size(p)`
- [ ] 用 `isnan()` 判斷 `audio_pipeline_4ch_bf_lambda()` 的失敗，不是拿它跟數值比較
- [ ] 同一個 instance 沒有被多 thread 同時處理

收尾：

- [ ] `destroy()` 在釋放 pool **之前**
- [ ] `create()` 建的 instance 只 `destroy()` 一次
- [ ] destroy 之後沒有再呼叫任何 accessor

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md) ｜ [下一份](integration_4ch_spatial_zh_TW_07_versioning.md) <!-- split-nav -->
