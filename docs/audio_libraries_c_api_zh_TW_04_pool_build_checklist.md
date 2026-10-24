[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md) <!-- split-nav -->
## 7. Static-pool 共通規則

三個 library 都遵守相同模式：

```text
config
  -> get_mem_size()/get_mem_requirements()
  -> caller 配置至少 16-byte aligned 的 pool
  -> init(pool, bytes, config)
  -> process...（不可 malloc）
  -> reset（需要時）
  -> destroy（不釋放 caller pool）
  -> caller 釋放 pool
```

規則：

- pool 大小必須由**同一份 resolved config**查詢；查完後不可再改 grid/preset/delay 欄位。
- 不要共用同一塊可寫 pool 給兩個 live instance。
- `_init()` 回傳 NULL 就停止，不可退回較小尺寸或默默改 config。
- static path 的 destroy 是 no-op 或只做狀態清理；pool ownership 永遠在 caller。
- 沒有「整條 pipeline」的 aggregate sizing API：mono 的每個函式庫各自查大小、整合者加總（[§6.10](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md#610-各函式庫各自初始化記憶體加總與單元的公開面)）；4ch 單元的
  `audio_pipeline_4ch_get_mem_requirements()` 只含單元自己，RES（`AecRes`）、NR、合成與 NN stage 另計。

---

## 8. 建置、SIMD 與連結

從 `SE/` 根目錄做參考建置（`libaudio_common.a` 在 `audiolib/common`；`audiolib` 根目錄的 `lib` 另建 `libspatial_common.a`、
`libdoa.a`、`libbf.a`、`libvad.a`，只有 4ch 單元需要）：

```bash
make -C audiolib/common lib BACKEND=ne10 SIMD=0
make -C AEC/c_impl      lib BACKEND=ne10 SIMD=0
make -C NR/c_impl       lib BACKEND=ne10 SIMD=0
```

ARM NEON 產品建置使用相同旗標：

```bash
make -C audiolib/common lib BACKEND=ne10 SIMD=1
make -C AEC/c_impl      lib BACKEND=ne10 SIMD=1
make -C NR/c_impl       lib BACKEND=ne10 SIMD=1
```

不要混用不同 `BACKEND`／`SIMD` 產物。archive 路徑由 build flags hash 決定，使用各 repo
的 `make -s ... print-lib-path` 取得，不要自行拼 `bin/` 路徑。

模型 pre/post 類別（[§6](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md#6-ai-模型-prepost-類別)）另有兩個 archive，同樣是 configuration-keyed：

```bash
make -C AIAEC lib ; make -C AIAEC -s print-lib-path   # libaiaec_prepost.a
make -C AINR  lib ; make -C AINR  -s print-lib-path   # libainr_prepost.a
```

兩者都落在自己 repo 的 `build/simd<N>-<sig>/` 下（`<sig>` 是各自 Makefile 的組態簽章 cksum：
AIAEC 蓋 `CC`/`AR`/`CFLAGS`/`CPPFLAGS`/`SIMD`/`WERROR`，AINR 蓋 `CC`/`AR`/`CFLAGS`/`LDFLAGS`/
`SIMD`/`SIMD_CPPFLAGS`/`BACKEND`/`WERROR`/`COMMON_LIB`；每個目錄另存 `config.manifest` 擋簽章碰撞），
所以路徑一律用 `print-lib-path` 取得。成員：`libaiaec_prepost.a` = `ulcnet_process`、
`ulcnet_model_io`、`ulcnet_prepost`、`ulcnet_accelerator_adapter`、`aiaec_process`、
`deepvqe_process`、`deepvqe_prepost`；`libainr_prepost.a` = `dfn2_process`、`dfn2_model_io`、
`dfn2_prepost`、`dfn3_process`、`dfn3_model_io`、`dfn3_prepost`、`gtcrn_process`。RNNoise-ERB **不在**任何 archive 裡（自己的 Makefile，
要用的板子把它的來源檔直接加進自己的 build）。`WERROR=1` 是 opt-in，且已納入組態簽章。

消費端至少要帶：

```text
-std=gnu99
-ffp-contract=off
-fno-math-errno
-I<AEC>/c_impl/include
-I<NR>/c_impl/include
-I<audiolib>/common/include
<libaec.a> <libanr.a> <libaudio_common.a> -lm
```

`SIMD_KERNELS_FORCE_SCALAR` 必須在 library 與 include header-only kernels 的 application TU
保持一致。不要使用 `-Ofast`／`-ffast-math`；它們會改變遞迴 DSP 數值與 parity。

---

## 9. 最小整合檢查清單

- [ ] sample rate、frame/FFT、hop 是同一個合法 grid。
- [ ] mic/ref 同 sample rate、同 clock；AEC ref 是實際 speaker render reference。
- [ ] 每次 AEC 恰好傳一個 hop。
- [ ] NR 前後使用 periodic sqrt-Hann 50% WOLA，且 frame 等於 FFT。
- [ ] 所有 pool 以 resolved config 查 size，base 至少 16-byte aligned。
- [ ] process hot path 沒有 malloc/free/file I/O/logging。
- [ ] stream boundary 同時 reset DSP instance 與 caller-owned ring/WOLA state。
- [ ] borrowed context/gain pointer 沒有跨下一個 process 呼叫保存。
- [ ] application 與 static libraries 使用相同 backend、SIMD、FP flags。
- [ ] 最終 float-to-PCM 前做 saturating conversion；需要 limiter 時只放在整條 chain 最後。
- [ ] 完整 AEC+NR/4ch 應用優先跑 pipeline tests，而不只跑各 library unit test。

[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md) <!-- split-nav -->
