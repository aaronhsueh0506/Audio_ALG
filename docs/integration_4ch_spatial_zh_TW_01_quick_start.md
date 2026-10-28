> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW.md) ｜ [下一份](integration_4ch_spatial_zh_TW_02_lifecycle.md) <!-- split-nav -->
## 2. Quick start

最小整合（heap 路徑；四條 lane、共用 estimator 與 mcab 由 `mcab_sample_flow_create()` 在同一塊記憶體建立，
SRP/BF helper 另一塊，RES、NR 與合成由應用自己初始化）。下列程式照下方編譯步驟建置並執行過，
輸出 `ok hop=128 effective_adapt_interval=1 lambda=0.995998025`（與 [§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值) 的第一列相同）：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mcab.h"
#include "mcab_sample_flow.h"
#include "mcab_sample_beam.h"
#include "anr.h"
#include "audio_stft.h"
#include "aec_res.h"
#include "mcab_sample_post.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "utility/simd_kernel_nn.h"   /* skn_all_finite_f32 */

int main(void) {
    McabConfig cfg = mcab_default_config(16000);          /* flow parameters only */
    McabSampleBeamParams bp = mcab_sample_beam_default_params();
    bp.geometry     = MCAB_SAMPLE_GEOMETRY_UCA;
    bp.uca_radius_m = 0.035f;                             /* 換成你陣列的真實半徑 */

    /* The application side of mcab.h: four lanes, the shared estimator and mcab in one heap block. */
    McabSampleFlow flow;
    if (mcab_sample_flow_create(&flow, &cfg) != 0) { fprintf(stderr, "flow init failed\n"); return 1; }
    /* The sample SRP-PHAT/BF helper, its own pool. */
    size_t beam_bytes;
    void* beam_mem = NULL;
    if (mcab_sample_beam_get_mem_size(&cfg, &bp, &beam_bytes) != 0 ||
        posix_memalign(&beam_mem, 16, beam_bytes) != 0) return 1;
    McabSampleBeam* beam = mcab_sample_beam_init(beam_mem, beam_bytes, &cfg, &bp, flow.flow);
    if (!beam) { fprintf(stderr, "helper init failed\n"); return 1; }

    /* RES (libaec AecRes), NR and synthesis are separate libraries the application sizes and
     * initialises itself; neither mcab nor the helper owns any of them. */
    const int hop = mcab_hop_size(flow.flow);
    const int n = mcab_n_freqs(flow.flow);
    const int fft = mcab_fft_size(flow.flow);
    AecConfig res_cfg;
    aec_config_from_level(&res_cfg, AEC_LEVEL_BALANCED, 16000);
    res_cfg.fft_size = fft;                                  /* the flow's grid */
    AnrConfig nr_cfg = anr_config_for_level_grid(16000, fft, ANR_LEVEL_BALANCED);
    AcStftConfig synth_cfg;
    ac_stft_config_default(&synth_cfg, fft);
    size_t res_bytes = aec_res_get_mem_size(&res_cfg);       /* 0 = invalid */
    size_t nr_bytes = anr_get_mem_size(&nr_cfg);
    size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);
    size_t vec = ALIGN16((size_t)n * sizeof(float));
    size_t total = ALIGN16(res_bytes) + ALIGN16(nr_bytes) + ALIGN16(synth_bytes) + 2 * vec +
                   ALIGN16((size_t)n * sizeof(Complex));
    void* post_pool;
    if (res_bytes == 0 || nr_bytes == 0 || synth_bytes == 0 ||
        posix_memalign(&post_pool, 16, total) != 0) return 1;
    uint8_t* cur = (uint8_t*)post_pool;
    AecRes* res = aec_res_init(cur, res_bytes, &res_cfg);    cur += ALIGN16(res_bytes);
    AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);      cur += ALIGN16(nr_bytes);
    AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg); cur += ALIGN16(synth_bytes);
    if (!res || !nr || !istft) return 1;
    float* total_gain = (float*)cur;  cur += vec;
    float* extra_noise = (float*)cur; cur += vec;
    Complex* spec = (Complex*)cur;
    uint32_t rng = MCAB_SAMPLE_CNG_RNG_SEED;

    float mics[4 * 512], ref[512], out[512];                 /* 512 = the largest hop (48 kHz) */
    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < hop; i++) {                      /* 換成你自己的擷取來源 */
            for (int ch = 0; ch < MCAB_CHANNELS; ch++)
                mics[i * MCAB_CHANNELS + ch] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }

        McabSampleBeamResult bres;
        McabSampleBeamInfo info;
        McabFrame f;
        const float* g_res = NULL;
        const float* g_nr = NULL;
        int status = mcab_sample_beam_process(beam, mics, ref, &bres, &info);  /* lanes + DOA + BF */
        if (status == MCAB_OK)
            status = mcab_sample_beam_fuse(beam, &f);         /* McabFrame; f.g_res_ctx is NULL */
        if (status == MCAB_OK) {
            AecResInput rin;
            mcab_sample_res_input(&f, &rin);
            g_res = aec_res_process(res, &rin);                /* RES gain from your AecRes */
        }
        if (g_res)
            g_nr = mcab_sample_anr_gain(nr, &f, extra_noise, 1);
        if (g_nr) {
            sk_min_f32(total_gain, g_nr, g_res, n);
            sk_capply_gain_f32(spec, f.b, total_gain, n);   /* 估計用 f.e，合成用 f.b */
            if (f.cn) aec_cng_fill(&rng, spec, f.cn, g_res, g_nr, MCAB_SAMPLE_CNG_NR_GAIN_FLOOR, n);
            (void)ac_istft_push(istft, spec, out);
        }
        if (!g_nr || !skn_all_finite_f32(out, (size_t)hop)) {
            fprintf(stderr, "status=%d\n", status);
            mcab_sample_beam_reset(beam);                    /* a refused call: reset every stage */
            aec_res_reset(res);
            anr_reset(nr);
            ac_istft_reset(istft);
            rng = MCAB_SAMPLE_CNG_RNG_SEED;
            continue;
        }
        /* 消費 out：hop 個 float，已增強的 mono 音訊；
         * info.doa_used_rad / info.vad_out / info.delay 與 bres.spectrum 是唯讀 telemetry。 */
    }

    printf("ok hop=%d effective_adapt_interval=%d lambda=%.9f\n", hop,
           mcab_sample_beam_bf_effective_adapt_interval(beam),
           (double)mcab_sample_beam_bf_lambda(beam));
    anr_destroy(nr);                                       /* AecRes has no destroy */
    free(post_pool);
    mcab_sample_beam_destroy(beam);                        /* before the flow and the lanes */
    free(beam_mem);
    mcab_sample_flow_destroy(&flow);
    return 0;
}
```

編譯（`mcab_sample_beam.o`、`mcab_sample_flow.o` 與各 archive 由本目錄的 Makefile 建出；路徑用 `print-*` 目標取得，不要硬寫）：

```sh
make -C pipelines/4ch_aec_bf_nr_res libs          # 各 archive 與兩個 sample object
make -s -C pipelines/4ch_aec_bf_nr_res print-obj-dir   # mcab_sample_beam.o、mcab_sample_flow.o 所在目錄
make -s -C pipelines/4ch_aec_bf_nr_res print-lib-path  # libaudio_pipeline_mcab.a
cc -std=gnu99 -O2 -Wall -Wextra -c app.c -o app.o \
  -I pipelines/4ch_aec_bf_nr_res \
  -I lib/aec/c_impl/include \
  -I lib/nr/c_impl/include \
  -I <audiolib>/common/include \
  -I <audiolib>/common \
  -I <audiolib>/doa \
  -I <audiolib>/bf
c++ -o app app.o \
  <4ch-obj>/mcab_sample_beam.o <4ch-obj>/mcab_sample_flow.o \
  <4ch-bin>/libaudio_pipeline_mcab.a \
  <doa-bin>/libdoa.a <bf-bin>/libbf.a <spatial-bin>/libspatial_common.a \
  <aec-bin>/libaec.a \
  <nr-bin>/libanr.a \
  <audiolib-bin>/libaudio_common.a \
  -lm
```

要接 ULCNet 後濾波時，不呼叫 `mcab_sample_beam_fuse()`、`enable_post = 0`，把 `beam.spectrum`、同一次呼叫的**原始** far hop 與
`info.delay.changed`／`info.delay.solid` 交給 `ulcnet_nn_stage_begin_complex()`（見 `pipelines/ulcnet/4ch_alignulcnet/README.md`）。

要自己寫 init 順序（不用 `mcab_sample_flow_*`）：照 `mcab.h` 檔頭註解與 `pipelines/4ch_aec_bf_nr_res/mcab_static.c` 的
`unit_plan()`／`unit_init()`／`unit_destroy()`（[§3](integration_4ch_spatial_zh_TW_02_lifecycle.md#3-lifecycle) 表列其步驟）。

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW.md) ｜ [下一份](integration_4ch_spatial_zh_TW_02_lifecycle.md) <!-- split-nav -->
