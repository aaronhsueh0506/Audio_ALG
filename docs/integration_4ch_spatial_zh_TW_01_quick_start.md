> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW.md) ｜ [下一份](integration_4ch_spatial_zh_TW_02_lifecycle.md) <!-- split-nav -->
## 2. Quick start

完整、可編譯、已實際跑過的最小整合（單元 heap 路徑；RES、NR 與合成由應用自己初始化）：

```c
#include <stdio.h>
#include <stdlib.h>
#include "audio_pipeline_4ch.h"
#include "anr.h"
#include "audio_stft.h"
#include "aec_res.h"
#include "samples/se_anr_stage.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

int main(void) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
    cfg.geometry     = AUDIO_PIPELINE_4CH_GEOMETRY_UCA;
    cfg.uca_radius_m = 0.035f;                     /* 換成你陣列的真實半徑 */

    AudioPipeline4Ch* p = audio_pipeline_4ch_create(&cfg);   /* heap path */
    if (!p) { fprintf(stderr, "create failed\n"); return 1; }

    /* RES (libaec AecRes), NR and synthesis are separate libraries the application sizes and
     * initialises itself; the unit owns none of them. */
    const int hop = audio_pipeline_4ch_hop_size(p);
    const int n = audio_pipeline_4ch_n_freqs(p);
    const int fft = audio_pipeline_4ch_fft_size(p);
    AecConfig res_cfg;
    aec_config_from_preset(&res_cfg, AEC_PRESET_BALANCED, 16000);
    res_cfg.fft_size = fft;                                  /* the unit's grid */
    AnrConfig nr_cfg = anr_config_for_mode_grid(16000, fft, ANR_MODE_BALANCED);
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
    uint32_t rng = SE_CNG_RNG_SEED;

    float mics[4 * 1024], ref[1024], out[1024];
    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < hop; i++) {                    /* 換成你自己的擷取來源 */
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ch++)
                mics[i * FOUR_AEC_NR_RES_CHANNELS + ch] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }

        AudioPipeline4ChBeam beam;
        AudioPipeline4ChFrameInfo info;
        SeSpecFrame f;
        const float* g_res = NULL;
        const float* g_nr = NULL;
        int status = audio_pipeline_4ch_process(p, mics, ref, &beam, &info);   /* lanes + DOA + BF */
        if (status == FOUR_AEC_NR_RES_OK)
            status = audio_pipeline_4ch_fuse(p, &f);          /* SeSpecFrame; f.g_res_ctx is NULL */
        if (status == FOUR_AEC_NR_RES_OK)
            g_res = se_res_stage_gain(res, &f);               /* RES gain from your AecRes */
        if (g_res)
            g_nr = se_anr_stage_gain(nr, &f, extra_noise, 1);
        if (g_nr) {
            sk_min_f32(total_gain, g_nr, g_res, n);
            sk_capply_gain_f32(spec, f.e, total_gain, n);
            if (f.cn) aec_cng_fill(&rng, spec, f.cn, g_res, g_nr, SE_CNG_NR_GAIN_FLOOR, n);
            (void)ac_istft_push(istft, spec, out);
        }
        if (!g_nr || !se_hop_is_finite(out, hop)) {
            fprintf(stderr, "status=%d\n", status);
            audio_pipeline_4ch_reset(p);                   /* a refused call: reset every stage */
            aec_res_reset(res);
            anr_reset(nr);
            ac_istft_reset(istft);
            rng = SE_CNG_RNG_SEED;
            continue;
        }
        /* 消費 out：hop 個 float，已增強的 mono 音訊；
         * info.doa_used_rad / info.vad_out / info.delay 與 beam.spectrum 是唯讀 telemetry。 */
    }

    printf("ok hop=%d effective_adapt_interval=%d lambda=%.9f\n", hop,
           audio_pipeline_4ch_bf_effective_adapt_interval(p),
           (double)audio_pipeline_4ch_bf_lambda(p));
    anr_destroy(nr);                                       /* AecRes has no destroy */
    free(post_pool);
    audio_pipeline_4ch_destroy(p);
    return 0;
}
```

編譯：

```sh
cc -std=gnu99 -O2 -Wall -Wextra -c app.c -o app.o \
  -I pipelines/4ch_aec_bf_nr_res \
  -I pipelines \
  -I lib/aec/c_impl/include \
  -I lib/nr/c_impl/include \
  -I <audiolib>/common/include \
  -I <audiolib>/common
c++ -o app app.o \
  <4ch-bin>/libaudio_pipeline_4ch.a \
  <spatial-bin>/libdoa.a <spatial-bin>/libbf.a <spatial-bin>/libspatial_common.a \
  <aec-bin>/libaec.a \
  <nr-bin>/libanr.a \
  <audiolib-bin>/libaudio_common.a \
  -lm
```

要接 ULCNet 後濾波時，不呼叫 `fuse()`、`core.enable_post = 0`，把 `beam.spectrum`、同一次呼叫的**原始** far hop 與
`info.delay.changed`／`info.delay.solid` 交給 `ulcnet_nn_stage_begin_complex()`（見 `pipelines/4ch_alignulcnet/README.md`）。

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW.md) ｜ [下一份](integration_4ch_spatial_zh_TW_02_lifecycle.md) <!-- split-nav -->
