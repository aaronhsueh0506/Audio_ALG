/* Binary stdin/stdout driver around the Align-ULCNet _freq family (aiaec_*_freq), for
 * AIAEC/tests/test_aiaec_packet_ort.py: the host runs the accelerator (ONNX Runtime) between a PRE and a POST
 * request. Align-ULCNet product, 16k/256, built with -DAIAEC_TEST_CATALOG (the N=8 row). Native-endian int32 / float32.
 *
 *   OPEN  cmd=1, n                            -> rc, state_generation, slot
 *   PRE   cmd=2, slot, err_re err_im far_re far_im [n * 129 floats each], changed[n], solid[n] (int32)
 *                                             -> rc, action, work.state_generation, live generation, input_count,
 *                                                (elements, floats) per input tensor, output elements
 *   POST  cmd=3, slot, infer_status, mask_elements, mask floats
 *                                             -> rc, live generation, re[n * 129], im[n * 129]
 *   QUIT  cmd=9
 * Slots share nothing; a slot's pool is allocated at OPEN and kept. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"

#if AIAEC_PRODUCT_MODEL_ID != AIAEC_MODEL_ID_ALIGN_ULCNET
#error "packet_driver.c is the Align-ULCNet product's driver"
#endif

#define BINS 129
#define MAX_SLOTS 4
#define MAX_N 64

typedef struct Slot {
    AiaecFreqCtx *ctx;
    void *pool;
    int n;
    AiaecWork work;
} Slot;

static Slot g_slot[MAX_SLOTS];
static int g_slots;

static int rd(void *p, size_t bytes) { return fread(p, 1, bytes, stdin) == bytes ? 0 : -1; }
static void wr(const void *p, size_t bytes) { if (fwrite(p, 1, bytes, stdout) != bytes) exit(3); }
static void wri(int v) { wr(&v, sizeof(v)); }

int main(void) {
    int cmd;

    for (;;) {
        if (rd(&cmd, sizeof(cmd)) != 0) return 0;
        if (cmd == 9) return 0;
        if (cmd == 1) {
            int n, e = 0, rc;
            AiaecConfig cfg;
            AiaecInfo info;
            StreamMemReq rq;
            if (rd(&n, sizeof(n)) != 0) return 2;
            cfg.sr = 16000; cfg.fft = 256; cfg.hop_in = 128 * n;
            rc = aiaec_get_mem_size_freq(&cfg, &rq);
            if (rc != AIAEC_OK || g_slots >= MAX_SLOTS || n < 1 || n > MAX_N) { wri(rc ? rc : -99); wri(0); wri(-1); fflush(stdout); continue; }
            if (posix_memalign(&g_slot[g_slots].pool, 16, (size_t)rq.bytes) != 0) return 2;
            g_slot[g_slots].ctx = NULL;
            e = aiaec_init_freq(&cfg, g_slot[g_slots].pool, (size_t)rq.bytes, &g_slot[g_slots].ctx, &info);
            g_slot[g_slots].n = n;
            wri(g_slot[g_slots].ctx ? AIAEC_OK : e);
            wri((int)aiaec_state_generation_freq(g_slot[g_slots].ctx));
            wri(g_slots);
            ++g_slots;
        } else if (cmd == 2) {
            int slot, i, rc;
            Slot *s;
            size_t plane;
            float *p[4];
            int changed[MAX_N], solid[MAX_N];
            AiaecAlignEvent ev[MAX_N];
            AiaecFreqInput fin;

            if (rd(&slot, sizeof(slot)) != 0 || slot < 0 || slot >= g_slots) return 2;
            s = &g_slot[slot];
            plane = (size_t)s->n * BINS;
            for (i = 0; i < 4; ++i) {
                p[i] = malloc(plane * sizeof(float));
                if (!p[i] || rd(p[i], plane * sizeof(float)) != 0) return 2;
            }
            if (rd(changed, sizeof(int) * (size_t)s->n) != 0 || rd(solid, sizeof(int) * (size_t)s->n) != 0) return 2;
            for (i = 0; i < s->n; ++i) { ev[i].changed = changed[i]; ev[i].solid = solid[i]; }
            memset(&fin, 0, sizeof(fin));
            fin.kind = AIAEC_INPUT_ALIGN_ULCNET; fin.frames = s->n;
            fin.u.align_ulcnet.err_re = p[0]; fin.u.align_ulcnet.err_im = p[1];
            fin.u.align_ulcnet.far_re = p[2]; fin.u.align_ulcnet.far_im = p[3];
            fin.u.align_ulcnet.events = ev; fin.u.align_ulcnet.event_count = s->n;
            rc = aiaec_preprocess_freq(s->ctx, &fin, &s->work);
            wri(rc);
            wri((int)s->work.action);
            wri((int)s->work.state_generation);
            wri((int)aiaec_state_generation_freq(s->ctx));
            wri(rc == AIAEC_OK ? s->work.input_count : 0);
            for (i = 0; rc == AIAEC_OK && i < s->work.input_count; ++i) {
                wri((int)s->work.inputs[i].elements);
                wr(s->work.inputs[i].data, s->work.inputs[i].elements * sizeof(float));
            }
            wri(rc == AIAEC_OK && s->work.output_count > 0 ? (int)s->work.outputs[0].elements : 0);
            for (i = 0; i < 4; ++i) free(p[i]);
        } else if (cmd == 3) {
            int slot, status, me, rc;
            Slot *s;
            float *re, *im;

            if (rd(&slot, sizeof(slot)) != 0 || slot < 0 || slot >= g_slots) return 2;
            s = &g_slot[slot];
            if (rd(&status, sizeof(status)) != 0 || rd(&me, sizeof(me)) != 0) return 2;
            if (me > 0) {
                if ((size_t)me != s->work.outputs[0].elements || rd(s->work.outputs[0].data, (size_t)me * sizeof(float)) != 0) return 2;
            }
            re = malloc((size_t)s->n * BINS * sizeof(float));
            im = malloc((size_t)s->n * BINS * sizeof(float));
            rc = aiaec_postprocess_freq(s->ctx, status, re, im);
            wri(rc);
            wri((int)aiaec_state_generation_freq(s->ctx));
            wr(re, (size_t)s->n * BINS * sizeof(float));
            wr(im, (size_t)s->n * BINS * sizeof(float));
            free(re); free(im);
        } else {
            return 2;
        }
        fflush(stdout);
    }
}
