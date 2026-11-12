/* Prints every row of ai3_model_io_deepvqes.h and ai3_model_io_alignulcnet.h, expanded, for
 * AIAEC/tests/test_aiaec_bind_info_pin.py to compare with a fresh export. Built by `make model-io-dumper`.
 *   row <id> layout <v> counts <signals> <heads> <inputs> <outputs>
 *   in <i> <b> <c> <h> <w> <elements>      inputs in order: signals, then states
 *   out <i> <b> <c> <h> <w> <elements>     outputs in order: heads, then states
 * <id> is deepvqe_s_<sample_rate>_<fft>_n<N> or align_ulcnet_<sample_rate>_<fft>_n<N>. */

#include <stdio.h>

#include "deepvqe_prepost.h"
#include "ulcnet_model_io.h"
#include "ai3_model_io.h"
#include "ai3_model_io_deepvqes.h"
#include "ai3_model_io_alignulcnet.h"

_Static_assert(AI3_MODEL_IO_DEEPVQES_LAYOUT == DEEPVQE_PREPOST_LAYOUT_VERSION, "DeepVQE-S layout version");
_Static_assert(AI3_MODEL_IO_ALIGNULCNET_LAYOUT_ONE_FRAME == ULCNET_MODEL_IO_LAYOUT_VERSION, "Align-ULCNet layout 18");
_Static_assert(AI3_MODEL_IO_ALIGNULCNET_LAYOUT_PACKET == ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION, "Align-ULCNet layout 19");

static void dump(const char *name, const Ai3ModelIoConfig *c)
{
    const StreamWiring *w = &c->wiring;
    int i;

    printf("row %s_%d_%d_n%d layout %d counts %d %d %d %d\n", name, c->sample_rate, c->fft, c->n, c->layout_version,
           w->signal_in_count, w->head_out_count, w->input_count, w->output_count);
    for (i = 0; i < w->input_count; ++i)
        printf("in %d %u %u %u %u %zu\n", i, w->inputs[i].b, w->inputs[i].c, w->inputs[i].h, w->inputs[i].w,
               w->inputs[i].elements);
    for (i = 0; i < w->output_count; ++i)
        printf("out %d %u %u %u %u %zu\n", i, w->outputs[i].b, w->outputs[i].c, w->outputs[i].h, w->outputs[i].w,
               w->outputs[i].elements);
}

int main(void)
{
    Ai3ModelIoConfig c;
    int i;

    for (i = 0; i < AI3_MODEL_IO_DEEPVQES_ROW_COUNT; ++i) {
        if (ai3_model_io_deepvqes_row(i, &c) != AI3_MODEL_IO_OK) return 1;
        dump("deepvqe_s", &c);
    }
    for (i = 0; i < AI3_MODEL_IO_ALIGNULCNET_ROW_COUNT; ++i) {
        if (ai3_model_io_alignulcnet_row(i, &c) != AI3_MODEL_IO_OK) return 1;
        dump("align_ulcnet", &c);
    }
    return 0;
}
