# C/Python inference parity and alignment events

The C ULCNet stage (`pipelines/ulcnet/ulcnet_nn_stage.c`, `gate_delay`) and
Python inference (`inference.py` -> `_streaming.py`) differ at alignment events:

| | C stage | Python inference |
|---|---|---|
| event | `delay.changed`, or the first transition to solid after the first hop | not detected: `lib/aec/python` has `get_post_reset_serial` but no equivalent of `AecLinearContext.delay_state` |
| model state | `model->reset` zeroes every state tensor (attention ring, logit history, recurrent state) | never reset, runs continuously |
| frames | `ULCNET_NN_STAGE_REPRIME_FRAMES` (1) identity frame | every frame goes through the model |

The reset policy is C-only for now; this is a documented divergence (function
map D-44, row PL-ULC-01) and predates `aiaec-linear-error-v4`. It does not
change the front-end contract or the model I/O layout. Asking `lib/aec` for a
Python event accessor so that Python can mirror the policy is a separate
decision.

Consequently the C/Python inference comparison of a checkpoint is defined only
over spans without alignment events:

1. **Event-free by construction (primary).** Python
   `inference.py --input-is-linear-error` against the C stage fed the same
   `linear_error` and `far` with `delay = {changed 0, solid 1}` on every hop.
   Neither side has an AEC, so neither has an event; both start from zeroed
   state. Compare the whole stream after hop #0.
2. **AEC in the loop.** Every real clip has at least one event (the lock), so
   compare only (a) from the stream start up to the frame before the first
   event hop, and (b) from the last event hop plus the attention depth `D`
   frames plus a settling margin. The margin has to be measured on the
   delivered checkpoint (the recurrent state is not guaranteed to forget at a
   known rate); frames in between are expected to differ and are not a defect.
   Take the event hops from the C side (the stage's `delay` argument, logged per
   hop). A change of `applied_delay_samples` in the sweep alignment trace is
   only a proxy for them.

No committed test drives the stage against Python (function map T-15); until
the owner's checkpoint exists, either comparison can run only on a synthetic
checkpoint. The model-level pieces (`ulcnet_model_io`, `ulcnet_prepost`) are
independent of alignment events and keep their own C tests.

[返回本主題](../README.md)
