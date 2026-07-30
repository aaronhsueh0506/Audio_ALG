# JointAECNR — one network for AEC + RES + NR

    in : (Y, X)      microphone, far-end reference
    out: S_hat       the finished near-end speech
    target: S        (the `near_speech` stem)

Bake-off entry **#5**: instead of a cascade, a single dual-input network does
echo cancellation, residual-echo suppression and noise reduction together.
There is **no `D_hat` handoff**, no `min(g_nr, g_res)` fusion, and no classical
comfort-noise generator driven by `g_res` — this model removes all three by
construction.

## What it consumes

The packed AEC corpus from `dataset_gen/aec/pack_aec_dataset.py`: a directory
of `.pt` shards, six stems per chunk, parent sequences of 20–60 s cut into
consecutive fixed-length chunks that share a `sequence_id`.

    Y = mic_postclip     X = far_render     D = echo
    S = near_speech      N = local_noise

The trainer derives everything it needs from the stems and stores nothing new.
Channel order, the STFT grid and the sequence sampler all come from
`dataset_gen/aec/aec_features.py` — this project re-declares none of them, so a
drift between the AEC models is not possible without editing that one file.

## What it emits

| output | always? | what it is |
|---|---|---|
| `speech_spec` | yes | **S_hat, the default final output** |
| `mask` | yes | the bounded complex mask applied to Y |
| `echo_spec` | `aux_echo_head` | D_hat, supervised by the independent `echo` stem |
| `noise_log_psd` | `aux_noise_psd_head` | log₁₀ local-noise PSD per bin |
| `ref_gate` | yes | per-frame "has the far end played recently?" in [0, 1) |

### ⚠ The auxiliary heads are what make #5 comparable to #4 + #3 at all

A cascade can be measured stage by stage: how much echo did the linear filter
remove, how much did the residual suppressor remove, what did the noise
suppressor cost the near end. A joint model has no such seams — a worse score
says "worse", never "worse at what".

* `aux_echo_head` supplies D_hat from a model that never had to produce one,
  so "failed to cancel" can be separated from "failed to suppress what it did
  not cancel". It costs ~300 parameters.
* `aux_noise_psd_head` supplies the noise floor that the classical CNG used to
  get from `g_res`. Without it there is no honest way to run comfort noise:
  estimating the floor from the model's own output is circular, because the
  output is where the floor was just removed.

Both are config switches. Turning them off is a legitimate shipping decision
and makes the model unattributable; the trainer banner says so on every run.

## The hard gate

**X == 0 must leave the microphone essentially unmodified.** For a joint model
that statement needs scoping, because this model also removes noise — with a
silent reference and a noisy mic, changing the mic is correct behaviour. The
honest version is enforced in two layers:

1. **Structural.** The entire reference pathway (ref encoder, temporal-context
   conv, fusion projections) is bias-free with positively homogeneous
   activations and no normalisation offsets, so a reference that has been
   exactly zero for the pathway's receptive field contributes **exactly zero**
   — the network is then bit-for-bit a mic-only NR model.
   `tests/test_joint_aecnr.py` asserts this by randomising every
   reference-branch parameter and checking the output does not move.
2. **Structural, for the echo estimate.** D_hat is multiplied by a
   reference-activity gate that is exactly 0 once the far end has been silent
   for `echo_gate_memory_sec`, so a silent reference gives `D_hat == 0.0`
   exactly. ⚠ That memory is a *physics* parameter — the echo of the last
   sample played arrives one bulk delay later and decays over the room's RT60,
   so a gate that closed instantly would forbid cancelling the tail.
3. **Numeric, for a trained model.** `model.idle_gate_report(model, y_spec)`
   returns `mic_delta_db` and `echo_energy_db`. Feed it a clean near-only mic;
   `mic_delta_db` must be deeply negative and `echo_energy_db` must be `-inf`.

## Post-processing policy

`speech_spec` is the **default final output** — nothing in `postproc.py` is
applied automatically. It provides the three things a cascade used to supply
downstream: a safety attenuation limiter, comfort noise driven by
`aux_noise_psd_head`, and a classical-fallback *blend* (replacement, not a
second stage).

⚠ What must not be chained is a **second full noise-suppression stage**.
`PostProcessChain.add(..., suppresses=True)` raises `DoubleSuppressionError` at
runtime with a message explaining why; a caller with a real reason passes
`allow_double_suppression=True` and owns it. This is deliberately a runtime
check rather than a compile-time impossibility — an explicit greppable
argument beats a type-system workaround nobody can find later.

## How to train

    python train.py --config config.ini --packed-dir <packed shard dir> --gpu 0
    python train.py --config config.ini --packed-dir <packed shard dir> \
        --resume output/joint_aecnr_last.pth

* `--seed` defaults to **42** and fixes both the split and the lane layout.
* The split is over **whole sequences**, not chunks. A chunk-level split would
  leak (same talker, same room, two seconds later) and would break the
  sampler's requirement that a sequence's `chunk_index` run be complete. The
  permutation still comes from the shared `locality_preserving_random_split`.
* `batch_size` is the number of **sequence lanes**. Lane *k* walks one 20–60 s
  sequence in order across consecutive batches with its recurrent state intact;
  `lane_reset_mask` zeroes a lane only when it starts a new sequence.
* Checkpoints record the contract, the split indices and the sequence ids;
  resuming across an architecture or version change is refused rather than
  silently accepted.

## Inference

    python denoise.py --config config.ini --model output/joint_aecnr_best.pth \
        --mic mic.wav --ref far_end.wav --output enhanced.wav

⚠ Two inputs. Omitting `--ref` substitutes silence, which asserts *nothing was
played* — a valid experiment (it exercises the idle gate), not "inference
without a reference". `--chunk-sec` walks the file in chunks with the state
carried; it matches whole-file processing to ~1e-8, which is the check that the
model really is the causal streaming model its latency figures claim.

## Grids

16 kHz first (`n_fft = 512`, hop 256, 257 bins, 62.5 fps). The 48 kHz variant
is `sr = 48000 / n_fft = 1024 / win_len = 1024 / hop_len = 512` **and nothing
else**: every duration in `config.ini` is in seconds and every band edge in
hertz, converted against the grid's frame rate exactly once, in
`JointAECNR.from_config`. `tests/test_train_contract.py` fails the build if a
frame count appears in the config.

## Files

| file | role |
|---|---|
| `config.ini` | grid, architecture, loss weights, training — heavily commented |
| `model.py` | the network, the reference-activity gate, the hard-gate report |
| `postproc.py` | optional limiter / CNG / fallback + the double-suppression check |
| `train.py` | contract gate, loss, sequence-level split, state-carrying loop |
| `denoise.py` | inference, streaming or whole-file, with exact reconstruction |
| `tests/` | hard gate, aux-head switches, causality, state carry, contract |

Default configuration: **1,341,483 parameters** (52,032 of them in the
reference pathway).
