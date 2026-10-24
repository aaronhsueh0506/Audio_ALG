# Gain stages

`gain/` holds three stand-alone gain stages. It has no Makefile of its own and
no shipped pipeline target links it.

- `fix_gain` (`fix_gain.h`): per-channel input gain with an optional clip; one
  channel gives a single-gain mono path.
- `nr_gain` (`nr_gain.h`): post-NR broadcast gain chosen by whether a DOA
  estimate exists; the header maps the config fields.
- `post_gain` (`post_gain.h`): directional per-bin post-gain over audiolib's
  `Complex` spectrum (`{float r; float i;}`); convert other complex types with a
  cast at the call site.

All three follow the pool-first convention: `*_get_mem_size()` then `*_init()`,
plus a `*_create()` heap convenience. The caller owns the pool and
`*_destroy()` does not free it. `fix_gain_get_mem_size()` refuses a config with
a non-finite gain or clip value.

Retuning between hops: `fix_gain` takes its config at construction only (build
a new instance to change gains). `nr_gain` takes its config per call and holds
none; `nr_enable` is live state, so refresh a cached config every hop.
`post_gain` takes its config per call, but `F` is fixed for the instance's life
and `gain_match` seeds the smoother at construction.

The stages share audiolib's kernels: `fix_gain` and `nr_gain` scale through
`sk_scale_f32`, `fix_gain` clips through `sk_clip_f32`, and `post_gain` clamps
and applies its per-bin gain through `sk_clip_f32` / `sk_capply_gain_f32`. Two
`post_gain` loops stay open-coded: the box-average frequency smoother (split
into edge/interior/edge to keep the addition order) and the attack/release
smoother (`sk_asym_ema_f32` selects its falling coefficient on `x < s`, this
module selects attack on `target > prev`, so the two differ at
`target == prev`).

The spatial pieces (SRP-PHAT/DOA, BF, VAD) live in the sibling `audiolib`
checkout and are tested there (`make -C audiolib SIMD=1 test`). The 4-channel
pipeline supports `16k/256/128`, `16k/512/256` and `48k/1024/512`
(`sample-rate/frame/hop`, frame equal to FFT), one grid at a time; see
`4ch_aec_bf_nr_res/README.md`.
