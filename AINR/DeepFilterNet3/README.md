# DeepFilterNet3

Preserved 48 kHz band-split branch for controlled comparison with the restored
DFN2 cascade/alpha model.

## Architecture contract

DFN3 predicts an ERB mask and low-band five-tap complex deep filter:

```text
bins 0..95:       complex deep filter on the input spectrum
bins 96..Nyquist: ERB-masked spectrum
```

The two paths are composed in parallel at the crossover. There is no learned
alpha residual blend. This is the former v5 implementation moved out of
`DeepFilterNet2/`; it is intentionally kept behaviorally unchanged.

Default signal contract:

- 48 kHz;
- FFT/window/hop `1024/1024/512`;
- 32 ERB bands;
- `df_bins=96`, `df_order=5`;
- mask/DF lookahead `1/1`;
- optional post-filter disabled.

## Checkpoints

Current model version:

```text
dfn3_fmajor_flatten_pathway_add_no_lsnr_v5
```

Existing v5 band-split checkpoints belong to this directory. They are not
compatible with DFN2 v6, despite the shared feature version and the large
overlap in parameter names.

## Train and infer

```bash
python3 train.py --config config.ini --packed-dir /path/to/data_48k

python3 denoise.py --config config.ini \
    --model /path/to/dfn3_checkpoint.pt \
    --input input_48k.wav --output output_48k.wav
```

DFN3 uses the same 48 kHz noisy/clean dataset contract as DFN2. A dataset
regeneration is not required merely because the output graph differs.

## Comparison guidance

For a fair DFN2/DFN3 bake-off, hold dataset split, seed, epoch size, loss,
optimizer, lookahead, and evaluation cohorts fixed. Report the models by their
full contract version. A result from a force-loaded or relabeled checkpoint is
invalid.

Use DFN3 as the diagnostic control for the low-frequency disappearance issue:
if only DFN2 fails, inspect cascade ordering and alpha/DF behavior; if both
fail, prioritize dataset distribution, target quality, and shared
normalization/loss settings.

The detailed v5 audit is in
[`UPSTREAM_ALIGNMENT.md`](UPSTREAM_ALIGNMENT.md).
