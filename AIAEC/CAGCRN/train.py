#!/usr/bin/env python3
"""CAGCRN training -- backup end-to-end AEC + RES + NR candidate.

用法:
    python3 train.py --config config.ini --packed-dir data_aec_16k/packed/all --gpu 0 --mmap
    python3 train.py --config config.ini --device cpu
    python3 train.py --config config.ini --resume output/cagcrn_e5.pth
    python3 train.py --config config.ini --resume output/cagcrn_e5.pth --reset-optimizer
    python3 train.py --config config.ini --seed 123

config.ini sections (see the shipped config.ini for every knob, documented):
    [signal]   sample rate + FFT grid. The paper used 16 kHz/512/512/256,
               exactly matching this project's 16 kHz grid; MUST equal the
               grid AIAEC/dataset_gen/config.ini rendered with.
    [data]     one packed corpus path + DataLoader batch size / workers +
               val_fraction (held out at LOAD time, see below)
    [model]    every CAGCRN constructor keyword (see model.py's __init__);
               an unknown key raises rather than being silently ignored, so a
               typo cannot build a differently-shaped model
    [training] optimizer, seed, epoch budget, checkpoint/log locations
    [loss]     paper MSE + SI-SNR + L1 objective weights

dataset:
    Data is rendered and packed by AIAEC/dataset_gen/ only -- see
    AIAEC/dataset_gen/README.md and ../Align_CRUSE/train.py's docstring for
    the full generation walkthrough (identical for every candidate) --
    including why this generates one unified pool (``--split all``) and
    performs a deterministic random split over individual 10-second chunks.
    CAGCRN is end-to-end -- mic + unaligned far-end go straight to one
    network, no linear AEC in front -- so like Align_CRUSE (and unlike
    the other candidate trainers) there is no persistent external state to
    preserve across chunks, and training draws chunks with a plain SHUFFLED
    DataLoader over each split.

task (see ../README.md's decision matrix and
../dataset_gen/model_views.py MODEL_TASKS['CAGCRN']):
    mic + unaligned far-end in; denoised, echo-free, early/dereverberated
    near-end speech out. This common product target extends the paper's
    non-dereverberating task and is recorded in the checkpoint contract.

⚠ See model.py/README.md: the paper's proposed learnable integer floor(D)
delay window cannot receive useful ordinary autograd through a tensor-shape
operation. This implementation uses a differentiable soft delay-window gate
instead -- trainable, but checkpoint-incompatible with any unpublished
author implementation.

inference: see inference.py's own top-of-file usage comment.
"""

import argparse
import configparser
import os
import sys
import time

import torch
import torch.nn as nn

# Run directly as `python3 train.py` from this directory, exactly like every
# AINR trainer -- so the repo root (parent of AIAEC/) must go on sys.path
# before the AIAEC.* imports below, the same way AINR/RNNoise-ERB/train.py
# adds its own repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.CAGCRN import CAGCRN
from AIAEC.dataset_gen import (
    AecStems,
    MODEL_TASKS,
    PACKED_STEM_ORDER,
    build_model_view,
    build_spectral_model_view,
)
from AIAEC.training_common import (
    GradNormLog,
    NonFiniteTraining,
    WeightScaleGuard,
    build_arg_parser,
    build_plain_loaders,
    auto_device,
    halt_on_non_finite,
    make_checkpoint_contract,
    read_grids,
    read_model_kwargs,
    require_checkpoint_contract,
    scan_non_finite,
    set_seed,
    si_snr_loss,
    spectrum_to_waveform,
    training_progress,
)


MODEL_NAME = 'CAGCRN'
TASK = MODEL_TASKS[MODEL_NAME]
LOSS_VERSION = 'cagcrn_mse_sisnr_l1_adamw_v1'


def build_parser() -> argparse.ArgumentParser:
    return build_arg_parser('Train CAGCRN (backup end-to-end AEC+RES+NR)')


def forward_batch(model, stems_batch, aec_grid, device):
    # dtype rides the same .to() as the device move, not the dataset: a
    # --dtype float16 corpus arrives here still half and dies in
    # torch.stft, which has no half CPU kernel. Widening here keeps the
    # smaller dtype across the loader. float32 corpora: no-op.
    stems = AecStems(
        stems_batch.to(device=device, dtype=torch.float32),
        PACKED_STEM_ORDER,
    )
    view = build_model_view(stems, MODEL_NAME, sample_rate=aec_grid.sr)
    spectral = build_spectral_model_view(view, aec_grid)
    output = model(**spectral.inputs)
    return output, spectral


def run_epoch(model, loader, aec_grid, device, loss_cfg, optimizer=None, *,
             epoch=0, global_step=0, output_dir=None, sr=None,
             grad_clip=1.0, checkpoint_for_halt=None, grad_log=None,
             max_epochs=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches = 0.0, 0
    # Fixed once the model is built. Counting it inside the batch loop walked
    # every parameter tensor in Python on every step for a constant.
    parameter_count = sum(p.numel() for p in model.parameters())

    batches = training_progress(
        loader, training=training, epoch=epoch, max_epochs=max_epochs
    )
    for batch_idx, (stems_batch, _meta) in enumerate(batches):
        with torch.set_grad_enabled(training):
            output, spectral = forward_batch(model, stems_batch, aec_grid, device)
            spectral_mse = (output.enhanced - spectral.target).abs().square().mean()
            estimate_wave = spectrum_to_waveform(output.enhanced, aec_grid)
            target_wave = spectrum_to_waveform(spectral.target, aec_grid)
            sisnr = si_snr_loss(
                estimate_wave, target_wave,
                eps=loss_cfg.getfloat('eps', fallback=1e-8),
            )
            # The paper calls L1 a sparsity regularizer but does not state a
            # normalization.  Mean absolute parameter value keeps its scale
            # invariant when the model width changes; a raw sum would make the
            # coefficient secretly depend on parameter count.
            l1 = sum(p.abs().sum() for p in model.parameters()) / parameter_count
            loss = (loss_cfg.getfloat('mse_weight', fallback=1.0) * spectral_mse
                    + loss_cfg.getfloat('si_snr_weight', fallback=1.0) * sisnr
                    + loss_cfg.getfloat('l1_weight', fallback=1.0) * l1)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach())
            if not torch.isfinite(loss.detach()):
                halt_on_non_finite(
                    'non-finite loss', model=model,
                    mic=spectral.inputs['microphone'], target=spectral.target,
                    epoch=epoch, batch_idx=batch_idx, global_step=global_step,
                    loss_value=loss_value, total_norm=None,
                    output_dir=output_dir, sr=sr, checkpoint=checkpoint_for_halt,
                    enhanced=output.enhanced,
                )
            loss.backward()
            # error_if_nonfinite=True is what stops clipping from CREATING the
            # NaN.  Without it a non-finite norm gives
            # clip_coef = grad_clip/(inf+1e-6) = 0.0, and inf*0.0 = NaN, so every
            # gradient is already NaN by the time a check downstream of the clip
            # can look at it -- the dump then describes the clip's damage instead
            # of what backward produced.  With the flag the raise lands before
            # any scaling.
            try:
                total_norm = float(nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip, error_if_nonfinite=True))
            except RuntimeError as exc:
                halt_on_non_finite(
                    f'non-finite gradient norm: {exc}',
                    model=model,
                    mic=spectral.inputs['microphone'], target=spectral.target,
                    epoch=epoch, batch_idx=batch_idx, global_step=global_step,
                    loss_value=loss_value, total_norm='non-finite',
                    output_dir=output_dir, sr=sr, checkpoint=checkpoint_for_halt,
                    enhanced=output.enhanced,
                )
            optimizer.step()
            if grad_log is not None:
                grad_log.record(
                    total_norm, epoch=epoch, batch_idx=batch_idx,
                    global_step=global_step, loss_value=loss_value,
                    noisy=spectral.inputs['microphone'], clean=spectral.target,
                    output_dir=output_dir, enhanced=output.enhanced,
                )
            global_step += 1
        else:
            loss_value = float(loss.detach())

        total_loss += loss_value
        n_batches += 1
        if training:
            batches.set_postfix(
                loss=f"{loss_value:.4f}", gnorm=f"{total_norm:.2e}",
                refresh=False,
            )

    return total_loss / max(1, n_batches), global_step


def main(args):
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")

    set_seed(args.seed)
    device = auto_device(args.device, args.gpu)
    print(f"Device: {device}")

    aec_grid, model_grid = read_grids(cfg)
    model_kwargs = read_model_kwargs(cfg, CAGCRN)
    model = CAGCRN(model_grid, **model_kwargs).to(device)
    print(f"CAGCRN: {sum(p.numel() for p in model.parameters())} params, "
          f"grid={model_grid}")

    train_loader, val_loader, data_contract = build_plain_loaders(
        cfg, aec_grid, seed=args.seed,
        packed_dir=args.packed_dir, mmap=args.mmap,
    )
    contract = make_checkpoint_contract(
        model_name=MODEL_NAME, task=TASK, grid=model_grid,
        model_kwargs=model_kwargs, loss_version=LOSS_VERSION,
        data_contract=data_contract,
    )

    output_dir = cfg.get('training', 'output_dir', fallback='output')
    os.makedirs(output_dir, exist_ok=True)
    lr = cfg.getfloat('training', 'lr', fallback=1e-3)
    max_epochs = cfg.getint('training', 'max_epochs', fallback=50)
    optimizer_name = cfg.get('training', 'optimizer', fallback='adamw').lower()
    if optimizer_name != 'adamw':
        raise ValueError("CAGCRN paper recipe requires optimizer=adamw")
    if cfg.get('training', 'scheduler', fallback='constant').lower() != 'constant':
        raise ValueError("CAGCRN paper reports a constant LR")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=cfg.getfloat('training', 'weight_decay', fallback=5e-7),
        amsgrad=cfg.getboolean('training', 'amsgrad', fallback=False),
    )

    start_epoch, global_step, best_val = 0, 0, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        require_checkpoint_contract(ckpt, contract, context=args.resume)
        model.load_state_dict(ckpt['state_dict'])
        poisoned = scan_non_finite(model)
        if poisoned:
            raise NonFiniteTraining(
                f"{args.resume} contains non-finite weights: {poisoned[:5]}")
        if not args.reset_optimizer:
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            global_step = ckpt['global_step']
            best_val = ckpt.get('best_val', best_val)
        resumed_lr = optimizer.param_groups[0]['lr']
        print(f"Resumed from {args.resume} at epoch {start_epoch}"
              f"{' (fresh optimizer)' if args.reset_optimizer else ''}"
              f", lr={resumed_lr:.4e}")

    loss_cfg = cfg['loss'] if cfg.has_section('loss') else {
        'mse_weight': '1.0', 'si_snr_weight': '1.0',
        'l1_weight': '1.0', 'eps': '1e-8'}
    patience = cfg.getint('training', 'early_stop_patience', fallback=50)
    grad_clip = cfg.getfloat('training', 'grad_clip', fallback=1.0)
    grad_log = GradNormLog(os.path.join(output_dir, 'grad_norm.csv'), aec_grid.sr)
    # Per-tensor, because grad_norm.csv above is a GLOBAL norm and stays healthy
    # while one branch's weights decay to nothing.  Built here, after any
    # --resume load, so a resumed run measures against what it resumed from.
    weight_guard = WeightScaleGuard(model)

    no_improve = 0
    for epoch in range(start_epoch, max_epochs):
        checkpoint_for_halt = {
            'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch - 1, 'global_step': global_step, 'contract': contract,
        }
        started = time.time()
        train_loss, global_step = run_epoch(
            model, train_loader, aec_grid, device, loss_cfg, optimizer,
            epoch=epoch, global_step=global_step, output_dir=output_dir,
            sr=aec_grid.sr, grad_clip=grad_clip,
            checkpoint_for_halt=checkpoint_for_halt, grad_log=grad_log,
            max_epochs=max_epochs,
        )
        msg = f"epoch {epoch}: train_loss={train_loss:.4f} ({time.time()-started:.0f}s)"

        val_loss = train_loss
        if val_loader is not None:
            val_loss, _ = run_epoch(model, val_loader, aec_grid, device, loss_cfg)
            msg += f" val_loss={val_loss:.4f}"
        print(msg)

        checkpoint = {
            'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch, 'global_step': global_step, 'contract': contract,
            'best_val': min(best_val, val_loss),
        }
        poisoned = scan_non_finite(model)
        if poisoned:
            raise NonFiniteTraining(
                f"refusing to save non-finite weights: {poisoned[:5]}")
        weight_guard.check(epoch=epoch, global_step=global_step)
        torch.save(checkpoint, os.path.join(output_dir, f'{MODEL_NAME.lower()}_last.pth'))
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(checkpoint, os.path.join(output_dir, f'{MODEL_NAME.lower()}_best.pth'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stop: no improvement in {patience} epochs")
                break
    grad_log.close()


if __name__ == '__main__':
    main(build_parser().parse_args())
