#!/usr/bin/env python3
"""Align-CRUSE training -- joint end-to-end AEC+RES+NR.

用法:
    python3 train.py --config config.ini --packed-dir data_aec_16k/packed/all --gpu 0 --mmap
    python3 train.py --config config.ini --device cpu
    python3 train.py --config config.ini --resume output/align_cruse_e5.pth
    python3 train.py --config config.ini --resume output/align_cruse_e5.pth --reset-optimizer
    python3 train.py --config config.ini --seed 123

config.ini sections (see the shipped config.ini for every knob, documented):
    [signal]   sample rate + FFT grid. MUST equal the grid the packed corpus
               below was rendered with (AIAEC/dataset_gen/config.ini's
               [signal]); AecGrid/SignalGrid do not re-derive it from the data.
    [data]     one packed corpus path + DataLoader batch size / workers +
               val_fraction (held out at LOAD time, see below)
    [model]    every AlignCRUSE constructor keyword (see model.py's
               __init__); an unknown key raises rather than being silently
               ignored, so a typo cannot build a differently-shaped model
    [training] optimizer, seed, epoch budget, checkpoint/log locations
    [loss]     compressed_spectral_loss term weights

dataset:
    Data is rendered and packed by AIAEC/dataset_gen/ only -- see
    AIAEC/dataset_gen/README.md and its config.example.ini for the full
    generation walkthrough. This project generates ONE unified pool (no
    source-disjoint split at generation time) and splits
    train/val at LOAD time instead:
        cp AIAEC/dataset_gen/config.example.ini AIAEC/dataset_gen/config.ini
        python3 -m AIAEC.dataset_gen.gen_aec_dataset --config ... --split all --hours 100 --output data_aec_16k
        python3 -m AIAEC.dataset_gen.pack_aec_dataset --input data_aec_16k/all --output data_aec_16k/packed/all
    ``[data] val_fraction`` then holds out individual 10-second chunks via
    ``training_common.split_dataset_by_sample``, deterministically from
    ``--seed``. Chunks from the same sequence/speaker/RIR may intentionally
    straddle the split, so generalisation requires a separate held-out set.
    Training uses a plain shuffled DataLoader; validation does not shuffle.

task (see ../README.md's decision matrix and
../dataset_gen/model_views.py MODEL_TASKS['Align_CRUSE']):
    mic + unaligned far-end in; near_target (early/dereverberated, denoised,
    echo-cancelled near speech) out. This candidate previously ran its own
    AEC/RES-only route -- background noise deliberately preserved for a
    later, independent NR stage -- but that route was retired: there is no
    more standalone AEC-only candidate, so Align-CRUSE now shares the same
    joint end-to-end AEC+RES+NR dereverb task and target as DeepVQE-S.

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

from AIAEC.Align_CRUSE import AlignCRUSE
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
    build_arg_parser,
    build_plain_loaders,
    auto_device,
    compressed_spectral_loss,
    halt_on_non_finite,
    fast_forward_scheduler,
    make_checkpoint_contract,
    make_scheduler,
    read_grids,
    read_model_kwargs,
    require_checkpoint_contract,
    scan_non_finite,
    set_seed,
    training_progress,
)


MODEL_NAME = 'Align_CRUSE'
TASK = MODEL_TASKS[MODEL_NAME]
LOSS_VERSION = 'aiaec_compressed_spectral_v1'


def build_parser() -> argparse.ArgumentParser:
    return build_arg_parser('Train Align-CRUSE (joint end-to-end AEC+RES+NR)')


def forward_batch(model, stems_batch, aec_grid, device):
    stems = AecStems(stems_batch.to(device), PACKED_STEM_ORDER)
    view = build_model_view(stems, MODEL_NAME, sample_rate=aec_grid.sr)
    spectral = build_spectral_model_view(view, aec_grid)
    output = model(**spectral.inputs)
    return output, spectral


def run_epoch(model, loader, aec_grid, device, loss_cfg, optimizer=None, *,
             epoch=0, global_step=0, output_dir=None, sr=None,
             grad_clip=1.0, checkpoint_for_halt=None, grad_log=None,
             max_epochs=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches = 0.0, 0

    batches = training_progress(
        loader, training=training, epoch=epoch, max_epochs=max_epochs
    )
    for batch_idx, (stems_batch, _meta) in enumerate(batches):
        with torch.set_grad_enabled(training):
            output, spectral = forward_batch(model, stems_batch, aec_grid, device)
            loss = compressed_spectral_loss(
                output.enhanced, spectral.target,
                compression=loss_cfg.getfloat('compression'),
                magnitude_weight=loss_cfg.getfloat('magnitude_weight'),
                complex_weight=loss_cfg.getfloat('complex_weight'),
            )

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
            scheduler.step()
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
    model_kwargs = read_model_kwargs(cfg, AlignCRUSE)
    model = AlignCRUSE(model_grid, **model_kwargs).to(device)
    print(f"Align-CRUSE: {sum(p.numel() for p in model.parameters())} params, "
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
    min_lr = cfg.getfloat('training', 'min_lr', fallback=1e-6)
    warmup_lr = cfg.getfloat('training', 'lr_warmup', fallback=1e-4)
    warmup_ep = cfg.getint('training', 'warmup_epochs', fallback=3)
    max_epochs = cfg.getint('training', 'max_epochs', fallback=100)
    weight_decay = cfg.getfloat('training', 'weight_decay', fallback=1e-4)
    amsgrad = cfg.getboolean('training', 'amsgrad', fallback=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay, amsgrad=amsgrad)
    # Per-step linear warmup into cosine annealing.  This trainer previously had
    # no scheduler at all: the lr stayed at its initial value for the whole run,
    # so the late epochs kept taking early-epoch-sized steps and the weights
    # never settled.  The LR trajectory is part of the comparison protocol --
    # candidates trained over "the same 100 epochs" must be on the same one.
    total_steps = max_epochs * len(train_loader)
    warmup_steps = min(warmup_ep * len(train_loader), total_steps - 1)
    scheduler = make_scheduler(
        optimizer, warmup_steps, total_steps, lr, min_lr, warmup_lr,
    )

    start_epoch, global_step, best_val = 0, 0, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        require_checkpoint_contract(ckpt, contract, context=args.resume)
        poisoned = scan_non_finite(model)
        model.load_state_dict(ckpt['state_dict'])
        if scan_non_finite(model) and not poisoned:
            raise NonFiniteTraining(f"{args.resume} contains non-finite weights")
        if not args.reset_optimizer:
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            global_step = ckpt['global_step']
            best_val = ckpt.get('best_val', best_val)
        # Rebuilt, never restored -- fast_forward_scheduler()'s docstring has
        # the measured reason a stored T_max must not come back.
        resumed_lr = fast_forward_scheduler(scheduler, global_step)
        print(f"Resumed from {args.resume} at epoch {start_epoch}"
              f"{' (fresh optimizer)' if args.reset_optimizer else ''}"
              f", lr={resumed_lr:.4e}")

    loss_cfg = cfg['loss'] if cfg.has_section('loss') else {
        'compression': '0.3', 'magnitude_weight': '1.0', 'complex_weight': '1.0'}
    patience = cfg.getint('training', 'early_stop_patience', fallback=15)
    grad_clip = cfg.getfloat('training', 'grad_clip', fallback=1.0)
    grad_log = GradNormLog(os.path.join(output_dir, 'grad_norm.csv'), aec_grid.sr)

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
            max_epochs=max_epochs, scheduler=scheduler,
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
