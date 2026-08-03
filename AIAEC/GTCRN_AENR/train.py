#!/usr/bin/env python3
"""GTCRN-AENR training -- linear AEC -> joint RES+NR project variant.

用法:
    python3 train.py --config config.ini --packed-dir data_aec_16k/packed/all --gpu 0 --mmap
    python3 train.py --config config.ini --device cpu
    python3 train.py --config config.ini --resume output/gtcrn_aenr_e5.pth
    python3 train.py --config config.ini --resume output/gtcrn_aenr_e5.pth --reset-optimizer
    python3 train.py --config config.ini --seed 123

config.ini sections (see the shipped config.ini for every knob, documented):
    [signal]     sample rate + FFT grid. This candidate is LOCKED to upstream
                 16 kHz / 512/512/256 (GTCRNAENR itself rejects other grids);
                 must equal the grid AIAEC/dataset_gen/config.ini rendered with.
    [data]       one packed five-stem corpus path + ordinary batch size +
                 per-chunk val_fraction
    [model]      every GTCRNAENR constructor keyword (see model.py)
    [training]   optimizer, seed, epoch budget, checkpoint/log locations
    [loss]       compressed_spectral_loss term weights

dataset:
    AIAEC/dataset_gen renders each complete parent sequence, runs one stateful
    frozen Python PBFDKF over it, stores the resulting ``linear_error`` as
    the last channel, then cuts 10-second chunks. This trainer never executes
    PBFDKF. It uses the common deterministic per-chunk random split and an
    ordinary shuffled training DataLoader.

task (see ../README.md's decision matrix and
../dataset_gen/model_views.py MODEL_TASKS['GTCRN_AENR']):
    frozen-linear-AEC error + far-end reference in; clean near-end speech out.
    Requires the stored real linear-AEC error; an oracle residual is rejected
    by ``build_model_view``.

inference: see denoise.py's own top-of-file usage comment.
"""

import argparse
import configparser
import os
import sys
import time

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.GTCRN_AENR import GTCRNAENR
from AIAEC.dataset_gen import (
    AecStems,
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
    make_checkpoint_contract,
    read_grids,
    read_model_kwargs,
    require_checkpoint_contract,
    scan_non_finite,
    set_seed,
)


MODEL_NAME = 'GTCRN_AENR'
TASK = 'linear_aec_postfilter_res_nr'
LOSS_VERSION = 'aiaec_compressed_spectral_v1'


def build_parser() -> argparse.ArgumentParser:
    return build_arg_parser('Train GTCRN-AENR (linear AEC -> RES+NR)')


def forward_batch(model, stems_batch, aec_grid, device):
    stems = AecStems(stems_batch.to(device))
    view = build_model_view(stems, MODEL_NAME, sample_rate=aec_grid.sr)
    spectral = build_spectral_model_view(view, aec_grid)
    output = model(**spectral.inputs)
    return output, spectral


def run_epoch(model, loader, aec_grid, device, loss_cfg, optimizer=None, *,
             epoch=0, global_step=0, output_dir=None, sr=None,
             grad_clip=1.0, checkpoint_for_halt=None, grad_log=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches = 0.0, 0

    for batch_idx, (stems_batch, _meta) in enumerate(loader):
        with torch.set_grad_enabled(training):
            output, spectral = forward_batch(
                model, stems_batch, aec_grid, device
            )
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
                    'non-finite loss', model=model, optimizer=optimizer,
                    mic=spectral.inputs['linear_error'], target=spectral.target,
                    epoch=epoch, batch_idx=batch_idx, global_step=global_step,
                    loss_value=loss_value, total_norm=float('nan'),
                    output_dir=output_dir, sr=sr, checkpoint=checkpoint_for_halt,
                    enhanced=output.enhanced,
                )
            loss.backward()
            total_norm = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            if not torch.isfinite(torch.tensor(total_norm)):
                halt_on_non_finite(
                    'non-finite gradient norm', model=model, optimizer=optimizer,
                    mic=spectral.inputs['linear_error'], target=spectral.target,
                    epoch=epoch, batch_idx=batch_idx, global_step=global_step,
                    loss_value=loss_value, total_norm=total_norm,
                    output_dir=output_dir, sr=sr, checkpoint=checkpoint_for_halt,
                    enhanced=output.enhanced,
                )
            optimizer.step()
            if grad_log is not None:
                grad_log.record(
                    total_norm, epoch=epoch, batch_idx=batch_idx,
                    global_step=global_step, loss_value=loss_value,
                    noisy=spectral.inputs['linear_error'], clean=spectral.target,
                    output_dir=output_dir, enhanced=output.enhanced,
                )
            global_step += 1
        else:
            loss_value = float(loss.detach())

        total_loss += loss_value
        n_batches += 1

    return total_loss / max(1, n_batches), global_step


def main(args):
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")

    set_seed(args.seed)
    device = auto_device(args.device, args.gpu)
    print(f"Device: {device}")

    aec_grid, model_grid = read_grids(cfg)
    model_kwargs = read_model_kwargs(cfg, GTCRNAENR)
    model = GTCRNAENR(model_grid, **model_kwargs).to(device)
    print(f"GTCRN-AENR: {sum(p.numel() for p in model.parameters())} params, "
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
    weight_decay = cfg.getfloat('training', 'weight_decay', fallback=1e-4)
    amsgrad = cfg.getboolean('training', 'amsgrad', fallback=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay, amsgrad=amsgrad)

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
        print(f"Resumed from {args.resume} at epoch {start_epoch}"
              f"{' (fresh optimizer)' if args.reset_optimizer else ''}")

    loss_cfg = cfg['loss'] if cfg.has_section('loss') else {
        'compression': '0.3', 'magnitude_weight': '1.0', 'complex_weight': '1.0'}
    max_epochs = cfg.getint('training', 'max_epochs', fallback=100)
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
        )
        msg = f"epoch {epoch}: train_loss={train_loss:.4f} ({time.time()-started:.0f}s)"

        val_loss = train_loss
        if val_loader is not None:
            val_loss, _ = run_epoch(
                model, val_loader, aec_grid, device, loss_cfg
            )
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
