#!/usr/bin/env python3
"""Align-ULCNet training -- linear AEC -> joint RES+NR paper reference.

用法:
    python3 train.py --config config.ini
    python3 train.py --config config.ini --device cpu
    python3 train.py --config config.ini --resume output/align_ulcnet_e5.pth
    python3 train.py --config config.ini --resume output/align_ulcnet_e5.pth --reset-optimizer
    python3 train.py --config config.ini --seed 123

config.ini sections (see the shipped config.ini for every knob, documented):
    [signal]     sample rate + FFT grid. Paper defaults are 16 kHz/512/512/256;
                 this project also accepts 48 kHz/1024/1024/512 (see
                 model.py/README.md) -- must equal the grid
                 AIAEC/dataset_gen/config.ini rendered with.
    [data]       packed corpus paths + batch size (= lane count, see below)
    [linear_aec] which frozen preset the linear-AEC frontend runs
    [model]      every AlignULCNet constructor keyword (see model.py)
    [training]   optimizer, seed, epoch budget, checkpoint/log locations
    [loss]       compressed_spectral_loss term weights

dataset:
    Data is rendered and packed by AIAEC/dataset_gen/ only -- see
    AIAEC/dataset_gen/README.md and ../Align_CRUSE/train.py's docstring for
    the full generation walkthrough (identical for every candidate).

    UNLIKE Align_CRUSE, this candidate's input includes the FROZEN PRODUCTION
    LINEAR AEC's error signal (AIAEC/training_common.LinearAecEngine), and
    that linear filter's own convergence state -- cold right after a sequence
    starts, progressively converged afterwards -- is real information the
    corpus's long stateful sequences were built to expose (see
    ../dataset_gen/README.md, "Why sequences are long"). Losing that
    stratification would make every training chunk look like a cold start.
    So training here uses ``SequenceChunkSampler`` (batch size IS lane count:
    lane k walks one physical sequence across consecutive steps) instead of
    Align_CRUSE's plain shuffle, and ``lane_reset_mask`` tells the shared
    LinearAecEngine exactly when a lane's filter must restart cold. The
    NEURAL model itself still sees one independent 3-second chunk per
    forward call -- see AIAEC/README.md's "public forwards are clip-level"
    note; carrying the model's own recurrent state across chunks is
    deliberately out of scope here, same as every other AIAEC candidate.

task (see ../README.md's decision matrix and
../dataset_gen/model_views.py MODEL_TASKS['Align_ULCNet']):
    frozen-linear-AEC error + far-end reference in; clean near-end speech out.
    Requires the real linear AEC (an oracle residual is rejected by
    ``build_model_view``), so this trainer is slower per step than
    Align_CRUSE/DeepVQE_S/CAGCRN -- see LinearAecEngine's own docstring for
    the reason and the (not-yet-built) offline-caching alternative.

inference: see denoise.py's own top-of-file usage comment.
"""

import argparse
import configparser
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.dataset_gen import (
    AecStems,
    PackedAecDataset,
    SequenceChunkSampler,
    aec_collate,
    build_model_view,
    build_spectral_model_view,
    lane_reset_mask,
)
from AIAEC.training_common import (
    GradNormLog,
    LinearAecEngine,
    NonFiniteTraining,
    build_arg_parser,
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


MODEL_NAME = 'Align_ULCNet'
TASK = 'linear_aec_postfilter_res_nr'
LOSS_VERSION = 'aiaec_compressed_spectral_v1'


def build_parser() -> argparse.ArgumentParser:
    return build_arg_parser('Train Align-ULCNet (linear AEC -> RES+NR)')


def make_loader_and_sampler(cfg, section: str, aec_grid, n_lanes: int, seed: int,
                            shuffle: bool):
    path = cfg.get(section, 'packed_dir')
    dataset = PackedAecDataset(path, expected_sr=aec_grid.sr)
    sampler = SequenceChunkSampler.from_dataset(
        dataset, n_lanes=n_lanes, seed=seed, shuffle=shuffle,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=aec_collate,
                        num_workers=cfg.getint('data', 'num_workers', fallback=0))
    return loader, sampler


def forward_batch(model, stems_batch, meta, aec_grid, device, linear_aec):
    reset_mask = lane_reset_mask([m['chunk_index'] for m in meta]).tolist()
    linear_aec.arm_reset(reset_mask)
    stems = AecStems(stems_batch.to(device))
    view = build_model_view(stems, MODEL_NAME, sample_rate=aec_grid.sr,
                            linear_aec=linear_aec)
    spectral = build_spectral_model_view(view, aec_grid)
    output = model(**spectral.inputs)
    return output, spectral


def run_epoch(model, loader, aec_grid, device, loss_cfg, linear_aec, optimizer=None, *,
             epoch=0, global_step=0, output_dir=None, sr=None,
             grad_clip=1.0, checkpoint_for_halt=None, grad_log=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches = 0.0, 0

    for batch_idx, (stems_batch, meta) in enumerate(loader):
        with torch.set_grad_enabled(training):
            output, spectral = forward_batch(model, stems_batch, meta, aec_grid,
                                             device, linear_aec)
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
    device = auto_device(args.device)
    print(f"Device: {device}")

    aec_grid, model_grid = read_grids(cfg)
    model_kwargs = read_model_kwargs(cfg, AlignULCNet)
    model = AlignULCNet(model_grid, **model_kwargs).to(device)
    print(f"Align-ULCNet: {sum(p.numel() for p in model.parameters())} params, "
          f"grid={model_grid}")

    contract = make_checkpoint_contract(
        model_name=MODEL_NAME, task=TASK, grid=model_grid,
        model_kwargs=model_kwargs, loss_version=LOSS_VERSION,
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

    batch_size = cfg.getint('data', 'batch_size')
    train_loader, train_sampler = make_loader_and_sampler(
        cfg, 'data', aec_grid, n_lanes=batch_size, seed=args.seed, shuffle=True)
    val_loader = None
    if cfg.has_section('val_data'):
        val_loader, _ = make_loader_and_sampler(
            cfg, 'val_data', aec_grid, n_lanes=batch_size, seed=args.seed,
            shuffle=False)

    linear_aec_cfg = cfg['linear_aec'] if cfg.has_section('linear_aec') else {}
    preset = linear_aec_cfg.get('preset', 'balanced')
    train_linear_aec = LinearAecEngine(batch_size, aec_grid.sr, preset=preset)
    val_linear_aec = LinearAecEngine(batch_size, aec_grid.sr, preset=preset) \
        if val_loader is not None else None

    loss_cfg = cfg['loss'] if cfg.has_section('loss') else {
        'compression': '0.3', 'magnitude_weight': '1.0', 'complex_weight': '1.0'}
    max_epochs = cfg.getint('training', 'max_epochs', fallback=100)
    patience = cfg.getint('training', 'early_stop_patience', fallback=15)
    grad_clip = cfg.getfloat('training', 'grad_clip', fallback=1.0)
    grad_log = GradNormLog(os.path.join(output_dir, 'grad_norm.csv'), aec_grid.sr)

    no_improve = 0
    for epoch in range(start_epoch, max_epochs):
        train_sampler.set_epoch(epoch)   # reshuffles which sequence sits in which lane
        checkpoint_for_halt = {
            'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch - 1, 'global_step': global_step, 'contract': contract,
        }
        started = time.time()
        train_loss, global_step = run_epoch(
            model, train_loader, aec_grid, device, loss_cfg, train_linear_aec,
            optimizer, epoch=epoch, global_step=global_step, output_dir=output_dir,
            sr=aec_grid.sr, grad_clip=grad_clip,
            checkpoint_for_halt=checkpoint_for_halt, grad_log=grad_log,
        )
        msg = f"epoch {epoch}: train_loss={train_loss:.4f} ({time.time()-started:.0f}s)"

        val_loss = train_loss
        if val_loader is not None:
            val_loss, _ = run_epoch(model, val_loader, aec_grid, device, loss_cfg,
                                    val_linear_aec)
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
