#!/usr/bin/env python3
"""Align-ULCNet training -- linear AEC -> joint RES+NR paper reference.

用法 (--config 必填，每個板端 grid 一份自足的 config: config_16k_256.ini / config_16k_512.ini / config_48k.ini):
    python3 train.py --config config_16k_512.ini --packed-dir data_aec_16k/packed/all --gpu 0 --mmap
    python3 train.py --config config_16k_512.ini --device cpu
    python3 train.py --config config_16k_512.ini --resume output_16k_512/align_ulcnet_e5.pth
    python3 train.py --config config_16k_512.ini --resume output_16k_512/align_ulcnet_e5.pth --reset-optimizer
    python3 train.py --config config_16k_512.ini --seed 123
    ([loss] stft_consistency = true refuses MPS: pass --device cpu or a CUDA device)

config_<grid>.ini sections (see a shipped config for every knob, documented):
    [signal]     sample rate + FFT grid. Paper defaults are 16 kHz/512/512/256;
                 this project also accepts 48 kHz/1024/1024/512 (see
                 model.py/README.md) -- must equal the grid
                 AIAEC/dataset_gen/config.ini rendered with.
    [data]       one packed four-stem corpus path + ordinary batch size +
                 per-chunk val_fraction
    [model]      every AlignULCNet constructor keyword (see model.py)
    [training]   optimizer, seed, epoch budget, checkpoint/log locations
    [loss]       PLCPA on the iSTFT->STFT-consistent estimate, as DeepVQE_S
                 and Align_CRUSE train; optional aux_stft_weight / aux_n_fft
                 term (training_common.AuxStftLoss)

dataset:
    AIAEC/dataset_gen renders each complete parent sequence, runs one stateful
    frozen Python PBFDKF over it, stores the resulting ``linear_error`` as the
    last channel, then cuts 10-second chunks. This trainer never executes
    PBFDKF; it uses the common deterministic per-chunk random split and an
    ordinary shuffled training DataLoader.

task (see ../README.md's decision matrix and
../dataset_gen/model_views.py MODEL_TASKS['Align_ULCNet']):
    frozen-linear-AEC error + far-end reference in; denoised, echo-free,
    early/dereverberated near-end speech out. Requires the stored real
    linear-AEC error; an oracle residual is rejected by ``build_model_view``.

inference: see inference.py's own top-of-file usage comment.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.Align_ULCNet import AlignULCNet
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
    aux_loss_version,
    aux_recipe,
    aux_stft_loss_from_config,
    contract_model_kwargs,
    fast_forward_scheduler,
    halt_on_non_finite,
    make_scheduler,
    make_checkpoint_contract,
    plcpa_log,
    plcpa_training_loss,
    read_config,
    read_grids,
    read_model_kwargs,
    require_checkpoint_contract,
    scan_non_finite,
    set_seed,
    training_progress,
)


MODEL_NAME = 'Align_ULCNet'
TASK = MODEL_TASKS[MODEL_NAME]
LOSS_VERSION = 'align_ulcnet_plcpa_c03_beta07_consistent_v1'


def build_parser() -> argparse.ArgumentParser:
    return build_arg_parser('Train Align-ULCNet (linear AEC -> RES+NR)',
                            default_config=None)


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
             max_epochs=None, scheduler=None, aux_loss=None):
    """One pass over ``loader``; returns ``(mean loss, global_step, mean
    PLCPA)``. The PLCPA alone is tracked only while ``aux_loss`` adds its
    term (None otherwise), so runs with and without it compare on one scale.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, plcpa_sum, n_batches = 0.0, 0.0, 0

    batches = training_progress(
        loader, training=training, epoch=epoch, max_epochs=max_epochs
    )
    for batch_idx, (stems_batch, _meta) in enumerate(batches):
        with torch.set_grad_enabled(training):
            output, spectral = forward_batch(
                model, stems_batch, aec_grid, device
            )
            loss, plcpa = plcpa_training_loss(
                output.enhanced, spectral.target, aec_grid, loss_cfg,
                consistency=loss_cfg.getboolean('stft_consistency'),
                aux_loss=aux_loss,
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach())
            if not torch.isfinite(loss.detach()):
                halt_on_non_finite(
                    'non-finite loss', model=model,
                    mic=spectral.inputs['linear_error'], target=spectral.target,
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
                    mic=spectral.inputs['linear_error'], target=spectral.target,
                    epoch=epoch, batch_idx=batch_idx, global_step=global_step,
                    loss_value=loss_value, total_norm='non-finite',
                    output_dir=output_dir, sr=sr, checkpoint=checkpoint_for_halt,
                    enhanced=output.enhanced,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
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
        if aux_loss is not None:
            plcpa_sum = plcpa_sum + plcpa.detach()
        n_batches += 1
        if training:
            batches.set_postfix(
                loss=f"{loss_value:.4f}", gnorm=f"{total_norm:.2e}",
                refresh=False,
            )

    plcpa_mean = (None if aux_loss is None
                  else float(plcpa_sum) / max(1, n_batches))
    return total_loss / max(1, n_batches), global_step, plcpa_mean


def main(args):
    cfg = read_config(args.config)

    set_seed(args.seed)
    device = auto_device(args.device, args.gpu)
    print(f"Device: {device}")

    aec_grid, model_grid = read_grids(cfg)
    model_kwargs = read_model_kwargs(cfg, AlignULCNet)
    model = AlignULCNet(model_grid, **model_kwargs).to(device)
    print(f"Align-ULCNet: {sum(p.numel() for p in model.parameters())} params, "
          f"grid={model_grid}, max_delay_frames={model.max_delay_frames}")

    train_loader, val_loader, data_contract = build_plain_loaders(
        cfg, aec_grid, seed=args.seed,
        packed_dir=args.packed_dir, mmap=args.mmap,
    )
    # The resolved numeric recipe, read ONCE and used both to record the
    # contract and to build the optimizer below -- so what the checkpoint says
    # trained the weights cannot drift from what did. No fallbacks: every key
    # is in the shipped config, and a fallback would let one be deleted and
    # silently re-supplied from source.
    recipe = {
        'name': cfg.get('training', 'optimizer').lower(),
        'lr': cfg.getfloat('training', 'lr'),
        'weight_decay': cfg.getfloat('training', 'weight_decay'),
        'amsgrad': cfg.getboolean('training', 'amsgrad'),
        'schedule': cfg.get('training', 'scheduler').lower(),
        'min_lr': cfg.getfloat('training', 'min_lr'),
        'max_epochs': cfg.getint('training', 'max_epochs', fallback=50),
        'batch_size': cfg.getint('data', 'batch_size'),
        'grad_clip': cfg.getfloat('training', 'grad_clip', fallback=1.0),
        'early_stop_patience': cfg.getint(
            'training', 'early_stop_patience', fallback=0),
        'loss_compression': cfg.getfloat('loss', 'compression'),
        'loss_complex_weight': cfg.getfloat('loss', 'complex_weight'),
        'loss_stft_consistency': cfg.getboolean('loss', 'stft_consistency'),
    }
    aux_loss = aux_stft_loss_from_config(cfg)
    recipe.update(aux_recipe(aux_loss))
    if recipe['schedule'] == 'warmup_cosine':
        # These two values and the denominator above define the entire step LR
        # curve, so all are checkpoint compatibility fields.
        recipe.update({
            'lr_warmup': cfg.getfloat('training', 'lr_warmup'),
            'warmup_epochs': cfg.getint('training', 'warmup_epochs'),
        })
    elif recipe['schedule'] == 'reduce_on_plateau':
        recipe.update({
            'lr_decay_factor': cfg.getfloat('training', 'lr_decay_factor'),
            'lr_patience': cfg.getint('training', 'lr_patience'),
        })
    if recipe['early_stop_patience'] < 0:
        raise ValueError("early_stop_patience must be >= 0 (0 disables it)")
    stop_label = (str(recipe['early_stop_patience'])
                  if recipe['early_stop_patience'] else 'off')
    print(f"Training recipe: optimizer={recipe['name']} lr={recipe['lr']:.4g} "
          f"schedule={recipe['schedule']} early_stop={stop_label}")
    contract = make_checkpoint_contract(
        model_name=MODEL_NAME, task=TASK, grid=model_grid,
        model_kwargs=contract_model_kwargs(AlignULCNet, model_kwargs),
        loss_version=aux_loss_version(LOSS_VERSION, aux_loss),
        data_contract=data_contract, optimizer=recipe,
    )

    output_dir = cfg.get('training', 'output_dir', fallback='output')
    os.makedirs(output_dir, exist_ok=True)
    lr = recipe['lr']
    max_epochs = recipe['max_epochs']
    optimizer_classes = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}
    if recipe['name'] not in optimizer_classes:
        raise ValueError(
            f"unknown optimizer {recipe['name']!r}; expected one of "
            f"{sorted(optimizer_classes)}")
    optimizer = optimizer_classes[recipe['name']](
        model.parameters(), lr=lr,
        weight_decay=recipe['weight_decay'], amsgrad=recipe['amsgrad'],
    )
    # Two schedules, and which one is live decides where it is stepped:
    # warmup_cosine is indexed by optimizer step and advances inside
    # run_epoch; reduce_on_plateau consumes a validation metric that exists
    # only once per epoch and advances in the epoch loop. `step_scheduler` is
    # what run_epoch receives, so the two can never both fire.
    step_scheduler = None
    epoch_scheduler = None
    if recipe['schedule'] == 'warmup_cosine':
        # Deliberately NOT stored in the checkpoint: fast_forward_scheduler()'s
        # docstring carries the measured reason a saved T_max must not come
        # back on a resume with a different horizon.
        total_steps = max_epochs * len(train_loader)
        warmup_steps = min(
            recipe['warmup_epochs'] * len(train_loader), total_steps - 1)
        step_scheduler = make_scheduler(
            optimizer, warmup_steps, total_steps, lr,
            recipe['min_lr'], recipe['lr_warmup'],
        )
    elif recipe['schedule'] == 'reduce_on_plateau':
        # min_lr is a floor, not a paper value: the recipe gives a factor and
        # a patience but no bound, and PyTorch's default is 0.
        epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=recipe['lr_decay_factor'], patience=recipe['lr_patience'],
            min_lr=recipe['min_lr'],
        )
    else:
        raise ValueError(
            f"unknown scheduler {recipe['schedule']!r}; expected "
            "'warmup_cosine' or 'reduce_on_plateau'")

    start_epoch, global_step, best_val, no_improve = 0, 0, float('inf'), 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        require_checkpoint_contract(ckpt, contract, context=args.resume)
        model.load_state_dict(ckpt['state_dict'])
        poisoned = scan_non_finite(model)
        if poisoned:
            raise NonFiniteTraining(
                f"{args.resume} contains non-finite weights: {poisoned[:5]}"
            )
        if not args.reset_optimizer:
            optimizer.load_state_dict(ckpt['optimizer'])
            if epoch_scheduler is not None:
                # Plateau state IS restored: its counters (best, num_bad_epochs,
                # cooldown) are earned history with no horizon baked in, so
                # dropping them would hand a stalled run a fresh patience.
                epoch_scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            global_step = ckpt['global_step']
            best_val = ckpt.get('best_val', best_val)
            # Early stopping is training state just like the optimizer.  If it
            # restarts at zero on every resume, repeated short jobs can evade
            # the configured patience indefinitely.
            no_improve = ckpt.get('no_improve', 0)
        # The step schedule is rebuilt for THIS run's horizon and re-indexed by
        # global_step, never restored -- see fast_forward_scheduler().
        if step_scheduler is not None and not args.reset_optimizer:
            resumed_lr = fast_forward_scheduler(step_scheduler, global_step)
        else:
            resumed_lr = optimizer.param_groups[0]['lr']
        print(f"Resumed from {args.resume} at epoch {start_epoch}"
              f"{' (fresh optimizer)' if args.reset_optimizer else ''}"
              f", lr={resumed_lr:.4e}")

    loss_cfg = cfg['loss']
    patience = recipe['early_stop_patience']
    grad_clip = recipe['grad_clip']
    grad_log = GradNormLog(os.path.join(output_dir, 'grad_norm.csv'), aec_grid.sr)
    # Per-tensor, because grad_norm.csv above is a GLOBAL norm and stays healthy
    # while one branch's weights decay to nothing.  Built here, after any
    # --resume load, so a resumed run measures against what it resumed from.
    weight_guard = WeightScaleGuard(model)

    for epoch in range(start_epoch, max_epochs):
        checkpoint_for_halt = {
            'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': (epoch_scheduler.state_dict()
                          if epoch_scheduler is not None else None),
            'epoch': epoch - 1, 'global_step': global_step, 'contract': contract,
            'best_val': best_val, 'no_improve': no_improve,
        }
        started = time.time()
        train_loss, global_step, train_plcpa = run_epoch(
            model, train_loader, aec_grid, device, loss_cfg, optimizer,
            epoch=epoch, global_step=global_step, output_dir=output_dir,
            sr=aec_grid.sr, grad_clip=grad_clip,
            checkpoint_for_halt=checkpoint_for_halt, grad_log=grad_log,
            max_epochs=max_epochs, scheduler=step_scheduler,
            aux_loss=aux_loss,
        )
        msg = f"epoch {epoch}: train_loss={train_loss:.4f} ({time.time()-started:.0f}s)"

        val_loss, val_plcpa = train_loss, None
        if val_loader is not None:
            val_loss, _, val_plcpa = run_epoch(
                model, val_loader, aec_grid, device, loss_cfg,
                aux_loss=aux_loss,
            )
            msg += f" val_loss={val_loss:.4f}"
        msg += plcpa_log(train=train_plcpa, val=val_plcpa)
        if epoch_scheduler is not None:
            epoch_scheduler.step(val_loss)
        msg += f" lr={optimizer.param_groups[0]['lr']:.4e}"
        print(msg)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1

        checkpoint = {
            'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': (epoch_scheduler.state_dict()
                          if epoch_scheduler is not None else None),
            'epoch': epoch, 'global_step': global_step, 'contract': contract,
            'best_val': best_val, 'no_improve': no_improve,
        }
        poisoned = scan_non_finite(model)
        if poisoned:
            raise NonFiniteTraining(
                "refusing to write a checkpoint with non-finite weights: "
                f"{poisoned[:5]}"
            )
        weight_guard.check(epoch=epoch, global_step=global_step)
        torch.save(checkpoint, os.path.join(output_dir, f'{MODEL_NAME.lower()}_last.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(output_dir, f'{MODEL_NAME.lower()}_best.pth'))
        elif patience > 0 and no_improve >= patience:
            print(f"early stop: no improvement in {patience} epochs")
            break
    grad_log.close()


if __name__ == '__main__':
    main(build_parser().parse_args())
