"""AINR trainer 共用機制：非有限值攔截、權重塌陷攔截、梯度範數追蹤、LR schedule。

這裡放的是「每個 trainer 都需要、而且必須完全一致」的東西。理由與
``dataset_gen`` 相同：sampler / seeder / train-val split 若被各個 trainer 各
自複製一份就會漂移——例如一個留 5% 驗證、另一個留 10%，被拿來互相比較的
兩個模型就訓練在不同的語料上。非有限值的處置與 LR 軌跡同樣是比較協定
的一部分：兩個模型若一個在第 32 個 epoch 還停在初始 lr、另一個早已 cosine 退火
到底，那「跑滿 100 epochs」在兩邊不是同一件事。

因此這個模組是唯一定義，trainer 只負責提供自己的目標函式與資料。與模型有關的
部分透過參數傳入（見 ``dump_batch`` 的 ``hazard_mag``），不在這裡分支。
"""

import os
from collections import deque

import torch


class NonFiniteTraining(RuntimeError):
    """Raised to HALT training the first time a NaN/inf reaches the optimizer.

    ⚠ Deliberately stricter than upstream, which SKIPS the batch and tolerates
    up to MAX_NANS = 50 (df/train.py:43,380-419). Halting on the first hit buys
    what skipping cannot: model and optimizer state at that moment are still
    UNCONTAMINATED because nothing has stepped, and that is the state worth
    resuming from.
    """


def scan_non_finite(module, include_buffers=False):
    """Non-finite (name, n_nan, n_inf, numel) rows, worst first.

    Answers what a batch dump alone cannot: is this the FIRST hit, or were the
    weights already poisoned by an earlier step? ⚠ Buffers are EXCLUDED by
    default, and that is load-bearing: ``BatchNorm2d`` updates its running
    stats inside ``forward()`` in train mode, so a forward-side fault poisons
    every BN buffer before the loss exists and scanning them would report "an
    earlier step already wrote NaN" at step 0.
    """
    tensors = list(module.named_parameters())
    if include_buffers:
        tensors += list(module.named_buffers())
    with torch.no_grad():
        floats = [(n, t) for n, t in tensors if t.is_floating_point()]
        if not floats:
            return []
        # One fused check, one device sync.  Counting per tensor up front cost two
        # syncs each (168 for this model) to answer a question that is "no" on
        # every call but the last one of a run.
        if bool(torch.stack([torch.isfinite(t).all() for _, t in floats]).all()):
            return []
        bad = [
            (n, int(t.isnan().sum()), int(t.isinf().sum()), t.numel())
            for n, t in floats
            if not bool(torch.isfinite(t).all())
        ]
    bad.sort(key=lambda row: row[1] + row[2], reverse=True)
    return bad


class VanishedWeights(RuntimeError):
    """Raised to HALT training when a parameter tensor's scale has collapsed.

    Deliberately NOT a ``NonFiniteTraining``: the values are FINITE. A GRU
    block decayed to 1e-25 passes ``scan_non_finite`` and every
    ``torch.isfinite`` assertion, exports to an ONNX graph that agrees with
    PyTorch bit for bit, and has no offending batch to dump.
    """


#: A tensor counts as collapsed only when BOTH hold; ``WeightScaleGuard``'s
#: docstring has the measurements behind the two numbers.
VANISHED_RATIO = 1e-6
VANISHED_FLOOR = 1e-12


class WeightScaleGuard:
    """Per-tensor ``max|w|`` watch that halts when one tensor's scale
    collapses.

    ⚠ PER-TENSOR is the whole point: one of two parallel subband GRU blocks can
    collapse to ~1e-25 while its twin stays at 0.6, and the global gradient
    norm, the loss, ``scan_non_finite`` and an ONNX parity check all stay
    healthy. Each tensor is judged against ITS OWN scale, since many biases
    initialize to exactly zero and a bare ``max|w| < eps`` would fire at step
    0. The baseline is the RUNNING maximum of ``max|w|`` (so a zero-initialized
    bias that grows and then collapses is caught); under ``--resume`` it starts
    from the loaded weights.

    ``floor`` (1e-12) and ``ratio`` (1e-6) are ANDed: a false positive needs a
    tensor that both fell by a factor of a million AND ended below 1e-12, while
    healthy decay shrinks a peak by a fraction of a decade over a whole run. A
    tensor whose running peak never exceeded ``floor`` is skipped. Callers run
    it ONCE PER EPOCH, before the checkpoint write.
    """

    def __init__(self, model, ratio=VANISHED_RATIO, floor=VANISHED_FLOOR):
        # The module is held rather than passed per call so the peaks cannot be
        # compared against a different module than they were keyed from -- the
        # trainer that wraps its model for compilation has both to hand.
        self.model = model
        self.ratio = ratio
        self.floor = floor
        self.peak = {}
        self._update()

    def _update(self):
        """Raise each tracked peak to today's ``max|w|``; return today's values.

        NaN never becomes a peak and never trips the test: every comparison
        against it is False.  That is ``scan_non_finite``'s question, not this
        one.
        """
        floats = [(name, param) for name, param in self.model.named_parameters()
                  if param.is_floating_point()]
        if not floats:
            return {}
        with torch.no_grad():
            # One stack, one device sync -- the same reason scan_non_finite
            # fuses its check instead of syncing once per tensor.
            values = torch.stack(
                [param.detach().abs().max() for _, param in floats]
            ).tolist()
        current = {}
        for (name, _), value in zip(floats, values):
            current[name] = value
            # max() keeps its FIRST argument when the comparison is False, so a
            # NaN never becomes a peak.
            self.peak[name] = max(self.peak.get(name, 0.0), value)
        return current

    def check(self, *, epoch, global_step):
        """Halt if any tensor's ``max|w|`` has collapsed away from its own peak."""
        current = self._update()
        collapsed = [
            (name, self.peak[name], value, value / self.peak[name])
            for name, value in current.items()
            if self.peak[name] > self.floor
            and value < self.floor
            and value < self.ratio * self.peak[name]
        ]
        if not collapsed:
            return
        collapsed.sort(key=lambda row: row[3])

        report = [
            f'epoch         : {epoch}',
            f'global step   : {global_step}',
            f'test          : max|w| < {self.floor:.1e} AND < {self.ratio:.1e} '
            f"x the tensor's own running peak",
            '',
            f'{len(collapsed)} parameter tensor(s) collapsed:',
        ]
        for name, peak, value, ratio in collapsed:
            report.append(
                f'    {name}: peak {peak:.6e} -> now {value:.6e} '
                f'(ratio {ratio:.3e})'
            )
        print('\n' + '=' * 68)
        print('HALTED: a parameter tensor has decayed to nothing')
        print('=' * 68)
        print('\n'.join(report))
        print('\nthese values are FINITE, so no NaN check and no ONNX parity')
        print('check can see them -- an export reproduces the dead branch exactly')
        print('no checkpoint was written for this epoch; the newest one on disk')
        print('is the previous epoch\'s')
        print('=' * 68)
        raise VanishedWeights(
            'collapsed parameter tensor(s): ' + ', '.join(
                f'{name} {peak:.3e}->{value:.3e}'
                for name, peak, value, _ in collapsed[:5]
            )
        )

def dump_batch(dump_dir, noisy, clean, sr, enhanced=None, hazard_mag=None):
    """Write the batch as a .pt plus one wav pair per lane, and describe it.

    The audio IS the evidence: a spike's cause (near-silent target, heavy
    spectral tilt, silent top bands) is audible in a spectrogram but invisible
    in any logged scalar. ``enhanced`` is the model output for the same batch
    (the gradient hazard lives on the PREDICTION side). ``hazard_mag`` is the
    prediction magnitude at which THIS model's objective has its worst gradient
    gain (None if it has none; the near_hazard%% column then reads '-'). ⚠ Read
    the column as a HINT: the supplied thresholds come from an STFT magnitude
    while the column measures |x| on the waveform, so a near-zero percentage
    does not establish the batch was clear of the hazard.
    """
    os.makedirs(dump_dir, exist_ok=True)
    payload = {'noisy': noisy.detach().cpu(), 'clean': clean.detach().cpu()}
    if enhanced is not None:
        payload['enhanced'] = enhanced.detach().cpu()
    torch.save(payload, os.path.join(dump_dir, 'batch.pt'))

    signals = [('noisy', noisy), ('clean', clean)]
    if enhanced is not None:
        signals.append(('enhanced', enhanced))
    try:
        import torchaudio
        for i in range(noisy.shape[0]):
            for tag, wav in signals:
                torchaudio.save(
                    os.path.join(dump_dir, f'{tag}_{i:02d}.wav'),
                    wav[i:i + 1].detach().cpu().float(), sr,
                )
    except Exception as exc:                       # noqa: BLE001
        print(f'  (wav dump skipped: {exc}; batch.pt still written)')

    # Per-lane summary: which lane is the suspect, without opening the wavs.
    # near_hazard counts samples inside a decade of hazard_mag -- a time-domain
    # proxy, but a lane with many of them is the one to open.
    lines = ['lane  clean_rms   noisy_rms   clean_peak  '
             'silent_target?  enh_rms     near_hazard%']
    lo = hi = None
    if hazard_mag is not None:
        lo, hi = hazard_mag / 3.0, hazard_mag * 3.0
    with torch.no_grad():
        for i in range(noisy.shape[0]):
            c, n = clean[i].detach().float(), noisy[i].detach().float()
            c_rms = float(c.pow(2).mean().sqrt())
            if enhanced is not None:
                e = enhanced[i].detach().float()
                e_rms = f'{float(e.pow(2).mean().sqrt()):.3e}'
                if lo is None:
                    pct = '-'
                else:
                    mag = e.abs()
                    frac = float(((mag > lo) & (mag < hi)).float().mean())
                    pct = f'{100.0 * frac:.2f}'
            else:
                e_rms, pct = '-', '-'
            lines.append(
                f'{i:>4}  {c_rms:.3e}   {float(n.pow(2).mean().sqrt()):.3e}   '
                f'{float(c.abs().max()):.3e}   {str(c_rms < 1e-8):<14}  '
                f'{e_rms:<11} {pct}'
            )
    with open(os.path.join(dump_dir, 'lanes.txt'), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def halt_on_non_finite(reason, *, model, noisy, clean,
                       epoch, batch_idx, global_step, loss_value, total_norm,
                       output_dir, sr, checkpoint, enhanced=None,
                       hazard_mag=None):
    """Dump everything needed to diagnose the hit, then raise NonFiniteTraining.

    Called BEFORE optimizer.step(), so `checkpoint` still holds pre-contamination
    weights and moments.  The optimizer and the scheduler are NOT
    parameters: everything this function reports about them already travels
    inside ``checkpoint``, and taking them anyway made every caller assemble two
    values that went nowhere.
    """
    dump_dir = os.path.join(output_dir, 'nan_halt',
                            f'e{epoch}_b{batch_idx}_s{global_step}')
    os.makedirs(dump_dir, exist_ok=True)

    poisoned = scan_non_finite(model)
    report = [
        f'reason        : {reason}',
        f'epoch         : {epoch}',
        f'batch index   : {batch_idx}',
        f'global step   : {global_step}',
        f'loss          : {loss_value}',
        f'pre-clip norm : {total_norm}',
        '',
        'model PARAMETERS already non-finite before this step:',
    ]
    if poisoned:
        report.append(
            '  ⚠ NOT the first hit -- an earlier step already wrote NaN into '
            'the weights, so this batch is a symptom, not the cause.'
        )
        for name, n_nan, n_inf, numel in poisoned:
            report.append(f'    {name}: {n_nan} NaN, {n_inf} inf of {numel}')
    else:
        report.append(
            '  none -- weights and Adam moments are clean, so THIS batch '
            'produced the non-finite value.  The dumped audio is the evidence.'
        )
    # ⚠ Reported separately, and NOT used for the first-hit verdict: BatchNorm
    # writes running_mean/running_var during forward() in train mode, so a
    # forward-side fault poisons every BN buffer before the loss exists.  Those
    # are a consequence of THIS batch, not evidence of an earlier one.
    tainted_buffers = [
        row for row in scan_non_finite(model, include_buffers=True)
        if row not in poisoned
    ]
    if tainted_buffers:
        report.append('')
        report.append(
            f'buffers written by THIS batch (BatchNorm running stats etc.), '
            f'{len(tainted_buffers)} of them -- expected on a forward-side '
            f'fault, not a sign of earlier damage:'
        )
        for name, n_nan, n_inf, numel in tainted_buffers[:6]:
            report.append(f'    {name}: {n_nan} NaN, {n_inf} inf of {numel}')
    text = '\n'.join(report)
    with open(os.path.join(dump_dir, 'report.txt'), 'w') as handle:
        handle.write(text + '\n')

    dump_batch(dump_dir, noisy, clean, sr, enhanced=enhanced,
               hazard_mag=hazard_mag)
    # ⚠ Strip non-finite buffers before saving.  Weights and optimizer moments
    # really are pre-step, but BatchNorm's running stats were written by this
    # batch's forward pass, and leaving them in makes --resume reject the very
    # artefact this banner tells the operator to resume from.  A BN buffer is
    # re-estimated within a few hundred steps; the weights are what matter.
    stripped = []
    for name, tensor in list(model.named_buffers()):
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            checkpoint['state_dict'][name] = torch.nan_to_num(
                tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0
            )
            stripped.append(name)
    torch.save(checkpoint, os.path.join(dump_dir, 'pre_step.pth'))

    print('\n' + '=' * 68)
    print('HALTED: non-finite value reached the optimizer')
    print('=' * 68)
    print(text)
    print(f'\ndumped to {dump_dir}')
    print('  batch.pt / *.wav  -- the offending batch, listen to it')
    print('  lanes.txt         -- per-lane RMS/peak, finds the suspect lane')
    print('  pre_step.pth      -- pre-step weights and Adam moments; '
          'resume from this')
    if stripped:
        print(f'                      ({len(stripped)} non-finite buffer(s) '
              f'zeroed so --resume accepts it; BN stats re-estimate)')
    print('=' * 68)
    raise NonFiniteTraining(reason)


class GradNormLog:
    """Per-step pre-clip gradient-norm trace, plus finite-spike batch dumps.

    ⚠ Needed for the diagnosis: with grad_clip = 1.0 every gradient above 1.0
    is scaled to exactly 1.0, so the loss curve cannot distinguish a few
    enormous isolated spikes (a pathological batch) from bounded gradients with
    a wrong step direction (an optimizer-state problem), which have opposite
    fixes. A finite spike is DUMPED but does not halt (still a legal step);
    only NaN/inf halts.
    """

    def __init__(self, path, sr, spike_ratio=100.0, window=1000,
                 min_samples=100, max_dumps=20, hazard_mag=None):
        self.sr = sr
        self.spike_ratio = spike_ratio
        self.min_samples = min_samples
        self.max_dumps = max_dumps
        self.hazard_mag = hazard_mag
        self.n_dumps = 0
        self.window = deque(maxlen=window)
        new = not os.path.exists(path)
        # Line buffering: the trace survives a hard kill, and one ~48-byte line
        # per step is free next to a training step.
        self.handle = open(path, 'a', buffering=1)
        if new:
            self.handle.write('global_step,epoch,batch_idx,total_norm,loss\n')

    def record(self, norm, *, epoch, batch_idx, global_step, loss_value,
               noisy, clean, output_dir, enhanced=None):
        """Log one step; dump the batch if the norm is a finite outlier.

        ``norm`` and ``loss_value`` arrive as PYTHON FLOATS already synced by
        the caller, avoiding a second device->host copy of two scalars.
        """
        self.handle.write(
            f'{global_step},{epoch},{batch_idx},{norm:.6e},{loss_value:.6e}\n'
        )
        median = None
        if len(self.window) >= self.min_samples:
            ordered = sorted(self.window)
            median = ordered[len(ordered) // 2]
            if (
                median > 0
                and norm > self.spike_ratio * median
                and self.n_dumps < self.max_dumps
            ):
                self.n_dumps += 1
                dump_dir = os.path.join(
                    output_dir, 'grad_spikes',
                    f'e{epoch}_b{batch_idx}_s{global_step}',
                )
                dump_batch(dump_dir, noisy, clean, self.sr,
                           enhanced=enhanced, hazard_mag=self.hazard_mag)
                with open(os.path.join(dump_dir, 'report.txt'), 'w') as handle:
                    handle.write(
                        f'finite gradient spike\n'
                        f'pre-clip norm   : {norm:.6e}\n'
                        f'rolling median  : {median:.6e}\n'
                        f'ratio           : {norm / median:.1f}x\n'
                        f'loss            : {loss_value:.6e}\n'
                        f'epoch/batch/step: {epoch}/{batch_idx}/{global_step}\n'
                    )
                print(f'\n  ⚠ grad spike {norm / median:.0f}x median at '
                      f'epoch {epoch} batch {batch_idx} -> {dump_dir}')
                if self.n_dumps == self.max_dumps:
                    print('  (spike dump cap reached; the CSV keeps recording '
                          'every step)')
        self.window.append(norm)

    def close(self):
        self.handle.close()


# ============================================================
# Scheduler
# ============================================================

def make_scheduler(
    optimizer,
    warmup_steps,
    total_steps,
    base_lr,
    min_lr,
    warmup_lr,
):
    start_factor = warmup_lr / base_lr
    if not 0 < start_factor <= 1:
        raise ValueError('lr_warmup must be in (0, lr]')
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=max(1, warmup_steps),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[max(1, warmup_steps)],
    )



def fast_forward_scheduler(scheduler, global_step):
    """Advance a freshly built scheduler to ``global_step`` and return the lr.

    ⚠ A resumed run must NOT restore the scheduler from its checkpoint:
    ``load_state_dict`` brings back ``CosineAnnealingLR``'s ``T_max`` from the
    run that wrote it, overwriting the freshly built ``max(1, total_steps -
    warmup_steps)``, and neither ``epochs`` nor ``batch_size`` is in the
    checkpoint contract, so a resume at a different epoch count ends at the
    wrong terminal lr.
    """
    for _ in range(global_step):
        scheduler.step()
    return scheduler.optimizer.param_groups[0]['lr']
