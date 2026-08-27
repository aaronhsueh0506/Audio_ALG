import sys
from pathlib import Path

import pytest
import torch

AINR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AINR))

from training_common import (  # noqa: E402
    VanishedWeights,
    WeightScaleGuard,
    fast_forward_scheduler,
    make_scheduler,
    scan_non_finite,
)

#: Every trainer that runs per-step warmup->cosine.  GTCRN joined when its
#: ReduceLROnPlateau was replaced: on a real run that schedule never fired once
#: in 32 epochs, because it compares against the best epoch ever seen.
_COSINE_TRAINERS = (
    "RNNoise-ERB/train.py",
    "GTCRN/train.py",
    "DeepFilterNet2/train.py",
)


def _source(relative: str) -> str:
    return (AINR / relative).read_text(encoding="utf-8")


def test_every_ainr_trainer_rejects_nonfinite_gradients():
    for relative in _COSINE_TRAINERS:
        source = _source(relative)
        assert "error_if_nonfinite=True" in source, relative
        assert "non-finite training loss" in source or "loss is non-finite" in source


def test_every_ainr_trainer_dumps_the_batch_it_halted_on():
    """A bare raise names the epoch and nothing else.

    The batch that produced the non-finite value is the only evidence of WHY,
    and it is gone the moment the process exits, so every trainer has to route
    its halt through the dumper rather than raising on its own.
    """
    for relative in _COSINE_TRAINERS:
        source = _source(relative)
        assert "halt_on_non_finite(" in source, relative
        assert "GradNormLog(" in source, relative
        assert "grad_log.record(" in source, relative


def test_every_ainr_trainer_watches_for_vanished_weights():
    """A finite collapse needs its own guard; the NaN path cannot reach it."""
    for relative in _COSINE_TRAINERS:
        source = _source(relative)
        assert "WeightScaleGuard(" in source, relative
        assert "weight_guard.check(" in source, relative


def test_cosine_trainers_rebuild_rather_than_restore_the_scheduler():
    for relative in _COSINE_TRAINERS:
        source = _source(relative)
        assert "scheduler.load_state_dict" not in source, relative
        assert "'global_step': global_step" in source, relative
        assert "fast_forward_scheduler(" in source, relative


def test_restoring_a_stale_tmax_really_does_break_the_floor():
    """The behaviour the test above is protecting, demonstrated both ways.

    ⚠ Written so it can FAIL: the ``restored`` branch reproduces the defect, and
    if ``load_state_dict`` ever stopped carrying ``T_max`` across, that branch
    would land on min_lr too and this test would say so instead of passing
    vacuously.
    """
    LR, MIN_LR, WARMUP, SHORT, LONG, RESUME_AT = 1e-3, 1e-6, 30, 1000, 1200, 600

    def build(total_steps):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([param], lr=LR)
        return opt, make_scheduler(opt, WARMUP, total_steps, LR, MIN_LR, 1e-4)

    # A checkpoint written by a 1000-step run, stopped at step 600.
    opt, sched = build(SHORT)
    for _ in range(RESUME_AT):
        sched.step()
    stale = sched.state_dict()

    # Wrong: restore into a run that was reconfigured for 1200 steps.
    opt_bad, sched_bad = build(LONG)
    sched_bad.load_state_dict(stale)
    for _ in range(LONG - RESUME_AT):
        sched_bad.step()
    restored = opt_bad.param_groups[0]['lr']

    # Right: rebuild for 1200 and index the fresh schedule by step.
    opt_ok, sched_ok = build(LONG)
    fast_forward_scheduler(sched_ok, RESUME_AT)
    for _ in range(LONG - RESUME_AT):
        sched_ok.step()
    rebuilt = opt_ok.param_groups[0]['lr']

    assert rebuilt == pytest.approx(MIN_LR, rel=0.05), rebuilt
    assert restored > 10 * MIN_LR, (
        f"restoring a stale T_max landed on {restored:.3e}, which is close "
        f"enough to min_lr that this test no longer proves anything"
    )


# Moved here from DeepFilterNet2's contract suite: the functions these two
# exercise left that model for training_common, so a per-model file is no
# longer where a reader would look for them.
def test_non_finite_gradient_is_refused_before_it_touches_the_model():
    """error_if_nonfinite must raise BEFORE clipping scales anything.

    Without the flag, total_norm=inf gives clip_coef=1/(inf+1e-6)=0 and inf*0
    becomes NaN, which the next step writes into the weights and into Adam's
    moments -- unrecoverable.  This asserts the ordering, not just that an
    exception appears: after the raise, gradients and weights are untouched.
    """
    model = torch.nn.Linear(4, 1)
    before = [p.detach().clone() for p in model.parameters()]
    (model(torch.ones(1, 4)).sum() * float('inf')).backward()

    with pytest.raises(RuntimeError):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )

    assert any(not torch.isfinite(p.grad).all() for p in model.parameters()), (
        'gradients were modified; the raise must happen before scaling'
    )
    for param, original in zip(model.parameters(), before):
        assert torch.equal(param.detach(), original)
    assert scan_non_finite(model) == []


def test_scan_non_finite_excludes_buffers_by_default():
    """⚠ Buffers must be OFF by default, and that default is load-bearing.

    BatchNorm writes running_mean/running_var during forward() in train mode, so a
    forward-side fault poisons every BN buffer before the loss exists.  Including
    them made the halt report claim "an earlier step already wrote NaN into the
    weights" at global step 0 -- the opposite of the truth -- and put non-finite
    buffers into the checkpoint it told the operator to resume from, which the
    resume guard then rejected.
    """
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    assert scan_non_finite(model) == []

    with torch.no_grad():
        model[1].running_mean[0] = float('nan')
    assert scan_non_finite(model) == [], (
        'a poisoned BN buffer must NOT register as a poisoned parameter'
    )
    with_buffers = scan_non_finite(model, include_buffers=True)
    assert [row[0] for row in with_buffers] == ['1.running_mean']

    with torch.no_grad():
        model[0].weight[0, 0] = float('inf')
    params_only = scan_non_finite(model)
    assert [row[0] for row in params_only] == ['0.weight']
    assert params_only[0][1:3] == (0, 1)


# ============================================================
# Vanished weights: the finite failure scan_non_finite cannot see
# ============================================================

def test_a_collapsed_tensor_halts_and_names_itself(capsys):
    """The premise and the catch, in one test.

    ⚠ Written so it can FAIL twice over: the first assertion states the premise
    (the collapsed values are FINITE, so a guard that merely re-ran
    ``scan_non_finite`` would be redundant), and the halt is matched on the
    TENSOR NAME, so a guard that fired on the wrong tensor -- or reported only
    "the model" -- would not satisfy it.
    """
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    guard = WeightScaleGuard(model)
    with torch.no_grad():
        model[1].weight.fill_(1e-20)

    assert scan_non_finite(model) == [], (
        'the collapsed tensor is finite; if scan_non_finite can see it, this '
        'guard is redundant'
    )
    with pytest.raises(VanishedWeights, match=r'1\.weight'):
        guard.check(epoch=7, global_step=700)

    banner = capsys.readouterr().out
    # Actionable means: which tensor, from what, to what, by how much, when.
    for expected in ('1.weight', 'epoch         : 7', 'global step   : 700',
                     '-> now 1.000000e-20', 'ratio'):
        assert expected in banner, banner


def test_a_zero_initialized_bias_does_not_halt_at_step_zero():
    """Why an absolute threshold alone is not a usable test.

    8 of Align-ULCNet's 67 parameter tensors are norm biases that initialize to
    exactly zero, so ``max|w| < eps`` fires on all of them before the first
    step.  The tensor is still TRACKED -- it is skipped for lack of a scale to
    fall from, not ignored.
    """
    model = torch.nn.Linear(4, 3)
    torch.nn.init.zeros_(model.bias)
    guard = WeightScaleGuard(model)

    assert guard.peak['bias'] == 0.0
    guard.check(epoch=0, global_step=0)


def test_a_tensor_that_never_had_a_scale_is_skipped_not_reported():
    """What the ``peak > floor`` skip is actually load-bearing for.

    A bias sitting at exactly 0.0 is already protected by the ratio test alone
    (0.0 is not less than 1e-06 x 0.0).  A tensor that has only ever been SMALL
    is not: a peak of 1e-14 falling to 1e-25 clears the absolute floor and the
    ratio both, and without the skip it would be reported as a collapse it never
    had the scale to suffer.

    ⚠ Written so it can FAIL: remove ``peak > floor`` from the test and the
    second check below halts.
    """
    model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.fill_(1e-14)
        model.bias.fill_(1e-14)
    guard = WeightScaleGuard(model)

    with torch.no_grad():
        model.weight.fill_(1e-25)
    guard.check(epoch=1, global_step=100)


def test_a_bias_that_grows_then_collapses_is_still_caught():
    """Why the baseline is a RUNNING max rather than the initial snapshot.

    ⚠ Written so it can FAIL: against a fixed initial snapshot this bias's
    baseline stays 0.0 for the whole run, the skip above applies forever, and
    the collapse at the end goes unreported.
    """
    model = torch.nn.Linear(4, 3)
    torch.nn.init.zeros_(model.bias)
    guard = WeightScaleGuard(model)
    guard.check(epoch=0, global_step=0)

    with torch.no_grad():
        model.bias.fill_(0.5)
    guard.check(epoch=1, global_step=100)
    assert guard.peak['bias'] == pytest.approx(0.5)

    with torch.no_grad():
        model.bias.fill_(1e-20)
    with pytest.raises(VanishedWeights, match='bias'):
        guard.check(epoch=2, global_step=200)


def test_a_healthy_tensor_never_halts_across_a_long_run():
    """No false positive on ordinary training motion, decay included.

    200 epochs x 500 steps of the decoupled shrink AdamW actually applies at the
    shipped lr 1e-3 / weight_decay 1e-4, plus per-epoch noise.  1e5 steps of it
    costs the tensor about 1% of its peak; the guard asks for a factor of 1e6.
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8)
    guard = WeightScaleGuard(model)
    assert set(guard.peak) == {'weight', 'bias'}

    shrink = (1.0 - 1e-3 * 1e-4) ** 500
    for epoch in range(200):
        with torch.no_grad():
            model.weight.mul_(shrink).add_(
                torch.randn_like(model.weight) * 1e-3
            )
            model.bias.mul_(shrink)
        guard.check(epoch=epoch, global_step=epoch * 500)

    assert float(model.weight.detach().abs().max()) > 1e-2
    assert float(model.bias.detach().abs().max()) > 1e-2
