import sys
from pathlib import Path

import pytest
import torch

AINR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AINR))

from training_common import (  # noqa: E402
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
