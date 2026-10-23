"""Checkpoint helpers shared by train.py / inference.py.

Kept out of model.py so that file stays identical to upstream
gtcrn_github/gtcrn.py and the vendored published checkpoints keep loading.
"""

# Top-level weights key, in priority order: 'state_dict' (this repo's train.py) or
# 'model' (upstream gtcrn).
_STATE_DICT_KEYS = ('state_dict', 'model')


def extract_state_dict(ckpt, source=''):
    """Return the weight dict from a checkpoint saved by either convention.

    Accepts a raw state_dict too (a mapping whose values are all tensors),
    so ``torch.save(model.state_dict(), ...)`` output still loads.
    """
    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Checkpoint is {type(ckpt).__name__}, expected a dict"
            + (f": {source}" if source else "")
        )

    for key in _STATE_DICT_KEYS:
        if key in ckpt:
            return ckpt[key]

    # A bare state_dict has no bookkeeping keys; detect it by the absence of
    # the ones a training checkpoint always carries.
    if not ({'epoch', 'optimizer', 'scheduler', 'best_val_loss'} & set(ckpt)):
        return ckpt

    raise KeyError(
        f"Checkpoint has none of {_STATE_DICT_KEYS}; top-level keys are "
        f"{sorted(ckpt)}" + (f": {source}" if source else "")
    )
