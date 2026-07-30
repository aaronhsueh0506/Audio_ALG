"""Put both trees on ``sys.path`` for every test under ``AIAEC/``.

⚠ AIAEC/ deliberately depends on ``ainr/dataset_gen`` and must not fork it. Two
things live there that cannot be duplicated:

* the augmentation DSP the AEC corpus reuses -- RIR, RT60, biquad, clipping;
* the train/val split and the seeder that **every** model in the repo shares. A
  second copy of the split is how two models being compared silently end up
  trained on different corpora (it happened once: 5% held out on one side, 10% on
  the other). ``ainr/tests/test_bakeoff_protocol.py`` enforces the single
  definition across both trees.

So the AEC entry points need `AIAEC/` (for ``dataset_gen_aec``) *and* `ainr/` (for
``dataset_gen``). Each `train.py` / `denoise.py` sets this up itself, because they
are run directly as scripts; this file does the same for pytest, which imports the
test modules rather than the scripts.

⚠ The AEC corpus package is named ``dataset_gen_aec``, NOT ``dataset_gen``,
precisely because both directories are on this one path -- two packages under the
same name would shadow whichever landed second, and the loser would be chosen by
`sys.path` order rather than by intent.

This file does NOT replace the per-project ``tests/conftest.py``. Those solve a
different problem: every project here has a top-level ``model.py`` / ``train.py``,
so the first one imported wins the bare name in ``sys.modules`` for the whole
session. See any of them for the full explanation.
"""

import os
import sys

_AIAEC = os.path.dirname(os.path.abspath(__file__))
_AINR = os.path.join(os.path.dirname(_AIAEC), 'ainr')

for _path in (_AIAEC, _AINR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
