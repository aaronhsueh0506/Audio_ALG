"""Make this project's own ``model`` / ``train`` / ``denoise`` win the import.

⚠ WHY THIS FILE EXISTS -- it is not boilerplate.

Every project here is a plain directory of top-level modules with the SAME
names: ``model.py``, ``train.py``, ``denoise.py``, ``postproc.py``.  The house
style is ``sys.path.insert(0, ROOT)`` followed by ``from model import ...``,
which is fine when one project is tested alone but not when several are
collected in one pytest session: the first project to import ``train`` puts it
in ``sys.modules`` under that bare name, and every later project silently gets
THAT module back.  ``sys.path`` order is irrelevant -- an already-imported
module is never re-read.

The symptom is not an ImportError.  It is a sibling's trainer being exercised
against this project's config.ini, which surfaces as a nonsense error about a
missing config key (``No option 'batch_size' in section: 'training'``) and looks
like a broken config rather than a cross-project collision.

Two evictions are needed, and the second is the one that is easy to miss:

1. At COLLECTION time, so the ``from model import ...`` at the top of each test
   module reads this project's files.  pytest imports a directory's conftest
   before the test modules in it.
2. Before EVERY TEST, because a test that does ``import train`` inside its own
   body runs long after collection -- by which point a later-collected sibling
   (RNNoise-ERB, say) has re-registered the bare name.  Without the autouse
   fixture below, such a test calls the sibling's ``train()`` against this
   project's config and fails with a missing-key error from a file it never
   mentions.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The bare names every sibling project also uses.  Anything imported under one
# of these that did NOT come from this project's directory belongs to a sibling
# and must go, so the `from model import ...` below re-reads the right file.
_SHARED_MODULE_NAMES = ('model', 'train', 'denoise', 'postproc', 'frontends')


def _evict_sibling_modules():
    for name in _SHARED_MODULE_NAMES:
        module = sys.modules.get(name)
        if module is None:
            continue
        path = getattr(module, '__file__', None) or ''
        if os.path.dirname(os.path.abspath(path)) != ROOT:
            del sys.modules[name]


def _claim_import_path():
    _evict_sibling_modules()
    if ROOT in sys.path:
        sys.path.remove(ROOT)
    sys.path.insert(0, ROOT)


# (1) collection time, for the module-level imports in each test file.
_claim_import_path()


# (2) run time, for tests that import inside the function body.
@pytest.fixture(autouse=True)
def _own_modules_win():
    _claim_import_path()
    yield
