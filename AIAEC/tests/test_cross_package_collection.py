"""The Audio_ALG test roots must collect together, in either argument order.

Every package here may carry its own ``conftest.py``, and pytest imports each
of them under the same top-level module name ``conftest``. So a test module
that does ``from conftest import ...`` gets whichever package's conftest was
imported first. On its own the suite passes; collected next to a sibling that
also has one, the import resolves to the wrong module and the WHOLE run dies
at collection -- including the tests that were fine.

That is a failure the per-package runs cannot see, which is exactly why it
survived: `pytest AIAEC` was green the entire time `pytest AIAEC AINR
pipelines` could not start. The gate therefore drives a real pytest in a
subprocess rather than pattern-matching import lines: any future shadowing, by
whatever mechanism, fails here.

Both orders are checked because argument order decides which ``conftest`` wins
the module name -- a fix that only works one way round is not a fix.

``--collect-only`` because the question is whether the modules IMPORT together;
running them is what the ordinary suites already do.
"""
import os
import subprocess
import sys

import pytest

_AUDIO_ALG = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


@pytest.mark.parametrize("roots", [
    ("AIAEC", "AINR", "pipelines"),
    ("AINR", "AIAEC"),
    ("AIAEC", "AINR"),
], ids=lambda roots: "+".join(roots))
def test_the_roots_collect_in_one_invocation(roots):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", *roots],
        cwd=_AUDIO_ALG,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        "`pytest %s` failed at collection:\n%s"
        % (" ".join(roots), result.stdout.decode()[-4000:]))
