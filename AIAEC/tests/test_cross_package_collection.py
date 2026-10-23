"""The Audio_ALG test roots must collect together, in either argument order.

Every package may carry its own ``conftest.py`` and pytest imports each under
the same top-level module name, so ``from conftest import ...`` gets whichever
package's conftest was imported first. Collected next to a sibling that also
has one, the import resolves to the wrong module and the WHOLE run dies at
collection, which per-package runs cannot see. The gate drives a real pytest
(``--collect-only``: the question is whether the modules IMPORT together) in a
subprocess in both argument orders, since order decides which ``conftest``
wins.
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
