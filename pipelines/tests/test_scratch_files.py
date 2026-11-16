"""Test programs make their scratch files with erb_fixture_scratch_path(), which creates the file with O_EXCL: a symlink
or a file planted under a guessed name is refused, never followed. No C test source writes to a fixed /tmp name."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'stream/ainr/dfn2/tests/erb_fixture.h'


def _c_sources():
    out = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '*.c', '*.h', '*.cpp'],
                         capture_output=True, text=True, check=True).stdout.split()
    return [ROOT / name for name in out]


def test_no_c_source_writes_to_a_fixed_tmp_name():
    """A "/tmp/<name>" format string in a test is a predictable name that fopen("wb") follows through a symlink. The only
    "/tmp/" literals are the two erb_fixture_scratch_path() definitions, which create their file with O_EXCL."""
    helper_files = {HELPER, ROOT / 'stream/ainr/tests/link_only_ainr.c'}
    bad = []
    for path in _c_sources():
        if path in helper_files:
            continue
        for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            if re.search(r'"/tmp/', line):
                bad.append('%s:%d: %s' % (path.relative_to(ROOT), number, line.strip()))
    assert not bad, 'fixed /tmp names in C sources:\n' + '\n'.join(bad)


@pytest.mark.parametrize('source', [HELPER, ROOT / 'stream/ainr/tests/link_only_ainr.c'])
def test_the_scratch_helpers_use_o_excl(source):
    text = source.read_text()
    assert 'erb_fixture_scratch_path' in text and 'O_CREAT | O_EXCL' in text


def test_helper_refuses_a_planted_symlink(tmp_path):
    """The program plants a symlink to a victim file under the name of every attempt the helper would try first
    (the pid is known to it), then asks for a scratch file: the helper takes a later name and the victim is intact."""
    program = tmp_path / 'probe.c'
    program.write_text(r'''
#include <string.h>
#include <sys/stat.h>
#include "erb_fixture.h"
int main(int argc, char **argv) {
    char path[128], planted[128];
    FILE *fp;
    char victim_text[16] = {0};
    int attempt;
    if (argc != 2) return 2;
    fp = fopen(argv[1], "wb"); if (!fp) return 2; fputs("VICTIM", fp); fclose(fp);
    for (attempt = 0; attempt < 3; ++attempt) {
        snprintf(planted, sizeof planted, "/tmp/probe_scratch_%d_%d.bin", (int)getpid(), attempt);
        unlink(planted);
        if (symlink(argv[1], planted) != 0) return 3;
    }
    erb_fixture_scratch_path(path, sizeof path, "probe_scratch");
    for (attempt = 0; attempt < 3; ++attempt) {
        snprintf(planted, sizeof planted, "/tmp/probe_scratch_%d_%d.bin", (int)getpid(), attempt);
        if (strcmp(planted, path) == 0) return 4;      /* took a planted name */
    }
    fp = fopen(path, "wb"); if (!fp) return 5; fputs("OVERWRITTEN", fp); fclose(fp);
    fp = fopen(argv[1], "rb"); if (!fp) return 6; if (!fgets(victim_text, sizeof victim_text, fp)) return 7; fclose(fp);
    for (attempt = 0; attempt < 3; ++attempt) {
        snprintf(planted, sizeof planted, "/tmp/probe_scratch_%d_%d.bin", (int)getpid(), attempt);
        unlink(planted);
    }
    unlink(path);
    return strcmp(victim_text, "VICTIM") == 0 ? 0 : 8;
}
''')
    exe = tmp_path / 'probe'
    subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-D_POSIX_C_SOURCE=200809L',
                    '-I', str(HELPER.parent), str(program), '-o', str(exe)], check=True)
    done = subprocess.run([str(exe), str(tmp_path / 'victim.txt')])
    assert done.returncode == 0, 'probe exit %d' % done.returncode
