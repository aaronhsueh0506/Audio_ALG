"""Fixed inputs of pyc_parity_pipelines.sh.

The only external data is the pair of 10 s / 16 kHz recordings tracked in
lib/aec/wav/aec_record/ (their sha256 is pinned in inputs.sha256); everything
else is derived from them deterministically:

  16 kHz   mic, ref                         the recordings as float32
  48 kHz   mic, ref                         resample_poly(x, 3, 1), clipped, stored as
                                            16-bit PCM and read back (the quantisation
                                            of a real 48 kHz capture)
  4 mics   mic4[:, k] = g[k] * roll(mic, k) + N(0, 2e-4^2)     (k = 0..3)
           g = [1.0, 0.9, 1.1, 0.8], integer shifts 0..3, noise from
           numpy.random.RandomState(7), drawn 16 kHz first, then 48 kHz

usage: gen_inputs.py generate <out_dir> <manifest>   write the files, verify against the manifest
       gen_inputs.py print    <out_dir>              print the sha256 manifest of the files in <out_dir>
Exit status is non-zero when a sha256 differs from the manifest: the tolerances of the comparisons are
meaningless on another input, so a changed input is a failure, never a warning.
"""
import hashlib
import io
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
ALG = os.path.abspath(os.path.join(HERE, '..', '..', '..'))      # .../Audio_ALG
SRC_DIR = os.path.join(ALG, 'lib', 'aec', 'wav', 'aec_record')
SRC = {'mic': 'aec_record_mic_10s.wav', 'ref': 'aec_record_ref_10s.wav'}
GAINS = np.array([1.0, 0.9, 1.1, 0.8], np.float32)
SEED = 7
NOISE = 2e-4


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def load_manifest(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                digest, name = line.split(None, 1)
                out[name.strip()] = digest
    return out


def pcm16_roundtrip(x32, sr):
    """What writing a 16-bit WAV and reading it back does to a float32 signal."""
    buf = io.BytesIO()
    sf.write(buf, x32, sr, subtype='PCM_16', format='WAV')
    buf.seek(0)
    y, _ = sf.read(buf, dtype='float32')
    return y


def build():
    """Returns {file name: float32 array}; deterministic."""
    files = {}
    base = {}
    for key, name in SRC.items():
        x, sr = sf.read(os.path.join(SRC_DIR, name), dtype='float64')
        if sr != 16000 or x.ndim != 1:
            raise SystemExit('source %s must be 16 kHz mono, got sr=%s shape=%s' % (name, sr, x.shape))
        base[(key, '16k')] = sf.read(os.path.join(SRC_DIR, name), dtype='float32')[0]
        up = np.clip(resample_poly(x, 3, 1), -1, 1).astype(np.float32)
        base[(key, '48k')] = pcm16_roundtrip(up, 48000)
    rs = np.random.RandomState(SEED)
    for tag in ('16k', '48k'):
        mic = base[('mic', tag)]
        ref = base[('ref', tag)]
        m4 = np.zeros((len(mic), 4), np.float32)
        for k in range(4):
            m4[:, k] = GAINS[k] * np.roll(mic, k) + (rs.randn(len(mic)) * NOISE).astype(np.float32)
        files['mic4_%s.f32' % tag] = m4
        files['ref_%s.f32' % tag] = ref.astype(np.float32)
        files['mic1_%s.f32' % tag] = mic.astype(np.float32)
    return files


def main(argv):
    if len(argv) < 3 or argv[1] not in ('generate', 'print'):
        sys.stderr.write(__doc__)
        return 2
    out_dir = argv[2]
    os.makedirs(out_dir, exist_ok=True)
    if argv[1] == 'generate':
        manifest = load_manifest(argv[3])
        bad = []
        for key, name in SRC.items():
            want = manifest.get('src/' + name)
            got = sha256_file(os.path.join(SRC_DIR, name))
            if want != got:
                bad.append(('src/' + name, want, got))
        for name, arr in sorted(build().items()):
            path = os.path.join(out_dir, name)
            np.ascontiguousarray(arr).tofile(path)
            want, got = manifest.get(name), sha256_file(path)
            if want != got:
                bad.append((name, want, got))
        if bad:
            for name, want, got in bad:
                sys.stderr.write('INPUT_SHA_MISMATCH %s: manifest=%s actual=%s\n' % (name, want, got))
            return 1
        print('inputs ok: %d files match %s' % (len(manifest), os.path.basename(argv[3])))
        return 0
    for name in sorted(os.listdir(out_dir)):
        if name.endswith('.f32'):
            print('%s  %s' % (sha256_file(os.path.join(out_dir, name)), name))
    for key, name in SRC.items():
        print('%s  src/%s' % (sha256_file(os.path.join(SRC_DIR, name)), name))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
