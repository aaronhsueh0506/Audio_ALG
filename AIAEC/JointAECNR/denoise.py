"""JointAECNR inference.

Single pair:
    python denoise.py --config config.ini --model output/joint_aecnr_best.pth \
        --mic mic.wav --ref far_end.wav --output enhanced.wav

Batch (files paired by name):
    python denoise.py --config config.ini --model output/joint_aecnr_best.pth \
        --mic-dir mics/ --ref-dir refs/ --output-dir enhanced/

⚠ This model takes TWO inputs.  ``--ref`` is the far-end reference the device
rendered, on the same clock and time origin as the microphone; it is not
optional in the way it looks.  Omitting it substitutes silence, which is a
legitimate thing to run (it exercises the idle gate) but is NOT "inference
without a reference" -- it is inference on the claim that nothing was played.
"""

import argparse
import configparser
import glob
import os
import sys

import torch
import torchaudio

from model import JointAECNR
from postproc import (
    PostProcessChain,
    apply_safety_attenuation,
    comfort_noise_from_log_psd,
)
from train import build_contract, require_checkpoint_contract

_AIAEC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AINR = os.path.join(os.path.dirname(_AIAEC), 'ainr')
sys.path.insert(0, _AIAEC)   # dataset_gen_aec: the AEC corpus, owned by AIAEC/
sys.path.insert(0, _AINR)    # dataset_gen: the SHARED loader/split/seed + DSP
# ⚠ AIAEC/ deliberately depends on ainr/dataset_gen and must not fork it.  Two
# things live there that cannot be duplicated: the augmentation DSP the AEC corpus
# reuses (RIR, RT60, biquad, clipping), and the train/val split + seeder that every
# model in the repo shares.  A second copy of the split is how two models being
# compared silently end up trained on different corpora -- see dataset_gen/loader.py.
# The package is named dataset_gen_aec, NOT dataset_gen, because both directories
# sit on this sys.path and a same-named package would shadow whichever came second.
from dataset_gen_aec import AecGrid, istft, stft  # noqa: E402


def load_model(args):
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")
    grid = AecGrid.from_config(cfg)
    center = cfg.getboolean('signal', 'center', fallback=False)

    device = torch.device(args.device)
    model = JointAECNR.from_config(cfg, grid).to(device)

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    # The same gate the resume path applies.  Shape mismatches are caught by
    # load_state_dict, but the switches that matter most here -- which auxiliary
    # heads exist, whether the echo head is reference-gated, the deep-filter
    # geometry -- can change the model while leaving every shape intact.
    require_checkpoint_contract(ckpt, build_contract(cfg, grid, model),
                                context=args.model, allow_missing=True)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    model.eval()
    return model, grid, center, device


def load_wav(path, grid, length=None):
    audio, sr = torchaudio.load(path)
    audio = audio[0]
    if sr != grid.sr:
        audio = torchaudio.functional.resample(audio, sr, grid.sr)
    if length is not None:
        if audio.shape[-1] < length:
            audio = torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))
        else:
            audio = audio[:length]
    return audio


def analysis_pad(length, grid, center):
    """Zeros to prepend/append so the whole signal survives the round trip.

    ``center=True``  -- torch pads internally and ``istft(center=True,
    length=...)`` is the exact inverse, so nothing is needed here.

    ``center=False`` -- ⚠ two separate losses to cover.  The first and last
    half-window of a WOLA reconstruction receive a contribution from a single
    frame, where the sqrt-Hann overlap-add envelope goes to zero and nothing is
    recoverable (see ``train.waveform_for_loss``); and the final partial frame
    is dropped by the analysis entirely.  Padding by a whole window at each end
    and rounding up to a frame boundary puts every real sample strictly inside
    the reconstructible interior.
    """
    if center:
        return 0, 0
    front = tail = grid.win_len
    remainder = (length + front + tail - grid.win_len) % grid.hop_len
    tail += (grid.hop_len - remainder) % grid.hop_len
    return front, tail


def reconstruct(spec, grid, center, pad_front, length):
    """Spectrum -> waveform aligned sample-for-sample with the original input."""
    if center:
        return istft(spec, grid, length=length, center=True)
    # center=True on the inverse of a center=False analysis returns the
    # interior, starting n_fft//2 samples into the padded signal.
    wav = istft(spec, grid, center=True)
    offset = pad_front - grid.n_fft // 2
    return wav[..., offset:offset + length]


def process_pair(mic_path, ref_path, output_path, model, grid, center, device,
                 args):
    mic = load_wav(mic_path, grid)
    length = mic.shape[-1]
    ref = (load_wav(ref_path, grid, length=length) if ref_path
           else torch.zeros_like(mic))

    pad_front, pad_tail = analysis_pad(length, grid, center)
    if pad_front or pad_tail:
        mic = torch.nn.functional.pad(mic, (pad_front, pad_tail))
        ref = torch.nn.functional.pad(ref, (pad_front, pad_tail))

    y_spec = stft(mic.unsqueeze(0).to(device), grid, center=center)
    x_spec = stft(ref.unsqueeze(0).to(device), grid, center=center)

    # Streaming path: the same forward, walked in chunks with the state carried.
    # ⚠ Worth running at least once against --chunk-sec 0.  If the two disagree
    # by more than float noise, some part of the model is reading the whole
    # chunk at once and is not the causal model the latency figures claim.
    chunk_frames = (int(round(args.chunk_sec * grid.frame_rate))
                    if args.chunk_sec > 0 else y_spec.shape[-1])
    state = None
    speech_chunks, echo_chunks, psd_chunks = [], [], []
    with torch.no_grad():
        for start in range(0, y_spec.shape[-1], chunk_frames):
            stop = min(start + chunk_frames, y_spec.shape[-1])
            outputs, state = model(y_spec[..., start:stop],
                                   x_spec[..., start:stop], state)
            speech_chunks.append(outputs.speech_spec)
            if outputs.echo_spec is not None:
                echo_chunks.append(outputs.echo_spec)
            if outputs.noise_log_psd is not None:
                psd_chunks.append(outputs.noise_log_psd)

    speech = torch.cat(speech_chunks, dim=-1)
    noise_log_psd = torch.cat(psd_chunks, dim=-1) if psd_chunks else None

    # Post-processing is OPT-IN.  The joint output is the default final output;
    # PostProcessChain is used here for its runtime double-suppression check
    # rather than because anything below is required.
    chain = PostProcessChain()
    if args.safety_attenuation_db is not None:
        limit = args.safety_attenuation_db
        chain.add('safety_attenuation',
                  lambda spec, mic_spec, **_: apply_safety_attenuation(
                      spec, mic_spec, max_attenuation_db=limit))
    if args.cng:
        if noise_log_psd is None:
            raise ValueError(
                "--cng needs aux_noise_psd_head, which this checkpoint was "
                "built without; there is no honest floor estimate to drive it")
        chain.add('comfort_noise',
                  lambda spec, mic_spec, **_: comfort_noise_from_log_psd(
                      spec, mic_spec, noise_log_psd,
                      cng_level_db=args.cng_level_db))
    speech = chain(speech, mic_spec=y_spec)

    wav = reconstruct(speech, grid, center, pad_front, length).cpu()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torchaudio.save(output_path, wav, grid.sr)

    if args.emit_echo:
        if not echo_chunks:
            raise ValueError("--emit-echo needs aux_echo_head")
        echo = reconstruct(torch.cat(echo_chunks, dim=-1), grid, center,
                           pad_front, length).cpu()
        torchaudio.save(_suffixed(output_path, 'echo'), echo, grid.sr)
    if args.emit_noise_psd:
        if noise_log_psd is None:
            raise ValueError("--emit-noise-psd needs aux_noise_psd_head")
        torch.save(noise_log_psd.cpu(),
                   os.path.splitext(output_path)[0] + '.noise_log_psd.pt')


def _suffixed(path, tag):
    stem, ext = os.path.splitext(path)
    return f"{stem}.{tag}{ext}"


def run_batch(args, model, grid, center, device):
    mics = sorted(glob.glob(os.path.join(args.mic_dir, '**', '*.wav'),
                            recursive=True))
    if not mics:
        raise FileNotFoundError(f"no .wav under {args.mic_dir}")
    print(f"{len(mics)} file(s) -> {args.output_dir}")
    failed = []
    for mic_path in mics:
        rel = os.path.relpath(mic_path, args.mic_dir)
        ref_path = os.path.join(args.ref_dir, rel) if args.ref_dir else None
        if ref_path and not os.path.isfile(ref_path):
            # ⚠ Loud, not silent.  Silently substituting zeros would report a
            # score for "AEC with no reference" as if it were an AEC score.
            failed.append((rel, f"no reference at {rel}"))
            continue
        try:
            process_pair(mic_path, ref_path,
                         os.path.join(args.output_dir, rel), model, grid,
                         center, device, args)
        except Exception as exc:                        # noqa: BLE001
            failed.append((rel, str(exc)))
    print(f"done: {len(mics) - len(failed)}/{len(mics)}")
    for rel, err in failed:
        print(f"  FAILED {rel}: {err}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JointAECNR inference')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--model', required=True)
    parser.add_argument('--mic', default=None)
    parser.add_argument('--ref', default=None,
                        help='far-end reference; omitted means "nothing was '
                             'played", not "reference unknown"')
    parser.add_argument('--output', default=None)
    parser.add_argument('--mic-dir', default=None)
    parser.add_argument('--ref-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--chunk-sec', type=float, default=0.0,
                        help='process in chunks of this many seconds with the '
                             'recurrent state carried; 0 = whole file at once')
    parser.add_argument('--safety-attenuation-db', type=float, default=None,
                        help='cap the attenuation the model may apply (see '
                             'postproc.apply_safety_attenuation)')
    parser.add_argument('--cng', action='store_true',
                        help='refill emptied bins from aux_noise_psd_head')
    parser.add_argument('--cng-level-db', type=float, default=-6.0)
    parser.add_argument('--emit-echo', action='store_true',
                        help='also write D_hat from aux_echo_head')
    parser.add_argument('--emit-noise-psd', action='store_true')
    args = parser.parse_args()

    model, grid, center, device = load_model(args)
    if args.mic_dir and args.output_dir:
        run_batch(args, model, grid, center, device)
    elif args.mic and args.output:
        process_pair(args.mic, args.ref, args.output, model, grid, center,
                     device, args)
        print(f"wrote {args.output}")
    else:
        parser.error('give (--mic + --output) or (--mic-dir + --output-dir)')
