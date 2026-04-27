"""
DNS4 Dataset + DeepFilterNet v2 風格 augmentation

參考:
- Braun et al., "Towards Efficient Models for Real-Time Deep Noise Suppression" (ICASSP 2021)
- DeepFilterNet v2 (Schröter et al., 2022)
- DNS Challenge noisyspeech_synthesizer_singleprocess.py
- PercepNet (Valin et al.) — early RIR dereverberation target

用法:
    from dataset import DNS4Dataset, load_config
    cfg = load_config('config.ini')
    dataset = DNS4Dataset(cfg)
"""

import configparser
import glob
import hashlib
import json
import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


# ============================================================
# Config 工具
# ============================================================

def load_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def parse_snr_ranges(s: str) -> List[Tuple[float, float, float]]:
    """
    解析 SNR 分段取樣字串, 格式: "low:high:weight, ..."
    回傳: [(low, high, weight), ...]
    """
    ranges = []
    for part in s.split(','):
        part = part.strip()
        lo, hi, w = part.split(':')
        ranges.append((float(lo), float(hi), float(w)))
    # normalize weights
    total = sum(r[2] for r in ranges)
    return [(lo, hi, w / total) for lo, hi, w in ranges]


def sample_snr(ranges: List[Tuple[float, float, float]]) -> float:
    """從分段 SNR 範圍中取樣"""
    r = random.random()
    cum = 0.0
    for lo, hi, w in ranges:
        cum += w
        if r <= cum:
            return random.uniform(lo, hi)
    # fallback
    lo, hi, _ = ranges[-1]
    return random.uniform(lo, hi)


# ============================================================
# Biquad Filter (ref: DeepFilterNet RandBiquadFilter)
# Web Audio API EQ cookbook 公式
# ============================================================

BIQUAD_TYPES = [
    ('lowpass',   4000, 8000),
    ('highpass',  40,   400),
    ('lowshelf',  40,   1000),
    ('highshelf', 1000, 8000),
    ('peaking',   40,   4000),
    ('notch',     40,   4000),
]


def _biquad_coeffs(ftype: str, freq: float, sr: float,
                   q: float, gain_db: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    計算 biquad filter 的 (b, a) 係數
    ref: Web Audio API EQ cookbook (Robert Bristow-Johnson)
    """
    w0 = 2.0 * math.pi * freq / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    if ftype == 'lowpass':
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif ftype == 'highpass':
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif ftype == 'lowshelf':
        amp = 10 ** (gain_db / 40.0)
        sq = 2 * math.sqrt(amp) * alpha
        b0 = amp * ((amp + 1) - (amp - 1) * cos_w0 + sq)
        b1 = 2 * amp * ((amp - 1) - (amp + 1) * cos_w0)
        b2 = amp * ((amp + 1) - (amp - 1) * cos_w0 - sq)
        a0 = (amp + 1) + (amp - 1) * cos_w0 + sq
        a1 = -2 * ((amp - 1) + (amp + 1) * cos_w0)
        a2 = (amp + 1) + (amp - 1) * cos_w0 - sq
    elif ftype == 'highshelf':
        amp = 10 ** (gain_db / 40.0)
        sq = 2 * math.sqrt(amp) * alpha
        b0 = amp * ((amp + 1) + (amp - 1) * cos_w0 + sq)
        b1 = -2 * amp * ((amp - 1) + (amp + 1) * cos_w0)
        b2 = amp * ((amp + 1) + (amp - 1) * cos_w0 - sq)
        a0 = (amp + 1) - (amp - 1) * cos_w0 + sq
        a1 = 2 * ((amp - 1) - (amp + 1) * cos_w0)
        a2 = (amp + 1) - (amp - 1) * cos_w0 - sq
    elif ftype == 'peaking':
        amp = 10 ** (gain_db / 40.0)
        b0 = 1 + alpha * amp
        b1 = -2 * cos_w0
        b2 = 1 - alpha * amp
        a0 = 1 + alpha / amp
        a1 = -2 * cos_w0
        a2 = 1 - alpha / amp
    elif ftype == 'notch':
        b0 = 1
        b1 = -2 * cos_w0
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    else:
        raise ValueError(f"Unknown biquad type: {ftype}")

    b = torch.tensor([b0 / a0, b1 / a0, b2 / a0], dtype=torch.float32)
    a = torch.tensor([1.0, a1 / a0, a2 / a0], dtype=torch.float32)
    return b, a


def apply_biquad(audio: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Direct Form II biquad filter"""
    return torchaudio.functional.biquad(audio, b[0], b[1], b[2], a[0], a[1], a[2])


def rand_biquad_filter(audio: torch.Tensor, sr: int,
                       n_filters: int = 3,
                       gain_db: float = 15.0,
                       q_min: float = 0.5,
                       q_max: float = 1.5) -> torch.Tensor:
    """
    隨機套用 1~n_filters 個 biquad filter
    套用後 RMS normalize 保持能量
    """
    rms_in = audio.pow(2).mean().sqrt() + 1e-10
    n = random.randint(1, n_filters)

    for _ in range(n):
        ftype, freq_lo, freq_hi = random.choice(BIQUAD_TYPES)
        freq = random.uniform(freq_lo, min(freq_hi, sr / 2 - 1))
        q = random.uniform(q_min, q_max)
        gain = random.uniform(-gain_db, gain_db)
        b, a = _biquad_coeffs(ftype, freq, sr, q, gain)
        audio = apply_biquad(audio, b, a)

    # RMS normalize (energy preservation)
    rms_out = audio.pow(2).mean().sqrt() + 1e-10
    audio = audio * (rms_in / rms_out)

    # prevent clipping from filter
    peak = audio.abs().max()
    if peak > 1.0:
        audio = audio / peak

    return audio


# ============================================================
# RT60 估算 (Schroeder 積分法)
# ============================================================

def estimate_rt60(rir: torch.Tensor, sr: int) -> float:
    """
    Schroeder 積分法估算 RT60
    1. h²(n) → 從尾端反向累加
    2. 轉 dB
    3. 線性回歸 -5dB ~ -25dB → 外推到 -60dB (T20 × 3)
    """
    h2 = rir.pow(2).numpy()
    # backward integration
    energy = np.cumsum(h2[::-1])[::-1].copy()
    energy = energy / (energy[0] + 1e-20)
    energy_db = 10.0 * np.log10(energy + 1e-20)

    # find -5dB and -25dB points
    idx_5 = np.argmax(energy_db < -5)
    idx_25 = np.argmax(energy_db < -25)

    if idx_25 <= idx_5 or idx_5 == 0:
        # fallback: cannot estimate
        return 0.0

    # linear regression
    x = np.arange(idx_5, idx_25)
    y = energy_db[idx_5:idx_25]
    if len(x) < 2:
        return 0.0

    slope = np.polyfit(x, y, 1)[0]
    if slope >= 0:
        return 0.0

    # T20: time for 20dB decay, extrapolate to 60dB
    rt60 = (-60.0 / slope) / sr
    return float(rt60)


# ============================================================
# RIR 處理
# ============================================================

def prepare_rir(rir: torch.Tensor, sr: int,
                early_ms: float = 50.0,
                pre_delay_keep_ms: float = 1.0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RIR 前處理:
    1. 移除 pre-delay (peak 前只保留 pre_delay_keep_ms)
    2. 產生 early_rir (target 用) 和 full_rir (noisy 混合用)

    回傳: (early_rir, full_rir)
    """
    # 移除 pre-delay
    peak_idx = torch.argmax(torch.abs(rir)).item()
    keep_samples = int(sr * pre_delay_keep_ms / 1000)
    start_idx = max(0, peak_idx - keep_samples)
    rir = rir[start_idx:]

    # full_rir = 去 delay 後的完整 RIR
    full_rir = rir.clone()

    # early_rir = peak 後保留 early_ms + fade-out
    new_peak = min(keep_samples, len(rir) - 1)
    early_samples = int(sr * early_ms / 1000)
    end_idx = min(new_peak + early_samples, len(rir))
    early_rir = rir[:end_idx].clone()

    # fade-out window (最後 5ms half-Hann)
    fade_len = min(int(sr * 5 / 1000), early_samples // 4)
    if fade_len > 0 and fade_len < len(early_rir):
        fade = torch.hann_window(fade_len * 2)[fade_len:]
        early_rir[-fade_len:] *= fade

    return early_rir, full_rir


def fftconvolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """FFT-based convolution, 截回原始長度"""
    n = signal.shape[-1] + kernel.shape[-1] - 1
    # next power of 2 for efficiency
    fft_size = 1
    while fft_size < n:
        fft_size *= 2

    sig_fd = torch.fft.rfft(signal, n=fft_size)
    ker_fd = torch.fft.rfft(kernel, n=fft_size)
    out = torch.fft.irfft(sig_fd * ker_fd, n=fft_size)
    return out[:signal.shape[-1]]


# ============================================================
# Active RMS (ref: DNS Challenge audiolib.py)
# ============================================================

def active_rms(audio: torch.Tensor, sr: int,
               window_ms: float = 100.0,
               threshold_db: float = -50.0) -> float:
    """
    計算 active RMS — 只算有聲音片段的 RMS
    window_ms: 視窗大小 (ms)
    threshold_db: 低於此能量的 window 不計入
    """
    window_samples = int(sr * window_ms / 1000)
    if window_samples < 1 or len(audio) < window_samples:
        return audio.pow(2).mean().sqrt().item() + 1e-10

    # 計算每個 window 的 energy
    n_windows = len(audio) // window_samples
    audio_trunc = audio[:n_windows * window_samples].view(n_windows, window_samples)
    window_energy = audio_trunc.pow(2).mean(dim=1)

    # 過濾低於 threshold 的 window
    threshold_linear = 10 ** (threshold_db / 10)
    active_mask = window_energy > threshold_linear

    if active_mask.sum() == 0:
        return audio.pow(2).mean().sqrt().item() + 1e-10

    active_energy = window_energy[active_mask].mean()
    return active_energy.sqrt().item()


# ============================================================
# Bandwidth Limitation
# ============================================================

def bandwidth_limit(audio: torch.Tensor, sr: int,
                    target_sr: int) -> torch.Tensor:
    """降頻到 target_sr 再升回 sr，模擬頻寬受限"""
    if target_sr >= sr:
        return audio
    down = torchaudio.functional.resample(audio, sr, target_sr)
    up = torchaudio.functional.resample(down, target_sr, sr)
    # 長度可能有微小差異
    if len(up) > len(audio):
        up = up[:len(audio)]
    elif len(up) < len(audio):
        up = F.pad(up, (0, len(audio) - len(up)))
    return up


# ============================================================
# Clipping Distortion
# ============================================================

def apply_clipping(audio: torch.Tensor,
                   clip_snr_min: float = 0.0,
                   clip_snr_max: float = 20.0) -> torch.Tensor:
    """
    Clipping distortion — 隨機放大使之 clip
    clip_snr: 10*log10(power_original / power_clipping_noise)
    較低的 clip_snr → 更嚴重的 clipping
    """
    clip_snr = random.uniform(clip_snr_min, clip_snr_max)
    # 目標 clipping level: 越低越嚴重
    # 簡化做法: 用 gain 放大再 clamp
    rms = audio.pow(2).mean().sqrt() + 1e-10
    # gain 使 peak 超過 1.0
    peak = audio.abs().max() + 1e-10
    # clip_factor: 1.0 = no clip, >1 = more clip
    clip_factor = 10 ** ((20 - clip_snr) / 20.0)
    clip_factor = max(clip_factor, 1.01)

    audio_gained = audio * clip_factor
    audio_clipped = audio_gained.clamp(-1.0, 1.0)
    # normalize back to original RMS
    rms_after = audio_clipped.pow(2).mean().sqrt() + 1e-10
    audio_clipped = audio_clipped * (rms / rms_after)
    return audio_clipped


# ============================================================
# Clipping Prevention (ref: DNS synthesizer)
# ============================================================

def prevent_clipping(*signals: torch.Tensor,
                     threshold: float = 0.99) -> Tuple[torch.Tensor, ...]:
    """
    若任何信號超出 [-threshold, threshold]，等比例縮小所有信號
    """
    max_amp = max(s.abs().max().item() for s in signals)
    if max_amp > threshold:
        scale = threshold / max_amp
        return tuple(s * scale for s in signals)
    return signals


# ============================================================
# DNS4 Dataset
# ============================================================

class DNS4Dataset(Dataset):
    """
    DNS4 Dataset + DeepFilterNet v2 風格 augmentation

    Pipeline:
    1. Load speech (48kHz → resample) + random crop
    2. Speech augmentation: RandBiquadFilter
    3. RIR convolution (early for target, full for noisy)
    4. Load noise (multi-source) + noise augmentation
    5. Segmental SNR mixing
    6. Gain randomization
    7. Bandwidth limitation (noisy + target)
    8. Clipping distortion (noisy only)
    9. Clipping prevention
    10. STFT → features + target gains  (return_raw=False)
        OR return (noisy, clean) raw audio tensors (return_raw=True)
    """

    def __init__(self, cfg: configparser.ConfigParser, return_raw: bool = False):
        # signal params
        self.sr = cfg.getint('signal', 'sr')
        self.n_fft = cfg.getint('signal', 'n_fft')
        self.win_len = cfg.getint('signal', 'win_len', fallback=self.n_fft)
        self.hop_len = cfg.getint('signal', 'hop_len', fallback=self.win_len // 2)
        self.n_bands = cfg.getint('signal', 'n_bands')
        self.hybrid_cutoff_hz = cfg.getint('signal', 'hybrid_cutoff_hz', fallback=0)
        self.n_erb_high_bands = cfg.getint('signal', 'n_erb_high_bands', fallback=0)

        # audio
        self.segment_sec = cfg.getfloat('audio', 'segment_sec')
        self.segment_samples = int(self.segment_sec * self.sr)

        # mixing
        self.snr_ranges = parse_snr_ranges(cfg.get('mixing', 'snr_ranges'))
        self.target_rms_min = cfg.getfloat('mixing', 'target_rms_min')
        self.target_rms_max = cfg.getfloat('mixing', 'target_rms_max')

        # rir
        self.p_rir = cfg.getfloat('rir', 'p_rir')
        self.rt60_min = cfg.getfloat('rir', 'rt60_min')
        self.rt60_max = cfg.getfloat('rir', 'rt60_max')
        self.early_rir_ms = cfg.getfloat('rir', 'early_rir_ms')
        self.pre_delay_keep_ms = cfg.getfloat('rir', 'pre_delay_keep_ms')

        # noise
        self.max_noise_mix = cfg.getint('noise', 'max_noise_mix')

        # augmentation
        self.p_biquad = cfg.getfloat('augmentation', 'p_biquad')
        self.n_biquad_filters = cfg.getint('augmentation', 'n_biquad_filters')
        self.biquad_gain_db = cfg.getfloat('augmentation', 'biquad_gain_db')
        self.biquad_q_min = cfg.getfloat('augmentation', 'biquad_q_min')
        self.biquad_q_max = cfg.getfloat('augmentation', 'biquad_q_max')
        self.p_resample = cfg.getfloat('augmentation', 'p_resample')
        self.resample_sr_min = cfg.getint('augmentation', 'resample_sr_min')
        self.resample_sr_max = cfg.getint('augmentation', 'resample_sr_max')
        self.p_clipping = cfg.getfloat('augmentation', 'p_clipping')
        self.clip_snr_min = cfg.getfloat('augmentation', 'clip_snr_min')
        self.clip_snr_max = cfg.getfloat('augmentation', 'clip_snr_max')

        self.return_raw = return_raw

        # epoch size
        self.epoch_size = cfg.getint('training', 'epoch_size')

        # scan files (遞迴掃描所有子資料夾)
        speech_dir = cfg.get('paths', 'speech_dir')
        noise_dir = cfg.get('paths', 'noise_dir')
        rir_dir = cfg.get('paths', 'rir_dir', fallback=None)

        self.speech_files = sorted(
            glob.glob(os.path.join(speech_dir, '**', '*.wav'), recursive=True)
        )
        self.noise_files = sorted(
            glob.glob(os.path.join(noise_dir, '**', '*.wav'), recursive=True)
        )

        if not self.speech_files:
            raise FileNotFoundError(f"No .wav files found in {speech_dir}")
        if not self.noise_files:
            raise FileNotFoundError(f"No .wav files found in {noise_dir}")

        # RIR: load paths + filter by RT60 (with cache)
        self.rir_files = []
        if rir_dir and os.path.isdir(rir_dir):
            self.rir_files = self._load_rir_paths_cached(rir_dir)

        # ERB bands
        self.bin_edges = self._compute_erb_bands()

        print(f"DNS4Dataset: {len(self.speech_files)} speech, "
              f"{len(self.noise_files)} noise, "
              f"{len(self.rir_files)} RIR files")

        # epoch shuffle indices
        self._shuffle_indices()

    def _load_rir_paths_cached(self, rir_dir: str) -> List[str]:
        """
        掃描 RIR 目錄並用 RT60 過濾，結果 cache 到 JSON 檔。
        Cache key = hash(檔案清單 + sr + rt60 range)，設定或檔案變動時自動重算。
        """
        all_rir_paths = sorted(
            glob.glob(os.path.join(rir_dir, '**', '*.wav'), recursive=True)
        )
        if not all_rir_paths:
            return []

        # 計算 cache key: 檔案清單 + 設定
        key_str = json.dumps({
            'paths': all_rir_paths,
            'sr': self.sr,
            'rt60_min': self.rt60_min,
            'rt60_max': self.rt60_max,
        }, sort_keys=True)
        cache_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        cache_path = os.path.join(rir_dir, f'.rir_rt60_cache_{cache_hash}.json')

        # 嘗試讀 cache
        if os.path.isfile(cache_path):
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            # 驗證 cache 內的檔案仍存在
            valid = [p for p in cached['rir_files'] if os.path.isfile(p)]
            if len(valid) == len(cached['rir_files']):
                print(f"RIR cache hit: {cache_path} ({len(valid)} files)")
                return valid
            print(f"RIR cache stale (missing files), rescanning...")

        # Cache miss: 逐檔計算 RT60
        print(f"Scanning {len(all_rir_paths)} RIR files for RT60 filtering "
              f"[{self.rt60_min:.1f}s, {self.rt60_max:.1f}s]...")
        passed = []
        rt60_map = {}
        for rp in all_rir_paths:
            try:
                rir_audio, rir_sr = torchaudio.load(rp)
                rir_audio = rir_audio[0]
                if rir_sr != self.sr:
                    rir_audio = torchaudio.functional.resample(
                        rir_audio, rir_sr, self.sr)
                rt60 = estimate_rt60(rir_audio, self.sr)
                rt60_map[rp] = rt60
                if self.rt60_min <= rt60 <= self.rt60_max:
                    passed.append(rp)
            except Exception:
                continue
        print(f"  → {len(passed)} / {len(all_rir_paths)} RIRs passed RT60 filter")

        # 寫入 cache
        cache_data = {
            'rir_files': passed,
            'rt60_map': rt60_map,
        }
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            print(f"  → Cache saved: {cache_path}")
        except OSError:
            pass  # 寫不了就算了，下次重算

        return passed

    def _shuffle_indices(self):
        """每 epoch 開始時 shuffle，取 epoch_size 個"""
        indices = list(range(len(self.speech_files)))
        random.shuffle(indices)
        if self.epoch_size > 0:
            self._indices = indices[:self.epoch_size]
        else:
            self._indices = indices

    def __len__(self):
        return len(self._indices)

    # --------------------------------------------------------
    # ERB bands
    # --------------------------------------------------------

    def _compute_erb_bands(self):
        from train import compute_hybrid_bands, compute_erb_bands
        if self.hybrid_cutoff_hz > 0 and self.n_erb_high_bands > 0:
            bin_edges, self.n_bands = compute_hybrid_bands(
                self.n_fft, self.sr, self.n_erb_high_bands, self.hybrid_cutoff_hz)
            return bin_edges
        else:
            return compute_erb_bands(self.n_fft, self.sr, self.n_bands)

    # --------------------------------------------------------
    # Audio loading helpers
    # --------------------------------------------------------

    def _load_and_crop(self, path: str, target_len: int) -> torch.Tensor:
        """載入音檔 → resample → 隨機裁切。空檔或損壞檔會 raise RuntimeError。"""
        audio, orig_sr = torchaudio.load(path)
        audio = audio[0]  # mono
        if audio.numel() == 0:
            raise RuntimeError(f"Empty audio: {path}")
        if orig_sr != self.sr:
            audio = torchaudio.functional.resample(audio, orig_sr, self.sr)
        if len(audio) >= target_len:
            start = random.randint(0, len(audio) - target_len)
            audio = audio[start:start + target_len]
        else:
            audio = F.pad(audio, (0, target_len - len(audio)))
        return audio

    def _load_noise(self, target_len: int) -> torch.Tensor:
        """載入噪音 → resample → loop 到足夠長"""
        path = random.choice(self.noise_files)
        audio, orig_sr = torchaudio.load(path)
        audio = audio[0]
        if orig_sr != self.sr:
            audio = torchaudio.functional.resample(audio, orig_sr, self.sr)
        if len(audio) < target_len:
            repeats = (target_len // len(audio)) + 1
            audio = audio.repeat(repeats)
        start = random.randint(0, len(audio) - target_len)
        return audio[start:start + target_len]

    def _load_rir(self) -> torch.Tensor:
        """載入隨機 RIR → resample"""
        path = random.choice(self.rir_files)
        audio, orig_sr = torchaudio.load(path)
        audio = audio[0]
        if orig_sr != self.sr:
            audio = torchaudio.functional.resample(audio, orig_sr, self.sr)
        return audio

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        """Root Hann window STFT → complex spectrum (n_frames, n_bins)"""
        window = torch.sqrt(torch.hann_window(self.win_len, device=audio.device))
        spec = torch.stft(audio, self.n_fft,
                          hop_length=self.hop_len,
                          win_length=self.win_len,
                          window=window,
                          return_complex=True,
                          center=True)
        return spec.T  # (n_bins, n_frames) → (n_frames, n_bins)

    def _band_energy(self, power_spec: torch.Tensor) -> torch.Tensor:
        """power spectrum → ERB band energy"""
        bands = []
        for b in range(len(self.bin_edges) - 1):
            lo, hi = self.bin_edges[b], self.bin_edges[b + 1]
            bands.append(power_spec[..., lo:hi].sum(dim=-1))
        return torch.stack(bands, dim=-1)

    def _extract_features(self, power_spec: torch.Tensor) -> torch.Tensor:
        """power spectrum → normalized log ERB band energy"""
        energy = self._band_energy(power_spec)
        log_energy = torch.log(energy + 1e-10)
        mean = log_energy.mean(dim=-2, keepdim=True)
        std = log_energy.std(dim=-2, keepdim=True) + 1e-8
        return (log_energy - mean) / std

    def _compute_gain_target(self, clean_power: torch.Tensor,
                             noisy_power: torch.Tensor) -> torch.Tensor:
        """per-band ideal ratio mask (amplitude domain), clipped [0, 1]"""
        clean_energy = self._band_energy(clean_power)
        noisy_energy = self._band_energy(noisy_power)
        ratio = clean_energy / (noisy_energy + 1e-10)
        gain = torch.sqrt(torch.clamp(ratio, 0.0, 1.0))
        return gain

    # --------------------------------------------------------
    # __getitem__
    # --------------------------------------------------------

    def __getitem__(self, idx):
        # Retry: 遇到空檔/損壞檔時隨機換一個 sample，最多重試 5 次
        for _retry in range(5):
            try:
                return self._getitem_impl(idx)
            except (RuntimeError, Exception) as e:
                idx = random.randint(0, len(self._indices) - 1)
        return self._getitem_impl(idx)

    def _getitem_impl(self, idx):
        real_idx = self._indices[idx]
        target_len = self.segment_samples

        # Noise-only sample (10%): 讓模型學會「純噪音 → gain 全 0」
        noise_only = random.random() < 0.1

        if not noise_only:
            # 1. Load clean speech
            speech = self._load_and_crop(self.speech_files[real_idx], target_len)

            # 2. Speech augmentation: RandBiquadFilter (混合前)
            if random.random() < self.p_biquad:
                speech = rand_biquad_filter(
                    speech, self.sr,
                    n_filters=self.n_biquad_filters,
                    gain_db=self.biquad_gain_db,
                    q_min=self.biquad_q_min,
                    q_max=self.biquad_q_max)

            # 3. RIR convolution
            if self.rir_files and random.random() < self.p_rir:
                rir = self._load_rir()
                early_rir, full_rir = prepare_rir(
                    rir, self.sr,
                    early_ms=self.early_rir_ms,
                    pre_delay_keep_ms=self.pre_delay_keep_ms)
                target = fftconvolve(speech, early_rir)
                reverbed = fftconvolve(speech, full_rir)
            else:
                target = speech.clone()
                reverbed = speech.clone()

        # 4. Load noise (multi-source)
        n_noises = random.randint(1, self.max_noise_mix)
        noise = torch.zeros(target_len)
        for _ in range(n_noises):
            noise = noise + self._load_noise(target_len)

        # 5. Noise augmentation: RandBiquadFilter (混合前, 獨立隨機參數)
        if random.random() < self.p_biquad:
            noise = rand_biquad_filter(
                noise, self.sr,
                n_filters=self.n_biquad_filters,
                gain_db=self.biquad_gain_db,
                q_min=self.biquad_q_min,
                q_max=self.biquad_q_max)

        if noise_only:
            # Noise-only: noisy = noise, target = silence
            noisy = noise.clone()
            target = torch.zeros(target_len)
        else:
            # 6. Segmental SNR mixing
            snr_db = sample_snr(self.snr_ranges)
            speech_rms = active_rms(reverbed, self.sr)
            noise_rms = active_rms(noise, self.sr)
            noise_scaled = noise * (speech_rms / noise_rms) * (10 ** (-snr_db / 20))
            noisy = reverbed + noise_scaled

        # 7. Gain randomization
        target_rms_db = random.uniform(self.target_rms_min, self.target_rms_max)
        target_rms_linear = 10 ** (target_rms_db / 20)
        if noise_only:
            current_rms = active_rms(noisy, self.sr)
        else:
            current_rms = active_rms(target, self.sr)
        scale = target_rms_linear / current_rms
        target = target * scale
        noisy = noisy * scale

        # 8. Bandwidth limitation (同時套用在 noisy 和 target, 相同 target SR)
        if random.random() < self.p_resample:
            bw_sr = random.randint(self.resample_sr_min, self.resample_sr_max)
            noisy = bandwidth_limit(noisy, self.sr, bw_sr)
            target = bandwidth_limit(target, self.sr, bw_sr)

        # 9. Clipping distortion (只影響 noisy)
        if random.random() < self.p_clipping:
            noisy = apply_clipping(noisy, self.clip_snr_min, self.clip_snr_max)

        # 10. Clipping prevention
        target, noisy = prevent_clipping(target, noisy)

        if self.return_raw:
            return noisy, target

        # 11. STFT → features + target gains
        clean_spec = self._stft(target)
        noisy_spec = self._stft(noisy)

        clean_power = clean_spec.abs().pow(2)
        noisy_power = noisy_spec.abs().pow(2)

        features = self._extract_features(noisy_power)
        target_gains = self._compute_gain_target(clean_power, noisy_power)

        return features, target_gains


# ============================================================
# Precomputed Dataset (讀取 gen_dataset.py 產生的 .pt shard)
# ============================================================

class WavPairDataset(Dataset):
    """
    讀取 gen_dataset.py (WAV 模式) 產生的 noisy/clean WAV 對。

    目錄結構:
        data_dir/
            noisy/000000.wav, 000001.wav, ...
            clean/000000.wav, 000001.wav, ...
            meta.json

    用法:
        dataset = WavPairDataset('data/')
        noisy_wav, clean_wav = dataset[0]  # shape: (T,)
    """

    def __init__(self, data_dir: str):
        import json
        self.noisy_dir = os.path.join(data_dir, 'noisy')
        self.clean_dir = os.path.join(data_dir, 'clean')

        if not os.path.isdir(self.noisy_dir):
            raise FileNotFoundError(f"noisy/ not found in {data_dir}")

        meta_path = os.path.join(data_dir, 'meta.json')
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        self.files = sorted(f for f in os.listdir(self.noisy_dir) if f.endswith('.wav'))
        print(f"WavPairDataset: {len(self.files)} pairs from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        noisy, _ = torchaudio.load(os.path.join(self.noisy_dir, name))
        clean, _ = torchaudio.load(os.path.join(self.clean_dir, name))
        return noisy.squeeze(0), clean.squeeze(0)


class PrecomputedDataset(Dataset):
    """
    讀取離線預生成的 .pt shard 檔，跳過所有即時 augmentation。

    用法:
        dataset = PrecomputedDataset('data/')
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, 'meta.pt')
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.pt not found in {data_dir}")

        meta = torch.load(meta_path, weights_only=False)
        self.n_shards = meta['n_shards']
        self.n_total = meta['n_total']
        self.shard_size = meta['shard_size']

        # 載入所有 shard 到記憶體 (concat 成一個大 tensor)
        all_features = []
        all_targets = []
        for i in range(self.n_shards):
            shard_path = os.path.join(data_dir, f'shard_{i:04d}.pt')
            shard = torch.load(shard_path, weights_only=False)
            all_features.append(shard['features'])
            all_targets.append(shard['targets'])

        self.features = torch.cat(all_features, dim=0)  # (N, seq_len, n_bands)
        self.targets = torch.cat(all_targets, dim=0)     # (N, seq_len, n_bands)
        print(f"PrecomputedDataset: {len(self)} samples loaded from {data_dir}")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
