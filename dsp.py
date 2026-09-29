"""
nSSTV Modular Component
"""
import os
import sys
import math
import time
import json
import struct
import shutil
import random
import logging
import tempfile
import warnings
import threading
import subprocess
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, find_peaks, medfilt, spectrogram

logger = logging.getLogger("nsstv")

def apply_bandpass_filter(audio, fs, lowcut=1000.0, highcut=2500.0, order=4):
    """
    Apply a 1100 Hz - 2300 Hz Butterworth bandpass filter to reject sub-audible hums (<1000 Hz)
    and high-frequency noise (>2500 Hz).
    """
    audio = np.asarray(audio, dtype=np.float32)
    nyq = 0.5 * fs
    low = max(100.0, lowcut) / nyq
    high = min(nyq - 100.0, highcut) / nyq
    if low >= high or high >= 1.0:
        return audio

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, audio)
    return filtered.astype(np.float32)


def apply_whistle_notch_filter(audio, fs, quality=30.0, min_freq=500.0, max_freq=3000.0):
    """
    Auto-detect and notch out interfering heterodyne whistle tones (CW/carriers) before FM demodulation.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < fs * 0.1:
        return audio

    try:
        freqs, pxx = spectrogram(audio, fs=fs, nperseg=int(min(fs * 0.25, len(audio))))
        mean_pxx = np.mean(pxx, axis=1)

        peaks, props = find_peaks(mean_pxx, prominence=np.max(mean_pxx) * 0.3)
        if len(peaks) == 0:
            return audio

        from scipy.signal import iirnotch, lfilter
        filtered = audio.copy()
        nyq = 0.5 * fs

        for p in peaks:
            f0 = freqs[p]
            if min_freq <= f0 <= max_freq:
                w0 = f0 / nyq
                if 0.01 < w0 < 0.99:
                    b, a = iirnotch(w0, quality)
                    filtered = lfilter(b, a, filtered)

        return filtered.astype(np.float32)
    except Exception:
        return audio


def measure_frequency(audio_path, t_start, t_end, method="zero_crossing"):
    """
    Measure the dominant tone frequency in a window of an SSTV audio file.

    method:
        zero_crossing  sub-sample-resolution zero-crossing estimator (default).
                       Works best on clean single-tone windows (VIS bits,
                       leader tones, sync pulses) and resolves ppm-level
                       drift that the Hilbert estimator quantises to 1/fs.
        hilbert        weighted-mean of the Hilbert/unwrap instantaneous
                       frequency - the FMDemodulator default.  Better for
                       noisy multi-tone windows where zero-crossings get
                       confused, but resolution is bounded by the sample
                       rate.

    Returns the frequency in Hz as a float.

    This is the public, top-level entry point that exercises the new
    sub-sample estimator.  It's useful for VIS-code analyzers, sync
    diagnostics, and any place a few ppm of drift matters.

    Raises ValueError if the window is invalid (t_start < 0, t_end <= t_start,
    window outside the audio).
    """
    # Validate the window before anything else.
    t_start = float(t_start)
    t_end = float(t_end)

    if t_start < 0:
        raise ValueError(f"t_start must be >= 0, got {t_start}")

    if t_end <= t_start:
        raise ValueError(
            f"t_end ({t_end}) must be > t_start ({t_start}); "
            "the window has zero or negative duration."
        )

    fs, samples = load_audio_mono(audio_path)
    audio_duration = len(samples) / fs

    if t_start >= audio_duration:
        raise ValueError(
            f"t_start ({t_start}) is past the end of the audio "
            f"({audio_duration:.3f}s)."
        )

    # Clip t_end to audio length so callers don't have to know the exact
    # duration to measure near the end.
    t_end = min(t_end, audio_duration)

    # Don't include FMDemodulator's full demod if the user only asked for
    # zero-crossing - it's expensive and unnecessary.
    if method == "zero_crossing":
        # Build a minimal demodulator so we can reuse the cached zero
        # crossing computation.
        from scipy.signal import butter, sosfiltfilt

        nyq = fs * 0.5
        lo = max(800.0, min(1000.0, nyq * 0.4))
        hi = min(2600.0, nyq * 0.95)

        sos = butter(4, (lo, hi), btype="bandpass", fs=fs, output="sos")

        a = max(0, int(round(t_start * fs)) - 8192)
        b = min(len(samples), int(round(t_end * fs)) + 8192)

        seg = sosfiltfilt(sos, np.asarray(samples[a:b], dtype=np.float64))

        rel_a = max(0, int(round(t_start * fs)) - a)
        rel_b = min(len(seg), int(round(t_end * fs)) - a)

        if rel_b - rel_a < 8:
            raise ValueError(
                f"Window [{t_start}, {t_end}) is too short or outside the audio."
            )

        win = seg[rel_a:rel_b]

        s0 = win[:-1]
        s1 = win[1:]
        prod = s0 * s1
        idx = np.where(prod < 0)[0]

        if len(idx) < 4:
            # Fall back to a simple Goertzel power scan for the dominant
            # frequency in the band - better than returning 0 when the
            # window is so short that no crossings are present.
            det = ToneDetector(fs, samples)
            best_f = 0.0
            best_p = 0.0

            for f in np.arange(1000.0, 2600.0, 5.0):
                p = det.tone_power(f, t_start, t_end - t_start)
                if p > best_p:
                    best_p = p
                    best_f = f

            return float(best_f)

        y0 = s0[idx]
        y1 = s1[idx]

        denom = y1 - y0
        frac = np.where(np.abs(denom) > 1e-12, -y0 / denom, 0.5)
        frac = np.clip(frac, 0.0, 1.0)

        crossing_samples = idx + frac
        rising_mask = y1 > y0

        if rising_mask.sum() < 2:
            return float(fs / np.mean(np.diff(crossing_samples)))

        rising_samples = crossing_samples[rising_mask]
        period_samples = float(np.mean(np.diff(rising_samples)))

        if period_samples <= 0:
            return 0.0

        return float(fs / period_samples)

    if method == "hilbert":
        demod = FMDemodulator(fs, samples)
        return float(demod.mean_freq(t_start, t_end))

    raise ValueError(f"Unknown method {method!r}; use 'zero_crossing' or 'hilbert'.")


def smoothing_seconds_for_layout(layout, divisor=None, basis=None):
    if layout is None:
        return SMOOTHING_MAX_SECONDS

    divisor = SMOOTHING_PIXEL_DIVISOR if divisor is None else float(divisor)
    basis = SMOOTHING_BASIS if basis is None else basis

    pixel_seconds = (
        layout.luma_pixel_seconds if basis == "luma" else layout.fastest_pixel_seconds
    )

    if not np.isfinite(pixel_seconds) or pixel_seconds <= 0 or divisor <= 0:
        return SMOOTHING_MAX_SECONDS

    return float(np.clip(
        pixel_seconds / divisor,
        SMOOTHING_MIN_SECONDS,
        SMOOTHING_MAX_SECONDS,
    ))


def _normalize_audio(samples):
    """
    Convert int/float mono/stereo audio to mono float64 in approximately [-1, 1].
    """
    samples = np.asarray(samples)

    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)

        if info.min == 0:
            midpoint = info.max / 2.0
            samples = (samples.astype(np.float64) - midpoint) / midpoint
        else:
            scale = max(abs(info.min), abs(info.max))
            samples = samples.astype(np.float64) / scale
    else:
        samples = samples.astype(np.float64)

    samples[~np.isfinite(samples)] = 0.0

    if len(samples) > 0:
        samples -= np.mean(samples)
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples /= peak

    return samples.astype(np.float64)


def _analytic(seg):
    """
    Analytic signal (real + j*Hilbert) using rfft/irfft, about half the cost
    of scipy.signal.hilbert, which runs a full complex fft/ifft pair.

    The Hilbert transformer is -j*sign(f), so multiplying the one-sided
    spectrum by -j and transforming back gives the real quadrature part.
    """
    n = len(seg)
    N = next_fast_len(n)

    spectrum = np.fft.rfft(seg, n=N)

    spectrum[1:-1] *= -1j
    spectrum[0] = 0.0

    if N % 2 == 0:
        spectrum[-1] = 0.0

    quad = np.fft.irfft(spectrum, n=N)[:n]

    return seg + 1j * quad


def _median_kernel(fs, seconds):
    k = int(round(float(seconds) * float(fs)))
    k = max(3, k)

    if k % 2 == 0:
        k += 1

    return k


def _simple_inferno_colormap(norm):
    stops = np.array([
        [0.00, 0, 0, 4],
        [0.15, 31, 12, 72],
        [0.30, 85, 15, 109],
        [0.45, 136, 34, 106],
        [0.60, 186, 54, 85],
        [0.75, 227, 89, 51],
        [0.90, 249, 164, 24],
        [1.00, 252, 255, 164],
    ], dtype=np.float64)

    x = np.clip(norm, 0.0, 1.0)
    rgb = np.zeros((*x.shape, 3), dtype=np.float64)

    for i in range(len(stops) - 1):
        x0, r0, g0, b0 = stops[i]
        x1, r1, g1, b1 = stops[i + 1]

        mask = (x >= x0) & (x <= x1)

        if not np.any(mask):
            continue

        t = (x[mask] - x0) / max(x1 - x0, 1e-12)

        rgb[mask, 0] = r0 + t * (r1 - r0)
        rgb[mask, 1] = g0 + t * (g1 - g0)
        rgb[mask, 2] = b0 + t * (b1 - b0)

    return np.clip(rgb, 0, 255).astype(np.uint8)

