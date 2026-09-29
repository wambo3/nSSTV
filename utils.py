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

def save_image_file(pil_img, path, image_format=None, quality=95):
    """
    Save PIL image as JPEG/PNG based on extension or image_format.
    """
    _ensure_parent(path)

    if image_format is None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        image_format = ext or "jpg"

    image_format = image_format.lower().lstrip(".")

    if image_format in ("jpg", "jpeg"):
        if not path.lower().endswith((".jpg", ".jpeg")):
            path = _replace_ext(path, "jpg")
        pil_img.convert("RGB").save(path, "JPEG", quality=quality)
    elif image_format == "png":
        if not path.lower().endswith(".png"):
            path = _replace_ext(path, "png")
        pil_img.save(path, "PNG")
    else:
        raise ValueError(f"Unsupported image_format {image_format!r}; use jpg or png")

    return path


def save_metadata_sidecar(image_path, metadata):
    base, _ = os.path.splitext(image_path)
    return write_json(base + ".json", metadata)


def write_json(path, data):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


def append_wav_file(path, fs, audio, max_bytes=1_500_000_000):
    """
    Append float audio [-1, 1] as int16 PCM frames to an existing WAV file (or create if missing).
    Modifies RIFF header sizes in-place to avoid re-encoding the entire file under CPU load.
    Auto-segments into _part2.wav, _part3.wav etc. if file size approaches 32-bit RIFF limits.
    """
    fs = int(fs)
    pcm_bytes = _pcm16(audio).tobytes()
    if not pcm_bytes:
        return path

    target_path = path
    if os.path.exists(path):
        current_size = os.path.getsize(path)
        if current_size + len(pcm_bytes) > max_bytes:
            base, ext = os.path.splitext(path)
            segment_idx = 2
            while True:
                candidate = f"{base}_part{segment_idx}{ext}"
                if not os.path.exists(candidate) or os.path.getsize(candidate) + len(pcm_bytes) <= max_bytes:
                    target_path = candidate
                    break
                segment_idx += 1

    if not os.path.exists(target_path) or os.path.getsize(target_path) < 44:
        return write_wav_file(target_path, fs, audio)

    with open(target_path, "r+b") as f:
        f.seek(0, os.SEEK_END)
        f.write(pcm_bytes)
        file_len = f.tell()
        f.seek(4)
        f.write(struct.pack("<I", file_len - 8))
        f.seek(40)
        f.write(struct.pack("<I", file_len - 44))

    return target_path


def _metrics(reference, decoded_path):
    decoded = np.asarray(Image.open(decoded_path).convert("RGB"), dtype=np.float64)

    if decoded.shape != reference.shape:
        decoded = np.asarray(
            Image.open(decoded_path).convert("RGB").resize(
                (reference.shape[1], reference.shape[0]),
                Image.Resampling.LANCZOS,
            ),
            dtype=np.float64,
        )

    return _score(reference, decoded)


def find_ffmpeg():
    """
    Cross-platform ffmpeg discovery.

    Search order:
    - FFMPEG_BINARY
    - IMAGEIO_FFMPEG_EXE
    - PATH: ffmpeg, avconv
    """
    return _find_program(
        env_names=("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"),
        program_names=("ffmpeg", "avconv"),
    )


def find_ffprobe():
    """
    Cross-platform ffprobe discovery.
    """
    return _find_program(
        env_names=("FFPROBE_BINARY",),
        program_names=("ffprobe", "avprobe"),
    )


def configure_pydub_ffmpeg(require=False):
    """
    Configure pydub with ffmpeg/ffprobe if available.

    No machine-specific hardcoded paths are used.
    """
    AudioSegment = _get_audio_segment()
    if AudioSegment is None:
        if require:
            raise RuntimeError(
                "MP3/non-WAV support requires pydub.\n"
                "Install with:\n"
                "  pip install pydub"
            )
        return None

    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe()

    if ffmpeg:
        AudioSegment.converter = ffmpeg
        AudioSegment.ffmpeg = ffmpeg

    if ffprobe:
        AudioSegment.ffprobe = ffprobe

    if require and not ffmpeg:
        raise RuntimeError(
            "Could not find ffmpeg/avconv, so MP3 cannot be read/written.\n\n"
            "Install ffmpeg and make sure it is on PATH.\n\n"
            "macOS:\n"
            "  brew install ffmpeg\n\n"
            "Linux:\n"
            "  sudo apt install ffmpeg\n\n"
            "Windows:\n"
            "  Install ffmpeg and add its bin directory to PATH.\n\n"
            "Or set:\n"
            "  FFMPEG_BINARY=/path/to/ffmpeg"
        )

    return ffmpeg


def _get_audio_segment():
    global _audio_segment
    if _audio_segment is None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv.*")
                from pydub import AudioSegment as _AS
                _audio_segment = _AS
        except Exception:
            _audio_segment = False
    return _audio_segment if _audio_segment is not False else None


def ffmpeg_is_available():
    return configure_pydub_ffmpeg(require=False) is not None


def _find_program(env_names, program_names):
    candidates = []

    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            candidates.append(value)

    for name in program_names:
        found = shutil.which(name)
        if found:
            candidates.append(found)

    for path in candidates:
        if path and os.path.exists(path) and os.access(path, os.X_OK):
            if _program_runs(path):
                return path

    return None


def _program_runs(path):
    try:
        subprocess.run(
            [path, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
        return True
    except Exception:
        return False


def _call_hook(hooks, name, ctx):
    """
    Call hooks.<name>(ctx) if it is set.

    If the callable returns a dict, its key/value pairs are merged back
    into ctx so subsequent hooks and the caller can see any updates (e.g.
    a rig frequency read-back stored under ctx["rig_freq_hz"]).

    Parameters
    ----------
    hooks : RigHooks or None
    name  : str  — attribute name on RigHooks
    ctx   : dict — mutable context passed to the callback
    """
    if hooks is None:
        return

    fn = getattr(hooks, name, None)

    if fn is None or not callable(fn):
        return

    result = fn(ctx)

    if isinstance(result, dict):
        ctx.update(result)


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value

    return None


def _abs(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _clean_path(path):
    if path is None:
        return None
    path = str(path).strip()
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _replace_ext(path, image_format):
    base, _ = os.path.splitext(path)
    image_format = image_format.lower().lstrip(".")
    return base + "." + image_format


def _safe_filename_part(text):
    text = str(text)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "unknown"


def save_decode_report(report_path, report):
    return write_json(report_path, report)


def _env(name, default=None):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return value


def _env_bool(name, default=False):
    value = _env(name, None)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name, default):
    value = _env(name, None)
    return default if value is None else float(value)


def _env_int(name, default):
    value = _env(name, None)
    return default if value is None else int(value)


def _env_paths(name):
    value = _env(name, None)
    if not value:
        return None

    parts = [
        _clean_path(p)
        for p in value.split(os.pathsep)
        if p.strip()
    ]

    return parts or None


def _env_modes(name):
    """Comma separated mode names, or a single word like random."""
    value = _env(name, None)

    if not value:
        return None

    parts = [p.strip() for p in value.split(",") if p.strip()]

    return parts or None


def _score(a, b):
    diff = np.abs(b - a)
    mae = float(diff.mean())
    mse = float((diff ** 2).mean())
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 0 else 99.0

    x = a - a.mean()
    y = b - b.mean()
    denom = float(np.sqrt((x ** 2).sum() * (y ** 2).sum()))
    correlation = float((x * y).sum() / denom) if denom > 0 else 0.0

    return {
        "mae": round(mae, 3),
        "psnr": round(psnr, 3),
        "correlation": round(correlation, 4),
    }


def _weak_reasons(result):
    reasons = []

    if not result.get("ok"):
        reasons.append(result.get("error") or "roundtrip failed")
        return reasons

    m = result["metrics"]
    quality = (result.get("quality") or {}).get("overall", 0)

    if result.get("detected_mode") != result.get("mode"):
        reasons.append("VIS decoded as %s" % result.get("detected_mode"))
    if not result.get("sync_lock"):
        reasons.append("sync lock weak")
    if m["psnr"] < WEAK_PSNR_DB:
        reasons.append("psnr %.1f dB < %.1f" % (m["psnr"], WEAK_PSNR_DB))
    if m["correlation"] < WEAK_CORRELATION:
        reasons.append("correlation %.3f < %.2f" % (m["correlation"], WEAK_CORRELATION))
    if m["mae"] > WEAK_MAE:
        reasons.append("mae %.1f > %.1f" % (m["mae"], WEAK_MAE))
    if quality < 0.5:
        reasons.append("quality %.2f" % quality)

    return reasons


def _resave_as_png(path, image_format):
    """
    Re-save a just-written JPEG as PNG when image_format is "png" and drop
    the JPEG. The style painters always write JPEG first.
    """
    if image_format != "png":
        return path

    png_path = _replace_ext(path, "png")
    Image.open(path).save(png_path, "PNG")

    if os.path.exists(path) and path.lower().endswith((".jpg", ".jpeg")):
        os.remove(path)

    return png_path


def _reference_array(image_path, width, height, fit="contain", background=(0, 0, 0)):
    img = prepare_image_for_mode(
        image_path,
        width=width,
        height=height,
        fit=fit,
        background=background,
    )
    return np.asarray(img, dtype=np.float64)


def _roundtrip_first_image(
        audio_path,
        decode_kwargs,
        encoded_wav_out,
        encoded_mp3_out=None,
        sample_rate=48000,
        amplitude=0.80,
        mp3_bitrate="320k",
        custom_modes_json=None,
        raw_missing="ERROR: raw output missing.",
):
    """
    Shared roundtrip plumbing: decode every transmission in `audio_path`,
    then re-encode the first decoded raw image.

    Used by the script-mode and CLI roundtrip actions. Prints the result and
    returns the info dict, or None when nothing usable was decoded.
    """
    decoded = decode_all_audio_to_images(audio_path=audio_path, **decode_kwargs)

    if decoded.get("decoded_count", 0) < 1:
        pprint(decoded)
        return None

    first = decoded["images"][0]
    raw_path = first["decode_image"]["paths"].get("raw")

    if not raw_path:
        pprint(decoded)
        print(raw_missing)
        return None

    encoded = encode_image_to_sstv_audio_v2(
        image_path=raw_path,
        wav_out=encoded_wav_out,
        mp3_out=encoded_mp3_out,
        mode_name=first["mode_name"],
        sample_rate=sample_rate,
        amplitude=amplitude,
        mp3_bitrate=mp3_bitrate,
        custom_modes_json=custom_modes_json,
        include_vis=True,
    )

    info = {
        "decode_all": decoded,
        "encode_first_image": encoded,
    }

    pprint(info)

    return info


def _default_encoded_wav_for_image(image_path):
    base, _ = os.path.splitext(image_path)
    return base + "_sstv.wav"


def _default_output_base_for_audio(audio_path):
    base, _ = os.path.splitext(audio_path)
    return base + "_decoded.jpg"


def _default_roundtrip_wav_for_audio(audio_path):
    base, _ = os.path.splitext(audio_path)
    return base + "_roundtrip_sstv.wav"


def _auto_find_audio_file(folder):
    folder = _clean_path(folder)

    if not folder or not os.path.isdir(folder):
        return None

    exts = (".wav", ".wave", ".mp3")

    files = []

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(name)[1].lower()

        if ext not in exts:
            continue

        files.append(path)

    if not files:
        return None

    bad_markers = (
        "encoded",
        "roundtrip",
        "_sstv",
        "_decoded",
    )

    preferred = [
        p for p in files
        if not any(marker in os.path.basename(p).lower() for marker in bad_markers)
    ]

    return preferred[0] if preferred else files[0]


def _auto_find_image_file(folder):
    folder = _clean_path(folder)

    if not folder or not os.path.isdir(folder):
        return None

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    files = []

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(name)[1].lower()

        if ext not in exts:
            continue

        files.append(path)

    if not files:
        return None

    bad_markers = (
        "output",
        "decoded",
        "polaroid",
        "spectrogram",
        "stack",
    )

    preferred = [
        p for p in files
        if not any(marker in os.path.basename(p).lower() for marker in bad_markers)
    ]

    return preferred[0] if preferred else files[0]


def _api_catalog_dict():
    """
    Build {group: [entry, ...]} with live signatures/docstrings.

    Every entry is:
        {
            "name", "signature", "summary", "example", "doc", "object"
        }
    """
    import inspect

    def _signature_of(name, obj):
        if callable(obj):
            try:
                return f"{name}{inspect.signature(obj)}"
            except (TypeError, ValueError):
                return f"{name}(...)"
        return f"{name} = {obj!r}"

    catalog = {}

    for group, entries in _API_CATALOG:
        items = []

        for entry in entries:
            name = entry[0]
            summary = entry[1]
            example = entry[2]
            alias_of = entry[3] if len(entry) > 3 else None

            obj = globals().get(name)

            if obj is None:
                continue

            signature = _signature_of(name, obj)

            if alias_of:
                target = globals().get(alias_of)
                if target is not None:
                    signature = f"{signature}  ->  {_signature_of(alias_of, target)}"

            items.append({
                "name": name,
                "group": group,
                "signature": signature,
                "summary": summary,
                "example": example,
                "doc": (
                    inspect.getdoc(obj)
                    if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj)
                    else ""
                ),
                "alias_of": alias_of,
                "object": obj,
            })

        catalog[group] = items

    return catalog


class LossyMP3Warning(UserWarning):
    pass


class UnknownModeError(KeyError, ValueError):
    """
    Raised for a mode name the registry does not know.

    Subclasses both KeyError and ValueError so existing handlers keep working
    while new code can catch it as a plain input-validation error. The message
    lists the modes that ARE available.
    """

    def __init__(self, name, available=None):
        self.mode_name = name
        self.available = sorted(available) if available else []
        hint = ""
        if self.available:
            hint = "\nAvailable modes: " + ", ".join(self.available)
        super().__init__(f"Unknown mode {name!r}.{hint}")

    def __str__(self):
        return " ".join(self.args)


class ProtocolConstants:
    vis_bit_seconds: float = 0.030
    vis_bits_total: int = 10
    leader_tone_seconds: float = 0.300
    break_tone_seconds: float = 0.010

    @property
    def vis_header_seconds(self):
        return self.vis_bits_total * self.vis_bit_seconds

    @property
    def full_leader_and_break_seconds(self):
        return 2 * self.leader_tone_seconds + self.break_tone_seconds

    @property
    def full_vis_preamble_seconds(self):
        return self.full_leader_and_break_seconds + self.vis_header_seconds


class SSTVMode:
    name: str
    pixel_seconds: float
    sync_seconds: float
    porch_seconds: float
    experimental: bool = False


def _channels_tuple(channels):
    if channels is None:
        return None

    return tuple((str(k), float(off), float(dur)) for k, off, dur in channels)


class ToneDetector:
    """
    Sliding single-frequency detector using a normalized DFT bin.
    """

    def __init__(self, fs, samples):
        self.fs = fs
        self.samples = samples

    @staticmethod
    def goertzel(digital_freq, samples, window_size):
        """
        Normalized single-bin DFT power.

        Power = |sum_n x[n] exp(-j*w*n)|^2 / N^2
        """
        window_size = int(window_size)
        x = np.asarray(samples[:window_size], dtype=np.float64)
        n = np.arange(window_size)
        X = np.dot(x, np.exp(-1j * digital_freq * n))
        return (abs(X) ** 2) / (window_size ** 2)

    def tone_power(self, freq_hz, start_time, duration):
        fs = self.fs
        start = int(round(start_time * fs))
        size = int(round(duration * fs))

        if start < 0 or start + size > len(self.samples) or size <= 0:
            return 0.0

        omega = 2 * np.pi * freq_hz / fs
        return self.goertzel(omega, self.samples[start:start + size], size)

    def scan_for_tone(
            self,
            target_freq,
            window_seconds,
            step_seconds,
            scan_start_seconds,
            scan_duration_seconds,
            block_windows=4096,
    ):
        fs = self.fs
        window_size = max(round(fs * window_seconds), 1)
        step_size = max(round(fs * step_seconds), 1)

        scan_start_sample = max(0, round(fs * scan_start_seconds))
        scan_end_sample = round(fs * (scan_start_seconds + scan_duration_seconds))
        scan_end_sample = min(scan_end_sample, len(self.samples))

        if scan_end_sample <= scan_start_sample:
            return []

        seg = self.samples[scan_start_sample:scan_end_sample + window_size - 1]

        if len(seg) < window_size:
            return []

        n = np.arange(window_size)
        kernel = np.exp(-2j * np.pi * target_freq / fs * n)

        views = np.lib.stride_tricks.sliding_window_view(seg, window_size)[::step_size]

        n_windows = min(
            len(views),
            math.ceil((scan_end_sample - scan_start_sample) / step_size),
        )

        views = views[:n_windows]
        powers = np.empty(n_windows, dtype=np.float64)

        for b in range(0, n_windows, block_windows):
            blk = views[b:b + block_windows]
            powers[b:b + len(blk)] = np.abs(blk @ kernel) ** 2

        powers /= window_size ** 2

        times = (scan_start_sample + np.arange(n_windows) * step_size) / fs
        return list(zip(times.tolist(), powers.tolist()))

    @staticmethod
    def auto_threshold(results, factor=5):
        if not results:
            return 1e-12

        powers = np.array([p for _, p in results])
        return max(np.median(powers) * factor, 1e-12)

    @staticmethod
    def find_regions(
            results,
            threshold,
            min_duration_seconds,
            tolerance=0.7,
            max_gap_seconds=0.0,
    ):
        """
        Find above-threshold tone regions, allowing short internal gaps.
        """
        if len(results) < 2:
            return []

        step = results[1][0] - results[0][0]
        above = [p > threshold for _, p in results]
        n = len(above)

        min_windows = max(round(min_duration_seconds * tolerance / step), 1)
        max_gap_windows = int(round(max_gap_seconds / step))

        regions = []
        i = 0

        while i < n:
            if not above[i]:
                i += 1
                continue

            start = i
            last_above = i
            j = i + 1

            while j < n:
                if above[j]:
                    last_above = j
                elif j - last_above > max_gap_windows:
                    break
                j += 1

            if last_above - start + 1 >= min_windows:
                regions.append((results[start][0], results[last_above][0]))

            i = last_above + 1

        return regions

    @staticmethod
    def find_tone_edges(results, threshold, min_consecutive):
        above = [power > threshold for _, power in results]

        start_index = None
        for i in range(0, len(above) - min_consecutive + 1):
            if all(above[i:i + min_consecutive]):
                start_index = i
                break

        if start_index is None:
            return None, None

        end_index = None
        for i in range(start_index, len(above) - min_consecutive + 1):
            if all(not v for v in above[i:i + min_consecutive]):
                end_index = i
                break

        start_time = results[start_index][0]
        end_time = results[end_index][0] if end_index is not None else None

        return start_time, end_time

    @staticmethod
    def refine_edges(results, run_start, run_end, window_seconds, fraction=0.25):
        """
        Convert threshold-crossing window-start times to approximate tone boundaries.

        A rectangular-window tone power rises roughly as overlap^2. The quarter
        power crossing corresponds approximately to half overlap, meaning the
        window center is on the real boundary.
        """
        times = np.array([t for t, _ in results])
        powers = np.array([p for _, p in results])

        in_run = (times >= run_start) & (times < run_end)

        if not np.any(in_run):
            return run_start, run_end

        plateau = np.median(powers[in_run])
        edge_thr = fraction * plateau

        search = (
                (times >= run_start - window_seconds) &
                (times <= run_end + window_seconds) &
                (powers >= edge_thr)
        )

        idx = np.where(search)[0]

        if len(idx) == 0:
            return run_start, run_end

        half = window_seconds / 2
        return times[idx[0]] + half, times[idx[-1]] + half


class VISHeaderReader:
    FREQ_ZERO = VIS_ZERO_HZ
    FREQ_ONE = VIS_ONE_HZ

    def __init__(self, detector: ToneDetector, protocol: ProtocolConstants):
        self.detector = detector
        self.protocol = protocol

    def read_full_vis_metrics(self, vis_start_time, freq_offset_hz=0.0):
        """
        Read all ten VIS bit slots and report the tone powers behind them.

        Bit 0 is the 1200 Hz start bit, bits 1 to 7 are the data bits LSB
        first, bit 8 is parity, and bit 9 is the 1200 Hz stop bit. Every slot
        is probed at 1100, 1200, 1300 and 1900 Hz so the caller can tell a
        confident bit read from a weak one. That distinction matters most for
        VIS 0, Robot 12 Color, whose seven data bits are all zero: any reader
        that shrugs and reports zero for an ambiguous bit will happily produce
        a parity-clean all-zero code and steal the frame from another mode.

        Returns complete=False when the window would run off the end of the
        recording, in which case the caller must ignore the result.
        """
        fs = self.detector.fs
        samples = self.detector.samples
        bit_s = self.protocol.vis_bit_seconds

        window_s = bit_s * 0.6
        window_size = max(round(fs * window_s), 1)
        side = (bit_s - window_s) / 2

        probe_freqs = {
            "p1100": 1100.0 + freq_offset_hz,
            "p1200": 1200.0 + freq_offset_hz,
            "p1300": 1300.0 + freq_offset_hz,
            "p1900": 1900.0 + freq_offset_hz,
        }

        records = []

        for bit_index in range(self.protocol.vis_bits_total):
            t = vis_start_time + bit_index * bit_s + side
            start = round(fs * t)

            if start < 0 or start + window_size > len(samples):
                return {
                    "complete": False,
                    "records": records,
                    "bits": [],
                    "powers": [],
                    "vis_value": None,
                    "parity_ok": False,
                }

            chunk = samples[start:start + window_size]

            rec = {"bit_index": bit_index}

            for key, freq in probe_freqs.items():
                omega = 2 * np.pi * freq / fs
                rec[key] = self.detector.goertzel(omega, chunk, window_size)

            records.append(rec)

        bits = []
        powers = []

        for rec in records[1:9]:
            p_zero = rec["p1300"]
            p_one = rec["p1100"]

            bits.append(1 if p_one > p_zero else 0)
            powers.append((p_zero, p_one))

        vis_value, parity_ok = self.decode(bits)

        return {
            "complete": True,
            "records": records,
            "bits": bits,
            "powers": powers,
            "vis_value": vis_value,
            "parity_ok": parity_ok,
        }

    def read_bits_with_powers(self, vis_start_time, freq_offset_hz=0.0):
        metrics = self.read_full_vis_metrics(vis_start_time, freq_offset_hz)
        return metrics["bits"], metrics["powers"]

    def read_bits(self, vis_start_time, freq_offset_hz=0.0):
        bits, _ = self.read_bits_with_powers(vis_start_time, freq_offset_hz)
        return bits

    @staticmethod
    def decode(bits):
        """
        Decode 7-bit VIS value + even parity.
        """
        if len(bits) < 8:
            return None, False

        value = 0

        for i, b in enumerate(bits[:7]):
            value |= (b << i)

        parity_bit = bits[7]
        ones_count = sum(bits[:7])
        parity_ok = (ones_count % 2) == parity_bit

        return value, parity_ok

    def sync_bit_ratio(self, vis_start_time, bit_index, freq_offset_hz=0.0):
        bit_s = self.protocol.vis_bit_seconds
        window_s = bit_s * 0.6
        side = (bit_s - window_s) / 2
        t = vis_start_time + bit_index * bit_s + side

        p1200 = self.detector.tone_power(1200 + freq_offset_hz, t, window_s)
        p1100 = self.detector.tone_power(1100 + freq_offset_hz, t, window_s)
        p1300 = self.detector.tone_power(1300 + freq_offset_hz, t, window_s)
        p1900 = self.detector.tone_power(1900 + freq_offset_hz, t, window_s)

        return p1200 / max(p1100, p1300, p1900, 1e-12)


class ToneSynth:
    """
    Continuous-phase tone synthesizer.
    """

    def __init__(self, fs=48000, amplitude=0.80):
        self.fs = int(fs)
        self.amplitude = float(amplitude)
        self.phase = 0.0
        self.chunks = []

    def append_silence(self, seconds):
        n = int(round(seconds * self.fs))
        if n > 0:
            self.chunks.append(np.zeros(n, dtype=np.float32))

    def append_tone(self, freq_hz, seconds):
        n = int(round(seconds * self.fs))
        if n <= 0:
            return

        freqs = np.full(n, float(freq_hz), dtype=np.float64)
        self.append_freqs(freqs)

    def append_freqs(self, freqs):
        freqs = np.asarray(freqs, dtype=np.float64)

        if len(freqs) == 0:
            return

        phase_inc = 2.0 * np.pi * freqs / self.fs
        phase = self.phase + np.cumsum(phase_inc)

        y = self.amplitude * np.sin(phase)

        self.phase = float(phase[-1] % (2.0 * np.pi))
        self.chunks.append(y.astype(np.float32))

    def finalize(self, fade_seconds=0.005):
        if not self.chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(self.chunks).astype(np.float32)

        fade_n = int(round(fade_seconds * self.fs))
        fade_n = min(fade_n, len(audio) // 2)

        if fade_n > 1:
            fade_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

            audio[:fade_n] *= fade_in
            audio[-fade_n:] *= fade_out

        return audio


class SSTVEncoder:
    BLACK_HZ = SSTV_BLACK_HZ
    WHITE_HZ = SSTV_WHITE_HZ

    def __init__(
            self,
            registry: ModeRegistry = None,
            mode_name="PD-180",
            fs=48000,
            amplitude=0.80,
            protocol: ProtocolConstants = None,
            vis_code=None,
            image_fit="contain",
            image_background=(0, 0, 0),
            line_structure="standard",
            oversample=4,
    ):
        self.registry = registry or ModeRegistry()
        self.mode_name = mode_name
        self.fs = int(fs)
        if self.fs <= 0:
            raise ValueError(f"sample_rate must be a positive integer, got {fs}")
        self.amplitude = float(amplitude)
        if self.amplitude <= 0:
            raise ValueError(f"amplitude must be > 0, got {amplitude}")
        if self.amplitude > 1.0:
            warnings.warn(
                f"amplitude {amplitude} is above full scale (1.0); the output "
                "will be clipped. Use <= 1.0 for a clean recording.",
                stacklevel=2,
            )
        self.protocol = protocol or ProtocolConstants()
        self.image_fit = image_fit
        self.image_background = image_background
        # Sub-pixel oversampling.  When >1 the source image is resized to
        # (width * oversample, height * oversample); the extra rows are
        # box-averaged back to one row per scan line (see _load_image_rgb)
        # and the encoder uses linear interpolation along each line when
        # painting the per-sample tone, so the audio preserves source
        # detail beyond the mode's nominal pixel grid instead of
        # collapsing each scan to a staircase.
        # Set to 1 to restore the old "downsample to mode dims then hold"
        # behaviour.
        #
        # Validation: 1..64 is the practical range.  oversample > 64 would
        # allocate > 1.2 GB of float64 for a 640x496 mode (the work buffer
        # alone, before the tone array), which is way past what most
        # consumer hardware can handle without paging.  We reject negative
        # or zero values explicitly because the symptom (silent acceptance
        # via max(1, ...)) is worse than a clear error.
        if not isinstance(oversample, int) or isinstance(oversample, bool):
            raise ValueError(
                f"oversample must be an integer >= 1, got {oversample!r}"
            )
        if oversample < 1:
            raise ValueError(
                f"oversample must be >= 1, got {oversample}"
            )
        if oversample > 64:
            raise ValueError(
                f"oversample must be <= 64 (would exhaust memory), got {oversample}. "
                "Use a smaller value; 4 is the recommended default."
            )
        self.oversample = oversample

        if mode_name not in self.registry.image_layouts:
            raise ValueError(
                f"Mode {mode_name!r} cannot currently be encoded.\n"
                f"Supported encoder modes: {self.registry.supported_encoder_modes()}\n"
                "For custom modes, call registry.add_custom_mode(...) first "
                "or load a custom modes JSON file."
            )

        # Pre-flight memory check: the encoder allocates a float64 buffer of
        # size (width * oversample) * (height * oversample) * 3 channels
        # plus the per-line plane arrays.  Cap the total at ~1 GB so a
        # user can't accidentally OOM their machine by combining a large
        # mode with a high oversample.
        layout_w = self.registry.get_layout(mode_name).width
        layout_h = self.registry.get_layout(mode_name).height
        eff_w = layout_w * self.oversample
        eff_h = layout_h * self.oversample
        # 8 bytes/float64 * 3 channels for the RGB buffer + ~2x for planes
        # (planes are indexed per mode-height but per-mode-width so they
        # have ~1/oversample the pixels of the RGB buffer, but they're
        # allocated as full 2D arrays in _image_to_planes).
        estimated_bytes = eff_w * eff_h * 3 * 8 * 2
        if estimated_bytes > 1024 * 1024 * 1024:  # 1 GB
            raise ValueError(
                f"oversample={self.oversample} on {mode_name!r} "
                f"({layout_w}x{layout_h}) would allocate ~{estimated_bytes // (1024*1024)} MB "
                f"of working memory.  Reduce --oversample (4 is the recommended default) "
                f"or pick a smaller mode."
            )

        self.line_structure = line_structure
        self.layout = self.registry.get_layout(
            mode_name,
            line_structure=line_structure,
        )

        if vis_code is not None:
            self.vis_code = int(vis_code)
        else:
            self.vis_code = self.registry.find_vis_code_for_mode(mode_name)

        # Pull the 8-bit extended VIS byte (if any) for 16-bit-VIS modes.
        # append_vis_header() emits it right after the standard 8-bit VIS.
        self.extended_vis_code = self.registry.extended_vis_code_for_mode(mode_name)

    def _load_image_rgb(self, image_path):
        L = self.layout

        # When oversampling is on, the image is resized to
        # (width * oversample, height * oversample) instead of (width, height).
        # That keeps source detail that the old "downsample to mode dims"
        # behaviour would have thrown away before the encoder ever saw it.
        # _paint_channel then linearly interpolates between the oversampled
        # buffer's pixel centres as each output audio sample is emitted,
        # giving sub-pixel-accurate tones instead of a per-pixel staircase.
        w_eff = L.width * self.oversample
        h_eff = L.height * self.oversample

        img = prepare_image_for_mode(
            image_path,
            width=w_eff,
            height=h_eff,
            fit=getattr(self, "image_fit", "contain"),
            background=getattr(self, "image_background", (0, 0, 0)),
        )

        arr = np.asarray(img, dtype=np.float64)

        # Oversampling only makes sense HORIZONTALLY: _paint_channel
        # interpolates along a scan line, which is a time axis, so extra
        # columns become sub-pixel tone detail.  Vertically an SSTV mode
        # has exactly L.height lines, and everything downstream (direct
        # planes[...][row] lookups, the PD Y[0::2]/Y[1::2] pairing, the
        # Robot chroma line sharing) indexes rows 0..L.height-1.  Handing
        # those consumers an (L.height * oversample)-row buffer made line
        # r read oversampled row r, i.e. source row r / oversample - so the
        # picture came out as its top 1/oversample, stretched vertically.
        # Box-average each group of `oversample` rows back down to one row
        # per line, which keeps the extra vertical detail as a proper
        # area-average instead of throwing it away or mis-indexing it.
        if self.oversample > 1:
            arr = arr.reshape(
                L.height, self.oversample, arr.shape[1], arr.shape[2]
            ).mean(axis=1)

        return arr

    @staticmethod
    def _rgb_to_ycrcb(rgb):
        """
        RGB to the classic SSTV Y, R-Y, B-Y triple, scaled 0-255.

        ITU-R BT.601 studio-range, exactly as specified in Appendix B of the
        Dayton SSTV mode paper and used by the Robot and PD modes: black
        lands at Y=16 and white at Y=235, and both colour-difference signals
        are centred on 128. "Cr" is R-Y and "Cb" is B-Y.
        """
        R = rgb[..., 0]
        G = rgb[..., 1]
        B = rgb[..., 2]

        Y = 16.0 + (
                65.738 * R
                + 129.057 * G
                + 25.064 * B
        ) / 256.0

        Cr = 128.0 + (
                112.439 * R
                - 94.154 * G
                - 18.285 * B
        ) / 256.0

        Cb = 128.0 + (
                -37.945 * R
                - 74.494 * G
                + 112.439 * B
        ) / 256.0

        Y = np.clip(Y, 0.0, 255.0)
        Cr = np.clip(Cr, 0.0, 255.0)
        Cb = np.clip(Cb, 0.0, 255.0)

        return Y, Cr, Cb

    def _image_to_planes(self, rgb):
        L = self.layout

        if L.color in ("rgb", "rgb3"):
            return {
                "R": rgb[..., 0],
                "G": rgb[..., 1],
                "B": rgb[..., 2],
            }

        if L.color == "bw":
            # Robot B&W modes: single luminance plane (Rec. 601 luma).
            R = rgb[..., 0]
            G = rgb[..., 1]
            B = rgb[..., 2]
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            return {"Y": Y}

        Y, Cr, Cb = self._rgb_to_ycrcb(rgb)

        if L.color == "ycrcb":
            return {
                "Y": Y,
                "Cr": Cr,
                "Cb": Cb,
            }

        if L.color == "ycrcb420":
            return {
                "Y": Y,
                "Cr": Cr,
                "Cb": Cb,
            }

        if L.color == "pd":
            n = L.n_lines

            Y0 = Y[0::2][:n]
            Y1 = Y[1::2][:n]

            Cr_pair = 0.5 * (Cr[0::2][:n] + Cr[1::2][:n])
            Cb_pair = 0.5 * (Cb[0::2][:n] + Cb[1::2][:n])

            return {
                "Y0": Y0,
                "Y1": Y1,
                "Cr": Cr_pair,
                "Cb": Cb_pair,
            }

        raise ValueError(f"Unsupported color model for encoding: {L.color!r}")

    def _lum_to_freq(self, values):
        values = np.asarray(values, dtype=np.float64)
        values = np.clip(values, 0.0, 255.0)

        return self.BLACK_HZ + (values / 255.0) * (self.WHITE_HZ - self.BLACK_HZ)

    def _paint_channel(self, freq_line, offset_s, duration_s, values):
        """Paint per-sample tones for one scan.

        ``values`` is the row of the image plane to transmit.  When
        ``self.oversample > 1`` it has width ``L.width * oversample``
        and we use linear interpolation between adjacent samples so the
        output tone reflects sub-pixel detail; otherwise the original
        nearest-neighbour (zero-order hold) behaviour is preserved.
        """
        n = len(freq_line)

        s0 = int(round(offset_s * self.fs))
        s1 = int(round((offset_s + duration_s) * self.fs))

        s0 = max(0, min(s0, n))
        s1 = max(0, min(s1, n))

        if s1 <= s0:
            return

        values = np.asarray(values, dtype=np.float64)
        W = len(values)

        sample_times = (np.arange(s0, s1, dtype=np.float64) + 0.5) / self.fs

        if self.oversample > 1:
            # Sub-pixel position in the oversampled buffer.
            # Map sample time -> fractional pixel index, then linearly
            # interpolate between adjacent pixels.  This is what makes
            # the audio carry information at sub-mode-pixel resolution
            # instead of being a staircase at the mode's pixel rate.
            x = (sample_times - offset_s) / duration_s * W
            x = np.clip(x, 0.0, float(W - 1))
            interp = np.interp(x, np.arange(W, dtype=np.float64), values)
            freq_line[s0:s1] = self._lum_to_freq(interp)
        else:
            pix = np.floor(((sample_times - offset_s) / duration_s) * W).astype(np.int64)
            pix = np.clip(pix, 0, W - 1)
            freq_line[s0:s1] = self._lum_to_freq(values[pix])

    def _make_line_freqs(self, line_index, planes):
        L = self.layout
        line_n = int(round(L.line_seconds * self.fs))

        if L.color == "rgb3":
            order = L.line_channel_order or ("R", "G", "B")
            n_components = len(order)

            row = line_index // n_components

            freq_line = np.full(line_n, self.BLACK_HZ, dtype=np.float64)

            if row >= L.height:
                return freq_line

            if L.sync_seconds > 0:
                s0 = int(round(L.sync_offset * self.fs))
                s1 = int(round((L.sync_offset + L.sync_seconds) * self.fs))

                s0 = max(0, min(s0, line_n))
                s1 = max(0, min(s1, line_n))

                if s1 > s0:
                    freq_line[s0:s1] = SSTV_SYNC_HZ

            _component, off, dur = L.channels[0]

            self._paint_channel(
                freq_line,
                off,
                dur,
                planes[order[line_index % n_components]][row],
            )

            return freq_line

        freq_line = np.full(line_n, self.BLACK_HZ, dtype=np.float64)

        if L.sync_seconds > 0:
            s0 = int(round(L.sync_offset * self.fs))
            s1 = int(round((L.sync_offset + L.sync_seconds) * self.fs))

            s0 = max(0, min(s0, line_n))
            s1 = max(0, min(s1, line_n))

            if s1 > s0:
                freq_line[s0:s1] = SSTV_SYNC_HZ

        if L.channels_odd is not None and line_index % 2 == 1:
            channels = L.channels_odd
        else:
            channels = L.channels

        for key, off, dur in channels:
            self._paint_channel(
                freq_line,
                off,
                dur,
                planes[key][line_index],
            )

        return freq_line

    def append_vis_header(self, synth):
        """
        Append standard VIS sounds:

        1900 Hz for 300 ms
        1200 Hz for 10 ms
        1900 Hz for 300 ms
        1200 Hz start bit
        7 data bits, LSB first:
            1100 Hz = 1
            1300 Hz = 0
        parity bit
        1200 Hz stop bit

        For 16-bit-VIS MMSSTV modes (primary VIS = 0x23), the stop bit
        above is followed by another 7 data bits + parity holding the
        extended mode byte, with no extra leader tones. This is the
        convention used by MMSSTV's MP/MR family per the SSTV Handbook
        "16-bit VIS" note.
        """
        if self.vis_code is None:
            raise ValueError(
                f"Mode {self.mode_name!r} has no VIS code. "
                "Add one with registry.add_vis_code(...) or encode with include_vis=False."
            )

        if not 0 <= int(self.vis_code) <= 127:
            raise ValueError("Standard VIS code must be in 0..127")

        p = self.protocol
        code = int(self.vis_code)

        synth.append_tone(SSTV_LEADER_HZ, p.leader_tone_seconds)
        synth.append_tone(SSTV_SYNC_HZ, p.break_tone_seconds)
        synth.append_tone(SSTV_LEADER_HZ, p.leader_tone_seconds)

        synth.append_tone(SSTV_SYNC_HZ, p.vis_bit_seconds)

        data_bits = [(code >> i) & 1 for i in range(7)]

        for bit in data_bits:
            synth.append_tone(VIS_ONE_HZ if bit else VIS_ZERO_HZ, p.vis_bit_seconds)

        parity_bit = sum(data_bits) % 2
        synth.append_tone(VIS_ONE_HZ if parity_bit else VIS_ZERO_HZ, p.vis_bit_seconds)

        synth.append_tone(SSTV_SYNC_HZ, p.vis_bit_seconds)

        # 16-bit extended VIS: append extended byte as a full second 10-slot
        # frame (start bit + 7 data bits + parity + stop bit), starting right
        # after the primary stop bit.  This is the convention the decoder
        # expects (read_full_vis_metrics reads a complete 10-bit frame).
        ext_code = self.extended_vis_code
        if ext_code is not None:
            # Start bit for the extended byte's own frame.
            synth.append_tone(SSTV_SYNC_HZ, p.vis_bit_seconds)

            ext_bits = [(int(ext_code) >> i) & 1 for i in range(7)]

            for bit in ext_bits:
                synth.append_tone(VIS_ONE_HZ if bit else VIS_ZERO_HZ, p.vis_bit_seconds)

            ext_parity = sum(ext_bits) % 2
            synth.append_tone(VIS_ONE_HZ if ext_parity else VIS_ZERO_HZ, p.vis_bit_seconds)

            # Stop bit for the extended byte's frame.
            synth.append_tone(SSTV_SYNC_HZ, p.vis_bit_seconds)

    def verify_vis_header_audio(self, audio, pre_silence=0.50):
        """
        Verify generated audio contains the expected VIS header.
        """
        if self.vis_code is None:
            return {
                "ok": False,
                "reason": "No VIS code for this mode",
            }

        samples = np.asarray(audio, dtype=np.float64)
        det = ToneDetector(self.fs, samples)
        reader = VISHeaderReader(det, self.protocol)
        p = self.protocol

        leader_start = float(pre_silence)
        vis_start = leader_start + p.full_leader_and_break_seconds

        bits = reader.read_bits(vis_start, freq_offset_hz=0.0)
        value, parity_ok = reader.decode(bits)

        first_1900 = det.tone_power(SSTV_LEADER_HZ, leader_start + 0.050, 0.200)
        first_1200 = det.tone_power(SSTV_SYNC_HZ, leader_start + 0.050, 0.200)

        break_1200 = det.tone_power(
            SSTV_SYNC_HZ,
            leader_start + p.leader_tone_seconds,
            p.break_tone_seconds,
        )
        break_1900 = det.tone_power(
            SSTV_LEADER_HZ,
            leader_start + p.leader_tone_seconds,
            p.break_tone_seconds,
        )

        second_1900 = det.tone_power(
            SSTV_LEADER_HZ,
            leader_start + p.leader_tone_seconds + p.break_tone_seconds + 0.050,
            0.200,
        )
        second_1200 = det.tone_power(
            SSTV_SYNC_HZ,
            leader_start + p.leader_tone_seconds + p.break_tone_seconds + 0.050,
            0.200,
        )

        start_ratio = reader.sync_bit_ratio(vis_start, 0)
        stop_ratio = reader.sync_bit_ratio(vis_start, 9)

        first_ratio = first_1900 / max(first_1200, 1e-12)
        break_ratio = break_1200 / max(break_1900, 1e-12)
        second_ratio = second_1900 / max(second_1200, 1e-12)

        ok = (
                value == int(self.vis_code)
                and parity_ok
                and first_ratio > 3.0
                and break_ratio > 1.3
                and second_ratio > 3.0
                and start_ratio > 3.0
                and stop_ratio > 3.0
        )

        # FIX: For 16-bit extended VIS modes (MMSSTV MP/MR family), the
        # encoder appends a second 10-slot VIS frame carrying the extended
        # byte right after the primary stop bit.  The old code only checked
        # the primary byte (VIS 35) and returned ok=True even when the
        # extended byte was corrupt or missing, giving a false green light.
        # We now also decode the extended frame and verify it matches what
        # was encoded.
        ext_bits = None
        ext_value = None
        ext_parity_ok = None
        if ok and self.extended_vis_code is not None:
            ext_bits = reader.read_bits(
                vis_start + p.vis_header_seconds,
                freq_offset_hz=0.0,
            )
            ext_value, ext_parity_ok = reader.decode(ext_bits)
            extended_ok = (
                ext_parity_ok
                and ext_value == int(self.extended_vis_code)
            )
            ok = ok and extended_ok

        result = {
            "ok": bool(ok),
            "expected_vis_code": int(self.vis_code),
            "decoded_vis_code": value,
            "bits": bits,
            "parity_ok": bool(parity_ok),
            "vis_start_seconds": float(vis_start),
            "vis_header_duration_seconds": float(p.full_vis_preamble_seconds),
            "first_1900_vs_1200_ratio": float(first_ratio),
            "break_1200_vs_1900_ratio": float(break_ratio),
            "second_1900_vs_1200_ratio": float(second_ratio),
            "start_bit_1200_ratio": float(start_ratio),
            "stop_bit_1200_ratio": float(stop_ratio),
        }

        if self.extended_vis_code is not None:
            result["expected_extended_vis_code"] = int(self.extended_vis_code)
            result["decoded_extended_vis_code"] = ext_value
            result["extended_bits"] = ext_bits
            result["extended_parity_ok"] = bool(ext_parity_ok) if ext_parity_ok is not None else None

        return result

    def encode_image_to_audio(
            self,
            image_path,
            pre_silence=0.50,
            post_silence=0.50,
            include_vis=True,
    ):
        """
        Encode image to SSTV audio samples.

        include_vis=True means the audio includes standard VIS leader/header.
        For modes with no standard VIS code (such as the MMSSTV -N modes
        that use N-VIS), include_vis is automatically downgraded to False
        with a warning, since emitting the leader without a valid VIS
        would just confuse receivers.  The decoder still finds these
        transmissions via line-rate detection.
        """
        rgb = self._load_image_rgb(image_path)
        planes = self._image_to_planes(rgb)

        synth = ToneSynth(
            fs=self.fs,
            amplitude=self.amplitude,
        )

        synth.append_silence(pre_silence)

        # N-VIS modes: now that we register the N-VIS code as the standard
        # VIS code, the encoder emits a normal VIS header.  We just warn
        # the user if the N-VIS code collides with another mode's VIS code
        # so they know the receiver may misidentify the mode.
        if include_vis and self.vis_code is not None:
            nvis = self.registry.nvis_code_for_mode(self.mode_name)
            if nvis is not None and int(nvis) != int(self.vis_code):
                # Shouldn't happen - both should be the same value now.
                pass

        if include_vis and self.vis_code is None:
            # Truly no VIS code (custom mode without one) - skip header.
            warnings.warn(
                f"Mode {self.mode_name!r} has no VIS code; skipping the "
                "VIS header. Decode with --force-mode or rely on line-rate "
                "detection.",
                stacklevel=2,
            )
            include_vis = False

        if include_vis:
            self.append_vis_header(synth)

        if self.layout.leadin_seconds > 0:
            synth.append_tone(SSTV_SYNC_HZ, self.layout.leadin_seconds)

        for line in range(self.layout.n_lines):
            freq_line = self._make_line_freqs(line, planes)
            synth.append_freqs(freq_line)

        synth.append_silence(post_silence)

        return synth.finalize()

    def encode_image_file(
            self,
            image_path,
            wav_out=None,
            mp3_out=None,
            mp3_bitrate="320k",
            pre_silence=0.50,
            post_silence=0.50,
            include_vis=True,
            fail_on_mp3_error=False,
    ):
        audio = self.encode_image_to_audio(
            image_path=image_path,
            pre_silence=pre_silence,
            post_silence=post_silence,
            include_vis=include_vis,
        )

        paths = {}
        mp3_error = None

        if wav_out:
            write_wav_file(wav_out, self.fs, audio)
            paths["wav"] = wav_out

        if mp3_out:
            try:
                write_mp3_file(mp3_out, self.fs, audio, bitrate=mp3_bitrate)
                paths["mp3"] = mp3_out
            except Exception as e:
                mp3_error = str(e)

                if fail_on_mp3_error:
                    raise

                if os.path.exists(mp3_out) and os.path.getsize(mp3_out) < 1024:
                    try:
                        os.remove(mp3_out)
                    except Exception:
                        pass

        vis_header_check = None
        if include_vis:
            vis_header_check = self.verify_vis_header_audio(
                audio,
                pre_silence=pre_silence,
            )

        return {
            "mode_name": self.mode_name,
            "vis_code": int(self.vis_code) if self.vis_code is not None else None,
            "include_vis": bool(include_vis),
            "vis_header_check": vis_header_check,
            "sample_rate": int(self.fs),
            "duration_seconds": float(len(audio) / self.fs),
            "paths": paths,
            "mp3_bitrate": mp3_bitrate if mp3_out else None,
            "mp3_warning": (
                "MP3 output is lossy even at 320k. Use WAV for clean/archival SSTV."
                if mp3_out else None
            ),
            "mp3_error": mp3_error,
        }


class FMDemodulator:
    """
    SSTV FM audio-tone demodulator.

    Method:
    - bandpass audio
    - analytic signal by Hilbert transform
    - unwrap phase
    - differentiate phase to get instantaneous frequency
    - amplitude-weight averages

    Runs in chunks and stores float32, so memory is bounded by the chunk size
    plus three arrays of one float32 per sample, not by four float64 arrays
    per sample plus a full-length FFT.
    """

    CHUNK_SAMPLES = 1 << 21
    PAD_SAMPLES = 8192

    def __init__(
            self,
            fs,
            samples,
            freq_offset_hz=0.0,
            band=(900.0, 2600.0),
            smoothing_seconds=SMOOTHING_MAX_SECONDS,
            chunk_samples=None,
    ):
        self.fs = fs
        self.n = len(samples)
        self.smoothing_seconds = (
            SMOOTHING_MAX_SECONDS if smoothing_seconds is None else float(smoothing_seconds)
        )

        nyq = fs * 0.5
        lo = max(50.0, band[0])
        hi = min(band[1], nyq * 0.95)

        if hi <= lo:
            raise ValueError(f"Invalid band {band} for sample rate {fs}")

        sos = butter(4, (lo, hi), btype="bandpass", fs=fs, output="sos")

        chunk = int(chunk_samples or self.CHUNK_SAMPLES)
        chunk = max(4 * self.PAD_SAMPLES, chunk)
        pad = self.PAD_SAMPLES

        samples32 = np.asarray(samples, dtype=np.float32)

        freq = np.empty(self.n, dtype=np.float32)
        amp = np.empty(self.n, dtype=np.float32)

        start = 0

        while start < self.n:
            end = min(start + chunk, self.n)

            a = max(0, start - 1 - pad)
            b = min(self.n, end + pad)

            seg = sosfiltfilt(sos, samples32[a:b].astype(np.float64))
            analytic = _analytic(seg)

            k0 = start - a
            k1 = end - a

            kept = analytic[k0:k1 + 1] if k1 + 1 <= len(analytic) else analytic[k0:]

            if len(kept) < 2:
                freq[start:end] = SSTV_LEADER_HZ
                amp[start:end] = 1.0
                start = end
                continue

            phase = np.unwrap(np.angle(kept))

            if len(phase) > 1:
                f = np.diff(phase) * fs / (2 * np.pi)
            else:
                f = np.zeros(1, dtype=np.float64)

            if len(f) < (end - start):
                f = np.pad(f, (0, (end - start) - len(f)), mode="edge")

            f = f[:end - start] - freq_offset_hz
            f[~np.isfinite(f)] = SSTV_LEADER_HZ
            np.clip(f, 800.0, 2800.0, out=f)

            ampvals = np.abs(kept[1:])

            if len(ampvals) < (end - start):
                ampvals = np.pad(ampvals, (0, (end - start) - len(ampvals)), mode="edge")

            freq[start:end] = f.astype(np.float32)
            amp[start:end] = ampvals[:end - start].astype(np.float32)

            del analytic, phase, seg, kept, f, ampvals
            start = end

        k = _median_kernel(fs, self.smoothing_seconds)
        self.smoothing_samples = k

        if 1 < k < self.n:
            half = k // 2
            start = 0

            while start < self.n:
                end = min(start + chunk, self.n)
                a = max(0, start - half)
                b = min(self.n, end + half)

                smoothed = medfilt(freq[a:b], kernel_size=k)
                freq[start:end] = smoothed[start - a:end - a].astype(np.float32)

                del smoothed
                start = end

        np.clip(freq, 1000.0, 2500.0, out=freq)

        stride = max(1, amp.size // 2000000)
        q = float(np.percentile(amp[::stride], 75)) + 1e-12

        weight = np.empty(self.n, dtype=np.float32)
        start = 0

        while start < self.n:
            end = min(start + chunk, self.n)
            w = np.clip(amp[start:end] / q, 0.0, 1.0)
            weight[start:end] = (0.05 + 0.95 * w * w).astype(np.float32)
            del w
            start = end

        del amp

        self.freq = freq
        self.weight = weight
        self.freq_weight = (freq * weight).astype(np.float32)

    def _window_sums(self, arr, starts, ends):
        if np.all(ends[:-1] <= starts[1:]):
            lo = int(starts[0])
            hi = min(int(ends[-1]), self.n)

            if hi <= lo:
                return np.zeros(len(starts), dtype=np.float64)

            cs = np.concatenate(
                ([0.0], np.cumsum(arr[lo:hi], dtype=np.float64))
            )

            return cs[ends - lo] - cs[starts - lo]

        return np.array(
            [arr[a:b].sum(dtype=np.float64) for a, b in zip(starts, ends)],
            dtype=np.float64,
        )

    def mean_freq(self, t_start, t_end):
        """
        Weighted mean frequency over [t_start, t_end).
        Accepts scalar or vector time arrays.
        """
        s = np.clip(np.round(np.asarray(t_start) * self.fs).astype(np.int64), 0, self.n)
        e = np.clip(np.round(np.asarray(t_end) * self.fs).astype(np.int64), 0, self.n)

        e = np.maximum(e, np.minimum(s + 1, self.n))

        if s.ndim == 0:
            num = float(self.freq_weight[s:e].sum(dtype=np.float64))
            den = float(self.weight[s:e].sum(dtype=np.float64))
            return num / max(den, 1e-12)

        num = self._window_sums(self.freq_weight, s, e)
        den = self._window_sums(self.weight, s, e)

        return num / np.maximum(den, 1e-12)

    # ---------------------------------------------------------------
    # Sub-sample zero-crossing frequency measurement
    # ---------------------------------------------------------------
    #
    # The Hilbert/unwrapped-phase estimator in FMDemodulator is excellent
    # for noisy signals but its frequency resolution is bounded by the
    # audio sample rate: one diff of the unwrapped phase gives one
    # instantaneous-frequency value per sample, quantised to 1/fs.
    #
    # Zero crossings are different.  A 1900 Hz tone has a zero crossing
    # roughly every 1/(2*1900) ~= 0.26 ms, and at 48 kHz that's 12.6
    # samples between crossings - but linear interpolation between the
    # two bracketing samples pins the crossing to a fraction of a sample
    # (~1/fs / slope), so the period estimate has resolution well below
    # 1/fs.  This is the "sub-sample zero-crossing" technique used in
    # frequency counters and is what lets us measure ppm-level drift
    # on long modes (PD-290, MR320) where the crystal wanders a few
    # Hz over the transmission.
    #
    # We precompute crossing times once per demodulator and offer two
    # entry points:
    #   * zero_crossing_frequency(t_start, t_end) -> scalar Hz
    #   * zero_crossing_frequency_array(t_starts, t_ends) -> vector Hz
    #
    # Both are useful for VIS bit decisions and for sync-pulse refinement
    # where the Hilbert estimator's quantisation would otherwise smear
    # timing across the line.

    def _compute_zero_crossings(self):
        """Find every zero crossing in the band-passed source audio.

        We work on the raw samples (not the FM-demodulated freq) because
        zero crossings only make sense on the actual waveform.  A
        band-limited pre-filter removes high-frequency noise that would
        otherwise create spurious crossings.

        Returns (crossing_times, crossing_signs) where crossing_times is
        a float32 array of crossing times in seconds and crossing_signs
        is +1 / -1 indicating rising/falling.
        """
        from scipy.signal import butter, sosfiltfilt

        nyq = self.fs * 0.5
        lo = max(800.0, min(1000.0, nyq * 0.4))
        hi = min(2600.0, nyq * 0.95)

        if hi <= lo:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int8)

        sos = butter(4, (lo, hi), btype="bandpass", fs=self.fs, output="sos")

        # Process in chunks to keep memory bounded (same pattern as the
        # main FM demodulator).
        chunk = max(4 * 8192, 1 << 18)
        pad = 8192

        samples32 = np.asarray(self.samples, dtype=np.float32)

        all_times = []
        all_signs = []

        start = 0

        while start < self.n:
            end = min(start + chunk, self.n)

            a = max(0, start - pad)
            b = min(self.n, end + pad)

            seg = sosfiltfilt(sos, samples32[a:b].astype(np.float64))

            # Sign changes inside the chunk only (we'll catch the boundary
            # crossings on the next chunk too, but de-dup via clip later).
            rel_a = start - a
            rel_b = end - a

            seg_chunk = seg[rel_a:rel_b + 1] if rel_b + 1 <= len(seg) else seg[rel_a:]

            if len(seg_chunk) < 2:
                start = end
                continue

            # Sign change anywhere seg[k] * seg[k+1] < 0.
            s0 = seg_chunk[:-1]
            s1 = seg_chunk[1:]

            prod = s0 * s1

            crossings_idx = np.where(prod < 0)[0]

            if len(crossings_idx) == 0:
                start = end
                continue

            y0 = s0[crossings_idx]
            y1 = s1[crossings_idx]

            # Linear interpolation: t_cross = k + (-y0) / (y1 - y0)
            # = k + |y0| / (|y0| + |y1|)  (signs opposite)
            denom = (y1 - y0)
            frac = np.where(np.abs(denom) > 1e-12, -y0 / denom, 0.5)
            frac = np.clip(frac, 0.0, 1.0)

            sample_positions = (crossings_idx + frac).astype(np.float64)
            crossing_times = (a + rel_a + sample_positions) / float(self.fs)
            crossing_signs = np.where(y1 > y0, 1, -1).astype(np.int8)

            mask = (crossing_times >= start / float(self.fs)) & (crossing_times < end / float(self.fs))

            all_times.append(crossing_times[mask])
            all_signs.append(crossing_signs[mask])

            start = end

        if not all_times:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int8)

        times = np.concatenate(all_times)
        signs = np.concatenate(all_signs)

        # Sort just in case (chunk boundaries should already be ordered).
        order = np.argsort(times, kind="stable")
        return times[order], signs[order]

    def zero_crossing_frequency(self, t_start, t_end):
        """Sub-sample-resolution frequency over [t_start, t_end).

        Counts rising zero crossings in the window and divides by the
        time span they cover.  Resolution is ~1/(2*N) of the average
        period, where N is the number of crossings in the window; for
        a 1900 Hz tone at 48 kHz sampled for 30 ms, that's ~57 crossings
        and resolution ~0.017 Hz, far below what the Hilbert estimator
        can resolve.

        Returns the Hilbert estimator's value as a fallback when the
        window has fewer than 4 crossings (very short or noisy).
        """
        if not hasattr(self, "_zero_crossings_cache"):
            self._zero_crossings_cache = None

        if self._zero_crossings_cache is None:
            try:
                self._zero_crossings_cache = self._compute_zero_crossings()
            except Exception:
                self._zero_crossings_cache = (np.array([]), np.array([]))

        times, signs = self._zero_crossings_cache

        if len(times) < 4:
            return self.mean_freq(t_start, t_end)

        in_window = (times >= float(t_start)) & (times < float(t_end))

        if in_window.sum() < 4:
            return self.mean_freq(t_start, t_end)

        rising = in_window & (signs > 0)

        if rising.sum() < 2:
            return self.mean_freq(t_start, t_end)

        rising_times = times[rising]

        # Span = last rising crossing - first rising crossing.
        span = float(rising_times[-1] - rising_times[0])

        if span <= 0:
            return self.mean_freq(t_start, t_end)

        n_cycles = len(rising_times) - 1

        return float(n_cycles) / span

    def zero_crossing_frequency_array(self, t_starts, t_ends):
        """Vectorised version of zero_crossing_frequency.

        Accepts array-like t_starts and t_ends of equal length and
        returns a numpy array of frequencies.  Falls back to mean_freq
        for any window that has too few crossings.
        """
        t_starts = np.atleast_1d(np.asarray(t_starts, dtype=np.float64))
        t_ends = np.atleast_1d(np.asarray(t_ends, dtype=np.float64))

        if t_starts.shape != t_ends.shape:
            raise ValueError("t_starts and t_ends must have the same shape")

        out = np.empty(t_starts.shape, dtype=np.float64)

        for i in range(t_starts.size):
            out.flat[i] = self.zero_crossing_frequency(float(t_starts.flat[i]),
                                                       float(t_ends.flat[i]))

        return out

    def sync_score(
            self,
            sync_seconds,
            step_seconds,
            t_from,
            t_to,
            center=SSTV_SYNC_HZ,
            half_width=150.0,
    ):
        """
        Weighted fraction of samples near the sync frequency in each window.
        """
        w = max(int(round(sync_seconds * self.fs)), 1)
        step = max(int(round(step_seconds * self.fs)), 1)

        i0 = max(int(round(t_from * self.fs)), 0)
        i1 = min(int(round(t_to * self.fs)), self.n - w)

        if i1 <= i0:
            return np.array([]), np.array([])

        idx = np.arange(i0, i1, step, dtype=np.int64)

        lo = i0
        hi = min(self.n, i1 + w)

        near = (np.abs(self.freq[lo:hi] - center) < half_width).astype(np.float32)
        weighted = near * self.weight[lo:hi]

        del near

        hit = sliding_window_view(weighted, w)[::step][:len(idx)].sum(axis=1, dtype=np.float64)
        den = sliding_window_view(self.weight[lo:hi], w)[::step][:len(idx)].sum(axis=1, dtype=np.float64)

        del weighted

        return idx / self.fs, hit / np.maximum(den, 1e-12)


class ImageDecoder:
    BLACK_HZ = SSTV_BLACK_HZ
    WHITE_HZ = SSTV_WHITE_HZ

    # When spline slant correction is enabled, the linear-fit result is
    # upgraded to a cubic smoothing spline through the same sync pulses.
    # The spline is what handles non-linear drift caused by temperature
    # changes in the 12 MHz crystal during long modes like PD-290.
    # "linear" keeps the historical a + b*k fit; "spline" adds the
    # per-line curvature; "auto" picks spline only when there are enough
    # pulses and the residuals are noticeably non-linear.
    SLANT_MODES = ("linear", "spline", "auto")

    def __init__(
            self,
            fs,
            samples,
            layout: ImageLayout,
            vis_end,
            freq_offset_hz=0.0,
            slant_search=0.03,
            smoothing_seconds=None,
            slant_mode="auto",
    ):
        self.fs = fs
        self.samples = np.asarray(samples, dtype=np.float32)
        self.layout = layout
        self.vis_end = vis_end
        self.freq_offset_hz = freq_offset_hz
        self.slant_search = slant_search
        if slant_mode not in self.SLANT_MODES:
            # Don't silently fall back to "auto" - that hides typos from
            # users who would expect --slant-mode spline to actually use
            # the spline.  Log a warning and fall back, but make the
            # invalid value visible in the info dict.
            import warnings as _warnings
            _warnings.warn(
                f"Unknown slant_mode {slant_mode!r}; falling back to 'auto'. "
                f"Valid values: {', '.join(self.SLANT_MODES)}.",
                stacklevel=2,
            )
            self.slant_mode = "auto"
            self._slant_mode_requested = str(slant_mode)
        else:
            self.slant_mode = slant_mode
            self._slant_mode_requested = slant_mode
        self.total_duration = len(samples) / fs

        if smoothing_seconds is None:
            smoothing_seconds = smoothing_seconds_for_layout(layout)

        self.smoothing_seconds = float(smoothing_seconds)
        self.demod = FMDemodulator(
            fs,
            samples,
            freq_offset_hz,
            smoothing_seconds=self.smoothing_seconds,
        )
        self.info = {}
        # Spline state, populated by _fit_timing when slant_mode is spline
        # or auto and the linear residual has detectable curvature.
        self._timing_spline = None
        self._timing_spline_kind = None

    def _expected_first_sync(self):
        L = self.layout
        return self.vis_end + L.leadin_seconds + L.sync_offset

    def _detect_sync_pulses(self):
        L = self.layout
        T = L.line_seconds

        expected0 = self._expected_first_sync()
        image_duration = L.n_lines * T
        drift_allow = self.slant_search * image_duration

        t_from = max(expected0 - 1.25 * T, 0.0)
        t_to = min(
            expected0 + image_duration + drift_allow + 2 * T,
            self.total_duration,
        )

        step_seconds = min(L.sync_seconds / 8.0, 0.002) if L.sync_seconds > 0 else 0.002

        times, score = self.demod.sync_score(
            L.sync_seconds,
            step_seconds,
            t_from,
            t_to,
            center=SSTV_SYNC_HZ + getattr(self, "freq_offset_hz", 0.0),
            half_width=160.0,
        )

        if len(times) == 0:
            return np.array([])

        smooth_n = max(1, int(round((max(L.sync_seconds, 0.001) / 4.0) / step_seconds)))

        if smooth_n > 1:
            kernel = np.ones(smooth_n) / smooth_n
            score_smooth = np.convolve(score, kernel, mode="same")
        else:
            score_smooth = score

        med = np.median(score_smooth)
        mad = np.median(np.abs(score_smooth - med)) + 1e-12

        min_dist = max(int(round(0.45 * T / step_seconds)), 1)

        attempts = [
            (max(0.25, med + 4.0 * mad), max(0.08, 3.0 * mad)),
            (max(0.15, med + 3.0 * mad), max(0.04, 2.0 * mad)),
            (max(0.08, med + 2.0 * mad), max(0.02, 1.5 * mad)),
        ]

        best_peaks = np.array([], dtype=int)

        for height, prominence in attempts:
            peaks, _ = find_peaks(
                score_smooth,
                height=height,
                prominence=prominence,
                distance=min_dist,
            )

            best_peaks = peaks

            if len(peaks) >= max(4, int(0.1 * L.n_lines)):
                break

        return self._refine_pulse_edges(times[best_peaks])

    def _refine_pulse_edges(self, pulses):
        L = self.layout
        freq = self.demod.freq
        fs = self.demod.fs
        sync = L.sync_seconds

        if len(pulses) == 0 or sync <= 0 or freq is None:
            return pulses

        smooth = max(1, int(round(0.00008 * fs)))
        kernel = np.ones(smooth) / float(smooth) if smooth > 1 else None

        refined = []

        for t in np.asarray(pulses, dtype=np.float64):
            a = int(max(0, round((t - 1.3 * sync) * fs)))
            b = int(min(len(freq), round((t + 1.6 * sync) * fs)))

            if b - a < 16:
                refined.append(float(t))
                continue

            seg = freq[a:b]

            if kernel is not None:
                seg = np.convolve(seg, kernel, mode="same")

            idx = int(np.clip(int(round(t * fs)) - a, 1, len(seg) - 2))

            ins_a = min(len(seg) - 4, idx + int(round(0.20 * sync * fs)))
            ins_b = min(len(seg), idx + int(round(0.70 * sync * fs)))

            if ins_b - ins_a < 4:
                refined.append(float(t))
                continue

            level_sync = float(np.median(seg[ins_a:ins_b]))
            thr = 0.5 * (level_sync + SSTV_BLACK_HZ)

            j = min(len(seg) - 2, idx + int(round(0.30 * sync * fs)))
            limit = min(len(seg) - 2, idx + int(round(1.45 * sync * fs)))

            while j < limit and seg[j] < thr:
                j += 1

            if j >= limit or j < 1:
                refined.append(float(t))
                continue

            y0 = float(seg[j - 1])
            y1 = float(seg[j])
            frac = (thr - y0) / (y1 - y0) if y1 != y0 else 0.0
            frac = float(np.clip(frac, 0.0, 1.0))

            t_end = (a + j - 1 + frac) / float(fs)
            t_edge = t_end - sync

            if abs(t_edge - t) > 1.0 * sync:
                refined.append(float(t))
            else:
                refined.append(t_edge)

        return np.asarray(refined, dtype=np.float64)

    def _fit_timing(self, pulses):
        """
        Fit sync_time(line k) = a + b*k.

        When slant_mode is "spline" or "auto", the linear fit is upgraded
        in place to a cubic smoothing spline through the same pulses, so
        the per-line sync times used by _read_planes() reflect non-linear
        drift instead of a single straight line.
        """
        L = self.layout
        T = L.line_seconds
        expected0 = self._expected_first_sync()

        if len(pulses) < 4:
            return expected0, T, 0

        pulses = np.asarray(pulses, dtype=np.float64)

        cands = T * (
                1.0 + np.linspace(-self.slant_search, self.slant_search, 4001)
        )

        best = None

        for Tp in cands:
            rel = pulses - expected0

            z = np.exp(2j * np.pi * rel / Tp)
            phase_offset = np.angle(z.mean()) / (2 * np.pi) * Tp
            a0 = expected0 + phase_offset

            k = np.round((pulses - a0) / Tp).astype(int)
            valid = (k >= 0) & (k < L.n_lines)

            resid = pulses - (a0 + Tp * k)

            tol = max(1.5 * L.sync_seconds, 0.012)
            inl = valid & (np.abs(resid) < tol)

            if inl.sum() < 4 or len(np.unique(k[inl])) < 2:
                continue

            med_abs = float(np.median(np.abs(resid[inl])))

            score = (
                int(inl.sum()),
                -med_abs,
                -abs(Tp / T - 1.0),
            )

            if best is None or score > best[0]:
                best = (score, a0, Tp)

        if best is None:
            return expected0, T, 0

        _, a, b = best

        n_used = 0
        tol = max(1.25 * L.sync_seconds, 0.010)

        for _ in range(5):
            k = np.round((pulses - a) / b).astype(int)
            valid = (k >= 0) & (k < L.n_lines)

            resid = pulses - (a + b * k)
            inl = valid & (np.abs(resid) < tol)

            if inl.sum() < 4 or len(np.unique(k[inl])) < 2:
                break

            slope, intercept = np.polyfit(k[inl], pulses[inl], 1)

            if abs(slope / T - 1.0) > self.slant_search * 1.5:
                break

            a = float(intercept)
            b = float(slope)
            n_used = int(inl.sum())

        # Try to upgrade the linear fit to a cubic smoothing spline when
        # asked or when there's measurable curvature in the residuals.
        self._maybe_fit_spline(pulses, a, b, n_used)

        return a, b, n_used

    def _maybe_fit_spline(self, pulses, a, b, n_used):
        """Try to fit a cubic smoothing spline to the sync pulses.

        Populates self._timing_spline with a callable sync_time(k) when
        the spline fit is meaningfully better than the linear fit, or
        when the user explicitly asked for slant_mode="spline".
        """
        self._timing_spline = None
        self._timing_spline_kind = None

        if n_used < 8 or b <= 0:
            # Not enough points for a stable spline; keep the linear fit.
            return

        L = self.layout
        T = L.line_seconds

        pulses = np.asarray(pulses, dtype=np.float64)
        k_lin = np.round((pulses - a) / b).astype(int)
        valid = (k_lin >= 0) & (k_lin < L.n_lines)

        tol = max(1.25 * L.sync_seconds, 0.010)
        resid = pulses - (a + b * k_lin)
        inl = valid & (np.abs(resid) < tol)

        if inl.sum() < 8:
            return

        k_used = k_lin[inl]
        t_used = pulses[inl]

        # Deduplicate (the sync detector occasionally reports two peaks for
        # the same line); keep the median time per line.
        uniq_k = np.unique(k_used)

        if len(uniq_k) < 8:
            return

        uniq_t = np.array([np.median(t_used[k_used == kk]) for kk in uniq_k])

        # Spline needs monotonically increasing x.
        order = np.argsort(uniq_k)
        uniq_k = uniq_k[order]
        uniq_t = uniq_t[order]

        linear_residual = uniq_t - (a + b * uniq_k)
        max_lin_resid = float(np.max(np.abs(linear_residual))) if len(linear_residual) else 0.0

        # Decide whether to bother.  If the linear fit already explains the
        # pulses to within a fraction of a sync pulse, a spline can only
        # overfit.  We require either an explicit user request
        # (slant_mode == "spline") or residuals that exceed ~1/3 of the
        # sync pulse width - that's the threshold where uncorrected drift
        # starts visibly smearing pixels in long modes.
        trigger = (
            self.slant_mode == "spline"
            or (self.slant_mode == "auto"
                and max_lin_resid > 0.35 * max(L.sync_seconds, 0.001))
        )

        if not trigger:
            return

        try:
            from scipy.interpolate import UnivariateSpline
        except ImportError:
            return

        try:
            # Smoothing factor scales with the number of pulses so the
            # spline doesn't chase noise on short modes.  s = N * sigma^2
            # where sigma is roughly the sync-edge jitter (~0.2 ms).
            sigma = max(0.0002, 0.05 * L.sync_seconds)
            s = len(uniq_k) * (sigma ** 2)

            spline = UnivariateSpline(
                uniq_k.astype(np.float64),
                uniq_t,
                k=min(3, max(1, len(uniq_k) - 1)),
                s=s,
            )

            # Sanity: the spline's value at k=0 should be within ~1 sync
            # of the linear a; otherwise we've fit garbage.
            t0_spline = float(spline(0.0))

            if not np.isfinite(t0_spline) or abs(t0_spline - a) > 2.0 * max(L.sync_seconds, 0.001):
                return

            # Verify the spline actually reduces residual on the training
            # set; if it doesn't, keep the linear fit.
            spline_resid = uniq_t - spline(uniq_k)
            spline_max = float(np.max(np.abs(spline_resid)))

            if self.slant_mode != "spline" and spline_max >= max_lin_resid * 0.95:
                return

            self._timing_spline = spline
            self._timing_spline_kind = "univariate"
        except Exception:
            # Spline fit failed for some reason; silently fall back.
            self._timing_spline = None
            self._timing_spline_kind = None

    def _sync_time_for_line(self, line_index):
        """Return the actual sync time for a given line.

        Uses the spline when one was fit, otherwise the linear fit.
        """
        if self._timing_spline is not None:
            try:
                t = float(self._timing_spline(float(line_index)))
                if np.isfinite(t):
                    return t
            except Exception:
                pass

        # Fallback: caller has a and b in scope, but here we don't.
        # _read_planes passes a and b explicitly, so this branch is only
        # used when called from outside _read_planes.
        return None

    def _line_rate_for_line(self, line_index, a, b):
        """Estimate the per-line line rate at line_index.

        For the spline case, this is the spline's local derivative.  For
        the linear case it's just b.
        """
        if self._timing_spline is not None:
            try:
                # Numerical derivative of the spline over a small window.
                h = 1.0
                t0 = float(self._timing_spline(float(line_index)))
                t1 = float(self._timing_spline(float(line_index + h)))

                if np.isfinite(t0) and np.isfinite(t1):
                    return (t1 - t0) / h
            except Exception:
                pass

        return b

    def _freq_to_lum(self, f):
        return np.clip(
            (f - self.BLACK_HZ) / (self.WHITE_HZ - self.BLACK_HZ) * 255.0,
            0.0,
            255.0,
        )

    def _tally_update(self, tally, key, freq):
        """
        Accumulate the per-plane frequency stats behind the level report.
        """
        item = tally.get(key)

        if item is None:
            item = tally[key] = {
                "n": 0,
                "below": 0,
                "above": 0,
                "total": 0.0,
                "squares": 0.0,
                "lo": math.inf,
                "hi": -math.inf,
            }

        item["n"] += int(freq.size)
        item["below"] += int((freq < self.BLACK_HZ).sum())
        item["above"] += int((freq > self.WHITE_HZ).sum())
        item["total"] += float(freq.sum())
        item["squares"] += float((freq.astype(np.float64) ** 2).sum())
        item["lo"] = min(item["lo"], float(freq.min()))
        item["hi"] = max(item["hi"], float(freq.max()))

    def _read_planes(self, a, b, lines=None):
        L = self.layout
        r = b / L.line_seconds
        W = L.width
        n_lines = int(L.n_lines if lines is None else min(L.n_lines, max(0, lines)))

        if L.color == "rgb3":
            keys = set(L.line_channel_order or ("R", "G", "B"))
        else:
            keys = {k for k, _, _ in L.channels}

            if L.channels_odd is not None:
                keys |= {k for k, _, _ in L.channels_odd}

        planes = {
            k: np.zeros((L.n_lines, W), dtype=np.float64)
            for k in keys
        }

        x = np.arange(W)
        tally = {}

        # When spline slant correction is active, sync_time and the per-line
        # rate vary across the image.  We compute both per line so pixels are
        # read at the right audio time even when the local crystal has
        # drifted several ppm from start to end of the transmission.
        use_spline = self._timing_spline is not None

        if L.color == "rgb3":
            order = L.line_channel_order or ("R", "G", "B")
            n_components = len(order)

            _component, off, dur = L.channels[0]

            for tx_line in range(n_lines):
                row = tx_line // n_components

                if row >= L.height:
                    break

                key = order[tx_line % n_components]

                if use_spline:
                    sync_time = float(self._timing_spline(float(tx_line)))
                    r_line = self._line_rate_for_line(tx_line, a, b) / L.line_seconds
                else:
                    sync_time = a + b * tx_line
                    r_line = r

                line_start = sync_time - L.sync_offset * r_line

                px = dur / W * r_line
                t0 = line_start + off * r_line + x * px

                freq = self.demod.mean_freq(t0, t0 + px)

                planes[key][row] = self._freq_to_lum(freq)

                self._tally_update(tally, key, freq)

            self.level_tally = tally

            return planes

        for line in range(n_lines):
            if use_spline:
                sync_time = float(self._timing_spline(float(line)))
                r_line = self._line_rate_for_line(line, a, b) / L.line_seconds
            else:
                sync_time = a + b * line
                r_line = r

            line_start = sync_time - L.sync_offset * r_line

            chans = L.channels

            if L.channels_odd is not None and line % 2 == 1:
                chans = L.channels_odd

            for key, off, dur in chans:
                px = dur / W * r_line
                t0 = line_start + off * r_line + x * px

                freq = self.demod.mean_freq(t0, t0 + px)

                planes[key][line] = self._freq_to_lum(freq)

                self._tally_update(tally, key, freq)

        self.level_tally = tally

        return planes

    @staticmethod
    def _ycrcb_to_rgb(Y, Cr, Cb):
        """
        SSTV Y, R-Y, B-Y back to RGB.

        The inverse of the Appendix B transform: Cr is R-Y and Cb is B-Y,
        both centred on 128, and Y is studio-range with black at 16.
        """
        y = np.asarray(Y, dtype=np.float64) - 16.0
        ry = np.asarray(Cr, dtype=np.float64) - 128.0
        by = np.asarray(Cb, dtype=np.float64) - 128.0

        R = (298.082 * y + 408.583 * ry) / 256.0
        G = (298.082 * y - 208.120 * ry - 100.291 * by) / 256.0
        B = (298.082 * y + 516.411 * by) / 256.0

        return np.dstack([R, G, B])

    # Time span (seconds) at each end of a chroma scan that is corrupted by
    # demodulator transients from the adjacent separator / sync tones.
    CHROMA_EDGE_SECONDS = 0.004

    @classmethod
    def _repair_chroma_edges(cls, plane, px_seconds):
        """Overwrite transient-corrupted edge pixels with the nearest clean value."""
        plane = np.array(plane, dtype=np.float64, copy=True)
        W = plane.shape[1]

        if px_seconds <= 0 or W < 16:
            return plane

        k = int(math.ceil(cls.CHROMA_EDGE_SECONDS / px_seconds))
        k = max(2, min(k, W // 8))
        ref = 4

        left = np.median(plane[:, k:k + ref], axis=1)
        right = np.median(plane[:, W - k - ref:W - k], axis=1)

        plane[:, :k] = left[:, None]
        plane[:, W - k:] = right[:, None]

        return plane

    @staticmethod
    def _repair_luma_edges(plane):
        """The outermost luma pixels blend with the porch/separator tone."""
        plane = np.array(plane, dtype=np.float64, copy=True)

        if plane.shape[1] >= 8:
            plane[:, 0] = plane[:, 1]
            plane[:, -1] = plane[:, -2]

        return plane

    def _assemble(self, planes):
        L = self.layout

        if L.color in ("rgb", "rgb3"):
            rows = L.height if L.color == "rgb3" else None

            rgb = np.dstack([
                planes["R"][:rows],
                planes["G"][:rows],
                planes["B"][:rows],
            ])

        elif L.color == "bw":
            # Robot B&W modes: a single Y plane becomes a grayscale RGB
            # image (R=G=B=Y).
            Y = planes["Y"]
            rgb = np.dstack([Y, Y, Y])

        elif L.color == "ycrcb":
            rgb = self._ycrcb_to_rgb(
                planes["Y"],
                planes["Cr"],
                planes["Cb"],
            )

        elif L.color == "ycrcb420":
            n = L.n_lines

            # SSTV 4:2:0: even scan lines carry Y+Cr, odd scan lines carry Y+Cb.
            # Row 0 is always inside the sync pulse window (near-zero after
            # freq_to_lum), so the first real Cr sample lands at scan row 2
            # and the first real Cb sample at scan row 1.
            #
            # Each chroma sample covers a PAIR of image rows:
            #   Cr from scan row 2 covers image rows 2 and 3
            #   Cr from scan row 4 covers image rows 4 and 5  …
            #   Cb from scan row 1 covers image rows 0,1 and 2,3? No—
            #   Cb from scan row 1 covers image rows 0 and 1
            #   Cb from scan row 3 covers image rows 2 and 3  …
            #
            # Old code: planes["Cr"][0::2] picks row 0 (zero/garbage) as the
            # first Cr sample, shifting every Cr value two rows late.
            # Old code: the extra vstack prepend on Cb gave three Cb_data[0]
            # rows at the top, shifting Cb one pair early for all even rows.
            #
            # Correct assembly:
            #   Cr_data = planes["Cr"][2::2]   (127 rows: real scans at 2,4,…)
            #   Cb_data = planes["Cb"][1::2]   (128 rows: real scans at 1,3,…)
            #
            #   Cr_full: rows 0,1 padded with Cr_data[0] (nearest available),
            #            then Cr_data repeated ×2 → covers rows 2…255
            #   Cb_full: Cb_data repeated ×2 → rows 0,1 = cb1, rows 2,3 = cb3, …
            #            (Cb_data[0] naturally covers image rows 0 and 1)

            # Edge repair: the FM demodulator smears the neighbouring
            # separator (1500 Hz) / next-line sync (1200 Hz) tones into the
            # first and last few chroma pixels, and rings for a few more.
            # Both chroma planes then dip toward 0, which renders as a green
            # stripe (magenta ringing just inside it) down each side.  Replace
            # the contaminated edge pixels with the nearest clean chroma value.
            planes = dict(planes)
            _scans = list(L.channels) + list(L.channels_odd or ())
            for _key, _off, _dur in _scans:
                if _key in ("Cr", "Cb"):
                    planes[_key] = self._repair_chroma_edges(
                        planes[_key], _dur / L.width)
            planes["Y"] = self._repair_luma_edges(planes["Y"])

            Cr_data = planes["Cr"][2::2]   # (n//2 - 1, W): scans at rows 2,4,…
            Cb_data = planes["Cb"][1::2]   # (n//2,     W): scans at rows 1,3,…

            Cr = np.vstack([
                Cr_data[[0]],                       # pad rows 0–1 with nearest Cr
                Cr_data[[0]],
                np.repeat(Cr_data, 2, axis=0),      # rows 2–(2*len-1)
            ])[:n]

            # No extra prepend for Cb: np.repeat already gives
            # [cb1, cb1, cb3, cb3, cb5, cb5, …] which correctly maps
            # cb1 → rows 0,1 and cb3 → rows 2,3 etc.
            Cb = np.repeat(Cb_data, 2, axis=0)[:n]

            rgb = self._ycrcb_to_rgb(
                planes["Y"],
                Cr,
                Cb,
            )

        elif L.color == "pd":
            n = L.n_lines

            Y = np.empty((2 * n, L.width), dtype=np.float64)
            Y[0::2] = planes["Y0"]
            Y[1::2] = planes["Y1"]

            Cr = np.repeat(planes["Cr"], 2, axis=0)
            Cb = np.repeat(planes["Cb"], 2, axis=0)

            rgb = self._ycrcb_to_rgb(Y, Cr, Cb)

        else:
            raise ValueError(f"Unknown color model {L.color}")

        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _lines_in_audio(self, a, b):
        L = self.layout

        if b <= 0:
            return 0

        available = self.total_duration - a

        return int(min(L.n_lines, max(0, math.floor(available / b))))

    def decode(self):
        """
        Decode the image. Audio that ends early still decodes: the lines that
        arrived are kept, the rest are black, and info carries
        lines_decoded / lines_expected / partial.
        """
        pulses = self._detect_sync_pulses()
        # Cache the raw pulses so the diagnostics builder can plot them.
        self._last_pulses = pulses
        a, b, n_used = self._fit_timing(pulses)

        L = self.layout
        lines = self._lines_in_audio(a, b)

        self.info = {
            "sync_pulses_detected": int(len(pulses)),
            "sync_pulses_used_in_fit": int(n_used),
            "sync_lock_ok": False,
            "lines_decoded": int(lines),
            "lines_expected": int(L.n_lines),
            "partial": bool(lines < L.n_lines),
            "nominal_line_seconds": float(L.line_seconds),
            "measured_line_seconds": float(b),
            "line_structure": "rgb3" if L.color == "rgb3" else "standard",
            "slant_ratio": float(b / L.line_seconds),
            "first_line_sync": float(a),
            "image_size": (L.width, L.height),
            "slant_mode_requested": self.slant_mode,
            "slant_correction": (
                "spline" if self._timing_spline is not None else "linear"
            ),
            "levels": {},
        }

        if lines < 2:
            self.info["error"] = (
                    "audio ends %d line(s) into the image, %d expected; "
                    "transmission truncated" % (lines, L.n_lines)
            )
            return None

        planes = self._read_planes(a, b, lines)

        scale = 255.0 / (self.WHITE_HZ - self.BLACK_HZ)

        self.info["levels"] = {
            key: {
                "mean_hz": round(item["total"] / max(item["n"], 1), 1),
                "min_hz": round(item["lo"], 1),
                "max_hz": round(item["hi"], 1),
                "mean_level": round(
                    (item["total"] / max(item["n"], 1) - self.BLACK_HZ) * scale, 1
                ),
                "std_level": round(
                    math.sqrt(
                        max(
                            item["squares"] / max(item["n"], 1)
                            - (item["total"] / max(item["n"], 1)) ** 2,
                            0.0,
                        )
                    ) * scale,
                    2,
                ),
                "below_black_fraction": round(item["below"] / max(item["n"], 1), 4),
                "above_white_fraction": round(item["above"] / max(item["n"], 1), 4),
            }
            for key, item in sorted(self.level_tally.items())
        }

        rgb = self._assemble(planes)

        lock_ok = n_used >= max(4, int(0.15 * min(lines, L.n_lines)))
        self.info["sync_lock_ok"] = bool(lock_ok)

        return rgb


class SSTVDecoder:
    FREQ_LEADER = SSTV_LEADER_HZ

    LEADER_OFFSETS_HZ = (
        0,
        -25, 25,
        -50, 50,
        -75, 75,
        -100, 100,
        -125, 125,
        -150, 150,
    )

    def __init__(
            self,
            audio_path,
            registry: ModeRegistry = None,
            protocol: ProtocolConstants = None,
    ):
        self.audio_path = audio_path
        self.fs, samples = load_audio_mono(audio_path)
        self.samples = np.asarray(samples, dtype=np.float32)

        self.registry = registry or ModeRegistry()
        self.protocol = protocol or ProtocolConstants()

        self.detector = ToneDetector(self.fs, self.samples)
        self.vis_reader = VISHeaderReader(self.detector, self.protocol)

        self.total_duration = len(self.samples) / self.fs

        self.leader_start = None
        self.leader_end = None
        self.vis_start = None
        self.vis_end = None
        self.mode = None
        self.freq_offset_hz = 0.0
        self.leader_check = None
        self.vis_value = None

    def force_mode(self, mode_name, vis_end=None, first_sync_time=None, freq_offset_hz=0.0):
        """
        Decode image data without VIS auto-detection.

        Useful for custom/no-VIS audio.

        If you know first line sync, pass first_sync_time.
        Otherwise pass vis_end, the time before layout.leadin_seconds.
        """
        self.registry._require_mode_name(mode_name)
        self.registry._require_layout_name(mode_name)

        layout = self.registry.get_layout(mode_name)

        if vis_end is None:
            if first_sync_time is not None:
                vis_end = float(first_sync_time) - layout.leadin_seconds - layout.sync_offset
            else:
                vis_end = 0.0

        self.mode = self.registry.get_mode(mode_name)
        self.vis_end = float(vis_end)
        self.freq_offset_hz = float(freq_offset_hz)
        self.vis_value = self.registry.find_vis_code_for_mode(mode_name)

        return {
            "forced_mode": True,
            "mode_name": mode_name,
            "experimental": is_experimental_mode(mode_name),
            "notice": experimental_notice(mode_name),
            "vis_end": self.vis_end,
            "first_sync_time": first_sync_time,
            "freq_offset_hz": self.freq_offset_hz,
            "vis_value": self.vis_value,
        }

    def _coarse_leader_candidates(self):
        p = self.protocol

        coarse_window_seconds = p.leader_tone_seconds / 10.0
        coarse_step_seconds = coarse_window_seconds / 2.0

        max_gap = p.break_tone_seconds + coarse_window_seconds

        candidates = []

        for offset in self.LEADER_OFFSETS_HZ:
            results = self.detector.scan_for_tone(
                self.FREQ_LEADER + offset,
                window_seconds=coarse_window_seconds,
                step_seconds=coarse_step_seconds,
                scan_start_seconds=0,
                scan_duration_seconds=self.total_duration,
            )

            if not results:
                continue

            threshold = self.detector.auto_threshold(results)
            noise_floor = threshold / 5.0

            regions = self.detector.find_regions(
                results,
                threshold,
                min_duration_seconds=p.full_leader_and_break_seconds,
                tolerance=0.7,
                max_gap_seconds=max_gap,
            )

            for rs, re in regions:
                region_powers = np.array([
                    pw for t, pw in results
                    if rs <= t <= re
                ])

                if len(region_powers) == 0:
                    continue

                score = np.median(region_powers) / max(noise_floor, 1e-12)

                candidates.append({
                    "offset": float(offset),
                    "threshold": float(threshold),
                    "region_start": float(rs),
                    "region_end": float(re),
                    "score": float(score),
                })

        candidates.sort(key=lambda c: (c["region_start"], -c["score"]))
        return candidates

    def _fine_leader_from_candidate(self, cand):
        p = self.protocol

        fine_window_seconds = p.vis_bit_seconds
        fine_step_seconds = fine_window_seconds / 15.0

        zoom_start = max(cand["region_start"] - 0.15, 0.0)

        zoom_end = min(
            cand["region_end"] + p.vis_header_seconds + 0.25,
            self.total_duration,
        )

        fine_results = self.detector.scan_for_tone(
            self.FREQ_LEADER + cand["offset"],
            window_seconds=fine_window_seconds,
            step_seconds=fine_step_seconds,
            scan_start_seconds=zoom_start,
            scan_duration_seconds=zoom_end - zoom_start,
        )

        if not fine_results:
            return None

        min_consecutive = max(round(fine_window_seconds / fine_step_seconds), 1)

        run_start, run_end = self.detector.find_tone_edges(
            fine_results,
            cand["threshold"],
            min_consecutive,
        )

        if run_start is None or run_end is None:
            return None

        max_bridge_seconds = p.break_tone_seconds + 2.0 * fine_window_seconds
        single_leader_seconds = p.full_leader_and_break_seconds * 0.9

        times_all = [t for t, _ in fine_results]

        while (run_end - run_start) < single_leader_seconds:
            cut = 0
            while cut < len(times_all) and times_all[cut] < run_end:
                cut += 1
            tail = fine_results[cut:]
            if not tail:
                break
            nxt_start, nxt_end = self.detector.find_tone_edges(
                tail,
                cand["threshold"],
                min_consecutive,
            )
            if nxt_start is None or (nxt_start - run_end) > max_bridge_seconds:
                break
            run_end = nxt_end if nxt_end is not None else tail[-1][0]

        leader_start, leader_end = self.detector.refine_edges(
            fine_results,
            run_start,
            run_end,
            fine_window_seconds,
        )

        measured = leader_end - leader_start

        return {
            "leader_start": float(leader_start),
            "leader_end": float(leader_end),
            "leader_duration": float(measured),
        }

    def _probe_sync_grid(self, layout, vis_start, freq_offset_hz):
        """
        Probe the first few line syncs a layout predicts.

        Returns how many of them actually carry a 1200 Hz pulse, which is
        what tells a real line rate from one that merely divides into it.
        """
        p = self.protocol

        # If this layout's mode uses the 16-bit extended VIS, the header
        # is twice as long (8 bits primary + 8 bits extended), so the
        # first sync pulse arrives one full vis_header_seconds later.
        ext_code = self.registry.extended_vis_code_for_mode(layout.name)
        header_blocks = 2 if ext_code is not None else 1
        vis_end = vis_start + header_blocks * p.vis_header_seconds
        first_sync = vis_end + layout.leadin_seconds + layout.sync_offset

        if first_sync >= self.total_duration:
            return {
                "available": False,
                "ok": True,
                "score": 0.0,
                "reason": "first expected sync is outside audio",
            }

        probe_lines = min(8, layout.n_lines)

        sync_dur = max(layout.sync_seconds * 0.8, min(layout.sync_seconds, 0.003))
        search_radius = max(0.030, layout.sync_seconds * 3.0)
        step = min(max(layout.sync_seconds / 4.0, 0.001), 0.004)

        ratios = []
        offsets = []

        for k in range(probe_lines):
            nominal = first_sync + k * layout.line_seconds

            if nominal + sync_dur >= self.total_duration:
                break

            times = np.arange(
                nominal - search_radius,
                nominal + search_radius + step,
                step,
            )

            best_ratio = 0.0
            best_dt = 0.0

            for t in times:
                if t < 0 or t + sync_dur >= self.total_duration:
                    continue

                p_sync = self.detector.tone_power(1200 + freq_offset_hz, t, sync_dur)

                others = [
                    self.detector.tone_power(f + freq_offset_hz, t, sync_dur)
                    for f in (1500, 1900, 2300)
                ]

                ratio = p_sync / max(max(others), 1e-12)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_dt = t - nominal

            ratios.append(best_ratio)
            offsets.append(best_dt)

        if len(ratios) < 3:
            return {
                "available": True,
                "ok": True,
                "score": 0.0,
                "reason": "too few sync probes",
                "ratios": ratios,
                "offsets": offsets,
            }

        ratios = np.asarray(ratios, dtype=np.float64)
        offsets = np.asarray(offsets, dtype=np.float64)

        hits = int(np.sum(ratios > 2.0))

        needed = max(3, math.ceil(len(ratios) * 0.75))

        median_ratio = float(np.median(ratios))
        median_abs_offset_ms = float(np.median(np.abs(offsets)) * 1000.0)

        ok = hits >= needed

        score = math.log(max(median_ratio, 1e-12)) - 0.02 * median_abs_offset_ms

        return {
            "available": True,
            "ok": bool(ok),
            "score": float(score),
            "hits": hits,
            "needed": needed,
            "median_ratio": median_ratio,
            "median_abs_offset_ms": median_abs_offset_ms,
            "ratios": ratios.tolist(),
            "offsets_ms": (offsets * 1000.0).tolist(),
        }

    def _mode_sync_sanity(self, mode_name, vis_start, freq_offset_hz):
        """
        Check that the decoded mode's line rate really is in the audio.

        Run after a VIS code reads cleanly, this asks whether the first few
        line-sync pulses the decoded mode predicts are actually there. A weak
        all-zero VIS passes parity and looks like Robot 12 Color, but Robot 12
        expects a sync every 100 ms, so on a Scottie DX recording, whose syncs
        are 428 ms apart, most of the probes come back empty and the read is
        rejected.

        Wraase SC-2 can be transmitted two ways, so both are tried when the
        mode has a component-per-line variant: one sync per image row, or one
        per colour component at a third of the line time. Whichever grid the
        pulses land on is reported as the line structure to decode with.

        Results are memoised per mode per 5 ms of start time.
        """
        cache = getattr(self, "_sync_sanity_cache", None)

        if cache is None:
            cache = self._sync_sanity_cache = {}

        cache_key = (
            mode_name,
            float(freq_offset_hz),
            int(round(vis_start * 200.0)),
        )

        if cache_key in cache:
            return cache[cache_key]

        layout = self.registry.image_layouts.get(mode_name)

        if layout is None or layout.sync_seconds <= 0:
            result = {
                "available": False,
                "ok": True,
                "score": 0.0,
                "reason": "no layout or no sync",
                "line_structure": "standard",
            }

            cache[cache_key] = result

            return result

        result = dict(self._probe_sync_grid(layout, vis_start, freq_offset_hz))
        result["line_structure"] = "standard"

        if not result.get("ok") and mode_name in self.registry.line_structure_variants:
            variant = self.registry.get_layout(mode_name, line_structure="rgb3")
            alternative = self._probe_sync_grid(variant, vis_start, freq_offset_hz)

            if alternative.get("ok"):
                result = dict(alternative)
                result["line_structure"] = "rgb3"

        if result.get("ok"):
            self._sync_sanity_structure = result["line_structure"]

        cache[cache_key] = result

        return result

    def _try_decode_vis_near(self, leader_end, freq_offset_hz):
        """
        Search around the leader for a VIS code that can be trusted.

        Four gates, cheapest first. Start and stop bits must clearly be
        1200 Hz. Each data bit must favour 1100 or 1300 by a real margin and
        stay clear of the non-data tones. A VIS code of 0, which is Robot 12
        Color, has to clear a higher bar because seven zeros plus a zero
        parity bit is what any weak or mistimed read degenerates into. Finally
        the mode's own line rate has to be present in the audio.

        The sync-probe result is folded into the score, so between two codes
        that both read cleanly, the one whose syncs line up wins.

        When the primary VIS code reads 0x23 (35) - the marker for a
        16-bit extended VIS - we also read 8 more bits starting right after
        the stop bit, decode that as a second 7-bit + parity byte, and look
        it up in the registry's extended_vis_codes table to pick the
        specific MMSSTV mode.
        """
        best = None
        unknown = getattr(self, "_unknown_vis_codes", None)

        if unknown is None:
            unknown = self._unknown_vis_codes = {}

        shifts = np.arange(-0.035, 0.0351, 0.001)

        for shift in shifts:
            vis_start = leader_end + float(shift)

            metrics = self.vis_reader.read_full_vis_metrics(
                vis_start,
                freq_offset_hz=freq_offset_hz,
            )

            if not metrics.get("complete", False):
                continue

            bits = metrics["bits"]
            records = metrics["records"]
            vis_value = metrics["vis_value"]
            parity_ok = metrics["parity_ok"]

            mode_name = self.registry.lookup_vis(vis_value) if vis_value is not None else None

            # If this primary VIS is the extended-VIS marker, try to read
            # the 8-bit extended byte that follows. The extended byte
            # disambiguates between MMSSTV MP73 / MR90 / etc., which all
            # share primary VIS 35.
            extended_vis_value = None
            extended_parity_ok = None
            resolved_mode_name = mode_name

            if vis_value == EXTENDED_VIS_PRIMARY and parity_ok:
                # The standard 8-bit VIS frame is 10 bit-slots:
                # index 0 = start bit, 1-7 = data, 8 = parity, 9 = stop.
                # The extended byte is a second 10-slot frame that begins
                # right after the primary stop bit.  Its own start bit lives
                # at vis_start + 10*bit_seconds; read_full_vis_metrics reads
                # 10 bits starting from the supplied start time, so we pass
                # that offset.
                ext_metrics = self.vis_reader.read_full_vis_metrics(
                    vis_start + 10 * self.protocol.vis_bit_seconds,
                    freq_offset_hz=freq_offset_hz,
                )

                if ext_metrics.get("complete", False):
                    extended_vis_value = ext_metrics["vis_value"]
                    extended_parity_ok = ext_metrics["parity_ok"]

                    if extended_vis_value is not None and extended_parity_ok:
                        ext_mode = self.registry.find_extended_vis_mode(extended_vis_value)

                        if ext_mode is not None:
                            resolved_mode_name = ext_mode
                            mode_name = ext_mode

            if not parity_ok or mode_name is None:
                if parity_ok and vis_value is not None:
                    unknown[vis_value] = unknown.get(vis_value, 0) + 1
                continue

            start_rec = records[0]
            stop_rec = records[9]

            start_ratio = start_rec["p1200"] / max(
                start_rec["p1100"],
                start_rec["p1300"],
                start_rec["p1900"],
                1e-12,
            )

            stop_ratio = stop_rec["p1200"] / max(
                stop_rec["p1100"],
                stop_rec["p1300"],
                stop_rec["p1900"],
                1e-12,
            )

            if start_ratio < 1.8 or stop_ratio < 1.8:
                continue

            data_margins = []
            data_purities = []

            for rec in records[1:9]:
                p_zero = rec["p1300"]
                p_one = rec["p1100"]

                chosen = max(p_zero, p_one)
                other_data = min(p_zero, p_one)

                data_margins.append(chosen / max(other_data, 1e-12))
                data_purities.append(chosen / max(rec["p1200"], rec["p1900"], 1e-12))

            median_margin = float(np.median(data_margins))
            median_purity = float(np.median(data_purities))

            bit_score = float(
                np.mean(np.log(np.maximum(data_margins, 1.0)))
                + 0.5 * np.mean(np.log(np.maximum(data_purities, 1.0)))
            )

            if median_margin < 1.45 or median_purity < 1.15:
                continue

            if vis_value == 0:
                if median_margin < 2.25 or median_purity < 1.50 or bit_score < 0.65:
                    continue

            # For 16-bit extended VIS, require parity OK on the extended
            # byte too - otherwise we'd misidentify the mode.
            if vis_value == EXTENDED_VIS_PRIMARY and extended_vis_value is not None:
                if not extended_parity_ok:
                    continue

            # FIX: use resolved_mode_name (the specific MMSSTV mode identified
            # via the extended byte) rather than mode_name (which for an
            # extended-VIS transmission may still be the sentinel stored in
            # vis_codes[35]).  _mode_sync_sanity probes for sync pulses at
            # the mode's actual line rate, so using the wrong name here would
            # cause it to reject a valid transmission.
            sync_sanity = self._mode_sync_sanity(
                resolved_mode_name,
                vis_start,
                freq_offset_hz,
            )

            if sync_sanity.get("available") and not sync_sanity.get("ok"):
                continue

            sync_score = float(sync_sanity.get("score", 0.0))

            score = (
                    bit_score
                    + 0.4 * math.log(max(start_ratio, 1e-12))
                    + 0.4 * math.log(max(stop_ratio, 1e-12))
                    + 0.6 * sync_score
                    - 15.0 * abs(shift)
            )

            item = {
                "vis_start": float(vis_start),
                "vis_shift_ms": float(shift * 1000.0),
                "bits": bits,
                "vis_value": int(vis_value),
                "parity_ok": bool(parity_ok),
                "mode_name": mode_name,
                "bit_score": float(bit_score),
                "median_data_margin": float(median_margin),
                "median_data_purity": float(median_purity),
                "start_bit_1200_ratio": float(start_ratio),
                "stop_bit_1200_ratio": float(stop_ratio),
                "mode_sync_sanity": sync_sanity,
                "score": float(score),
                "extended_vis_value": (
                    int(extended_vis_value) if extended_vis_value is not None else None
                ),
                "extended_parity_ok": (
                    bool(extended_parity_ok) if extended_parity_ok is not None else None
                ),
            }

            if best is None or item["score"] > best["score"]:
                best = item

        return best

    def _verify_leader_header(
            self,
            leader_start,
            vis_start,
            freq_offset_hz,
            vis_info,
    ):
        p = self.protocol

        first_mid_start = leader_start + 0.050
        second_mid_start = (
                leader_start
                + p.leader_tone_seconds
                + p.break_tone_seconds
                + 0.050
        )

        first_1900 = self.detector.tone_power(
            SSTV_LEADER_HZ + freq_offset_hz,
            first_mid_start,
            0.200,
        )

        first_1200 = self.detector.tone_power(
            SSTV_SYNC_HZ + freq_offset_hz,
            first_mid_start,
            0.200,
        )

        second_1900 = self.detector.tone_power(
            SSTV_LEADER_HZ + freq_offset_hz,
            second_mid_start,
            0.200,
        )

        second_1200 = self.detector.tone_power(
            SSTV_SYNC_HZ + freq_offset_hz,
            second_mid_start,
            0.200,
        )

        break_start = leader_start + p.leader_tone_seconds

        break_1200 = self.detector.tone_power(
            SSTV_SYNC_HZ + freq_offset_hz,
            break_start,
            p.break_tone_seconds,
        )

        break_1900 = self.detector.tone_power(
            SSTV_LEADER_HZ + freq_offset_hz,
            break_start,
            p.break_tone_seconds,
        )

        measured_leader = vis_start - leader_start
        expected_leader = p.full_leader_and_break_seconds
        duration_error_ms = (measured_leader - expected_leader) * 1000.0

        first_ratio = first_1900 / max(first_1200, 1e-12)
        second_ratio = second_1900 / max(second_1200, 1e-12)
        break_ratio = break_1200 / max(break_1900, 1e-12)

        ok = (
                abs(duration_error_ms) < 60.0 and
                first_ratio > 3.0 and
                second_ratio > 3.0 and
                break_ratio > 1.3 and
                vis_info["parity_ok"] and
                vis_info["mode_name"] is not None
        )

        return {
            "ok": bool(ok),
            "expected_leader_seconds": float(expected_leader),
            "measured_leader_seconds": float(measured_leader),
            "duration_error_ms": float(duration_error_ms),
            "first_1900_vs_1200_ratio": float(first_ratio),
            "break_1200_vs_1900_ratio": float(break_ratio),
            "second_1900_vs_1200_ratio": float(second_ratio),
            "vis_start_adjust_ms": float(vis_info["vis_shift_ms"]),
            "start_bit_1200_ratio": float(vis_info["start_bit_1200_ratio"]),
            "stop_bit_1200_ratio": float(vis_info["stop_bit_1200_ratio"]),
        }

    def _measure_line_rate(self, scan_seconds=60.0):
        """
        Measure the sync-pulse interval in the audio, assuming no mode.

        Only called when VIS detection has already failed, so the cost of
        demodulating is paid on the unhappy path only. It finds 1200 Hz
        pulses, takes the median gap between them, and reports which modes
        run at that line rate. Measured against all 29 modes the median gap
        lands within 0.35% of the true line time, so a 1.5% tolerance keeps
        the right mode and excludes its neighbours.
        """
        cached = getattr(self, "_line_rate_cache", None)

        if cached is not None:
            return cached

        info = {
            "measured_line_seconds": None,
            "sync_pulses_found": 0,
            "line_rate_confidence": 0.0,
            "line_rate_candidates": [],
        }

        self._line_rate_cache = info

        try:
            demod = getattr(self, "_demod", None)

            if demod is None:
                demod = self._demod = FMDemodulator(
                    self.fs,
                    self.samples,
                    freq_offset_hz=self.freq_offset_hz,
                )

            t_from = min(1.5, 0.25 * self.total_duration)
            t_to = min(self.total_duration, t_from + scan_seconds)

            if t_to - t_from < 1.0:
                return info

            step = 0.002

            times, score = demod.sync_score(
                0.005,
                step,
                t_from,
                t_to,
                center=SSTV_SYNC_HZ + self.freq_offset_hz,
                half_width=160.0,
            )

            if len(times) < 4:
                return info

            med = np.median(score)
            mad = np.median(np.abs(score - med)) + 1e-12

            peaks, _ = find_peaks(
                score,
                height=max(0.25, med + 4.0 * mad),
                distance=max(int(round(0.02 / step)), 1),
            )

            if len(peaks) < 4:
                return info

            gaps = np.diff(times[peaks])
            measured = float(np.median(gaps))

            if measured <= 0:
                return info

            info["measured_line_seconds"] = measured
            info["sync_pulses_found"] = int(len(peaks))
            info["line_rate_confidence"] = float(
                np.mean(np.abs(gaps - measured) <= 0.02 * measured)
            )
            info["line_rate_candidates"] = self._modes_matching_line_rate(
                measured
            )
        except Exception:
            return info

        return info

    def _modes_matching_line_rate(self, measured, tolerance=0.015):
        """
        Every mode whose line time matches the measured pulse interval.

        Several pairs genuinely share a line time - Martin M1 and M3, Scottie
        S1 and S3, Martin M2 and M4, Scottie S2 and S4 - so this returns all
        of them and lets the caller choose, rather than guessing one.
        """
        rows = []

        for name, layout in self.registry.image_layouts.items():
            line_seconds = float(layout.line_seconds)

            if line_seconds <= 0 or layout.sync_seconds <= 0:
                continue

            error = abs(measured - line_seconds) / line_seconds

            if error <= tolerance:
                rows.append(
                    {
                        "mode_name": name,
                        "line_seconds": round(line_seconds, 6),
                        "error_percent": round(100.0 * error, 2),
                    }
                )

        rows.sort(key=lambda row: (row["error_percent"], row["mode_name"]))

        return rows

    def _line_rate_hint(self):
        """
        A sentence naming the modes that fit the measured line rate.
        """
        info = self._measure_line_rate()
        candidates = info["line_rate_candidates"]

        if not candidates or info["measured_line_seconds"] is None:
            return None

        names = ", ".join(
            repr(row["mode_name"]) for row in candidates[:4]
        )

        return (
                "The audio carries sync pulses every %.1f ms (%d pulses, %.0f%% "
                "of the gaps agree), which matches %s. Retry with "
                "forced_mode_name= set to one of them."
                % (
                    info["measured_line_seconds"] * 1000.0,
                    info["sync_pulses_found"],
                    100.0 * info["line_rate_confidence"],
                    names,
                )
        )

    def detect_mode(self):
        p = self.protocol

        candidates = self._coarse_leader_candidates()

        if not candidates:
            return {
                "error": "No leader candidates found.",
                "tried_offsets_hz": self.LEADER_OFFSETS_HZ,
                "line_rate": self._measure_line_rate(),
                "hint": self._line_rate_hint(),
            }

        attempted = []

        for cand in candidates[:80]:
            fine = self._fine_leader_from_candidate(cand)

            if fine is None:
                continue

            vis_info = self._try_decode_vis_near(
                fine["leader_end"],
                cand["offset"],
            )

            if vis_info is None:
                attempted.append({
                    **cand,
                    **fine,
                    "vis": "no valid VIS near leader end",
                })
                continue

            check = self._verify_leader_header(
                fine["leader_start"],
                vis_info["vis_start"],
                cand["offset"],
                vis_info,
            )

            if check["ok"]:
                self.freq_offset_hz = cand["offset"]
                self.leader_start = fine["leader_start"]
                self.leader_end = fine["leader_end"]
                self.vis_start = vis_info["vis_start"]
                # When the detected mode is a 16-bit extended VIS mode,
                # the header is twice as long (8 bits primary + 8 bits
                # extended), so vis_end moves back by another vis_header
                # block.
                ext_code = self.registry.extended_vis_code_for_mode(vis_info["mode_name"])
                header_blocks = 2 if ext_code is not None else 1
                self.vis_end = self.vis_start + header_blocks * p.vis_header_seconds
                self.mode = self.registry.get_mode(vis_info["mode_name"])
                self.vis_value = vis_info["vis_value"]
                self.leader_check = check

                return {
                    "leader_start": self.leader_start,
                    "leader_end": self.leader_end,
                    "leader_duration_raw": fine["leader_duration"],
                    "vis_start": self.vis_start,
                    "vis_end": self.vis_end,
                    "vis_value": vis_info["vis_value"],
                    "bits": vis_info["bits"],
                    "mode_name": vis_info["mode_name"],
                    "experimental": is_experimental_mode(vis_info["mode_name"]),
                    "notice": experimental_notice(vis_info["mode_name"]),
                    "freq_offset_hz": self.freq_offset_hz,
                    "leader_candidate_score": cand["score"],
                    "leader_check": check,
                    "vis_bit_score": float(vis_info.get("bit_score", 0.0)),
                    "vis_data_margin": float(
                        vis_info.get("median_data_margin", 0.0)
                    ),
                    "vis_data_purity": float(
                        vis_info.get("median_data_purity", 0.0)
                    ),
                    "extended_vis_value": vis_info.get("extended_vis_value"),
                    "extended_parity_ok": vis_info.get("extended_parity_ok"),
                    "mode_sync_sanity": dict(
                        vis_info.get("mode_sync_sanity") or {}
                    ),
                }

            attempted.append({
                **cand,
                **fine,
                "vis_value": vis_info["vis_value"],
                "mode_name": vis_info["mode_name"],
                "leader_check": check,
            })

        unknown = getattr(self, "_unknown_vis_codes", None) or {}
        error = ("Leader candidates were found, but none passed "
                 "VIS/header validation.")
        extra = {}

        if unknown:
            code, hits = max(unknown.items(), key=lambda kv: (kv[1], kv[0]))
            error = ("VIS code %d was read with valid parity %d time(s), "
                     "but no mode is registered for that code. Add one with "
                     "registry.add_vis_code(%d, ...) or decode with "
                     "forced_mode_name." % (code, hits, code))
            extra = {"unknown_vis_code": code, "unknown_vis_reads": hits}

        return {
            "error": error,
            "candidate_count": len(candidates),
            "attempted_first_few": attempted[:5],
            "line_rate": self._measure_line_rate(),
            "hint": self._line_rate_hint(),
            **extra,
        }

    def _resolve_line_structure(self, requested):
        """
        Decide between one sync per image row and one per colour component.

        "auto" measures the sync-pulse interval, which is 475 ms for
        standard SC2-120 and 163 ms for the component-per-line reading of
        the same pixel time - a factor of three apart, so the choice is not
        close. Anything that cannot be measured falls back to standard.
        """
        if requested in (None, "", "standard"):
            return "standard"

        if requested in ("rgb3", "component", "component-lines"):
            return "rgb3"

        if requested != "auto":
            return "standard"

        if self.mode is None:
            return "standard"

        if self.mode.name not in self.registry.line_structure_variants:
            return "standard"

        try:
            standard = self.registry.get_layout(self.mode.name)
            variant = self.registry.get_layout(
                self.mode.name,
                line_structure="rgb3",
            )

            measured = self._measure_line_rate().get("measured_line_seconds")

            if not measured:
                return "standard"

            error_standard = abs(measured - standard.line_seconds) / standard.line_seconds
            error_variant = abs(measured - variant.line_seconds) / variant.line_seconds

            if error_variant < error_standard and error_variant <= 0.03:
                return "rgb3"

            if error_standard <= 0.03:
                return "standard"
        except Exception:
            pass

        probed = getattr(self, "_sync_sanity_structure", None)

        if probed in ("standard", "rgb3"):
            return probed

        return "standard"

    def decode_image(
            self,
            out_path,
            slant_search=0.03,
            output_mode="all",
            line_structure="auto",
    ):
        if self.mode is None:
            raise RuntimeError("call detect_mode() or force_mode() first")

        if self.mode.name not in self.registry.image_layouts:
            return {
                "error": f"No image layout defined for {self.mode.name}",
                "mode_name": self.mode.name,
                "supported_image_layout_modes": self.registry.supported_decoder_image_modes(),
            }

        self.line_structure = self._resolve_line_structure(line_structure)

        layout = self.registry.get_layout(
            self.mode.name,
            line_structure=self.line_structure,
        )

        img = ImageDecoder(
            self.fs,
            self.samples,
            layout,
            self.vis_end,
            freq_offset_hz=self.freq_offset_hz,
            slant_search=slant_search,
        )

        rgb = img.decode()

        if rgb is None:
            return {
                "error": img.info.get("error", "image decode failed"),
                "mode_name": self.mode.name,
                "info": dict(img.info),
            }

        base, ext = os.path.splitext(out_path)

        if not ext:
            ext = ".jpg"

        if output_mode not in OUTPUT_IMAGE_MODES:
            return {
                "error": f"Unknown output_mode {output_mode!r}",
                "valid_output_modes": list(OUTPUT_IMAGE_MODES),
            }

        paths = {}

        mode_name = self.mode.name
        vis_text = f"VIS {self.vis_value}" if self.vis_value is not None else "VIS ?"
        caption = f"{mode_name} | {vis_text}"

        measured_line_seconds = img.info.get(
            "measured_line_seconds",
            layout.line_seconds,
        )

        first_sync = img.info.get(
            "first_line_sync",
            self.vis_end + layout.leadin_seconds,
        )

        slant_ratio = measured_line_seconds / layout.line_seconds

        image_audio_start = max(
            0.0,
            first_sync - layout.sync_offset * slant_ratio - 0.05,
        )

        image_audio_end = min(
            self.total_duration,
            image_audio_start + layout.n_lines * measured_line_seconds + 0.20,
        )

        if output_mode in ("raw", "all"):
            raw_path = out_path if output_mode == "raw" else f"{base}_raw{ext}"
            save_raw_jpg(rgb, raw_path)
            paths["raw"] = raw_path

        if output_mode in ("polaroid", "polaroid_text", "all"):
            polaroid_text_path = (
                out_path
                if output_mode in ("polaroid", "polaroid_text")
                else f"{base}_polaroid_text{ext}"
            )

            save_polaroid_jpg(
                rgb,
                polaroid_text_path,
                caption=caption,
                show_text=True,
            )

            paths["polaroid_text"] = polaroid_text_path

        if output_mode in ("polaroid_notext", "all"):
            polaroid_notext_path = (
                out_path
                if output_mode == "polaroid_notext"
                else f"{base}_polaroid_notext{ext}"
            )

            save_polaroid_jpg(
                rgb,
                polaroid_notext_path,
                caption=None,
                show_text=False,
            )

            paths["polaroid_notext"] = polaroid_notext_path

        if output_mode in ("spectrogram", "spectrogram_text", "all"):
            spectrogram_text_path = (
                out_path
                if output_mode in ("spectrogram", "spectrogram_text")
                else f"{base}_spectrogram_text{ext}"
            )

            save_image_with_spectrogram_jpg(
                rgb=rgb,
                samples=self.samples,
                fs=self.fs,
                t_start=image_audio_start,
                t_end=image_audio_end,
                out_path=spectrogram_text_path,
                caption=f"{caption} audio spectrogram",
                show_text=True,
            )

            paths["spectrogram_text"] = spectrogram_text_path

        if output_mode in ("spectrogram_notext", "all"):
            spectrogram_notext_path = (
                out_path
                if output_mode == "spectrogram_notext"
                else f"{base}_spectrogram_notext{ext}"
            )

            save_image_with_spectrogram_jpg(
                rgb=rgb,
                samples=self.samples,
                fs=self.fs,
                t_start=image_audio_start,
                t_end=image_audio_end,
                out_path=spectrogram_notext_path,
                caption=None,
                show_text=False,
            )

            paths["spectrogram_notext"] = spectrogram_notext_path

        if output_mode in ("stack", "all"):
            stack_path = (
                out_path
                if output_mode == "stack"
                else f"{base}_stack{ext}"
            )

            save_stacked_sstv_spectrogram_jpg(
                rgb=rgb,
                samples=self.samples,
                fs=self.fs,
                t_start=image_audio_start,
                t_end=image_audio_end,
                out_path=stack_path,
            )

            paths["stack"] = stack_path

        return {
            "paths": paths,
            "output_mode": output_mode,
            "spectrogram_audio_start": float(image_audio_start),
            "spectrogram_audio_end": float(image_audio_end),
            **img.info,
        }


def _detect_header_from_candidate(decoder, cand):
    fine = decoder._fine_leader_from_candidate(cand)
    if fine is None:
        return None

    vis_info = decoder._try_decode_vis_near(
        fine["leader_end"],
        cand["offset"],
    )

    if vis_info is None:
        return None

    check = decoder._verify_leader_header(
        fine["leader_start"],
        vis_info["vis_start"],
        cand["offset"],
        vis_info,
    )

    if not check.get("ok", False):
        return None

    return {
        "candidate": cand,
        "fine": fine,
        "vis_info": vis_info,
        "leader_check": check,
    }


def _shorten(text, width=96):
    text = str(text)
    return text if len(text) <= width else text[:width - 3] + "..."

