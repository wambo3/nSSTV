"""
nSSTV - SSTV audio/image encoder-decoder library.

Made by Nana Kwadjo Agyare.

Features:
- Decode WAV/MP3 SSTV audio into images.
- Encode images into SSTV WAV/MP3 audio.
- Standard VIS leader/header support.
- Custom mode support via Python or JSON.
- Multiple output image styles:
  raw, polaroid, spectrogram, stacked.
- CLI support and no-argument script mode.
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pprint import pprint

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import next_fast_len
from scipy.io import wavfile
from scipy.signal import (
    butter,
    sosfiltfilt,
    find_peaks,
    medfilt,
    spectrogram,
)

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv.*")
        from pydub import AudioSegment
except Exception:
    AudioSegment = None

__version__ = "1.0.0"
__author__ = "Nana Kwadjo Agyare"

OUTPUT_IMAGE_MODES = (
    "raw",
    "polaroid",
    "polaroid_text",
    "polaroid_notext",
    "spectrogram",
    "spectrogram_text",
    "spectrogram_notext",
    "stack",
    "all",
)

SSTV_SYNC_HZ = 1200.0
SSTV_BLACK_HZ = 1500.0
SSTV_WHITE_HZ = 2300.0
SSTV_LEADER_HZ = 1900.0
VIS_ONE_HZ = 1100.0
VIS_ZERO_HZ = 1300.0

SMOOTHING_PIXEL_DIVISOR = 3.0
SMOOTHING_MIN_SECONDS = 0.00005
SMOOTHING_MAX_SECONDS = 0.00015
SMOOTHING_BASIS = "fastest"

EXPERIMENTAL_MODES = ("Pasokon P3", "Pasokon P5", "Pasokon P7", "PD-50")

EXPERIMENTAL_DECODE_ALL_MODES = ("Scottie DX",)


def is_experimental_mode(name, context=None):
    """
    True when the mode `name` is marked experimental.

    context="decode-all" also flags modes that are experimental only
    inside the multi-transmission scanner (Scottie DX).
    """
    if not name:
        return False

    if name in EXPERIMENTAL_MODES:
        return True

    return context == "decode-all" and name in EXPERIMENTAL_DECODE_ALL_MODES


def experimental_notice(name, context=None):
    """
    One-line human explanation for an experimental flag, or None.
    """
    if not is_experimental_mode(name, context):
        return None

    if context == "decode-all" and name in EXPERIMENTAL_DECODE_ALL_MODES:
        return (
            "%s is experimental in decode-all: long transmissions are hard "
            "to tell apart from neighbouring signals, so check the result" % name
        )

    return (
        "%s is experimental: the layout is unverified and decoding may be "
        "unreliable" % name
    )


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


def warn_mp3_lossy(action=None, path=None, bitrate=None):
    msg = (
        "MP3 is lossy"

    )

    if action:
        msg += f" Action: {action}."

    if path:
        msg += f" File: {path!r}."

    if bitrate:
        msg += f" Bitrate: {bitrate}; high bitrate, but still lossy."

    warnings.warn(msg, LossyMP3Warning, stacklevel=2)


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


def ffmpeg_is_available():
    return configure_pydub_ffmpeg(require=False) is not None


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


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


def load_audio_mono(path):
    """
    Load audio as mono float64 samples.

    WAV:
        loaded with scipy.io.wavfile.

    MP3 / other:
        loaded with pydub + ffmpeg.

    Returns:
        fs, samples
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".wav", ".wave"):
        fs, samples = wavfile.read(path)
        return fs, _normalize_audio(samples)

    if ext == ".mp3":
        warn_mp3_lossy("reading MP3 SSTV audio", path)

    configure_pydub_ffmpeg(require=True)

    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:
        raise RuntimeError(
            f"Could not read audio file {path!r}. "
            "For MP3/non-WAV support, make sure ffmpeg is installed and on PATH."
        ) from e

    audio = audio.set_channels(1)
    fs = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())

    return fs, _normalize_audio(samples)


def _pcm16(audio):
    """Sanitize float audio [-1, 1] into int16 PCM samples."""
    audio = np.asarray(audio, dtype=np.float64)
    audio[~np.isfinite(audio)] = 0.0
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def write_wav_file(path, fs, audio):
    """
    Write float audio [-1, 1] as int16 WAV.
    """
    fs = int(fs)
    if fs <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {fs}")

    _ensure_parent(path)
    wavfile.write(path, fs, _pcm16(audio))

    return path


_MP3_BITRATE_RE = None


def _validate_mp3_bitrate(bitrate):
    """
    Reject nonsense MP3 bitrates early ("banana") instead of letting ffmpeg
    silently fall back to its default.
    """
    global _MP3_BITRATE_RE

    if _MP3_BITRATE_RE is None:
        import re
        _MP3_BITRATE_RE = re.compile(r"^\d+(\.\d+)?\s*[kKmM]?$")

    text = str(bitrate).strip()
    if not _MP3_BITRATE_RE.match(text):
        raise ValueError(
            f"Invalid MP3 bitrate {bitrate!r}. Use a number with an optional "
            "k/M suffix, e.g. '320k', '128k' or '256000'."
        )

    return text


def write_mp3_file(path, fs, audio, bitrate="320k"):
    """
    Write MP3 using pydub/ffmpeg.

    Uses a temporary file first so failed exports do not leave empty MP3 files.
    """
    bitrate = _validate_mp3_bitrate(bitrate)
    warn_mp3_lossy("writing SSTV MP3 audio", path, bitrate=bitrate)

    configure_pydub_ffmpeg(require=True)
    _ensure_parent(path)

    pcm = _pcm16(audio)

    segment = AudioSegment(
        data=pcm.tobytes(),
        sample_width=2,
        frame_rate=int(fs),
        channels=1,
    )

    out_dir = os.path.dirname(os.path.abspath(path)) or "."

    tmp_file = tempfile.NamedTemporaryFile(
        prefix=".nsstv_tmp_",
        suffix=".mp3",
        dir=out_dir,
        delete=False,
    )
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        exported = segment.export(tmp_path, format="mp3", bitrate=bitrate)

        try:
            exported.close()
        except Exception:
            pass

        if not os.path.exists(tmp_path):
            raise RuntimeError("ffmpeg did not create an MP3 file.")

        size = os.path.getsize(tmp_path)

        if size < 1024:
            raise RuntimeError(
                f"ffmpeg created an invalid/empty MP3 file: {tmp_path} "
                f"({size} bytes)."
            )

        os.replace(tmp_path, path)

    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return path


def write_audio_file(path, fs, audio, bitrate="320k"):
    """
    Write WAV or MP3 based on file extension.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".mp3":
        return write_mp3_file(path, fs, audio, bitrate=bitrate)

    return write_wav_file(path, fs, audio)


@dataclass
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


@dataclass
class SSTVMode:
    name: str
    pixel_seconds: float
    sync_seconds: float
    porch_seconds: float
    experimental: bool = False


@dataclass
class ImageLayout:
    name: str
    width: int
    height: int
    color: str
    line_seconds: float
    sync_seconds: float
    sync_offset: float
    channels: tuple
    channels_odd: tuple = None
    leadin_seconds: float = 0.0
    rows_per_line: int = 1

    transmitted_lines: int = None
    line_channel_order: tuple = None

    @property
    def n_lines(self):
        if self.transmitted_lines is not None:
            return self.transmitted_lines

        return self.height // self.rows_per_line

    @property
    def fastest_pixel_seconds(self):
        planes = list(self.channels or ()) + list(self.channels_odd or ())

        if not planes:
            return float("nan")

        return min(duration for _key, _start, duration in planes) / float(self.width)

    @property
    def luma_pixel_seconds(self):
        for key, _start, duration in (self.channels or ()):
            if key in ("Y", "G"):
                return duration / float(self.width)

        return self.fastest_pixel_seconds


def _martin(name, height, scan):
    sync, porch, sep = 0.004862, 0.000572, 0.000572

    g = sync + porch
    b = g + scan + sep
    r = b + scan + sep
    line = r + scan + sep

    return ImageLayout(
        name=name,
        width=320,
        height=height,
        color="rgb",
        line_seconds=line,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=(("G", g, scan), ("B", b, scan), ("R", r, scan)),
    )


def _scottie(name, height, scan):
    sync, porch, sep = 0.009, 0.0015, 0.0015

    g = sep
    b = g + scan + sep
    sync_off = b + scan
    r = sync_off + sync + porch
    line = r + scan

    return ImageLayout(
        name=name,
        width=320,
        height=height,
        color="rgb",
        line_seconds=line,
        sync_seconds=sync,
        sync_offset=sync_off,
        channels=(("G", g, scan), ("B", b, scan), ("R", r, scan)),
        leadin_seconds=sync,
    )


def _pd(name, width, height, scan):
    sync, porch = 0.020, 0.00208
    y = sync + porch

    return ImageLayout(
        name=name,
        width=width,
        height=height,
        color="pd",
        line_seconds=y + 4 * scan,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=(
            ("Y0", y, scan),
            ("Cr", y + scan, scan),
            ("Cb", y + 2 * scan, scan),
            ("Y1", y + 3 * scan, scan),
        ),
        rows_per_line=2,
    )


def _robot36():
    sync, porch, sep, cporch = 0.009, 0.003, 0.0045, 0.0015

    y = sync + porch
    c = y + 0.088 + sep + cporch

    return ImageLayout(
        name="Robot 36",
        width=320,
        height=240,
        color="ycrcb420",
        line_seconds=c + 0.044,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=(("Y", y, 0.088), ("Cr", c, 0.044)),
        channels_odd=(("Y", y, 0.088), ("Cb", c, 0.044)),
    )


def _robot12():
    """
    SSTV Handbook: 160x120, YCrCb 4:2:0, sync 7.0 ms, porch 3.0 ms,
    Y 60 ms, one chroma component of 30 ms per line, 600 lines/min.

    Even lines carry R-Y, odd lines carry B-Y, and each chroma line is shared
    by the pair, so the chroma planes are 60 rows of 160.
    """
    sync, porch = 0.007, 0.003

    y = sync + porch
    c = y + 0.060

    return ImageLayout(
        name="Robot 12 Color",
        width=160,
        height=120,
        color="ycrcb420",
        line_seconds=c + 0.030,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=(("Y", y, 0.060), ("Cr", c, 0.030)),
        channels_odd=(("Y", y, 0.060), ("Cb", c, 0.030)),
    )


def _robot24():
    """
    SSTV Handbook: 160x120, YCrCb 4:2:2, sync 12.0 ms, Y 88 ms, R-Y 44 ms,
    B-Y 44 ms, 6.0 ms before each chroma component, 300 lines/min.

    Both chroma components go out on every line, so this is 4:2:2, not the
    4:2:0 that Robot 12 and Robot 36 use.
    """
    sync, color_sync = 0.012, 0.006

    y = sync
    cr = y + 0.088 + color_sync
    cb = cr + 0.044 + color_sync

    return ImageLayout(
        name="Robot 24 Color",
        width=160,
        height=120,
        color="ycrcb",
        line_seconds=cb + 0.044,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=(("Y", y, 0.088), ("Cr", cr, 0.044), ("Cb", cb, 0.044)),
    )


def _robot72():
    sync, porch, sep, cporch = 0.009, 0.003, 0.0045, 0.0015

    y = sync + porch
    cr = y + 0.138 + sep + cporch
    cb = cr + 0.069 + sep + cporch

    return ImageLayout(
        name="Robot 72",
        width=320,
        height=240,
        color="ycrcb",
        line_seconds=cb + 0.069,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=(("Y", y, 0.138), ("Cr", cr, 0.069), ("Cb", cb, 0.069)),
    )


def _pasokon(name, sync, porch, scan):
    r = sync + porch
    g = r + scan + porch
    b = g + scan + porch

    return ImageLayout(
        name=name,
        width=640,
        height=496,
        color="rgb",
        line_seconds=b + scan + porch,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=(("R", r, scan), ("G", g, scan), ("B", b, scan)),
    )


def _wraase_sc2(name, pixel_seconds, order=("R", "G", "B")):
    """
    Shared builder for the Wraase SC-2 colour modes.

    One line is a 5.5225 ms sync, a 0.5 ms porch, and then the three colour
    scans back to back with no separators between them, which is what the
    published pixel times imply:

        SC-2 60    scan = 320 * 0.000244150 =  78.128 ms   line  240.407 ms
        SC-2 120   scan = 320 * 0.000489000 = 156.480 ms   line  475.463 ms
        SC-2 180   scan = 320 * 0.00073437500 = 235.040 ms   line  711.143 ms

    Order is (R, G, B) as specified. If a decode comes back aligned but with
    the colours permuted, pass a different order rather than touching timing.
    """
    width = 320
    height = 256

    sync = 0.0055225
    porch = 0.000500

    scan = width * pixel_seconds

    c0 = sync + porch
    c1 = c0 + scan
    c2 = c1 + scan

    line = c2 + scan

    channels = (
        (order[0], c0, scan),
        (order[1], c1, scan),
        (order[2], c2, scan),
    )

    return ImageLayout(
        name=name,
        width=width,
        height=height,
        color="rgb",
        line_seconds=line,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=channels,
    )


def _wraase_sc2_60():
    """Wraase SC2-60: 320x256, RGB, 61.544 s over 256 lines, VIS 59."""
    return _wraase_sc2("Wraase SC-2 60", 0.000244150)


def _wraase_sc2_120():
    """Wraase SC2-120: 320x256, RGB, 121.7 s over 256 lines, VIS 63.

    Uses half-res R and B channels (empirically derived from audio analysis).
    """
    return ImageLayout(
        name="Wraase SC-2 120",
        width=320,
        height=256,
        color="rgb",
        line_seconds=0.476038,
        sync_seconds=0.0055225,
        sync_offset=0.0,
        channels=(
            ("R", 0.0157, 0.0976),
            ("G", 0.1424, 0.1951),
            ("B", 0.3675, 0.0976),
        ),
    )


def _wraase_sc2_rgb3(name, pixel_seconds, order=("R", "G", "B")):
    """
    Wraase SC-2 as one transmitted line per colour component.

    The published SC-2 timing is one sync per image row followed by three
    scans, and that is what _wraase_sc2() builds. Some transmitters instead
    send a sync before every component:

        sync + porch + R row 0
        sync + porch + G row 0
        sync + porch + B row 0
        sync + porch + R row 1
        ...

    Feeding that to a one-scan-per-row decoder puts each colour plane on a
    different part of the line, which is the RGB ghosting this fixes. Each
    transmitted line is sync + porch + one scan, so a frame is three times
    as many lines and a little longer than the one-sync-per-row reading of
    the same pixel time.
    """
    width = 320
    height = 256

    sync = 0.0055225
    porch = 0.000500

    scan = width * pixel_seconds

    component_offset = sync + porch
    component_line = sync + porch + scan

    return ImageLayout(
        name=name,
        width=width,
        height=height,
        color="rgb3",
        line_seconds=component_line,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=(("component", component_offset, scan),),
        transmitted_lines=height * 3,
        line_channel_order=order,
    )


def _wraase_sc2_180():
    """Wraase SC2-180: 320x256, RGB, 182 s over 256 lines, VIS 55."""
    return _wraase_sc2("Wraase SC-2 180", 0.000734375)


def make_builtin_layouts():
    layouts = [
        _martin("Martin M1", 256, 0.146432),
        _martin("Martin M2", 256, 0.073216),
        _martin("Martin M3", 128, 0.146432),
        _martin("Martin M4", 128, 0.073216),

        _scottie("Scottie S1", 256, 0.13824),
        _scottie("Scottie S2", 256, 0.088064),
        _scottie("Scottie S3", 128, 0.13824),
        _scottie("Scottie S4", 128, 0.088064),
        _scottie("Scottie DX", 256, 0.3456),
        _scottie("Scottie DX2", 256, 0.169984),

        _robot12(),
        _robot24(),
        _robot36(),
        _robot72(),

        _pd("PD-50", 320, 256, 0.09152),
        _pd("PD-90", 320, 256, 0.17024),
        _pd("PD-120", 640, 496, 0.1216),
        _pd("PD-160", 512, 400, 0.195584),
        _pd("PD-180", 640, 496, 0.18304),
        _pd("PD-240", 640, 496, 0.24448),
        _pd("PD-290", 800, 616, 0.2288),

        _pasokon("Pasokon P3", 25 / 4800, 5 / 4800, 640 / 4800),
        _pasokon("Pasokon P5", 25 / 3200, 5 / 3200, 640 / 3200),
        _pasokon("Pasokon P7", 25 / 2400, 5 / 2400, 640 / 2400),

        _wraase_sc2_180(),
        _wraase_sc2_120(),
        _wraase_sc2_60(),
    ]

    return {layout.name: layout for layout in layouts}


def _channels_tuple(channels):
    if channels is None:
        return None

    return tuple((str(k), float(off), float(dur)) for k, off, dur in channels)


def make_rgb3_layout_variants():
    """
    Wraase SC-2 layouts for transmitters that sync before every component.

    Same pixel times as the standard layouts, so the scan rates and the
    published durations per component are unchanged; only the line
    structure differs. Auto-selected at decode time when the measured sync
    interval matches these rather than the standard layouts.
    """
    return {
        "Wraase SC-2 60": _wraase_sc2_rgb3("Wraase SC-2 60", 0.0002442),
        "Wraase SC-2 120": _wraase_sc2_rgb3("Wraase SC-2 120", 0.0004890),
        "Wraase SC-2 180": _wraase_sc2_rgb3("Wraase SC-2 180", 0.000734375),
    }


class ModeRegistry:
    """
    Registry of:
    - mode timing metadata
    - VIS code mappings
    - image layouts used by encoder/decoder

    Users can add custom modes programmatically or from JSON.
    """

    def __init__(self, load_builtin=True):
        self.modes = {}
        self.vis_codes = {}
        self.image_layouts = {}
        self.line_structure_variants = {}

        if load_builtin:
            self._load_builtin_modes()
            self._load_builtin_vis_codes()
            self._load_builtin_layouts()

    def _load_builtin_modes(self):
        specs = {
            "Martin M1": (0.0004576, 0.004862, 0.000572),
            "Martin M2": (0.0002288, 0.004862, 0.000572),
            "Martin M3": (0.0002288, 0.004862, 0.000572),
            "Martin M4": (0.0002288, 0.004862, 0.000572),
            "Scottie S1": (0.0004320, 0.009000, 0.001500),
            "Scottie S2": (0.0002752, 0.009000, 0.001500),
            "Scottie S3": (0.0002752, 0.009000, 0.001500),
            "Scottie S4": (0.0002752, 0.009000, 0.001500),
            "Scottie DX": (0.0010805, 0.009000, 0.001500),
            "Scottie DX2": (0.0005312, 0.009000, 0.001500),

            "Robot 12 Color": (0.0002083, 0.007000, 0.003000),
            "Robot 24 Color": (0.0002083, 0.012000, 0.003000),
            "Robot 36": (0.0001375, 0.010500, 0.003000),
            "Robot 72": (0.0002875, 0.012000, 0.003000),

            "PD-50": (0.000286, 0.020000, 0.00208),
            "PD-90": (0.000532, 0.020000, 0.00208),
            "PD-120": (0.000190, 0.020000, 0.00208),
            "PD-160": (0.000382, 0.020000, 0.00208),
            "PD-180": (0.000286, 0.020000, 0.00208),
            "PD-240": (0.000382, 0.020000, 0.00208),
            "PD-290": (0.000286, 0.020000, 0.00208),

            "Pasokon P3": (0.0002083, 0.005208, 0.001042),
            "Pasokon P5": (0.0003125, 0.007813, 0.001563),
            "Pasokon P7": (0.0004167, 0.010417, 0.002083),

            "Wraase SC-2 180": (0.000734375, 0.0055225, 0.000500),
            "Wraase SC-2 120": (0.0004890, 0.0055225, 0.000500),
            "Wraase SC-2 60": (0.0002442, 0.0055225, 0.000500),
        }

        for name, (pixel_s, sync_s, porch_s) in specs.items():
            self.add_mode(
                name=name,
                pixel_seconds=pixel_s,
                sync_seconds=sync_s,
                porch_seconds=porch_s,
                overwrite=True,
            )

        for name in EXPERIMENTAL_MODES:
            self.modes[name].experimental = True

    def _load_builtin_vis_codes(self):
        entries = [
            (44, "Martin M1"),
            (40, "Martin M2"),
            (36, "Martin M3"),
            (32, "Martin M4"),
            (60, "Scottie S1"),
            (56, "Scottie S2"),
            (52, "Scottie S3"),
            (48, "Scottie S4"),
            (76, "Scottie DX"),
            (80, "Scottie DX2"),

            (0, "Robot 12 Color"),
            (4, "Robot 24 Color"),
            (8, "Robot 36"),
            (12, "Robot 72"),

            (93, "PD-50"),
            (99, "PD-90"),
            (95, "PD-120"),
            (98, "PD-160"),
            (96, "PD-180"),
            (97, "PD-240"),
            (94, "PD-290"),

            (113, "Pasokon P3"),
            (114, "Pasokon P5"),
            (115, "Pasokon P7"),

            (55, "Wraase SC-2 180"),
            (63, "Wraase SC-2 120"),
            (59, "Wraase SC-2 60"),
        ]

        for code, name in entries:
            self.add_vis_code(code, name, overwrite=True)

    def _load_builtin_layouts(self):
        for layout in make_builtin_layouts().values():
            self.add_layout(layout, overwrite=True)

        self.line_structure_variants = make_rgb3_layout_variants()

    def add_mode(self, name, pixel_seconds, sync_seconds, porch_seconds=0.0, overwrite=False):
        if name in self.modes and not overwrite:
            raise ValueError(f"Mode {name!r} already exists")

        self.modes[name] = SSTVMode(
            name=str(name),
            pixel_seconds=float(pixel_seconds),
            sync_seconds=float(sync_seconds),
            porch_seconds=float(porch_seconds),
        )

    def add_vis_code(self, code, mode_name, overwrite=False):
        code = int(code)

        if not 0 <= code <= 127:
            raise ValueError("Standard VIS code must fit in 7 bits: 0..127")

        if mode_name not in self.modes:
            raise ValueError(f"Cannot add VIS {code}: unknown mode {mode_name!r}")

        if code in self.vis_codes and self.vis_codes[code] != mode_name and not overwrite:
            raise ValueError(
                f"VIS collision: {code} already maps to {self.vis_codes[code]!r}"
            )

        self.vis_codes[code] = mode_name

    def add_layout(self, layout: ImageLayout, overwrite=False):
        if layout.name in self.image_layouts and not overwrite:
            raise ValueError(f"Image layout for {layout.name!r} already exists")

        if layout.name not in self.modes:
            raise ValueError(
                f"Cannot add layout for {layout.name!r}: add the mode first"
            )

        layout.channels = _channels_tuple(layout.channels)
        layout.channels_odd = _channels_tuple(layout.channels_odd)

        self.image_layouts[layout.name] = layout

    def add_custom_mode(
            self,
            name,
            vis_code=None,
            pixel_seconds=None,
            sync_seconds=None,
            porch_seconds=0.0,
            layout: ImageLayout = None,
            width=None,
            height=None,
            color="rgb",
            line_seconds=None,
            sync_offset=0.0,
            channels=None,
            channels_odd=None,
            leadin_seconds=0.0,
            rows_per_line=1,
            overwrite=False,
    ):
        """
        Add a custom SSTV mode.

        If you want the encoder to include VIS sounds, provide `vis_code`.
        If `vis_code` is None, encode with include_vis=False.

        Example:

            registry.add_custom_mode(
                name="My RGB Mode",
                vis_code=123,
                width=320,
                height=240,
                color="rgb",
                line_seconds=0.500,
                sync_seconds=0.005,
                sync_offset=0.0,
                channels=(
                    ("R", 0.006, 0.160),
                    ("G", 0.166, 0.160),
                    ("B", 0.326, 0.160),
                ),
            )
        """
        name = str(name)

        if layout is None:
            if width is None or height is None or line_seconds is None or channels is None:
                raise ValueError(
                    "Custom mode needs either `layout=ImageLayout(...)` or "
                    "`width`, `height`, `line_seconds`, and `channels`."
                )

            if sync_seconds is None:
                raise ValueError("Custom mode needs `sync_seconds`")

            layout = ImageLayout(
                name=name,
                width=int(width),
                height=int(height),
                color=str(color),
                line_seconds=float(line_seconds),
                sync_seconds=float(sync_seconds),
                sync_offset=float(sync_offset),
                channels=_channels_tuple(channels),
                channels_odd=_channels_tuple(channels_odd),
                leadin_seconds=float(leadin_seconds),
                rows_per_line=int(rows_per_line),
            )
        else:
            layout = ImageLayout(
                name=name,
                width=int(layout.width),
                height=int(layout.height),
                color=str(layout.color),
                line_seconds=float(layout.line_seconds),
                sync_seconds=float(layout.sync_seconds),
                sync_offset=float(layout.sync_offset),
                channels=_channels_tuple(layout.channels),
                channels_odd=_channels_tuple(layout.channels_odd),
                leadin_seconds=float(layout.leadin_seconds),
                rows_per_line=int(layout.rows_per_line),
            )

        if sync_seconds is None:
            sync_seconds = layout.sync_seconds

        if pixel_seconds is None:
            durations = [dur / layout.width for _, _, dur in layout.channels]
            pixel_seconds = min(durations) if durations else 0.0

        self.add_mode(
            name=name,
            pixel_seconds=float(pixel_seconds),
            sync_seconds=float(sync_seconds),
            porch_seconds=float(porch_seconds),
            overwrite=overwrite,
        )

        self.add_layout(layout, overwrite=overwrite)

        if vis_code is not None:
            self.add_vis_code(int(vis_code), name, overwrite=overwrite)

        return layout

    def load_custom_modes_json(self, path, overwrite=False):
        """
        JSON format:

        {
          "modes": [
            {
              "name": "My RGB Mode",
              "vis_code": 123,
              "pixel_seconds": 0.0005,
              "sync_seconds": 0.005,
              "porch_seconds": 0.001,
              "layout": {
                "width": 320,
                "height": 240,
                "color": "rgb",
                "line_seconds": 0.500,
                "sync_seconds": 0.005,
                "sync_offset": 0.0,
                "channels": [
                  ["R", 0.006, 0.160],
                  ["G", 0.166, 0.160],
                  ["B", 0.326, 0.160]
                ],
                "leadin_seconds": 0.0,
                "rows_per_line": 1
              }
            }
          ]
        }
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            entries = data.get("modes", [data])
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError("Custom modes JSON must be an object or list")

        loaded = []

        for item in entries:
            try:
                name = item["name"]
                layout_data = item.get("layout", item)

                layout = ImageLayout(
                    name=name,
                    width=int(layout_data["width"]),
                    height=int(layout_data["height"]),
                    color=str(layout_data.get("color", "rgb")),
                    line_seconds=float(layout_data["line_seconds"]),
                    sync_seconds=float(layout_data.get("sync_seconds", item.get("sync_seconds", 0.0))),
                    sync_offset=float(layout_data.get("sync_offset", 0.0)),
                    channels=_channels_tuple(layout_data["channels"]),
                    channels_odd=_channels_tuple(layout_data.get("channels_odd")),
                    leadin_seconds=float(layout_data.get("leadin_seconds", 0.0)),
                    rows_per_line=int(layout_data.get("rows_per_line", 1)),
                )
            except (KeyError, TypeError, ValueError) as e:
                label = item.get("name", "?") if isinstance(item, dict) else item
                raise ValueError(
                    f"Invalid custom mode entry {label!r} in {path!r}: "
                    f"missing or bad field ({type(e).__name__}: {e}). "
                    "Each mode needs at least: name, layout.width, layout.height, "
                    "layout.line_seconds and layout.channels."
                ) from e

            self.add_custom_mode(
                name=name,
                vis_code=item.get("vis_code"),
                pixel_seconds=item.get("pixel_seconds"),
                sync_seconds=item.get("sync_seconds", layout.sync_seconds),
                porch_seconds=item.get("porch_seconds", 0.0),
                layout=layout,
                overwrite=overwrite,
            )

            loaded.append(name)

        return loaded

    def _require_mode_name(self, name):
        if name not in self.modes:
            raise UnknownModeError(name, sorted(self.modes))

    def _require_layout_name(self, name):
        if name not in self.image_layouts:
            raise UnknownModeError(name, sorted(self.image_layouts))

    def get_mode(self, name):
        self._require_mode_name(name)
        return self.modes[name]

    def get_layout(self, name, line_structure="standard"):
        """
        Layout for a mode, optionally the component-per-line variant.

        line_structure:
            "standard"  one sync per image row (the published SC-2 timing)
            "rgb3"      one sync per colour component, for transmitters
                        that send three syncs per row
        """
        if line_structure in ("rgb3", "component", "component-lines"):
            variant = self.line_structure_variants.get(name)

            if variant is not None:
                return variant

        self._require_layout_name(name)

        return self.image_layouts[name]

    def lookup_vis(self, value):
        return self.vis_codes.get(value)

    def find_vis_code_for_mode(self, mode_name):
        for code, name in sorted(self.vis_codes.items()):
            if name == mode_name:
                return code
        return None

    def supported_encoder_modes(self):
        return sorted(self.image_layouts)

    def supported_decoder_image_modes(self):
        return sorted(self.image_layouts)


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


def prepare_image_for_mode(
        image_path,
        width,
        height,
        fit="contain",
        background=(0, 0, 0),
):
    """
    Resize image to SSTV mode dimensions.

    fit:
      stretch -> force resize, may distort
      contain -> preserve aspect ratio, pad
      cover   -> preserve aspect ratio, crop
    """
    img = Image.open(image_path).convert("RGB")

    fit = fit.lower()

    if fit == "stretch":
        return img.resize((width, height), Image.Resampling.LANCZOS)

    src_w, src_h = img.size
    src_ratio = src_w / src_h
    dst_ratio = width / height

    if fit == "contain":
        if src_ratio > dst_ratio:
            new_w = width
            new_h = round(width / src_ratio)
        else:
            new_h = height
            new_w = round(height * src_ratio)

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (width, height), background)
        x = (width - new_w) // 2
        y = (height - new_h) // 2
        canvas.paste(resized, (x, y))
        return canvas

    if fit == "cover":
        if src_ratio > dst_ratio:
            new_h = height
            new_w = round(height * src_ratio)
        else:
            new_w = width
            new_h = round(width / src_ratio)

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        x = (new_w - width) // 2
        y = (new_h - height) // 2
        return resized.crop((x, y, x + width, y + height))

    raise ValueError("fit must be 'stretch', 'contain', or 'cover'")


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

        if mode_name not in self.registry.image_layouts:
            raise ValueError(
                f"Mode {mode_name!r} cannot currently be encoded.\n"
                f"Supported encoder modes: {self.registry.supported_encoder_modes()}\n"
                "For custom modes, call registry.add_custom_mode(...) first "
                "or load a custom modes JSON file."
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

    def _load_image_rgb(self, image_path):
        L = self.layout

        img = prepare_image_for_mode(
            image_path,
            width=L.width,
            height=L.height,
            fit=getattr(self, "image_fit", "contain"),
            background=getattr(self, "image_background", (0, 0, 0)),
        )

        return np.asarray(img, dtype=np.float64)

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

        return {
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
        """
        rgb = self._load_image_rgb(image_path)
        planes = self._image_to_planes(rgb)

        synth = ToneSynth(
            fs=self.fs,
            amplitude=self.amplitude,
        )

        synth.append_silence(pre_silence)

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

    def __init__(
            self,
            fs,
            samples,
            layout: ImageLayout,
            vis_end,
            freq_offset_hz=0.0,
            slant_search=0.03,
            smoothing_seconds=None,
    ):
        self.fs = fs
        self.samples = np.asarray(samples, dtype=np.float32)
        self.layout = layout
        self.vis_end = vis_end
        self.freq_offset_hz = freq_offset_hz
        self.slant_search = slant_search
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
            center=SSTV_SYNC_HZ,
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

        return a, b, n_used

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

        if L.color == "rgb3":
            order = L.line_channel_order or ("R", "G", "B")
            n_components = len(order)

            _component, off, dur = L.channels[0]

            for tx_line in range(n_lines):
                row = tx_line // n_components

                if row >= L.height:
                    break

                key = order[tx_line % n_components]

                sync_time = a + b * tx_line
                line_start = sync_time - L.sync_offset * r

                px = dur / W * r
                t0 = line_start + off * r + x * px

                freq = self.demod.mean_freq(t0, t0 + px)

                planes[key][row] = self._freq_to_lum(freq)

                self._tally_update(tally, key, freq)

            self.level_tally = tally

            return planes

        for line in range(n_lines):
            sync_time = a + b * line
            line_start = sync_time - L.sync_offset * r

            chans = L.channels

            if L.channels_odd is not None and line % 2 == 1:
                chans = L.channels_odd

            for key, off, dur in chans:
                px = dur / W * r
                t0 = line_start + off * r + x * px

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

    def _assemble(self, planes):
        L = self.layout

        if L.color in ("rgb", "rgb3"):
            rows = L.height if L.color == "rgb3" else None

            rgb = np.dstack([
                planes["R"][:rows],
                planes["G"][:rows],
                planes["B"][:rows],
            ])

        elif L.color == "ycrcb":
            rgb = self._ycrcb_to_rgb(
                planes["Y"],
                planes["Cr"],
                planes["Cb"],
            )

        elif L.color == "ycrcb420":
            n = L.n_lines

            Cr = np.repeat(planes["Cr"][0::2], 2, axis=0)[:n]
            Cb = np.repeat(planes["Cb"][1::2], 2, axis=0)[:n]

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


def _resize_pil(img, size, resample=None):
    if resample is None:
        resample = Image.Resampling.LANCZOS

    return img.resize(size, resample)


def save_raw_jpg(rgb, out_path, quality=95):
    _ensure_parent(out_path)
    Image.fromarray(rgb).save(out_path, "JPEG", quality=quality)
    return out_path


def save_polaroid_jpg(rgb, out_path, caption=None, show_text=True, quality=95):
    _ensure_parent(out_path)

    img = Image.fromarray(rgb)
    w, h = img.size

    side_border = max(24, int(w * 0.065))
    top_border = max(24, int(h * 0.065))
    bottom_border = max(70, int(h * 0.20))

    canvas_w = w + 2 * side_border
    canvas_h = h + top_border + bottom_border

    polaroid = Image.new("RGB", (canvas_w, canvas_h), (248, 248, 244))
    polaroid.paste(img, (side_border, top_border))

    draw = ImageDraw.Draw(polaroid)

    draw.rectangle(
        [
            side_border - 1,
            top_border - 1,
            side_border + w,
            top_border + h,
        ],
        outline=(215, 215, 210),
        width=1,
    )

    if show_text and caption:
        text = str(caption)

        font_size = max(16, int(w * 0.035))
        font = _font(font_size)
        max_text_w = canvas_w - 2 * side_border - 8

        while font_size > 10:
            bbox = draw.textbbox((0, 0), text, font=font)
            if (bbox[2] - bbox[0]) <= max_text_w:
                break
            font_size -= 1
            font = _font(font_size)

        bbox = draw.textbbox((0, 0), text, font=font)

        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        tx = (canvas_w - tw) // 2
        ty = top_border + h + (bottom_border - th) // 2 - 4

        draw.text((tx, ty), text, fill=(45, 45, 45), font=font)

    polaroid.save(out_path, "JPEG", quality=quality)
    return out_path


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


def make_spectrogram_image(
        samples,
        fs,
        t_start,
        t_end,
        width,
        height=220,
        f_min=900.0,
        f_max=2600.0,
):
    n0 = max(0, int(round(t_start * fs)))
    n1 = min(len(samples), int(round(t_end * fs)))

    if n1 <= n0 + 8:
        return Image.new("RGB", (width, height), (0, 0, 0))

    x = samples[n0:n1].astype(np.float64)

    nperseg = min(2048, len(x))

    if nperseg >= 64:
        nperseg = 2 ** int(np.floor(np.log2(nperseg)))
    else:
        nperseg = len(x)

    noverlap = min(int(nperseg * 0.75), nperseg - 1)

    f, tt, Sxx = spectrogram(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )

    band = (f >= f_min) & (f <= f_max)

    if not np.any(band):
        return Image.new("RGB", (width, height), (0, 0, 0))

    S = Sxx[band, :]
    db = 20.0 * np.log10(S + 1e-12)

    lo = np.percentile(db, 5)
    hi = np.percentile(db, 99.5)

    norm = (db - lo) / max(hi - lo, 1e-12)
    norm = np.clip(norm, 0.0, 1.0)

    norm = np.flipud(norm)

    rgb = _simple_inferno_colormap(norm)

    spec = Image.fromarray(rgb)
    spec = _resize_pil(spec, (width, height), Image.Resampling.BICUBIC)

    return spec


def save_image_with_spectrogram_jpg(
        rgb,
        samples,
        fs,
        t_start,
        t_end,
        out_path,
        quality=95,
        spec_height=None,
        caption="SSTV audio spectrogram",
        show_text=True,
):
    _ensure_parent(out_path)

    img = Image.fromarray(rgb)
    w, h = img.size

    if spec_height is None:
        spec_height = max(160, int(h * 0.35))

    spec = make_spectrogram_image(
        samples=samples,
        fs=fs,
        t_start=t_start,
        t_end=t_end,
        width=w,
        height=spec_height,
    )

    gap = 10
    label_h = 28 if show_text else 0

    font = _font(16)

    caption_text = str(caption) if caption is not None else ""
    hz_text = "900-2600 Hz"

    side_margin = 10
    label_gap = 24

    measure = ImageDraw.Draw(img)
    caption_box = measure.textbbox((0, 0), caption_text, font=font)
    caption_w = caption_box[2] - caption_box[0]
    hz_box = measure.textbbox((0, 0), hz_text, font=font)
    hz_w = hz_box[2] - hz_box[0]

    if show_text:
        needed_w = side_margin + caption_w + label_gap + hz_w + side_margin
    else:
        needed_w = w

    canvas_w = max(w, needed_w)
    x_offset = (canvas_w - w) // 2

    canvas_h = h + gap + label_h + spec_height

    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))

    canvas.paste(img, (x_offset, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, h, canvas_w, h + gap], fill=(245, 245, 245))

    spec_y = h + gap + label_h

    if show_text:
        label_y = h + gap + 6

        draw.text((side_margin, label_y), caption_text, fill=(235, 235, 235), font=font)
        draw.text(
            (canvas_w - side_margin - hz_w, label_y),
            hz_text,
            fill=(180, 180, 180),
            font=font,
        )

    canvas.paste(spec, (x_offset, spec_y))
    canvas.save(out_path, "JPEG", quality=quality)

    return out_path


def save_stacked_sstv_spectrogram_jpg(
        rgb,
        samples,
        fs,
        t_start,
        t_end,
        out_path,
        quality=95,
        spec_height=None,
):
    _ensure_parent(out_path)

    img = Image.fromarray(rgb)
    w, h = img.size

    if spec_height is None:
        spec_height = max(160, int(h * 0.35))

    spec = make_spectrogram_image(
        samples=samples,
        fs=fs,
        t_start=t_start,
        t_end=t_end,
        width=w,
        height=spec_height,
    )

    canvas = Image.new("RGB", (w, h + spec_height), (0, 0, 0))
    canvas.paste(img, (0, 0))
    canvas.paste(spec, (0, h))

    canvas.save(out_path, "JPEG", quality=quality)
    return out_path


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

        vis_end = vis_start + p.vis_header_seconds
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

            sync_sanity = self._mode_sync_sanity(
                mode_name,
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
                self.vis_end = self.vis_start + p.vis_header_seconds
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


def make_registry(custom_modes_json=None):
    registry = ModeRegistry()

    if custom_modes_json:
        if isinstance(custom_modes_json, (str, os.PathLike)):
            custom_modes_json = [custom_modes_json]

        for path in custom_modes_json:
            registry.load_custom_modes_json(path, overwrite=True)

    return registry


def decode_audio_to_images(
        audio_path,
        output_base,
        output_mode="all",
        slant_search=0.03,
        registry=None,
        protocol=None,
        custom_modes_json=None,
        forced_mode_name=None,
        forced_vis_end=None,
        first_sync_time=None,
        freq_offset_hz=0.0,
        line_structure="auto",
):
    """
    Decode SSTV audio file into one or more JPEG image files.

    Returns:
        {
            "detect_mode": {...},
            "decode_image": {...}
        }
    """
    registry = registry or make_registry(custom_modes_json)
    protocol = protocol or ProtocolConstants()

    decoder = SSTVDecoder(audio_path, registry=registry, protocol=protocol)

    if forced_mode_name:
        detect = decoder.force_mode(
            forced_mode_name,
            vis_end=forced_vis_end,
            first_sync_time=first_sync_time,
            freq_offset_hz=freq_offset_hz,
        )
    else:
        detect = decoder.detect_mode()

    result = {
        "detect_mode": detect,
        "decode_image": None,
    }

    if "error" in detect:
        return result

    image = decoder.decode_image(
        output_base,
        slant_search=slant_search,
        output_mode=output_mode,
        line_structure=line_structure,
    )

    result["decode_image"] = image
    return result


def encode_image_to_sstv_audio(
        image_path,
        wav_out=None,
        mp3_out=None,
        mode_name="PD-180",
        sample_rate=48000,
        amplitude=0.80,
        mp3_bitrate="320k",
        registry=None,
        protocol=None,
        custom_modes_json=None,
        vis_code=None,
        include_vis=True,
        pre_silence=0.50,
        post_silence=0.50,
        fail_on_mp3_error=False,
        image_fit="contain",
        image_background=(0, 0, 0),
        line_structure="standard",
):
    """
    Encode an image into SSTV audio.

    line_structure:
        standard   one sync per image row (default)
        rgb3       one sync per colour component, three transmitted lines
                   per image row, for transmitters that work that way

    WAV is recommended.
    MP3 is supported only if pydub + ffmpeg are installed, but MP3 is lossy.

    image_fit:
        contain (default) -> keep aspect ratio, pad with image_background
        cover             -> keep aspect ratio, crop the overflow
        stretch           -> old behavior, distort to fill the mode
    """
    registry = registry or make_registry(custom_modes_json)
    protocol = protocol or ProtocolConstants()

    encoder = SSTVEncoder(
        registry=registry,
        mode_name=mode_name,
        fs=sample_rate,
        amplitude=amplitude,
        protocol=protocol,
        vis_code=vis_code,
        image_fit=image_fit,
        image_background=image_background,
        line_structure=line_structure,
    )

    return encoder.encode_image_file(
        image_path=image_path,
        wav_out=wav_out,
        mp3_out=mp3_out,
        mp3_bitrate=mp3_bitrate,
        pre_silence=pre_silence,
        post_silence=post_silence,
        include_vis=include_vis,
        fail_on_mp3_error=fail_on_mp3_error,
    )


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


def _replace_ext(path, image_format):
    base, _ = os.path.splitext(path)
    image_format = image_format.lower().lstrip(".")
    return base + "." + image_format


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


def save_rgb_image(rgb, path, image_format=None, quality=95):
    return save_image_file(Image.fromarray(rgb), path, image_format=image_format, quality=quality)


def auto_levels_rgb(rgb, low_percentile=1.0, high_percentile=99.0):
    """
    Simple robust contrast stretch.
    """
    arr = np.asarray(rgb, dtype=np.float64)
    out = np.empty_like(arr)

    for c in range(3):
        ch = arr[..., c]
        lo = np.percentile(ch, low_percentile)
        hi = np.percentile(ch, high_percentile)
        out[..., c] = (ch - lo) * 255.0 / max(hi - lo, 1e-12)

    return np.clip(out, 0, 255).astype(np.uint8)


def denoise_rgb(rgb, preset="off"):
    """
    Lightweight post-decode denoise.

    Presets:
      off
      light
      medium
      strong
    """
    preset = (preset or "off").lower()

    if preset == "off":
        return rgb

    try:
        from scipy.ndimage import median_filter, gaussian_filter
    except Exception:
        return rgb

    arr = np.asarray(rgb, dtype=np.float64)

    if preset == "light":
        out = median_filter(arr, size=(1, 3, 1))
    elif preset == "medium":
        out = median_filter(arr, size=(1, 5, 1))
        out = gaussian_filter(out, sigma=(0.35, 0.35, 0))
    elif preset == "strong":
        out = median_filter(arr, size=(2, 5, 1))
        out = gaussian_filter(out, sigma=(0.55, 0.55, 0))
    else:
        raise ValueError("denoise preset must be off, light, medium, or strong")

    return np.clip(out, 0, 255).astype(np.uint8)


def add_caption_to_image(
        image_path,
        out_path=None,
        text="",
        position="bottom",
        font_size=None,
        text_color=(255, 255, 255),
        bg_color=(0, 0, 0),
        padding=12,
):
    """
    Add a caption overlay to an image before encoding.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if out_path is None:
        base, ext = os.path.splitext(image_path)
        out_path = base + "_captioned" + (ext or ".jpg")

    draw = ImageDraw.Draw(img)

    if font_size is None:
        font_size = max(18, int(w * 0.04))

    font = _font(font_size)

    text = str(text)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    bar_h = th + 2 * padding

    position = position.lower()

    if position == "top":
        y0 = 0
        y_text = padding
    elif position == "bottom":
        y0 = h - bar_h
        y_text = h - bar_h + padding
    else:
        raise ValueError("caption position must be top or bottom")

    draw.rectangle([0, y0, w, y0 + bar_h], fill=bg_color)
    draw.text(((w - tw) // 2, y_text), text, fill=text_color, font=font)

    save_image_file(img, out_path)
    return out_path


def make_testcard(
        out_path,
        width=640,
        height=496,
        title="nSSTV Test Card",
        subtitle=None,
        image_format=None,
):
    """
    Generate a simple SSTV test card image.
    """
    img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    colors = [
        (255, 255, 255),
        (255, 255, 0),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 255),
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 0),
    ]

    bar_h = int(height * 0.22)
    bar_w = width // len(colors)

    for i, color in enumerate(colors):
        x0 = i * bar_w
        x1 = width if i == len(colors) - 1 else (i + 1) * bar_w
        draw.rectangle([x0, 0, x1, bar_h], fill=color)

    ramp_y0 = bar_h + 12
    ramp_h = int(height * 0.12)

    for x in range(width):
        v = int(255 * x / max(width - 1, 1))
        draw.line([x, ramp_y0, x, ramp_y0 + ramp_h], fill=(v, v, v))

    grid_y0 = ramp_y0 + ramp_h + 20
    for x in range(0, width, max(width // 16, 1)):
        draw.line([x, grid_y0, x, height], fill=(70, 70, 70))
    for y in range(grid_y0, height, max(height // 16, 1)):
        draw.line([0, y, width, y], fill=(70, 70, 70))

    cx, cy = width // 2, int(height * 0.62)
    radius = min(width, height) // 7

    for r in range(radius, 0, -radius // 5 if radius >= 5 else 1):
        color = (230, 230, 230) if (r // max(radius // 5, 1)) % 2 == 0 else (30, 30, 30)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)

    font_big = _font(max(24, width // 18))
    font_small = _font(max(16, width // 32))

    title = str(title)
    bbox = draw.textbbox((0, 0), title, font=font_big)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, int(height * 0.80)), title, fill=(255, 255, 255), font=font_big)

    if subtitle is None:
        subtitle = f"{width}x{height} | Made with nSSTV"

    subtitle = str(subtitle)
    bbox = draw.textbbox((0, 0), subtitle, font=font_small)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, int(height * 0.89)), subtitle, fill=(220, 220, 220), font=font_small)

    draw.rectangle([0, 0, width - 1, height - 1], outline=(255, 255, 255), width=3)

    return save_image_file(img, out_path, image_format=image_format)


def make_testcard_for_mode(out_path, mode_name="PD-180", registry=None, title=None, subtitle=None):
    registry = registry or ModeRegistry()
    if mode_name not in registry.image_layouts:
        raise ValueError(f"No image layout for mode {mode_name!r}")

    layout = registry.get_layout(mode_name)

    return make_testcard(
        out_path=out_path,
        width=layout.width,
        height=layout.height,
        title=title or f"nSSTV Test Card - {mode_name}",
        subtitle=subtitle,
    )


def estimate_decode_quality(detect_info, image_info):
    """
    Return a simple 0..1 quality estimate from VIS/header + sync lock metadata.
    """
    if not detect_info or "error" in detect_info:
        return {
            "overall": 0.0,
            "vis_confidence": 0.0,
            "sync_confidence": 0.0,
            "frequency_stability": 0.0,
            "notes": ["mode detection failed"],
        }

    notes = []

    leader_check = detect_info.get("leader_check", {}) or {}

    leader_ok = bool(leader_check.get("ok", False))
    vis_conf = 1.0 if leader_ok else 0.45

    start_ratio = float(leader_check.get("start_bit_1200_ratio", 1.0) or 1.0)
    stop_ratio = float(leader_check.get("stop_bit_1200_ratio", 1.0) or 1.0)

    vis_ratio_score = min(1.0, math.log(max(start_ratio * stop_ratio, 1.0)) / math.log(100.0))
    vis_conf = 0.65 * vis_conf + 0.35 * vis_ratio_score

    if image_info is None or "error" in image_info:
        return {
            "overall": 0.25 * vis_conf,
            "vis_confidence": float(vis_conf),
            "sync_confidence": 0.0,
            "frequency_stability": 0.0,
            "notes": ["image decode failed"],
        }

    sync_lock_ok = bool(image_info.get("sync_lock_ok", False))
    detected = int(image_info.get("sync_pulses_detected", 0) or 0)
    used = int(image_info.get("sync_pulses_used_in_fit", 0) or 0)

    sync_ratio = used / max(detected, 1)
    sync_conf = min(1.0, sync_ratio * 1.5)

    if sync_lock_ok:
        sync_conf = max(sync_conf, 0.75)
    else:
        notes.append("sync lock weak")

    lines_decoded = image_info.get("lines_decoded")
    lines_expected = image_info.get("lines_expected") or 0

    if lines_decoded is not None and lines_expected:
        fraction = float(lines_decoded) / float(lines_expected)

        if fraction < 0.999:
            sync_conf *= fraction
            notes.append(
                "partial: %d/%d lines, audio ends early"
                % (int(lines_decoded), int(lines_expected))
            )

    levels = image_info.get("levels") or {}
    pinned = []
    worst_pinned = 0.0

    for key in sorted(levels):
        item = levels[key] or {}

        above = float(item.get("above_white_fraction", 0.0) or 0.0)
        below = float(item.get("below_black_fraction", 0.0) or 0.0)

        if above > 0.5:
            pinned.append("%s above 2300 Hz on %.0f%% of pixels" % (key, 100.0 * above))
            worst_pinned = max(worst_pinned, above)
        elif below > 0.5:
            pinned.append("%s below 1500 Hz on %.0f%% of pixels" % (key, 100.0 * below))
            worst_pinned = max(worst_pinned, below)

    if pinned:
        notes.append(
            "tones outside the SSTV 1500-2300 Hz range: "
            + ", ".join(pinned)
            + "; this is not a readable SSTV picture, so check the sample "
              "rate, the sideband and the tuning of the recording"
        )

    flat = []

    for key in sorted(levels):
        spread = float((levels[key] or {}).get("std_level", 0.0) or 0.0)

        if spread < 2.5:
            flat.append("%s is almost constant (std %.1f of 255 levels)" % (key, spread))

    if flat:
        notes.append(
            "no picture in this audio: " + ", ".join(flat)
            + "; the scan lines are not carrying image data"
        )

    nominal = float(image_info.get("nominal_line_seconds", 1.0) or 1.0)
    measured = float(image_info.get("measured_line_seconds", nominal) or nominal)
    slant_err = abs(measured / nominal - 1.0)

    frequency_stability = max(0.0, 1.0 - slant_err / 0.03)

    overall = (
            0.35 * vis_conf +
            0.45 * sync_conf +
            0.20 * frequency_stability
    )

    if pinned:
        overall *= max(0.0, 1.0 - 0.9 * worst_pinned)

    if flat:
        overall *= 0.1

    return {
        "overall": float(np.clip(overall, 0.0, 1.0)),
        "vis_confidence": float(np.clip(vis_conf, 0.0, 1.0)),
        "sync_confidence": float(np.clip(sync_conf, 0.0, 1.0)),
        "frequency_stability": float(np.clip(frequency_stability, 0.0, 1.0)),
        "sync_pulses_detected": detected,
        "sync_pulses_used": used,
        "slant_error_fraction": float(slant_err),
        "lines_decoded": lines_decoded,
        "lines_expected": lines_expected or None,
        "partial": bool(
            lines_decoded is not None
            and lines_expected
            and lines_decoded < lines_expected
        ),
        "levels": levels,
        "notes": notes,
    }


def write_json(path, data):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


def save_metadata_sidecar(image_path, metadata):
    base, _ = os.path.splitext(image_path)
    return write_json(base + ".json", metadata)


def save_decode_report(report_path, report):
    return write_json(report_path, report)


def make_diagnostics_image(
        samples,
        fs,
        out_path,
        t_start=0.0,
        t_end=None,
        markers=None,
        title="nSSTV diagnostics",
        width=1200,
        height=520,
):
    """
    Diagnostic spectrogram with optional vertical markers.
    """
    if t_end is None:
        t_end = len(samples) / fs

    spec = make_spectrogram_image(
        samples=samples,
        fs=fs,
        t_start=t_start,
        t_end=t_end,
        width=width,
        height=height - 60,
    )

    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    canvas.paste(spec, (0, 60))

    draw = ImageDraw.Draw(canvas)

    font = _font(18)
    small = _font(13)

    draw.text((12, 12), title, fill=(245, 245, 245), font=font)
    draw.text((12, 36), f"{t_start:.3f}s to {t_end:.3f}s | 900-2600 Hz", fill=(180, 180, 180), font=small)

    markers = markers or []

    duration = max(t_end - t_start, 1e-12)

    for marker in markers:
        mt = float(marker.get("time", 0.0))
        label = str(marker.get("label", ""))
        color = marker.get("color", (255, 255, 255))

        if mt < t_start or mt > t_end:
            continue

        x = int(round((mt - t_start) / duration * (width - 1)))
        draw.line([x, 60, x, height - 1], fill=color, width=2)

        if label:
            draw.text((min(x + 4, width - 120), 62), label, fill=color, font=small)

    return save_image_file(canvas, out_path)


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


def _save_outputs_from_rgb(
        rgb,
        decoder,
        layout,
        img_info,
        out_path,
        output_mode="all",
        image_format="jpg",
        quality=95,
        metadata=None,
        write_sidecar=False,
):
    """
    Extended output saver with PNG/JPG and sidecars.
    """
    base, _ = os.path.splitext(out_path)
    ext = "." + image_format.lower().lstrip(".")

    if output_mode not in OUTPUT_IMAGE_MODES:
        return {
            "error": f"Unknown output_mode {output_mode!r}",
            "valid_output_modes": list(OUTPUT_IMAGE_MODES),
        }

    mode_name = decoder.mode.name if decoder.mode is not None else layout.name
    vis_text = f"VIS {decoder.vis_value}" if getattr(decoder, "vis_value", None) is not None else "VIS ?"
    caption = f"{mode_name} | {vis_text}"

    measured_line_seconds = img_info.get("measured_line_seconds", layout.line_seconds)
    first_sync = img_info.get("first_line_sync", decoder.vis_end + layout.leadin_seconds)
    slant_ratio = measured_line_seconds / layout.line_seconds

    image_audio_start = max(
        0.0,
        first_sync - layout.sync_offset * slant_ratio - 0.05,
    )

    image_audio_end = min(
        decoder.total_duration,
        image_audio_start + layout.n_lines * measured_line_seconds + 0.20,
    )

    paths = {}

    def _path(name):
        if output_mode == name:
            return base + ext
        if output_mode == "all":
            if name == "raw":
                return f"{base}_raw{ext}"
            return f"{base}_{name}{ext}"
        return base + ext

    if output_mode in ("raw", "all"):
        p = _path("raw")
        save_rgb_image(rgb, p, image_format=image_format, quality=quality)
        paths["raw"] = p

    def _paint(key, painter):
        p = _path(key)
        painter(p)
        paths[key] = _resave_as_png(p, image_format)

    if output_mode in ("polaroid", "polaroid_text", "all"):
        _paint("polaroid_text", lambda p: save_polaroid_jpg(
            rgb, p, caption=caption, show_text=True, quality=quality))

    if output_mode in ("polaroid_notext", "all"):
        _paint("polaroid_notext", lambda p: save_polaroid_jpg(
            rgb, p, caption=None, show_text=False, quality=quality))

    def _spectrogram(p, show_text):
        save_image_with_spectrogram_jpg(
            rgb=rgb,
            samples=decoder.samples,
            fs=decoder.fs,
            t_start=image_audio_start,
            t_end=image_audio_end,
            out_path=p,
            caption=f"{caption} audio spectrogram" if show_text else None,
            show_text=show_text,
            quality=quality,
        )

    if output_mode in ("spectrogram", "spectrogram_text", "all"):
        _paint("spectrogram_text", lambda p: _spectrogram(p, show_text=True))

    if output_mode in ("spectrogram_notext", "all"):
        _paint("spectrogram_notext", lambda p: _spectrogram(p, show_text=False))

    if output_mode in ("stack", "all"):
        _paint("stack", lambda p: save_stacked_sstv_spectrogram_jpg(
            rgb=rgb,
            samples=decoder.samples,
            fs=decoder.fs,
            t_start=image_audio_start,
            t_end=image_audio_end,
            out_path=p,
            quality=quality,
        ))

    if write_sidecar:
        for style_name, path in paths.items():
            sidecar = dict(metadata or {})
            sidecar["style"] = style_name
            sidecar["image_path"] = path
            save_metadata_sidecar(path, sidecar)

    return {
        "paths": paths,
        "output_mode": output_mode,
        "image_format": image_format,
        "spectrogram_audio_start": float(image_audio_start),
        "spectrogram_audio_end": float(image_audio_end),
        **img_info,
    }


def decode_image_extended(
        decoder,
        out_path,
        slant_search=0.03,
        output_mode="all",
        image_format="jpg",
        quality=95,
        auto_levels=False,
        denoise="off",
        write_sidecar=False,
        extra_metadata=None,
        smoothing_seconds=None,
        line_structure="auto",
):
    """
    Extended image decode wrapper.

    This does not require replacing SSTVDecoder.decode_image.

    line_structure:
        auto       choose from the audio (default)
        standard   one sync per image row
        rgb3       one sync per colour component
    """
    if decoder.mode is None:
        raise RuntimeError("call detect_mode() or force_mode() first")

    if decoder.mode.name not in decoder.registry.image_layouts:
        return {
            "error": f"No image layout defined for {decoder.mode.name}",
            "mode_name": decoder.mode.name,
            "supported_image_layout_modes": decoder.registry.supported_decoder_image_modes(),
        }

    decoder.line_structure = decoder._resolve_line_structure(line_structure)

    layout = decoder.registry.get_layout(
        decoder.mode.name,
        line_structure=decoder.line_structure,
    )

    img = ImageDecoder(
        decoder.fs,
        decoder.samples,
        layout,
        decoder.vis_end,
        freq_offset_hz=decoder.freq_offset_hz,
        slant_search=slant_search,
        smoothing_seconds=smoothing_seconds,
    )

    rgb = img.decode()

    if rgb is None:
        return {
            "error": img.info.get("error", "image decode failed"),
            "mode_name": decoder.mode.name,
            "info": dict(img.info),
            "quality": estimate_decode_quality(
                {"mode_name": decoder.mode.name, "leader_check": decoder.leader_check},
                img.info,
            ),
        }

    if denoise and denoise.lower() != "off":
        rgb = denoise_rgb(rgb, denoise)

    if auto_levels:
        rgb = auto_levels_rgb(rgb)

    detect_info = {
        "mode_name": decoder.mode.name,
        "vis_value": decoder.vis_value,
        "freq_offset_hz": decoder.freq_offset_hz,
        "smoothing_seconds": img.smoothing_seconds,
        "smoothing_samples": img.demod.smoothing_samples,
        "fastest_pixel_seconds": layout.fastest_pixel_seconds,
        "leader_check": decoder.leader_check,
    }

    quality_info = estimate_decode_quality(detect_info, img.info)

    metadata = {
        "library": "nSSTV",
        "version": __version__,
        "source_audio": getattr(decoder, "audio_path", None),
        "mode_name": decoder.mode.name,
        "vis_code": decoder.vis_value,
        "freq_offset_hz": decoder.freq_offset_hz,
        "leader_check": decoder.leader_check,
        "decode_info": img.info,
        "quality": quality_info,
        "smoothing_seconds": img.smoothing_seconds,
        "smoothing_samples": img.demod.smoothing_samples,
        "fastest_pixel_seconds": layout.fastest_pixel_seconds,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    result = _save_outputs_from_rgb(
        rgb=rgb,
        decoder=decoder,
        layout=layout,
        img_info=img.info,
        out_path=out_path,
        output_mode=output_mode,
        image_format=image_format,
        quality=quality,
        metadata=metadata,
        write_sidecar=write_sidecar,
    )

    result["quality"] = quality_info
    result["smoothing_seconds"] = img.smoothing_seconds
    result["smoothing_samples"] = img.demod.smoothing_samples
    result["fastest_pixel_seconds"] = layout.fastest_pixel_seconds

    return result


def _hint_modes_from_detect(detect):
    """Mode names the measured line rate vouches for, best first."""
    lr = detect.get("line_rate") or {}
    return [c["mode_name"] for c in (lr.get("line_rate_candidates") or [])
            if c.get("mode_name")]


def _single_confident_hint_mode(detect, min_confidence=0.9):
    """
    The one experimental mode name to retry with, or None.

    Auto-retry is limited to the experimental modes (Pasokon P3/P5/P7,
    PD-50): their off-air recordings are exactly the case that motivated
    the line-rate hint - VIS header too noisy to score, image otherwise
    decodable - and the result is flagged experimental either way.
    Established modes keep the old contract: the hint is reported and the
    caller decides. Pairs that share a line time (Martin M1/M3, Scottie
    S1/S3, ...) never trigger a retry either: two candidates means the
    caller still gets the hint but no guess.
    """
    lr = detect.get("line_rate") or {}
    cands = lr.get("line_rate_candidates") or []

    if len(cands) == 1 and float(lr.get("line_rate_confidence") or 0.0) >= min_confidence:
        name = cands[0].get("mode_name")

        if name and is_experimental_mode(name):
            return name

    return None


def decode_audio_to_images_v2(
        audio_path,
        output_base,
        output_mode="all",
        image_format="jpg",
        slant_search=0.03,
        registry=None,
        protocol=None,
        custom_modes_json=None,
        forced_mode_name=None,
        forced_vis_end=None,
        first_sync_time=None,
        freq_offset_hz=0.0,
        auto_levels=False,
        denoise="off",
        write_sidecar=True,
        diagnostics=False,
        diagnostics_path=None,
        report_path=None,
        smoothing_seconds=None,
        line_structure="auto",
):
    """
    Improved decode API with PNG, sidecar, diagnostics, quality score.
    """
    registry = registry or make_registry(custom_modes_json)
    protocol = protocol or ProtocolConstants()

    decoder = SSTVDecoder(audio_path, registry=registry, protocol=protocol)

    if forced_mode_name:
        detect = decoder.force_mode(
            forced_mode_name,
            vis_end=forced_vis_end,
            first_sync_time=first_sync_time,
            freq_offset_hz=freq_offset_hz,
        )
    else:
        detect = decoder.detect_mode()

        if "error" in detect:
            hint_modes = _hint_modes_from_detect(detect)

            if hint_modes:
                detect["hint_modes"] = hint_modes

                exp_named = [m for m in hint_modes if is_experimental_mode(m)]

                if exp_named:
                    detect["experimental"] = True
                    detect["notice"] = experimental_notice(exp_named[0])

                retry = _single_confident_hint_mode(detect)

                if retry:
                    detect = decoder.force_mode(
                        retry,
                        vis_end=forced_vis_end,
                        first_sync_time=first_sync_time,
                        freq_offset_hz=freq_offset_hz,
                    )
                    detect["line_rate_hint_used"] = True
                    detect["hint_modes"] = hint_modes

                    if detect.get("notice"):
                        detect["notice"] = (
                            "VIS unreadable; the measured sync interval "
                            "matches %s. %s" % (retry, detect["notice"])
                        )

    result = {
        "detect_mode": detect,
        "decode_image": None,
        "raw": None,
        "mode": None,
        "experimental": bool(detect.get("experimental")),
        "notice": detect.get("notice"),
        "partial": False,
        "lines_decoded": 0,
        "lines_expected": 0,
        "quality": 0.0,
        "diagnostics_path": None,
        "report_path": None,
    }

    if detect.get("line_rate_hint_used"):
        result["mode_source"] = "line-rate-hint"

    if "error" in detect:
        if report_path:
            result["report_path"] = save_decode_report(report_path, result)
        return result

    image = decode_image_extended(
        decoder=decoder,
        out_path=output_base,
        slant_search=slant_search,
        output_mode=output_mode,
        image_format=image_format,
        auto_levels=auto_levels,
        denoise=denoise,
        write_sidecar=write_sidecar,
        smoothing_seconds=smoothing_seconds,
        line_structure=line_structure,
    )

    if isinstance(image, dict):
        for key in ("smoothing_seconds", "smoothing_samples", "fastest_pixel_seconds"):
            if key in image:
                detect[key] = image[key]

    result["decode_image"] = image

    if isinstance(image, dict) and "error" not in image:
        result["raw"] = (image.get("paths") or {}).get("raw")
        result["mode"] = detect.get("mode_name")
        result["partial"] = bool(image.get("partial", False))
        result["lines_decoded"] = image.get("lines_decoded")
        result["lines_expected"] = image.get("lines_expected")
        result["quality"] = float((image.get("quality") or {}).get("overall", 0.0))

    if diagnostics:
        if diagnostics_path is None:
            base, _ = os.path.splitext(output_base)
            diagnostics_path = base + "_diagnostics.jpg"

        markers = []

        if decoder.leader_start is not None:
            markers.append({"time": decoder.leader_start, "label": "leader", "color": (0, 255, 255)})
        if decoder.vis_start is not None:
            markers.append({"time": decoder.vis_start, "label": "VIS", "color": (255, 255, 0)})
        if decoder.vis_end is not None:
            markers.append({"time": decoder.vis_end, "label": "image", "color": (0, 255, 0)})

        t0 = max(0.0, (decoder.leader_start or 0.0) - 0.5)
        t1 = min(decoder.total_duration, t0 + 10.0)

        result["diagnostics_path"] = make_diagnostics_image(
            samples=decoder.samples,
            fs=decoder.fs,
            out_path=diagnostics_path,
            t_start=t0,
            t_end=t1,
            markers=markers,
            title=f"nSSTV diagnostics | {detect.get('mode_name')}",
        )

    if report_path:
        result["report_path"] = save_decode_report(report_path, result)

    return result


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


def decode_all_audio_to_images(
        audio_path,
        out_dir=None,
        output_mode="raw",
        image_format="png",
        slant_search=0.03,
        registry=None,
        protocol=None,
        custom_modes_json=None,
        auto_levels=False,
        denoise="off",
        write_sidecar=True,
        diagnostics=False,
        min_gap_seconds=1.0,
        line_structure="auto",
        forced_mode_name=None,
        report_path=None,
):
    """
    Decode every SSTV transmission found in one recording.

    forced_mode_name skips VIS detection: every leader candidate is decoded
    as that mode, assuming a standard VIS header sits between the leader and
    the image.

    report_path writes the summary JSON there in addition to the automatic
    <audio>_summary.json next to the images.

    Returns a dict with a list under "images".
    """
    registry = registry or make_registry(custom_modes_json)
    protocol = protocol or ProtocolConstants()

    decoder = SSTVDecoder(audio_path, registry=registry, protocol=protocol)

    if out_dir is None:
        base_dir = os.path.dirname(os.path.abspath(audio_path)) or "."
        audio_base = os.path.splitext(os.path.basename(audio_path))[0]
        out_dir = os.path.join(base_dir, audio_base + "_decoded")

    os.makedirs(out_dir, exist_ok=True)

    candidates = decoder._coarse_leader_candidates()

    images = []
    skipped = []
    skip_until = -1.0

    audio_base = _safe_filename_part(os.path.splitext(os.path.basename(audio_path))[0])

    forced_vis_value = None

    if forced_mode_name:
        if forced_mode_name not in registry.modes:
            raise ValueError(f"Unknown mode {forced_mode_name!r}")

        if forced_mode_name not in registry.image_layouts:
            raise ValueError(f"No image layout for mode {forced_mode_name!r}")

        forced_vis_value = registry.find_vis_code_for_mode(forced_mode_name)

    for cand in candidates:
        if cand["region_start"] < skip_until:
            skipped.append({"reason": "inside previous transmission", **cand})
            continue

        if forced_mode_name:
            fine = decoder._fine_leader_from_candidate(cand)

            if fine is None:
                skipped.append({"reason": "candidate leader could not be refined", **cand})
                continue

            vis_info = {
                "vis_start": fine["leader_end"],
                "vis_value": forced_vis_value,
                "mode_name": forced_mode_name,
            }
            check = {"ok": True, "forced_mode": True}
        else:
            header = _detect_header_from_candidate(decoder, cand)

            if header is None:
                skipped.append({"reason": "candidate failed VIS/header validation", **cand})
                continue

            fine = header["fine"]
            vis_info = header["vis_info"]
            check = header["leader_check"]

        decoder.freq_offset_hz = cand["offset"]
        decoder.leader_start = fine["leader_start"]
        decoder.leader_end = fine["leader_end"]
        decoder.vis_start = vis_info["vis_start"]
        decoder.vis_end = decoder.vis_start + protocol.vis_header_seconds
        decoder.mode = registry.get_mode(vis_info["mode_name"])
        decoder.vis_value = vis_info["vis_value"]
        decoder.leader_check = check

        experimental_notice_text = experimental_notice(
            decoder.mode.name, context="decode-all"
        )

        mode_safe = _safe_filename_part(decoder.mode.name)
        index = len(images)
        out_base = os.path.join(out_dir, f"{audio_base}_{index:03d}_{mode_safe}.{image_format}")

        image = decode_image_extended(
            decoder=decoder,
            out_path=out_base,
            slant_search=slant_search,
            output_mode=output_mode,
            image_format=image_format,
            auto_levels=auto_levels,
            denoise=denoise,
            write_sidecar=write_sidecar,
            line_structure=line_structure,
            extra_metadata={
                "transmission_index": index,
                "multi_decode": True,
            },
        )

        layout = registry.get_layout(decoder.mode.name)
        measured_line = image.get("measured_line_seconds", layout.line_seconds)
        tx_end = decoder.vis_end + layout.leadin_seconds + layout.n_lines * measured_line + 0.5

        if image.get("spectrogram_audio_end") is not None:
            tx_end = max(tx_end, float(image["spectrogram_audio_end"]))

        skip_until = tx_end + min_gap_seconds

        item = {
            "index": index,
            "mode_name": decoder.mode.name,
            "mode": decoder.mode.name,
            "vis_code": decoder.vis_value,
            "leader_start": decoder.leader_start,
            "vis_start": decoder.vis_start,
            "vis_end": decoder.vis_end,
            "estimated_end": tx_end,
            "freq_offset_hz": decoder.freq_offset_hz,
            "leader_check": check,
            "decode_image": image,
            "quality": image.get("quality"),
            "experimental": experimental_notice_text is not None,
        }

        if experimental_notice_text:
            item["experimental_note"] = experimental_notice_text

        if isinstance(image, dict) and "error" not in image:
            item["raw"] = (image.get("paths") or {}).get("raw")
            item["partial"] = bool(image.get("partial", False))
            item["lines_decoded"] = image.get("lines_decoded")
            item["lines_expected"] = image.get("lines_expected")
            item["quality_score"] = float((image.get("quality") or {}).get("overall", 0.0))
            item["starts_at"] = round(float(decoder.leader_start), 3)

        if diagnostics:
            diag_path = os.path.join(out_dir, f"{audio_base}_{index:03d}_{mode_safe}_diagnostics.jpg")
            markers = [
                {"time": decoder.leader_start, "label": "leader", "color": (0, 255, 255)},
                {"time": decoder.vis_start, "label": "VIS", "color": (255, 255, 0)},
                {"time": decoder.vis_end, "label": "image", "color": (0, 255, 0)},
                {"time": tx_end, "label": "end", "color": (255, 0, 255)},
            ]
            item["diagnostics_path"] = make_diagnostics_image(
                decoder.samples,
                decoder.fs,
                diag_path,
                t_start=max(0.0, decoder.leader_start - 0.5),
                t_end=min(decoder.total_duration, tx_end + 0.5),
                markers=markers,
                title=f"nSSTV diagnostics | transmission {index} | {decoder.mode.name}",
            )

        images.append(item)

    summary = {
        "source_audio": audio_path,
        "out_dir": out_dir,
        "candidate_count": len(candidates),
        "decoded_count": len(images),
        "count": len(images),
        "modes": [i["mode_name"] for i in images],
        "paths": [i.get("raw") for i in images],
        "partials": [bool(i.get("partial", False)) for i in images],
        "quality": [i.get("quality_score", 0.0) for i in images],
        "experimental_count": sum(1 for i in images if i.get("experimental")),
        "experimental": [i["mode_name"] for i in images if i.get("experimental")],
        "images": images,
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
    }

    write_json(os.path.join(out_dir, f"{audio_base}_summary.json"), summary)

    if report_path:
        write_json(report_path, summary)

    return summary


def batch_decode(
        input_dir,
        out_dir=None,
        recursive=True,
        output_mode="all",
        image_format="jpg",
        slant_search=0.03,
        custom_modes_json=None,
        auto_levels=False,
        denoise="off",
        write_sidecar=True,
        diagnostics=False,
        continue_on_error=True,
):
    """
    Batch decode all WAV/MP3 files in a directory.
    """
    input_dir = os.path.abspath(os.path.expanduser(input_dir))

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"batch_decode: input directory does not exist: {input_dir}"
        )

    if out_dir is None:
        out_dir = os.path.join(input_dir, "nSSTV_decoded")

    os.makedirs(out_dir, exist_ok=True)

    exts = (".wav", ".wave", ".mp3")

    if recursive:
        found = (
            os.path.join(root, name)
            for root, _, files in os.walk(input_dir)
            for name in files
        )
    else:
        found = (os.path.join(input_dir, name) for name in os.listdir(input_dir))

    audio_files = sorted(
        path for path in found
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in exts
    )

    results = []
    errors = []

    for path in audio_files:
        rel = os.path.relpath(path, input_dir)
        rel_base = os.path.splitext(rel)[0]
        file_out_dir = os.path.join(out_dir, _safe_filename_part(rel_base))

        try:
            info = decode_all_audio_to_images(
                audio_path=path,
                out_dir=file_out_dir,
                output_mode=output_mode,
                image_format=image_format,
                slant_search=slant_search,
                custom_modes_json=custom_modes_json,
                auto_levels=auto_levels,
                denoise=denoise,
                write_sidecar=write_sidecar,
                diagnostics=diagnostics,
            )
            results.append(info)
        except Exception as e:
            err = {
                "audio_path": path,
                "error": str(e),
            }
            errors.append(err)
            if not continue_on_error:
                raise

    summary = {
        "input_dir": input_dir,
        "out_dir": out_dir,
        "file_count": len(audio_files),
        "success_count": len(results),
        "error_count": len(errors),
        "results": results,
        "errors": errors,
    }

    write_json(os.path.join(out_dir, "batch_summary.json"), summary)

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("source_audio,index,mode_name,vis_code,quality_overall,raw_path\n")
        for file_result in results:
            for img in file_result.get("images", []):
                raw_path = ""
                paths = img.get("decode_image", {}).get("paths", {})
                raw_path = paths.get("raw", "")
                q = img.get("quality", {}) or {}
                f.write(
                    f"{file_result.get('source_audio', '')},"
                    f"{img.get('index', '')},"
                    f"{img.get('mode_name', '')},"
                    f"{img.get('vis_code', '')},"
                    f"{q.get('overall', '')},"
                    f"{raw_path}\n"
                )

    summary["csv_path"] = csv_path
    return summary


def encode_image_to_sstv_audio_v2(
        image_path,
        wav_out=None,
        mp3_out=None,
        mode_name="PD-180",
        sample_rate=48000,
        amplitude=0.80,
        mp3_bitrate="320k",
        registry=None,
        protocol=None,
        custom_modes_json=None,
        vis_code=None,
        include_vis=True,
        pre_silence=0.50,
        post_silence=0.50,
        fail_on_mp3_error=False,
        caption=None,
        caption_position="bottom",
        image_fit="contain",
        image_background=(0, 0, 0),
        line_structure="standard",
):
    """
    Improved encode API with optional caption overlay and aspect-ratio fit.
    """
    temp_caption_path = None

    try:
        source_image = image_path

        if caption:
            base, ext = os.path.splitext(os.path.abspath(image_path))
            temp_caption_path = base + "_nsstv_caption_tmp" + (ext or ".jpg")
            source_image = add_caption_to_image(
                image_path=image_path,
                out_path=temp_caption_path,
                text=caption,
                position=caption_position,
            )

        info = encode_image_to_sstv_audio(
            image_path=source_image,
            wav_out=wav_out,
            mp3_out=mp3_out,
            mode_name=mode_name,
            sample_rate=sample_rate,
            amplitude=amplitude,
            mp3_bitrate=mp3_bitrate,
            registry=registry,
            protocol=protocol,
            custom_modes_json=custom_modes_json,
            vis_code=vis_code,
            include_vis=include_vis,
            pre_silence=pre_silence,
            post_silence=post_silence,
            fail_on_mp3_error=fail_on_mp3_error,
            image_fit=image_fit,
            image_background=image_background,
            line_structure=line_structure,
        )

        info["experimental"] = is_experimental_mode(mode_name)

        return info
    finally:
        if temp_caption_path and os.path.exists(temp_caption_path):
            try:
                os.remove(temp_caption_path)
            except Exception:
                pass


def live_record_then_decode(
        seconds=240,
        out_base="live_capture.jpg",
        sample_rate=48000,
        device=None,
        output_mode="all",
        image_format="jpg",
        slant_search=0.03,
        auto_levels=False,
        denoise="off",
        diagnostics=True,
):
    """
    Optional live feature: record microphone/virtual audio, then decode.

    Requires:
        pip install sounddevice
    """
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError(
            "Live capture requires sounddevice.\n"
            "Install with:\n"
            "  pip install sounddevice"
        ) from e

    seconds = float(seconds)
    sample_rate = int(sample_rate)

    print(f"Recording {seconds:.1f}s at {sample_rate} Hz...")
    audio = sd.rec(
        int(round(seconds * sample_rate)),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()

    audio = np.asarray(audio).reshape(-1)
    audio = _normalize_audio(audio)

    wav_path = _replace_ext(out_base, "live_capture.wav")
    write_wav_file(wav_path, sample_rate, audio)

    return decode_audio_to_images_v2(
        audio_path=wav_path,
        output_base=out_base,
        output_mode=output_mode,
        image_format=image_format,
        slant_search=slant_search,
        auto_levels=auto_levels,
        denoise=denoise,
        write_sidecar=True,
        diagnostics=diagnostics,
    )


def decode_all(*args, **kwargs):
    """
    Alias for decode_all_audio_to_images().
    """
    return decode_all_audio_to_images(*args, **kwargs)


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


def cut(audio, seconds, out=None):
    """
    Keep only the first `seconds` of an audio file.

    Makes a partial transmission to test with. Returns the new path.

    Uses the same loader as the decoder, so PCM and float WAV (and MP3)
    all work.
    """
    audio = _abs(audio)
    fs, samples = load_audio_mono(audio)
    n = max(0, int(round(float(seconds) * fs)))

    if out is None:
        out = os.path.splitext(audio)[0] + "_cut.wav"

    return write_audio_file(_abs(out), fs, samples[:n])


def join(audios, out=None, gap=2.0):
    """
    Concatenate audio files into one recording, silence between them.

    Makes a multi-image file to test with. Returns the new path.

    Uses the same loader as the decoder, so PCM and float WAV (and MP3)
    all work. Files must share a sample rate.
    """
    if isinstance(audios, (str, os.PathLike)):
        audios = [audios]

    audios = [_abs(a) for a in audios]

    if not audios:
        raise ValueError("join() needs at least one audio file")

    gap = float(gap)
    if not math.isfinite(gap) or gap < 0:
        raise ValueError(f"gap must be a non-negative number of seconds, got {gap!r}")

    if out is None:
        out = os.path.splitext(audios[0])[0] + "_joined.wav"

    pieces = []
    fs0 = None

    for i, path in enumerate(audios):
        fs, samples = load_audio_mono(path)

        if fs0 is None:
            fs0 = fs
        elif fs != fs0:
            raise ValueError(f"Sample rate mismatch: {path} is {fs} Hz, expected {fs0} Hz.")

        pieces.append(samples)

        if i < len(audios) - 1:
            pieces.append(np.zeros(int(round(gap * fs0)), dtype=np.float64))

    return write_audio_file(_abs(out), fs0, np.concatenate(pieces))


DEFAULT_MODE = "Martin M1"
DEFAULT_OUT_ROOT = "nSSTV_output"
DEFAULT_IMAGE_FORMAT = "png"

RANK_METRIC = "psnr"
WEAK_PSNR_DB = 15.0
WEAK_CORRELATION = 0.80
WEAK_MAE = 30.0

RANDOM_MODE_WORDS = ("random", "rand", "any", "shuffle")
DEFAULT_RANDOM_MODES = 5


def _random_modes(available, count, seed=None):
    """Pick `count` modes at random, without repeats."""
    rng = random.Random(seed)

    count = max(1, min(int(count), len(available)))

    return rng.sample(sorted(available), count)


def random_modes(count=DEFAULT_RANDOM_MODES, seed=None):
    """
    Pick mode names at random. Same seed, same picks.
    """
    return _random_modes(ModeRegistry().supported_encoder_modes(), count, seed)


def _resolve_bench_modes(available, modes, count, seed):
    """
    Turn the modes argument into a list of mode names.

    None               -> every mode
    "random" / 3       -> that many modes picked at random
    "Martin M4" / list -> exactly those

    Returns (modes, was_random).
    """
    if isinstance(modes, int) and not isinstance(modes, bool):
        count = modes
        modes = "random"

    if isinstance(modes, str):
        modes = [modes]

    if modes:
        modes = [str(m) for m in modes]

        if len(modes) == 1 and modes[0].lower() in RANDOM_MODE_WORDS:
            return _random_modes(available, count or DEFAULT_RANDOM_MODES, seed), True

        return modes, False

    return list(available), False


def _abs(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value

    return None


def _rt_out_dir(image_path, out_dir=None, sub=None):
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), DEFAULT_OUT_ROOT, _stem(image_path))
    if sub:
        out_dir = os.path.join(out_dir, _safe_filename_part(sub))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.abspath(out_dir)


FONT_CANDIDATES = (
    "Arial.ttf",
    "Helvetica.ttf",
    "Verdana.ttf",
    "Tahoma.ttf",
    "LiberationSans-Regular.ttf",
    "DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _font(size):
    """
    Best font available at `size`, so captions do not shrink to specks on
    machines without Arial.
    """
    size = max(8, int(size))

    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text_width(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _reference_array(image_path, width, height, fit="contain", background=(0, 0, 0)):
    img = prepare_image_for_mode(
        image_path,
        width=width,
        height=height,
        fit=fit,
        background=background,
    )
    return np.asarray(img, dtype=np.float64)


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


def compare(a, b):
    """
    Compare two image files. Returns mae, psnr, correlation.
    """
    with Image.open(a) as first, Image.open(b) as second:
        if second.size != first.size:
            second = second.resize(first.size, Image.Resampling.LANCZOS)

        reference = np.asarray(first.convert("RGB"), dtype=np.float64)
        other = np.asarray(second.convert("RGB"), dtype=np.float64)

    return _score(reference, other)


def side(a, b, out=None, left="ORIGINAL", right="DECODED FROM SSTV AUDIO", max_height=900):
    """
    Side-by-side PNG: a on the left, b on the right. Canvas widened so the
    labels are never clipped.
    """
    left_img = Image.open(a).convert("RGB")
    right_img = Image.open(b).convert("RGB")

    tallest = max(left_img.height, right_img.height)
    scale = 1.0

    if tallest > max_height:
        scale = max_height / float(tallest)

    def _scale(img):
        if scale != 1.0:
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.Resampling.LANCZOS,
            )
        return img

    left_img = _scale(left_img)
    right_img = _scale(right_img)

    if out is None:
        out = os.path.join(os.getcwd(), "comparison.png")

    _ensure_parent(out)

    gap = 12
    header = 38
    font = _font(18)

    measure = ImageDraw.Draw(left_img)
    left_w = _text_width(measure, left, font)
    right_w = _text_width(measure, right, font)

    left_col = max(left_img.width, left_w + 12)
    right_col = max(right_img.width, right_w + 12)

    canvas = Image.new(
        "RGB",
        (left_col + gap + right_col, max(left_img.height, right_img.height) + header),
        (16, 16, 16),
    )

    draw = ImageDraw.Draw(canvas)
    draw.text((6, 10), left, fill=(235, 235, 235), font=font)
    draw.text((left_col + gap + 6, 10), right, fill=(235, 235, 235), font=font)

    canvas.paste(left_img, ((left_col - left_img.width) // 2, header))
    canvas.paste(right_img, (left_col + gap + (right_col - right_img.width) // 2, header))
    canvas.save(out, "PNG")

    return out


def encode(
        image,
        out=None,
        mode=None,
        fit=None,
        background=None,
        sample_rate=48000,
        amplitude=0.80,
        caption=None,
        mp3=False,
        bitrate=None,
        vis=None,
        pre_silence=0.50,
        post_silence=0.50,
        registry=None,
        wav_out=None,
        mp3_out=None,
        mode_name=None,
        image_fit=None,
        image_background=None,
        mp3_bitrate=None,
        vis_code=None,
        line_structure="standard",
):
    """
    image -> SSTV audio. Out path is automatic when omitted.

    Short names (out, mode, fit, background, bitrate, vis) and long names
    (wav_out, mode_name, image_fit, image_background, mp3_bitrate, vis_code)
    both work.
    """
    image = _abs(image)

    out = _first_not_none(wav_out, out) or (os.path.splitext(image)[0] + "_sstv.wav")
    mode = _first_not_none(mode_name, mode, DEFAULT_MODE)
    fit = _first_not_none(image_fit, fit, "contain")
    background = _first_not_none(image_background, background, (0, 0, 0))
    bitrate = _first_not_none(mp3_bitrate, bitrate, "320k")

    raw_vis = _first_not_none(vis_code, vis)

    if isinstance(raw_vis, bool):
        vis, include_vis = None, raw_vis
    else:
        vis, include_vis = raw_vis, True

    out = _abs(out)

    if mp3_out is None and mp3:
        mp3_out = os.path.splitext(out)[0] + ".mp3"

    mp3_out = _abs(mp3_out) if mp3_out else None

    return encode_image_to_sstv_audio_v2(
        image_path=image,
        wav_out=out,
        mp3_out=mp3_out,
        mode_name=mode,
        sample_rate=sample_rate,
        line_structure=line_structure,
        amplitude=amplitude,
        mp3_bitrate=bitrate,
        vis_code=vis,
        include_vis=include_vis,
        pre_silence=pre_silence,
        post_silence=post_silence,
        caption=caption,
        image_fit=fit,
        image_background=background,
        registry=registry,
    )


def decode(
        audio,
        out=None,
        mode=None,
        styles=None,
        fmt=None,
        denoise="off",
        levels=None,
        slant=0.03,
        sidecar=True,
        diagnostics=False,
        report=None,
        output_base=None,
        output_mode=None,
        image_format=None,
        auto_levels=None,
        forced_mode_name=None,
        slant_search=None,
        write_sidecar=None,
        line_structure="auto",
        custom_modes_json=None,
):
    """
    SSTV audio -> image(s). Out path is automatic when omitted.

    line_structure:
        auto       choose from the audio (default)
        standard   one sync per image row, the published Wraase SC-2 timing
        rgb3       one sync per colour component

    Short names (out, styles, fmt, levels, mode, slant, sidecar) and long names
    (output_base, output_mode, image_format, auto_levels, forced_mode_name,
    slant_search, write_sidecar) both work.
    """
    audio = _abs(audio)

    out = _first_not_none(output_base, out)
    fmt = _first_not_none(image_format, fmt, DEFAULT_IMAGE_FORMAT)
    styles = _first_not_none(output_mode, styles, "raw")
    levels = bool(_first_not_none(auto_levels, levels, False))
    mode = _first_not_none(forced_mode_name, mode)
    slant = float(_first_not_none(slant_search, slant, 0.03))
    sidecar = bool(_first_not_none(write_sidecar, sidecar, True))

    if out is None:
        out = os.path.splitext(audio)[0] + "_decoded." + fmt.lstrip(".")

    return decode_audio_to_images_v2(
        audio_path=audio,
        output_base=_abs(out),
        output_mode=styles,
        image_format=fmt,
        slant_search=slant,
        forced_mode_name=mode,
        auto_levels=levels,
        denoise=denoise,
        write_sidecar=sidecar,
        diagnostics=diagnostics,
        report_path=report,
        line_structure=line_structure,
        custom_modes_json=custom_modes_json,
    )


def rt(
        image,
        out_dir=None,
        mode=DEFAULT_MODE,
        fit="contain",
        background=(0, 0, 0),
        styles="raw",
        fmt=DEFAULT_IMAGE_FORMAT,
        denoise="off",
        levels=False,
        sample_rate=48000,
        amplitude=0.80,
        comparison=True,
        keep_wav=True,
        verbose=True,
        line_structure="standard",
):
    """
    image -> SSTV audio -> image, with metrics and a side-by-side PNG.

    Returns a dict with everything: wav path, decoded paths, raw path,
    comparison path, quality, metrics, timings.
    """
    image = _abs(image)
    stem = _stem(image)
    out_dir = _rt_out_dir(image, out_dir)

    wav = os.path.join(out_dir, stem + "_sstv.wav")
    out_base = os.path.join(out_dir, stem + "_decoded." + fmt.lstrip("."))

    t0 = time.time()
    enc = encode(
        image,
        out=wav,
        mode=mode,
        fit=fit,
        background=background,
        sample_rate=sample_rate,
        amplitude=amplitude,
        line_structure=line_structure,
    )
    encode_seconds = time.time() - t0

    t0 = time.time()
    dec = decode(
        wav,
        out=out_base,
        styles=styles,
        fmt=fmt,
        denoise=denoise,
        levels=levels,
    )
    decode_seconds = time.time() - t0

    result = {
        "ok": False,
        "image": image,
        "mode": mode,
        "fit": fit,
        "out_dir": out_dir,
        "wav": wav,
        "wav_seconds": enc["duration_seconds"],
        "vis_code": enc["vis_code"],
        "vis_ok": bool(enc.get("vis_header_check", {}).get("ok", False)),
        "encode": enc,
        "decode": dec,
        "raw": None,
        "comparison": None,
        "metrics": None,
        "quality": None,
        "encode_seconds": round(encode_seconds, 3),
        "decode_seconds": round(decode_seconds, 3),
        "error": None,
    }

    smoothing = (dec.get("decode_image") or {}).get("smoothing_seconds")
    if smoothing is not None:
        result["smoothing_seconds"] = smoothing
        result["smoothing_samples"] = (dec.get("decode_image") or {}).get("smoothing_samples")

    detected = dec["detect_mode"]

    if "error" in detected:
        result["error"] = detected["error"]
        write_json(os.path.join(out_dir, "result.json"), result)
        if verbose:
            print(f"{mode}: FAILED - {result['error']}")
        return result

    decoded = dec["decode_image"]

    if decoded is None or "error" in decoded:
        result["error"] = (decoded or {}).get("error", "decode failed")
        write_json(os.path.join(out_dir, "result.json"), result)
        if verbose:
            print(f"{mode}: FAILED - {result['error']}")
        return result

    layout = ModeRegistry().get_layout(mode)
    raw = decoded["paths"]["raw"]

    reference = _reference_array(
        image,
        layout.width,
        layout.height,
        fit=fit,
        background=background,
    )

    metrics = _metrics(reference, raw)

    comparison_path = None
    if comparison:
        comparison_path = side(
            image,
            raw,
            os.path.join(out_dir, stem + "_comparison.png"),
            right="DECODED FROM SSTV AUDIO (%s, fit=%s)" % (mode, fit),
        )

    if not keep_wav:
        try:
            os.remove(wav)
        except OSError:
            pass
        result["wav"] = None

    result.update({
        "ok": True,
        "raw": raw,
        "comparison": comparison_path,
        "metrics": metrics,
        "quality": decoded.get("quality"),
        "paths": decoded.get("paths"),
        "detected_mode": detected.get("mode_name"),
        "sync_lock": decoded.get("sync_lock_ok"),
        "mode_size": [layout.width, layout.height],
    })

    write_json(os.path.join(out_dir, "result.json"), result)

    if verbose:
        print(f"{mode}: mae {metrics['mae']:.2f} | psnr {metrics['psnr']:.1f} dB | "
              f"corr {metrics['correlation']:.3f} | quality "
              f"{(result['quality'] or {}).get('overall', 0):.3f} | {out_dir}")

    return result


def bench(
        image,
        modes=None,
        out_dir=None,
        fit="contain",
        background=(0, 0, 0),
        fmt=DEFAULT_IMAGE_FORMAT,
        denoise="off",
        levels=False,
        comparison=False,
        keep_wav=False,
        sample_rate=48000,
        verbose=True,
        count=0,
        seed=None,
        include_experimental=False,
):
    """
    Run rt() across several modes and rank them.

    modes=None means every non-experimental mode the encoder supports;
    modes="random" (or modes=3) picks that many modes at random, and seed
    makes the pick repeatable. Experimental modes (Pasokon P3/P5/P7, PD-50)
    are skipped unless include_experimental is True or they are named
    explicitly in modes. Returns a summary dict with "ranked" (best first)
    and "weak" (modes that failed or decoded badly).
    """
    image = _abs(image)
    registry = ModeRegistry()

    available = registry.supported_encoder_modes()

    if include_experimental:
        excluded_experimental = []
    else:
        excluded_experimental = [
            m for m in available if registry.modes[m].experimental
        ]
        available = [m for m in available if m not in excluded_experimental]

    modes, random_pick = _resolve_bench_modes(available, modes, count, seed)

    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), DEFAULT_OUT_ROOT, _stem(image) + "_bench")
    else:
        out_dir = _abs(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    results = []

    for mode_name in modes:
        try:
            result = rt(
                image,
                out_dir=os.path.join(out_dir, _safe_filename_part(mode_name)),
                mode=mode_name,
                fit=fit,
                background=background,
                styles="raw",
                fmt=fmt,
                denoise=denoise,
                levels=levels,
                comparison=comparison,
                keep_wav=keep_wav,
                sample_rate=sample_rate,
                verbose=False,
            )
            result["weak_reasons"] = _weak_reasons(result)
        except Exception as e:
            result = {
                "ok": False,
                "image": image,
                "mode": mode_name,
                "error": str(e),
                "metrics": None,
                "quality": None,
                "weak_reasons": [str(e)],
            }

        results.append(result)

        del result
        gc.collect()

    ok = [r for r in results if r.get("ok")]
    ranked = sorted(ok, key=lambda r: -r["metrics"][RANK_METRIC])
    weak = [r for r in results if r.get("weak_reasons")]

    summary = {
        "image": image,
        "fit": fit,
        "out_dir": out_dir,
        "modes": list(modes),
        "random": bool(random_pick),
        "seed": seed,
        "count": len(results),
        "ok_count": len(ok),
        "weak_count": len(weak),
        "include_experimental": bool(include_experimental),
        "excluded_experimental": excluded_experimental,
        "ranked": [
            {
                "mode": r["mode"],
                "size": r.get("mode_size"),
                "wav_seconds": round(r.get("wav_seconds", 0), 2),
                "mae": r["metrics"]["mae"],
                "psnr": r["metrics"]["psnr"],
                "correlation": r["metrics"]["correlation"],
                "quality": round((r.get("quality") or {}).get("overall", 0), 4),
                "sync_lock": r.get("sync_lock"),
            }
            for r in ranked
        ],
        "weak": [
            {
                "mode": r["mode"],
                "reasons": r["weak_reasons"],
                "mae": (r.get("metrics") or {}).get("mae"),
                "psnr": (r.get("metrics") or {}).get("psnr"),
                "correlation": (r.get("metrics") or {}).get("correlation"),
            }
            for r in weak
        ],
        "results": results,
    }

    write_json(os.path.join(out_dir, "bench.json"), summary)
    _write_bench_csv(os.path.join(out_dir, "bench.csv"), summary)

    if verbose:
        _print_bench(summary)

    return summary


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


def _write_bench_csv(path, summary):
    lines = ["mode,size,audio_seconds,mae,psnr_db,correlation,quality,sync_lock,status"]

    for r in summary["ranked"]:
        lines.append(
            "%s,%s,%.2f,%.3f,%.3f,%.4f,%.4f,%s,%s"
            % (
                r["mode"],
                "%dx%d" % tuple(r["size"]) if r.get("size") else "",
                r["wav_seconds"],
                r["mae"],
                r["psnr"],
                r["correlation"],
                r["quality"],
                r.get("sync_lock"),
                "ok",
            )
        )

    for r in summary["weak"]:
        lines.append(
            "%s,,,%s,%s,%s,,,%s"
            % (
                r["mode"],
                r["mae"] if r["mae"] is not None else "",
                r["psnr"] if r["psnr"] is not None else "",
                r["correlation"] if r["correlation"] is not None else "",
                "weak: " + "; ".join(r["reasons"]),
            )
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return path


def _print_bench(summary):
    print()
    print("nSSTV bench: %s" % summary["image"])
    print("=" * 92)
    print("%-18s %8s %10s %8s %10s %8s %8s" % ("mode", "audio", "size", "mae", "psnr", "corr", "quality"))
    print("-" * 92)

    for r in summary["ranked"]:
        size = "%dx%d" % tuple(r["size"]) if r.get("size") else ""
        print("%-18s %7.0fs %10s %8.2f %8.1fdB %8.3f %8.3f"
              % (r["mode"], r["wav_seconds"], size, r["mae"], r["psnr"],
                 r["correlation"], r["quality"]))

    for r in summary["weak"]:
        if r["psnr"] is None:
            print("%-18s %8s %10s %8s %10s %8s %8s" % (r["mode"], "-", "-", "-", "FAILED", "-", "-"))
        else:
            print("%-18s %8s %10s %8.2f %8.1fdB %8.3f %8s  <- weak"
                  % (r["mode"], "-", "-", r["mae"], r["psnr"], r["correlation"], "-"))

    print("-" * 92)
    print("%d modes, %d usable, %d weak | out: %s"
          % (summary["count"], summary["ok_count"], summary["weak_count"], summary["out_dir"]))

    if summary.get("excluded_experimental"):
        print("experimental modes skipped: %s"
              % ", ".join(summary["excluded_experimental"]))

    if summary["weak"]:
        print()
        print("weak links:")
        for r in summary["weak"]:
            print("  %-18s %s" % (r["mode"], "; ".join(r["reasons"])))


def modes(custom_modes_json=None):
    """
    List modes the encoder supports.
    """
    return make_registry(custom_modes_json).supported_encoder_modes()


def card(mode=DEFAULT_MODE, out="testcard.png", title=None, subtitle=None):
    """
    Make a test card sized for a mode.
    """
    return make_testcard_for_mode(_abs(out), mode_name=mode, title=title, subtitle=subtitle)


def fit(image, mode=DEFAULT_MODE, out=None, how="contain", background=(0, 0, 0)):
    """
    Preview/save how an image will be fitted into a mode's frame.
    """
    image = _abs(image)

    if out is None:
        out = os.path.splitext(image)[0] + "_fit.png"

    out = _abs(out)
    registry = ModeRegistry()
    layout = registry.get_layout(mode)

    img = prepare_image_for_mode(
        image,
        width=layout.width,
        height=layout.height,
        fit=how,
        background=background,
    )

    _ensure_parent(out)
    img.save(out, "PNG")
    return out


def dec(*args, **kwargs):
    return decode(*args, **kwargs)


def enc(*args, **kwargs):
    return encode(*args, **kwargs)


_CLI_EXAMPLES = (
    "nsstv decode input.wav --out-base out.jpg [--image-format png] [--denoise medium] [--diagnostics]",
    "nsstv decode-all recording.wav --out-dir decoded",
    "nsstv batch ./recordings --out-dir ./decoded",
    "nsstv encode image.jpg --mode PD-180 --wav-out tx.wav [--caption \"TEXT\"]",
    "nsstv bench photo.jpg --modes random --count 4",
    "nsstv roundtrip input.wav --out-dir decoded --encoded-wav-out rt.wav",
    "nsstv testcard --mode PD-180 --out testcard.png",
    "nsstv live --seconds 240 --out-base live.jpg",
    "nsstv modes",
)

_API_CATALOG = (
    (
        "Quick start",
        (
            ("decode", "Decode one SSTV image from audio.",
             'info = nSSTV.decode("input.wav", out="output.png", styles="all")',
             "decode_audio_to_images_v2"),
            ("encode", "Encode an image to SSTV WAV/MP3 audio.",
             'info = nSSTV.encode("image.jpg", out="tx.wav", mode="PD-180")',
             "encode_image_to_sstv_audio_v2"),
            ("decode_all", "Decode every SSTV transmission in one recording. Out dir, format and style are automatic.",
             'info = nSSTV.decode_all("recording.wav")'),
            ("cut", "Keep only the first N seconds of an audio file; makes a partial transmission to test with.",
             'short = nSSTV.cut("tx.wav", 16.0)'),
            ("join", "Concatenate audio files with silence between them; makes a multi-image recording.",
             'multi = nSSTV.join(["a.wav", "b.wav"], "recording.wav")'),
            ("rt", "Roundtrip: image -> SSTV audio -> image, with metrics and a side-by-side PNG.",
             'r = nSSTV.rt("photo.jpg")'),
            ("bench", "Run rt() across every mode and rank them; finds weak modes.",
             'summary = nSSTV.bench("photo.jpg")'),
            ("compare", "Score two images: mae, psnr, correlation.",
             'nSSTV.compare("original.png", "decoded.png")'),
            ("side", "Side-by-side comparison PNG of two images.",
             'nSSTV.side("original.png", "decoded.png", "comparison.png")'),
            ("card", "Test card sized for a mode.",
             'nSSTV.card("PD-180", "testcard.png")'),
            ("fit", "Preview how an image is fitted into a mode's frame.",
             'nSSTV.fit("photo.jpg", mode="PD-180", how="contain")'),
            ("modes", "List the modes the encoder supports.",
             "names = nSSTV.modes()"),
            ("random_modes", "Pick mode names at random; same seed, same picks.",
             "picks = nSSTV.random_modes(4, seed=7)"),
        ),
    ),
    (
        "Decoding",
        (
            ("decode_audio_to_images", "Original single-image decode API.",
             'info = nSSTV.decode_audio_to_images("input.wav", "output.jpg")'),
            ("decode_audio_to_images_v2", "Full decode API: PNG/JPEG, sidecars, diagnostics, report, quality score.",
             'info = nSSTV.decode_audio_to_images_v2("input.wav", "out.png", image_format="png", diagnostics=True)'),
            ("decode_all_audio_to_images", "Scan a recording and decode every transmission found.",
             'info = nSSTV.decode_all_audio_to_images("recording.wav", out_dir="decoded")'),
            ("batch_decode", "Decode every WAV/MP3 in a folder; writes batch_summary.json and summary.csv.",
             'info = nSSTV.batch_decode("recordings", out_dir="decoded", denoise="light")'),
            ("decode_image_extended", "Decode one image from an already-configured SSTVDecoder.",
             'image = nSSTV.decode_image_extended(decoder, "out.png", image_format="png")'),
            ("SSTVDecoder", "Low-level decoder object: detect_mode(), force_mode(), decode_image().",
             'decoder = nSSTV.SSTVDecoder("input.wav"); decoder.detect_mode()'),
        ),
    ),
    (
        "Encoding",
        (
            ("encode_image_to_sstv_audio", "Original encode API.",
             'info = nSSTV.encode_image_to_sstv_audio("image.jpg", wav_out="tx.wav")'),
            ("encode_image_to_sstv_audio_v2", "Encode API with optional caption overlay.",
             'info = nSSTV.encode_image_to_sstv_audio_v2("image.jpg", wav_out="tx.wav", caption="NANA")'),
            ("SSTVEncoder", "Low-level encoder object: encode_image_to_audio(), encode_image_file().",
             'encoder = nSSTV.SSTVEncoder(mode_name="PD-180", fs=48000)'),
            ("ToneSynth", "Continuous-phase tone synthesizer used by the encoder.",
             "synth = nSSTV.ToneSynth(fs=48000, amplitude=0.80); synth.append_tone(1900.0, 0.3)"),
        ),
    ),
    (
        "Modes, registry, protocol",
        (
            ("ModeRegistry", "Mode/VIS/layout registry: add_custom_mode(), load_custom_modes_json().",
             "registry = nSSTV.ModeRegistry(); registry.supported_encoder_modes()"),
            ("make_registry", "Build a registry, optionally loading custom mode JSON files.",
             'registry = nSSTV.make_registry("examples/custom_modes.json")'),
            ("ImageLayout", "Dataclass describing width/height/color/channel timing for a mode.",
             "layout = registry.get_layout(\"PD-180\")"),
            ("SSTVMode", "Dataclass with pixel/sync/porch timing for a mode.",
             "mode = registry.get_mode(\"PD-180\")"),
            ("ProtocolConstants", "VIS leader, break, and bit timing constants.",
             "protocol = nSSTV.ProtocolConstants()"),
            ("is_experimental_mode", "True for experimental modes; context='decode-all' adds the decode-all-only flags.",
             'nSSTV.is_experimental_mode("Pasokon P3")'),
            ("experimental_notice", "One-line human explanation for an experimental mode, or None.",
             'nSSTV.experimental_notice("PD-50")'),
        ),
    ),
    (
        "Audio I/O",
        (
            ("load_audio_mono", "Load WAV/MP3 as mono float64 samples; returns (fs, samples).",
             "fs, samples = nSSTV.load_audio_mono(\"input.wav\")"),
            ("write_wav_file", "Write float audio [-1, 1] to int16 WAV.",
             'nSSTV.write_wav_file("out.wav", fs, audio)'),
            ("write_mp3_file", "Write MP3 using pydub/ffmpeg. Lossy.",
             'nSSTV.write_mp3_file("out.mp3", fs, audio, bitrate="320k")'),
            ("write_audio_file", "Write WAV or MP3 based on the file extension.",
             'nSSTV.write_audio_file("out.wav", fs, audio)'),
            ("find_ffmpeg", "Locate ffmpeg/avconv, or None.",
             "path = nSSTV.find_ffmpeg()"),
            ("find_ffprobe", "Locate ffprobe/avprobe, or None.",
             "path = nSSTV.find_ffprobe()"),
            ("ffmpeg_is_available", "True if a usable ffmpeg was found.",
             "ok = nSSTV.ffmpeg_is_available()"),
            ("configure_pydub_ffmpeg", "Point pydub at the discovered ffmpeg/ffprobe.",
             "nSSTV.configure_pydub_ffmpeg(require=True)"),
        ),
    ),
    (
        "Image helpers",
        (
            ("make_testcard", "Generate a color-bar test card image.",
             'nSSTV.make_testcard("testcard.png", width=640, height=496)'),
            ("make_testcard_for_mode", "Generate a test card sized for a specific SSTV mode.",
             'nSSTV.make_testcard_for_mode("testcard.png", mode_name="PD-180")'),
            ("add_caption_to_image", "Burn a caption bar into an image before encoding.",
             'nSSTV.add_caption_to_image("in.jpg", "out.jpg", text="NANA - GHANA")'),
            ("prepare_image_for_mode",
             "Resize an image to a mode's dimensions: contain (pad), cover (crop), stretch (distort).",
             'img = nSSTV.prepare_image_for_mode("photo.jpg", 640, 496, fit="contain")'),
            ("auto_levels_rgb", "Robust per-channel contrast stretch.",
             "rgb = nSSTV.auto_levels_rgb(rgb)"),
            ("denoise_rgb", "Post-decode denoise: off, light, medium, strong.",
             'rgb = nSSTV.denoise_rgb(rgb, preset="medium")'),
            ("save_image_file", "Save a PIL image as JPEG or PNG.",
             'nSSTV.save_image_file(pil_img, "out.png")'),
            ("save_rgb_image", "Save a NumPy RGB array as JPEG or PNG.",
             'nSSTV.save_rgb_image(rgb, "out.png", image_format="png")'),
            ("make_spectrogram_image", "Render a 900-2600 Hz spectrogram image.",
             "spec = nSSTV.make_spectrogram_image(samples, fs, 0.0, 10.0, width=1200)"),
            ("make_diagnostics_image", "Spectrogram with leader/VIS/image markers.",
             'nSSTV.make_diagnostics_image(samples, fs, "diag.jpg", markers=[{"time": 0.5, "label": "leader"}])'),
        ),
    ),
    (
        "Reports and quality",
        (
            ("estimate_decode_quality", "0..1 quality score from VIS header and sync lock metadata.",
             "score = nSSTV.estimate_decode_quality(detect_info, image_info)"),
            ("write_json", "Write JSON with indent=2, creating parent directories.",
             'nSSTV.write_json("report.json", data)'),
            ("save_metadata_sidecar", "Write <image>.json metadata next to an image.",
             'nSSTV.save_metadata_sidecar("out_raw.png", metadata)'),
            ("save_decode_report", "Write a full decode report JSON.",
             'nSSTV.save_decode_report("report.json", info)'),
        ),
    ),
    (
        "Signal processing internals",
        (
            ("ToneDetector", "Sliding single-bin DFT tone detector (Goertzel-style).",
             "detector = nSSTV.ToneDetector(fs, samples)"),
            ("VISHeaderReader", "Read and verify the VIS bits.",
             "reader = nSSTV.VISHeaderReader(detector, nSSTV.ProtocolConstants())"),
            ("FMDemodulator", "Instantaneous-frequency FM demodulator.",
             "demod = nSSTV.FMDemodulator(fs, samples)"),
            ("ImageDecoder", "Sync detection, line-timing fit, image assembly.",
             "rgb = nSSTV.ImageDecoder(fs, samples, layout, vis_end).decode()"),
        ),
    ),
    (
        "CLI and script mode",
        (
            ("main", "Console entry point: arguments -> cli_main(), no arguments -> script_main().",
             "raise SystemExit(nSSTV.main())"),
            ("cli_main", "Run the argparse CLI: decode, decode-all, batch, encode, roundtrip, testcard, live, modes.",
             'nSSTV.cli_main(["modes"])'),
            ("script_main", "Path-free no-argument mode driven by NSSTV_* environment variables.",
             "NSSTV_ACTION=decode NSSTV_AUDIO_INPUT=in.wav nsstv"),
            ("build_arg_parser", "Return the argparse parser used by cli_main().",
             "parser = nSSTV.build_arg_parser()"),
            ("info", "Show what nSSTV contains and how to use it.",
             'nSSTV.info()  |  nSSTV.info("decode")  |  nSSTV.info("cli")'),
        ),
    ),
    (
        "Constants",
        (
            ("OUTPUT_IMAGE_MODES", "Valid --output-mode values.",
             "print(nSSTV.OUTPUT_IMAGE_MODES)"),
            ("SSTV_SYNC_HZ", "Sync pulse frequency, 1200 Hz.",
             "print(nSSTV.SSTV_SYNC_HZ)"),
            ("SSTV_BLACK_HZ", "Black pixel frequency, 1500 Hz.",
             "print(nSSTV.SSTV_BLACK_HZ)"),
            ("SSTV_WHITE_HZ", "White pixel frequency, 2300 Hz.",
             "print(nSSTV.SSTV_WHITE_HZ)"),
            ("SSTV_LEADER_HZ", "VIS leader tone, 1900 Hz.",
             "print(nSSTV.SSTV_LEADER_HZ)"),
            ("VIS_ONE_HZ", "VIS bit 1, 1100 Hz.",
             "print(nSSTV.VIS_ONE_HZ)"),
            ("VIS_ZERO_HZ", "VIS bit 0, 1300 Hz.",
             "print(nSSTV.VIS_ZERO_HZ)"),
            ("LossyMP3Warning", "Warning class raised when MP3 paths are used.",
             "warnings.simplefilter(\"error\", nSSTV.LossyMP3Warning)"),
            ("UnknownModeError", "Error for an unknown mode name; lists the available modes.",
             "nSSTV.ModeRegistry().get_mode(\"Nope\")"),
        ),
    ),
)


def _shorten(text, width=96):
    text = str(text)
    return text if len(text) <= width else text[:width - 3] + "..."


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


def info(what=None, show=True):
    """
    Show what nSSTV contains and how to use it.

    Usage:

        import nSSTV

        nSSTV.info()             # list every public group + CLI examples
        nSSTV.info("decode")     # detail for one function, class, or constant
        nSSTV.info("cli")        # command-line examples
        nSSTV.info(show=False)   # return the catalog dict without printing

    Returns:
        - dict of {group: [entry, ...]} when `what` is None
        - one entry dict when `what` names a function, class, or constant
        - None when nothing matches
    """
    catalog = _api_catalog_dict()
    entries = [item for items in catalog.values() for item in items]

    lines = []
    bar = "=" * 78

    def _header():
        lines.append(bar)
        lines.append(f"nSSTV {__version__} - public API")
        lines.append(f"Made by {__author__}")
        lines.append(
            f"{len(entries)} documented names | "
            'nSSTV.info("name") for details | help(nSSTV.name) for full docs'
        )
        lines.append(bar)
        lines.append("")

    if what is None:
        result = catalog

        _header()

        for group, items in catalog.items():
            lines.append(group.upper())
            lines.append("-" * 78)

            for item in items:
                lines.append(f"  {_shorten(item['signature'])}")
                lines.append(f"      {item['summary']}")
                lines.append(f"      Example: {item['example']}")

            lines.append("")

        lines.append("COMMAND LINE")
        lines.append("-" * 78)

        for example in _CLI_EXAMPLES:
            lines.append(f"  {example}")

        lines.append("")
        lines.append("SCRIPT MODE (no arguments)")
        lines.append("-" * 78)
        lines.append("  NSSTV_ACTION=decode NSSTV_AUDIO_INPUT=input.wav nsstv")
        lines.append("  NSSTV_ACTION=batch NSSTV_WORKDIR=./recordings nsstv")
        lines.append("  NSSTV_ACTION=encode NSSTV_IMAGE_INPUT=image.jpg nsstv")
        lines.append("")
        lines.append("Tip: nSSTV.modes via CLI is `nsstv modes`; MP3 support needs ffmpeg.")

    elif str(what).strip().lower() in ("cli", "commands", "command-line", "usage", "shell"):
        result = list(_CLI_EXAMPLES)

        lines.append(bar)
        lines.append(f"nSSTV {__version__} - command line")
        lines.append(bar)
        lines.append("")

        for example in _CLI_EXAMPLES:
            lines.append(f"  {example}")

        lines.append("")
        lines.append("Run any command with --help for its flags, e.g. `nsstv decode --help`.")
        lines.append("With no arguments at all, `nsstv` runs script mode using NSSTV_* variables.")

    else:
        wanted = str(what).strip()
        match = None

        for item in entries:
            if item["name"].lower() == wanted.lower():
                match = item
                break

        result = match

        if match is None:
            import difflib

            close = difflib.get_close_matches(
                wanted,
                [item["name"] for item in entries],
                n=5,
                cutoff=0.5,
            )

            lines.append(bar)
            lines.append(f"nSSTV has no documented name {wanted!r}.")
            lines.append(bar)

            if close:
                lines.append("")
                lines.append("Did you mean:")
                for name in close:
                    lines.append(f"  nSSTV.info({name!r})")

            lines.append("")
            lines.append("Call nSSTV.info() to list everything.")

        else:
            lines.append(bar)
            lines.append(f"{match['name']}  ({match['group']})")
            lines.append(bar)
            lines.append("")
            import textwrap

            sig_lines = textwrap.wrap(
                match["signature"],
                width=88,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [match["signature"]]

            lines.append(f"  Use:      {sig_lines[0]}")
            for extra in sig_lines[1:]:
                lines.append(f"           {extra}")
            lines.append(f"  Summary:  {match['summary']}")
            lines.append("")
            lines.append("  Example:")
            lines.append(f"      {match['example']}")

            if match["doc"]:
                lines.append("")
                lines.append("  Docs:")
                for line in match["doc"].splitlines()[:12]:
                    lines.append(f"      {line.rstrip()}")
                if len(match["doc"].splitlines()) > 12:
                    lines.append("      ... (see help(nSSTV." + match["name"] + "))")

            lines.append("")

    if show:
        print("\n".join(lines))

    return result


def _clean_path(path):
    if path is None:
        return None
    path = str(path).strip()
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _env(name, default=None):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return value


def _env_float(name, default):
    value = _env(name, None)
    return default if value is None else float(value)


def _env_int(name, default):
    value = _env(name, None)
    return default if value is None else int(value)


def _env_modes(name):
    """Comma separated mode names, or a single word like random."""
    value = _env(name, None)

    if not value:
        return None

    parts = [p.strip() for p in value.split(",") if p.strip()]

    return parts or None


def _env_bool(name, default=False):
    value = _env(name, None)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


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


def _parse_color(text, default=(0, 0, 0)):
    """
    Parse an "r,g,b" string into an (r, g, b) tuple of ints.
    """
    if text is None:
        return default

    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]

    if len(parts) != 3:
        raise ValueError(f"Color must be 3 values like '0,0,0'; got {text!r}")

    values = []

    for part in parts:
        value = int(float(part))
        if not 0 <= value <= 255:
            raise ValueError(f"Color channel {value} out of range 0..255")
        values.append(value)

    return tuple(values)


def _default_output_base_for_audio(audio_path):
    base, _ = os.path.splitext(audio_path)
    return base + "_decoded.jpg"


def _default_encoded_wav_for_image(image_path):
    base, _ = os.path.splitext(image_path)
    return base + "_sstv.wav"


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


def script_main():
    """
    Path-free no-argument mode.

    Environment variables:

      NSSTV_ACTION=auto|decode|decode-all|batch|encode|roundtrip|testcard|live
      NSSTV_WORKDIR=/path
      NSSTV_AUDIO_INPUT=input.wav
      NSSTV_IMAGE_INPUT=image.jpg
      NSSTV_OUTPUT_BASE=out.jpg
      NSSTV_OUT_DIR=decoded/
      NSSTV_IMAGE_FORMAT=jpg|png
      NSSTV_DENOISE=off|light|medium|strong
      NSSTV_AUTO_LEVELS=1
      NSSTV_DIAGNOSTICS=1
      NSSTV_WRITE_SIDECAR=1
      NSSTV_BENCH_MODES=random|Martin M1,Robot 36
      NSSTV_BENCH_COUNT=5
      NSSTV_BENCH_SEED=7
      NSSTV_INCLUDE_EXPERIMENTAL=1
    """
    workdir = _clean_path(_env("NSSTV_WORKDIR", os.getcwd()))
    action = _env("NSSTV_ACTION", "auto").lower()

    audio_input = _clean_path(_env("NSSTV_AUDIO_INPUT", None))
    image_input = _clean_path(_env("NSSTV_IMAGE_INPUT", None))

    output_base = _clean_path(_env("NSSTV_OUTPUT_BASE", None))
    out_dir = _clean_path(_env("NSSTV_OUT_DIR", None))

    output_mode = _env("NSSTV_OUTPUT_MODE", "all")
    image_format = _env("NSSTV_IMAGE_FORMAT", "jpg").lower()
    slant_search = _env_float("NSSTV_SLANT_SEARCH", 0.03)

    encode_mode = _env("NSSTV_ENCODE_MODE", "PD-180")
    encoded_wav_out = _clean_path(_env("NSSTV_ENCODED_WAV_OUT", None))
    encoded_mp3_out = _clean_path(_env("NSSTV_ENCODED_MP3_OUT", None))

    sample_rate = _env_int("NSSTV_SAMPLE_RATE", 48000)
    amplitude = _env_float("NSSTV_AMPLITUDE", 0.80)
    mp3_bitrate = _env("NSSTV_MP3_BITRATE", "320k")

    custom_modes_json = _env_paths("NSSTV_CUSTOM_MODES_JSON")

    image_fit = _env("NSSTV_IMAGE_FIT", "contain").lower()
    image_background = _parse_color(_env("NSSTV_IMAGE_BACKGROUND", None))

    auto_levels = _env_bool("NSSTV_AUTO_LEVELS", False)
    denoise = _env("NSSTV_DENOISE", "off")
    diagnostics = _env_bool("NSSTV_DIAGNOSTICS", False)
    write_sidecar = _env_bool("NSSTV_WRITE_SIDECAR", True)

    decode_kwargs = dict(
        output_mode=output_mode,
        image_format=image_format,
        slant_search=slant_search,
        custom_modes_json=custom_modes_json,
        auto_levels=auto_levels,
        denoise=denoise,
        write_sidecar=write_sidecar,
        diagnostics=diagnostics,
    )

    def _resolve_input(value, finder, label):
        """Auto-find and validate an input file; None (with a message) on failure."""
        if value is None:
            value = finder(workdir)
        if not value or not os.path.exists(value):
            print(f"ERROR: no valid {label} input found.")
            return None
        return value

    print()
    print("nSSTV script mode")
    print("=================")
    print(f"Version: {__version__}")
    print(f"Made by: {__author__}")
    print(f"Working directory: {workdir}")
    print(f"Requested action: {action}")

    if action == "auto":
        if audio_input is None:
            audio_input = _auto_find_audio_file(workdir)

        if audio_input:
            action = "decode-all"
        else:
            if image_input is None:
                image_input = _auto_find_image_file(workdir)

            if image_input:
                action = "encode"
            else:
                print()
                print("ERROR: no input found.")
                print("Put a WAV/MP3 or JPG/PNG in the working directory,")
                print("or set NSSTV_AUDIO_INPUT / NSSTV_IMAGE_INPUT.")
                return 1

        print(f"Auto-selected action: {action}")

    if action == "decode":
        audio_input = _resolve_input(audio_input, _auto_find_audio_file, "audio")

        if audio_input is None:
            return 1

        if output_base is None:
            output_base = _default_output_base_for_audio(audio_input)

        info = decode_audio_to_images_v2(
            audio_path=audio_input,
            output_base=output_base,
            report_path=os.path.splitext(output_base)[0] + "_report.json",
            **decode_kwargs,
        )

        pprint(info)
        return 0 if info.get("decode_image") and "error" not in info["decode_image"] else 1

    if action == "decode-all":
        audio_input = _resolve_input(audio_input, _auto_find_audio_file, "audio")

        if audio_input is None:
            return 1

        info = decode_all_audio_to_images(audio_path=audio_input, out_dir=out_dir, **decode_kwargs)

        pprint(info)
        return 0 if info.get("decoded_count", 0) > 0 else 1

    if action == "batch":
        out_dir = out_dir or os.path.join(workdir, "nSSTV_decoded")

        info = batch_decode(
            input_dir=workdir,
            out_dir=out_dir,
            recursive=True,
            **decode_kwargs,
        )

        pprint(info)
        return 0

    if action == "encode":
        image_input = _resolve_input(image_input, _auto_find_image_file, "image")

        if image_input is None:
            return 1

        if encoded_wav_out is None:
            encoded_wav_out = _default_encoded_wav_for_image(image_input)

        caption = _env("NSSTV_CAPTION", None)

        info = encode_image_to_sstv_audio_v2(
            image_path=image_input,
            wav_out=encoded_wav_out,
            mp3_out=encoded_mp3_out,
            mode_name=encode_mode,
            sample_rate=sample_rate,
            amplitude=amplitude,
            mp3_bitrate=mp3_bitrate,
            custom_modes_json=custom_modes_json,
            include_vis=True,
            caption=caption,
            image_fit=image_fit,
            image_background=image_background,
        )

        pprint(info)

        if info.get("include_vis") and not info.get("vis_header_check", {}).get("ok", False):
            return 1

        return 0

    if action == "roundtrip":
        audio_input = _resolve_input(audio_input, _auto_find_audio_file, "audio")

        if audio_input is None:
            return 1

        if encoded_wav_out is None:
            encoded_wav_out = _default_roundtrip_wav_for_audio(audio_input)

        info = _roundtrip_first_image(
            audio_input,
            decode_kwargs={**decode_kwargs, "out_dir": out_dir},
            encoded_wav_out=encoded_wav_out,
            encoded_mp3_out=encoded_mp3_out,
            sample_rate=sample_rate,
            amplitude=amplitude,
            mp3_bitrate=mp3_bitrate,
            custom_modes_json=custom_modes_json,
            raw_missing="ERROR: raw image was not generated; use output_mode=all or raw.",
        )

        if info is None:
            return 1

        if not info["encode_first_image"].get("vis_header_check", {}).get("ok", False):
            return 1

        return 0

    if action in ("rt", "roundtrip-image"):
        image_input = _resolve_input(image_input, _auto_find_image_file, "image")

        if image_input is None:
            return 1

        result = rt(
            image_input,
            out_dir=out_dir,
            mode=encode_mode,
            fit=image_fit,
            background=image_background,
            styles=output_mode,
            fmt=image_format,
            denoise=denoise,
            levels=auto_levels,
        )

        pprint(result)
        return 0 if result["ok"] else 1

    if action == "bench":
        image_input = _resolve_input(image_input, _auto_find_image_file, "image")

        if image_input is None:
            return 1

        summary = bench(
            image_input,
            modes=_env_modes("NSSTV_BENCH_MODES"),
            out_dir=out_dir,
            fit=image_fit,
            background=image_background,
            denoise=denoise,
            levels=auto_levels,
            count=_env_int("NSSTV_BENCH_COUNT", 0),
            seed=_env_int("NSSTV_BENCH_SEED", None),
            include_experimental=_env_bool("NSSTV_INCLUDE_EXPERIMENTAL", False),
        )

        pprint(summary)
        return 0 if summary["ok_count"] > 0 else 1

    if action == "testcard":
        registry = make_registry(custom_modes_json)
        path = output_base or os.path.join(workdir, f"nSSTV_testcard_{_safe_filename_part(encode_mode)}.{image_format}")
        info_path = make_testcard_for_mode(path, mode_name=encode_mode, registry=registry)
        print(f"Wrote testcard: {info_path}")
        return 0

    if action == "live":
        output_base = output_base or os.path.join(workdir, f"nSSTV_live.{image_format}")
        seconds = _env_float("NSSTV_LIVE_SECONDS", 240.0)
        device = _env("NSSTV_LIVE_DEVICE", None)

        info = live_record_then_decode(
            seconds=seconds,
            out_base=output_base,
            sample_rate=sample_rate,
            device=device,
            output_mode=output_mode,
            image_format=image_format,
            slant_search=slant_search,
            auto_levels=auto_levels,
            denoise=denoise,
            diagnostics=diagnostics,
        )

        pprint(info)
        return 0 if info.get("decode_image") and "error" not in info["decode_image"] else 1

    print(f"ERROR: unknown action {action!r}")
    return 1


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="nSSTV: WAV/MP3 <-> SSTV images, with custom modes, batch decode, diagnostics, and test cards."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    denoise_choices = ("off", "light", "medium", "strong")
    image_format_choices = ("jpg", "jpeg", "png")

    line_structure_help = (
        "standard (default) sends one sync per image row; rgb3 sends one "
        "sync per colour component, three transmitted lines per row."
    )

    def _add_common_flags(p, out_flag, no_sidecar=True):
        """Flags shared by the decode / decode-all / batch / roundtrip subcommands."""
        if out_flag:
            p.add_argument(out_flag, default=None)
        p.add_argument("--output-mode", default="all", choices=OUTPUT_IMAGE_MODES)
        p.add_argument("--image-format", default="jpg", choices=image_format_choices)
        p.add_argument("--slant-search", type=float, default=0.03)
        p.add_argument("--custom-modes-json", action="append", default=None)
        p.add_argument("--auto-levels", action="store_true")
        p.add_argument("--denoise", default="off", choices=denoise_choices)
        if no_sidecar:
            p.add_argument("--no-sidecar", action="store_true")
        p.add_argument("--diagnostics", action="store_true")

    p_decode = sub.add_parser("decode", help="Decode one SSTV image from audio.")
    p_decode.add_argument("audio_path")
    _add_common_flags(p_decode, "--out-base")
    p_decode.add_argument("--force-mode", default=None)
    p_decode.add_argument("--vis-end", type=float, default=None)
    p_decode.add_argument("--first-sync-time", type=float, default=None)
    p_decode.add_argument("--freq-offset-hz", type=float, default=0.0)
    p_decode.add_argument("--report", default=None)
    p_decode.add_argument(
        "--line-structure",
        default="auto",
        choices=("auto", "standard", "rgb3"),
        help="auto (default) picks from the audio; rgb3 decodes Wraase SC-2 "
             "transmissions that send a sync before every colour component.",
    )

    p_all = sub.add_parser("decode-all", help="Decode all SSTV images in one audio file.")
    p_all.add_argument("audio_path")
    _add_common_flags(p_all, "--out-dir")
    p_all.add_argument("--force-mode", default=None,
                       help="Skip VIS detection and decode every transmission as this mode.")
    p_all.add_argument("--report", default=None, help="Also write the summary JSON to this path.")

    p_batch = sub.add_parser("batch", help="Batch decode a folder of WAV/MP3 files.")
    p_batch.add_argument("input_dir")
    p_batch.add_argument("--no-recursive", action="store_true")
    _add_common_flags(p_batch, "--out-dir")
    p_batch.add_argument("--stop-on-error", action="store_true")

    p_encode = sub.add_parser("encode", help="Encode image to SSTV WAV/MP3.")
    p_encode.add_argument("image_path")
    p_encode.add_argument("--mode", default="PD-180")
    p_encode.add_argument("--wav-out", default=None)
    p_encode.add_argument("--mp3-out", default=None)
    p_encode.add_argument("--sample-rate", type=int, default=48000)
    p_encode.add_argument("--amplitude", type=float, default=0.80)
    p_encode.add_argument("--mp3-bitrate", default="320k")
    p_encode.add_argument("--custom-modes-json", action="append", default=None)
    p_encode.add_argument("--vis-code", type=int, default=None)
    p_encode.add_argument("--no-vis", action="store_true")
    p_encode.add_argument("--pre-silence", type=float, default=0.50)
    p_encode.add_argument("--post-silence", type=float, default=0.50)
    p_encode.add_argument("--strict-mp3", action="store_true")
    p_encode.add_argument("--caption", default=None)
    p_encode.add_argument("--caption-position", default="bottom", choices=("top", "bottom"))
    p_encode.add_argument("--fit", default="contain", choices=("contain", "cover", "stretch"),
                          help="How the image fills the mode: contain pads, cover crops, stretch distorts.")
    p_encode.add_argument("--background", default="0,0,0",
                          help="Padding color for --fit contain, as r,g,b. Default 0,0,0.")
    p_encode.add_argument("--line-structure", default="standard", choices=("standard", "rgb3"),
                          help=line_structure_help)

    p_roundtrip = sub.add_parser("roundtrip", help="Decode audio, then encode first decoded raw image to WAV.")
    p_roundtrip.add_argument("audio_path")
    p_roundtrip.add_argument("--encoded-wav-out", default=None)
    p_roundtrip.add_argument("--encoded-mp3-out", default=None)
    p_roundtrip.add_argument("--sample-rate", type=int, default=48000)
    p_roundtrip.add_argument("--amplitude", type=float, default=0.80)
    p_roundtrip.add_argument("--mp3-bitrate", default="320k")
    _add_common_flags(p_roundtrip, "--out-dir", no_sidecar=False)

    p_test = sub.add_parser("testcard", help="Generate a test card image.")
    p_test.add_argument("--out", default="nSSTV_testcard.png")
    p_test.add_argument("--mode", default="PD-180")
    p_test.add_argument("--custom-modes-json", action="append", default=None)
    p_test.add_argument("--title", default=None)
    p_test.add_argument("--subtitle", default=None)

    p_rt = sub.add_parser("rt", help="Roundtrip: image -> SSTV audio -> image.")
    p_rt.add_argument("image_path")
    p_rt.add_argument("--out-dir", default=None)
    p_rt.add_argument("--mode", default=DEFAULT_MODE)
    p_rt.add_argument("--fit", default="contain", choices=("contain", "cover", "stretch"))
    p_rt.add_argument("--background", default="0,0,0")
    p_rt.add_argument("--styles", "--output-mode", dest="styles", default="raw",
                      choices=OUTPUT_IMAGE_MODES, help="Output style (alias: --output-mode).")
    p_rt.add_argument("--image-format", default="png", choices=image_format_choices)
    p_rt.add_argument("--denoise", default="off", choices=denoise_choices)
    p_rt.add_argument("--auto-levels", action="store_true")
    p_rt.add_argument("--line-structure", default="standard", choices=("standard", "rgb3"),
                      help=line_structure_help)
    p_rt.add_argument("--no-comparison", action="store_true")

    p_bench = sub.add_parser("bench", help="Roundtrip across modes and rank them.")
    p_bench.add_argument("image_path")
    p_bench.add_argument("--modes", default=None, help="Comma separated list, or 'random'; default: every mode.")
    p_bench.add_argument("--count", type=int, default=0, help="How many random modes, when --modes random.")
    p_bench.add_argument("--seed", type=int, default=None, help="Repeat a random pick.")
    p_bench.add_argument("--out-dir", default=None)
    p_bench.add_argument("--fit", default="contain", choices=("contain", "cover", "stretch"))
    p_bench.add_argument("--background", default="0,0,0")
    p_bench.add_argument("--denoise", default="off", choices=denoise_choices)
    p_bench.add_argument("--auto-levels", action="store_true")
    p_bench.add_argument("--comparison", action="store_true")
    p_bench.add_argument("--keep-wav", action="store_true", help="Keep the SSTV WAV of every mode.")
    p_bench.add_argument("--sample-rate", type=int, default=48000)
    p_bench.add_argument(
        "--include-experimental",
        action="store_true",
        help="Also bench experimental modes (Pasokon P3/P5/P7, PD-50); "
             "they are skipped by default.",
    )

    p_cmp = sub.add_parser("compare", help="Score two images (mae, psnr, correlation).")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")

    p_side = sub.add_parser("side", help="Side-by-side PNG of two images.")
    p_side.add_argument("a")
    p_side.add_argument("b")
    p_side.add_argument("out_pos", nargs="?", default=None, help="Output path (same as --out).")
    p_side.add_argument("--out", default=None)
    p_side.add_argument("--left", default="ORIGINAL")
    p_side.add_argument("--right", default="DECODED FROM SSTV AUDIO")

    p_card = sub.add_parser("card", help="Test card sized for a mode.")
    p_card.add_argument("--mode", default=DEFAULT_MODE)
    p_card.add_argument("--out", default="testcard.png")
    p_card.add_argument("--title", default=None)
    p_card.add_argument("--subtitle", default=None)

    p_fit = sub.add_parser("fit", help="Preview how an image fits a mode's frame.")
    p_fit.add_argument("image_path")
    p_fit.add_argument("--mode", default=DEFAULT_MODE)
    p_fit.add_argument("--out", default=None)
    p_fit.add_argument("--how", default="contain", choices=("contain", "cover", "stretch"))
    p_fit.add_argument("--background", default="0,0,0")

    p_live = sub.add_parser("live", help="Record microphone/virtual audio, then decode.")
    p_live.add_argument("--seconds", type=float, default=240.0)
    p_live.add_argument("--out-base", default="nSSTV_live.jpg")
    p_live.add_argument("--sample-rate", type=int, default=48000)
    p_live.add_argument("--device", default=None)
    p_live.add_argument("--output-mode", default="all", choices=OUTPUT_IMAGE_MODES)
    p_live.add_argument("--image-format", default="jpg", choices=image_format_choices)
    p_live.add_argument("--slant-search", type=float, default=0.03)
    p_live.add_argument("--auto-levels", action="store_true")
    p_live.add_argument("--denoise", default="off", choices=denoise_choices)
    p_live.add_argument("--diagnostics", action="store_true")

    p_modes = sub.add_parser("modes", help="List modes.")
    p_modes.add_argument("--custom-modes-json", action="append", default=None)

    return parser


def cli_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    def _common():
        """Keyword args shared by the decode-all / batch / roundtrip commands."""
        return dict(
            output_mode=args.output_mode,
            image_format=args.image_format,
            slant_search=args.slant_search,
            custom_modes_json=args.custom_modes_json,
            auto_levels=args.auto_levels,
            denoise=args.denoise,
            diagnostics=args.diagnostics,
        )

    try:
        if args.command == "decode":
            out_base = args.out_base or _default_output_base_for_audio(args.audio_path)
            report = args.report
            if report is None and args.diagnostics:
                report = os.path.splitext(out_base)[0] + "_report.json"

            info = decode_audio_to_images_v2(
                audio_path=args.audio_path,
                output_base=out_base,
                output_mode=args.output_mode,
                image_format=args.image_format,
                slant_search=args.slant_search,
                custom_modes_json=args.custom_modes_json,
                forced_mode_name=args.force_mode,
                forced_vis_end=args.vis_end,
                first_sync_time=args.first_sync_time,
                freq_offset_hz=args.freq_offset_hz,
                auto_levels=args.auto_levels,
                denoise=args.denoise,
                write_sidecar=not args.no_sidecar,
                diagnostics=args.diagnostics,
                report_path=report,
                line_structure=args.line_structure,
            )

            pprint(info)

            if info["detect_mode"] and "error" in info["detect_mode"]:
                _print_detect_hint(info["detect_mode"])
                return 1
            if info["decode_image"] and "error" in info["decode_image"]:
                return 1
            return 0

        if args.command == "decode-all":
            info = decode_all_audio_to_images(
                audio_path=args.audio_path,
                out_dir=args.out_dir,
                write_sidecar=not args.no_sidecar,
                forced_mode_name=args.force_mode,
                report_path=args.report,
                **_common(),
            )

            pprint(info)
            return 0 if info.get("decoded_count", 0) > 0 else 1

        if args.command == "batch":
            info = batch_decode(
                input_dir=args.input_dir,
                out_dir=args.out_dir,
                recursive=not args.no_recursive,
                write_sidecar=not args.no_sidecar,
                continue_on_error=not args.stop_on_error,
                **_common(),
            )

            pprint(info)
            return 0 if info.get("error_count", 0) == 0 else 1

        if args.command == "encode":
            wav_out = args.wav_out
            if wav_out is None and args.mp3_out is None:
                wav_out = _default_encoded_wav_for_image(args.image_path)

            info = encode_image_to_sstv_audio_v2(
                image_path=args.image_path,
                wav_out=wav_out,
                mp3_out=args.mp3_out,
                mode_name=args.mode,
                sample_rate=args.sample_rate,
                amplitude=args.amplitude,
                mp3_bitrate=args.mp3_bitrate,
                custom_modes_json=args.custom_modes_json,
                vis_code=args.vis_code,
                include_vis=not args.no_vis,
                pre_silence=args.pre_silence,
                post_silence=args.post_silence,
                fail_on_mp3_error=args.strict_mp3,
                caption=args.caption,
                caption_position=args.caption_position,
                image_fit=args.fit,
                image_background=_parse_color(args.background),
                line_structure=args.line_structure,
            )

            pprint(info)

            if info.get("include_vis") and not info.get("vis_header_check", {}).get("ok", False):
                return 1
            if args.strict_mp3 and info.get("mp3_error"):
                return 1
            return 0

        if args.command == "roundtrip":
            encoded_wav_out = args.encoded_wav_out or _default_roundtrip_wav_for_audio(args.audio_path)

            info = _roundtrip_first_image(
                args.audio_path,
                decode_kwargs={"out_dir": args.out_dir, "write_sidecar": True, **_common()},
                encoded_wav_out=encoded_wav_out,
                encoded_mp3_out=args.encoded_mp3_out,
                sample_rate=args.sample_rate,
                amplitude=args.amplitude,
                mp3_bitrate=args.mp3_bitrate,
                custom_modes_json=args.custom_modes_json,
            )

            if info is None:
                return 1

            if not info["encode_first_image"].get("vis_header_check", {}).get("ok", False):
                return 1

            return 0

        if args.command == "rt":
            result = rt(
                args.image_path,
                out_dir=args.out_dir,
                mode=args.mode,
                fit=args.fit,
                background=_parse_color(args.background),
                styles=args.styles,
                fmt=args.image_format,
                denoise=args.denoise,
                levels=args.auto_levels,
                comparison=not args.no_comparison,
                line_structure=getattr(args, "line_structure", "standard"),
            )

            print()
            print("Roundtrip result")
            print("================")
            print(f"image       : {result['image']}")
            print(f"mode        : {result['mode']} ({result.get('detected_mode')})")
            print(f"wav         : {result['wav']} ({result['wav_seconds']:.2f}s)")
            print(f"decoded     : {result.get('raw')}")
            print(f"comparison  : {result.get('comparison')}")
            if result.get("metrics"):
                m = result["metrics"]
                print(f"mae         : {m['mae']:.2f} / 255")
                print(f"psnr        : {m['psnr']:.2f} dB")
                print(f"correlation : {m['correlation']:.4f}")
            print(f"output dir  : {result['out_dir']}")

            return 0 if result["ok"] else 1

        if args.command == "bench":
            mode_list = None
            if args.modes:
                mode_list = [m.strip() for m in args.modes.split(",") if m.strip()]

            summary = bench(
                args.image_path,
                modes=mode_list,
                out_dir=args.out_dir,
                fit=args.fit,
                background=_parse_color(args.background),
                denoise=args.denoise,
                levels=args.auto_levels,
                comparison=args.comparison,
                keep_wav=args.keep_wav,
                sample_rate=args.sample_rate,
                count=args.count,
                seed=args.seed,
                include_experimental=args.include_experimental,
            )

            return 0 if summary["ok_count"] > 0 else 1

        if args.command == "compare":
            pprint(compare(args.a, args.b))
            return 0

        if args.command == "side":
            out = _first_not_none(args.out, args.out_pos)
            print(side(args.a, args.b, out=out, left=args.left, right=args.right))
            return 0

        if args.command == "card":
            print(card(args.mode, args.out, title=args.title, subtitle=args.subtitle))
            return 0

        if args.command == "fit":
            print(fit(args.image_path, mode=args.mode, out=args.out,
                      how=args.how, background=_parse_color(args.background)))
            return 0

        if args.command == "testcard":
            registry = make_registry(args.custom_modes_json)
            path = make_testcard_for_mode(
                args.out,
                mode_name=args.mode,
                registry=registry,
                title=args.title,
                subtitle=args.subtitle,
            )
            print(f"Wrote testcard: {path}")
            return 0

        if args.command == "live":
            info = live_record_then_decode(
                seconds=args.seconds,
                out_base=args.out_base,
                sample_rate=args.sample_rate,
                device=args.device,
                output_mode=args.output_mode,
                image_format=args.image_format,
                slant_search=args.slant_search,
                auto_levels=args.auto_levels,
                denoise=args.denoise,
                diagnostics=args.diagnostics,
            )

            pprint(info)

            if info["detect_mode"] and "error" in info["detect_mode"]:
                _print_detect_hint(info["detect_mode"])
                return 1
            if info["decode_image"] and "error" in info["decode_image"]:
                return 1
            return 0

        if args.command == "modes":
            registry = make_registry(args.custom_modes_json)

            print("nSSTV modes with image layouts:")

            for name in registry.supported_encoder_modes():
                code = registry.find_vis_code_for_mode(name)
                code_text = f"VIS {code}" if code is not None else "no VIS"
                layout = registry.get_layout(name)
                flag = " | experimental" if registry.modes[name].experimental else ""
                print(
                    f"  {name} ({code_text}) | "
                    f"{layout.width}x{layout.height} | "
                    f"{layout.color} | "
                    f"line {layout.line_seconds * 1000:.3f} ms{flag}"
                )

            return 0

        parser.print_help()
        return 2

    except Exception as e:
        print()
        print("ERROR:")
        print(str(e))
        return 1


def _print_detect_hint(detect_info):
    """
    Tell the user what to try next when no VIS code could be trusted.

    The line-rate report says which modes the audio's sync pulses fit, so a
    damaged header becomes a choice instead of a dead end.
    """
    if not isinstance(detect_info, dict):
        return

    hint = detect_info.get("hint")

    if hint:
        print()
        print(hint)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv:
        return cli_main(argv)

    return script_main()


__all__ = [
    "__version__",
    "__author__",
    "OUTPUT_IMAGE_MODES",
    "LossyMP3Warning",
    "UnknownModeError",
    "ProtocolConstants",
    "SSTVMode",
    "ImageLayout",
    "ModeRegistry",
    "EXPERIMENTAL_MODES",
    "EXPERIMENTAL_DECODE_ALL_MODES",
    "is_experimental_mode",
    "experimental_notice",
    "ToneDetector",
    "VISHeaderReader",
    "ToneSynth",
    "SSTVEncoder",
    "FMDemodulator",
    "ImageDecoder",
    "SSTVDecoder",
    "make_registry",
    "decode_audio_to_images",
    "encode_image_to_sstv_audio",
    "load_audio_mono",
    "write_wav_file",
    "write_mp3_file",
    "write_audio_file",
    "find_ffmpeg",
    "ffmpeg_is_available",

    "decode_audio_to_images_v2",
    "decode_all_audio_to_images",
    "decode_all",
    "cut",
    "join",
    "batch_decode",
    "encode_image_to_sstv_audio_v2",
    "decode",
    "encode",
    "make_testcard",
    "make_testcard_for_mode",
    "add_caption_to_image",
    "prepare_image_for_mode",
    "auto_levels_rgb",
    "denoise_rgb",
    "estimate_decode_quality",
    "save_decode_report",
    "make_diagnostics_image",
    "live_record_then_decode",
    "compare",
    "side",
    "rt",
    "bench",
    "modes",
    "random_modes",
    "card",
    "fit",
    "dec",
    "enc",
    "info",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
