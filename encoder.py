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
        callsign=None,
        callsign_position="bottom-right",
        callsign_opacity=1.0,
        oversample=4,
        rig_hooks=None,
        overlays=None,
):
    """
    image -> SSTV audio. Out path is automatic when omitted.
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

    info = encode_image_to_sstv_audio_v2(
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
        callsign=callsign,
        callsign_position=callsign_position,
        callsign_opacity=callsign_opacity,
        oversample=oversample,
        overlays=overlays,
    )

    tx_ctx = {
        "action": "transmit",
        "wav_path": out,
        "mode_name": mode,
        "duration_seconds": info.get("duration_seconds"),
    }
    _call_hook(rig_hooks, "before_transmit", tx_ctx)

    # before_transmit is the PTT key-up point.  The actual audio playback
    # is the caller's responsibility (e.g. via sounddevice or aplay);
    # after_transmit fires immediately after so the caller can key down
    # as soon as playback ends.
    _call_hook(rig_hooks, "after_transmit", tx_ctx)

    if tx_ctx.get("rig"):
        info["rig"] = tx_ctx["rig"]

    return info


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
        oversample=4,
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

    oversample (default 4) resizes the source image to
    (width*oversample, height*oversample), averages the extra vertical
    rows back down to one per scan line, and emits per-sample tones by
    linearly interpolating along each line across the oversampled
    columns, so sub-pixel detail is preserved instead of collapsing to a
    staircase at the mode's pixel rate.  Set to 1 for the old
    nearest-neighbour behaviour.
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
        oversample=oversample,
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
        callsign=None,
        callsign_position="bottom-right",
        callsign_opacity=1.0,
        oversample=4,
        overlays=None,
):
    """
    Improved encode API with optional caption overlay and aspect-ratio fit.

    callsign overlays a callsign (e.g. "N0CALL") at callsign_position, one
    of the 3x3 grid positions: top-left/top-center/top-right,
    middle-left/middle-center/middle-right, bottom-left/bottom-center/
    bottom-right.  callsign_opacity (0..1) sets bar+text opacity for a
    translucent overlay so the picture shows through.

    oversample (default 4) is the sub-pixel oversampling factor used to
    fix the "image downsampled to mode dims before signal generation"
    problem.  Set to 1 to restore the old nearest-neighbour behaviour.
    """
    temp_overlay_path = None
    temp_caption_path = None

    try:
        source_image = image_path

        if overlays:
            base, ext = os.path.splitext(os.path.abspath(image_path))
            fd, temp_overlay_path = tempfile.mkstemp(prefix="_nsstv_overlay_", suffix=ext or ".png")
            os.close(fd)
            add_texts_to_image(image_path, temp_overlay_path, overlays=overlays)
            image_path = temp_overlay_path

        # Callsign is applied first so the caption (if any) wins on
        # overlapping pixels.
        if callsign:
            base, ext = os.path.splitext(os.path.abspath(image_path))
            temp_overlay_path = base + "_nsstv_callsign_tmp" + (ext or ".jpg")
            source_image = add_callsign_to_image(
                image_path=image_path,
                out_path=temp_overlay_path,
                callsign=callsign,
                position=callsign_position,
                opacity=callsign_opacity,
            )
            image_path = source_image

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
            oversample=oversample,
        )

        info["experimental"] = is_experimental_mode(mode_name)

        return info
    finally:
        # Clean up both temp files if we made them.
        for tmp in (temp_overlay_path, temp_caption_path):
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass


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


def write_mp3_file(path, fs, audio, bitrate="320k"):
    """
    Write MP3 using pydub/ffmpeg.

    Uses a temporary file first so failed exports do not leave empty MP3 files.
    """
    bitrate = _validate_mp3_bitrate(bitrate)
    warn_mp3_lossy("writing SSTV MP3 audio", path, bitrate=bitrate)

    configure_pydub_ffmpeg(require=True)
    AudioSegment = _get_audio_segment()
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


def warn_mp3_lossy(action=None, path=None, bitrate=None):
    msg = (
        "MP3 audio compression uses lossy frequency masking that alters sub-audible SSTV tones, "
        "resulting in noisier decoded pictures. WAV format is strongly recommended for SSTV transmissions."
    )

    if action:
        msg += f" Action: {action}."

    if path:
        msg += f" File: {path!r}."

    if bitrate:
        msg += f" Bitrate: {bitrate}; high bitrate mitigates loss, but uncompressed WAV is preferred."

    warnings.warn(msg, LossyMP3Warning, stacklevel=2)


def enc(*args, **kwargs):
    return encode(*args, **kwargs)

