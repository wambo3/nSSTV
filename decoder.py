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
        slant_mode="auto",
        rig_hooks=None,
        raise_on_error=False,
):
    """
    SSTV audio -> image(s). Out path is automatic when omitted.

    line_structure:
        auto       choose from the audio (default)
        standard   one sync per image row, the published Wraase SC-2 timing
        rgb3       one sync per colour component

    slant_mode:
        linear   straight line fit (historical behaviour)
        spline   force a cubic smoothing spline through the sync pulses,
                 useful for long modes (PD-290, MR320) where the 12 MHz
                 crystal drifts ppm over the transmission
        auto     use spline only when residuals indicate visible drift
                 (default)

    raise_on_error:
        If True, raises RuntimeError when decoding fails (e.g. silence, noise,
        or unrecognised mode) instead of returning a result dictionary containing
        the error details.

    Short names (out, styles, fmt, levels, mode, slant, sidecar) and long names
    (output_base, output_mode, image_format, auto_levels, forced_mode_name,
    slant_search, write_sidecar) both work.

    rig_hooks : RigHooks or None
        Optional rig-control callbacks.  ``before_decode`` fires before the
        audio file is processed; ``after_decode`` fires once decoding
        completes.  Any dict returned by ``after_decode`` is stored under
        ``result["rig"]``.  See RigHooks for the full callback contract.
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

    rx_ctx = {"action": "decode", "audio_path": audio}
    _call_hook(rig_hooks, "before_decode", rx_ctx)

    result = decode_audio_to_images_v2(
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
        slant_mode=slant_mode,
    )

    rx_ctx["result"] = result
    _call_hook(rig_hooks, "after_decode", rx_ctx)

    if rx_ctx.get("rig"):
        result["rig"] = rx_ctx["rig"]

    if raise_on_error:
        err = (result.get("detect_mode") or {}).get("error") or (result.get("decode_image") or {}).get("error")
        if err:
            raise RuntimeError(f"SSTV decode failed: {err}")

    return result


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
        slant_mode="auto",
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
        slant_mode=slant_mode,
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

        # Build a residual-curve dict from the most recent decode_image's
        # ImageDecoder sync pulses + spline.  We need to peek inside the
        # ImageDecoder that ran inside decode_image_extended to do this.
        # If we don't have access, fall back to drawing just the spectrogram.
        residual_curve = None
        sync_pulses = None
        try:
            img_dec = getattr(decoder, "_last_image_decoder", None)
            if img_dec is not None:
                # Pull the linear fit parameters (a, b) and the spline.
                a = img_dec.info.get("first_line_sync")
                b_nominal = img_dec.info.get("nominal_line_seconds")
                b = img_dec.info.get("measured_line_seconds")
                spline = img_dec._timing_spline
                pulses = getattr(img_dec, "_last_pulses", None)

                if pulses is not None and a is not None and b is not None:
                    # Convert pulses (absolute times) to (line_index, time)
                    # pairs by snapping to the nearest integer line.
                    pulse_pairs = []
                    for t in pulses:
                        k = int(round((float(t) - a) / max(b, 1e-12)))
                        if 0 <= k < img_dec.layout.n_lines:
                            pulse_pairs.append((k, float(t)))

                    linear_fn = (lambda k, a=a, b=b: a + b * k)
                    spline_fn = (spline if spline is not None else None)

                    residual_curve = {
                        "linear": linear_fn,
                        "spline": spline_fn,
                        "pulses": pulse_pairs,
                    }
                    sync_pulses = pulse_pairs
        except Exception:
            pass

        result["diagnostics_path"] = make_diagnostics_image(
            samples=decoder.samples,
            fs=decoder.fs,
            out_path=diagnostics_path,
            t_start=t0,
            t_end=t1,
            markers=markers,
            title=f"nSSTV diagnostics | {detect.get('mode_name')}",
            residual_curve=residual_curve,
            sync_pulses=sync_pulses,
        )

    if report_path:
        result["report_path"] = save_decode_report(report_path, result)

    return result


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
        slant_mode="auto",
):
    """
    Extended image decode wrapper.

    This does not require replacing SSTVDecoder.decode_image.

    line_structure:
        auto       choose from the audio (default)
        standard   one sync per image row
        rgb3       one sync per colour component

    slant_mode:
        linear   straight line fit (historical behaviour)
        spline   force a cubic smoothing spline through the sync pulses
        auto     use spline only when the linear residual shows drift
                 that's big enough to visibly smear pixels (default)
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
        slant_mode=slant_mode,
    )

    # Expose the ImageDecoder on the SSTVDecoder so the diagnostics
    # builder can pull the sync pulses + spline residual curve.
    decoder._last_image_decoder = img

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


def decode_all(*args, **kwargs):
    """
    Alias for decode_all_audio_to_images().
    """
    return decode_all_audio_to_images(*args, **kwargs)


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
        # Extended VIS: header is twice as long when the mode uses 16-bit VIS.
        ext_code = registry.extended_vis_code_for_mode(vis_info["mode_name"])
        header_blocks = 2 if ext_code is not None else 1
        decoder.vis_end = decoder.vis_start + header_blocks * protocol.vis_header_seconds
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
        try:
            fs, samples = wavfile.read(path)
        except (struct.error, ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid or corrupt WAV file {path!r}: header truncated or corrupted ({e})"
            ) from e
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


def dec(*args, **kwargs):
    return decode(*args, **kwargs)

