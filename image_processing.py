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

class TextOverlay:
    """
    Specification for a single text overlay stamped onto an image.

    text           the string to draw (may contain newlines).
    position       Either a named 3x3 grid position from _OVERLAY_POSITIONS
                   (e.g. "bottom-right", "top-center", "middle-left"), OR
                   an (x, y) coordinate pair for free placement anywhere on
                   (or off) the image. Each of x, y is one of:
                     - a "NN%" string, e.g. "50%" - always a percentage of
                       the image's width/height, however `position_units`
                       is set. Unambiguous - use this if in doubt.
                     - a plain number, read according to `position_units`.
                   See `anchor` for what point of the text (x, y) refers to.
    position_units How to read a plain (non-percent-string) number in
                   `position`: "px" (always absolute pixels), "fraction"
                   (always 0..1 of the image size), or None (default) to
                   auto-infer - a float in 0..1 means fraction, anything
                   else means pixels. The auto-infer default is convenient
                   but ambiguous at the edges (e.g. y=1.0 reads as "100%
                   down", not "1 pixel down"); set this explicitly, or use
                   a "NN%" string, when that matters. Ignored for named
                   positions and for "NN%" strings.
    anchor         Only used when `position` is an (x, y) pair. One of the
                   3x3 grid names, saying which point of the text's box
                   sits at (x, y): "top-left" pins the box's top-left
                   corner there, "bottom-right" pins its bottom-right
                   corner there, etc. Defaults to "middle-center" - (x, y)
                   is the center of the text. Ignored for named positions.
    font_size      point size; None = auto based on image size.
    text_color     (R, G, B) tuple, 0-255.
    bg_color       (R, G, B) tuple for the bar behind the text.
    padding        pixels of bar margin around the text. For named
                   positions this also nudges the text in from the image
                   edge; for (x, y) positions it only pads the bar/text box.
    opacity        0..1; <1 produces a translucent overlay.
    rotation       degrees; 0/90/180/270 or arbitrary.  Rotates the
                   bar+text layer around its own center.
    font_style     "normal", "bold", "italic", "bold_italic".
    font_family    path to a TTF/TTC/OTF file, or None to use the
                   built-in font candidate list.
    bar_style      "solid" (default), "outline" (no fill, just a border),
                   or "none" (no bar at all - text floats on the image).
    outline_color  (R, G, B) tuple for a stroke drawn around each glyph,
                   or None for no stroke.  Handy when bar_style="none":
                   a contrasting outline keeps the text (any color)
                   readable over a busy image without a background box.
    outline_width  stroke width in pixels; None = auto (scaled to
                   font_size) when outline_color is set, 0 otherwise.

    Examples
    --------
    Named grid position (as before)::

        TextOverlay(text="WB1GCM", position="bottom-right")

    Free placement by percentage - unambiguous regardless of the numbers
    involved - centered horizontally, 30% of the way down::

        TextOverlay(text="CQSSTV", position=("50%", "30%"))

    Free placement by exact pixel, pinned by its top-left corner - handy
    for lining several texts up against each other. `position_units="px"`
    makes the "always pixels" reading explicit instead of relying on
    auto-inference::

        TextOverlay(text="DE WB1GCM", position=(20, 20), anchor="top-left",
                    position_units="px")
    """
    text: str = ""
    position: str = "bottom-right"
    position_units: str = None
    anchor: str = None
    font_size: int = None
    text_color: tuple = (255, 255, 255)
    bg_color: tuple = (0, 0, 0)
    padding: int = 10
    opacity: float = 1.0
    rotation: float = 0.0
    font_style: str = "normal"
    font_family: str = None
    bar_style: str = "solid"
    outline_color: tuple = None
    outline_width: int = None


def add_texts_to_image(image_path, out_path=None, overlays=None, context=None, ctx=None):
    """
    Stamp multiple text overlays onto an image in a single pass.

    overlays is a list of TextOverlay objects (or dicts with the same
    keys).  Each overlay is rendered onto its own RGBA layer, then all
    layers are alpha-composited onto the source image in order.

    This is the right API for stamping a callsign AND a timestamp AND a
    location label onto the same image - each at its own position,
    rotation, font style, and opacity.

    Example:
        overlays = [
            TextOverlay(text="DE {callsign} GRID {grid}", position="bottom-right",
                        font_style="bold", opacity=0.85),
            TextOverlay(text="2024-01-15 14:32 UTC", position="top-left",
                        font_size=14, font_style="italic"),
            TextOverlay(text="QTH: 35.6N 139.7E", position="bottom-left",
                        bar_style="outline"),
        ]
        add_texts_to_image("photo.jpg", "out.jpg", overlays, ctx={"callsign": "N0CALL", "grid": "FN31"})
    """
    if overlays is None:
        overlays = []

    ctx_dict = _first_not_none(ctx, context)

    # Accept dicts too for JSON-friendliness.
    normalized = []
    for ov in overlays:
        if isinstance(ov, TextOverlay):
            normalized.append(ov)
        elif isinstance(ov, dict):
            normalized.append(TextOverlay(**ov))
        else:
            raise TypeError(f"overlay must be TextOverlay or dict, got {type(ov)}")

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if out_path is None:
        base, ext = os.path.splitext(image_path)
        out_path = base + "_overlaid" + (ext or ".jpg")

    if not normalized:
        # Nothing to do; just save a copy.
        save_image_file(img, out_path)
        return out_path

    # Composite each overlay onto an RGBA copy of the source.
    canvas = img.convert("RGBA")
    for ov in normalized:
        layer = _draw_text_overlay(ov, w, h, context=ctx_dict)
        canvas = Image.alpha_composite(canvas, layer)

    save_image_file(canvas.convert("RGB"), out_path)
    return out_path


def add_callsign_to_image(
        image_path,
        out_path=None,
        callsign="",
        position="bottom-right",
        font_size=None,
        text_color=(255, 255, 255),
        bg_color=(0, 0, 0),
        padding=10,
        opacity=1.0,
        rotation=0.0,
        font_style="bold",
        font_family=None,
        bar_style="none",
        outline_color=(255, 255, 255),
        outline_width=None,
):
    """
    Stamp a callsign onto an image before encoding.

    position is the 3x3 grid from _OVERLAY_POSITIONS (top-left,
    top-center, top-right, middle-left, middle-center, middle-right,
    bottom-left, bottom-center, bottom-right).  Defaults to bottom-right,
    the traditional SSTV callsign corner.

    There is no background box by default (bar_style="none") - the
    callsign floats directly on the image, big and bold like a classic
    CQSSTV testcard signature.  Pass bar_style="solid" or "outline" to
    bring a box back.

    The callsign is bold, large, and outlined by default so it stays
    readable without a box: font_style defaults to "bold"; font_size,
    when not given, is auto-sized relative to the image (much bigger than
    a regular caption); and outline_color draws a contrasting stroke
    (white by default) around the glyphs. text_color controls the fill
    color of the letters themselves and can be set to anything - e.g.
    text_color=(30, 90, 220) for blue text with the default white
    outline. Set outline_color=None for plain, unoutlined text.

    opacity (0..1) is applied to the text (and bar, if any); <1 gives a
    translucent overlay so the picture shows through.  Useful when
    stamping on top of busy photo areas.

    rotation is in degrees (0/90/180/270 or arbitrary).  Non-zero rotates
    the text (+bar) layer around its center before compositing onto the
    image.

    font_style is one of "normal", "bold", "italic", "bold_italic".
    font_family, if given, is a path to a TTF/TTC/OTF file that overrides
    the default font search.

    For multiple text overlays on the same image, use add_texts_to_image()
    instead - it accepts a list of TextOverlay objects and composites them
    in a single pass.
    """
    callsign = str(callsign).strip()

    if not callsign:
        # Nothing to do; just save a copy if asked.
        if out_path is not None:
            img = Image.open(image_path).convert("RGB")
            save_image_file(img, out_path)
            return out_path
        return image_path

    if font_size is None:
        with Image.open(image_path) as _im:
            _w, _h = _im.size
        # Big, testcard-style lettering - noticeably bigger than the
        # generic ~0.045 auto-size used for plain text overlays, so the
        # callsign reads clearly at a glance without needing a box.
        font_size = max(32, int(min(_w, _h) * 0.13))

    overlay = TextOverlay(
        text=callsign,
        position=position,
        font_size=font_size,
        text_color=text_color,
        bg_color=bg_color,
        padding=padding,
        opacity=opacity,
        rotation=rotation,
        font_style=font_style,
        font_family=font_family,
        bar_style=bar_style,
        outline_color=outline_color,
        outline_width=outline_width,
    )

    return add_texts_to_image(image_path, out_path=out_path, overlays=[overlay])


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

    position accepts the classic "top" / "bottom" plus a 3x3 grid:
    top-left, top-center, top-right, middle-left, middle-center,
    middle-right, bottom-left, bottom-center, bottom-right.  For the
    middle row the caption floats on a small bar instead of a full-width
    band so it doesn't cover the whole image.
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
    vert, horiz = _resolve_overlay_position(position)

    # Decide the bar geometry based on the vertical band.
    if vert == "top":
        y0 = 0
        y_text = padding
    elif vert == "bottom":
        y0 = h - bar_h
        y_text = h - bar_h + padding
    else:  # middle
        y0 = (h - bar_h) // 2
        y_text = y0 + padding

    # Decide horizontal geometry based on the horizontal alignment.
    # For "top" / "bottom", the bar still spans the full width (back-compat
    # with the old API).  For "middle" rows the bar only covers the text
    # plus padding so the picture is still visible around it.
    if vert == "middle":
        bar_w = tw + 2 * padding

        if horiz == "left":
            x0 = padding
        elif horiz == "right":
            x0 = w - bar_w - padding
        else:  # center
            x0 = (w - bar_w) // 2

        x_text = x0 + padding
        draw.rectangle([x0, y0, x0 + bar_w, y0 + bar_h], fill=bg_color)
    else:
        # Full-width band for top/bottom (matches old behaviour).
        if horiz == "left":
            x_text = padding
        elif horiz == "right":
            x_text = w - tw - padding
        else:  # center
            x_text = (w - tw) // 2

        draw.rectangle([0, y0, w, y0 + bar_h], fill=bg_color)

    # Account for bbox[0] (the glyph origin x-offset) when placing text so
    # the visible pixels actually line up with the requested alignment.
    draw.text((x_text - bbox[0], y_text - bbox[1]), text, fill=text_color, font=font)

    save_image_file(img, out_path)
    return out_path


def format_overlay_text(text, ctx=None):
    """
    Format template variables in overlay text (e.g. {callsign}, {grid}, {datetime}, {mode}, {report}, {freq_hz}).
    """
    if not text or "{" not in text:
        return text
    ctx = ctx or {}
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    defaults = {
        "callsign": ctx.get("callsign", "N0CALL"),
        "grid": ctx.get("grid", "FN31"),
        "datetime": now,
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "time": time.strftime("%H:%M:%S UTC", time.gmtime()),
        "mode": ctx.get("mode_name", ctx.get("mode", "SSTV")),
        "report": ctx.get("report", "599"),
        "freq_hz": f"{ctx.get('rig_freq_hz', 14230000) / 1e6:.3f} MHz" if ctx.get("rig_freq_hz") else "14.230 MHz",
    }
    try:
        return text.format(**defaults)
    except Exception:
        return text


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
        residual_curve=None,
        sync_pulses=None,
):
    """
    Diagnostic spectrogram with optional vertical markers.

    residual_curve: optional dict with keys:
        - "linear": list of (line_index, expected_time) tuples for the
          linear fit, OR a callable(line_index) -> time.
        - "spline": same shape, the spline fit.
        - "pulses": list of (line_index, measured_time) tuples.
        When provided, a residual subplot is drawn below the spectrogram
        showing how the spline fit diverges from the linear fit at each
        sync pulse - this is the visual signature of crystal drift on
        long modes.

    sync_pulses: list of (line_index, measured_time) tuples - just the
        raw sync pulse detections, drawn as dots on the residual subplot.
        (Comes from ImageDecoder._detect_sync_pulses + _fit_timing.)
    """
    if t_end is None:
        t_end = len(samples) / fs

    # Make room for the residual subplot when one is requested.
    has_residual = bool(residual_curve or sync_pulses)
    spec_h = height - 60 - (160 if has_residual else 0)

    spec = make_spectrogram_image(
        samples=samples,
        fs=fs,
        t_start=t_start,
        t_end=t_end,
        width=width,
        height=spec_h,
    )

    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    canvas.paste(spec, (0, 60))

    draw = ImageDraw.Draw(canvas)

    font = _font(18)
    small = _font(13)
    tiny = _font(11)

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
        draw.line([x, 60, x, 60 + spec_h], fill=color, width=2)

        if label:
            draw.text((min(x + 4, width - 120), 62), label, fill=color, font=small)

    # Residual subplot: shows the per-line timing residual after fit.
    # This is what visualises crystal drift: a flat line means no drift,
    # a curve means the crystal's ppm wandered during the transmission.
    if has_residual:
        sub_y0 = 60 + spec_h + 10
        sub_h = 150
        sub_x0 = 20
        sub_w = width - 40

        # Frame.
        draw.rectangle([sub_x0, sub_y0, sub_x0 + sub_w, sub_y0 + sub_h],
                       outline=(80, 80, 80), width=1)
        draw.text((sub_x0 + 6, sub_y0 - 16), "Timing residual (sync drift)",
                  fill=(200, 200, 200), font=small)

        rc = residual_curve or {}

        pulses = sync_pulses or rc.get("pulses") or []
        linear = rc.get("linear")  # list of (k, t) or callable
        spline = rc.get("spline")

        # Determine the line-index range we'll plot.
        if pulses:
            k_min = min(k for k, _ in pulses)
            k_max = max(k for k, _ in pulses)
        else:
            k_min, k_max = 0, 1

        k_span = max(1, k_max - k_min)

        # Determine residual range.
        residuals = []
        for k, t in pulses:
            t_lin = linear(k) if callable(linear) else None
            if t_lin is None and linear:
                # linear is a list of (k, t); find closest.
                t_lin = min((tt for kk, tt in linear if kk <= k), default=None,
                            key=lambda tt: 0)
            t_spl = spline(k) if callable(spline) else None

            if t_lin is not None:
                residuals.append(("lin", k, t - t_lin))
            if t_spl is not None:
                residuals.append(("spl", k, t - t_spl))

        if not residuals:
            draw.text((sub_x0 + 20, sub_y0 + sub_h // 2),
                      "No sync pulse data available",
                      fill=(120, 120, 120), font=small)
        else:
            abs_max = max(abs(r) for _, _, r in residuals) or 1e-3
            # Scale: 1ms = 1 pixel up to +/- 50ms.
            scale_y = min(sub_h / 2 / max(abs_max * 1000, 1.0), sub_h / 2 / 50.0)
            zero_y = sub_y0 + sub_h // 2

            # Zero line.
            draw.line([sub_x0, zero_y, sub_x0 + sub_w, zero_y],
                      fill=(80, 80, 80), width=1)
            draw.text((sub_x0 + 4, zero_y - 14), "0 ms", fill=(120, 120, 120), font=tiny)

            # Plot linear residuals (grey) and spline residuals (cyan).
            for kind, color in (("lin", (160, 160, 160)), ("spl", (80, 220, 220))):
                pts = [(k, r) for knd, k, r in residuals if knd == kind]
                if not pts:
                    continue
                px_prev = py_prev = None
                for k, r in pts:
                    px = sub_x0 + int((k - k_min) / k_span * sub_w)
                    py = zero_y - int(r * 1000 * scale_y)  # r is in seconds
                    py = max(sub_y0, min(sub_y0 + sub_h, py))
                    if px_prev is not None:
                        draw.line([px_prev, py_prev, px, py], fill=color, width=1)
                    px_prev, py_prev = px, py

            # Mark the raw pulses as dots.
            for k, t in pulses:
                px = sub_x0 + int((k - k_min) / k_span * sub_w)
                py = zero_y
                draw.rectangle([px - 1, py - 1, px + 1, py + 1],
                               fill=(220, 220, 80))

            # Legend.
            lx = sub_x0 + sub_w - 200
            ly = sub_y0 + 6
            draw.rectangle([lx, ly, lx + 8, ly + 8], fill=(160, 160, 160))
            draw.text((lx + 12, ly - 2), "linear fit", fill=(160, 160, 160), font=tiny)
            draw.rectangle([lx, ly + 14, lx + 8, ly + 22], fill=(80, 220, 220))
            draw.text((lx + 12, ly + 12), "spline fit", fill=(80, 220, 220), font=tiny)
            draw.rectangle([lx, ly + 28, lx + 8, ly + 36], fill=(220, 220, 80))
            draw.text((lx + 12, ly + 26), "sync pulses", fill=(220, 220, 80), font=tiny)

    return save_image_file(canvas, out_path)


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


def _draw_text_overlay(overlay, img_w, img_h, context=None):
    """Render a single TextOverlay onto a transparent RGBA layer.

    Returns an RGBA image the same size as the source, with the overlay
    composited at the right position and rotation.  Caller alpha-blends
    this onto the source.
    """
    layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    text = format_overlay_text(str(overlay.text), ctx=context)
    if not text:
        return layer

    font_size = overlay.font_size
    if font_size is None:
        font_size = max(20, int(min(img_w, img_h) * 0.045))

    font = _font(font_size, style=overlay.font_style, family=overlay.font_family)

    stroke_width = overlay.outline_width
    if stroke_width is None:
        stroke_width = max(2, font_size // 12) if overlay.outline_color else 0

    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = int(overlay.padding)
    bar_h = th + 2 * pad
    bar_w = tw + 2 * pad

    pos = _normalize_position(overlay.position)
    if _is_free_position(pos):
        x0, y0 = _resolve_free_position(
            pos, overlay.anchor, overlay.position_units,
            bar_w, bar_h, img_w, img_h,
        )
    else:
        vert, horiz = _resolve_overlay_position(pos)

        if vert == "top":
            y0 = pad  # small margin from edge
        elif vert == "bottom":
            y0 = img_h - bar_h - pad
        else:  # middle
            y0 = (img_h - bar_h) // 2

        if horiz == "left":
            x0 = pad
        elif horiz == "right":
            x0 = img_w - bar_w - pad
        else:  # center
            x0 = (img_w - bar_w) // 2

    x_text = x0 + pad - bbox[0]
    y_text = y0 + pad - bbox[1]

    alpha = int(round(max(0.0, min(1.0, overlay.opacity)) * 255))
    if overlay.bg_color:
        bg_rgba = (int(overlay.bg_color[0]), int(overlay.bg_color[1]),
                   int(overlay.bg_color[2]), alpha)
    else:
        bg_rgba = (0, 0, 0, 0)
    text_rgba = (int(overlay.text_color[0]), int(overlay.text_color[1]),
                 int(overlay.text_color[2]), alpha)

    stroke_rgba = None
    if overlay.outline_color and stroke_width > 0:
        stroke_rgba = (int(overlay.outline_color[0]), int(overlay.outline_color[1]),
                       int(overlay.outline_color[2]), alpha)

    # Draw the bar.
    if overlay.bar_style == "outline":
        draw.rectangle([x0, y0, x0 + bar_w, y0 + bar_h], outline=bg_rgba, width=2)
    elif overlay.bar_style != "none":
        draw.rectangle([x0, y0, x0 + bar_w, y0 + bar_h], fill=bg_rgba)

    # Draw the text, with an optional stroke outline around each glyph so it
    # stays readable over a busy image even without a background bar.
    draw.text(
        (x_text, y_text), text, fill=text_rgba, font=font,
        stroke_width=stroke_width, stroke_fill=stroke_rgba,
    )

    # Rotate the layer around the bar's center if requested.
    rot = float(overlay.rotation) % 360.0
    if rot != 0.0:
        # Crop to the bar's bounding box plus padding, rotate, paste back.
        crop_pad = max(4, abs(rot) // 10)
        crop_box = (
            max(0, x0 - crop_pad),
            max(0, y0 - crop_pad),
            min(img_w, x0 + bar_w + crop_pad),
            min(img_h, y0 + bar_h + crop_pad),
        )
        cropped = layer.crop(crop_box)
        # Rotate with expand=True so the rotated content isn't clipped.
        rotated = cropped.rotate(rot, expand=True, resample=Image.BICUBIC)
        # Center the rotated layer on the original bar position.
        new_x = (x0 + bar_w // 2) - rotated.width // 2
        new_y = (y0 + bar_h // 2) - rotated.height // 2
        # Clear the layer and paste the rotated version.
        layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        layer.paste(rotated, (max(0, new_x), max(0, new_y)), rotated)

    return layer


def _resize_pil(img, size, resample=None):
    if resample is None:
        resample = Image.Resampling.LANCZOS

    return img.resize(size, resample)


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


def _font(size, style="normal", family=None):
    """
    Best font available at `size`, so captions do not shrink to specks on
    machines without Arial.

    style is one of "normal", "bold", "italic", "bold_italic".
    family, if given, is a path to a TTF/TTC/OTF file that overrides the
    candidate list (useful for embedded fonts or custom typography).
    """
    size = max(8, int(size))

    # If the user supplied an explicit font file, try it first.
    if family:
        try:
            return ImageFont.truetype(family, size)
        except Exception:
            import warnings as _w
            _w.warn(
                f"Could not load font family {family!r}; falling back to "
                "the default font search.",
                stacklevel=2,
            )

    # Pick the candidate list based on style.
    if style == "bold":
        candidates = FONT_CANDIDATES_BOLD
    elif style == "italic":
        candidates = FONT_CANDIDATES_ITALIC
    elif style == "bold_italic":
        candidates = FONT_CANDIDATES_BOLD_ITALIC
    else:
        candidates = FONT_CANDIDATES

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue

    # Fall back to regular style if the requested style isn't available.
    if style != "normal":
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


def _normalize_position(position):
    """Normalize position parameter: converts comma-separated coordinate strings
    like '10%, 90%' or '50, 100' into a free-position tuple ('10%', '90%')."""
    if isinstance(position, str) and "," in position:
        parts = [p.strip() for p in position.split(",", 1)]
        if len(parts) == 2 and all(_is_coord_value(p) for p in parts):
            return (parts[0], parts[1])
    return position


def _is_coord_value(value):
    """True if `value` is something _resolve_coord can turn into a
    pixel position: a plain number, or a "NN%" string."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.strip().rstrip("%"))
            return True
        except ValueError:
            return False
    return False


def _is_fraction(value):
    """True for a plain number meant as a 0..1 fraction of the image size
    rather than an absolute pixel coordinate. Used only for the legacy
    auto-inferred case (position_units=None); explicit units or a "NN%"
    string never need this guess."""
    return isinstance(value, float) and 0.0 <= value <= 1.0


def _is_free_position(position):
    """True when `position` is an (x, y) coordinate pair rather than a
    named grid position like "bottom-right". Each of x, y may be a
    number or a "NN%" string."""
    pos = _normalize_position(position)
    return (
        isinstance(pos, (tuple, list))
        and len(pos) == 2
        and all(_is_coord_value(v) for v in pos)
    )


def _resolve_coord(value, size, units):
    """
    Resolve one coordinate component (x or y) to an absolute pixel value.

    value  a number, or a "NN%" string.
    size   the image's width or height, for fraction/percent conversion.
    units  how to read a plain number: "px" (always pixels), "fraction"
           (always 0..1 of `size`), or None to auto-infer as before -
           a float in 0..1 is a fraction, anything else is pixels.
           A "NN%" string always means a percentage of `size`,
           regardless of `units`, since it can't be mistaken for pixels.
    """
    if isinstance(value, str):
        return float(value.strip().rstrip("%")) / 100.0 * size

    if units == "px":
        return float(value)
    if units == "fraction":
        return float(value) * size

    # units is None: auto-infer, same rule as before this was configurable.
    return value * size if _is_fraction(value) else float(value)


def _resolve_free_position(position, anchor, units, bar_w, bar_h, img_w, img_h):
    """
    Compute the (x0, y0) top-left corner of a bar/text box for a free-form
    (x, y) position, anywhere in (or outside) the image.

    x, y are each one of:
      - a "NN%" string, e.g. "50%" - always a percentage of the image's
        width/height. Unambiguous regardless of `units`.
      - a plain number, read according to `units` ("px", "fraction", or
        None to auto-infer: a float in 0..1 is a fraction, else pixels).

    anchor says which point of the text's bounding box (x, y) refers to:
    one of the 3x3 grid names ("top-left", "center", "bottom-right", ...).
    Defaults to "middle-center" - (x, y) is the center of the text - which
    is the most natural default when placing a text box's *point* anywhere
    on the canvas.
    """
    x, y = position

    anchor_x = _resolve_coord(x, img_w, units)
    anchor_y = _resolve_coord(y, img_h, units)

    anchor_vert, anchor_horiz = _resolve_overlay_position(anchor or "middle-center")

    if anchor_horiz == "left":
        x0 = anchor_x
    elif anchor_horiz == "right":
        x0 = anchor_x - bar_w
    else:  # center
        x0 = anchor_x - bar_w / 2.0

    if anchor_vert == "top":
        y0 = anchor_y
    elif anchor_vert == "bottom":
        y0 = anchor_y - bar_h
    else:  # middle
        y0 = anchor_y - bar_h / 2.0

    return int(round(x0)), int(round(y0))


def _resolve_overlay_position(position):
    """Return a (vertical, horizontal) tuple for a position name."""
    pos = _normalize_position(position)
    key = str(pos or "").strip().lower().replace("_", "-")

    if key not in _OVERLAY_POSITIONS:
        raise ValueError(
            f"Unknown overlay position {position!r}. "
            "Use standard grid positions (e.g. 'bottom-right', 'top-left'), "
            "intuitive aliases (e.g. 'watermark', 'stamp', 'callsign', 'header', 'footer', 'ur', 'll'), "
            "or coordinate pairs (e.g. (10, 20) or '10%, 90%')."
        )

    return _OVERLAY_POSITIONS[key]


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


def save_raw_jpg(rgb, out_path, quality=95):
    _ensure_parent(out_path)
    Image.fromarray(rgb).save(out_path, "JPEG", quality=quality)
    return out_path


def save_rgb_image(rgb, path, image_format=None, quality=95):
    return save_image_file(Image.fromarray(rgb), path, image_format=image_format, quality=quality)


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

