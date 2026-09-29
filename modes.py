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
        # Per-mode extended VIS code (8 bits) emitted after the primary 0x23
        # for 16-bit-VIS MMSSTV modes. None for everything else.
        self.extended_vis_codes = {}
        # Per-mode N-VIS code (separate digital header, recorded for
        # documentation only - we still emit a standard VIS when one is set).
        self.nvis_codes = {}

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

        # MMSSTV modes derive their pixel_seconds from the layout's fastest
        # scan, since the SSTV Handbook only specifies the line rate.
        mmsstv_layouts = make_mmsstv_layouts()

        for name, _w, _h, _lpm, _kind, vis_kind, vis_value in _mmsstv_mode_specs():
            layout = mmsstv_layouts[name]
            pixel_s = _mmsstv_pixel_seconds(layout)

            self.add_mode(
                name=name,
                pixel_seconds=pixel_s,
                sync_seconds=layout.sync_seconds,
                porch_seconds=0.0,
                overwrite=True,
            )

            if vis_kind == "extended":
                # Primary standard VIS is 0x23 (35) - the extended byte is
                # stored on the mode so the encoder can append it after the
                # standard 8-bit VIS, and the decoder can look it up after
                # reading 0x23.
                self.extended_vis_codes[name] = int(vis_value)
            elif vis_kind == "nvis":
                self.nvis_codes[name] = int(vis_value)

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

        # MMSSTV modes:
        # - "extended" 16-bit VIS modes share primary VIS code 0x23 (35).
        #   The decoder reads the extended byte to pick the right mode.
        # - "nvis" modes use N-VIS, which in MMSSTV's actual implementation
        #   is just a standard VIS frame whose value is the N-VIS code.  The
        #   "N" stands for Narrow filtering at the receiver, not a different
        #   signal format.
        #
        # Collision handling: Some N-VIS codes overlap with standard VIS
        # codes for other modes (e.g., MP73-N N-VIS=2 collides with Robot B&W 8
        # VIS=1/2/3).  We do NOT overwrite the existing VIS code in that
        # case - the existing mode keeps it.  N-VIS modes that collide
        # are stored in self.nvis_codes only (not in self.vis_codes), and
        # the encoder will still emit the N-VIS code as the standard VIS
        # byte; the receiver uses --force-mode to disambiguate.
        #
        # FIX: EXTENDED_VIS_PRIMARY (35) must be registered exactly once.
        # The old code called add_vis_code(35, name, overwrite=True) for
        # every extended mode, leaving vis_codes[35] pointing only at the
        # last mode in _mmsstv_mode_specs() (MMSSTV MR320).  That caused
        # _mode_sync_sanity() to probe MR320's line time for every MMSSTV
        # transmission, rejecting MP73/MP115/MR90/etc. as invalid.
        # The actual mode resolution for extended VIS is done by
        # find_extended_vis_mode() in the decoder after reading the second
        # 10-slot frame, so vis_codes[35] just needs to exist as a sentinel
        # that tells the decoder "extended byte follows".  We store the
        # first extended mode name as the sentinel value and register it
        # only once; the decoder always overwrites it via extended byte
        # look-up before using it.
        _ext_primary_registered = False
        for name, _w, _h, _lpm, _kind, vis_kind, vis_value in _mmsstv_mode_specs():
            if vis_kind == "extended":
                if not _ext_primary_registered:
                    self.add_vis_code(EXTENDED_VIS_PRIMARY, name, overwrite=False)
                    _ext_primary_registered = True
                # Do NOT call add_vis_code again for subsequent extended modes;
                # that would overwrite the sentinel and point vis_codes[35] at
                # whichever mode happened to be last in the list.
            elif vis_kind == "nvis":
                existing = self.vis_codes.get(int(vis_value))
                if existing is None:
                    # No collision - register the N-VIS code as the standard
                    # VIS code for this mode.
                    self.add_vis_code(int(vis_value), name, overwrite=True)
                elif existing == name:
                    pass  # already registered
                else:
                    # Collision - keep the existing mode's VIS code.  The
                    # N-VIS mode is still encodable (the encoder uses
                    # find_vis_code_for_mode which checks nvis_codes too)
                    # but it won't be auto-detected; users need --force-mode.
                    pass

    def extended_vis_code_for_mode(self, mode_name):
        """Return the 8-bit extended VIS byte for a 16-bit-VIS MMSSTV mode, or None."""
        return self.extended_vis_codes.get(mode_name)

    def nvis_code_for_mode(self, mode_name):
        """Return the N-VIS code recorded for a mode, or None."""
        return self.nvis_codes.get(mode_name)

    def find_extended_vis_mode(self, extended_byte):
        """Look up a mode name by its 8-bit extended VIS byte."""
        for name, code in self.extended_vis_codes.items():
            if int(code) == int(extended_byte):
                return name
        return None

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

            # Validate dimensions and timing - these can crash the encoder
            # silently if we accept zero/negative values (e.g. division
            # by zero in fastest_pixel_seconds or _paint_channel).
            width = int(width)
            height = int(height)
            if width <= 0:
                raise ValueError(f"Custom mode width must be >= 1, got {width}")
            if height <= 0:
                raise ValueError(f"Custom mode height must be >= 1, got {height}")
            if line_seconds <= 0:
                raise ValueError(
                    f"Custom mode line_seconds must be > 0, got {line_seconds}"
                )
            if sync_seconds < 0:
                raise ValueError(
                    f"Custom mode sync_seconds must be >= 0, got {sync_seconds}"
                )
            if not channels:
                raise ValueError("Custom mode needs at least one channel")

            layout = ImageLayout(
                name=name,
                width=width,
                height=height,
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
            # Validate the caller-supplied layout too.
            if layout.width <= 0:
                raise ValueError(f"Custom layout width must be >= 1, got {layout.width}")
            if layout.height <= 0:
                raise ValueError(f"Custom layout height must be >= 1, got {layout.height}")
            if layout.line_seconds <= 0:
                raise ValueError(
                    f"Custom layout line_seconds must be > 0, got {layout.line_seconds}"
                )
            if layout.sync_seconds < 0:
                raise ValueError(
                    f"Custom layout sync_seconds must be >= 0, got {layout.sync_seconds}"
                )
            if not layout.channels:
                raise ValueError("Custom layout needs at least one channel")

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
        # Modes with a 16-bit extended VIS all share the primary 0x23 (35)
        # code - we report 35 here even though `vis_codes[35]` only stores
        # one of them; the extended byte (in self.extended_vis_codes) is
        # what disambiguates them on decode.
        if mode_name in self.extended_vis_codes:
            return EXTENDED_VIS_PRIMARY

        for code, name in sorted(self.vis_codes.items()):
            if name == mode_name:
                return code

        # N-VIS modes that collided with an existing VIS code are stored in
        # self.nvis_codes only.  The encoder still emits the N-VIS code as
        # a standard VIS byte.
        if mode_name in self.nvis_codes:
            return self.nvis_codes[mode_name]

        return None

    def supported_encoder_modes(self):
        return sorted(self.image_layouts)

    def supported_decoder_image_modes(self):
        return sorted(self.image_layouts)


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


def _wraase_sc2_180():
    """Wraase SC2-180: 320x256, RGB, 182 s over 256 lines, VIS 55."""
    return _wraase_sc2("Wraase SC-2 180", 0.000734375)


def make_mmsstv_layouts():
    """Build every MMSSTV ImageLayout, keyed by mode name."""
    out = {}

    for name, width, height, lpm, kind, _vk, _vv in _mmsstv_mode_specs():
        out[name] = _mmsstv_layout(name, width, height, lpm, kind)

    return out


def _mmsstv_mode_specs():
    """All MMSSTV modes as (name, width, height, lpm, builder_kind, vis_kind, vis_value).

    vis_kind is one of:
        "extended"  -> standard VIS 0x23 + extended byte (16-bit VIS)
        "nvis"      -> N-VIS code emitted as standard VIS (no extended byte)

    For "extended" modes, vis_value is the 7-bit data byte of the
    extended VIS (the SSTV Handbook lists values like 0x85 for MR180 -
    we strip the top parity bit and store 0x05 here).
    """
    return [
        # name, width, height, lpm, kind, vis_kind, vis_value
        ("MMSSTV MC110-N", 320, 256, 137.143, "mc", "nvis", 0x14),
        ("MMSSTV MC140-N", 320, 256, 109.389, "mc", "nvis", 0x15),
        ("MMSSTV MC180-N", 320, 256, 85.167, "mc", "nvis", 0x16),

        ("MMSSTV MP73-N", 320, 256, 210.526, "y420", "nvis", 0x02),
        ("MMSSTV MP140-N", 320, 256, 110.092, "y420", "nvis", 0x05),

        ("MMSSTV MP73", 320, 256, 210.526, "y420", "extended", 0x25),
        ("MMSSTV MP115", 320, 256, 133.038, "y420", "extended", 0x29),
        ("MMSSTV MP140", 320, 256, 110.092, "y420", "extended", 0x2A),
        ("MMSSTV MP175", 320, 256, 87.591, "y420", "extended", 0x2C),

        ("MMSSTV MR73", 320, 256, 419.141, "y420", "extended", 0x45),
        ("MMSSTV MR90", 320, 256, 340.619, "y420", "extended", 0x46),
        ("MMSSTV MR115", 320, 256, 266.489, "y420", "extended", 0x49),
        ("MMSSTV MR140", 320, 256, 218.858, "y420", "extended", 0x4A),
        ("MMSSTV MR175", 320, 256, 175.362, "y420", "extended", 0x4C),

        # SSTV Handbook lists 0x85/0x86/0x89/0x8A for MR180/MR240/MR280/MR320.
        # The top bit (0x80) is the parity bit; we store the 7-bit data byte.
        ("MMSSTV MR180", 640, 496, 330.306, "y420", "extended", 0x05),
        ("MMSSTV MR240", 640, 496, 248.293, "y420", "extended", 0x06),
        ("MMSSTV MR280", 640, 496, 212.277, "y420", "extended", 0x09),
        ("MMSSTV MR320", 640, 496, 185.960, "y420", "extended", 0x0A),
    ]


def _mmsstv_pixel_seconds(layout):
    """Compute the per-pixel duration of the fastest scan in a layout.

    Stored on the SSTVMode entry so `nSSTV modes` and the rest of the API
    can report it the same way as the built-in modes.
    """
    planes = list(layout.channels or ()) + list(layout.channels_odd or ())

    if not planes:
        return float("nan")

    return min(duration for _k, _o, duration in planes) / float(layout.width)


def _mmsstv_y420(name, line_seconds, width, height):
    """YCrCb 4:2:0 layout, Robot-36-style: one sync per row, alternating Cr/Cb.

    The Y scan is 2/3 of the remaining time after sync + porch + separator +
    chroma porch; the chroma scan gets the other 1/3. Same shape as Robot 36
    so a receiver that handles Robot 36 will accept the timing.
    """
    sync, porch, sep, cporch = 0.009, 0.003, 0.0045, 0.0015

    y = sync + porch
    remaining = line_seconds - y - sep - cporch
    y_scan = remaining * (2.0 / 3.0)
    c_scan = remaining - y_scan
    c = y + y_scan + sep + cporch

    return ImageLayout(
        name=name,
        width=width,
        height=height,
        color="ycrcb420",
        line_seconds=line_seconds,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=(("Y", y, y_scan), ("Cr", c, c_scan)),
        channels_odd=(("Y", y, y_scan), ("Cb", c, c_scan)),
    )


def _mmsstv_mc_n(name, line_seconds):
    """MMSSTV MC-N mode: 320x256 RGB, Martin-style 3-scan-per-line layout.

    Uses Martin M1's sync/porch/separation times so the mode still decodes
    cleanly on receivers that only know the Martin family. The scan
    duration is whatever is left after subtracting sync, porch, and two
    separators from `line_seconds`, split evenly across R/G/B.
    """
    sync, porch, sep = 0.004862, 0.000572, 0.000572

    scan_total = line_seconds - sync - porch - 2 * sep
    scan = scan_total / 3.0

    g = sync + porch
    b = g + scan + sep
    r = b + scan + sep

    return ImageLayout(
        name=name,
        width=320,
        height=256,
        color="rgb",
        line_seconds=line_seconds,
        sync_seconds=sync,
        sync_offset=0.0,
        channels=(("G", g, scan), ("B", b, scan), ("R", r, scan)),
    )


def _mmsstv_layout(name, width, height, lpm, kind):
    """Build an ImageLayout for one MMSSTV mode based on its kind."""
    line_seconds = 60.0 / float(lpm)

    if kind == "mc":
        return _mmsstv_mc_n(name, line_seconds)

    if kind == "y420":
        return _mmsstv_y420(name, line_seconds, width, height)

    raise ValueError(f"Unknown MMSSTV kind: {kind!r}")


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

    # MMSSTV modes are added on top of the standard built-ins.
    layouts.extend(make_mmsstv_layouts().values())

    return {layout.name: layout for layout in layouts}


def make_registry(custom_modes_json=None):
    registry = ModeRegistry()

    if custom_modes_json:
        if isinstance(custom_modes_json, (str, os.PathLike)):
            custom_modes_json = [custom_modes_json]

        for path in custom_modes_json:
            if not os.path.exists(path):
                # A missing custom-modes JSON is recoverable: we still have
                # all the built-in modes, so decode/encode can continue.
                # Warn rather than crash so a typo doesn't kill the whole
                # pipeline.
                import warnings as _warnings
                _warnings.warn(
                    f"Custom modes JSON {path!r} not found; "
                    "continuing with built-in modes only.",
                    stacklevel=2,
                )
                continue

            try:
                registry.load_custom_modes_json(path, overwrite=True)
            except (ValueError, OSError, json.JSONDecodeError) as e:
                import warnings as _warnings
                _warnings.warn(
                    f"Failed to load custom modes JSON {path!r}: {e}",
                    stacklevel=2,
                )

    return registry

