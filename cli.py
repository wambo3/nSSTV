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
            slant_mode=args.slant_mode,
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
                slant_mode=args.slant_mode,
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
                callsign=args.callsign,
                callsign_position=args.callsign_position,
                callsign_opacity=args.callsign_opacity,
                oversample=args.oversample,
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
                callsign=args.callsign,
                callsign_position=args.callsign_position,
                callsign_opacity=args.callsign_opacity,
                oversample=args.oversample,
                slant_mode=args.slant_mode,
            )

            print()
            print("Roundtrip result")
            print("================")
            print(f"image       : {result['image']}")
            print(f"mode        : {result['mode']} ({result.get('detected_mode')})")
            print(f"wav         : {result['wav']} ({result['wav_seconds']:.2f}s)")
            print(f"decoded     : {result.get('raw')}")
            print(f"comparison  : {result.get('comparison')}")
            if result.get("callsign"):
                print(f"callsign    : {result['callsign']} @ {args.callsign_position}")
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
                callsign=args.callsign,
                callsign_position=args.callsign_position,
                callsign_opacity=args.callsign_opacity,
                oversample=args.oversample,
                slant_mode=args.slant_mode,
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

        if args.command in ("listen", "live-listen"):
            info = live_listen_and_decode(
                out_dir=args.out_dir,
                sample_rate=args.sample_rate,
                device=args.device,
                output_mode=args.output_mode,
                image_format=args.image_format,
                slant_search=args.slant_search,
                auto_levels=args.auto_levels,
                denoise=args.denoise,
                diagnostics=args.diagnostics,
                poll_seconds=args.poll_seconds,
                max_seconds=args.max_seconds,
                keep_wav=args.keep_wav,
            )

            print(f"\n{info['count']} image(s) decoded to {info['out_dir']}")
            for path in info["images"]:
                print(" ", path.get("raw"))
            return 0

        if args.command in ("transmit", "tx"):
            rig_hooks = None
            if args.rig_model or args.rigctld_host:
                rig_hooks = make_rigctl_hooks(
                    rig_model=args.rig_model,
                    rig_file=args.rig_file,
                    baud=args.baud,
                    rigctld_host=args.rigctld_host,
                    rigctld_port=args.rigctld_port,
                )
            elif args.gpio_pin is not None:
                rig_hooks = make_rpi_gpio_hooks(ptt_pin=args.gpio_pin)
            elif args.serial_port:
                rig_hooks = make_serial_dtr_rts_hooks(port=args.serial_port, line=args.serial_line)

            input_path = args.input_path
            ext = os.path.splitext(input_path)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".bmp"):
                tmp_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="tx_enc_")
                os.close(tmp_fd)
                try:
                    encode(input_path, wav_path, mode=args.mode, sample_rate=args.sample_rate)
                    tx_res = transmit(wav_path, sample_rate=args.sample_rate, device=args.device, rig_hooks=rig_hooks, mode_name=args.mode)
                finally:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
            else:
                tx_res = transmit(input_path, sample_rate=args.sample_rate, device=args.device, rig_hooks=rig_hooks, mode_name=args.mode)

            print(f"Transmission completed: {tx_res['duration_seconds']:.2f} seconds")
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
        p.add_argument(
            "--slant-mode",
            default="auto",
            choices=("linear", "spline", "auto"),
            help="linear: straight-line timing fit (historical). spline: cubic "
                 "smoothing spline for non-linear crystal drift on long modes "
                 "(PD-290, MR320). auto (default): spline only when residuals "
                 "show visible drift.",
        )
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
    p_encode.add_argument(
        "--caption-position",
        default="bottom",
        choices=(
            "top", "bottom",
            "top-left", "top-center", "top-centre", "top-right",
            "middle-left", "middle-center", "middle-centre", "middle-right",
            "bottom-left", "bottom-center", "bottom-centre", "bottom-right",
            "left", "right", "center", "centre", "middle",
        ),
        help="Where to put the caption (3x3 grid for finer control).",
    )
    p_encode.add_argument(
        "--callsign",
        default=None,
        help="Callsign (e.g. N0CALL) to overlay before encoding.",
    )
    p_encode.add_argument(
        "--callsign-position",
        default="bottom-right",
        choices=(
            "top-left", "top-center", "top-centre", "top-right",
            "middle-left", "middle-center", "middle-centre", "middle-right",
            "bottom-left", "bottom-center", "bottom-centre", "bottom-right",
            "top", "bottom", "left", "right", "center", "centre", "middle",
        ),
        help="Where to put the callsign overlay (default: bottom-right).",
    )
    p_encode.add_argument(
        "--callsign-opacity",
        type=float,
        default=1.0,
        help="Callsign overlay opacity 0..1 (1 = solid, <1 = translucent).",
    )
    p_encode.add_argument(
        "--oversample",
        type=int,
        default=4,
        help="Sub-pixel oversampling factor (default 4). Set to 1 to restore "
             "the old 'downsample to mode dims then hold' encoder behaviour.",
    )
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
    p_rt.add_argument("--callsign", default=None,
                      help="Stamp this callsign onto the source image before encoding.")
    p_rt.add_argument("--callsign-position", default="bottom-right",
                      choices=tuple(_OVERLAY_POSITIONS.keys()),
                      help="Where to place the callsign (default: bottom-right).")
    p_rt.add_argument("--callsign-opacity", type=float, default=1.0,
                      help="Callsign opacity 0..1 (default 1.0 = solid).")
    p_rt.add_argument("--oversample", type=int, default=4,
                      help="Encoder sub-pixel oversample factor (default 4).")
    p_rt.add_argument("--slant-mode", default="auto",
                      choices=("linear", "spline", "auto"),
                      help="Decoder slant correction mode (default auto).")

    p_bench = sub.add_parser("bench", help="Roundtrip across modes and rank them.")
    p_bench.add_argument("image_path")
    p_bench.add_argument("--modes", default=None, help="Comma-separated list, or 'random'; default: every mode.")
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
    p_bench.add_argument("--callsign", default=None,
                         help="Stamp this callsign onto the source image before each encode.")
    p_bench.add_argument("--callsign-position", default="bottom-right",
                         choices=tuple(_OVERLAY_POSITIONS.keys()),
                         help="Where to place the callsign (default: bottom-right).")
    p_bench.add_argument("--callsign-opacity", type=float, default=1.0,
                         help="Callsign opacity 0..1 (default 1.0 = solid).")
    p_bench.add_argument("--oversample", type=int, default=4,
                         help="Encoder sub-pixel oversample factor (default 4).")
    p_bench.add_argument("--slant-mode", default="auto",
                         choices=("linear", "spline", "auto"),
                         help="Decoder slant correction mode (default auto).")

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

    p_listen = sub.add_parser(
        "listen",
        aliases=["live-listen"],
        help="Listen continuously and decode SSTV transmissions as they arrive "
             "(unlike 'live', doesn't need a fixed duration up front).",
    )
    p_listen.add_argument("--out-dir", "--out", dest="out_dir", default="live_decoded")
    p_listen.add_argument("--sample-rate", "--rate", dest="sample_rate", type=int, default=48000)
    p_listen.add_argument("--device", default=None)
    p_listen.add_argument("--output-mode", default="raw", choices=OUTPUT_IMAGE_MODES)
    p_listen.add_argument("--image-format", default="png", choices=image_format_choices)
    p_listen.add_argument("--slant-search", type=float, default=0.03)
    p_listen.add_argument("--auto-levels", action="store_true")
    p_listen.add_argument("--denoise", default="off", choices=denoise_choices)
    p_listen.add_argument("--diagnostics", action="store_true")
    p_listen.add_argument("--poll-seconds", type=float, default=3.0)
    p_listen.add_argument("--max-seconds", type=float, default=None,
                          help="Stop after this many seconds (default: run until Ctrl+C).")
    p_listen.add_argument("--keep-wav", action="store_true",
                          help="Also save the full session's audio as a WAV file.")

    p_tx = sub.add_parser(
        "transmit",
        aliases=["tx"],
        help="Transmit SSTV audio or encode an image and transmit over radio.",
    )
    p_tx.add_argument("input_path", help="WAV/MP3 audio file or PNG/JPG image to encode and transmit.")
    p_tx.add_argument("--mode", default="Martin M1", help="SSTV mode name (if input is an image).")
    p_tx.add_argument("--device", default=None, help="Soundcard output device ID or name.")
    p_tx.add_argument("--sample-rate", "--rate", dest="sample_rate", type=int, default=48000)
    p_tx.add_argument("--rig-model", type=int, default=None, help="Hamlib rig model ID.")
    p_tx.add_argument("--rig-file", default="/dev/ttyUSB0", help="Serial device path for CAT rig control.")
    p_tx.add_argument("--baud", type=int, default=9600, help="CAT serial baud rate.")
    p_tx.add_argument("--rigctld-host", default=None, help="Hamlib rigctld daemon host.")
    p_tx.add_argument("--rigctld-port", type=int, default=4532, help="Hamlib rigctld daemon TCP port.")
    p_tx.add_argument("--gpio-pin", type=int, default=None, help="Raspberry Pi GPIO pin number for PTT keying.")
    p_tx.add_argument("--serial-port", default=None, help="Serial port for RTS/DTR line PTT keying.")
    p_tx.add_argument("--serial-line", default="dtr", choices=["dtr", "rts"], help="Serial control line for PTT.")

    p_modes = sub.add_parser("modes", help="List modes.")
    p_modes.add_argument("--custom-modes-json", action="append", default=None)

    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv:
        return cli_main(argv)

    return script_main()


def script_main():
    """
    Path-free no-argument mode.

    Environment variables:

      NSSTV_ACTION=auto|decode|decode-all|batch|encode|roundtrip|testcard|live|listen
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
        callsign=None,
        callsign_position="bottom-right",
        callsign_opacity=1.0,
        oversample=4,
        slant_mode="auto",
        overlays=None,
):
    """
    image -> SSTV audio -> image, with metrics and a side-by-side PNG.

    Returns a dict with everything: wav path, decoded paths, raw path,
    comparison path, quality, metrics, timings.

    callsign / callsign_position / callsign_opacity stamp a callsign onto
    the source image before encoding, so it appears in both the SSTV
    transmission and the decoded roundtrip image.  Useful for test
    reports that need to identify which transmission is which.

    For more complex overlays (multiple texts, rotation, font styles),
    pass `overlays=[TextOverlay(...), ...]` - those are stamped via
    add_texts_to_image() and take precedence over callsign if both are
    given.

    oversample and slant_mode forward to the encoder/decoder respectively.
    """
    image = _abs(image)
    stem = _stem(image)
    out_dir = _rt_out_dir(image, out_dir)

    # Apply text overlays before encoding so they appear in the SSTV
    # transmission.  We write to a temp file so the original isn't mutated.
    source_image = image
    temp_overlay_path = None
    try:
        if overlays:
            base, ext = os.path.splitext(os.path.abspath(image))
            temp_overlay_path = base + "_nsstv_rt_overlays_tmp" + (ext or ".jpg")
            source_image = add_texts_to_image(image, out_path=temp_overlay_path,
                                              overlays=overlays)
        elif callsign:
            base, ext = os.path.splitext(os.path.abspath(image))
            temp_overlay_path = base + "_nsstv_rt_callsign_tmp" + (ext or ".jpg")
            source_image = add_callsign_to_image(
                image_path=image,
                out_path=temp_overlay_path,
                callsign=callsign,
                position=callsign_position,
                opacity=callsign_opacity,
            )

        wav = os.path.join(out_dir, stem + "_sstv.wav")
        out_base = os.path.join(out_dir, stem + "_decoded." + fmt.lstrip("."))

        t0 = time.time()
        enc = encode(
            source_image,
            out=wav,
            mode=mode,
            fit=fit,
            background=background,
            sample_rate=sample_rate,
            amplitude=amplitude,
            line_structure=line_structure,
            oversample=oversample,
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
            slant_mode=slant_mode,
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
            "callsign": callsign,
            "oversample": oversample,
            "slant_mode": slant_mode,
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

        # Use the overlaid image as the reference for metrics if we stamped
        # an overlay, so the comparison fairly reflects what was transmitted.
        reference = _reference_array(
            source_image,
            layout.width,
            layout.height,
            fit=fit,
            background=background,
        )

        metrics = _metrics(reference, raw)

        comparison_path = None
        if comparison:
            comparison_path = side(
                source_image,
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
    finally:
        if temp_overlay_path and os.path.exists(temp_overlay_path):
            try:
                os.remove(temp_overlay_path)
            except Exception:
                pass


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
        callsign=None,
        callsign_position="bottom-right",
        callsign_opacity=1.0,
        oversample=4,
        slant_mode="auto",
        overlays=None,
):
    """
    Run rt() across several modes and rank them.

    modes=None means every non-experimental mode the encoder supports;
    modes="random" (or modes=3) picks that many modes at random, and seed
    makes the pick repeatable. Experimental modes (Pasokon P3/P5/P7, PD-50)
    are skipped unless include_experimental is True or they are named
    explicitly in modes. Returns a summary dict with "ranked" (best first)
    and "weak" (modes that failed or decoded badly).

    callsign / callsign_position / callsign_opacity stamp a callsign onto
    the source image before each encode, so the bench output images are
    labelled with the callsign.  Useful for test reports that compare
    modes on the same source.

    For more complex overlays, pass `overlays=[TextOverlay(...), ...]`.
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
                callsign=callsign,
                callsign_position=callsign_position,
                callsign_opacity=callsign_opacity,
                oversample=oversample,
                slant_mode=slant_mode,
                overlays=overlays,
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


def info(what=None, show=True):
    """
    Show what nSSTV contains and how to use it.

    Usage:

        import nsstv

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
        lines.append("  NSSTV_ACTION=decode NSSTV_AUDIO_INPUT=input.wav nSSTV")
        lines.append("  NSSTV_ACTION=batch NSSTV_WORKDIR=./recordings nSSTV")
        lines.append("  NSSTV_ACTION=encode NSSTV_IMAGE_INPUT=image.jpg nSSTV")
        lines.append("")
        lines.append("Tip: nSSTV.modes via CLI is `nSSTV modes`; MP3 support needs ffmpeg.")

    elif str(what).strip().lower() in ("cli", "commands", "command-line", "usage", "shell"):
        result = list(_CLI_EXAMPLES)

        lines.append(bar)
        lines.append(f"nSSTV {__version__} - command line")
        lines.append(bar)
        lines.append("")

        for example in _CLI_EXAMPLES:
            lines.append(f"  {example}")

        lines.append("")
        lines.append("Run any command with --help for its flags, e.g. `nSSTV decode --help`.")
        lines.append("With no arguments at all, `nSSTV` runs script mode using NSSTV_* variables.")

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


def modes(custom_modes_json=None):
    """
    List modes the encoder supports.
    """
    return make_registry(custom_modes_json).supported_encoder_modes()


def random_modes(count=DEFAULT_RANDOM_MODES, seed=None):
    """
    Pick mode names at random. Same seed, same picks.
    """
    return _random_modes(ModeRegistry().supported_encoder_modes(), count, seed)


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


def _rt_out_dir(image_path, out_dir=None, sub=None):
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), DEFAULT_OUT_ROOT, _stem(image_path))
    if sub:
        out_dir = os.path.join(out_dir, _safe_filename_part(sub))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.abspath(out_dir)


def _random_modes(available, count, seed=None):
    """Pick `count` modes at random, without repeats."""
    rng = random.Random(seed)

    count = max(1, min(int(count), len(available)))

    return rng.sample(sorted(available), count)

