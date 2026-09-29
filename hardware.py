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

def transmit(
        audio,
        sample_rate=48000,
        device=None,
        rig_hooks=None,
        mode_name="SSTV",
):
    """
    Transmit SSTV audio over a radio transceiver or soundcard.
    Keys PTT (before_transmit hook), plays audio, and guarantees PTT is unkeyed
    in a try...finally block (after_transmit hook).
    """
    created_temp_wav = False
    if isinstance(audio, str):
        wav_path = audio
        fs, pcm = load_audio_mono(wav_path)
    else:
        pcm = np.asarray(audio, dtype=np.float32)
        fs = int(sample_rate)
        tmp_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="_nsstv_tx_")
        os.close(tmp_fd)
        write_wav_file(wav_path, fs, pcm)
        created_temp_wav = True

    duration_s = len(pcm) / fs if fs else 0.0
    tx_ctx = {
        "action": "transmit",
        "wav_path": wav_path,
        "mode_name": mode_name,
        "duration_seconds": duration_s,
    }

    try:
        _call_hook(rig_hooks, "before_transmit", tx_ctx)
        try:
            import sounddevice as sd
            sd.play(pcm, fs, device=device)
            sd.wait()
        except Exception as sd_err:
            if sys.platform.startswith("linux"):
                cmd = ["aplay"]
                if device:
                    cmd.extend(["-D", str(device)])
                cmd.append(wav_path)
                subprocess.run(cmd, check=True)
            elif sys.platform == "darwin":
                subprocess.run(["afplay", wav_path], check=True)
            else:
                raise RuntimeError(
                    f"Audio playback failed: {sd_err}\n"
                    "If running on Linux or Raspberry Pi, ensure libportaudio2 is installed:\n"
                    "  sudo apt install libportaudio2"
                ) from sd_err
    finally:
        try:
            _call_hook(rig_hooks, "after_transmit", tx_ctx)
        finally:
            if created_temp_wav and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    return tx_ctx


def make_rigctl_hooks(
        rig_model=None,
        rig_file="/dev/ttyUSB0",
        baud=9600,
        rigctld_host=None,
        rigctld_port=4532,
        freq_hz=None,
        mode="USB",
        ptt_delay=0.2,
        max_ptt_seconds=180.0,
):
    """
    Create a pre-configured RigHooks instance using Hamlib `rigctl` CLI or `rigctld` TCP socket daemon.
    Includes emergency PTT-off recovery and socket connection caching.
    """
    state = {"sock": None, "tx_start": None}

    def _build_rigctl_base_cmd():
        base_cmd = ["rigctl"]
        if rig_model is not None:
            base_cmd.extend(["-m", str(rig_model)])
        if rig_file:
            base_cmd.extend(["-r", str(rig_file)])
        if baud:
            base_cmd.extend(["-s", str(baud)])
        return base_cmd

    def _send_rigctld_cmd(cmd_str):
        import socket
        try:
            if state["sock"] is None:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3.0)
                s.connect((rigctld_host, int(rigctld_port)))
                state["sock"] = s
            s = state["sock"]
            s.sendall(f"{cmd_str}\n".encode("utf-8"))
            resp = s.recv(1024).decode("utf-8").strip()
            if "RPRT" in resp:
                lines = [l.strip() for l in resp.splitlines() if "RPRT" in l]
                for rprt_line in lines:
                    parts = rprt_line.split()
                    if len(parts) >= 2 and parts[0] == "RPRT":
                        if parts[1] != "0":
                            raise RuntimeError(f"rigctld error response for '{cmd_str}': {resp}")
            return resp
        except Exception as e:
            if state["sock"]:
                try:
                    state["sock"].close()
                except Exception:
                    pass
                state["sock"] = None
            raise e

    def _run_rigctl(cmd_args):
        if rigctld_host:
            cmd_str = " ".join(cmd_args)
            return _send_rigctld_cmd(cmd_str)
        else:
            base_cmd = _build_rigctl_base_cmd()
            full_cmd = base_cmd + cmd_args
            res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=5)
            if res.returncode != 0:
                raise RuntimeError(
                    f"rigctl command failed (code {res.returncode}): {res.stderr.strip() or res.stdout.strip()}"
                )
            return res.stdout.strip()

    def before_rec(ctx):
        updates = {}
        if freq_hz or mode:
            cmd = []
            if freq_hz:
                cmd.extend(["F", str(int(freq_hz))])
            if mode:
                cmd.extend(["M", str(mode), "0"])
            try:
                _run_rigctl(cmd)
            except Exception as e:
                warnings.warn(f"Rigctl frequency set error: {e}")
        try:
            freq_str = _run_rigctl(["f"])
            if freq_str and freq_str.isdigit():
                updates["rig_freq_hz"] = int(freq_str)
        except Exception:
            pass
        return updates

    def before_tx(ctx):
        dur = (ctx or {}).get("duration_seconds", 0.0)
        if max_ptt_seconds and dur > max_ptt_seconds:
            raise ValueError(
                f"Transmit duration ({dur:.1f}s) exceeds max_ptt_seconds limit ({max_ptt_seconds:.1f}s)"
            )
        state["tx_start"] = time.time()
        _run_rigctl(["T", "1"])
        if ptt_delay > 0:
            time.sleep(ptt_delay)

    def after_tx(ctx):
        if max_ptt_seconds and state.get("tx_start"):
            elapsed = time.time() - state["tx_start"]
            if elapsed > max_ptt_seconds:
                warnings.warn(
                    f"PTT watchdog warning: total transmit time ({elapsed:.1f}s) exceeded max_ptt_seconds ({max_ptt_seconds:.1f}s)"
                )
        unkeyed = False
        for attempt in range(3):
            try:
                _run_rigctl(["T", "0"])
                unkeyed = True
                break
            except Exception as e:
                warnings.warn(f"Rigctl PTT unkey attempt {attempt+1} failed: {e}")
                time.sleep(0.1)

        if not unkeyed:
            try:
                base_cmd = _build_rigctl_base_cmd()
                subprocess.run(base_cmd + ["T", "0"], capture_output=True, timeout=3)
            except Exception as e:
                warnings.warn(f"Emergency rigctl unkey also failed: {e}")

            if state["sock"]:
                try:
                    state["sock"].close()
                except Exception:
                    pass
                state["sock"] = None

            raise RuntimeError("CRITICAL: Failed to unkey PTT after all attempts! Radio may remain keyed!")

        if state["sock"]:
            try:
                state["sock"].close()
            except Exception:
                pass
            state["sock"] = None

    return RigHooks(
        before_record=before_rec,
        before_transmit=before_tx,
        after_transmit=after_tx,
    )


def make_rpi_gpio_hooks(ptt_pin=17, active_high=True, ptt_delay=0.2):
    """
    Create a RigHooks instance for Raspberry Pi GPIO PTT transmitter control.
    Supports Pi 5 (RP1 chip) and Pi 4/3/Zero 2 W, persisting line handles during playback.
    """
    state = {"gpiod_line": None, "rpi_gpio": False}

    def before_tx(ctx):
        high = True if active_high else False
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(ptt_pin, GPIO.OUT)
            GPIO.output(ptt_pin, GPIO.HIGH if high else GPIO.LOW)
            state["rpi_gpio"] = True
        except (ImportError, RuntimeError):
            try:
                import gpiod
                chip_name = None
                for candidate in ["gpiochip4", "gpiochip0"]:
                    if os.path.exists(f"/dev/{candidate}"):
                        chip_name = candidate
                        break
                chip_name = chip_name or "gpiochip0"
                chip = gpiod.Chip(chip_name)
                line = chip.get_line(ptt_pin)
                line.request(consumer="nSSTV", type=gpiod.LINE_REQ_DIR_OUT)
                line.set_value(1 if high else 0)
                state["gpiod_line"] = line
            except Exception as e:
                warnings.warn(f"Raspberry Pi GPIO PTT warning: RPi.GPIO or gpiod not available ({e})")

        if ptt_delay > 0:
            time.sleep(ptt_delay)

    def after_tx(ctx):
        low = False if active_high else True
        try:
            if state["rpi_gpio"]:
                import RPi.GPIO as GPIO
                GPIO.output(ptt_pin, GPIO.HIGH if low else GPIO.LOW)
                GPIO.cleanup(ptt_pin)
        except Exception:
            pass

        try:
            line = state["gpiod_line"]
            if line is not None:
                line.set_value(1 if low else 0)
                line.release()
        except Exception:
            pass
        finally:
            state["gpiod_line"] = None
            state["rpi_gpio"] = False

    return RigHooks(
        before_transmit=before_tx,
        after_transmit=after_tx,
    )


def make_serial_dtr_rts_hooks(port="/dev/ttyUSB0", line="dtr", active_high=True, ptt_delay=0.2):
    """
    Create a RigHooks instance for serial port DTR or RTS line PTT keying (SignaLink / interfaces).
    Persists open serial connection throughout transmission so lines are held HIGH during playback.
    """
    state = {"ser": None}

    def before_tx(ctx):
        try:
            import serial
            if state["ser"] is None or not state["ser"].is_open:
                state["ser"] = serial.Serial(port)
            ser = state["ser"]
            val = True if active_high else False
            if line.lower() == "rts":
                ser.rts = val
            else:
                ser.dtr = val
        except ImportError:
            warnings.warn("Serial PTT keying requires pyserial (pip install pyserial).")
        except Exception as e:
            warnings.warn(f"Serial PTT keying error on {port}: {e}")

        if ptt_delay > 0:
            time.sleep(ptt_delay)

    def after_tx(ctx):
        try:
            ser = state["ser"]
            if ser and ser.is_open:
                val = False if active_high else True
                if line.lower() == "rts":
                    ser.rts = val
                else:
                    ser.dtr = val
                ser.close()
        except Exception as e:
            warnings.warn(f"Error releasing serial PTT on {port}: {e}")
        finally:
            state["ser"] = None

    return RigHooks(
        before_transmit=before_tx,
        after_transmit=after_tx,
    )


def make_rpi_systemd_service(
    service_name="nsstv-listener",
    out_dir="/home/pi/sstv_images",
    device=None,
    sample_rate=48000,
    keep_wav=False,
    python_path=None,
    user="pi",
):
    """
    Generate a Linux systemd service unit file for automated Raspberry Pi background SSTV listening.
    """
    python_path = python_path or sys.executable
    cmd = f'{python_path} -m nSSTV listen --out-dir "{out_dir}" --sample-rate {sample_rate}'
    if device:
        cmd += f" --device {device}"
    if keep_wav:
        cmd += " --keep-wav"

    return f"""[Unit]
Description=nSSTV Radio encoding and decoding ({service_name})
After=network.target sound.target

[Service]
Type=simple
User={user}
ExecStart={cmd}
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""


def live_listen_and_decode(
        out_dir="live_decoded",
        sample_rate=48000,
        device=None,
        output_mode="raw",
        image_format="png",
        slant_search=0.03,
        auto_levels=False,
        denoise="off",
        write_sidecar=True,
        diagnostics=False,
        poll_seconds=3.0,
        max_seconds=None,
        max_buffer_seconds=1800.0,
        keep_wav=False,
        on_image=None,
        on_preview=None,
        stop_event=None,
        registry=None,
        protocol=None,
        custom_modes_json=None,
        verbose=True,
        rig_hooks=None,
):
    """
    Listen continuously and decode SSTV transmissions AS THEY ARRIVE,
    instead of live_record_then_decode()'s "record for N seconds, then
    decode" - which can only ever produce one image per call and makes
    you commit to a total listening time up front. This keeps listening
    (and can decode any number of transmissions back to back) until you
    stop it - Ctrl+C, `max_seconds` elapses, or `stop_event` is set.

    Requires: pip install sounddevice

    How it works: a background audio callback appends microphone/virtual
    -cable audio to an in-memory buffer; every `poll_seconds`, the buffer
    captured so far is rescanned (via _live_scan_buffer) for new leader
    tones. A transmission already fully inside the buffer is FINALIZED -
    image written, on_image(item) called, counted done. One that's still
    arriving is reported as a PREVIEW instead - same file, filled in a
    little more each poll - via on_preview(item), and is retried (not
    skipped) on the next poll once more audio has come in.

    on_image(item)      called once, when a transmission finishes.
    on_preview(item)    optional; called every poll while a transmission
                        is still arriving. Leave as None to skip previews
                        entirely (finalization still happens normally).
    stop_event           a threading.Event; set() it from another thread
                        to end the session early. Ctrl+C also works.
    max_buffer_seconds   caps memory: once this much audio has been
                        finalized, older finalized audio is dropped from
                        the in-memory buffer. Doesn't limit session
                        length, only how much old audio is kept around.
    keep_wav             if truthy, the full session's audio is also
                        written to this path (or an auto-named .wav next
                        to out_dir if keep_wav=True) as it's captured -
                        handy for replaying a session later with
                        decode_all_audio_to_images().

    rig_hooks : RigHooks or None
        before_record/after_record fire once, around the whole listening
        session (not per-transmission); before_decode/after_decode fire
        once per finalized image, ctx={"action": "decode", "result": item}.

    Returns a summary dict: {"images": [...], "out_dir", "wav_path"}.
    """
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError(
            "Live capture requires sounddevice.\n"
            "Install with:\n"
            "  pip install sounddevice"
        ) from e

    registry = registry or make_registry(custom_modes_json)
    protocol = protocol or ProtocolConstants()

    sample_rate = int(sample_rate)
    poll_seconds = float(poll_seconds)

    os.makedirs(out_dir, exist_ok=True)
    audio_base = _safe_filename_part(
        os.path.splitext(os.path.basename(out_dir.rstrip("/\\")))[0] or "live"
    )

    wav_path = None
    if keep_wav:
        wav_path = keep_wav if isinstance(keep_wav, str) else os.path.join(out_dir, f"{audio_base}_session.wav")

    lock = threading.Lock()
    state = {"chunks": [], "stop": False}

    def _callback(indata, frames, time_info, status):
        if status:
            if status.input_overflow and verbose:
                warnings.warn(f"Sounddevice buffer overflow detected ({status}) - CPU load may have dropped audio samples.")
            elif status.input_underflow and verbose:
                warnings.warn(f"Sounddevice buffer underflow detected ({status}).")
        if indata.ndim > 1 and indata.shape[1] > 1:
            samples = np.mean(indata, axis=1, dtype=np.float32)
        else:
            samples = np.asarray(indata[:, 0] if indata.ndim > 1 else indata, dtype=np.float32)
        with lock:
            state["chunks"].append(samples.copy())

    rec_ctx = {"action": "record", "seconds": max_seconds, "sample_rate": sample_rate}
    _call_hook(rig_hooks, "before_record", rec_ctx)

    if verbose:
        print(f"Listening at {sample_rate} Hz - Ctrl+C to stop"
              + (f" (max {max_seconds:.0f}s)" if max_seconds else "") + " ...")

    finished_all = []
    processed_until = 0.0
    pending_index_by_key = {}
    next_index = 0
    trimmed_seconds = 0.0
    written_wav_samples = 0
    start_time = time.time()

    executor = ThreadPoolExecutor(max_workers=1)
    scan_future = None

    stream = sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32",
        device=device, callback=_callback,
    )

    try:
        with stream:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                    break

                time.sleep(poll_seconds)

                with lock:
                    if not state["chunks"]:
                        continue
                    buf = np.concatenate(state["chunks"])

                if wav_path:
                    new_samples = buf[written_wav_samples:]
                    if len(new_samples) > 0:
                        append_wav_file(wav_path, sample_rate, new_samples)
                        written_wav_samples = len(buf)

                # Process result of background scan if ready
                if scan_future is not None and scan_future.done():
                    try:
                        result = scan_future.result()
                        processed_until = result["processed_until"]
                        pending_index_by_key = result["pending_index_by_key"]
                        next_index = result["next_index"]

                        for item in result["finished"]:
                            finished_all.append(item)
                            if verbose:
                                q = item.get("quality_score")
                                print(f"  [{item['index']:03d}] {item['mode_name']}: DONE"
                                      + (f" quality {q:.2f}" if q is not None else ""))
                            dx_ctx = {"action": "decode", "result": item}
                            _call_hook(rig_hooks, "before_decode", dx_ctx)
                            if on_image:
                                on_image(item)
                            _call_hook(rig_hooks, "after_decode", dx_ctx)

                        for item in result["previews"]:
                            if verbose:
                                ld, le = item.get("lines_decoded"), item.get("lines_expected")
                                print(f"  [{item['index']:03d}] {item['mode_name']}: "
                                      f"receiving... {ld}/{le} lines")
                            if on_preview:
                                on_preview(item)
                    except Exception as exc:
                        if verbose:
                            warnings.warn(f"Background buffer scan error: {exc}")
                    scan_future = None

                # Submit new background scan job if worker is free
                if scan_future is None:
                    scan_future = executor.submit(
                        _live_scan_buffer,
                        buf, sample_rate,
                        processed_until=processed_until,
                        pending_index_by_key=pending_index_by_key,
                        next_index=next_index,
                        out_dir=out_dir,
                        audio_base=audio_base,
                        registry=registry,
                        protocol=protocol,
                        output_mode=output_mode,
                        image_format=image_format,
                        slant_search=slant_search,
                        auto_levels=auto_levels,
                        denoise=denoise,
                        write_sidecar=write_sidecar,
                        diagnostics=diagnostics,
                    )

                # Cap memory: once audio before `processed_until` is no
                # longer needed (nothing pending still references it),
                # drop it from the buffer and remember the offset so
                # future "seconds" bookkeeping (leader_start etc. in
                # returned items) stays relative to the full session.
                buf_seconds = len(buf) / sample_rate
                if buf_seconds > max_buffer_seconds and not pending_index_by_key:
                    keep_from = max(0.0, processed_until - 2.0)
                    drop_samples = int(keep_from * sample_rate)
                    if drop_samples > 0:
                        with lock:
                            state["chunks"] = [buf[drop_samples:]]
                        processed_until -= keep_from
                        trimmed_seconds += keep_from
                        written_wav_samples = max(0, written_wav_samples - drop_samples)
    except KeyboardInterrupt:
        if verbose:
            print("Stopped.")
    finally:
        executor.shutdown(wait=False)
        rec_ctx["wav_path"] = wav_path
        _call_hook(rig_hooks, "after_record", rec_ctx)

    return {
        "images": finished_all,
        "count": len(finished_all),
        "out_dir": out_dir,
        "wav_path": wav_path,
        "modes": [i["mode_name"] for i in finished_all],
        "quality": [i.get("quality_score", 0.0) for i in finished_all],
    }


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
        rig_hooks=None,
):
    """
    Optional live feature: record microphone/virtual audio, then decode.

    This blocks for the full `seconds` before decoding anything, and only
    ever produces one image per call - you have to know roughly how long
    to record ahead of time. For an open-ended session that decodes each
    transmission as soon as it's finished (no fixed duration, multiple
    images per session), see live_listen_and_decode() instead.

    Requires:
        pip install sounddevice

    rig_hooks : RigHooks or None
        Optional rig-control callbacks fired around the record and decode
        steps.  Call order:

        1. ``before_record``  — tune the rig, set USB/LSB, etc.
        2. (audio capture)
        3. ``after_record``   — rig read-back, log the actual frequency, etc.
        4. ``before_decode``  — called just before the decoder runs.
        5. (decode)
        6. ``after_decode``   — inspect or log the decode result.

        See RigHooks for the full callback contract.
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

    rec_ctx = {
        "action": "record",
        "seconds": seconds,
        "sample_rate": sample_rate,
    }
    _call_hook(rig_hooks, "before_record", rec_ctx)

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

    rec_ctx["wav_path"] = wav_path
    _call_hook(rig_hooks, "after_record", rec_ctx)

    rx_ctx = {"action": "decode", "audio_path": wav_path}
    _call_hook(rig_hooks, "before_decode", rx_ctx)

    result = decode_audio_to_images_v2(
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

    rx_ctx["result"] = result
    _call_hook(rig_hooks, "after_decode", rx_ctx)

    if rx_ctx.get("rig"):
        result["rig"] = rx_ctx["rig"]

    return result


def sdr_listen_and_decode(
        frequency_hz=145800000,
        sample_rate=2000000,
        audio_sample_rate=48000,
        gain="auto",
        out_dir="sdr_decoded",
        output_mode="raw",
        image_format="png",
        max_seconds=None,
        stop_event=None,
        verbose=True,
        rig_hooks=None,
):
    """
    Capture and decode SSTV signals directly from an RTL-SDR USB dongle (requires optional pyrtlsdr).
    No physical radio rig or external soundcard required!
    """
    try:
        from rtlsdr import RtlSdr
    except ImportError as e:
        raise RuntimeError(
            "RTL-SDR receiver support requires pyrtlsdr.\n"
            "Install with:\n"
            "  pip install pyrtlsdr"
        ) from e

    sdr = RtlSdr()
    sdr.sample_rate = int(sample_rate)
    sdr.center_freq = int(frequency_hz)
    sdr.gain = gain

    if verbose:
        print(f"RTL-SDR listening on {frequency_hz / 1e6:.3f} MHz (Ctrl+C to stop) ...")

    try:
        raw_iq = sdr.read_samples(int(sample_rate * 2.0))
        angles = np.angle(raw_iq[1:] * np.conj(raw_iq[:-1]))
        decim_factor = max(1, int(sample_rate // audio_sample_rate))
        audio_pcm = angles[::decim_factor].astype(np.float32)
        audio_pcm /= np.pi

        res = _live_scan_buffer(
            audio_pcm, audio_sample_rate,
            processed_until=0.0,
            pending_index_by_key={},
            next_index=0,
            out_dir=out_dir,
            audio_base=f"sdr_{int(frequency_hz/1e3)}kHz",
        )
        return res
    finally:
        try:
            sdr.close()
        except Exception as close_err:
            warnings.warn(f"Warning closing RTL-SDR device: {close_err}")


def _live_scan_buffer(
        buf_samples,
        fs,
        processed_until,
        pending_index_by_key,
        next_index,
        out_dir,
        audio_base,
        registry=None,
        protocol=None,
        output_mode="raw",
        image_format="png",
        slant_search=0.03,
        auto_levels=False,
        denoise="off",
        write_sidecar=True,
        diagnostics=False,
        min_gap_seconds=1.0,
        line_structure="auto",
):
    """
    One incremental scan/decode step over audio captured so far.

    This is the hardware-independent core behind live_listen_and_decode():
    it takes a snapshot of the audio buffer (as plain samples, not a live
    stream) and returns what changed, so it can be driven either by a real
    microphone poll loop or, for testing, by literally growing a numpy
    array a bit at a time with no sound device involved at all.

    buf_samples / fs   audio captured so far, from time 0.
    processed_until     seconds already finalized in a previous call;
                        candidates entirely before this are ignored.
    pending_index_by_key  {leader_start_rounded: image_index} for
                        transmissions seen but not yet finished, so a
                        transmission's preview and final image share one
                        index/filename across repeated calls instead of
                        each poll grabbing a new one.
    next_index          the next fresh image index to hand out.

    A transmission is only FINALIZED (counted as processed, sidecar
    written, processed_until advanced past it) once decode_image_extended
    reports partial=False, i.e. enough audio had arrived to decode every
    line. While partial=True it's returned as a PREVIEW instead: same
    filename/index, image gets overwritten with more lines each call, and
    processed_until does NOT move past it, so the next call retries it
    with whatever new audio has since arrived.

    Returns a dict:
      finished              list of finalized item dicts (final=True)
      previews               list of in-progress item dicts (final=False)
      processed_until        updated cursor to pass in next time
      pending_index_by_key   updated map to pass in next time
      next_index             updated counter to pass in next time
    """
    registry = registry or make_registry()
    protocol = protocol or ProtocolConstants()

    total_seconds = len(buf_samples) / fs if fs else 0.0

    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="_nsstv_live_")
    os.close(tmp_fd)
    try:
        write_wav_file(tmp_wav, fs, buf_samples)
        decoder = SSTVDecoder(tmp_wav, registry=registry, protocol=protocol)
        candidates = decoder._coarse_leader_candidates()

        finished = []
        previews = []

        # Candidates come out in chronological order. Audio arrives
        # chronologically too, so once we hit one that isn't decodable
        # yet, every later candidate is at least as incomplete - stop
        # the scan for this call rather than churning through them.
        for cand in candidates:
            if cand["region_start"] < processed_until - 0.25:
                continue

            header = _detect_header_from_candidate(decoder, cand)

            if header is None:
                # Could be a still-arriving leader (not enough audio yet
                # for the VIS header to confirm), or plain noise. Only
                # give up on it once it's comfortably behind the live
                # edge - a real leader's VIS header finishes within
                # ~1.2s of its start, so 3s of margin is generous.
                if cand["region_start"] > total_seconds - 3.0:
                    break
                continue

            fine = header["fine"]
            vis_info = header["vis_info"]
            check = header["leader_check"]

            decoder.freq_offset_hz = cand["offset"]
            decoder.leader_start = fine["leader_start"]
            decoder.leader_end = fine["leader_end"]
            decoder.vis_start = vis_info["vis_start"]
            ext_code = registry.extended_vis_code_for_mode(vis_info["mode_name"])
            header_blocks = 2 if ext_code is not None else 1
            decoder.vis_end = decoder.vis_start + header_blocks * protocol.vis_header_seconds
            decoder.mode = registry.get_mode(vis_info["mode_name"])
            decoder.vis_value = vis_info["vis_value"]
            decoder.leader_check = check

            key = round(float(fine["leader_start"]), 2)
            if key in pending_index_by_key:
                index = pending_index_by_key[key]
            else:
                index = next_index
                next_index += 1
                pending_index_by_key[key] = index

            mode_safe = _safe_filename_part(decoder.mode.name)
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
                extra_metadata={"transmission_index": index, "live": True},
            )

            layout = registry.get_layout(decoder.mode.name)
            measured_line = image.get("measured_line_seconds", layout.line_seconds)
            tx_end = decoder.vis_end + layout.leadin_seconds + layout.n_lines * measured_line + 0.5
            if image.get("spectrogram_audio_end") is not None:
                tx_end = max(tx_end, float(image["spectrogram_audio_end"]))

            is_final = isinstance(image, dict) and "error" not in image and not image.get("partial", True)

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
                "final": is_final,
            }

            if isinstance(image, dict) and "error" not in image:
                item["raw"] = (image.get("paths") or {}).get("raw")
                item["partial"] = bool(image.get("partial", False))
                item["lines_decoded"] = image.get("lines_decoded")
                item["lines_expected"] = image.get("lines_expected")
                item["quality_score"] = float((image.get("quality") or {}).get("overall", 0.0))
                item["starts_at"] = round(float(decoder.leader_start), 3)

            if diagnostics and is_final:
                diag_path = os.path.join(out_dir, f"{audio_base}_{index:03d}_{mode_safe}_diagnostics.jpg")
                markers = [
                    {"time": decoder.leader_start, "label": "leader", "color": (0, 255, 255)},
                    {"time": decoder.vis_start, "label": "VIS", "color": (255, 255, 0)},
                    {"time": decoder.vis_end, "label": "image", "color": (0, 255, 0)},
                    {"time": tx_end, "label": "end", "color": (255, 0, 255)},
                ]
                item["diagnostics_path"] = make_diagnostics_image(
                    decoder.samples, decoder.fs, diag_path,
                    t_start=max(0.0, decoder.leader_start - 0.5),
                    t_end=min(decoder.total_duration, tx_end + 0.5),
                    markers=markers,
                    title=f"nSSTV diagnostics | transmission {index} | {decoder.mode.name}",
                )

            if is_final:
                pending_index_by_key.pop(key, None)
                processed_until = max(processed_until, tx_end + min_gap_seconds)
                finished.append(item)
            else:
                previews.append(item)
                # This transmission isn't done, so nothing after it in
                # the recording is either - stop the scan here.
                break
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass

    return {
        "finished": finished,
        "previews": previews,
        "processed_until": processed_until,
        "pending_index_by_key": pending_index_by_key,
        "next_index": next_index,
    }


class RigHooks:
    """
    Optional rig-control callbacks for encode(), decode(),
    live_record_then_decode(), and live_listen_and_decode().

    Every field is a callable or None.  nSSTV calls each one at the
    appropriate moment and passes a context dict so the callback can
    read or update state (e.g. log the actual rig frequency into the
    sidecar).  A callback that returns a dict is merged into the
    context; any other return value is ignored.  Exceptions raised by
    a callback propagate normally so the caller can decide whether to
    abort or continue.

    Typical Hamlib wiring
    ---------------------
    ::

        import Hamlib

        rig = Hamlib.Rig(Hamlib.RIG_MODEL_FT817ND)
        rig.set_conf("rig_pathname", "/dev/ttyUSB0")
        rig.set_conf("serial_speed", "9600")
        rig.open()

        def set_rx(ctx):
            rig.set_freq(Hamlib.RIG_VFO_A, 14230000)
            rig.set_mode(Hamlib.RIG_MODE_USB, 0)
            ctx["rig_freq_hz"] = rig.get_freq()

        def ptt_on(ctx):
            rig.set_ptt(Hamlib.RIG_VFO_A, Hamlib.RIG_PTT_ON)

        def ptt_off(ctx):
            rig.set_ptt(Hamlib.RIG_VFO_A, Hamlib.RIG_PTT_OFF)

        hooks = nsstv.RigHooks(
            before_record=set_rx,
            before_transmit=ptt_on,
            after_transmit=ptt_off,
        )

        # RX
        nsstv.decode("recording.wav", rig_hooks=hooks)

        # TX
        nsstv.encode("image.jpg", rig_hooks=hooks)

    Fields
    ------
    before_record
        Called before audio capture starts (live_record_then_decode).
        Use it to tune the rig to the receive frequency and set USB/LSB.
        Receives: ``{"action": "record", "seconds": float, "sample_rate": int}``

    after_record
        Called after audio capture finishes, before decoding begins.
        Receives: ``{"action": "record", "wav_path": str, "seconds": float}``

    before_decode
        Called before an audio file (live or pre-recorded) is decoded.
        Receives: ``{"action": "decode", "audio_path": str}``

    after_decode
        Called after decoding completes.
        Receives: ``{"action": "decode", "audio_path": str, "result": dict}``
        The callback may add keys to the context; they are stored under
        ``result["rig"]`` in the returned decode result dict.

    before_transmit
        Called immediately before the encoded audio is played / written,
        intended for PTT key-up.
        Receives: ``{"action": "transmit", "wav_path": str, "mode_name": str,
        "duration_seconds": float}``

    after_transmit
        Called after transmit completes, intended for PTT key-down.
        Receives: ``{"action": "transmit", "wav_path": str, "mode_name": str,
        "duration_seconds": float}``
    """

    before_record:    object = None  # callable(ctx) or None
    after_record:     object = None
    before_decode:    object = None
    after_decode:     object = None
    before_transmit:  object = None
    after_transmit:   object = None

