<p align="center">
  <img src="https://github.com/user-attachments/assets/8eed1fbe-a135-4d00-b7f0-d51395689181" alt="nSSTV Logo" width="650" />
</p>

<h1 align="center">nSSTV v1.1</h1>

<p align="center">
  <strong>Complete SSTV (Slow-Scan Television) encoder, decoder, live radio receiver and rig controller for Python.</strong><br/>
  Turn images into SSTV radio audio, and SSTV audio back into images.
</p>

<p align="center">
  <a href="#what-is-sstv">What is SSTV?</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#python-api-handbook">Python API</a> •
  <a href="#command-line-interface">CLI</a> •
  <a href="#rig-control">Rig Control</a> •
  <a href="#supported-modes">Modes</a> •
  <a href="#notes--operating-tips">Tips</a>
</p>

---

## What is SSTV?

**Slow-Scan Television** is a mode used by amateur radio operators to transmit still images over HF, VHF, and UHF radio bands, or via satellite links such as the International Space Station. Each pixel's brightness becomes an audio tone and the picture is painted one line at a time:

| Tone | Meaning |
| :--- | :--- |
| 1200 Hz | Sync |
| 1500 Hz | Black |
| 2300 Hz | White |

---

## Features

### Encoding and decoding

- **Full-spectrum encoding**: convert any image (JPEG, PNG, BMP, WEBP) into SSTV audio (WAV or MP3) with custom sample rates, padding, and aspect-ratio handling.
- **Universal decoding**: recover weak, noisy, or off-frequency SSTV audio with automatic mode detection and VIS header verification.
- **Decode-all**: scan a long recording and extract every transmission into separate images.
- **Batch processing**: recursively decode whole directories of WAV or MP3 files in one command.
- **Broad mode support**: Martin, Scottie, Robot, PD, Pasokon, MMSSTV (including 16-bit extended VIS and N-VIS), and Wraase families.
- **Custom modes**: define your own resolution, timing, and RGB/YUV scan pattern from Python or JSON by extending `SSTVMode` and `ModeRegistry`.

### Radio and hardware

- **Managed transmission (`transmit`)**: automated PTT keying before playback, with guaranteed release in a `finally` block so your transmitter never gets stuck on.
- **Rig control**: Hamlib `rigctl`/`rigctld` CAT control, Raspberry Pi GPIO (Pi 5 on `/dev/gpiochip4`, Pi 4/3 on `/dev/gpiochip0`), and serial RTS/DTR interface lines. Fully opt-in through the `RigHooks` dataclass.
- **Live soundcard receiver**: record from a mic or audio cable, or listen continuously and decode automatically.
- **Direct RTL-SDR receiver**: decode broadcasts straight from an RTL-SDR USB dongle (for example 145.800 MHz FM for ISS passes, or 14.230 MHz HF) with no external radio hardware.
- **Background station daemons**: generate a systemd unit (`nsstv-rx.service`) for 24/7 monitoring on Raspberry Pi OS.
- **WAV auto-rollover**: long-running sessions roll over into `_part2.wav` files before the 32-bit RIFF size limit is hit.

### Image and signal quality

- **Smart image fit**: `contain` (pad), `cover` (crop), or `stretch`, so photos are not squashed.
- **Multiple output styles**: raw, Polaroid frame, annotated spectrogram, stacked composite, or all at once.
- **Text overlays and watermarks**: template callsign, grid square, timestamp, frequency, and mode text with custom TTF/OTF fonts, bold/italic styles, opacity, arbitrary rotation, and sub-pixel anti-aliased rendering.
- **Auto-levels** (`auto_levels=True`): stretch dynamic range to restore dark or faded images.
- **Denoise presets**: `off`, `light`, `medium`, `strong`.
- **Adaptive whistle notch filters**: detect and remove interfering carrier or CW tones before FM demodulation.
- **Bandpass pre-filtering**: a 1100 to 2300 Hz Butterworth filter isolates the SSTV band from sub-audible hum and high-frequency noise.
- **Auto-slant calibration** (`slant_search=0.03`): corrects clock drift between transmit and receive soundcards to remove image tilt.
- **Diagnostics**: annotated spectrograms showing leader tones, VIS bit breakdown, sync alignment, and signal-to-noise metrics.

### Analysis tools

- **Quality metrics**: MAE, PSNR (dB), and correlation on every roundtrip.
- **Benchmarking (`bench`)**: encode and decode your image across every mode and rank the results.
- **Test cards**: generate calibrated test images for any mode.

<p align="center">
  <img width="1226" height="636" alt="nSSTV image output comparison showcase" src="https://github.com/user-attachments/assets/4273c915-325b-4b72-ae69-8ebeb3df75ac" />
</p>

---

## Installation

```bash
pip install nSSTV            # core encoder/decoder
pip install "nSSTV[mp3]"     # + MP3 support (needs ffmpeg)
pip install "nSSTV[live]"    # + live capture
pip install "nSSTV[audio]"   # + live soundcard playback and mic recording (sounddevice)
pip install "nSSTV[serial]"  # + serial DTR/RTS PTT keying (pyserial)
pip install "nSSTV[sdr]"     # + direct RTL-SDR USB receiver (pyrtlsdr)
pip install "nSSTV[all]"     # everything
```

MP3 files and some audio features need `ffmpeg` and PortAudio:

```bash
# macOS (Homebrew)
brew install ffmpeg

# Linux (Debian / Ubuntu / Raspberry Pi OS)
sudo apt update && sudo apt install -y ffmpeg libportaudio2
```

> [!NOTE]
> MP3 compression damages SSTV tones, so MP3 decodes come out noisier. It is fine for casual sharing, but WAV is strongly recommended for the best picture.

---

## Quick Start

```python
import nsstv

nSSTV.encode("photo.jpg")           # → photo_sstv.wav
nSSTV.decode("recording.wav")       # → recording_decoded.png
nSSTV.decode_all("recording.wav")   # → every image in one recording
nSSTV.rt("photo.jpg")               # roundtrip + metrics + comparison
nSSTV.bench("photo.jpg")            # try every mode, rank them
```

Or from the terminal:

```bash
nSSTV rt photo.jpg          # test the whole pipeline
nSSTV bench photo.jpg       # find the best mode for your image
```

---

## Python API Handbook

Every function works out of the box with sensible defaults. Only input paths are required; all tuning parameters are optional.

### Function overview

```python
nSSTV.encode("photo.jpg")               # → photo_sstv.wav
nSSTV.decode("recording.wav")           # → recording_decoded.png
nSSTV.decode_all("recording.wav")       # every image in a recording
nSSTV.batch_decode("recordings/")       # whole folder
nSSTV.rt("photo.jpg")                   # encode + decode + metrics + comparison
nSSTV.bench("photo.jpg")                # benchmark every mode
nSSTV.compare("a.png", "b.png")         # MAE / PSNR / correlation
nSSTV.side("a.png", "b.png", "out.png") # side-by-side PNG
nSSTV.card("PD-180", "testcard.png")    # test card for a mode
nSSTV.fit("photo.jpg", mode="PD-180")   # preview image fit
nSSTV.modes()                           # list all modes
nSSTV.info()                            # full API reference
nSSTV.info("decode")                    # docs for one function
```

### 1. Encoding images

```python
import nsstv

info = nSSTV.encode(
    image_path="photo.jpg",         # REQUIRED
    wav_out=None,                   # default creates "photo_sstv.wav"
    mode="Martin M1",               # SSTV mode name (default "PD-180")
    sample_rate=48000,              # audio sample rate in Hz
    amplitude=0.80,                 # volume, 0.0 to 1.0
    callsign=None,                  # station callsign, e.g. "W1AW"
    callsign_position="watermark",  # "watermark", "stamp", "header", "footer", "ur", "ll"
    callsign_opacity=1.0,           # overlay transparency, 0.0 to 1.0
    image_fit="contain",            # "contain" (pad), "cover" (crop), "stretch"
    oversample=4,                   # sub-pixel oversampling factor
)

print("Generated audio duration:", info["duration_seconds"], "seconds")
```

### 2. Managed radio transmission with PTT

`nSSTV.transmit()` keys your transceiver before audio playback begins and guarantees PTT release in a `finally` block.

```python
import nsstv

# Option A: Hamlib CAT control (rigctl / rigctld)
rig_hooks = nSSTV.make_rigctl_hooks(
    rig_model=2024,             # Hamlib rig model ID
    rig_file="/dev/ttyUSB0",    # serial port path
    baud=9600,
    freq_hz=14230000,           # 14.230 MHz
    mode="USB",
)
nSSTV.transmit("photo_sstv.wav", rig_hooks=rig_hooks)

# Option B: Raspberry Pi 5 / 4 GPIO
gpio_hooks = nSSTV.make_rpi_gpio_hooks(ptt_pin=17, active_high=True, ptt_delay=0.2)
nSSTV.transmit("photo_sstv.wav", rig_hooks=gpio_hooks)

# Option C: serial RTS/DTR line (SignaLink and USB interfaces)
serial_hooks = nSSTV.make_serial_dtr_rts_hooks(port="/dev/ttyUSB0", line="dtr")
nSSTV.transmit("photo_sstv.wav", rig_hooks=serial_hooks)
```

### 3. Decoding: single, multi-transmission, and batch

```python
import nsstv

# Single transmission
single = nSSTV.decode(
    audio_path="received_recording.wav",  # REQUIRED
    out="decoded_images/photo_1",         # output base path
    output_mode="raw",                    # "raw", "polaroid", "spectrogram", "stack", "all"
    auto_levels=True,                     # auto-stretch contrast (default False)
    denoise="light",                      # "off", "light", "medium", "strong"
    diagnostics=True,                     # save annotated spectrogram (default False)
)

# Decode-all: every transmission in a long recording
multi = nSSTV.decode_all(
    audio_path="long_field_day_recording.wav",  # REQUIRED
    out_dir="field_day_decoded",
    slant_search=0.03,                          # auto-slant search range
    output_mode="all",
)
print("Decoded transmissions:", multi["decoded_count"])

# Batch: a whole directory, recursively
batch = nSSTV.batch_decode(
    input_dir="./field_recordings",             # REQUIRED
    out_dir="./batch_decoded",
    recursive=True,
    output_mode="raw",
)
print("Batch decoded files:", batch["processed_count"])
```

### 4. Continuous soundcard receiver

```python
import nsstv

nSSTV.live_listen_and_decode(
    out_dir="live_rx_images",  # default "live_decoded"
    sample_rate=48000,
    device=None,               # soundcard index or name
    keep_wav=True,             # save the session recording
)
```

### 5. Direct RTL-SDR receiver

```python
import nsstv

# 145.800 MHz FM for International Space Station passes
nSSTV.sdr_listen_and_decode(
    frequency_hz=145800000,   # default 145800000
    sample_rate=2000000,      # SDR IQ sample rate
    out_dir="iss_images",
)
```

### 6. Roundtrip testing and benchmarking

```python
import nsstv

# Encode -> decode -> score (MAE, PSNR dB, correlation)
rt_res = nSSTV.rt(
    image_path="photo.jpg",
    mode="PD-180",
    fit="cover",
    denoise="light",
)
print("Roundtrip PSNR:", rt_res["metrics"]["psnr"], "dB")

# Rank every mode by quality on your image
bench_summary = nSSTV.bench("photo.jpg")
print("Top ranked mode:", bench_summary["best_mode"])
```

What `rt()` returns:

```python
r = nSSTV.rt("photo.jpg")

r["ok"]             # True/False
r["mode"]           # mode name
r["vis_ok"]         # True
r["wav"]            # path to audio
r["raw"]            # path to decoded image
r["comparison"]     # path to comparison PNG
r["metrics"]        # {"mae": 7.55, "psnr": 27.0, "correlation": 0.96}
r["quality"]        # {"overall": 1.0, ...}
```

### 7. Image comparison

```python
import nsstv

metrics = nSSTV.compare("original.png", "decoded.png")
print("MAE:", metrics["mae"], "PSNR:", metrics["psnr"], "dB")

side_path = nSSTV.side("original.png", "decoded.png", out="side_by_side.png")
```

### 8. Handy extras

```python
nSSTV.cut("tx.wav", 16.0)                       # first 16s → tx_cut.wav
nSSTV.join(["a.wav", "b.wav"], gap=5.0)         # join with silence between
nSSTV.add_caption_to_image("in.jpg", "out.jpg", text="NANA")
nSSTV.random_modes(4, seed=7)                   # repeatable random mode pick
nSSTV.add_callsign_to_image(
    "photo.jpg", "out.jpg",
    callsign="W1AW", position="bottom-right", opacity=0.7,
)
```

### 9. Raspberry Pi background service

```python
import nsstv

service_unit = nSSTV.make_rpi_systemd_service(
    service_name="nsstv-rx",
    out_dir="/home/pi/sstv_images",
    sample_rate=48000,
    keep_wav=True,
)
print(service_unit)
```

Generated unit:

```ini
[Unit]
Description=nSSTV Radio encoding and decoding
After=network.target sound.target

[Service]
Type=simple
User=pi
ExecStart=/usr/bin/python3 -m nsstv listen --out-dir /home/pi/sstv_images --sample-rate 48000 --keep-wav
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

---

## Command Line Interface

Every command runs as `nSSTV <command>` or `python3 -m nsstv <command>`. The two you will use most are **`rt`** (test the whole pipeline) and **`bench`** (find the best mode).

### `rt`: roundtrip

Encode → decode → metrics → comparison PNG. The best starting point.

```bash
nSSTV rt photo.jpg
nSSTV rt photo.jpg --mode PD-180
nSSTV rt photo.jpg --fit cover
nSSTV rt photo.jpg --output-mode all
nSSTV rt photo.jpg --output-mode stack
nSSTV rt photo.jpg --image-format png
nSSTV rt photo.jpg --denoise light
nSSTV rt photo.jpg --auto-levels
nSSTV rt photo.jpg --no-comparison
```

### `bench`: benchmark every mode

Runs `rt` across every mode, ranks the results, flags weak ones, and writes `bench.json` and `bench.csv`.

```bash
nSSTV bench photo.jpg
nSSTV bench photo.jpg --modes random             # 5 random modes
nSSTV bench photo.jpg --modes random --count 3   # 3 random modes
nSSTV bench photo.jpg --modes random --seed 7    # repeatable pick
nSSTV bench photo.jpg --modes 3                  # same as random --count 3
nSSTV bench photo.jpg --modes "PD-50"            # bench one experimental mode
nSSTV bench photo.jpg --include-experimental     # include experimental modes
```

### `encode`: image → WAV/MP3

```bash
nSSTV encode image.jpg
nSSTV encode image.jpg --mode PD-180
nSSTV encode image.jpg --wav-out encoded.wav
nSSTV encode image.jpg --mp3-out encoded.mp3
nSSTV encode image.jpg --mp3-bitrate 320k
nSSTV encode image.jpg --fit contain             # default: letterbox, no distortion
nSSTV encode image.jpg --fit cover               # crop to fill frame
nSSTV encode image.jpg --fit stretch             # squash
nSSTV encode image.jpg --background 255,255,255  # white padding instead of black
nSSTV encode image.jpg --callsign W1AW
nSSTV encode image.jpg --caption "Test - PD-180"
nSSTV encode image.jpg --caption "Test" --caption-position top
nSSTV encode image.jpg --caption "Test" --caption-position bottom
nSSTV encode image.jpg --no-vis                  # skip VIS header tones
nSSTV encode image.jpg --sample-rate 48000
nSSTV encode image.jpg --amplitude 0.80
nSSTV encode image.jpg --custom-modes-json examples/custom_modes.json
```

### `transmit`: key the radio and send

```bash
# Raspberry Pi GPIO PTT
nSSTV transmit photo.jpg --mode "Martin M1" --gpio-pin 17

# Hamlib CAT control
nSSTV transmit photo.jpg --mode "PD-180" --rig-model 2024 --rig-file /dev/ttyUSB0
```

### `decode`: WAV/MP3 → image

```bash
nSSTV decode input.wav
nSSTV decode input.mp3
nSSTV decode input.wav --out-base output.jpg
nSSTV decode input.wav --out-base output.png --image-format png
nSSTV decode input.wav --output-mode raw
nSSTV decode input.wav --output-mode polaroid_text
nSSTV decode input.wav --output-mode polaroid_notext
nSSTV decode input.wav --output-mode spectrogram_text
nSSTV decode input.wav --output-mode spectrogram_notext
nSSTV decode input.wav --output-mode stack
nSSTV decode input.wav --output-mode all
nSSTV decode input.wav --auto-levels
nSSTV decode input.wav --denoise off|light|medium|strong
nSSTV decode input.wav --diagnostics             # spectrogram with markers
nSSTV decode input.wav --report report.json      # full JSON decode report
nSSTV decode input.wav --no-sidecar              # skip .json sidecars
nSSTV decode input.wav --force-mode "Martin M1"  # skip VIS detection
nSSTV decode input.wav --slant-search 0.03       # line-timing search (default 0.03)
nSSTV decode input.wav --custom-modes-json examples/custom_modes.json
```

### `decode-all`: every image in one recording

Finds and decodes every transmission inside one long WAV/MP3.

```bash
nSSTV decode-all recording.wav
nSSTV decode-all recording.wav --out-dir decoded
nSSTV decode-all recording.wav --out-dir decoded --output-mode all
nSSTV decode-all recording.wav --image-format png
nSSTV decode-all recording.wav --auto-levels
nSSTV decode-all recording.wav --denoise medium
nSSTV decode-all recording.wav --diagnostics
nSSTV decode-all recording.wav --no-sidecar
nSSTV decode-all recording.wav --report report.json
nSSTV decode-all recording.wav --force-mode "Martin M1"
nSSTV decode-all recording.wav --slant-search 0.03
nSSTV decode-all recording.wav --custom-modes-json examples/custom_modes.json
```

Output:

```text
decoded/
├── recording_summary.json
├── recording_000_PD-180_raw.jpg
├── recording_000_PD-180_polaroid_text.jpg
├── recording_001_Martin_M1_raw.jpg
└── ...
```

### `batch`: decode a whole folder

```bash
nSSTV batch ./recordings
nSSTV batch ./recordings --out-dir ./decoded
nSSTV batch ./recordings --out-dir ./decoded --output-mode all
nSSTV batch ./recordings --image-format png
nSSTV batch ./recordings --auto-levels
nSSTV batch ./recordings --denoise light
nSSTV batch ./recordings --diagnostics
nSSTV batch ./recordings --no-sidecar
nSSTV batch ./recordings --no-recursive          # don't search subfolders
nSSTV batch ./recordings --stop-on-error
nSSTV batch ./recordings --custom-modes-json examples/custom_modes.json
```

Output:

```text
decoded/
├── batch_summary.json
├── summary.csv
├── recording_1/
│   ├── recording_1_000_PD-180_raw.jpg
│   └── ...
└── recording_2/
```

### `live`: record, then decode

Requires `pip install "nSSTV[live]"`.

```bash
nSSTV live
nSSTV live --seconds 240
nSSTV live --seconds 240 --out-base live_record.jpg
nSSTV live --out-base live.jpg --diagnostics
nSSTV live --device 2                            # specific audio device index
nSSTV live --output-mode all
nSSTV live --auto-levels
nSSTV live --denoise medium
nSSTV live --image-format png
```

### `listen` / `live-listen`: continuous receiver

```bash
nSSTV listen --out-dir live_rx --sample-rate 48000
nSSTV live-listen --out-dir live_rx --sample-rate 48000   # alias
```

### `roundtrip`: decode audio, then re-encode it

```bash
nSSTV roundtrip input.wav
nSSTV roundtrip input.wav --out-dir decoded
nSSTV roundtrip input.wav --encoded-wav-out roundtrip.wav
```

### `compare`, `side`, `fit`

```bash
nSSTV compare original.png decoded.png          # MAE, PSNR, correlation
nSSTV side original.png decoded.png --out side.png
nSSTV fit photo.jpg --mode PD-180 --how cover   # preview image fit
```

### `testcard` / `card`: calibrated test card

```bash
nSSTV testcard --mode PD-180
nSSTV testcard --mode "Scottie S1" --out testcard.png
nSSTV card --mode PD-180 --out testcard.png     # same thing
```

<p align="center">
  <img width="640" height="496" alt="nSSTV calibrated test card" src="https://github.com/user-attachments/assets/5b78aa13-a53a-4c60-a74b-56de24965e9f" />
</p>

### `modes`: list all modes and VIS codes

```bash
nSSTV modes
```

### Script mode (environment variables)

Run `nSSTV` with no arguments and it reads its configuration from the environment:

```bash
NSSTV_ACTION=decode NSSTV_AUDIO_INPUT=input.wav nSSTV
NSSTV_ACTION=encode NSSTV_IMAGE_INPUT=image.jpg nSSTV

NSSTV_ACTION=batch \
NSSTV_WORKDIR=./recordings \
NSSTV_OUT_DIR=./decoded \
NSSTV_IMAGE_FORMAT=png \
NSSTV_DIAGNOSTICS=1 \
NSSTV_AUTO_LEVELS=1 \
NSSTV_DENOISE=medium \
nSSTV
```

Key variables: `NSSTV_ACTION`, `NSSTV_AUDIO_INPUT`, `NSSTV_IMAGE_INPUT`, `NSSTV_OUTPUT_BASE`, `NSSTV_OUT_DIR`, `NSSTV_WORKDIR`, `NSSTV_IMAGE_FORMAT`, `NSSTV_OUTPUT_MODE`, `NSSTV_ENCODE_MODE`, `NSSTV_CAPTION`, `NSSTV_IMAGE_FIT`, `NSSTV_DENOISE`, `NSSTV_AUTO_LEVELS`, `NSSTV_DIAGNOSTICS`, `NSSTV_SAMPLE_RATE`, `NSSTV_LIVE_SECONDS`, `NSSTV_CUSTOM_MODES_JSON`. Full list: `nSSTV.info("cli")`.

---

## Output Styles and Image Fit

### Output styles (`--output-mode` / `output_mode`)

| Style | Preview | Description |
| :--- | :---: | :--- |
| `raw` (default) | <img width="240" alt="raw" src="https://github.com/user-attachments/assets/c39476f5-fa31-4b83-bbbf-b517e91f41b3" /> | Plain decoded image |
| `polaroid_text` | <img width="240" alt="polaroid_text" src="https://github.com/user-attachments/assets/ab46a3af-c3bd-42f2-81c8-5d1114f2a803" /> | Polaroid border with mode, timestamp, and quality metadata |
| `polaroid_notext` | <img width="240" alt="polaroid_notext" src="https://github.com/user-attachments/assets/03b1d2b2-d8a5-4d84-a653-4aa7b08d09ed" /> | Polaroid border, no text |
| `spectrogram_text` | <img width="240" alt="spectrogram_text" src="https://github.com/user-attachments/assets/00b658d8-bcb8-4510-a6d6-0b8bf28e3d8d" /> | Image plus spectrogram with signal timing annotations |
| `spectrogram_notext` | <img width="240" alt="spectrogram_notext" src="https://github.com/user-attachments/assets/bc1b9b49-6b7f-4574-b8f4-96fdf4b7e455" /> | Image plus clean spectrogram |
| `stack` | <img width="240" alt="stack" src="https://github.com/user-attachments/assets/e8d268a8-bd6a-483e-9448-6e6f9959dd05" /> | Image stacked above spectrogram |
| `all` | n/a | Every style in one pass |

The short aliases `polaroid` and `spectrogram` are also accepted in the Python API.

```bash
nSSTV decode input.wav --output-mode all
```

### Image fit (`--fit` / `image_fit`)

Every SSTV mode has fixed dimensions (PD-180 is always 640×496), so nSSTV fits your photo instead of squashing it:

- **`contain`** (default): keep aspect ratio and pad the edges. No distortion.
- **`cover`**: keep aspect ratio and crop to fill the frame.
- **`stretch`**: distort the image to the exact frame dimensions.

```bash
nSSTV encode portrait.jpg --mode PD-180 --fit cover
nSSTV encode portrait.jpg --mode PD-180 --fit contain --background 255,255,255
```

---

## Diagnostics

Add `--diagnostics` (or `diagnostics=True`) to any decode for a spectrogram annotated with the leader tone, VIS header bits, sync alignment, and image boundaries.

<p align="center">
  <img width="1200" height="520" alt="nSSTV diagnostic spectrogram" src="https://github.com/user-attachments/assets/5c84dc3c-70da-4c5b-9eea-e4a6296d802e" />
</p>

```bash
nSSTV decode input.wav --diagnostics
```

---

## Callsign and Text Overlays

Stamp station details onto an image before transmission. Templates support `{callsign}`, `{grid}`, timestamp, frequency, and mode tags.

```python
import nsstv

overlays = [
    nSSTV.TextOverlay(
        text="DE {callsign} GRID {grid}",
        position="bottom-right",      # aliases: watermark, stamp, callsign, header, footer, ur, ll
        font_style="bold",
        text_color=(255, 255, 255),
        bg_color=(20, 20, 20),
        outline_color=(0, 0, 0),
        outline_width=3,
        opacity=0.90,
    ),
    nSSTV.TextOverlay(
        text="2024-01-15 14:32 UTC | Scottie S1",
        position="top-left",
        font_size=16,
        font_style="italic",
        bg_color=(0, 0, 0),
        opacity=0.75,
        # rotation=15.0,              # arbitrary angle; bounding box expands automatically
    ),
]

ctx = {"callsign": "N0CALL", "grid": "FN31"}
nSSTV.add_texts_to_image(
    image_path="ufo_decoded.png",
    out_path="ufo_stamped.png",
    overlays=overlays,
)
```

<p align="center">
  <img width="320" height="256" alt="nSSTV callsign overlay demo" src="https://github.com/user-attachments/assets/8eccb4e1-e659-44e0-8a27-7c054d160d90" />
</p>

---

## Rig Control

Rig control is entirely opt-in through the `RigHooks` dataclass. Without `rig_hooks`, nSSTV behaves exactly as it would with no rig attached, and Hamlib is never required.

### `RigHooks`

```python
@dataclass
class RigHooks:
    before_record:   callable = None  # live RX: tune rig, set mode
    after_record:    callable = None  # live RX: read back actual frequency
    before_decode:   callable = None  # any decode: fired before processing
    after_decode:    callable = None  # any decode: fired after processing
    before_transmit: callable = None  # encode: PTT key-up
    after_transmit:  callable = None  # encode: PTT key-down
```

Each callback receives a `ctx` dict. If it returns a dict, those key/value pairs are merged back into `ctx`, so data can flow between hooks. Anything stored under `ctx["rig"]` ends up in `result["rig"]` on the returned decode or encode result.

### Wiring Hamlib (Python bindings)

```python
import Hamlib
import nsstv

rig = Hamlib.Rig(Hamlib.RIG_MODEL_FT817ND)
rig.set_conf("rig_pathname", "/dev/ttyUSB0")
rig.set_conf("serial_speed", "9600")
rig.open()

hooks = nSSTV.RigHooks(
    # Tune to 14.230 MHz before recording starts
    before_record=lambda ctx: rig.set_freq(Hamlib.RIG_VFO_A, 14_230_000),

    # Read the actual frequency back and log it in the result
    after_record=lambda ctx: {"rig": {"freq_hz": rig.get_freq()}},

    # Key up before transmitting
    before_transmit=lambda ctx: rig.set_ptt(Hamlib.RIG_VFO_A, Hamlib.RIG_PTT_ON),

    # Key down when done
    after_transmit=lambda ctx: rig.set_ptt(Hamlib.RIG_VFO_A, Hamlib.RIG_PTT_OFF),
)
```

### Using `rigctl` instead

If `python-hamlib` is not available on your platform, `rigctl` via subprocess works fine:

```python
import subprocess
import nsstv

def rigctl(*args, model=361, port="/dev/ttyUSB0"):
    subprocess.run(
        ["rigctl", "-m", str(model), "-r", port, *args],
        check=True,
        timeout=10,
    )

hooks = nSSTV.RigHooks(
    before_record=lambda ctx: rigctl("M", "USB", "0", "F", "14230000"),
    before_transmit=lambda ctx: rigctl("T", "1"),
    after_transmit=lambda ctx: rigctl("T", "0"),
)
```

### Live receive

```python
result = nSSTV.live_record_then_decode(
    seconds=120,
    out_base="capture.jpg",
    rig_hooks=hooks,
)

# result["rig"]["freq_hz"] holds the frequency read back from the rig
```

Hook order for `live_record_then_decode`:

```text
before_record  →  [audio capture]  →  after_record
               →  before_decode    →  [decode]  →  after_decode
```

### Decode a file

```python
result = nSSTV.decode("recording.wav", rig_hooks=hooks)
```

Only `before_decode` and `after_decode` fire here, since there is no record step.

### Transmit

```python
info = nSSTV.encode("image.jpg", rig_hooks=hooks)
```

`before_transmit` fires after the WAV is written and `after_transmit` fires immediately after. When using `encode` with raw hooks, playing the audio is your responsibility (`sounddevice`, `aplay`, or any player): key up in `before_transmit`, play the file, then key down in `after_transmit`. For fully managed playback with guaranteed PTT release, use [`nSSTV.transmit()`](#2-managed-radio-transmission-with-ptt).

---

## Custom Modes

Define your own mode in Python:

```python
import nsstv

reg = nSSTV.ModeRegistry()

reg.add_custom_mode(
    name="My Mode",
    vis_code=123,
    width=320,
    height=240,
    color="rgb",
    line_seconds=0.5,
    sync_seconds=0.005,
    sync_offset=0.0,
    channels=(
        ("R", 0.006, 0.16),
        ("G", 0.166, 0.16),
        ("B", 0.326, 0.16),
    ),
)

nSSTV.encode(
    "image.jpg",
    mode_name="My Mode",
    registry=reg,
    wav_out="tx.wav",
)
```

Or load modes from a JSON file and pass it to any command with `--custom-modes-json`.

---

## Supported Modes

| Family | Mode | Resolution | Line time | VIS | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Martin** | Martin M1 | 320×256 | 448.0 ms | 0x2B | Standard European HF mode |
| **Martin** | Martin M2 | 320×256 | 224.2 ms | 0x28 | High-speed Martin mode |
| **Scottie** | Scottie S1 | 320×256 | 428.8 ms | 0x3C | Standard US HF mode |
| **Scottie** | Scottie S2 | 320×256 | 269.8 ms | 0x38 | High-speed Scottie mode |
| **Scottie** | Scottie DX | 320×256 | 1045.6 ms | 0x4C | Long-distance weak-signal mode. *Experimental in `decode-all`* |
| **Robot** | Robot 36 | 320×240 | 150.0 ms | 0x08 | Y/C YUV mode (ISS pass default) |
| **Robot** | Robot 72 | 320×240 | 300.0 ms | 0x0C | High-resolution Y/C mode |
| **PD** | PD-50 | 512×400 | 252.8 ms | 0x5D | *Experimental* |
| **PD** | PD-90 | 512×400 | 452.8 ms | 0x63 | High resolution |
| **PD** | PD-120 | 640×496 | 379.2 ms | 0x5F | Standard high-resolution mode |
| **PD** | PD-160 | 512×400 | 800.0 ms | 0x62 | High resolution |
| **PD** | PD-180 | 640×496 | 569.6 ms | 0x60 | Popular high-resolution mode (default) |
| **PD** | PD-240 | 640×496 | 758.4 ms | 0x58 | Ultra high-resolution mode |
| **PD** | PD-290 | 800×616 | 940.8 ms | 0x5E | Maximum-resolution PD mode |
| **MMSSTV** | MP73 to MR320 | 320×256 / 640×496 | Variable | 0x23 | 16-bit extended VIS and N-VIS modes |
| **Pasokon** | P3, P5, P7 | 640×496 | Variable | 0x71 to 0x73 | *Experimental* |
| **Wraase** | SC-180 | 320×256 | 705.0 ms | 0x37 | Classic 180 s RGB mode |

Run `nSSTV modes` for the live list, or `nSSTV bench photo.jpg` to rank modes by quality on your own image.

### Experimental modes

Five modes are marked **experimental**. They still encode, decode, and appear in `nSSTV modes` (tagged `experimental`), but treat their results with care.

| Mode | VIS | Experimental scope |
| :--- | :--- | :--- |
| Pasokon P3 | 113 | everywhere |
| Pasokon P5 | 114 | everywhere |
| Pasokon P7 | 115 | everywhere |
| PD-50 | 93 | everywhere |
| Scottie DX | 76 | `decode-all` only |

- **Pasokon P3/P5/P7 and PD-50**: layouts are unverified and decodes may be unreliable. Their VIS codes stay in the table on purpose, so nSSTV can name these signals and flag them as experimental instead of failing to recognize them.
- **Scottie DX**: plain encode and decode are fine, but inside `decode-all` its very long lines are easy to misjudge in a recording holding several transmissions, so the scanner flags it.

`bench` skips experimental modes by default. Name one explicitly or pass `--include-experimental`. When a decode detects an experimental mode, the result carries `"experimental": true` plus a short notice, and `decode-all` adds the same flag to each affected transmission and to the summary.

---

## Notes & Operating Tips

> [!NOTE]
> **Clock drift and slanted images.** Slanted pictures come from small clock-speed differences between the transmitting and receiving soundcards. Enable auto-slant correction with `--slant-search 0.03` or `slant_search=0.03`.

> [!NOTE]
> **Audio levels.** Set your radio interface output so the transmitter runs in linear SSB without clipping or over-driving ALC. When receiving, adjust the soundcard input so signals fill roughly 50% to 80% of the dynamic range.

> [!NOTE]
> **MP3 compression.** MP3 alters the audio frequencies SSTV depends on. nSSTV supports it, but WAV gives the cleanest pictures.

> [!NOTE]
> **Raspberry Pi permissions.** For GPIO PTT control, make sure your user belongs to the `gpio` and `dialout` groups:
> ```bash
> sudo usermod -aG gpio,dialout pi
> ```

---

## License

nSSTV is released under the **MIT License**. Free for amateur radio operators, open-source projects, educational research, and commercial use.

MIT © Nana Kwadjo Agyare
