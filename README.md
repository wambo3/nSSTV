<p align="center">
  <img src="https://github.com/user-attachments/assets/8eed1fbe-a135-4d00-b7f0-d51395689181" alt="nSSTV" width="600" />
</p>

<p align="center">
  Turn images into SSTV radio audio, and SSTV audio back into images
</p>


# nSSTV

A Python SSTV encoder/decoder for turning images into WAV/MP3 audio and SSTV audio back into images.


## What is SSTV?

**Slow-Scan Television** is a way to send still pictures over radio as sound. Each
pixel's brightness becomes an audio tone (1500 Hz = black, 2300 Hz = white), and the
image is painted one line at a time.

## Features

- **Encode** any image into SSTV audio (WAV or MP3)
- **Decode** SSTV audio back into images, with automatic mode detection
- **28 modes** — Martin, Scottie, Robot, PD, Pasokon, and Wraase families
- **Decode-all** — pull every transmission out of one long recording
- **Batch decode** — process an entire folder of recordings at once
- **Live capture** — record from a mic or audio cable, then decode
- **Smart image fit** — `contain`, `cover`, or `stretch` so photos aren't squashed
- **Output styles** — raw, Polaroid frame, spectrogram, or stacked, with optional captions
- **Diagnostics** — annotated spectrograms showing leader, VIS, and image boundaries
- **Quality metrics** — MAE, PSNR, and correlation on every roundtrip
- **Benchmarking** — rank every mode by quality on your own image
- **Custom modes** — define your own from Python or JSON
- **Cleanup tools** — auto-levels and denoise presets for noisy recordings
- **Test cards** — generate calibrated test images for any mode

## Install

```bash
pip install nSSTV            # core
pip install "nSSTV[mp3]"    # + MP3 support (needs ffmpeg)
pip install "nSSTV[live]"   # + live capture
pip install "nSSTV[all]"    # everything
```

MP3 needs ffmpeg:

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Linux
```

> MP3 compression damages SSTV tones, so MP3 decodes come out noisier. Fine for casual sharing.

---

## Quick start

```python
import nSSTV

nSSTV.encode("photo.jpg")           # → photo_sstv.wav
nSSTV.decode("recording.wav")       # → recording_decoded.png
nSSTV.decode_all("recording.wav")   # → every image in one recording
nSSTV.rt("photo.jpg")               # roundtrip + metrics + comparison
nSSTV.bench("photo.jpg")            # try every mode, rank them
```

<img width="1226" height="636" alt="photo-1_comparison" src="https://github.com/user-attachments/assets/4273c915-325b-4b72-ae69-8ebeb3df75ac" />

---

# Commands

Every command works from the terminal as `nsstv <command>`. The two you'll use
most are `rt` (test the whole pipeline) and `bench` (find the best mode).

## `rt` — Roundtrip

Encode → decode → metrics → comparison PNG. The best starting point.

```bash
nsstv rt photo.jpg
nsstv rt photo.jpg --mode PD-180
nsstv rt photo.jpg --fit cover
nsstv rt photo.jpg --output-mode all
nsstv rt photo.jpg --image-format png
nsstv rt photo.jpg --denoise light
nsstv rt photo.jpg --auto-levels
nsstv rt photo.jpg --no-comparison
```

## `bench` — Benchmark every mode

Runs `rt` across every mode, ranks the results, flags weak ones. Writes
`bench.json` and `bench.csv`.

```bash
nsstv bench photo.jpg
nsstv bench photo.jpg --modes random           # 5 random modes
nsstv bench photo.jpg --modes random --count 3 # 3 random modes
nsstv bench photo.jpg --modes random --seed 7  # repeatable pick
nsstv bench photo.jpg --modes 3                # same as random --count 3
```

## `encode` — Image → WAV/MP3

```bash
nsstv encode image.jpg
nsstv encode image.jpg --mode PD-180
nsstv encode image.jpg --wav-out encoded.wav
nsstv encode image.jpg --mp3-out encoded.mp3
nsstv encode image.jpg --mp3-bitrate 320k
nsstv encode image.jpg --fit contain            # default — letterbox, no distortion
nsstv encode image.jpg --fit cover              # crop to fill frame
nsstv encode image.jpg --fit stretch            # squash (old behavior)
nsstv encode image.jpg --background 255,255,255 # white padding instead of black
nsstv encode image.jpg --caption "Test - PD-180"
nsstv encode image.jpg --caption "Test" --caption-position top
nsstv encode image.jpg --caption "Test" --caption-position bottom
nsstv encode image.jpg --no-vis                 # skip VIS header tones
nsstv encode image.jpg --sample-rate 48000
nsstv encode image.jpg --amplitude 0.80
nsstv encode image.jpg --custom-modes-json examples/custom_modes.json
```

## `decode` — WAV/MP3 → image

```bash
nsstv decode input.wav
nsstv decode input.mp3
nsstv decode input.wav --out-base output.jpg
nsstv decode input.wav --out-base output.png --image-format png
nsstv decode input.wav --output-mode raw
nsstv decode input.wav --output-mode polaroid_text
nsstv decode input.wav --output-mode polaroid_notext
nsstv decode input.wav --output-mode spectrogram_text
nsstv decode input.wav --output-mode spectrogram_notext
nsstv decode input.wav --output-mode stack
nsstv decode input.wav --output-mode all
nsstv decode input.wav --auto-levels
nsstv decode input.wav --denoise off|light|medium|strong
nsstv decode input.wav --diagnostics            # spectrogram with markers
nsstv decode input.wav --report report.json     # full JSON decode report
nsstv decode input.wav --no-sidecar             # skip .json sidecars
nsstv decode input.wav --force-mode "Martin M1" # skip VIS detection
nsstv decode input.wav --slant-search 0.03      # line-timing search (default 0.03)
nsstv decode input.wav --custom-modes-json examples/custom_modes.json
```

## `decode-all` — Every image in one recording

Finds and decodes every transmission inside one long WAV/MP3.

```bash
nsstv decode-all recording.wav
nsstv decode-all recording.wav --out-dir decoded
nsstv decode-all recording.wav --out-dir decoded --output-mode all
nsstv decode-all recording.wav --image-format png
nsstv decode-all recording.wav --auto-levels
nsstv decode-all recording.wav --denoise medium
nsstv decode-all recording.wav --diagnostics
nsstv decode-all recording.wav --no-sidecar
nsstv decode-all recording.wav --report report.json
nsstv decode-all recording.wav --force-mode "Martin M1"
nsstv decode-all recording.wav --slant-search 0.03
nsstv decode-all recording.wav --custom-modes-json examples/custom_modes.json
```

Output:
```
decoded/
├── recording_summary.json
├── recording_000_PD-180_raw.jpg
├── recording_000_PD-180_polaroid_text.jpg
├── recording_001_Martin_M1_raw.jpg
└── ...
```

## `batch` — Decode a whole folder

```bash
nsstv batch ./recordings
nsstv batch ./recordings --out-dir ./decoded
nsstv batch ./recordings --out-dir ./decoded --output-mode all
nsstv batch ./recordings --image-format png
nsstv batch ./recordings --auto-levels
nsstv batch ./recordings --denoise light
nsstv batch ./recordings --diagnostics
nsstv batch ./recordings --no-sidecar
nsstv batch ./recordings --no-recursive         # don't search subfolders
nsstv batch ./recordings --stop-on-error
nsstv batch ./recordings --custom-modes-json examples/custom_modes.json
```

Output:
```
decoded/
├── batch_summary.json
├── summary.csv
├── recording_1/
│   ├── recording_1_000_PD-180_raw.jpg
│   └── ...
└── recording_2/
```

## `roundtrip` — Decode audio, then re-encode it

```bash
nsstv roundtrip input.wav
nsstv roundtrip input.wav --out-dir decoded
nsstv roundtrip input.wav --encoded-wav-out roundtrip.wav
```

## `compare` — Score two images

Returns MAE, PSNR, and correlation.

```bash
nsstv compare a.png b.png
```

## `side` — Side-by-side PNG

```bash
nsstv side a.png b.png
nsstv side a.png b.png out.png
```

## `testcard` / `card` — Generate a test card

```bash
nsstv testcard --mode PD-180
nsstv testcard --mode PD-180 --out testcard.png
nsstv card --mode PD-180 --out testcard.png     # same thing
```

<img width="640" height="496" alt="testcard" src="https://github.com/user-attachments/assets/5b78aa13-a53a-4c60-a74b-56de24965e9f" />

## `live` — Record, then decode

```bash
pip install "nSSTV[live]"

nsstv live
nsstv live --seconds 240
nsstv live --out-base live.jpg
nsstv live --out-base live.jpg --diagnostics
nsstv live --device 2                            # specific audio device index
nsstv live --output-mode all
nsstv live --auto-levels
nsstv live --denoise medium
nsstv live --image-format png
```

## `modes` — List all modes

```bash
nsstv modes
```

---

## Output styles (`--output-mode`)

| Style | Preview | Description |
|-------|---------|-------------|
| `raw` | <img width="240" alt="raw" src="https://github.com/user-attachments/assets/c39476f5-fa31-4b83-bbbf-b517e91f41b3" /> | Plain decoded image |
| `polaroid_text` | <img width="240" alt="polaroid_text" src="https://github.com/user-attachments/assets/ab46a3af-c3bd-42f2-81c8-5d1114f2a803" /> | Polaroid border + metadata text |
| `polaroid_notext` | <img width="240" alt="polaroid_notext" src="https://github.com/user-attachments/assets/03b1d2b2-d8a5-4d84-a653-4aa7b08d09ed" /> | Polaroid border, no text |
| `spectrogram_text` | <img width="240" alt="spectrogram_text" src="https://github.com/user-attachments/assets/00b658d8-bcb8-4510-a6d6-0b8bf28e3d8d" /> | Image + spectrogram + text |
| `spectrogram_notext` | <img width="240" alt="spectrogram_notext" src="https://github.com/user-attachments/assets/bc1b9b49-6b7f-4574-b8f4-96fdf4b7e455" /> | Image + spectrogram, no text |
| `stack` | <img width="240" alt="stack" src="https://github.com/user-attachments/assets/e8d268a8-bd6a-483e-9448-6e6f9959dd05" /> | Image stacked above spectrogram |
| `all` | — | Every style at once |

```bash
nsstv decode input.wav --output-mode all
```

---

## Image fit: no more stretched photos

Every SSTV mode has fixed dimensions (PD-180 is always 640×496), so nSSTV fits
your photo instead of squashing it:

- `contain` — keep aspect ratio, pad the edges (**default**, no distortion)
- `cover` — keep aspect ratio, crop to fill
- `stretch` — squash to fit

```bash
nsstv encode portrait.jpg --mode PD-180 --fit cover
nsstv encode portrait.jpg --mode PD-180 --fit contain --background 255,255,255
```

---

## Picking a mode

```bash
nsstv modes            # list all 28
nsstv bench photo.jpg  # rank them by quality on your image
```

---

## Python API

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

Handy extras:

```python
nSSTV.cut("tx.wav", 16.0)                       # first 16s → tx_cut.wav
nSSTV.join(["a.wav", "b.wav"], gap=5.0)         # join with silence between
nSSTV.add_caption_to_image("in.jpg", "out.jpg", text="NANA - GHANA")
nSSTV.random_modes(4, seed=7)                   # repeatable random mode pick
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

---

## Diagnostics

Add `--diagnostics` to any decode for a spectrogram marked with the leader, VIS
header, and image boundaries.

<img width="1200" height="520" alt="out_diagnostics" src="https://github.com/user-attachments/assets/5c84dc3c-70da-4c5b-9eea-e4a6296d802e" />

```bash
nsstv decode input.wav --diagnostics
```

---

## Custom modes

```python
import nSSTV
reg = nSSTV.ModeRegistry()
reg.add_custom_mode(
    name="My Mode", vis_code=123, width=320, height=240, color="rgb",
    line_seconds=0.5, sync_seconds=0.005, sync_offset=0.0,
    channels=(("R", 0.006, 0.16), ("G", 0.166, 0.16), ("B", 0.326, 0.16)),
)
nSSTV.encode("image.jpg", mode_name="My Mode", registry=reg, wav_out="tx.wav")
```

Or load from JSON and pass to any command with `--custom-modes-json`.

---

## Script mode (environment variables)

Run `nsstv` with no arguments and it reads these instead:

```bash
NSSTV_ACTION=batch \
NSSTV_WORKDIR=./recordings \
NSSTV_OUT_DIR=./decoded \
NSSTV_IMAGE_FORMAT=png \
NSSTV_DIAGNOSTICS=1 \
NSSTV_AUTO_LEVELS=1 \
NSSTV_DENOISE=medium \
nsstv
```

Key variables: `NSSTV_ACTION`, `NSSTV_AUDIO_INPUT`, `NSSTV_IMAGE_INPUT`,
`NSSTV_OUTPUT_BASE`, `NSSTV_OUT_DIR`, `NSSTV_IMAGE_FORMAT`, `NSSTV_OUTPUT_MODE`,
`NSSTV_ENCODE_MODE`, `NSSTV_CAPTION`, `NSSTV_IMAGE_FIT`, `NSSTV_DENOISE`,
`NSSTV_DIAGNOSTICS`, `NSSTV_SAMPLE_RATE`, `NSSTV_LIVE_SECONDS`,
`NSSTV_CUSTOM_MODES_JSON`. Full list: `nSSTV.info("cli")`.

---

## License

MIT © Nana Kwadjo Agyare
