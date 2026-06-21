# Moodito

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=georgiosnikitas_moodito&metric=alert_status)](https://sonarcloud.io/summary/overall?id=georgiosnikitas_moodito)
[![CI](https://github.com/georgiosnikitas/moodito/actions/workflows/ci.yml/badge.svg)](https://github.com/georgiosnikitas/moodito/actions/workflows/ci.yml)
[![Release](https://github.com/georgiosnikitas/moodito/actions/workflows/release.yml/badge.svg?event=push)](https://github.com/georgiosnikitas/moodito/actions/workflows/release.yml)
[![Python](https://img.shields.io/badge/Python-3.11–3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white)](https://www.apple.com/macos/)
[![License](https://img.shields.io/github/license/georgiosnikitas/moodito)](LICENSE)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/georgiosnikitas)

A macOS menu bar app that watches your face through the webcam and shows your
current emotion as an emoji + label in the menu bar.

Emotion recognition is powered by **Google's MediaPipe Face Landmarker**, which
produces ARKit-style facial *blendshapes*. Moodito maps those blendshapes to a
small set of coarse emotions: happy, sad, surprised, angry, and neutral.

> Note on Google ML Kit: ML Kit is a mobile-only SDK (Android/iOS) and does not
> run on macOS. MediaPipe is Google's cross-platform equivalent that does run on
> macOS, so it is used here.

## Requirements

- macOS
- Python 3.11–3.13 (MediaPipe wheels are not yet published for 3.13+)
- A webcam

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

On first launch Moodito downloads the MediaPipe `face_landmarker.task` model
(~3.7 MB) next to `app.py`. macOS will prompt for **camera permission** the
first time — grant it, otherwise the app shows a "could not open webcam" error.

The menu bar shows the live emotion (e.g. `😀 happy`). The dropdown menu lets you:

- See the detected emotion and confidence
- **Pause / Resume** detection (stops processing webcam frames)
- **Quit**

## Build a standalone .app

To produce a double-clickable `Moodito.app` that doesn't need a terminal or an
activated virtualenv:

```bash
./build.sh
```

This uses PyInstaller (see [`moodito.spec`](moodito.spec)) and writes
`dist/Moodito.app`. The bundle is configured as a menu-bar-only app
(`LSUIElement`) and declares `NSCameraUsageDescription`, so on first launch
macOS prompts for camera access for **Moodito** itself.

```bash
open dist/Moodito.app            # run it
cp -R dist/Moodito.app /Applications/   # install it
```

The app is unsigned, so the first launch may require right-click -> **Open** (or
allowing it under System Settings -> Privacy & Security). The model is cached in
`~/Library/Application Support/Moodito/`.

## How it works

1. `app.py` captures webcam frames with OpenCV on a background thread.
2. Each frame is passed to MediaPipe Face Landmarker (`detect_for_video`).
3. The 52 blendshape scores are mapped to an emotion in `emotion.py`.
4. A `rumps` timer refreshes the menu bar title every 0.3s.

## Tuning

The emotion heuristics live in [`emotion.py`](emotion.py) in `infer_emotion()`.
Adjust the linear weights or the `0.25` neutral threshold to make detection more
or less sensitive.

## Support

If you enjoy Moodito, you can support development by buying me a coffee ☕

[buymeacoffee.com/georgiosnikitas](https://buymeacoffee.com/georgiosnikitas)

<img src="bmc_qr.png" alt="Buy Me a Coffee QR code" width="200">

There's also a **Buy Me a Coffee ☕** item in the menu bar dropdown that opens
the page directly.

## License

Moodito is released under the [MIT License](LICENSE).

