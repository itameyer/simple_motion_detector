# Simple motion detector

A real-time motion detection pipeline that plays a local video, detects motion frame-by-frame, and displays the results overlaid on the original video at the video's native frame rate.

---

## How it works

Three independent processes run in parallel:

- **Reader** — decodes the video and writes frames into shared memory and metadata to queues
- **Detector** — consume metadata from queue, reads corresponding frame, runs motion detection, and emits bounding boxes
- **Viewer** — consume metadata from 2 queues (detections + frames), joins frames with their detections and displays the result at the correct FPS

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py --video path/to/video.mp4
```

| Flag | Description |
|---|---|
| `--video PATH` | Path to the input video file (required) |
| `-b, --blur-detections` | Blur detected regions instead of drawing bounding boxes |

## Exception handeling

a keyboard interrup exception is caught gracefully by the system.
other exception are handled as a fast-fail (FileNotFound, etc.) by the init_successful variable, but aren't addressed specifically.