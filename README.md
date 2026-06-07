# Playing Card Detector

**Stack:** Python · OpenCV · NumPy  
**Platform:** Raspberry Pi camera or USB webcam

---

## Overview

A real-time computer vision system that detects and identifies standard playing cards from a live video feed. Using OpenCV image processing (thresholding, contour detection, perspective transform, and template matching), the system isolates cards in frame, extracts their rank and suit, and displays the identification on screen at ~10 FPS.

---

## Demo

See [`demo/Playing_card_project.mp4`](demo/Playing_card_project.mp4) for a live demo of the card detection in action.

---

## Features

- Real-time card detection from camera feed (Raspberry Pi camera or USB webcam)
- Template matching against 13 ranks × 4 suits = 52 card reference images
- Perspective transform to flatten and normalize detected card regions
- Contour-based card isolation (handles multiple cards in frame)
- Live FPS display overlay
- Threaded video stream for performance

---

## Project Structure

```
Playing_Card_Detector/
├── Card_Detector.py        # Main program — camera loop, detection, display
├── Cards.py                # Card image processing, rank/suit matching logic
├── VideoStream.py          # Threaded camera stream class
├── Card_Imgs/              # Reference images for template matching
│   ├── Ace.jpg
│   ├── Two.jpg  ...  King.jpg    (13 rank images)
│   ├── Spades.jpg
│   ├── Hearts.jpg
│   ├── Clubs.jpg
│   └── Diamonds.jpg        (4 suit images)
├── demo/
│   └── Playing_card_project.mp4  # Live demo video
└── README.md
```

---

## Setup & Run

### Requirements
```bash
pip install opencv-python numpy
```
For Raspberry Pi camera support:
```bash
pip install picamera
```

### Run
```bash
python Card_Detector.py
```

**Camera selection** — Line 29 in `Card_Detector.py`:
```python
# Raspberry Pi camera:
videostream = VideoStream.VideoStream((1280, 720), 10, 1, 0).start()

# USB webcam:
videostream = VideoStream.VideoStream((1280, 720), 10, 2, 0).start()
# Change the third argument from 1 to 2
```

Press `q` to quit.

---

## How It Works

1. **Frame Capture** — `VideoStream.py` grabs frames in a background thread to prevent main loop blocking.
2. **Preprocessing** — Each frame is converted to grayscale, blurred, and thresholded to isolate card edges.
3. **Contour Detection** — OpenCV finds contours; large rectangles are identified as potential cards.
4. **Perspective Transform** — The card region is warped into a flat, standard-orientation image.
5. **Rank & Suit Extraction** — Corner regions of the flattened card are cropped and matched against `Card_Imgs/` templates using template matching.
6. **Display** — Identified rank and suit are drawn onto the original frame with bounding box and FPS counter.

---

## Notes

- Reference images in `Card_Imgs/` must be present for detection to work.
- Accuracy depends on lighting conditions and card quality.
- Camera resolution is set to 1280×720 at 10 FPS — adjust in `Card_Detector.py` if needed.
