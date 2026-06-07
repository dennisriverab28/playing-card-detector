# VideoStream.py
# PC/Webcam version for the EdjeElectronics Playing Card Detector
# Keeps the same API as the original PiCamera version.

import cv2
import threading
import time

class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30, usePiCamera=0, src=0):
        """
        resolution: (width, height)
        framerate : target FPS (best effort)
        usePiCamera: kept for API compatibility; ignored on PC
        src       : camera index (0 is default webcam)
        """
        self.width, self.height = int(resolution[0]), int(resolution[1])
        self.framerate = int(framerate)
        self.src = src

        # On Windows, CAP_DSHOW often avoids long startup delays
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            # Fallback without CAP_DSHOW if needed
            self.stream = cv2.VideoCapture(self.src)

        # Try to set properties (best effort)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.stream.set(cv2.CAP_PROP_FPS,          self.framerate)

        # Warm-up
        time.sleep(0.3)

        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed:
            raise RuntimeError(f"VideoStream: unable to read from camera (index {self.src})")

        self.stopped = False
        self.read_lock = threading.Lock()
        self.thread = None

    def start(self):
        """Start the background frame grab thread."""
        if self.thread is None:
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
        return self

    def update(self):
        """Background thread: keep the most recent frame."""
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            # Small sleep to avoid pegging CPU if driver ignores FPS setting
            if self.framerate > 0:
                time.sleep(1.0 / (self.framerate * 1.5))

        # Clean up when stopped
        self.stream.release()

    def read(self):
        """Return the latest frame (or None if grab failed)."""
        with self.read_lock:
            if not self.grabbed:
                return None
            return self.frame.copy()

    def stop(self):
        """Signal the thread to stop and release the camera."""
        self.stopped = True
        # Give update() a moment to exit gracefully
        time.sleep(0.05)
