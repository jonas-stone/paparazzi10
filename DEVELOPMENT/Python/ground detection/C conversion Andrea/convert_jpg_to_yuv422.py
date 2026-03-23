#!/usr/bin/env python3
"""
convert_jpg_to_yuv422.py

Convert JPEG (or any image OpenCV can read) to raw UYVY (YUV422) bytes
that the C test binary can consume.

USAGE:
    python3 convert_jpg_to_yuv422.py  <input.jpg>  [output.yuv422]

    If no output path given, writes to <input_stem>.yuv422

Prints the width and height so you can pass them to the C binary:
    ./test_obstacle_detection  output.yuv422  <width>  <height>  results/
"""

import sys
import os
import numpy as np
import cv2


def bgr_to_uyvy(bgr: np.ndarray) -> bytes:
    h, w = bgr.shape[:2]
    if w % 2:
        bgr = bgr[:, :w-1, :]
        w -= 1

    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    Y = yuv[:, :, 0]
    U = yuv[:, :, 1]
    V = yuv[:, :, 2]

    # Average U/V over each horizontal pixel pair
    Up = ((U[:, 0::2].astype(np.int16) + U[:, 1::2].astype(np.int16)) // 2).astype(np.uint8)
    Vp = ((V[:, 0::2].astype(np.int16) + V[:, 1::2].astype(np.int16)) // 2).astype(np.uint8)

    raw = np.zeros((h, w * 2), dtype=np.uint8)
    raw[:, 0::4] = Up          # U
    raw[:, 1::4] = Y[:, 0::2]  # Y0
    raw[:, 2::4] = Vp          # V
    raw[:, 3::4] = Y[:, 1::2]  # Y1

    return raw.tobytes(), w, h


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]}  <input.jpg>  [output.yuv422]")
        sys.exit(1)

    inp = sys.argv[1]
    if len(sys.argv) >= 3:
        out = sys.argv[2]
    else:
        stem = os.path.splitext(inp)[0]
        out = stem + ".yuv422"

    bgr = cv2.imread(inp)
    if bgr is None:
        print(f"ERROR: cannot read {inp}")
        sys.exit(1)

    raw, w, h = bgr_to_uyvy(bgr)

    with open(out, "wb") as f:
        f.write(raw)

    print(f"Converted: {inp}")
    print(f"  Output:  {out}  ({len(raw)} bytes)")
    print(f"  Size:    {w} x {h}")
    print()
    print(f"Run the C test with:")
    print(f"  ./test_obstacle_detection  {out}  {w}  {h}  results/")
    print()
    print(f"Run multi-frame (same image repeated 10x):")
    print(f"  ./test_multiframe  {out}  {w}  {h}  10")


if __name__ == "__main__":
    main()
