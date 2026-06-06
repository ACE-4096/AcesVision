"""Generate a printable ArUco scale marker.

    python make_marker.py            # 50mm marker -> marker.png

Print it at 100% / "actual size" (no fit-to-page) so the black square measures
exactly MARKER_MM across. Place it flat in the camera's view next to whatever
you're measuring — everything is scaled against it.
"""
import cv2
import numpy as np

MARKER_MM = 50.0
DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
DPI = 300


def main():
    px = int(MARKER_MM / 25.4 * DPI)        # marker size in pixels at DPI
    d = cv2.aruco.getPredefinedDictionary(DICT)
    marker = cv2.aruco.generateImageMarker(d, MARKER_ID, px)

    pad = int(px * 0.25)
    canvas = np.full((px + 2 * pad, px + 2 * pad), 255, np.uint8)
    canvas[pad:pad + px, pad:pad + px] = marker
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, f"Print actual size — this square = {MARKER_MM:.0f} mm",
                (pad, pad - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite("marker.png", canvas)
    print(f"wrote marker.png ({MARKER_MM:.0f}mm, DICT_4X4_50 id {MARKER_ID}, {DPI}dpi)")
    print("Print at 100% scale, then measure the black square to confirm it's "
          f"{MARKER_MM:.0f}mm.")


if __name__ == "__main__":
    main()
