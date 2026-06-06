"""Generate a print-ready ArUco marker PDF at EXACT physical size.

    python make_marker_pdf.py        # -> marker.pdf  (black square = 50mm)

PNG prints get scaled ("fit to page") and ruin the scale. This PDF places the
marker at an exact 50mm and adds a 50mm reference line so you can ruler-check
the printout. Print at 100% / "Actual size".
"""
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

MARKER_MM = 50.0
DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
PNG = "/tmp/_aruco_marker.png"
QUIET = 0.25   # white border = 25% of marker on each side (ArUco quiet zone)


def main():
    d = cv2.aruco.getPredefinedDictionary(DICT)
    size = 1000
    pad = int(size * QUIET)
    marker = cv2.aruco.generateImageMarker(d, MARKER_ID, size)
    padded = np.full((size + 2 * pad, size + 2 * pad), 255, np.uint8)
    padded[pad:pad + size, pad:pad + size] = marker
    cv2.imwrite(PNG, padded)

    c = canvas.Canvas("marker.pdf", pagesize=A4)
    W, H = A4
    img_mm = MARKER_MM * (size + 2 * pad) / size   # so the black square stays 50mm
    x = (W - img_mm * mm) / 2
    y = H - 55 * mm - img_mm * mm
    c.drawImage(PNG, x, y, img_mm * mm, img_mm * mm)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y + img_mm * mm + 8 * mm,
                 f"ArUco DICT_4X4_50  id {MARKER_ID}   —   black square = {MARKER_MM:.0f} mm")
    c.setFont("Helvetica", 10)
    c.drawString(x, y - 9 * mm,
                 "Print at 100% / Actual size (NOT fit-to-page).")

    # 50mm verification ruler with end ticks
    ly = y - 22 * mm
    c.line(x, ly, x + 50 * mm, ly)
    for tx in (x, x + 50 * mm):
        c.line(tx, ly - 2 * mm, tx, ly + 2 * mm)
    c.setFont("Helvetica", 9)
    c.drawString(x, ly - 7 * mm, "This line should measure exactly 50 mm with a ruler.")
    c.showPage()
    c.save()
    print("wrote marker.pdf")


if __name__ == "__main__":
    main()
