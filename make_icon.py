"""Generate the Moodito app icon (moodito.icns).

Draws a friendly smiley on a warm rounded-square background and packages it
into a multi-resolution macOS .icns file. Re-run after changing the design:

    python make_icon.py
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

# Supersample everything, then downscale with LANCZOS for crisp anti-aliasing.
SCALE = 4
BASE = 1024
R = BASE * SCALE

# Palette.
FACE = (255, 255, 255)        # white lines and border


def render_master() -> Image.Image:
    """Render the icon at full supersampled resolution."""
    icon = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)

    # Scalloped ("ruffled") ring border on a transparent background: the radius
    # oscillates with the angle to create evenly spaced wavy bumps.
    border = round(0.04 * R)
    cx0, cy0 = 0.5 * R, 0.5 * R
    base_r = 0.40 * R          # mean radius of the ring
    amplitude = 0.025 * R      # depth of each bump
    scallops = 16              # number of bumps around the edge
    samples = 1440             # smoothness of the curve

    points = []
    for i in range(samples + 1):
        t = 2 * math.pi * i / samples
        r = base_r + amplitude * math.cos(scallops * t)
        points.append((cx0 + r * math.cos(t), cy0 + r * math.sin(t)))
    draw.line(points, fill=FACE, width=border, joint="curve")

    # Eyes.
    eye_ry = 0.072 * R
    eye_rx = 0.044 * R
    eye_y = 0.42 * R
    for cx in (0.385 * R, 0.615 * R):
        draw.ellipse(
            (cx - eye_rx, eye_y - eye_ry, cx + eye_rx, eye_y + eye_ry),
            fill=FACE,
        )

    # Smile: lower arc of an ellipse with rounded end caps.
    cx, cy = 0.50 * R, 0.49 * R
    ax, ay = 0.20 * R, 0.19 * R
    width = round(0.05 * R)
    bbox = (cx - ax, cy - ay, cx + ax, cy + ay)
    draw.arc(bbox, start=20, end=160, fill=FACE, width=width)

    cap = width / 2
    for ang in (20, 160):
        ex = cx + ax * math.cos(math.radians(ang))
        ey = cy + ay * math.sin(math.radians(ang))
        draw.ellipse((ex - cap, ey - cap, ex + cap, ey + cap), fill=FACE)

    return icon.resize((BASE, BASE), Image.LANCZOS)


def build_icns() -> Path:
    master = render_master()
    ASSETS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "moodito.iconset"
        iconset.mkdir()
        # (point size, @ scale) -> filename pairs required by iconutil.
        specs = [
            (16, 1), (16, 2),
            (32, 1), (32, 2),
            (128, 1), (128, 2),
            (256, 1), (256, 2),
            (512, 1), (512, 2),
        ]
        for size, factor in specs:
            px = size * factor
            suffix = f"@{factor}x" if factor == 2 else ""
            img = master.resize((px, px), Image.LANCZOS)
            img.save(iconset / f"icon_{size}x{size}{suffix}.png")

        out = ASSETS / "moodito.icns"
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
            check=True,
        )
        # Keep a PNG preview alongside the .icns.
        master.save(ASSETS / "moodito.png")
        return out


if __name__ == "__main__":
    path = build_icns()
    print(f"Wrote {path}")
