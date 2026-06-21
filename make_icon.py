"""Generate the Moodito app icon (moodito.icns).

Draws a friendly smiley on a warm rounded-square background and packages it
into a multi-resolution macOS .icns file. Re-run after changing the design:

    python make_icon.py
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent

# Supersample everything, then downscale with LANCZOS for crisp anti-aliasing.
SCALE = 4
BASE = 1024
R = BASE * SCALE

# Palette.
GRAD_TOP = (255, 209, 102)   # warm yellow
GRAD_BOTTOM = (255, 138, 76)  # orange
FACE = (74, 45, 8)            # deep warm brown


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _vertical_gradient(size: int) -> Image.Image:
    """Build a top-to-bottom gradient image."""
    column = Image.new("RGB", (1, size))
    for y in range(size):
        column.putpixel((0, y), _lerp(GRAD_TOP, GRAD_BOTTOM, y / (size - 1)))
    return column.resize((size, size))


def render_master() -> Image.Image:
    """Render the icon at full supersampled resolution."""
    # Rounded-square background matching macOS icon proportions.
    pad = round(0.0977 * R)            # content inset
    radius = round(0.180 * R)          # corner radius
    rect = (pad, pad, R - pad, R - pad)

    mask = Image.new("L", (R, R), 0)
    ImageDraw.Draw(mask).rounded_rectangle(rect, radius=radius, fill=255)

    icon = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    icon.paste(_vertical_gradient(R), (0, 0), mask)

    draw = ImageDraw.Draw(icon)

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

        out = HERE / "moodito.icns"
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
            check=True,
        )
        # Keep a PNG preview alongside the .icns.
        master.save(HERE / "moodito.png")
        return out


if __name__ == "__main__":
    path = build_icns()
    print(f"Wrote {path}")
