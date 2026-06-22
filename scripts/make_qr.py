#!/usr/bin/env python3
"""Regenerate the Buy Me a Coffee QR code (bmc_qr.png).

Dev-only helper (not bundled into the app). Run after changing the URL:

    python scripts/make_qr.py
"""
from __future__ import annotations

from pathlib import Path

import qrcode

BMC_URL = "https://buymeacoffee.com/georgiosnikitas"
OUTPUT = Path(__file__).resolve().parent.parent / "assets" / "bmc_qr.png"


def main() -> None:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2,
    )
    qr.add_data(BMC_URL)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(OUTPUT)
    print(f"Wrote {OUTPUT} -> {BMC_URL}")


if __name__ == "__main__":
    main()
