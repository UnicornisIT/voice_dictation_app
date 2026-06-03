#!/usr/bin/env python
"""Convert PNG to ICO for PyInstaller."""
from pathlib import Path

from PIL import Image


icon_png = Path(__file__).parent / "app_logo.png"
icon_ico = Path(__file__).parent / "app_icon.ico"

if icon_png.exists():
    try:
        img = Image.open(str(icon_png))
        if img.size[0] > 256 or img.size[1] > 256:
            img.thumbnail((256, 256), Image.Resampling.LANCZOS)
        img.save(str(icon_ico))
        print(f"OK: icon converted: {icon_ico}")
    except Exception as exc:
        print(f"ERROR: icon conversion failed: {exc}")
        raise SystemExit(1) from exc
else:
    print(f"ERROR: file not found: {icon_png}")
    raise SystemExit(1)
