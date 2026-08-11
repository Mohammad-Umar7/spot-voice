"""Regenerate the placeholder camera frame used by mock mode.

The bundled ``spot_voice/assets/test_scene.jpg`` is a synthetic facility scene,
not a real photograph -- it exists so ``capture_image`` returns something
plausible at a desk. Drop a real 640x480 JPEG in its place if you want mock mode
to describe your actual site.

Usage::

    python scripts/make_test_scene.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 640, 480
OUT = Path(__file__).resolve().parent.parent / "spot_voice" / "assets" / "test_scene.jpg"


def build() -> Image.Image:
    """Draw a simple industrial corridor: floor, wall, pipes, panel, signage."""
    image = Image.new("RGB", (WIDTH, HEIGHT), (108, 112, 118))
    draw = ImageDraw.Draw(image)

    # Back wall and floor, with a vanishing-point corridor.
    draw.polygon([(0, 300), (WIDTH, 300), (WIDTH, HEIGHT), (0, HEIGHT)], fill=(74, 76, 80))
    draw.polygon([(200, 150), (440, 150), (440, 330), (200, 330)], fill=(126, 130, 136))
    for x in range(0, WIDTH + 1, 80):
        draw.line([(x, HEIGHT), (320, 300)], fill=(88, 90, 94), width=2)
    draw.line([(0, 300), (WIDTH, 300)], fill=(60, 62, 66), width=3)

    # Overhead pipe run.
    for index, y in enumerate((60, 92, 124)):
        shade = 150 - index * 18
        draw.rectangle([0, y, WIDTH, y + 20], fill=(shade, shade - 6, shade - 14))
        for x in range(40, WIDTH, 160):
            draw.rectangle([x, y - 4, x + 14, y + 24], fill=(70, 72, 76))

    # Control panel on the left wall.
    draw.rectangle([40, 200, 190, 330], fill=(58, 66, 74), outline=(30, 34, 38), width=3)
    for row in range(2):
        for column in range(3):
            cx = 70 + column * 42
            cy = 232 + row * 46
            draw.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill=(210, 214, 218))
            draw.line([cx, cy, cx + 9, cy - 9], fill=(180, 40, 40), width=2)
    draw.rectangle([48, 300, 182, 322], fill=(24, 40, 30))
    for index in range(6):
        height = 6 + (index * 3) % 14
        draw.rectangle(
            [56 + index * 20, 318 - height, 68 + index * 20, 318], fill=(90, 220, 130)
        )

    # Electrical cabinet on the right.
    draw.rectangle([460, 180, 600, 340], fill=(96, 100, 104), outline=(40, 42, 46), width=3)
    draw.line([530, 180, 530, 340], fill=(40, 42, 46), width=2)
    draw.ellipse([548, 250, 560, 262], fill=(230, 190, 60))

    # Yellow hazard sign above the cabinet.
    draw.polygon([(530, 118), (566, 172), (494, 172)], fill=(226, 190, 40))
    draw.polygon([(530, 132), (556, 166), (504, 166)], outline=(30, 30, 30))
    draw.line([530, 140, 530, 156], fill=(30, 30, 30), width=4)
    draw.ellipse([527, 160, 533, 166], fill=(30, 30, 30))

    # Floor markings and a small spill to give the model something to notice.
    draw.rectangle([220, 380, 420, 396], fill=(206, 176, 46))
    draw.ellipse([250, 410, 372, 452], fill=(52, 58, 70))
    draw.ellipse([262, 418, 342, 442], fill=(64, 72, 88))

    # A pallet in the corridor.
    draw.rectangle([420, 348, 560, 386], fill=(140, 112, 74), outline=(96, 76, 50), width=2)
    for x in range(428, 556, 22):
        draw.rectangle([x, 348, x + 12, 386], fill=(158, 128, 86))

    return image


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build().save(OUT, "JPEG", quality=88)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
