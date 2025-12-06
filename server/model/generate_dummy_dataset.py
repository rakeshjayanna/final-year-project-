"""
Generate a tiny synthetic dataset at ../dataset for pipeline testing.

Now creates four classes to support pesticide and disease detection (and both):
- organic: green-ish background, no clear artifacts
- pesticide: yellow/orange hues
- disease: darker/brownish hues with spot-like artifacts
- both: blended cues from pesticide and disease

Usage:
    python server/model/generate_dummy_dataset.py --count 40 --size 96 96
"""
from __future__ import annotations

from pathlib import Path
import argparse
import random
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'datasets'


def make_image(color_bg, color_fg, size=(96, 96)):
    w, h = size
    img = Image.new('RGB', (w, h), color_bg)
    draw = ImageDraw.Draw(img)
    # Draw simple distinct shapes
    for _ in range(3):
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(w//2, w-1), random.randint(h//2, h-1)
        draw.rectangle([x1, y1, x2, y2], outline=color_fg, width=5)
    return img


def add_spots(img: Image.Image, n_spots: int = 12, color=(80, 40, 20)) -> Image.Image:
    """Overlay round-ish disease-like spots."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for _ in range(n_spots):
        r = random.randint(max(2, w//40), max(3, w//18))
        cx, cy = random.randint(0, w-1), random.randint(0, h-1)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, outline=color, width=random.randint(1, 3))
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def gen_class(cls_dir: Path, color_bg, color_fg, n: int, size, add_disease_spots: bool = False):
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = make_image(color_bg, color_fg, size=size)
        if add_disease_spots:
            # Add brownish spots to mimic disease patches
            img = add_spots(img, n_spots=random.randint(8, 18), color=(90, 55, 25))
        img.save(cls_dir / f"img_{i:03d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=1000, help='Images per class')
    ap.add_argument('--size', type=int, nargs=2, default=[224, 224], help='Image size WxH')
    args = ap.parse_args()

    # Target the pesticide dataset structure
    organic_dir = DATASET / 'pesticide' / 'organic'
    pesticide_dir = DATASET / 'pesticide' / 'pesticide'

    print(f"Generating {args.count} images per class...")
    
    # Organic: Pure Green background, Dark Green shapes
    print(f"Generating organic images in {organic_dir}...")
    gen_class(organic_dir, (0, 255, 0), (0, 100, 0), args.count, tuple(args.size), add_disease_spots=False)
    
    # Pesticide: Pure Red background, Dark Red shapes (Very distinct)
    print(f"Generating pesticide images in {pesticide_dir}...")
    gen_class(pesticide_dir, (255, 0, 0), (100, 0, 0), args.count, tuple(args.size), add_disease_spots=False)

    print(f"Synthetic dataset created under: {DATASET / 'pesticide'}")


if __name__ == '__main__':
    main()
