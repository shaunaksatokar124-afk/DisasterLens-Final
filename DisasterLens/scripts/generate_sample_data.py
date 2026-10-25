"""
Generate a synthetic pre/post-disaster sample pair for demos and smoke tests.

    python scripts/generate_sample_data.py            # writes sample_data/*.png

The scene is procedurally generated (terrain, roads, buildings).  In the
"post" image, buildings near a handful of impact epicentres are progressively
damaged (colour shift -> debris -> rubble).  The images are entirely synthetic
and depict no real place or event.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIZE = 512


def _terrain(rng: np.random.Generator) -> np.ndarray:
    """Soft green/tan ground texture."""
    base = np.zeros((SIZE, SIZE, 3), np.float32)
    base[:] = (96, 112, 78)
    for scale, amp in ((64, 26), (24, 14), (8, 7)):
        n = rng.normal(0, 1, (SIZE // scale + 2, SIZE // scale + 2)).astype(np.float32)
        n = cv2.resize(n, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
        base += n[..., None] * amp * np.array([0.9, 1.0, 0.7], np.float32)
    # dry patches
    patch = cv2.resize(rng.normal(0, 1, (6, 6)).astype(np.float32), (SIZE, SIZE),
                       interpolation=cv2.INTER_CUBIC)
    base += np.clip(patch, 0, None)[..., None] * np.array([28, 18, -6], np.float32)
    return np.clip(base, 0, 255)


def build_scene(seed: int = 7):
    rng = np.random.default_rng(seed)
    ground = _terrain(rng)
    pre = ground.copy()

    # ── Roads (grid, slightly jittered) ───────────────────────────────────────
    road_mask = np.zeros((SIZE, SIZE), np.uint8)
    xs = [70, 190, 320, 440]
    ys = [60, 175, 300, 420]
    for x in xs:
        cv2.line(road_mask, (x, 0), (x + int(rng.integers(-8, 8)), SIZE), 255, 12)
    for y in ys:
        cv2.line(road_mask, (0, y), (SIZE, y + int(rng.integers(-8, 8))), 255, 12)
    road_col = np.array([128, 128, 130], np.float32)
    road_noise = rng.normal(0, 5, (SIZE, SIZE, 1)).astype(np.float32)
    pre[road_mask > 0] = (road_col + road_noise)[road_mask > 0]

    # ── Buildings inside the city blocks ──────────────────────────────────────
    buildings = []  # (x0, y0, x1, y1, roof_colour)
    roofs = [(176, 82, 68), (150, 150, 155), (196, 170, 140), (120, 96, 88), (205, 205, 210)]
    xb = [0] + xs + [SIZE]
    yb = [0] + ys + [SIZE]
    for i in range(len(xb) - 1):
        for j in range(len(yb) - 1):
            bx0, bx1 = xb[i] + 16, xb[i + 1] - 16
            by0, by1 = yb[j] + 16, yb[j + 1] - 16
            if bx1 - bx0 < 30 or by1 - by0 < 30:
                continue
            gx = max(1, (bx1 - bx0) // 44)
            gy = max(1, (by1 - by0) // 36)
            cw, ch = (bx1 - bx0) // gx, (by1 - by0) // gy
            for a in range(gx):
                for b in range(gy):
                    if rng.random() < 0.18:
                        continue
                    w = int(cw * rng.uniform(0.55, 0.85))
                    h = int(ch * rng.uniform(0.55, 0.85))
                    x0 = bx0 + a * cw + int(rng.integers(2, max(3, cw - w)))
                    y0 = by0 + b * ch + int(rng.integers(2, max(3, ch - h)))
                    roof = roofs[int(rng.integers(len(roofs)))]
                    buildings.append((x0, y0, x0 + w, y0 + h, roof))

    for x0, y0, x1, y1, roof in buildings:
        # drop shadow
        cv2.rectangle(pre, (x0 + 3, y0 + 3), (x1 + 3, y1 + 3), (35, 35, 35), -1)
    for x0, y0, x1, y1, roof in buildings:
        col = np.array(roof, np.float32) + rng.normal(0, 6, 3)
        cv2.rectangle(pre, (x0, y0), (x1, y1), tuple(float(c) for c in np.clip(col, 0, 255)), -1)
        cv2.rectangle(pre, (x0, y0), (x1, y1), (60, 60, 60), 1)
        cv2.line(pre, (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2), (90, 90, 90), 1)

    pre = np.clip(pre + rng.normal(0, 2.5, pre.shape), 0, 255).astype(np.uint8)

    # ── Post-disaster: damage fields around epicentres ────────────────────────
    post = ground.copy()
    post[road_mask > 0] = (road_col + road_noise)[road_mask > 0]
    yy, xx = np.mgrid[:SIZE, :SIZE]
    field = np.zeros((SIZE, SIZE), np.float32)
    for cx, cy, rad in ((150, 140, 110), (360, 330, 130), (400, 90, 70)):
        field += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (rad * 0.55) ** 2))
    field = np.clip(field, 0, 1)

    rubble_base = np.array([112, 98, 86], np.float32)
    for x0, y0, x1, y1, roof in buildings:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        sev = float(field[cy, cx]) + float(rng.normal(0, 0.08))
        w, h = x1 - x0, y1 - y0
        if sev < 0.28:                             # intact
            cv2.rectangle(post, (x0 + 3, y0 + 3), (x1 + 3, y1 + 3), (35, 35, 35), -1)
            cv2.rectangle(post, (x0, y0), (x1, y1), tuple(float(c) for c in roof), -1)
            cv2.rectangle(post, (x0, y0), (x1, y1), (60, 60, 60), 1)
            cv2.line(post, (x0, cy), (x1, cy), (90, 90, 90), 1)
        elif sev < 0.55:                           # minor: tarp / partial roof loss
            cv2.rectangle(post, (x0 + 3, y0 + 3), (x1 + 3, y1 + 3), (35, 35, 35), -1)
            cv2.rectangle(post, (x0, y0), (x1, y1), tuple(float(c) for c in roof), -1)
            cv2.rectangle(post, (x0, y0), (x0 + w // 2, y0 + h // 3), (60, 90, 170), -1)
            cv2.rectangle(post, (x0, y0), (x1, y1), (60, 60, 60), 1)
        elif sev < 0.8:                            # major: half collapsed
            cv2.rectangle(post, (x0, y0), (x1, y1), tuple(float(c) * 0.7 for c in roof), -1)
            sub = post[y0:y1, x0:x1]
            debris = rubble_base + rng.normal(0, 28, (sub.shape[0], sub.shape[1], 1))
            half = np.zeros(sub.shape[:2], bool)
            half[:, w // 3:] = True
            sub[half] = np.clip(debris + rng.normal(0, 14, sub.shape), 0, 255)[half]
        else:                                      # destroyed: rubble
            sub = post[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
            noise = rubble_base + rng.normal(0, 38, sub.shape)
            sub[:] = np.clip(noise, 0, 255)

    # scattered debris on roads/ground near the epicentres
    debris_pts = rng.random((SIZE, SIZE)) < (0.03 * field)
    post[debris_pts] = np.array([140, 120, 100], np.float32)
    post = cv2.GaussianBlur(post, (3, 3), 0)
    post = np.clip(post * 0.96 + rng.normal(0, 3, post.shape), 0, 255).astype(np.uint8)

    return pre, post


def main(out_dir: Path | None = None) -> None:
    from PIL import Image

    out = Path(out_dir) if out_dir else ROOT / "sample_data"
    out.mkdir(parents=True, exist_ok=True)
    pre, post = build_scene()
    Image.fromarray(pre).save(out / "pre_disaster_sample.png")
    Image.fromarray(post).save(out / "post_disaster_sample.png")
    print(f"Wrote {out / 'pre_disaster_sample.png'} and {out / 'post_disaster_sample.png'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
