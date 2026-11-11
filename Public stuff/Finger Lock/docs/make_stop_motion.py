"""Make docs/stop_motion.mp4: a silent, paper-cutout stop-motion movie that
shows how my finger-counting CNN works, using the REAL network.

    python docs/make_stop_motion.py            # video + keyframes
    python docs/make_stop_motion.py --stills   # keyframes only (fast check)

Every number in the movie comes from the project files:
  - weights:      models/finger_cnn.npz (run with finger_net.py, plain numpy)
  - the photo:    one 96x96 tile of docs/sample_photos.png (row k = k fingers),
                  resized to 64x64 RGB, 0..1, channels first, like prepare()
  - lock rules:   lock_policy.py (70%, 1 s, 10 s lockout, 20 s timeout)
  - training:     train.py (epochs), README.md (photo count, accuracy),
                  results/training.png and results/confusion.png

Needs numpy, Pillow and opencv-python (already used by the project) plus an
ffmpeg: imageio-ffmpeg's if it is installed, otherwise ffmpeg on the PATH.
"""

import argparse
import math
import re
import shutil
import subprocess
import sys
import zlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))
import lock_policy                                    # noqa: E402
from finger_net import FingerNet, conv3x3, maxpool2, relu, softmax  # noqa: E402

W, H, FPS = 1280, 720, 8
M = 8                       # transparent margin around every paper piece
PHOTO_ROW, PHOTO_COL = 4, 0  # the photo we follow: 4 fingers (row 4 = 4 fingers)
FILTER = 4                   # conv1 filter #4 finds finger edges


def sec(s):
    return int(round(s * FPS))


# ======================================================== real data ====
def load_real():
    net = FingerNet(ROOT / "models" / "finger_cnn.npz")
    sheet = np.array(Image.open(DOCS / "sample_photos.png").convert("RGB"))
    assert sheet.shape == (6 * 96, 8 * 96, 3), sheet.shape

    def tile(r, c):
        return sheet[r * 96:(r + 1) * 96, c * 96:(c + 1) * 96]

    def to_input(t):  # same as FingerNet.prepare after the crop
        t = cv2.resize(t, (64, 64), interpolation=cv2.INTER_AREA)
        return t.transpose(2, 0, 1).astype(np.float32) / 255.0

    R = {"net": net, "tile": tile, "to_input": to_input}
    R["photo96"] = tile(PHOTO_ROW, PHOTO_COL)
    x = to_input(R["photo96"])
    R["x"] = x
    (w1, b1), (w2, b2), (w3, b3) = net.layers
    R["w1"], R["b1"] = w1, b1
    c1 = conv3x3(x, w1, b1); r1 = relu(c1); p1 = maxpool2(r1)
    c2 = conv3x3(p1, w2, b2); p2 = maxpool2(relu(c2))
    c3 = conv3x3(p2, w3, b3); p3 = maxpool2(relu(c3))
    stages = net.features(x)
    for a, b in zip(stages[1:], (p1, p2, p3)):
        assert np.allclose(a, b)
    R.update(c1=c1, r1=r1, p1=p1, p2=p2, p3=p3)
    flat = p3.reshape(-1)
    scores = net.fc_w @ flat + net.fc_b
    probs = softmax(scores)
    assert np.allclose(probs, net.predict(x))
    R.update(flat=flat, scores=scores, probs=probs)
    assert int(probs.argmax()) == PHOTO_ROW, "chosen photo is misread"

    # weight counts straight from the .npz
    p = np.load(ROOT / "models" / "finger_cnn.npz")
    R["n_layer"] = [p[f"{n}.weight"].size + p[f"{n}.bias"].size
                    for n in ("conv1", "conv2", "conv3", "fc")]
    R["n_total"] = sum(p[k].size for k in p.files)
    assert R["n_total"] == sum(R["n_layer"])
    R["n_filters"] = [p[f"conv{i}.weight"].shape[0] for i in (1, 2, 3)]
    R["fc_shape"] = p["fc.weight"].shape

    # best-read sample photo for each count (for the lock scene)
    R["best"] = {}
    for k in range(6):
        best = max(range(8), key=lambda c: net.predict(to_input(tile(k, c)))[k])
        R["best"][k] = (tile(k, best), net.predict(to_input(tile(k, best))))

    # training facts from the project files
    train_src = (ROOT / "train.py").read_text()
    R["epochs"] = int(re.search(r"^EPOCHS\s*=\s*(\d+)", train_src, re.M).group(1))
    readme = (ROOT / "README.md").read_text()
    m = re.search(r"Accuracy:\*\*\s*([\d.]+)%\s*on held-out photos\s*\(([\d,]+) photos", readme)
    R["accuracy"], R["n_photos"] = m.group(1), m.group(2)
    R["code"] = lock_policy.load_code()
    R["policy"] = dict(conf=lock_policy.CONFIDENCE, hold=lock_policy.HOLD_TIME,
                       lockout=lock_policy.LOCKOUT_TIME, timeout=lock_policy.ENTRY_TIMEOUT,
                       max_wrong=lock_policy.MAX_TRIES)
    return R


def add_picks(R):
    """Which patches / positions the close-ups show (chosen from the real maps)."""
    R["relu_patch"] = pick_patch(R["c1"][FILTER], 6, "mixed")
    c1 = R["c1"][FILTER]
    inner = c1[4:60, 4:60]
    R["slow_pos"] = [(12, 8),
                     tuple(int(v) + 4 for v in np.unravel_index(inner.argmax(), inner.shape)),
                     tuple(int(v) + 4 for v in np.unravel_index(inner.argmin(), inner.shape))]
    R["pool_patch"] = tuple(v - v % 2 for v in pick_patch(R["r1"][FILTER], 4, "pool"))
    return R


# ======================================================= paper look ====
PAPER_RGB = (233, 224, 205)
CREAM = (250, 246, 236)
INK = (44, 40, 38)
SOFT = (110, 100, 92)
RED = (205, 62, 50)
BLUE = (52, 98, 170)
YELLOW = (238, 190, 70)
GREEN = (70, 150, 90)
BROWN = (170, 120, 78)
DARK = (58, 56, 60)

_rng = np.random.default_rng(193)


def _make_paper():
    n = _rng.normal(0, 1, (H // 4 + 2, W // 4 + 2))
    n = cv2.resize(cv2.GaussianBlur(n, (0, 0), 2), (W, H))
    fine = _rng.normal(0, 1, (H, W))
    tex = 1 + 0.025 * n + 0.018 * fine
    img = np.array(PAPER_RGB, np.float32)[None, None] * tex[..., None]
    # a few long paper fibres
    fib = np.zeros((H, W), np.float32)
    for _ in range(260):
        x0, y0 = _rng.integers(0, W), _rng.integers(0, H)
        a = _rng.uniform(0, math.pi)
        ln = _rng.integers(8, 30)
        cv2.line(fib, (int(x0), int(y0)),
                 (int(x0 + ln * math.cos(a)), int(y0 + ln * math.sin(a))), 1.0, 1)
    img *= (1 - 0.05 * cv2.GaussianBlur(fib, (0, 0), 0.6))[..., None]
    yy, xx = np.mgrid[0:H, 0:W]
    vign = 1 - 0.10 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2
    img *= vign[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


PAPER = _make_paper()
_g = (_rng.normal(0, 1, (900, 1400)) * 5 + 248).clip(225, 255).astype(np.uint8)
GRAIN = Image.fromarray(_g).convert("RGB")


def rnd(*key):
    return (zlib.crc32(repr(key).encode()) % 1000003) / 1000003


@lru_cache(None)
def font(size, kind="sans"):
    names = {"sans": ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"],
             "bold": ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"],
             "mono": ["DejaVuSansMono.ttf", "Menlo.ttc", "Courier New.ttf"]}[kind]
    dirs = ["/usr/share/fonts/truetype/dejavu", "/Library/Fonts",
            "/System/Library/Fonts", "/System/Library/Fonts/Supplemental",
            "C:/Windows/Fonts"]
    try:
        import matplotlib
        dirs.append(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf"))
    except ImportError:
        pass
    for d in dirs:
        for n in names:
            if (Path(d) / n).exists():
                return ImageFont.truetype(str(Path(d) / n), size)
    return ImageFont.load_default(size)


def text_size(s, size, kind="sans"):
    l, t, r, b = font(size, kind).getbbox(s)
    return r - l, b - t


def fmt(v, d=2):
    s = f"{v:+.{d}f}" if v != 0 else f"{0:.{d}f}"
    return s.replace("-", "\u2212")


def num(v):
    return f"{v:,}"


# ------------------------------------------------------------ pieces ----
def card(w, h, fill=CREAM, key="", cut=2.0, outline=None, width=2):
    """A hand-cut paper rectangle (RGBA, with margin M)."""
    im = Image.new("RGBA", (w + 2 * M, h + 2 * M), (0, 0, 0, 0))
    r = [(rnd(key, "c", i) * 2 - 1) * cut for i in range(8)]
    pts = [(M + r[0], M + r[1]), (M + w + r[2], M + r[3]),
           (M + w + r[4], M + h + r[5]), (M + r[6], M + h + r[7])]
    d = ImageDraw.Draw(im)
    d.polygon(pts, fill=tuple(fill) + (255,))
    gx = int(rnd(key, "gx") * (GRAIN.width - im.width - 1))
    gy = int(rnd(key, "gy") * (GRAIN.height - im.height - 1))
    rgb = ImageChops.multiply(im.convert("RGB"),
                              GRAIN.crop((gx, gy, gx + im.width, gy + im.height)))
    rgb.putalpha(im.getchannel("A"))
    if outline:
        ImageDraw.Draw(rgb).polygon(pts, outline=tuple(outline) + (255,), width=width)
    return rgb


def write(im, x, y, s, size=24, color=INK, kind="sans", anchor="la"):
    """Text on a piece (coordinates inside the piece, margin added)."""
    ImageDraw.Draw(im).text((x + M, y + M), s, font=font(size, kind),
                            fill=tuple(color) + (255,), anchor=anchor)


def photo(rgb, px, key, border=6):
    """A printed photo: white border + picture (nearest-neighbour so pixels show)."""
    img = Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
    img = img.resize((px, px), Image.NEAREST)
    c = card(px + 2 * border, px + 2 * border, (252, 251, 247), key, cut=1.5)
    c.paste(img, (M + border, M + border))
    return c


def cmap_div(a, vmax):
    t = np.clip(a / (vmax + 1e-9), -1, 1)[..., None]
    cream = np.array(CREAM, np.float32)
    neg = np.array(BLUE, np.float32)
    pos = np.array((214, 84, 44), np.float32)
    out = np.where(t < 0, cream + (neg - cream) * (-t), cream + (pos - cream) * t)
    return out.clip(0, 255).astype(np.uint8)


def cmap_pos(a, vmax):
    t = np.clip(a / (vmax + 1e-9), 0, 1)[..., None]
    cream = np.array(CREAM, np.float32)
    mid = np.array((235, 150, 60), np.float32)
    hi = np.array((120, 40, 30), np.float32)
    out = np.where(t < 0.5, cream + (mid - cream) * (t * 2), mid + (hi - mid) * (t * 2 - 1))
    return out.clip(0, 255).astype(np.uint8)


def robust_max(a):
    """Colour scale: 99.5th percentile of |a|, so one hot pixel at the image
    border doesn't wash the whole map out."""
    return float(np.percentile(np.abs(a), 99.5)) + 1e-9


def fmap(a, px, key, mode="div", border=4, vmax=None):
    vmax = vmax if vmax is not None else robust_max(a)
    rgb = cmap_div(a, vmax) if mode == "div" else cmap_pos(a, vmax)
    return photo(rgb, px, key, border)


def sign_fill(v, vmax):
    if v == 0:
        return (238, 234, 224)
    t = min(1.0, abs(v) / (vmax + 1e-9)) * 0.55 + 0.15
    base = np.array(CREAM, np.float32)
    tgt = np.array(BLUE if v < 0 else (214, 84, 44), np.float32)
    return tuple(int(c) for c in base + (tgt - base) * t)


def grid_piece(vals, cell, key, d=2, size=None, vmax=None, signed=True,
               outlines=None, bold_cells=(), title=None):
    """A paper grid of numbers. outlines: list of ((r0,c0,r1,c1), color)."""
    rows, cols = vals.shape
    top = 30 if title else 0
    c = card(cols * cell + 8, rows * cell + 8 + top, CREAM, key, cut=1.5)
    dr = ImageDraw.Draw(c)
    vmax = vmax if vmax is not None else (np.nanmax(np.abs(vals)) or 1)
    size = size or int(cell * 0.32)
    if title:
        write(c, (cols * cell + 8) // 2, 4, title, 20, INK, "bold", "ma")
    for i in range(rows):
        for j in range(cols):
            x0, y0 = M + 4 + j * cell, M + 4 + top + i * cell
            v = float(vals[i, j])
            if np.isnan(v):
                dr.rectangle((x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2), outline=(200, 192, 178), width=2)
                continue
            fill = sign_fill(v, vmax) if signed else (245, 240, 228)
            dr.rectangle((x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2), fill=fill)
            s = "0" if v == 0 else (fmt(v, d) if signed else f"{v:.{d}f}")
            dr.text((x0 + cell / 2, y0 + cell / 2), s,
                    font=font(size, "bold" if (i, j) in bold_cells else "mono"),
                    fill=INK, anchor="mm")
    for (r0, c0, r1, c1), col in outlines or []:
        dr.rectangle((M + 4 + c0 * cell, M + 4 + top + r0 * cell,
                      M + 4 + c1 * cell - 1, M + 4 + top + r1 * cell - 1),
                     outline=col, width=4)
    return c


# ------------------------------------------------------------- frame ----
def stepped(f, start, n=3, hold=1):
    """Stop-motion progress: None before start, then 1/n, 2/n ... 1."""
    if f < start:
        return None
    return min(n, (f - start) // hold + 1) / n


class Frame:
    def __init__(self, gf, lf):
        self.gf, self.f = gf, lf          # global / scene-local frame number
        self.im = PAPER.copy()
        self.d = ImageDraw.Draw(self.im)

    def jit(self, key, amp=1.4):
        a, b = rnd(key, self.gf, "x"), rnd(key, self.gf, "y")
        return int(round((a * 2 - 1) * amp)), int(round((b * 2 - 1) * amp))

    def put(self, piece, x, y, key, start=0, frm=(0, -40), n=3, shadow=True):
        """Place a piece (top-left of its content at x, y). It hops in from
        `frm` in n discrete steps starting at local frame `start`."""
        p = stepped(self.f, start, n)
        if p is None:
            return False
        dx, dy = self.jit(key)
        X = int(x - M + dx + frm[0] * (1 - p))
        Y = int(y - M + dy + frm[1] * (1 - p))
        if shadow:
            a = piece.getchannel("A").filter(ImageFilter.GaussianBlur(3))
            a = a.point(lambda v: int(v * 0.33))
            self.im.paste((70, 55, 40), (X + 3, Y + 4, X + 3 + a.width, Y + 4 + a.height), a)
        self.im.paste(piece, (X, Y), piece)
        return True

    def text(self, x, y, s, key, size=26, color=INK, kind="sans", anchor="la",
             start=0):
        if self.f < start:
            return
        dx, dy = self.jit(key, 1.0)
        self.d.text((x + dx, y + dy), s, font=font(size, kind), fill=color, anchor=anchor)

    def line(self, pts, key, color=INK, width=3, start=0):
        if self.f < start:
            return
        dx, dy = self.jit(key, 1.0)
        self.d.line([(a + dx, b + dy) for a, b in pts], fill=color, width=width)

    def arrow(self, x0, y0, x1, y1, key, color=INK, width=4, start=0, head=14):
        if self.f < start:
            return
        dx, dy = self.jit(key, 1.0)
        x0, y0, x1, y1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
        self.d.line([(x0, y0), (x1, y1)], fill=color, width=width)
        a = math.atan2(y1 - y0, x1 - x0)
        pts = [(x1, y1),
               (x1 - head * math.cos(a - 0.45), y1 - head * math.sin(a - 0.45)),
               (x1 - head * math.cos(a + 0.45), y1 - head * math.sin(a + 0.45))]
        self.d.polygon(pts, fill=color)

    def caption(self, s, key, start=0):
        if self.f < start:
            return
        tw, _ = text_size(s, 30, "bold")
        c = card(tw + 48, 52, (252, 250, 243), ("cap", key), cut=2.5)
        write(c, (tw + 48) // 2, 26, s, 30, INK, "bold", "mm")
        self.put(c, (W - tw - 48) // 2, 650, ("cap", key), start, frm=(0, 30))

    def tag(self, s, key):
        tw, _ = text_size(s, 24, "bold")
        c = card(tw + 32, 42, YELLOW, ("tag", key), cut=2)
        write(c, 16, 21, s, 24, INK, "bold", "lm")
        self.put(c, 28, 22, ("tag", key), 0, frm=(-60, 0))


def caption_seq(fr, seq, key):
    """seq: [(start_sec, text), ...]; shows the latest one that has started."""
    cur = None
    for i, (t, s) in enumerate(seq):
        if fr.f >= sec(t):
            cur = (i, t, s)
    if cur:
        i, t, s = cur
        fr.caption(s, (key, i), sec(t))


# ============================================================ scenes ====
def scene_lock(fr, R):
    f = fr.f
    if f < sec(3.5):  # title card
        c = card(860, 230, CREAM, "title", cut=3)
        write(c, 430, 70, "How my finger-counting", 52, INK, "bold", "mm")
        write(c, 430, 135, "neural network works", 52, RED, "bold", "mm")
        write(c, 430, 195, "Finger Lock  \u00b7  ME-193", 26, SOFT, "sans", "mm")
        fr.put(c, 210, 200, "title", 0)
        return
    g = f - sec(3.5)
    fr.tag("1 \u00b7 The lock", "s1")
    if f < sec(15):
        # the box
        fr.put(card(300, 40, (140, 96, 60), "lid"), 70, 300, "lid", sec(3.6))
        box = card(300, 220, BROWN, "box")
        fr.put(box, 70, 340, "box", sec(3.6))
        pressed = f >= sec(5.5)
        btn = Image.new("RGBA", (56 + 2 * M, 56 + 2 * M), (0, 0, 0, 0))
        ImageDraw.Draw(btn).ellipse((M, M, M + 56, M + 56),
                                    fill=((150, 40, 34) if pressed else RED) + (255,),
                                    outline=INK + (255,), width=3)
        fr.put(btn, 290, 470 + (3 if pressed else 0), "btn", sec(4.0))
        fr.text(318, 540, "button", "btnl", 18, CREAM, "bold", "ma", sec(4.0))
        if pressed:
            led = card(180, 90, DARK, "led", cut=1.5)
            dl = ImageDraw.Draw(led)
            for i in range(13):
                for j in range(8):
                    on = (j in (5, 6)) and (i % 4 != 0) and 1 <= i <= 11
                    col = (255, 200, 60) if on else (90, 86, 90)
                    cx, cy = M + 12 + i * 12.8, M + 10 + j * 10
                    dl.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=col)
            fr.put(led, 100, 370, "led", sec(5.5))
            fr.text(190, 470, "3 slots", "ledl", 20, CREAM, "bold", "ma", sec(5.5))
        # camera stand + paper + red square
        if f >= sec(7.5):
            fr.put(card(500, 0 + 150, (252, 251, 246), "sheet", cut=3), 440, 450, "sheet", sec(7.5))
            fr.put(card(18, 330, (150, 150, 155), "pole"), 470, 150, "pole", sec(7.6))
            fr.put(card(200, 16, (150, 150, 155), "arm"), 470, 150, "arm", sec(7.7))
            cam = card(100, 56, DARK, "cam")
            ImageDraw.Draw(cam).ellipse((M + 36, M + 30, M + 64, M + 58), fill=(30, 30, 34), outline=(150, 150, 160), width=3)
            fr.put(cam, 620, 120, "cam", sec(7.8))
            fr.line([(650, 190), (600, 450)], "fov1", SOFT, 2, sec(8.2))
            fr.line([(720, 190), (790, 450)], "fov2", SOFT, 2, sec(8.2))
            fr.put(photo(R["photo96"], 168, "hand1", border=0), 612, 462, "hand1",
                   sec(8.6), frm=(160, 0), n=4)
            sq = Image.new("RGBA", (180 + 2 * M, 180 + 2 * M), (0, 0, 0, 0))
            ImageDraw.Draw(sq).rectangle((M, M, M + 180, M + 180), outline=RED + (255,), width=4)
            fr.put(sq, 606, 456, "sq", sec(8.0), shadow=False)
        if f >= sec(10.5):
            fr.line([(786, 456), (900, 150)], "crop1", RED, 2)
            fr.line([(786, 636), (900, 446)], "crop2", RED, 2)
            if f < sec(12.5):
                fr.put(photo(R["photo96"], 288, "big96"), 900, 150, "big96", sec(10.5), frm=(40, 0))
                fr.text(1050, 460, "the red square", "bigl", 22, INK, "bold", "ma")
            else:
                img64 = (R["x"].transpose(1, 2, 0) * 255).round().astype(np.uint8)
                fr.put(photo(img64, 288, "big64"), 900, 150, "big64", sec(12.5), frm=(0, -30))
                fr.text(1050, 460, "64 \u00d7 64 pixels", "bigl2", 24, INK, "bold", "ma")
        caption_seq(fr, [(3.6, "Press the button"), (5.5, "The LEDs light up: 3 slots for 3 numbers"),
                         (7.5, "Hold up fingers under the camera"),
                         (10.5, "Only the inside of the red square is used"),
                         (12.5, "It gets shrunk to a 64 \u00d7 64 color image")], "s1")
        return
    # ---- the image is just numbers
    x = R["x"]
    img64 = (x.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    r0, c0 = 30, 24
    fr.put(photo(img64, 256, "n64"), 50, 170, "n64", sec(15))
    box = Image.new("RGBA", (32 + 2 * M, 32 + 2 * M), (0, 0, 0, 0))
    ImageDraw.Draw(box).rectangle((M, M, M + 32, M + 32), outline=RED + (255,), width=3)
    fr.put(box, 56 + c0 * 4, 176 + r0 * 4, "nbox", sec(15.6), shadow=False)
    patch = x[0, r0:r0 + 8, c0:c0 + 8]
    gp = grid_piece(patch, 44, "ngrid", d=2, size=14, signed=False, title="red channel, 8 \u00d7 8 corner")
    fr.put(gp, 352, 140, "ngrid", sec(15.8), frm=(-40, 0))
    fr.line([(56 + c0 * 4 + 32, 176 + r0 * 4), (352, 175)], "nl1", RED, 2, sec(15.8))
    fr.line([(56 + c0 * 4 + 32, 176 + r0 * 4 + 32), (352, 520)], "nl2", RED, 2, sec(15.8))
    names = [("R", (230, 70, 60)), ("G", (70, 160, 80)), ("B", (60, 100, 200))]
    for i, (nm, col) in enumerate(names):
        ch = x[i]
        rgb = (ch[..., None] * np.array(col)[None, None] + (1 - ch[..., None]) * 30).astype(np.uint8)
        fr.put(photo(rgb, 176, ("ch", i)), 840 + i * 48, 150 + i * 40, ("ch", i), sec(17 + i * 0.4))
        fr.text(840 + i * 48 + 12, 150 + i * 40 + 10, nm, ("chl", i), 26, CREAM, "bold", start=sec(17 + i * 0.4))
    n = x.size
    c, h, w = x.shape
    fr.text(1010, 450, f"{c} \u00d7 {h} \u00d7 {w} = {num(n)}", "eq", 34, INK, "bold", "ma", sec(18))
    fr.text(1010, 494, "numbers between 0 and 1", "eq2", 22, SOFT, "sans", "ma", sec(18))
    caption_seq(fr, [(15, "To the computer, the image is just numbers"),
                     (18, f"3 colors \u00d7 64 \u00d7 64 pixels = {num(n)} numbers")], "s1b")


def conv_at(R, i, j):
    """The 3x3x3 patch around output pixel (i, j) and the real multiply-add."""
    xp = np.pad(R["x"], ((0, 0), (1, 1), (1, 1)))
    patch = xp[:, i:i + 3, j:j + 3]
    w = R["w1"][FILTER]
    per = (patch * w).sum(axis=(1, 2))
    total = per.sum() + R["b1"][FILTER]
    assert abs(total - R["c1"][FILTER, i, j]) < 1e-4
    return patch, per, total


def scene_conv(fr, R):
    f = fr.f
    fr.tag("2 \u00b7 Convolution", "s2")
    x, w, b = R["x"], R["w1"][FILTER], R["b1"][FILTER]
    c1 = R["c1"][FILTER]
    vmax = robust_max(c1)
    img64 = (x.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    if f < sec(26):
        S = 5  # display pixels per image pixel
        fr.put(photo(img64, 64 * S, "cimg"), 30, 110, "cimg", 0)
        fr.text(190, 450, "input (64 \u00d7 64)", "cimgl", 22, INK, "bold", "ma")
        # the filter: real conv1 weights
        names = [("R", (200, 60, 50)), ("G", (60, 140, 70)), ("B", (50, 90, 180))]
        for k, (nm, col) in enumerate(names):
            gp = grid_piece(w[k], 40, ("fw", k), d=2, size=12, vmax=np.abs(w).max(),
                            title=f"weights {nm}")
            fr.put(gp, 390 + k * 150, 100, ("fw", k), sec(1 + 0.5 * k))
        fr.text(615, 274, f"+ bias {fmt(b, 3)}   (27 weights + 1 bias)", "fb", 20, INK, "bold", "ma", sec(2.5))
        # output map canvas
        out = np.full((64, 64), np.nan)
        slow = R["slow_pos"]
        t_slow = [5, 9, 13]
        cur = None
        for n_, (t0, pos) in enumerate(zip(t_slow, slow)):
            if f >= sec(t0):
                cur = (n_, t0, pos)
                if f >= sec(t0 + 2.5):
                    out[pos] = c1[pos]
            for (t1, p1) in zip(t_slow[:n_], slow[:n_]):
                out[p1] = c1[p1]
        sweep0 = sec(17)
        if f >= sweep0:
            rows = min(64, (f - sweep0 + 1) * 2)  # 2 rows per frame: ~4 s sweep
            out[:rows] = c1[:rows]
            cur = None
            wi, wj = min(63, rows - 1), (f * 23) % 62
            fr.put(_win(S), 30 + 6 + (wj - 1 + 1) * S - 0, 110 + 6 + (wi - 1) * S, "win", 0, shadow=False)
            fr.text(615, 470, f"{num(64 * 64)} positions, same 28 numbers each time", "sw", 22, INK, "bold", "ma")
        disp = np.where(np.isnan(out), 0, out)
        mrgb = cmap_div(disp, vmax)
        mrgb[np.isnan(out)] = (236, 230, 216)
        fr.put(photo(mrgb, 64 * S, "omap"), 910, 110, "omap", sec(3))
        fr.text(1070, 450, f"feature map (filter #{FILTER})", "omapl", 22, INK, "bold", "ma", sec(3))
        if cur is not None:
            n_, t0, (i, j) = cur
            lf = f - sec(t0)
            fr.put(_win(S), 30 + 6 + (j - 1) * S, 110 + 6 + (i - 1) * S, ("win", n_), 0, shadow=False)
            patch, per, total = conv_at(R, i, j)
            for k in range(3):
                gp = grid_piece(patch[k], 40, ("px", n_, k), d=2, size=12, signed=False,
                                title=f"pixels {'RGB'[k]}")
                fr.put(gp, 390 + k * 150, 350, ("px", n_, k), 0, frm=(0, 30))
                fr.text(458 + k * 150, 328, "\u00d7", ("x", k), 30, INK, "bold", "mm")
                if lf >= sec(1.2):
                    fr.text(458 + k * 150, 522, f"= {fmt(per[k], 3)}", ("ps", n_, k), 22, INK, "bold", "ma")
            if lf >= sec(2.2):
                s = (f"{fmt(per[0], 3)} {fmt(per[1], 3)} {fmt(per[2], 3)} {fmt(b, 3)} (bias)"
                     f" = {fmt(total, 3)}")
                c = card(text_size(s, 24, "bold")[0] + 30, 44, (255, 236, 170), ("tot", n_))
                write(c, 15, 22, s, 24, INK, "bold", "lm")
                fr.put(c, 615 - c.width // 2 + M, 560, ("tot", n_), 0)
                if lf >= sec(2.5):
                    px_, py_ = 910 + 6 + j * S + 2, 110 + 6 + i * S + 2
                    fr.arrow(615 + c.width // 2 - M, 572, px_ - 6, py_ + 6, ("ta", n_), RED, 3)
                    fr.d.ellipse((px_ - 9, py_ - 9, px_ + 9, py_ + 9), outline=RED, width=3)
        caption_seq(fr, [(0, "A filter is a tiny 3 \u00d7 3 grid of weights (one per color)"),
                         (5, "Multiply each pixel by its weight, then add it all up"),
                         (9, "Big answer = the patch matches the filter's pattern"),
                         (13, "Negative answer = the opposite pattern"),
                         (17, "Slide it across the whole image \u2192 a feature map"),
                         (22, "This filter lights up on the edges of fingers")], "s2")
        return
    # 16 real conv1 maps
    fr.put(photo(img64, 200, "cimg2"), 60, 200, "cimg2", sec(26))
    fr.arrow(290, 300, 380, 300, "a16", INK, 5, sec(26.5))
    fr.text(335, 260, "16 filters", "a16l", 22, INK, "bold", "ma", sec(26.5))
    for k in range(16):
        r_, c_ = divmod(k, 4)
        grp = k // 4
        pc = fmap(R["c1"][k], 104, ("m16", k))
        fr.put(pc, 420 + c_ * 124, 108 + r_ * 128, ("m16", k), sec(27 + grp * 0.75))
        if k == FILTER:
            fr.put(_ring(116), 420 + c_ * 124 - 2, 108 + r_ * 128 - 2, "ring", sec(27 + grp * 0.75), shadow=False)
    fr.text(950, 250, "red = positive", "lg1", 20, (190, 70, 40), "bold", start=sec(29))
    fr.text(950, 280, "blue = negative", "lg2", 20, BLUE, "bold", start=sec(29))
    fr.text(950, 320, f"each {R['c1'].shape[1]} \u00d7 {R['c1'].shape[2]}", "lg3", 20, INK, "sans", start=sec(29))
    caption_seq(fr, [(26, f"Layer 1 has {R['n_filters'][0]} filters \u2192 {R['c1'].shape[0]} feature maps"),
                     (30, "Each one looks for a different pattern")], "s2b")


def _win(S):
    im = Image.new("RGBA", (3 * S + 2 * M, 3 * S + 2 * M), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((M - 2, M - 2, M + 3 * S + 1, M + 3 * S + 1), outline=(255, 230, 0, 255), width=3)
    ImageDraw.Draw(im).rectangle((M - 4, M - 4, M + 3 * S + 3, M + 3 * S + 3), outline=INK + (255,), width=1)
    return im


def _ring(px):
    im = Image.new("RGBA", (px + 2 * M, px + 2 * M), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((M, M, M + px, M + px), outline=RED + (255,), width=4)
    return im


def pick_patch(a, size, want):
    """Find a size x size patch of map a: 'mixed' = most balanced signs,
    'pool' = most non-zero with varied values."""
    best, bp = -1, (0, 0)
    for i in range(0, a.shape[0] - size, 2):
        for j in range(0, a.shape[1] - size, 2):
            p = a[i:i + size, j:j + size]
            if want == "mixed":
                s = min((p < 0).sum(), (p > 0).sum()) + 0.1 * np.abs(p).mean()
            else:
                s = (p > 0).sum() + 10 * p.std()
            if s > best:
                best, bp = s, (i, j)
    return bp


def scene_relu(fr, R):
    f = fr.f
    fr.tag("3 \u00b7 ReLU", "s3")
    c1, r1 = R["c1"][FILTER], R["r1"][FILTER]
    vmax = robust_max(c1)
    i0, j0 = R["relu_patch"]
    # progress of the flip: one grid row every 0.75 s from 6 s
    flipped = 0 if f < sec(6) else min(6, (f - sec(6)) // sec(0.75) + 1)
    shown = c1.copy()
    band = int(64 * flipped / 6)
    shown[:band] = r1[:band]
    fr.put(fmap(shown, 288, "rmap", vmax=vmax), 40, 130, "rmap", 0)
    box = Image.new("RGBA", (27 + 2 * M, 27 + 2 * M), (0, 0, 0, 0))
    ImageDraw.Draw(box).rectangle((M, M, M + 27, M + 27), outline=INK + (255,), width=3)
    fr.put(box, 46 + j0 * 4.5, 136 + i0 * 4.5, "rbox", sec(1), shadow=False)
    fr.text(190, 442, f"feature map #{FILTER}", "rmapl", 22, INK, "bold", "ma")
    patch = c1[i0:i0 + 6, j0:j0 + 6].copy()
    patch[:flipped] = np.maximum(0, patch[:flipped])
    fr.put(grid_piece(patch, 54, ("rg", flipped), d=2, size=15, vmax=vmax), 380, 130, "rg", sec(1), frm=(-40, 0))
    fr.line([(46 + j0 * 4.5 + 27, 136 + i0 * 4.5), (380, 140)], "rl1", INK, 2, sec(1))
    fr.line([(46 + j0 * 4.5 + 27, 136 + i0 * 4.5 + 27), (380, 460)], "rl2", INK, 2, sec(1))
    # ReLU graph
    if f >= sec(4):
        g = card(300, 300, CREAM, "graph")
        dg = ImageDraw.Draw(g)
        ox, oy = M + 150, M + 200
        dg.line((M + 20, oy, M + 280, oy), fill=SOFT, width=2)
        dg.line((ox, M + 50, ox, M + 280), fill=SOFT, width=2)
        dg.line((M + 25, oy, ox, oy), fill=BLUE, width=6)
        dg.line((ox, oy, M + 270, oy - 120), fill=(214, 84, 44), width=6)
        write(g, 280, 202, "z", 22, INK, "bold", "ra")
        write(g, 150, 12, "ReLU(z) = max(0, z)", 22, INK, "bold", "ma")
        write(g, 75, 170, "negative \u2192 0", 18, BLUE, "bold", "ma")
        write(g, 222, 214, "positive stays", 18, (180, 70, 40), "bold", "ma")
        fr.put(g, 830, 130, "graph", sec(4))
    if f >= sec(11):
        c = card(330, 80, (255, 236, 170), "vfit")
        write(c, 165, 22, "Same ReLU as my", 24, INK, "bold", "mm")
        write(c, 165, 54, "V data fit!", 24, RED, "bold", "mm")
        fr.put(c, 815, 470, "vfit", sec(11))
    caption_seq(fr, [(0, "Some of the numbers are negative (blue)"),
                     (4, "ReLU: if it's negative, make it 0"),
                     (6, "Negatives \u2192 0, positives stay the same"),
                     (11, "Same ReLU as my V data fit: max(0, z)")], "s3")


def scene_pool(fr, R):
    f = fr.f
    fr.tag("4 \u00b7 Max pooling", "s4")
    r1, p1 = R["r1"][FILTER], R["p1"][FILTER]
    i0, j0 = R["pool_patch"]
    patch = r1[i0:i0 + 4, j0:j0 + 4]
    vmax = robust_max(r1)
    cols = [(205, 62, 50), (52, 98, 170), (70, 150, 90), (200, 140, 20)]
    blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
    nb = 0 if f < sec(2) else min(4, (f - sec(2)) // sec(2) + 1)
    outl = [((r, c, r + 2, c + 2), cols[k]) for k, (r, c) in enumerate(blocks[:max(nb, 0)])]
    bold = []
    for (r, c) in blocks[:nb]:
        b = patch[r:r + 2, c:c + 2]
        a = np.unravel_index(b.argmax(), b.shape)
        bold.append((r + a[0], c + a[1]))
    fr.put(grid_piece(patch, 68, ("pg", nb), d=2, size=17, vmax=vmax, outlines=outl, bold_cells=bold),
           60, 150, "pg", 0)
    fr.text(200, 450, "4 \u00d7 4 piece (after ReLU)", "pgl", 20, INK, "bold", "ma")
    if nb:
        o = np.full((2, 2), np.nan)
        for k, (r, c) in enumerate(blocks[:nb]):
            o[r // 2, c // 2] = patch[r:r + 2, c:c + 2].max()
        og = grid_piece(o, 68, ("po", nb), d=2, size=17, vmax=vmax,
                        outlines=[((r // 2, c // 2, r // 2 + 1, c // 2 + 1), cols[k])
                                  for k, (r, c) in enumerate(blocks[:nb])])
        fr.put(og, 450, 218, "po", sec(2))
        fr.arrow(370, 290, 435, 290, "pa", INK, 5)
        fr.text(520, 390, "2 \u00d7 2 result", "pol", 20, INK, "bold", "ma")
        # check against the real pool
        real = p1[i0 // 2:i0 // 2 + 2, j0 // 2:j0 // 2 + 2]
        for k, (r, c) in enumerate(blocks[:nb]):
            assert abs(real[r // 2, c // 2] - o[r // 2, c // 2]) < 1e-6
    if f >= sec(10):
        fr.put(fmap(r1, 256, "pm64", mode="pos", vmax=vmax), 690, 130, "pm64", sec(10))
        fr.text(818, 400, f"{r1.shape[0]} \u00d7 {r1.shape[1]}", "pm64l", 26, INK, "bold", "ma", sec(10))
        fr.arrow(965, 260, 1030, 260, "pma", INK, 5, sec(11))
        fr.put(fmap(p1, 128, "pm32", mode="pos", vmax=vmax), 1050, 194, "pm32", sec(11))
        fr.text(1114, 336, f"{p1.shape[0]} \u00d7 {p1.shape[1]}", "pm32l", 26, INK, "bold", "ma", sec(11))
        P = R["p1"]
        fr.text(960, 470, f"all {P.shape[0]} maps: {P.shape[0]} \u00d7 {P.shape[1]} \u00d7 {P.shape[2]}",
                "pall", 26, RED, "bold", "ma", sec(12.5))
    caption_seq(fr, [(0, "Look at each 2 \u00d7 2 block"),
                     (2, "Keep only the biggest number"),
                     (10, f"The map shrinks: {r1.shape[0]} \u00d7 {r1.shape[1]} \u2192 {p1.shape[0]} \u00d7 {p1.shape[1]}"),
                     (12.5, "A finger moved a few pixels still looks the same")], "s4")


def stack_piece(n_sheets, px, key, maps=None, mode="pos"):
    step = 6
    size = px + step * (n_sheets - 1) + 2 * M + 10
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for s in range(n_sheets - 1, -1, -1):
        if maps is not None:
            a = maps[min(s, len(maps) - 1)]
            p = fmap(a, px, (key, s), mode=mode, border=3) if mode != "img" else photo(a, px, (key, s), 3)
        else:
            p = card(px, px, CREAM, (key, s), outline=SOFT)
        im.alpha_composite(p, (s * step, s * step)) if p.width + s * step <= size else None
    return im


def scene_deeper(fr, R):
    f = fr.f
    fr.tag("5 \u00b7 Going deeper", "s5")
    img64 = (R["x"].transpose(1, 2, 0) * 255).round().astype(np.uint8)
    shapes = [R["x"].shape, R["p1"].shape, R["p2"].shape, R["p3"].shape]
    if f < sec(8):
        xs = [40, 330, 640, 930]
        pxs = [150, 110, 80, 56]
        sheets = [3, 8, 12, 16]
        data = [[img64] * 3, list(R["p1"]), list(R["p2"]), list(R["p3"])]
        for k in range(4):
            mode = "img" if k == 0 else "pos"
            sp = stack_piece(sheets[k], pxs[k], ("stk", k), data[k], mode)
            y = 330 - (pxs[k] + 6 * sheets[k]) // 2
            fr.put(sp, xs[k], y, ("stk", k), sec(0.3 + 1.5 * k))
            c, h, w = shapes[k]
            fr.text(xs[k] + (pxs[k] + 6 * sheets[k]) // 2, 470, f"{c} \u00d7 {h} \u00d7 {w}",
                    ("stkl", k), 26, INK, "bold", "ma", sec(0.3 + 1.5 * k))
            if k:
                ax = xs[k] - 70
                fr.arrow(ax, 330, ax + 55, 330, ("sa", k), INK, 5, sec(1.5 * k))
                fr.text(ax + 28, 360, f"{R['n_filters'][k - 1]} filters", ("saf", k), 18, RED, "bold", "ma", sec(1.5 * k))
                fr.text(ax + 28, 384, f"{num(R['n_layer'][k - 1])} weights", ("saw", k), 16, SOFT, "sans", "ma", sec(1.5 * k))
        fr.text(640, 540, "maps get smaller, but there are more of them", "s5t", 26, RED, "bold", "ma", sec(6))
        caption_seq(fr, [(0, "Each layer: convolution \u2192 ReLU \u2192 max pool"),
                         (3, f"Width = how many filters: {R['n_filters'][0]}, {R['n_filters'][1]}, {R['n_filters'][2]}"),
                         (6, "The shape shrinks while the number of maps grows")], "s5a")
        return
    if f < sec(16):
        P2, P3 = R["p2"], R["p3"]
        v2, v3 = P2.max(), P3.max()
        for k in range(P2.shape[0]):
            r_, c_ = divmod(k, 8)
            fr.put(fmap(P2[k], 52, ("l2", k), "pos", 3, vmax=v2), 40 + c_ * 64, 140 + r_ * 66, ("l2", k),
                   sec(8.2 + 0.25 * r_), n=2)
        fr.text(290, 420, f"layer 2: {P2.shape[0]} maps, {P2.shape[1]} \u00d7 {P2.shape[2]}", "l2l", 26, INK, "bold", "ma", sec(8.2))
        for k in range(P3.shape[0]):
            r_, c_ = divmod(k, 8)
            fr.put(fmap(P3[k], 40, ("l3", k), "pos", 2, vmax=v3), 650 + c_ * 52, 100 + r_ * 52, ("l3", k),
                   sec(11 + 0.2 * r_), n=2)
        fr.text(900, 530, f"layer 3: {P3.shape[0]} maps, {P3.shape[1]} \u00d7 {P3.shape[2]}", "l3l", 26, INK, "bold", "ma", sec(11))
        caption_seq(fr, [(8, f"Layer 2: {P2.shape[0]} maps, each {P2.shape[1]} \u00d7 {P2.shape[2]}"),
                         (11, f"Layer 3: {P3.shape[0]} maps, each {P3.shape[1]} \u00d7 {P3.shape[2]}"),
                         (13.5, "Deeper maps are blurrier, but mean more")], "s5b")
        return
    # depth tower
    names = ["conv 1", "conv 2", "conv 3", "linear"]
    outs = [f"{R['p1'].shape[0]}\u00d7{R['p1'].shape[1]}\u00d7{R['p1'].shape[2]}",
            f"{R['p2'].shape[0]}\u00d7{R['p2'].shape[1]}\u00d7{R['p2'].shape[2]}",
            f"{R['p3'].shape[0]}\u00d7{R['p3'].shape[1]}\u00d7{R['p3'].shape[2]}",
            f"{R['fc_shape'][0]} scores"]
    widths = [R["n_filters"][0], R["n_filters"][1], R["n_filters"][2], R["fc_shape"][0]]
    cols = [(240, 200, 150), (230, 170, 110), (215, 135, 80), (150, 190, 160)]
    for k in range(4):
        label = f"{names[k]}  \u2192  {outs[k]}"
        wpx = max(text_size(label, 22, "bold")[0] + 40, 150 + 4 * widths[k] if k < 3 else 0)
        c = card(wpx, 62, cols[k], ("tw", k))
        write(c, wpx // 2, 31, label, 22, INK, "bold", "mm")
        fr.put(c, 330 - wpx // 2, 520 - k * 86, ("tw", k), sec(16.2 + 0.8 * k), frm=(0, -80))
    fr.arrow(70, 590, 70, 190, "depth", RED, 5, sec(16.2))
    fr.text(86, 180, "depth = 4 layers", "depthl", 22, RED, "bold", start=sec(19.5))
    fr.text(86, 206, "(3 conv + 1 linear)", "depthl2", 18, SOFT, start=sec(19.5))
    words = [("layer 1: edges", 534), ("layer 2: finger parts", 448), ("layer 3: whole fingers", 362),
             ("linear: how many?", 276)]
    for k, (s, y) in enumerate(words):
        fr.text(720, y, s, ("wd", k), 28, INK, "bold", start=sec(19 + 0.8 * k))
    caption_seq(fr, [(16, "Depth = how many layers are stacked"),
                     (19, "Simple patterns first, then bigger ideas")], "s5c")


def scene_output(fr, R):
    f = fr.f
    fr.tag("6 \u00b7 The answer", "s6")
    p3, flat, scores, probs = R["p3"], R["flat"], R["scores"], R["probs"]
    win = int(probs.argmax())
    fr.put(photo(R["photo96"], 150, "o_ph"), 40, 100, "o_ph", 0)
    fr.put(stack_piece(10, 70, "o_stk", list(p3), "pos"), 70, 300, "o_stk", sec(0.5))
    c, h, w = p3.shape
    fr.text(130, 470, f"{c} \u00d7 {h} \u00d7 {w}", "o_stkl", 22, INK, "bold", "ma", sec(0.5))
    # flatten strip: first 100 of the 4,096 numbers
    if f >= sec(2):
        n_show = 96
        st = card(n_show * 8 + 8, 34, CREAM, "strip", cut=1)
        ds = ImageDraw.Draw(st)
        col = cmap_pos(flat[:n_show], flat.max())
        upto = n_show if f >= sec(3) else n_show // 2
        for k in range(upto):
            ds.rectangle((M + 4 + k * 8, M + 4, M + 4 + k * 8 + 6, M + 30), fill=tuple(int(v) for v in col[k]))
        fr.put(st, 300, 110, "strip", sec(2), frm=(-60, 0))
        fr.text(1100, 116, "\u2026", "dots", 30, INK, "bold", start=sec(3))
        fr.text(700, 156, f"flatten: {c} \u00d7 {h} \u00d7 {w} = {num(flat.size)} numbers in one line",
                "fl", 22, INK, "bold", "ma", sec(3))
        fr.arrow(200, 350, 290, 140, "fla", INK, 3, sec(2))
    xs = [380 + k * 140 for k in range(6)]
    if f >= sec(5):
        n_w = R["fc_shape"][0] * R["fc_shape"][1]
        c_ = card(560, 50, (255, 236, 170), "lin")
        write(c_, 280, 25, f"linear: {R['fc_shape'][0]} \u00d7 {num(R['fc_shape'][1])} weights + {R['fc_shape'][0]} = {num(R['n_layer'][3])}",
              22, INK, "bold", "mm")
        fr.put(c_, 440, 196, "lin", sec(5))
        for k in range(6):
            fr.line([(720, 250), (xs[k] + 50, 300)], ("fan", k), SOFT, 2, sec(5.5))
    if f >= sec(9):
        for k in range(6):
            sc = card(100, 64, CREAM, ("sc", k))
            write(sc, 50, 18, f"{k} fingers", 15, SOFT, "bold", "mm")
            write(sc, 50, 44, fmt(scores[k], 1), 22, INK, "bold", "mm")
            fr.put(sc, xs[k], 300, ("sc", k), sec(9 + 0.25 * k))
    if f >= sec(13):
        base = 610
        for k in range(6):
            hgt = max(3, int(200 * probs[k]))
            p = stepped(f, sec(13.5), 4, 2) or 0.0
            hh = max(3, int(hgt * p))
            col = RED if k == win and f >= sec(16) else BLUE
            bar = card(70, hh, col, ("bar", k), cut=1)
            fr.put(bar, xs[k] + 15, base - hh, ("bar", k), sec(13), frm=(0, 0), n=1)
            fr.text(xs[k] + 50, base - hh - 30, f"{probs[k] * 100:.1f}%", ("pct", k), 20,
                    RED if k == win and f >= sec(16) else INK, "bold", "ma", sec(14.5))
        fr.line([(xs[0] - 10, base), (xs[5] + 110, base)], "base", INK, 3)
        fr.text(1100, 470, f"sum = {probs.sum() * 100:.0f}%", "sum", 22, SOFT, "bold", start=sec(15))
        if f >= sec(16):
            t = card(250, 56, (255, 236, 170), "win", cut=3)
            write(t, 125, 28, f"{win} fingers! {probs[win] * 100:.1f}%", 26, RED, "bold", "mm")
            fr.put(t, 60, 540, "win", sec(16))
    caption_seq(fr, [(0, "The last layer gives 64 maps of 8 \u00d7 8"),
                     (2, f"Flatten: line up all {num(flat.size)} numbers"),
                     (5, "Each answer gets a weight for every number"),
                     (9, "Add them up \u2192 6 scores, one for each finger count"),
                     (13, "Softmax turns the scores into probabilities (sum = 100%)"),
                     (16, f"Winner: {win} fingers, {probs[win] * 100:.1f}% sure")], "s6")


def augment_views(t96):
    h, w = t96.shape[:2]
    def warp(M_):
        return cv2.warpAffine(t96, M_, (w, h), borderMode=cv2.BORDER_REPLICATE)
    rot = warp(cv2.getRotationMatrix2D((w / 2, h / 2), 20, 1.0))
    shift = warp(np.float32([[1, 0, 10], [0, 1, -8]]))
    mirror = t96[:, ::-1].copy()
    bright = np.clip(t96.astype(np.float32) * 1.25 + 10, 0, 255).astype(np.uint8)
    return [("rotate 20\u00b0", rot), ("shift", shift), ("mirror", mirror), ("brighter", bright)]


def scene_train(fr, R):
    f = fr.f
    fr.tag("7 \u00b7 Training", "s7")
    probs = R["probs"]
    right = PHOTO_ROW
    pr = float(probs[right])
    if f < sec(8):
        c = card(700, 90, CREAM, "lossf", cut=3)
        write(c, 350, 45, "loss = \u2212ln( probability of the right answer )", 28, INK, "bold", "mm")
        fr.put(c, 40, 110, "lossf", 0)
        ex1 = card(700, 62, (220, 240, 220), "ex1")
        write(ex1, 20, 31, f"this photo: right answer {right}, p = {pr:.3f} \u2192 loss = {-math.log(pr):.3f}", 22, INK, "bold", "lm")
        fr.put(ex1, 40, 240, "ex1", sec(1.5))
        ex2 = card(700, 62, (250, 220, 210), "ex2")
        write(ex2, 20, 31, f"if it had said p = 0.10 \u2192 loss = {-math.log(0.10):.2f}  (big!)", 22, INK, "bold", "lm")
        fr.put(ex2, 40, 330, "ex2", sec(4))
        g = card(420, 380, CREAM, "lcurve")
        dg = ImageDraw.Draw(g)
        ox, oy, sx, sy = M + 50, M + 330, 340, 90
        dg.line((ox, oy, ox + sx + 10, oy), fill=SOFT, width=2)
        dg.line((ox, oy, ox, M + 20), fill=SOFT, width=2)
        pts = [(ox + p * sx, oy - (-math.log(p)) * sy) for p in np.linspace(0.04, 1, 60)]
        dg.line(pts, fill=RED, width=4)
        write(g, 210, 340, "probability of the right answer", 16, SOFT, "bold", "ma")
        write(g, 10, 4, "loss", 18, SOFT, "bold")
        write(g, 50, 342, "0", 16, SOFT, "sans", "ma")
        write(g, 390, 342, "1", 16, SOFT, "sans", "ma")
        dot = (ox + pr * sx, oy + math.log(pr) * sy)
        dg.ellipse((dot[0] - 8, dot[1] - 8, dot[0] + 8, dot[1] + 8), fill=GREEN)
        d2 = (ox + 0.1 * sx, oy + math.log(0.1) * sy)
        if f >= sec(4):
            dg.ellipse((d2[0] - 8, d2[1] - 8, d2[0] + 8, d2[1] + 8), fill=RED)
        fr.put(g, 800, 110, "lcurve", sec(0.8))
        caption_seq(fr, [(0, "Training starts with the loss: how wrong was it?"),
                         (1.5, "Sure and right \u2192 small loss"),
                         (4, "Unsure or wrong \u2192 big loss")], "s7a")
        return
    if f < sec(15):
        names = ["conv 1", "conv 2", "conv 3", "linear", "loss"]
        xs = [60 + k * 240 for k in range(5)]
        for k in range(5):
            col = (250, 220, 210) if k == 4 else (240, 200, 150)
            c = card(170, 80, col, ("bp", k))
            write(c, 85, 40, names[k], 26, INK, "bold", "mm")
            fr.put(c, xs[k], 250, ("bp", k), sec(8 + 0.3 * k))
        for k in range(4):
            fr.arrow(xs[k] + 175, 270, xs[k + 1] - 8, 270, ("fw", k), SOFT, 3, sec(9))
        fr.text(640, 200, "forward: photo \u2192 answer", "fwl", 22, SOFT, "bold", "ma", sec(9))
        nb = 0 if f < sec(10) else min(4, (f - sec(10)) // sec(0.8) + 1)
        for k in range(nb):
            a = 3 - k
            fr.arrow(xs[a + 1] - 8, 315, xs[a] + 175, 315, ("bw", a), RED, 6)
            fr.text((xs[a + 1] + xs[a] + 170) // 2, 330, "chain rule", ("bwl", a), 16, RED, "bold", "ma")
        if f >= sec(10):
            fr.text(640, 380, "back-propagation: error goes backwards", "bwt", 24, RED, "bold", "ma")
        if f >= sec(13):
            c = card(760, 70, (255, 236, 170), "adam")
            write(c, 380, 35, f"Adam nudges all {num(R['n_total'])} weights a little downhill", 26, INK, "bold", "mm")
            fr.put(c, 260, 470, "adam", sec(13))
            for k in range(4):  # little knobs turning step by step
                ang = ((f - sec(13)) // 2) * 0.5 + k
                kx, ky = xs[k] + 85, 230
                fr.d.ellipse((kx - 12, ky - 12, kx + 12, ky + 12), fill=CREAM, outline=INK, width=2)
                fr.d.line((kx, ky, kx + 11 * math.cos(ang), ky + 11 * math.sin(ang)), fill=INK, width=3)
        caption_seq(fr, [(8, "Back-propagation uses the chain rule, like my V data fit"),
                         (10, "The error is sent back through every layer"),
                         (13, f"Gradient descent (Adam) nudges all {num(R['n_total'])} weights")], "s7b")
        return
    if f < sec(22):
        fr.put(photo(R["photo96"], 200, "au0"), 60, 180, "au0", sec(15))
        fr.text(166, 410, "original", "au0l", 22, INK, "bold", "ma")
        for k, (nm, img) in enumerate(augment_views(R["photo96"])):
            x0 = 330 + k * 230
            fr.put(photo(img, 180, ("au", k)), x0, 200, ("au", k), sec(16 + k * 1.2), frm=(-50, -30))
            fr.text(x0 + 96, 410, nm, ("aul", k), 22, INK, "bold", "ma", sec(16 + k * 1.2))
        fr.arrow(290, 290, 325, 290, "aua", INK, 4, sec(16))
        fr.text(640, 480, "a new version every time a photo is used", "aut", 26, RED, "bold", "ma", sec(20))
        caption_seq(fr, [(15, "Data augmentation: change the same photo a little"),
                         (16, "Rotate, shift, mirror, brighten"),
                         (20, "So it can't just memorize the photos")], "s7c")
        return
    tr = Image.open(ROOT / "results" / "training.png").convert("RGB")
    tw = 700
    tr = tr.resize((tw, int(tr.height * tw / tr.width)), Image.LANCZOS)
    pc = card(tr.width + 12, tr.height + 12, (252, 251, 247), "trpng", cut=1.5)
    pc.paste(tr, (M + 6, M + 6))
    fr.put(pc, 30, 170, "trpng", sec(22), frm=(-60, 0))
    cf = Image.open(ROOT / "results" / "confusion.png").convert("RGB")
    ch_ = 390
    cf = cf.resize((int(cf.width * ch_ / cf.height), ch_), Image.LANCZOS)
    pc2 = card(cf.width + 12, cf.height + 12, (252, 251, 247), "cfpng", cut=1.5)
    pc2.paste(cf, (M + 6, M + 6))
    fr.put(pc2, 780, 130, "cfpng", sec(24), frm=(60, 0))
    fr.text(380, 480, f"{R['epochs']} passes through {R['n_photos']} photos", "trl", 26, INK, "bold", "ma", sec(23))
    caption_seq(fr, [(22, f"The loss goes down over {R['epochs']} passes"),
                     (24, "Most mistakes are off by one (like 3 \u2194 4)"),
                     (27, f"{R['accuracy']}% right on held-out photos it never trained on")], "s7d")


def scene_policy(fr, R):
    f = fr.f
    fr.tag("8 \u00b7 The lock policy", "s8")
    code = R["code"]
    pol = R["policy"]
    conf = pol["conf"]
    # timeline of what the camera sees: (start, end, count, digit_index or None)
    plan = [(0, 3.5, code[0], 0), (3.5, 5, 0, None), (5, 8, code[1], 1), (8, 9, 0, None),
            (9, 12, code[2], 2)]
    t = f / FPS
    entered = []
    seg = None
    for (a, b, cnt, di) in plan:
        if t >= a:
            seg = (a, b, cnt, di)
        if di is not None and t >= a + 2.5:
            entered.append(cnt)
    if t < 15.5:
        a, b, cnt, di = seg if t < 12 else plan[-1]
        img, pr = R["best"][cnt]
        fr.put(photo(img, 190, ("cam8", cnt)), 50, 110, ("cam8", cnt), sec(a), frm=(30, 0))
        fr.text(152, 82, "camera", "camt", 20, SOFT, "bold", "ma")
        # bars with the 70% line
        bx, base = 40, 560
        for k in range(6):
            hh = max(2, int(150 * pr[k]))
            col = RED if k == int(pr.argmax()) and pr[k] >= conf else BLUE
            fr.d.rectangle((bx + k * 36, base - hh, bx + k * 36 + 26, base), fill=col)
            fr.text(bx + k * 36 + 13, base + 6, str(k), ("bl", k), 16, INK, "bold", "ma")
        ly = base - int(150 * conf)
        fr.line([(bx - 6, ly), (bx + 6 * 36, ly)], "line70", (200, 140, 20), 3)
        fr.text(bx + 6 * 36 + 6, ly - 12, f"{conf * 100:.0f}%", "l70", 18, (200, 140, 20), "bold")
        # hold timer: fills in 4 steps over the hold
        if di is not None and t < 12:
            p = min(1.0, max(0, (t - a - 0.5)) / 2.0)
            p = math.floor(p * 4) / 4
            tc = card(300, 40, CREAM, "hold", cut=1.5)
            ImageDraw.Draw(tc).rectangle((M + 4, M + 4, M + 4 + int(292 * p), M + 36), fill=GREEN)
            write(tc, 150, 20, f"hold {pol['hold']:.0f} s", 20, INK, "bold", "mm")
            fr.put(tc, 320, 190, "hold", sec(a))
            fr.text(470, 150, f"same count, \u2265 {conf * 100:.0f}% for {pol['hold']:.0f} second", "holdt", 20, INK, "bold", "ma", sec(a))
        elif t < 12:
            fr.text(470, 200, "hand away \u2192 0 fingers", "away", 26, BLUE, "bold", "ma", sec(a))
    # code slots
    wrong = t >= 15.5
    shown = entered if not wrong else [code[0], 5, code[2]]
    for k in range(3):
        sl = card(90, 110, DARK, ("slot", k), cut=1.5)
        if k < len(shown):
            write(sl, 45, 55, str(shown[k]), 64, (255, 200, 60), "bold", "mm")
        else:
            write(sl, 45, 70, "_", 64, (110, 106, 110), "bold", "mm")
        fr.put(sl, 820 + k * 110, 110, ("slot", k), 0)
    fr.text(985, 240, "3 digits", "slott", 22, INK, "bold", "ma")
    # the box with the lid
    open_p = 0 if t < 12.5 or wrong else min(3, int((t - 12.5) / 0.5) + 1) / 3
    ang = math.radians(55) * open_p
    hx, hy = 900, 480
    lid = [(0, -22), (240, -22), (240, 0), (0, 0)]
    pts = [(hx + x * math.cos(ang) + y * math.sin(ang), hy - x * math.sin(ang) + y * math.cos(ang)) for x, y in lid]
    fr.d.polygon(pts, fill=(140, 96, 60))
    fr.put(card(240, 150, BROWN, "box8"), 900, 480, "box8", 0)
    if 12 <= t < 15.5:
        ok = list(code) == list(entered)
        c = card(260, 60, (220, 240, 220), "ok")
        write(c, 130, 30, "code matches \u2713" if ok else "wrong", 26, GREEN, "bold", "mm")
        fr.put(c, 340, 190, "ok", sec(12))
        if t >= 12.5:
            fr.text(700, 520, "motors lift the lid!", "mot", 24, GREEN, "bold", "ma")
    if wrong:
        x0, y0 = 890, 300
        fr.line([(x0, y0), (x0 + 260, y0 + 150)], "x1", RED, 14)
        fr.line([(x0 + 260, y0), (x0, y0 + 150)], "x2", RED, 14)
        c = card(560, 150, CREAM, "lockrules", cut=2)
        write(c, 20, 18, "wrong code \u2192 \u2717, box stays shut", 24, INK, "bold")
        mw = pol["max_wrong"]
        write(c, 20, 60, f"{mw} wrong in a row \u2192 {pol['lockout']:.0f} s lockout", 24, RED, "bold")
        write(c, 20, 102, f"too slow (over {pol['timeout']:.0f} s) \u2192 starts over", 22, SOFT, "bold")
        fr.put(c, 60, 300, "lockrules", sec(16))
        if t >= 17.5:
            left = max(0, int(pol["lockout"]) - int((t - 17.5) * 2))
            lc = card(170, 70, DARK, "lockc")
            write(lc, 85, 35, f"locked {left} s", 26, (255, 200, 60), "bold", "mm")
            fr.put(lc, 250, 480, "lockc", sec(17.5))
    if t >= 20.5:
        c = card(520, 70, (255, 236, 170), "end", cut=3)
        write(c, 260, 35, "Finger Lock \u00b7 ME-193 \u00b7 thanks!", 28, INK, "bold", "mm")
        fr.put(c, 380, 20, "end", sec(20.5))
    caption_seq(fr, [(0, f"Hold a count for {pol['hold']:.0f} second at {conf * 100:.0f}% or more"),
                     (3.5, "Then move your hand away (0 fingers)"),
                     (5, "Next digit: hold it again"),
                     (9, "Third digit"),
                     (12, "3 digits match the code \u2192 unlock, the lid opens"),
                     (15.5, "Wrong code \u2192 \u2717 and it stays shut"),
                     (17.5, f"{pol['max_wrong']} wrong in a row \u2192 {pol['lockout']:.0f} s lockout")], "s8")


SCENES = [("The lock", 20, scene_lock),
          ("Convolution", 34, scene_conv),
          ("ReLU", 15, scene_relu),
          ("Max pooling", 15, scene_pool),
          ("Going deeper", 24, scene_deeper),
          ("The answer", 20, scene_output),
          ("Training", 30, scene_train),
          ("The lock policy", 22, scene_policy)]

KEYFRAMES = [("keyframe_1_convolution.png", 1, 11.5),
             ("keyframe_2_deeper.png", 4, 14.5),
             ("keyframe_3_answer.png", 5, 18.5),
             ("keyframe_4_training.png", 6, 29)]


def timeline():
    out, t = [], 0
    for name, dur, fn in SCENES:
        out.append((name, t, dur, fn))
        t += dur
    return out, t


def render(R, gf):
    tl, _ = timeline()
    for name, t0, dur, fn in tl:
        if gf < sec(t0 + dur):
            fr = Frame(gf, gf - sec(t0))
            fn(fr, R)
            return fr.im
    raise IndexError(gf)


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            sys.exit("Need ffmpeg: pip install imageio-ffmpeg")
        return exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true", help="only write the keyframe PNGs")
    ap.add_argument("--out", default=str(DOCS / "stop_motion.mp4"))
    args = ap.parse_args()

    R = add_picks(load_real())

    print("numbers used (all from the project files):")
    print(f"  photo: sample_photos.png row {PHOTO_ROW}, col {PHOTO_COL} -> input {R['x'].shape} = {R['x'].size}")
    print(f"  stages: {R['c1'].shape} -> {R['p1'].shape} -> {R['p2'].shape} -> {R['p3'].shape} -> {R['flat'].size}")
    print(f"  weights per layer: {R['n_layer']}  total: {R['n_total']}")
    print("  probabilities:", " ".join(f"{k}:{p * 100:.1f}%" for k, p in enumerate(R["probs"])))
    print(f"  epochs {R['epochs']}, photos {R['n_photos']}, accuracy {R['accuracy']}%, policy {R['policy']}, code {R['code']}")

    tl, total = timeline()
    print("timeline:")
    for name, t0, dur, _ in tl:
        print(f"  {int(t0) // 60}:{int(t0) % 60:02d}  {name} ({dur} s)")
    print(f"  total {int(total) // 60}:{int(total) % 60:02d}")

    starts = {i: t0 for i, (_, t0, _, _) in enumerate(tl)}
    for fname, si, t in KEYFRAMES:
        render(R, sec(starts[si] + t)).save(DOCS / fname)
        print("wrote", fname)
    if args.stills:
        return

    n = sec(total)
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for gf in range(n):
        proc.stdin.write(render(R, gf).tobytes())
        if gf % 80 == 0:
            print(f"  frame {gf}/{n}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", args.out, f"({n} frames, {n / FPS:.0f} s)")


if __name__ == "__main__":
    main()
