"""Video fuer "mid day out": Super-8 / 60er Look, 4:3, harte Schnitte im Takt.

18 Bilder/s (Super 8), 120 BPM -> 9 Frames pro Beat, 1080 Frames = 60 s.
Gibt rohe RGB-Frames auf stdout aus (fuer ffmpeg).
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 960, 720
FPS = 18
FPB = 9                     # Frames pro Beat
TOTAL = 1080
rng = np.random.default_rng(1966)

# ------------------------------------------------------------- Shotliste
# datei, beats, (cx0, cy0, zoom0) -> (cx1, cy1, zoom1), jumpcut-ziel (cx, cy, zoom) oder None, wackeln
S = "bilder/"
SHOTS = [
    ("TITLE", 8, None, None, None, 0),
    (S + "01_sperrbildschirm.png", 8, (0.5, 0.17, 1.0), (0.5, 0.30, 1.05), (0.5, 0.75, 1.4), 0.6),
    (S + "02_schuhe.jpg", 6, (0.5, 0.45, 1.0), (0.45, 0.78, 1.1), None, 1.2),
    (S + "03_treppe.jpg", 8, (0.5, 0.80, 1.0), (0.5, 0.45, 1.05), (0.62, 0.15, 1.8), 1.0),
    (S + "04_turm.jpg", 6, (0.5, 0.80, 1.0), (0.5, 0.22, 1.0), None, 0.7),
    (S + "05_selfie.jpg", 8, (0.5, 0.42, 1.0), (0.5, 0.42, 1.15), (0.56, 0.44, 1.9), 0.8),
    (S + "06_sakko.jpg", 6, (0.5, 0.42, 1.0), (0.5, 0.38, 1.15), None, 0.7),
    (S + "07_pelz_gruen.jpg", 8, (0.5, 0.45, 1.0), (0.52, 0.40, 1.2), (0.53, 0.30, 2.0), 0.7),
    (S + "08_pelz_grau.jpg", 6, (0.5, 0.30, 1.0), (0.5, 0.55, 1.0), None, 0.7),
    (S + "09_galerie_leer.jpg", 8, (0.52, 0.52, 2.2), (0.5, 0.48, 1.1), (0.52, 0.50, 3.2), 0.4),
    (S + "10_galerie_publikum.jpg", 6, (0.30, 0.42, 1.5), (0.68, 0.42, 1.5), None, 0.6),
    (S + "11_kunst_blau.jpg", 8, (0.53, 0.45, 1.0), (0.53, 0.45, 1.3), (0.53, 0.52, 2.4), 0.5),
    (S + "12_teddy.jpg", 6, (0.5, 0.38, 1.0), (0.5, 0.30, 1.35), None, 0.6),
    (S + "13_globus.jpg", 8, (0.5, 0.40, 1.0), (0.5, 0.36, 1.25), (0.42, 0.33, 2.2), 0.6),
    (S + "14_aktie.png", 6, (0.5, 0.18, 1.0), (0.5, 0.42, 1.0), None, 0.5),
    (S + "15_dom_nacht.jpg", 8, (0.46, 0.42, 1.9), (0.55, 0.45, 1.15), (0.32, 0.40, 2.6), 0.5),
    (S + "16_tanz.jpg", 6, (0.40, 0.48, 1.15), (0.60, 0.45, 1.25), None, 1.8),
]
assert sum(s[1] for s in SHOTS) * FPB == TOTAL

# ------------------------------------------------------------- Vorberechnungen
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
nx, ny = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
r2 = nx ** 2 + ny ** 2

# leichte Tonnenverzerrung (Roehren-TV)
k = 0.06
mapx = np.clip((nx * (1 + k * r2)) * (W / 2) + W / 2, 0, W - 1).astype(np.int32)
mapy = np.clip((ny * (1 + k * r2)) * (H / 2) + H / 2, 0, H - 1).astype(np.int32)

# Vignette
vign = np.clip(1.15 - 0.55 * r2 ** 1.2, 0, 1)[..., None]

# abgerundete Ecken / Bildfenster
mask_img = Image.new("L", (W, H), 0)
ImageDraw.Draw(mask_img).rounded_rectangle([14, 12, W - 15, H - 13], radius=70, fill=255)
mask = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(5)), dtype=np.float32)[..., None] / 255

# Scanlines
scan = (0.93 + 0.07 * (np.arange(H) % 2 == 0)).astype(np.float32)[:, None, None]


def load(path):
    im = Image.open(path).convert("RGB")
    s = 1300 / im.width
    return im.resize((1300, int(im.height * s)), Image.LANCZOS)


def smooth_noise(n, scale, seed):
    r = np.random.default_rng(seed)
    pts = r.standard_normal(n // scale + 3)
    x = np.arange(n) / scale
    i = x.astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    return pts[i] * (1 - f) + pts[i + 1] * f


def crop(im, cx, cy, zoom, jx=0.0, jy=0.0, rot=0.0):
    iw, ih = im.size
    cw = iw / zoom
    ch = cw * 3 / 4
    if ch > ih:                      # Querformat-Quelle: an Hoehe ausrichten
        ch = ih / zoom
        cw = ch * 4 / 3
    x = np.clip(cx * iw - cw / 2 + jx * cw, 0, iw - cw)
    y = np.clip(cy * ih - ch / 2 + jy * ch, 0, ih - ch)
    fr = im.resize((W, H), Image.BILINEAR, box=(x, y, x + cw, y + ch))
    if rot:
        fr = fr.rotate(rot, Image.BILINEAR, expand=False)
    return fr


# ------------------------------------------------------------- Titelkarte
def title_frames(n):
    frames = []
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 96)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 30)
    for i in range(n):
        im = Image.new("RGB", (W, H), (70, 22, 14))
        d = ImageDraw.Draw(im)
        # Sonnenstrahlen (Western-Plakat)
        cxp, cyp = W / 2, H * 0.62
        for a in range(0, 360, 12):
            ang = np.radians(a + i * 0.6)
            ang2 = np.radians(a + 6 + i * 0.6)
            d.polygon([(cxp, cyp), (cxp + 1400 * np.cos(ang), cyp + 1400 * np.sin(ang)),
                       (cxp + 1400 * np.cos(ang2), cyp + 1400 * np.sin(ang2))], fill=(110, 38, 18))
        d.ellipse([cxp - 120, cyp - 120, cxp + 120, cyp + 120], fill=(235, 150, 40))
        d.rectangle([0, int(H * 0.70), W, H], fill=(40, 14, 10))
        # Schrift, vertikal gestreckt
        txt = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
        td = ImageDraw.Draw(txt)
        word = "MID DAY OUT"
        tw = td.textlength(word, font=font_big) + 8 * (len(word) - 1)
        x = (W - tw) / 2
        for ch in word:
            td.text((x + 5, 25), ch, font=font_big, fill=(25, 8, 5, 255))
            td.text((x, 20), ch, font=font_big, fill=(250, 226, 170, 255))
            x += td.textlength(ch, font=font_big) + 8
        txt = txt.resize((W, 300))
        jx = int(rng.integers(-2, 3))
        jy = int(rng.integers(-2, 3))
        im.paste(txt, (jx, 120 + jy), txt)
        sub = "~  IN COLOR  ~  SUPER 8  ~"
        sw = d.textlength(sub, font=font_small)
        d.text(((W - sw) / 2 + jx, 445 + jy), sub, font=font_small, fill=(250, 226, 170))
        frames.append(im)
    return frames


# ------------------------------------------------------------- Film-Look
scratches = []          # (x, restliche frames, hell/dunkel)


def film_look(fr, fidx, shot_frame):
    a = np.asarray(fr, dtype=np.float32) / 255

    # Kodachrome-artige Farbe: warm, kontrastreich, leicht ausgebleicht
    lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    a = lum[..., None] + (a - lum[..., None]) * 1.12
    a = a * np.array([1.07, 1.0, 0.86], dtype=np.float32) + np.array([0.03, 0.015, 0.0], dtype=np.float32)
    a = np.clip(a, 0, 1)
    a = a * a * (3 - 2 * a) * 0.55 + a * 0.45          # S-Kurve
    a = 0.05 + 0.90 * a                                 # angehobene Schwarzwerte

    # Halation: helle Stellen bluten roetlich
    small = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W // 8, H // 8), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR)
    glow = np.asarray(small, dtype=np.float32) / 255
    a += np.clip(glow - 0.65, 0, 1) * np.array([0.6, 0.3, 0.15], dtype=np.float32)

    # Weichzeichnung (Super-8-Optik)
    soft = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W // 2, H // 2), Image.BILINEAR).resize((W, H), Image.BICUBIC)
    a = np.asarray(soft, dtype=np.float32) / 255

    # Belichtungsflackern
    flick = 1 + 0.05 * rng.standard_normal() + 0.03 * np.sin(fidx * 0.9)
    a *= flick

    # Splice-Blitz / Lichtleck am Schnitt
    if shot_frame == 0:
        a = a * 0.4 + np.array([1.0, 0.85, 0.6], dtype=np.float32) * 0.8
    elif shot_frame == 1:
        leak = np.clip(1.2 - (nx + 1.2) ** 2, 0, 1)[..., None] * np.array([1.0, 0.45, 0.1], dtype=np.float32)
        a += leak * 0.55

    # Korn (klumpig, helligkeitsabhaengig)
    g = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)
    a += g[..., None] * 0.055 * (0.6 + 0.4 * (1 - a))

    # Staub und Fusseln
    for _ in range(rng.integers(0, 6)):
        x, y = rng.integers(0, W), rng.integers(0, H)
        rad = int(rng.integers(1, 4))
        val = 0.95 if rng.random() < 0.5 else 0.0
        a[max(0, y - rad):y + rad, max(0, x - rad):x + rad] = val
    if rng.random() < 0.05:
        x0, y0 = rng.integers(0, W), rng.integers(0, H)
        t = np.linspace(0, 1, 120)
        px = (x0 + 60 * np.sin(t * 5 + rng.random() * 6) + t * 40).astype(int) % W
        py = (y0 + t * 140).astype(int) % H
        a[py, px] = 0.02

    # Kratzer (bleiben einige Frames stehen)
    if rng.random() < 0.08:
        scratches.append([int(rng.integers(40, W - 40)), int(rng.integers(4, 20)), rng.random() < 0.7])
    for s in scratches:
        s[0] += int(rng.integers(-2, 3))
        x = s[0] % W
        a[:, x:x + 2] = a[:, x:x + 2] * 0.5 + (0.6 if s[2] else 0.0)
        s[1] -= 1
    scratches[:] = [s for s in scratches if s[1] > 0]

    a *= vign

    # TV: Farbsaum, Tonnenverzerrung, Scanlines, runde Ecken
    a[..., 0] = np.roll(a[..., 0], 2, axis=1)
    a[..., 2] = np.roll(a[..., 2], -2, axis=1)
    a = a[mapy, mapx]
    a *= scan
    a = a * mask + 0.015 * (1 - mask)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------- Rendern
def main():
    out = sys.stdout.buffer
    fidx = 0
    weave_x = smooth_noise(TOTAL, 3, 1) * 0.004
    weave_y = smooth_noise(TOTAL, 2, 2) * 0.006
    for si, (path, beats, p0, p1, jump, shake) in enumerate(SHOTS):
        n = beats * FPB
        if path == "TITLE":
            frames = title_frames(n)
            for i, fr in enumerate(frames):
                out.write(film_look(fr, fidx, i if i > 0 else 0).tobytes())
                fidx += 1
            continue
        im = load(path)
        hx = smooth_noise(n, 7, 100 + si) * 0.012 * shake
        hy = smooth_noise(n, 9, 200 + si) * 0.012 * shake
        hr = smooth_noise(n, 11, 300 + si) * 0.5 * shake
        cut_at = n // 2 if jump else n + 1
        for i in range(n):
            # Ruckeln am Anfang jedes Shots (Kamera wird hochgerissen)
            settle = np.exp(-i / 3.0) * 0.02 * shake
            if i < cut_at:
                t = i / max(cut_at - 1, 1) if jump else i / (n - 1)
                t = t * t * (3 - 2 * t) * 0.6 + t * 0.4
                cx = p0[0] + (p1[0] - p0[0]) * t
                cy = p0[1] + (p1[1] - p0[1]) * t
                z = p0[2] + (p1[2] - p0[2]) * t
                sf = i
            else:
                t = (i - cut_at) / (n - cut_at)
                cx, cy = jump[0], jump[1]
                z = jump[2] * (1 + 0.08 * t)       # Crash-Zoom-Jumpcut
                sf = i - cut_at
            fr = crop(im, cx, cy, z,
                      hx[i] + weave_x[fidx] + settle, hy[i] + weave_y[fidx] - settle, hr[i])
            out.write(film_look(fr, fidx, sf if sf < 2 else 99).tobytes())
            fidx += 1
        print(f"shot {si} done ({fidx}/{TOTAL})", file=sys.stderr)
    assert fidx == TOTAL


if __name__ == "__main__":
    main()
