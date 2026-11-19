"""Video "mid day out II": dunkle Welt, ein Lichtkegel sucht durch die Bilder.

25 fps (fluessig), 100 BPM -> 15 Frames/Beat, 60 Frames/Takt, 7500 Frames = 300 s.
Super-8/Roehren-TV-Look wie Teil 1, aber weiche Kamerafahrten statt Ruckeln.
Rendert parallel und schreibt rohe RGB-Frames auf stdout.
"""
import sys
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 960, 720
FPS = 25
FPBAR = 60
TOTAL = 7500
S = "bilder/"

# keys: (t, cx, cy, zoom, rot)   light: (t, lx, ly) in Bildkoordinaten 0..1
SHOTS = [
    dict(f="TITLE", bars=4),
    # --- dunkle Welt (Intro/Puls)
    dict(f=S + "22_sperrbildschirm.png", bars=4, keys=[(0, .5, .62, 1.6, 0), (1, .5, .32, 1.3, 0)],
         light=[(0, .5, .7), (1, .5, .25)], rad=.25, pixel=True),
    dict(f=S + "19_nachtflug.jpg", bars=4, keys=[(0, .45, .35, 1.4, -6), (1, .5, .55, 1.7, 6)],
         light=[(0, .3, .4), (1, .6, .55)], rad=.25),
    dict(f=S + "01_platine.jpg", bars=4, keys=[(0, .35, .45, 1.6, 0), (1, .65, .50, 2.2, 0)],
         light=[(0, .2, .5), (.5, .5, .3), (1, .65, .5)], rad=.22, tr="pix"),
    dict(f=S + "18_apple_board.jpg", bars=4, keys=[(0, .30, .30, 1.8, 0), (1, .60, .55, 1.5, 0)],
         light=[(0, .35, .3), (.5, .55, .35), (1, .6, .6)], rad=.24, tr="pix"),
    dict(f=S + "02_schuhe_aufzug.jpg", bars=3, keys=[(0, .5, .40, 1.0, 0), (1, .47, .75, 1.3, 0)],
         light=[(0, .5, .2), (1, .4, .75)], rad=.28),
    dict(f=S + "03_beine.jpg", bars=3, keys=[(0, .5, .75, 1.0, 0), (1, .52, .32, 1.35, 0)],
         light=[(0, .6, .8), (1, .5, .35)], rad=.3),
    dict(f=S + "04_aktie.png", bars=3, keys=[(0, .22, .47, 2.1, 0), (1, .78, .41, 2.1, 0)],
         light=[(0, .3, .55), (1, .7, .4)], rad=.3, pixel=True, tr="pix"),
    dict(f=S + "05_profile.jpg", bars=3, keys=[(0, .5, .5, 1.25, -10), (1, .5, .5, 1.7, 8)],
         light=[(0, .15, .5), (1, .85, .5)], rad=.25),
    # --- Drive / Lead
    dict(f=S + "06_bruecke.jpg", bars=4, keys=[(0, .45, .42, 1.0, 0), (1, .62, .45, 1.8, 0)],
         light=[(0, .7, .45), (1, .5, .5)], rad=.28, tr="pix"),
    dict(f=S + "07_turm_hecke.jpg", bars=4, keys=[(0, .55, .78, 1.0, 0), (1, .55, .30, 1.05, 0)],
         light=[(0, .4, .8), (1, .55, .3)], rad=.32),
    dict(f=S + "08_turm.jpg", bars=4, keys=[(0, .5, .65, 1.35, 0), (1, .5, .25, 1.0, 0)],
         light=[(0, .5, .7), (1, .5, .2)], rad=.33, tr="pix"),
    dict(f=S + "09_kunst_blau.jpg", bars=4, keys=[(0, .53, .50, 1.7, 0), (1, .53, .45, 1.0, 0)],
         light=[(0, .5, .5), (1, .55, .45)], rad=.3),
    dict(f=S + "10_leuchtkaesten.jpg", bars=4, keys=[(0, .28, .47, 1.8, 0), (1, .73, .48, 1.8, 0)],
         light=[(0, .5, .5), (1, .5, .5)], rad=.35, tr="pix"),
    dict(f=S + "12_spiegelraum.jpg", bars=4, keys=[(0, .5, .5, 1.3, 0), (1, .5, .5, 1.7, 28)],
         light=[(0, .5, .6), (1, .5, .5)], rad=.33),
    # --- Breakdown: Licht im Dunkeln
    dict(f=S + "11_meer.jpg", bars=4, keys=[(0, .25, .50, 1.8, 0), (1, .75, .50, 1.8, 0)],
         light=[(0, .3, .45), (1, .7, .45)], rad=.33),
    dict(f=S + "17_globus.jpg", bars=4, keys=[(0, .50, .45, 1.6, 0), (1, .45, .40, 1.15, -12)],
         light=[(0, .5, .45), (1, .45, .4)], rad=.38),
    # --- Aufbau
    dict(f=S + "25_kirche.jpg", bars=4, keys=[(0, .45, .80, 1.3, 0), (1, .40, .55, 1.6, 0)],
         light=[(0, .35, .8), (1, .3, .55)], rad=.3, tr="pix"),
    dict(f=S + "24_wald.jpg", bars=4, keys=[(0, .35, .70, 1.0, 0), (1, .55, .45, 1.4, 0)],
         light=[(0, .3, .7), (1, .6, .4)], rad=.33),
    dict(f=S + "13_galerie_mann.jpg", bars=4, keys=[(0, .5, .55, 1.0, 0), (1, .46, .43, 2.0, 0)],
         light=[(0, .45, .45), (1, .5, .5)], rad=.36, tr="pix"),
    dict(f=S + "14_dachterrasse.jpg", bars=4, keys=[(0, .28, .70, 1.4, 0), (1, .55, .50, 1.0, 0)],
         light=[(0, .3, .6), (1, .55, .5)], rad=.4),
    # --- Hoehepunkt
    dict(f=S + "15_dom_nacht.jpg", bars=4, keys=[(0, .5, .70, 1.0, 0), (1, .32, .42, 2.2, 0)],
         light=[(0, .5, .6), (1, .5, .45)], rad=.45),
    dict(f=S + "16_selfie_nacht.jpg", bars=4, keys=[(0, .5, .45, 1.0, 0), (1, .5, .42, 1.2, 0)],
         light=[(0, .5, .5), (1, .5, .5)], rad=.55, tr="pix"),
    dict(f="FACES", faces=[S + "f1.jpg", S + "f2.jpg", S + "f3.jpg", S + "f4.jpg", S + "f5.jpg"], bars=8,
         keys=[(0, .5, .45, 1.0, 0), (1, .5, .45, 1.25, 0)], light=[(0, .5, .45), (1, .5, .45)], rad=.6),
    # --- Licht gefunden
    dict(f=S + "21_flughafen.jpg", bars=6, keys=[(0, .45, .62, 1.0, 0), (1, .45, .42, 1.5, 0)],
         light=[(0, .45, .45), (1, .45, .42)], rad=.55),
    dict(f=S + "20_fluegel.jpg", bars=6, keys=[(0, .45, .60, 1.0, 0), (1, .55, .32, 1.4, 0)],
         light=[(0, .4, .4), (1, .55, .35)], rad=.6),
    dict(f=S + "23_berg.jpg", bars=5, keys=[(0, .5, .60, 1.0, 0), (1, .5, .40, 1.5, 0)],
         light=[(0, .5, .4), (1, .5, .35)], rad=.65),
    dict(f=S + "26_meer_sonne.jpg", bars=6, keys=[(0, .45, .60, 1.0, 0), (1, .35, .40, 1.5, 0)],
         light=[(0, .35, .4), (1, .3, .38)], rad=.65),
    dict(f=S + "27_aussicht.jpg", bars=6, keys=[(0, .5, .30, 1.4, 0), (1, .5, .60, 1.0, 0)],
         light=[(0, .5, .4), (1, .5, .5)], rad=.75),
]
start = 0
for s in SHOTS:
    s["start"] = start
    s["n"] = s["bars"] * FPBAR
    start += s["n"]
assert start == TOTAL, start

# ------------------------------------------------------------------ Vorberechnungen
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
nx, ny = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
r2 = nx ** 2 + ny ** 2
k = 0.06
MAPX = np.clip((nx * (1 + k * r2)) * (W / 2) + W / 2, 0, W - 1).astype(np.int32)
MAPY = np.clip((ny * (1 + k * r2)) * (H / 2) + H / 2, 0, H - 1).astype(np.int32)
VIGN = np.clip(1.15 - 0.6 * r2 ** 1.2, 0, 1)[..., None]
_m = Image.new("L", (W, H), 0)
ImageDraw.Draw(_m).rounded_rectangle([14, 12, W - 15, H - 13], radius=70, fill=255)
MASK = np.asarray(_m.filter(ImageFilter.GaussianBlur(5)), dtype=np.float32)[..., None] / 255
SCAN = (0.92 + 0.08 * (np.arange(H) % 2 == 0)).astype(np.float32)[:, None, None]
XN, YN = xx / W, yy / H

prng = np.random.default_rng(42)
NP = 160
P0 = prng.uniform(0, 1, (NP, 2))
PV = prng.uniform(-0.004, 0.004, (NP, 2)) + np.array([0.001, -0.002])
PPH = prng.uniform(0, 6.28, NP)

_cache = {}


def load(path):
    if path not in _cache:
        im = Image.open(path).convert("RGB")
        s = 1400 / max(im.size)
        _cache[path] = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return _cache[path]


def hermite(keys, t, idx):
    """Weiche Interpolation (Catmull-Rom) einer Spalte idx ueber Keyframes."""
    ts = [kf[0] for kf in keys]
    vs = [kf[idx] for kf in keys]
    if len(keys) == 1:
        return vs[0]
    i = max(0, min(np.searchsorted(ts, t, side="right") - 1, len(ts) - 2))
    t0, t1 = ts[i], ts[i + 1]
    u = (t - t0) / (t1 - t0)
    def tan(j):
        a, b = max(j - 1, 0), min(j + 1, len(ts) - 1)
        return (vs[b] - vs[a]) / (ts[b] - ts[a])
    m0, m1 = tan(i) * (t1 - t0), tan(i + 1) * (t1 - t0)
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
    return h00 * vs[i] + h10 * m0 + h01 * vs[i + 1] + h11 * m1


def snoise(x, seed):
    """Glattes 1D-Rauschen."""
    r = np.random.default_rng(seed)
    pts = r.standard_normal(64)
    i = int(np.floor(x)) % 63
    f = x - np.floor(x)
    f = f * f * (3 - 2 * f)
    return pts[i] * (1 - f) + pts[i + 1] * f


def crop(im, cx, cy, zoom, rot):
    iw, ih = im.size
    cw = iw / zoom
    ch = cw * 3 / 4
    if ch > ih:
        ch = ih / zoom
        cw = ch * 4 / 3
    pad = 1.3 if rot else 1.0
    cw2, ch2 = min(cw * pad, iw), min(ch * pad, ih)
    x = np.clip(cx * iw - cw2 / 2, 0, iw - cw2)
    y = np.clip(cy * ih - ch2 / 2, 0, ih - ch2)
    ow, oh = int(round(W * cw2 / cw)), int(round(H * ch2 / ch))
    fr = im.resize((ow, oh), Image.BILINEAR, box=(x, y, x + cw2, y + ch2))
    if rot:
        fr = fr.rotate(rot, Image.BILINEAR)
        l, t = (ow - W) // 2, (oh - H) // 2
        fr = fr.crop((l, t, l + W, t + H))
    return np.asarray(fr, dtype=np.float32) / 255


def pixelate(a, block):
    if block <= 1:
        return a
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    small = img.resize((max(1, W // block), max(1, H // block)), Image.BILINEAR)
    return np.asarray(small.resize((W, H), Image.NEAREST), dtype=np.float32) / 255


# ------------------------------------------------------------------ Titel (Pixel-Schrift)
def pixel_text(text, size, scale):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size)
    tmp = Image.new("L", (1, 1))
    w = int(ImageDraw.Draw(tmp).textlength(text, font=font)) + 4
    im = Image.new("L", (w, size + 6), 0)
    ImageDraw.Draw(im).text((2, 0), text, font=font, fill=255)
    im = im.point(lambda v: 255 if v > 110 else 0)
    return im.resize((w * scale, (size + 6) * scale), Image.NEAREST)


TITLE_IMG = None


def title_base():
    global TITLE_IMG
    if TITLE_IMG is None:
        base = Image.new("L", (W, H), 0)
        t1 = pixel_text("MID DAY OUT", 22, 6)
        base.paste(t1, ((W - t1.width) // 2, 200))
        t2 = pixel_text("II - AUF DER SUCHE NACH DEM LICHT", 11, 3)
        base.paste(t2, ((W - t2.width) // 2, 400))
        TITLE_IMG = np.asarray(base, dtype=np.float32) / 255
    return TITLE_IMG


def press_start():
    t3 = pixel_text("PRESS START", 11, 3)
    a = np.zeros((H, W), np.float32)
    arr = np.asarray(t3, dtype=np.float32) / 255
    y0, x0 = 560, (W - t3.width) // 2
    a[y0:y0 + arr.shape[0], x0:x0 + arr.shape[1]] = arr
    return a


PS = None


# ------------------------------------------------------------------ Frame
def render(i):
    global PS
    T = i / TOTAL
    shot = next(s for s in reversed(SHOTS) if s["start"] <= i)
    li = i - shot["start"]
    n = shot["n"]
    t = li / (n - 1)
    rng = np.random.default_rng(i)
    beat_phase = (i % 15) / 15.0

    amb = 0.10 + 0.55 * T ** 1.4            # dunkle Welt -> heller
    sat = 0.25 + 0.75 * T                   # fast monochrom -> Farbe
    warmth = 0.3 + 0.7 * T

    if shot["f"] == "TITLE":
        base = title_base()
        lx = -0.2 + 1.4 * t
        ly = 0.42 + 0.08 * np.sin(t * 6)
        m = np.exp(-((XN - lx) ** 2 * 1.8 + (YN - ly) ** 2) / (2 * 0.16 ** 2))
        glow = np.asarray(Image.fromarray((base * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(8)), np.float32) / 255
        lum = base * (0.08 + 0.92 * m) + glow * (0.15 + 0.6 * m)
        if PS is None:
            PS = press_start()
        if li > 120 and (li // 15) % 2 == 0:
            lum = lum + PS * 0.7
        a = lum[..., None] * np.array([1.0, 0.92, 0.75], np.float32)
        a += m[..., None] * 0.05 * np.array([0.6, 0.8, 1.0], np.float32)
        lightm = m
    else:
        if shot["f"] == "FACES":
            seq = [0, 1, 2, 3, 4, 3, 2, 1]
            im = load(shot["faces"][seq[(li // 15) % len(seq)]])   # Stop-Motion im Beat
        else:
            im = load(shot["f"])
        keys = shot["keys"]
        cx, cy, z, rot = (hermite(keys, t, j) for j in (1, 2, 3, 4))
        # sanftes Handkamera-Schweben
        cx += 0.006 * snoise(i / 40, 1)
        cy += 0.006 * snoise(i / 45, 2)
        rot += 0.4 * snoise(i / 60, 3)
        a = crop(im, cx, cy, z, rot)
        if shot.get("pixel"):
            a = pixelate(a, 6)
            a = np.round(a * 5) / 5            # 8-Bit-Palette
        # Lichtkegel
        lx = hermite(shot["light"], t, 1) + 0.05 * snoise(i / 30, 7)
        ly = hermite(shot["light"], t, 2) + 0.05 * snoise(i / 33, 8)
        rad = shot["rad"] * (1 + 0.06 * snoise(i / 20, 9))
        lightm = np.exp(-((XN - lx) ** 2 * 1.78 + (YN - ly) ** 2) / (2 * rad ** 2))
        # Taschenlampen-Flackern in der dunklen ersten Haelfte
        torch = 1.0
        if T < 0.5 and snoise(i / 6, 11) > 1.1:
            torch = 0.55
        m = amb + (1 - amb) * lightm * torch
        lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
        a = lum[..., None] + (a - lum[..., None]) * sat
        cold = a * np.array([0.80, 0.95, 1.15], np.float32)
        warm = a * np.array([1.15, 1.0, 0.80], np.float32)
        w = (lightm * warmth)[..., None]
        a = (cold * (1 - w) + warm * w) * m[..., None]
        a = np.clip(a, 0, 1)
        a = a * a * (3 - 2 * a) * 0.5 + a * 0.5

    # Staub im Licht
    pos = (P0 + PV * i + 0.01 * np.stack([np.sin(i / 50 + PPH), np.cos(i / 63 + PPH)], 1)) % 1.0
    px = (pos[:, 0] * (W - 2)).astype(int)
    py = (pos[:, 1] * (H - 2)).astype(int)
    lv = lightm[py, px] * 0.5
    for dy in (0, 1):
        for dx in (0, 1):
            a[py + dy, px + dx] += lv[:, None] * np.array([1.0, 0.95, 0.85], np.float32)

    # Puls auf den Beat (nach dem Intro)
    if i > 8 * FPBAR and not (56 * FPBAR <= i < 64 * FPBAR):
        a *= 1 + 0.05 * np.exp(-beat_phase * 6)

    # Ende: Licht gefunden -> Bloom, dann Abblende
    if i > TOTAL - 4 * FPBAR:
        e = (i - (TOTAL - 4 * FPBAR)) / (4 * FPBAR)
        bloom = np.clip(e * 1.6, 0, 1) ** 2
        a = a + bloom * 0.6 * np.array([1.0, 0.9, 0.75], np.float32)
        if e > 0.7:
            a *= 1 - (e - 0.7) / 0.3

    # Uebergaenge
    tr_in = shot.get("tr", "dip")
    nxt = next((s for s in SHOTS if s["start"] == shot["start"] + n), None)
    tr_out = nxt.get("tr", "dip") if nxt else "none"
    F = 18
    if li < F and shot["start"] > 0:
        if tr_in == "pix":
            a = pixelate(a, int(1 + 30 * (1 - li / F) ** 2))
        else:
            a *= (li / F) ** 1.5
    if n - li <= F and tr_out != "none":
        u = (n - li) / F
        if tr_out == "pix":
            a = pixelate(a, int(1 + 30 * (1 - u) ** 2))
        else:
            a *= u ** 1.5

    # ---------------- Film-Look
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    small = img.resize((W // 8, H // 8), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR)
    glow = np.asarray(small, np.float32) / 255
    a = a + np.clip(glow - 0.45, 0, 1) * np.array([0.7, 0.35, 0.2], np.float32)
    soft = img.resize((W // 2, H // 2), Image.BILINEAR).resize((W, H), Image.BICUBIC)
    a = 0.6 * (np.asarray(soft, np.float32) / 255) + 0.4 * np.clip(a, 0, 1) + (a - np.clip(a, 0, 1))
    a *= 1 + 0.025 * rng.standard_normal()
    a += 0.02                                               # angehobene Schwarzwerte (Film)
    g = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)
    a += g[..., None] * 0.045
    for _ in range(rng.integers(0, 4)):
        x, y = rng.integers(0, W), rng.integers(0, H)
        rd = int(rng.integers(1, 3))
        a[max(0, y - rd):y + rd, max(0, x - rd):x + rd] = 0.8 if rng.random() < 0.5 else 0.0
    for j in range(max(0, i - 25), i + 1):                  # deterministische Kratzer
        rj = np.random.default_rng(100000 + j)
        if rj.random() < 0.04:
            dur = int(rj.integers(5, 25))
            if i - j < dur:
                x = int(rj.integers(40, W - 40)) + int(3 * np.sin((i - j) * 0.7))
                a[:, x:x + 2] = a[:, x:x + 2] * 0.5 + 0.35
    a *= VIGN
    a[..., 0] = np.roll(a[..., 0], 2, axis=1)
    a[..., 2] = np.roll(a[..., 2], -2, axis=1)
    a = a[MAPY, MAPX]
    a *= SCAN
    a = a * MASK + 0.01 * (1 - MASK)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes()


def main():
    frames = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(TOTAL)
    out = sys.stdout.buffer
    with Pool(4) as p:
        for k, buf in enumerate(p.imap(render, frames, chunksize=6)):
            out.write(buf)
            if k % 250 == 0:
                print(f"frame {frames[k]}/{TOTAL}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
