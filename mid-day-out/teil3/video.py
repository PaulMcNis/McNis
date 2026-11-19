"""Video "mid day out III - Deus Ex Machina".

Schwarz-Weiss, 28 fps (140 BPM -> 12 Frames/Beat, 48/Takt), 175 Takte = 8400 Frames = 300 s.
Menschen werden animiert:
  * Gesichter per Landmarken-Warp: Blinzeln, Laecheln/Trauer, Augenbrauen, Mund im Beat,
    fliessende Morphs zwischen den Selfie-Ausdruecken
  * Personen per Segmentierung freigestellt: eigene Ebene (2.5D-Parallaxe), Atmen, Schwanken
  * Android-HUD: Gesichts-Gitter, Ziel-Rahmen, Systemtexte
Startet mit der venv-Python (cv2). Schreibt rohe Graustufen-Frames (gray) auf stdout.
"""
import sys
import numpy as np
import cv2
from multiprocessing import Pool

cv2.setNumThreads(1)
W, H = 960, 720
FPS, FPB, FPBAR = 28, 12, 48
BARS = 175
TOTAL = BARS * FPBAR
S = "bilder/"
EDGES = np.load("build/mesh_edges.npy")
CONTOURS = np.load("build/mesh_contours.npy")


# =================================================================== Musik-Abschnitte (wie music.py)
def sec(bar):
    for name, a, b in (("boot", 0, 8), ("wake", 8, 24), ("travel", 24, 48), ("search", 48, 80), ("dive", 80, 104),
                       ("found", 104, 120), ("joy", 120, 130), ("sad", 130, 140), ("love", 140, 152),
                       ("finale", 152, 171), ("end", 171, 175)):
        if a <= bar < b:
            return name
    return "end"


ENERGY = {"boot": 0.1, "wake": 0.4, "travel": 0.7, "search": 0.9, "dive": 0.3, "found": 0.8, "joy": 0.8,
          "sad": 0.2, "love": 0.6, "finale": 1.0, "end": 0.1}

# =================================================================== Shotliste
# keys: (t, cx, cy, zoom, rot) ; hud: [(beat, text, x, y, scale)]
SHOTS = [
    dict(kind="boot", bars=4, f=S + "18_apple_board.jpg", keys=[(0, .5, .5, 1.3, 0), (1, .5, .5, 1.5, 0)]),
    dict(kind="title", bars=4, f=S + "01_platine.jpg", keys=[(0, .4, .5, 1.5, 0), (1, .6, .5, 1.9, 0)]),
    dict(f=S + "27_aussicht", bars=8, person=True, par=1.0, keys=[(0, .60, .63, 3.2, 0), (.5, .58, .68, 2.0, 0), (1, .5, .60, 1.2, 0)],
         hud=[(4, "UNIT M-01  //  ONLINE", 40, 60, 1.0), (10, "BIOMETRIE ......... OK", 40, 95, .7),
              (14, "EMOTION-CHIP ...... OFFLINE", 40, 120, .7), (20, "ZIEL: SUBJEKT F", 40, 145, .7)], lock=True, scan=True),
    dict(f=S + "22_sperrbildschirm.png", bars=4, keys=[(0, .5, .2, 1.2, 0), (1, .5, .45, 1.6, 0)],
         hud=[(2, "MISSION 14:39", 330, 330, 1.4), (8, "> SUBJEKT F FINDEN", 300, 400, .9)], pixel=True),
    dict(f=S + "02_schuhe_aufzug.jpg", bars=2, keys=[(0, .5, .45, 1.0, 0), (1, .47, .72, 1.3, 0)], walk=True),
    dict(f=S + "03_beine.jpg", bars=2, keys=[(0, .5, .70, 1.0, 0), (1, .52, .35, 1.3, 0)], walk=True),
    dict(f=S + "21_flughafen.jpg", bars=6, keys=[(0, .45, .62, 1.0, 0), (1, .45, .42, 1.6, 0)], flare=(.45, .44),
         hud=[(4, "GATE  //  BOARDING", 40, 60, .9), (12, "VERBINDUNG HERGESTELLT", 40, 95, .7)]),
    dict(f=S + "17_globus.jpg", bars=4, keys=[(0, .5, .45, 1.15, -8), (1, .5, .45, 1.4, 4)], route=True,
         hud=[(2, "ROUTE BERECHNET", 40, 60, .9)]),
    dict(f=S + "20_fluegel.jpg", bars=6, keys=[(0, .5, .55, 1.0, 0), (1, .55, .35, 1.4, 0)], bank=True,
         hud=[(4, "FLUGHOEHE 11.200 M", 40, 60, .8)]),
    dict(f=S + "19_nachtflug.jpg", bars=4, keys=[(0, .5, .45, 1.2, 0), (1, .5, .45, 2.2, 0)], speed=True),
    dict(f=S + "23_berg.jpg", bars=4, keys=[(0, .5, .60, 1.0, 0), (1, .5, .40, 1.4, 0)],
         hud=[(2, "LANDUNG  //  SEKTOR 7", 40, 60, .9)]),
    dict(f=S + "25_kirche.jpg", bars=4, keys=[(0, .45, .80, 1.3, 0), (1, .40, .55, 1.6, 0)], torch=True,
         hud=[(2, "SCAN ... KEIN TREFFER", 40, 60, .8)]),
    dict(f=S + "09_kunst_blau.jpg", bars=4, keys=[(0, .53, .50, 1.6, 0), (1, .53, .45, 1.0, 0)], torch=True, grid=True),
    dict(f=S + "10_leuchtkaesten.jpg", bars=4, keys=[(0, .28, .47, 1.8, 0), (1, .73, .48, 1.8, 0)], torch=True,
         hud=[(2, "RUINE 03 ... KEIN TREFFER", 40, 60, .8)]),
    dict(f=S + "13_galerie_mann", bars=4, person=True, par=1.4, keys=[(0, .5, .58, 1.0, 0), (1, .48, .48, 1.7, 0)], lock=True,
         hud=[(2, "M-01 SUCHT", 40, 60, .9)]),
    dict(f=S + "08_turm.jpg", bars=4, keys=[(0, .5, .70, 1.35, 0), (1, .5, .25, 1.0, 0)], torch=True),
    dict(f=S + "06_bruecke.jpg", bars=4, keys=[(0, .45, .42, 1.0, 0), (1, .62, .45, 1.8, 0)], torch=True, grid=True,
         hud=[(2, "SIGNAL SCHWACH", 40, 60, .8)]),
    dict(f=S + "24_wald.jpg", bars=4, keys=[(0, .35, .70, 1.0, 0), (1, .55, .45, 1.4, 0)], torch=True),
    dict(f=S + "26_meer_sonne.jpg", bars=4, keys=[(0, .45, .55, 1.0, 0), (1, .5, .85, 1.6, 0)],
         hud=[(4, "SIGNAL UNTER WASSER", 40, 60, .9), (12, "TAUCHGANG EINLEITEN", 40, 95, .8)]),
    dict(f=S + "11_meer.jpg", bars=8, keys=[(0, .5, .35, 1.7, 0), (1, .5, .85, 2.4, 0)], under=1,
         hud=[(4, "TIEFE 012 M", 40, 60, .8), (16, "TIEFE 040 M", 40, 60, .8)]),
    dict(f=S + "01_platine.jpg", bars=8, keys=[(0, .35, .45, 1.6, 0), (1, .65, .50, 2.3, 8)], under=2,
         hud=[(4, "WRACK GEFUNDEN", 40, 60, .8), (16, "TIEFE 090 M", 40, 95, .8)]),
    dict(f=S + "26_meer_sonne.jpg", bars=8, flipv=True, keys=[(0, .4, .75, 1.6, 0), (1, .35, .55, 1.0, 0)], under=3,
         hud=[(4, "AUFSTIEG", 40, 60, .8), (20, "LICHT ERKANNT", 40, 95, .8)]),
    dict(f=S + "14_dachterrasse", bars=8, person=True, par=1.2, keys=[(0, .45, .62, 1.0, 0), (1, .64, .55, 2.0, 0)], lock=True,
         found=True, hud=[(3, "SUBJEKT F", 40, 60, 1.1), (6, "UEBEREINSTIMMUNG 99,7 %", 40, 100, .8)]),
    dict(f=S + "12_spiegelraum", bars=4, person=True, par=0.6, keys=[(0, .5, .5, 1.3, 0), (1, .5, .5, 1.7, 25)],
         hud=[(2, "M-01 + F", 40, 60, 1.0)]),
    dict(f=S + "15_dom_nacht.jpg", bars=4, keys=[(0, .5, .70, 1.0, 0), (1, .32, .42, 2.0, 0)]),
    # Emotionen: Morph-Sequenzen (photo, smile, brow, beat-zeit relativ)
    dict(kind="morph", bars=10, label="GLUECK.EXE", keys=[(0, .5, .52, 1.0, 0), (1, .5, .48, 1.15, 0)],
         seq=[("f5", .3, .6, 0), ("f5", .8, .3, 6), ("f4", .9, .8, 10), ("f4", 1.0, .2, 16),
              ("16_selfie_nacht", .6, 0, 20), ("16_selfie_nacht", 1.0, .3, 28), ("f5", 1.0, .9, 34), ("f3", .9, .4, 40)]),
    dict(kind="morph", bars=10, label="TRAUER.EXE", rain=True, keys=[(0, .5, .50, 1.12, 0), (1, .5, .52, 1.0, 0)],
         seq=[("f3", .2, 0, 0), ("f1", -.6, -.3, 8), ("f1", -1.0, -.6, 16), ("f2", -.9, -.2, 24), ("f2", -1.0, -.7, 32), ("f1", -.8, -.4, 40)]),
    dict(kind="morph", bars=12, label="LIEBE.EXE", link=True, keys=[(0, .5, .52, 1.0, 0), (1, .5, .48, 1.2, 0)],
         seq=[("f1", -.4, 0, 0), ("f3", .4, .2, 8), ("f5", .9, .5, 16), ("16_selfie_nacht", .8, .2, 24),
              ("16_selfie_nacht", 1.0, .4, 36), ("16_selfie_nacht", .9, .2, 48)]),
]
FINALE = ["27_aussicht", "21_flughafen.jpg", "20_fluegel.jpg", "09_kunst_blau.jpg", "11_meer.jpg", "01_platine.jpg",
          "14_dachterrasse", "17_globus.jpg", "23_berg.jpg", "24_wald.jpg", "10_leuchtkaesten.jpg", "15_dom_nacht.jpg"]
for k, fn in enumerate(FINALE):
    pers = fn in ("27_aussicht", "14_dachterrasse")
    SHOTS.append(dict(f=S + fn, bars=1, person=pers, par=1.0, keys=[(0, .5, .5, 1.6, (-4) ** (k % 2)), (1, .5, .5, 1.1, 0)], glitchy=True))
SHOTS += [
    dict(kind="morph", bars=4, label="", keys=[(0, .5, .42, 1.0, 0), (1, .5, .42, 1.2, 0)],
         seq=[("16_selfie_nacht", 1.0, .3, 0), ("16_selfie_nacht", .7, .1, 16)]),
    dict(f=S + "27_aussicht", bars=3, person=True, par=1.0, keys=[(0, .5, .60, 1.2, 0), (1, .5, .55, 1.0, 0)]),
    dict(kind="end", bars=4),
]
st = 0
for s in SHOTS:
    s["start"] = st
    s["n"] = s["bars"] * FPBAR
    st += s["n"]
assert st == TOTAL, st


# =================================================================== Daten laden
_cache = {}


def resolve(name):
    import os
    base = name if name.startswith(S) else S + name
    for ext in ("", ".jpg", ".png"):
        if os.path.exists(base + ext):
            return base + ext
    raise FileNotFoundError(name)


def load(name, flipv=False):
    key = (name, flipv)
    if key in _cache:
        return _cache[key]
    import os
    path = resolve(name)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    s = 1400 / max(img.shape)
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    stem = os.path.splitext(os.path.basename(path))[0]
    an = np.load(f"build/analysis/{stem}.npz")
    person = cv2.resize(an["person"].astype(np.float32).squeeze(), (img.shape[1], img.shape[0]))
    person = np.clip((person - 0.35) / 0.4, 0, 1)
    faces = [f[:, :2] * np.array([img.shape[1], img.shape[0]], np.float32) for f in an["faces"]]
    faces.sort(key=lambda f: f[:, 0].mean())
    if flipv:
        img = img[::-1].copy(); person = person[::-1].copy()
    d = dict(img=img, person=person, faces=faces, bg=None)
    _cache[key] = d
    return d


def inpainted(d):
    if d["bg"] is None:
        img = d["img"]
        m = (cv2.dilate((d["person"] > 0.2).astype(np.uint8), np.ones((25, 25), np.uint8)) * 255)
        small = cv2.resize((img * 255).astype(np.uint8), None, fx=0.5, fy=0.5)
        ms = cv2.resize(m, (small.shape[1], small.shape[0]))
        fill = cv2.inpaint(small, ms, 9, cv2.INPAINT_TELEA)
        fill = cv2.resize(fill, (img.shape[1], img.shape[0])).astype(np.float32) / 255
        mf = cv2.GaussianBlur(m.astype(np.float32) / 255, (31, 31), 0)
        d["bg"] = img * (1 - mf) + fill * mf
    return d["bg"]


# =================================================================== Hilfen
def hermite(keys, t, idx):
    ts = [k[0] for k in keys]; vs = [k[idx] for k in keys]
    if len(keys) == 1:
        return vs[0]
    i = max(0, min(int(np.searchsorted(ts, t, side="right")) - 1, len(ts) - 2))
    t0, t1 = ts[i], ts[i + 1]
    u = np.clip((t - t0) / (t1 - t0), 0, 1)
    def tan(j):
        a, b = max(j - 1, 0), min(j + 1, len(ts) - 1)
        return (vs[b] - vs[a]) / (ts[b] - ts[a])
    m0, m1 = tan(i) * (t1 - t0), tan(i + 1) * (t1 - t0)
    return ((2 * u ** 3 - 3 * u ** 2 + 1) * vs[i] + (u ** 3 - 2 * u ** 2 + u) * m0
            + (-2 * u ** 3 + 3 * u ** 2) * vs[i + 1] + (u ** 3 - u ** 2) * m1)


def snoise(x, seed):
    pts = np.random.default_rng(seed).standard_normal(256)
    i = int(np.floor(x)) % 255
    f = x - np.floor(x); f = f * f * (3 - 2 * f)
    return pts[i] * (1 - f) + pts[i + 1] * f


def cam_matrix(iw, ih, cx, cy, zoom, rot, extra_scale=1.0):
    """Affine Matrix Quelle -> Ausgabe (4:3-Ausschnitt um cx,cy)."""
    cw = iw / zoom
    if cw * 3 / 4 > ih:
        cw = ih / zoom * 4 / 3
    sc = W / cw * extra_scale
    # Mittelpunkt so begrenzen, dass kein Rand sichtbar wird
    hw, hh = cw / 2 / extra_scale, cw * 3 / 8 / extra_scale
    px = np.clip(cx * iw, hw, iw - hw) if iw > 2 * hw else iw / 2
    py = np.clip(cy * ih, hh, ih - hh) if ih > 2 * hh else ih / 2
    M = cv2.getRotationMatrix2D((float(px), float(py)), float(rot), float(sc))
    M[0, 2] += W / 2 - px
    M[1, 2] += H / 2 - py
    return M


def apply_M(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


# =================================================================== Gesichts-Animation
UPPER = [159, 158, 160, 161, 157, 386, 385, 387, 384, 388]
LOWER = [145, 153, 144, 163, 154, 374, 380, 373, 390, 381]
BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
LIPLOW = [14, 17, 87, 317, 84, 314, 178, 402]
JAW = [152, 148, 377, 176, 400]
DENSE = np.arange(0, 468, 3)


def expression(lm, smile=0.0, brow=0.0, blink=0.0, mouth=0.0):
    """Verschiebt Landmarken: gibt Ziel-Landmarken zurueck."""
    out = lm.copy()
    mw = np.linalg.norm(lm[291] - lm[61]) + 1e-3
    for u, l in zip(UPPER, LOWER):
        out[u] = lm[u] + (lm[l] - lm[u]) * blink * 0.9
    if brow:
        for b in BROWS:
            out[b] = lm[b] + np.array([0, -mw * 0.10 * brow])
    if smile:
        up = -0.20 * mw * smile if smile > 0 else -0.16 * mw * smile
        out[61] = lm[61] + np.array([-0.08 * mw * max(smile, 0), up])
        out[291] = lm[291] + np.array([0.08 * mw * max(smile, 0), up])
        for i, j in ((40, 270), (91, 321), (185, 409), (146, 375)):
            out[i] = lm[i] + np.array([-0.03 * mw * max(smile, 0), up * 0.5])
            out[j] = lm[j] + np.array([0.03 * mw * max(smile, 0), up * 0.5])
        if smile > 0:
            for c in (205, 425, 50, 280):
                out[c] = lm[c] + np.array([0, -0.08 * mw * smile])
    if mouth:
        for i in LIPLOW:
            out[i] = lm[i] + np.array([0, mw * 0.16 * mouth])
        for i in JAW:
            out[i] = lm[i] + np.array([0, mw * 0.10 * mouth])
    return out


def warp_field(shape, src_list, dst_list):
    """Verschiebungsfeld (Ausgabe -> Quelle) aus Kontrollpunkten, normalisierte Gauss-Gewichte, grob berechnet."""
    h, w = shape
    step = 6
    gy, gx = np.mgrid[0:h:step, 0:w:step].astype(np.float32)
    num_x = np.zeros_like(gx); num_y = np.zeros_like(gx); den = np.full_like(gx, 0.03)
    any_ctrl = False
    for src, dst in zip(src_list, dst_list):
        fw = np.linalg.norm(src[454] - src[234]) + 1e-3
        d = dst - src
        moved = np.linalg.norm(d, axis=1) > 0.3
        idx = set(DENSE.tolist()) | set(np.nonzero(moved)[0].tolist())
        idx = np.array(sorted(idx))
        q = dst[idx]; dd = d[idx]
        if not moved[idx].any():
            continue
        # Anker zu nah an bewegten Punkten weglassen
        mv = q[moved[idx]]
        keep = moved[idx] | (np.min(np.linalg.norm(q[:, None] - mv[None], axis=2), axis=1) > fw * 0.045)
        q, dd = q[keep], dd[keep]
        sig = fw * 0.055
        x0, y0 = q[:, 0].min() - 4 * sig, q[:, 1].min() - 4 * sig
        x1, y1 = q[:, 0].max() + 4 * sig, q[:, 1].max() + 4 * sig
        sel = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
        if not sel.any():
            continue
        px, py = gx[sel], gy[sel]
        g = np.exp(-(((px[:, None] - q[None, :, 0]) ** 2 + (py[:, None] - q[None, :, 1]) ** 2) / (2 * sig * sig)))
        num_x[sel] += g @ dd[:, 0]; num_y[sel] += g @ dd[:, 1]; den[sel] += g.sum(1)
        any_ctrl = True
    if not any_ctrl:
        return None
    fx = cv2.resize(num_x / den, (w, h), interpolation=cv2.INTER_LINEAR)
    fy = cv2.resize(num_y / den, (w, h), interpolation=cv2.INTER_LINEAR)
    return fx, fy


def warp_image(img, field):
    if field is None:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xx - field[0], yy - field[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def blink_amount(i, seed):
    """Natuerliches Blinzeln alle 2-4 s (~5 Frames)."""
    r = np.random.default_rng(seed)
    times = np.cumsum(r.uniform(55, 110, 200)).astype(int)
    k = np.searchsorted(times, i)
    best = 0.0
    for tt in (times[max(k - 1, 0)], times[min(k, len(times) - 1)]):
        dt = abs(i - tt)
        if dt < 4:
            best = max(best, 1 - dt / 4)
    return best


# =================================================================== Effekte
yy0, xx0 = np.mgrid[0:H, 0:W].astype(np.float32)
nx, ny = (xx0 - W / 2) / (W / 2), (yy0 - H / 2) / (H / 2)
r2 = nx ** 2 + ny ** 2
MAPX = np.clip((nx * (1 + 0.06 * r2)) * (W / 2) + W / 2, 0, W - 1).astype(np.float32)
MAPY = np.clip((ny * (1 + 0.06 * r2)) * (H / 2) + H / 2, 0, H - 1).astype(np.float32)
VIGN = np.clip(1.15 - 0.55 * r2 ** 1.2, 0, 1).astype(np.float32)
_mask = np.zeros((H, W), np.uint8)
cv2.rectangle(_mask, (14 + 60, 12), (W - 15 - 60, H - 13), 255, -1)
cv2.rectangle(_mask, (14, 12 + 60), (W - 15, H - 13 - 60), 255, -1)
for cxy in ((74, 72), (W - 75, 72), (74, H - 73), (W - 75, H - 73)):
    cv2.circle(_mask, cxy, 60, 255, -1)
MASK = cv2.GaussianBlur(_mask.astype(np.float32) / 255, (11, 11), 0)
SCAN = (0.9 + 0.1 * (np.arange(H) % 2 == 0)).astype(np.float32)[:, None]
FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_PLAIN


def text(img, s, x, y, scale=0.8, val=1.0, thick=1, font=FONT):
    """putText braucht in OpenCV 5 uint8 -> in kleinem Patch zeichnen und einblenden."""
    if not s:
        return
    (tw, th), bl = cv2.getTextSize(s, font, scale, thick)
    x, y = int(x), int(y)
    pad = 4
    x0, y0 = max(x - pad, 0), max(y - th - pad, 0)
    x1, y1 = min(x + tw + pad, img.shape[1]), min(y + bl + pad, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    patch = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.putText(patch, s, (x - x0, y - y0), font, scale, 255, thick, cv2.LINE_AA)
    img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], patch.astype(np.float32) / 255 * val)


def brackets(img, x0, y0, x1, y1, val=1.0, L=22, th=2):
    x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
    for (ax, ay, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (ax, ay), (ax + dx * L, ay), val, th, cv2.LINE_AA)
        cv2.line(img, (ax, ay), (ax, ay + dy * L), val, th, cv2.LINE_AA)


def draw_mesh(ov, pts, alpha, reveal_y=None):
    p = pts.astype(np.int32)
    for a, b in EDGES[::2]:
        if reveal_y is not None and (p[a, 1] > reveal_y or p[b, 1] > reveal_y):
            continue
        cv2.line(ov, tuple(p[a]), tuple(p[b]), alpha * 0.55, 1, cv2.LINE_AA)
    for a, b in CONTOURS:
        if reveal_y is not None and (p[a, 1] > reveal_y or p[b, 1] > reveal_y):
            continue
        cv2.line(ov, tuple(p[a]), tuple(p[b]), alpha, 1, cv2.LINE_AA)


def glitch(a, rng, strength):
    out = a.copy()
    for _ in range(int(3 + 8 * strength)):
        y = rng.integers(0, H - 10); h = int(rng.integers(4, 40))
        out[y:y + h] = np.roll(out[y:y + h], int(rng.integers(-60, 60) * strength), axis=1)
    if strength > 0.6:
        for _ in range(4):
            x, y = rng.integers(0, W - 60), rng.integers(0, H - 30)
            out[y:y + 24, x:x + 48] = rng.random()
    return out


# =================================================================== Szenen-Renderer
def render_photo(shot, i, li, t, gt):
    beat = (i % FPB) / FPB
    bar = i // FPBAR
    d = load(shot["f"], shot.get("flipv", False))
    img = d["img"]
    ih, iw = img.shape
    cx, cy, z, rot = (hermite(shot["keys"], t, j) for j in (1, 2, 3, 4))
    cx += 0.004 * snoise(i / 30, 1); cy += 0.004 * snoise(i / 35, 2); rot += 0.3 * snoise(i / 50, 3)
    if shot.get("walk"):
        cy += 0.012 * abs(np.sin(np.pi * i / FPB)); rot += 1.2 * np.sin(np.pi * i / FPB)
    if shot.get("bank"):
        rot += 7 * np.sin(2 * np.pi * t * 0.8)
    punch = 1.0 + 0.025 * np.exp(-beat * 7) * ENERGY[sec(bar)]
    M = cam_matrix(iw, ih, cx, cy, z, rot, punch)

    # Gesichter animieren
    src_faces = d["faces"]
    dst_faces = []
    for k, lm in enumerate(src_faces):
        dst_faces.append(expression(lm, smile=0.25 * np.sin(i / 40 + k), brow=0.2 * np.sin(i / 23 + k * 2),
                                    blink=blink_amount(i, 11 + k)))
    if src_faces:
        img = warp_image(img, warp_field(img.shape, src_faces, dst_faces))

    if shot.get("person"):
        bg = inpainted(d)
        if src_faces:
            bg = img * 0 + bg
        a_bg = cv2.warpAffine(bg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = d["person"]
        ys, xs = np.nonzero(m > 0.5)
        if len(xs):
            ax, ay = xs.mean(), ys.max()           # Fusspunkt
        else:
            ax, ay = iw / 2, ih
        par = shot.get("par", 1.0)
        breath = 1 + 0.012 * np.sin(2 * np.pi * i / (FPBAR * 1.5))
        sway = 1.2 * np.sin(2 * np.pi * i / (FPBAR * 2.3)) * par
        P = cv2.getRotationMatrix2D((float(ax), float(ay)), float(sway), float(breath))
        P[0, 2] += par * 0.06 * iw * (t - 0.5)
        P3 = np.vstack([P, [0, 0, 1]]); M3 = np.vstack([M, [0, 0, 1]])
        MP = (M3 @ P3)[:2]
        a_p = cv2.warpAffine(img, MP, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        a_m = cv2.warpAffine(m, MP, (W, H), flags=cv2.INTER_LINEAR)
        found = shot.get("found")
        lift = 1.0 + (0.25 if found else 0.1)
        dim = 0.55 if found else 0.85
        a = a_bg * dim * (1 - a_m) + np.clip(a_p * lift, 0, 1) * a_m
        pm = a_m
        Mfaces = MP
    else:
        a = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        pm = None
        Mfaces = M

    if shot.get("pixel"):
        sm = cv2.resize(a, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
        a = cv2.resize(np.round(sm * 5) / 5, (W, H), interpolation=cv2.INTER_NEAREST)
    if shot.get("speed"):
        acc = a.copy()
        for k in range(1, 5):
            Z = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + 0.035 * k)
            acc += cv2.warpAffine(a, Z, (W, H)) * (1 - k * 0.15)
        a = acc / (1 + sum(1 - k * 0.15 for k in range(1, 5)))
    if shot.get("torch"):
        lx = 0.5 + 0.3 * snoise(i / 25, 7); ly = 0.5 + 0.25 * snoise(i / 29, 8)
        lm_ = np.exp(-((xx0 / W - lx) ** 2 * 1.78 + (yy0 / H - ly) ** 2) / (2 * 0.22 ** 2))
        a = a * (0.18 + 0.95 * lm_)
    if shot.get("flare"):
        fx, fy = apply_M(M, np.array([[shot["flare"][0] * iw, shot["flare"][1] * ih]]))[0]
        rr = ((xx0 - fx) ** 2 + (yy0 - fy) ** 2) / (W * 0.08) ** 2
        a = a + (0.5 + 0.3 * np.exp(-beat * 5)) * np.exp(-rr)
    u = shot.get("under")
    if u:
        a = underwater(a, i, li, shot["n"], u)

    # ---------------- HUD
    ov = np.zeros((H, W), np.float32)
    if shot.get("grid"):
        off = int(li * 2) % 40
        for x in range(-40 + off, W, 40):
            cv2.line(ov, (x, 0), (x, H), 0.12, 1)
        for y in range(-40 + off, H, 40):
            cv2.line(ov, (0, y), (W, y), 0.12, 1)
        sy = int((li * 9) % H)
        cv2.line(ov, (0, sy), (W, sy), 0.6, 2, cv2.LINE_AA)
    if shot.get("scan") and pm is not None:
        sy = int((li * 6) % (H + 200)) - 100
        band = np.exp(-((yy0 - sy) / 18) ** 2) * pm
        ov = np.maximum(ov, band * 0.8)
        edges = cv2.Canny((pm * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255
        ov = np.maximum(ov, edges * (yy0 < sy) * 0.7)
    if shot.get("lock") and pm is not None:
        ys, xs = np.nonzero(pm > 0.5)
        if len(xs) > 50:
            x0, x1, y0, y1 = np.percentile(xs, 2), np.percentile(xs, 98), np.percentile(ys, 2), np.percentile(ys, 98)
            k = min(1, li / 36)
            cxp, cyp = (x0 + x1) / 2, (y0 + y1) / 2
            grow = 1 + 1.5 * (1 - k) ** 2
            hw, hh = (x1 - x0) / 2 * grow + 10, (y1 - y0) / 2 * grow + 10
            val = 1.0 if (li // 6) % 2 or k >= 1 else 0.5
            brackets(ov, cxp - hw, cyp - hh, cxp + hw, cyp + hh, val)
            if k >= 1:
                lab = "SUBJEKT F  99,7%" if shot.get("found") else "M-01"
                text(ov, lab, cxp - hw, cyp - hh - 10, 0.55, 1.0)
    if shot.get("route"):
        k = min(1, li / (shot["n"] * 0.75))
        p0 = np.array([W * 0.30, H * 0.30]); p2 = np.array([W * 0.62, H * 0.62]); p1 = np.array([W * 0.72, H * 0.15])
        ts = np.linspace(0, k, max(2, int(60 * k)))
        pts = ((1 - ts) ** 2)[:, None] * p0 + (2 * (1 - ts) * ts)[:, None] * p1 + (ts ** 2)[:, None] * p2
        for j in range(0, len(pts) - 1, 2):
            cv2.line(ov, tuple(pts[j].astype(int)), tuple(pts[j + 1].astype(int)), 1.0, 2, cv2.LINE_AA)
        cv2.circle(ov, tuple(p0.astype(int)), 6, 1.0, 2, cv2.LINE_AA)
        cv2.circle(ov, tuple(pts[-1].astype(int)), 5 + int(4 * np.exp(-beat * 5)), 1.0, -1, cv2.LINE_AA)
        if k >= 1:
            cv2.circle(ov, tuple(p2.astype(int)), 12 + int(6 * np.exp(-beat * 4)), 1.0, 2, cv2.LINE_AA)
    if src_faces and not shot.get("person"):
        for dst in dst_faces:
            pts = apply_M(Mfaces, dst)
            draw_mesh(ov, pts, 0.35 + 0.3 * np.exp(-beat * 6))
    draw_hud_text(ov, shot, li)
    return a, ov


def underwater(a, i, li, n, stage):
    t = li / n
    ph = i / FPS
    dx = 6 * np.sin(2 * np.pi * (yy0 / 140 + ph * 0.6)) + 3 * np.sin(2 * np.pi * (yy0 / 47 - ph * 1.1))
    dy = 4 * np.cos(2 * np.pi * (xx0 / 160 + ph * 0.5))
    a = cv2.remap(a, xx0 + dx, yy0 + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = cv2.GaussianBlur(a, (0, 0), 1.6)
    depth = {1: 0.25 + 0.35 * t, 2: 0.6 + 0.2 * t, 3: 0.8 - 0.75 * t}[stage]
    grad = (1 - depth) * (1 - 0.6 * (yy0 / H) * depth)
    a = a * grad + 0.03
    # Lichtstrahlen
    rays = 0.5 + 0.5 * np.sin((xx0 - W / 2) / (yy0 + 260) * 22 + ph * 0.7) * np.sin((xx0 - W / 2) / (yy0 + 300) * 9 - ph * 0.4)
    a = a + np.clip(rays, 0, 1) ** 4 * (1 - yy0 / H) * (0.35 if stage == 3 else 0.18) * (1 - depth * 0.5)
    # Kaustiken
    c = np.sin(xx0 / 23 + ph * 2) * np.sin(yy0 / 19 - ph * 1.7) + np.sin((xx0 + yy0) / 31 + ph)
    a = a + np.clip(c - 1.2, 0, 1) * 0.12 * (1 - depth)
    # Luftblasen
    r = np.random.default_rng(77)
    for b in range(45):
        x0 = r.uniform(0, W); sp = r.uniform(60, 200); sz = r.uniform(2, 9); off = r.uniform(0, H + 100)
        y = (off - ph * sp) % (H + 100) - 50
        x = x0 + 12 * np.sin(ph * 3 + b)
        cv2.circle(a, (int(x), int(y)), int(sz), 0.85, 1, cv2.LINE_AA)
        cv2.circle(a, (int(x - sz * 0.3), int(y - sz * 0.3)), max(1, int(sz * 0.25)), 1.0, -1, cv2.LINE_AA)
    return a


def draw_hud_text(ov, shot, li):
    for (b, s, x, y, sc) in shot.get("hud", []):
        f0 = b * FPB
        if li >= f0:
            nchar = int((li - f0) / 1.2)
            vis = s[:nchar]
            cur = "_" if (li // 7) % 2 and nchar <= len(s) + 8 else ""
            text(ov, vis + cur, x, y, sc, 1.0, 2 if sc > 1 else 1)


MORPH_CACHE = {}


def render_morph(shot, i, li, t):
    beat = (i % FPB) / FPB
    seq = shot["seq"]
    tb = li / FPB
    k = 0
    while k + 1 < len(seq) and seq[k + 1][3] <= tb:
        k += 1
    A = seq[k]
    B = seq[min(k + 1, len(seq) - 1)]
    span = max(B[3] - A[3], 1e-3)
    u = np.clip((tb - A[3]) / span, 0, 1) if B is not A else 0.0
    u = u * u * (3 - 2 * u)
    smile = A[1] + (B[1] - A[1]) * u
    brow = A[2] + (B[2] - A[2]) * u
    dA = load(A[0]); dB = load(B[0])
    sad = smile < -0.3
    mouth = 0.0 if sad else 0.25 * max(0, np.sin(np.pi * (i % FPB) / FPB)) * (1 if shot["label"] == "GLUECK.EXE" else 0.4)
    fa, fb = dA["faces"], dB["faces"]
    blinks = [blink_amount(i, 31 + j) for j in range(2)]
    # Morph nur zwischen verschiedenen Fotos mit gleicher Gesichterzahl
    if A[0] != B[0] and len(fa) == len(fb) and dA["img"].shape == dB["img"].shape:
        mids = [(1 - u) * la + u * lb for la, lb in zip(fa, fb)]
        tg = [expression(m, smile, brow, blinks[j % 2], mouth) for j, m in enumerate(mids)]
        ia = warp_image(dA["img"], warp_field(dA["img"].shape, fa, tg))
        ib = warp_image(dB["img"], warp_field(dB["img"].shape, fb, tg))
        img = ia * (1 - u) + ib * u
        faces_out = tg
        d = dA
    else:
        d = dA if u < 0.5 else dB
        tg = [expression(m, smile, brow, blinks[j % 2], mouth) for j, m in enumerate(d["faces"])]
        img = warp_image(d["img"], warp_field(d["img"].shape, d["faces"], tg))
        faces_out = tg
    ih, iw = img.shape
    cx, cy, z, rot = (hermite(shot["keys"], t, j) for j in (1, 2, 3, 4))
    if d is load("16_selfie_nacht"):
        cy = 0.56; z = 1.0 + (z - 1) * 0.4
    rot += 0.6 * snoise(i / 40, 4)
    punch = 1 + 0.02 * np.exp(-beat * 7) * ENERGY[sec(i // FPBAR)]
    M = cam_matrix(iw, ih, cx, cy, z, rot, punch)
    a = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ov = np.zeros((H, W), np.float32)
    lab = shot["label"]
    if shot.get("rain"):
        r = np.random.default_rng(i)
        a = a * 0.8
        for _ in range(140):
            x, y = r.uniform(0, W), r.uniform(0, H)
            cv2.line(a, (int(x), int(y)), (int(x - 6), int(y + 34)), 0.75, 1, cv2.LINE_AA)
    pts_all = [apply_M(M, f) for f in faces_out]
    reveal = None
    if li < 60:
        reveal = li * 14
    for pts in pts_all:
        draw_mesh(ov, pts, 0.25 + 0.35 * np.exp(-beat * 6), reveal)
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        brackets(ov, x0 - 12, y0 - 12, x1 + 12, y1 + 12, 0.8, 16, 1)
    if shot.get("link") and len(pts_all) == 2:
        for j in (33, 263, 1, 13):
            p, q = pts_all[0][j], pts_all[1][j]
            mid = (p + q) / 2 + np.array([0, -40 * np.sin(np.pi * ((li / 30) % 1))])
            for a0, a1 in ((p, mid), (mid, q)):
                cv2.line(ov, tuple(a0.astype(int)), tuple(a1.astype(int)), 0.8, 1, cv2.LINE_AA)
        if li > shot["n"] * 0.5:
            hx, hy = W / 2, 110
            sz = 18 * (1 + 0.25 * np.exp(-beat * 5))
            tt = np.linspace(0, 2 * np.pi, 60)
            hxs = hx + sz * 16 * np.sin(tt) ** 3 / 16
            hys = hy - sz * (13 * np.cos(tt) - 5 * np.cos(2 * tt) - 2 * np.cos(3 * tt) - np.cos(4 * tt)) / 16
            cv2.polylines(ov, [np.stack([hxs, hys], 1).astype(np.int32)], True, 1.0, 2, cv2.LINE_AA)
    if lab:
        n = int(li / 1.5)
        text(ov, lab[:n], 40, 70, 1.1, 1.0, 2)
        pct = int(min(100, li / shot["n"] * 160))
        cv2.rectangle(ov, (40, 90), (40 + 260, 104), 0.7, 1)
        cv2.rectangle(ov, (42, 92), (42 + int(256 * pct / 100), 102), 0.9, -1)
        text(ov, f"{pct:3d} %", 312, 103, 0.5, 1.0)
    return a, ov


def render_boot(shot, i, li, t):
    d = load(shot["f"])
    img = d["img"]
    M = cam_matrix(img.shape[1], img.shape[0], hermite(shot["keys"], t, 1), hermite(shot["keys"], t, 2),
                   hermite(shot["keys"], t, 3), 0)
    a = cv2.warpAffine(img, M, (W, H)) * (0.05 + 0.25 * t)
    ov = np.zeros((H, W), np.float32)
    lines = ["DEUS EX MACHINA OS  v3.0", "SPEICHER ........ 640 KB OK", "NEURONALES NETZ ... LADEN",
             "ANDROID M-01 ...... AKTIVIERT", "EMOTIONSMODUL ..... FEHLT", "", "> MISSION STARTEN_"]
    y = 120
    shown = li / 2.2
    for s_ in lines:
        n = int(min(len(s_), shown))
        text(ov, s_[:n], 70, y, 0.75, 1.0, 1, MONO if False else FONT)
        shown -= len(s_) + 6
        y += 44
        if shown <= 0:
            break
    pb = min(1, li / (shot["n"] - 20))
    cv2.rectangle(ov, (70, 520), (890, 545), 0.8, 1)
    cv2.rectangle(ov, (74, 524), (74 + int(812 * pb), 541), 0.9, -1)
    return a, ov


def render_title(shot, i, li, t):
    d = load(shot["f"])
    img = d["img"]
    M = cam_matrix(img.shape[1], img.shape[0], hermite(shot["keys"], t, 1), hermite(shot["keys"], t, 2),
                   hermite(shot["keys"], t, 3), 0, 1 + 0.03 * np.exp(-((i % FPB) / FPB) * 7))
    a = cv2.warpAffine(img, M, (W, H)) * 0.35
    ov = np.zeros((H, W), np.float32)
    tl = np.zeros((H, W), np.float32)
    s1 = "MID DAY OUT III"
    (tw, th), _ = cv2.getTextSize(s1, FONT, 2.4, 5)
    text(tl, s1, (W - tw) / 2, 330, 2.4, 1.0, 5)
    s2 = "D E U S   E X   M A C H I N A"
    (tw2, _), _ = cv2.getTextSize(s2, FONT, 0.9, 2)
    text(tl, s2, (W - tw2) / 2, 400, 0.9, 1.0, 2)
    rng = np.random.default_rng(i)
    if li < 30 or rng.random() < 0.12:
        tl = glitch(tl, rng, 1.0)
    ov = np.maximum(ov, tl)
    ov = np.maximum(ov, cv2.GaussianBlur(tl, (0, 0), 8) * 0.6)
    return a, ov


def render_end(shot, i, li, t):
    a = np.zeros((H, W), np.float32)
    ov = np.zeros((H, W), np.float32)
    s1 = "END OF LINE"
    (tw, _), _ = cv2.getTextSize(s1, FONT, 1.6, 3)
    if li < shot["n"] - 40:
        text(ov, s1[:int(li / 3)], (W - tw) / 2, 350, 1.6, 1.0, 3)
        if li > 70:
            s2 = "M-01 + F  //  EMOTIONSMODUL INSTALLIERT"
            (tw2, _), _ = cv2.getTextSize(s2, FONT, 0.6, 1)
            text(ov, s2[:int((li - 70) / 1.2)], (W - tw2) / 2, 400, 0.6, 1.0, 1)
    return a, ov


# =================================================================== Frame
def render(i):
    shot = next(s for s in reversed(SHOTS) if s["start"] <= i)
    li = i - shot["start"]
    t = li / max(shot["n"] - 1, 1)
    gt = i / TOTAL
    bar = i // FPBAR
    beat = (i % FPB) / FPB
    rng = np.random.default_rng(i * 7 + 1)
    kind = shot.get("kind", "photo")
    if kind == "boot":
        a, ov = render_boot(shot, i, li, t)
    elif kind == "title":
        a, ov = render_title(shot, i, li, t)
    elif kind == "morph":
        a, ov = render_morph(shot, i, li, t)
    elif kind == "end":
        a, ov = render_end(shot, i, li, t)
    else:
        a, ov = render_photo(shot, i, li, t, gt)

    # Kontrast (S-Kurve, SW)
    a = np.clip(a, 0, 1.5)
    a = np.clip(a, 0, 1)
    a = a * a * (3 - 2 * a) * 0.6 + a * 0.4
    # Beat-Blitz
    e = ENERGY[sec(bar)]
    a = a * (1 + 0.12 * e * np.exp(-beat * 8))
    # Schnitt-Blitz
    if li < 2 and shot["start"] > 0 and kind != "end":
        a = a * 0.3 + 0.8
    # HUD einblenden (mit Glow)
    glow = cv2.GaussianBlur(ov, (0, 0), 3)
    a = np.maximum(a, ov) + glow * 0.35
    # Glitch: auf Schnitten, in energiereichen Abschnitten auf der letzten 16tel
    gl = 0.0
    if li < 4 and shot["start"] > 0:
        gl = 0.8
    if e >= 0.7 and (i % FPBAR) >= FPBAR - 3 and (bar % 2 == 1):
        gl = max(gl, 0.6)
    if shot.get("glitchy") and (i % FPB) < 2:
        gl = max(gl, 0.7)
    if gl:
        a = glitch(a, rng, gl)
    # Film/CRT
    small = cv2.resize(a, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2), (W, H))
    a = a + np.clip(bloom - 0.55, 0, 1) * 0.6
    a = a * (1 + 0.03 * rng.standard_normal())
    g = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
    a = a + cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST) * 0.05 + 0.015
    a = a * VIGN
    a = cv2.remap(a, MAPX, MAPY, cv2.INTER_LINEAR)
    a = a * SCAN
    # CRT-Ausschalten am Ende
    if i >= TOTAL - 36:
        k = (i - (TOTAL - 36)) / 36
        if k < 0.5:
            sy = max(0.004, 1 - k * 2)
            Z = np.float32([[1, 0, 0], [0, sy, H / 2 * (1 - sy)]])
            a = cv2.warpAffine(a, Z, (W, H)) * (1 + 2 * k)
        else:
            sx = max(0.003, 1 - (k - 0.5) * 2)
            Z = np.float32([[sx, 0, W / 2 * (1 - sx)], [0, 0.004, H / 2]])
            a = cv2.warpAffine(np.full((H, W), 1.0, np.float32), Z, (W, H)) * (1 - (k - 0.5) * 1.6)
    a = a * MASK + 0.008 * (1 - MASK)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes()


def main():
    frames = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(TOTAL)
    out = sys.stdout.buffer
    with Pool(4) as p:
        for k, buf in enumerate(p.imap(render, frames, chunksize=4)):
            out.write(buf)
            if k % 400 == 0:
                print(f"frame {frames[k]}/{TOTAL}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
