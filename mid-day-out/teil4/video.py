"""Video "mid day out IV - EAT SLEEP FILM REPEAT".

16:9, 1280x720, 24 fps, 144 BPM -> 10 Frames/Beat, 40/Takt, 180 Takte = 7200 Frames = 300 s.
Hell, hart, digital. Generierte Elemente: Blitze/Strom, Drohnenschwarm, Klone, Roboter-Gesichter, EKG, Schrift.
Abschnitte wie music.py. Laeuft mit der venv-Python (cv2). Schreibt rohe RGB-Frames auf stdout.
"""
import os
import sys
import numpy as np
import cv2
from multiprocessing import Pool

cv2.setNumThreads(1)
W, H = 1280, 720
FPS, FPB, FPBAR = 24, 10, 40
BARS = 180
TOTAL = BARS * FPBAR
B = "bilder/"
EDGES = np.load("build/mesh_edges.npy")
CONTOURS = np.load("build/mesh_contours.npy")


def sec(bar):
    for name, a, b in (("intro", 0, 8), ("life", 8, 32), ("current", 32, 56), ("drones", 56, 80), ("death", 80, 96),
                       ("charge", 96, 104), ("trans", 104, 136), ("rave", 136, 168), ("repeat", 168, 180)):
        if a <= bar < b:
            return name
    return "repeat"


ENERGY = dict(intro=0.4, life=0.7, current=0.85, drones=0.9, death=0.0, charge=0.5, trans=0.8, rave=1.0, repeat=0.6)
CHANT_BARS = set([b for b in range(0, 8) if b % 2 == 1] + [8, 16, 24] + [104, 112, 120, 128] +
                 [b for b in range(136, 168) if b % 2 == 0] + [b for b in range(168, 180) if b % 2 == 0])
WORDS = ["EAT", "SLEEP", "FILM", "REPEAT"]

# ====================================================================== Shotliste
LIFE = ["a_selfie", "a_tanz", "a_sakko", "a_pelz_gruen", "02_schuhe_aufzug", "03_beine", "a_treppe", "a_pelz_grau",
        "a_galerie_publikum", "a_teddy", "27_aussicht", "16_selfie_nacht"]
CURRENT = ["18_apple_board", "01_platine", "22_sperrbildschirm", "19_nachtflug", "15_dom_nacht", "a_dom_nacht",
           "14_dachterrasse", "06_bruecke", "05_profile", "04_aktie", "08_turm", "a_galerie_leer"]
DRONES = ["23_berg", "26_meer_sonne", "21_flughafen", "20_fluegel", "27_aussicht", "24_wald", "11_meer",
          "07_turm_hecke", "13_galerie_mann", "17_globus", "a_globus", "14_dachterrasse"]
PERSON = {"a_tanz", "a_sakko", "a_pelz_gruen", "a_pelz_grau", "27_aussicht", "13_galerie_mann", "14_dachterrasse",
          "a_teddy", "12_spiegelraum"}
FACES = ["f1", "f2", "f3", "f4", "f5", "a_selfie", "16_selfie_nacht"]
ALL = sorted(os.path.splitext(f)[0] for f in os.listdir(B))

SHOTS = [dict(kind="intro", bars=4), dict(f="01_platine", bars=4, fx={"flow"}, z=(1.6, 2.2))]
for n in LIFE:
    SHOTS.append(dict(f=n, bars=2, fx={"clones"} if n in PERSON else set(), z=(1.0, 1.25)))
for n in CURRENT:
    SHOTS.append(dict(f=n, bars=2, fx={"flow", "bolts"}, z=(1.1, 1.6)))
for n in DRONES:
    SHOTS.append(dict(f=n, bars=2, fx={"drones"} | ({"target"} if n in PERSON else set()), z=(1.0, 1.3)))
for n, b in (("09_kunst_blau", 4), ("10_leuchtkaesten", 4), ("25_kirche", 3), ("a_galerie_leer", 3), ("12_spiegelraum", 2)):
    SHOTS.append(dict(f=n, bars=b, fx={"death"}, z=(1.3, 1.0)))
SHOTS.append(dict(kind="charge", bars=8))
SHOTS += [dict(kind="robot", bars=8, seq=["f2", "f1", "f3", "f4", "f5"]),
          dict(kind="grid", bars=8),
          dict(f="a_sakko", bars=4, fx={"wire", "bolts"}, z=(1.0, 1.4)),
          dict(f="a_pelz_gruen", bars=4, fx={"wire", "bolts"}, z=(1.0, 1.4)),
          dict(kind="robot", bars=8, seq=["16_selfie_nacht", "a_selfie", "16_selfie_nacht"])]
SHOTS.append(dict(kind="rave", bars=32))
SHOTS += [dict(f="27_aussicht", bars=8, fx={"clones", "infinite"}, z=(1.0, 1.0)), dict(kind="outro", bars=4)]
st = 0
for s in SHOTS:
    s["start"] = st; s["n"] = s["bars"] * FPBAR; st += s["n"]
assert st == TOTAL, st

# ====================================================================== Laden
_cache = {}


def path_of(name):
    for ext in (".jpg", ".png", ".jpeg"):
        if os.path.exists(B + name + ext):
            return B + name + ext
    raise FileNotFoundError(name)


def load(name):
    if name in _cache:
        return _cache[name]
    img = cv2.imread(path_of(name), cv2.IMREAD_GRAYSCALE)
    s = 1500 / max(img.shape)
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    # hell, hart: High-Key
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip((img - lo) / max(hi - lo, 1e-3), 0, 1)
    img = np.clip(img * 1.08 + 0.04, 0, 1) ** 0.85
    an = np.load(f"build/analysis/{name}.npz")
    person = cv2.resize(an["person"].astype(np.float32).squeeze(), (img.shape[1], img.shape[0]))
    person = np.clip((person - 0.35) / 0.4, 0, 1)
    faces = [f[:, :2] * np.array([img.shape[1], img.shape[0]], np.float32) for f in an["faces"]]
    faces.sort(key=lambda f: f[:, 0].mean())
    ih, iw = img.shape
    if faces:
        allp = np.concatenate(faces)
        c = (allp[:, 0].mean() / iw, allp[:, 1].mean() / ih)
    elif person.mean() > 0.02 and name in PERSON:
        ys, xs = np.nonzero(person > 0.5)
        c = (xs.mean() / iw, (ys.min() * 0.6 + ys.mean() * 0.4) / ih)
    else:
        c = (0.5, 0.5)
    d = dict(img=img, person=person, faces=faces, center=c, name=name)
    _cache[name] = d
    return d


# ====================================================================== Helfer
def snoise(x, seed):
    pts = np.random.default_rng(seed).standard_normal(256)
    i = int(np.floor(x)) % 255
    f = x - np.floor(x); f = f * f * (3 - 2 * f)
    return pts[i] * (1 - f) + pts[i + 1] * f


def cam_matrix(iw, ih, cx, cy, zoom, rot=0.0):
    cw = iw / zoom
    if cw * H / W > ih:
        cw = ih / zoom * W / H
    sc = W / cw
    hw, hh = cw / 2, cw * H / W / 2
    px = np.clip(cx * iw, hw, iw - hw) if iw > 2 * hw else iw / 2
    py = np.clip(cy * ih, hh, ih - hh) if ih > 2 * hh else ih / 2
    M = cv2.getRotationMatrix2D((float(px), float(py)), float(rot), float(sc))
    M[0, 2] += W / 2 - px; M[1, 2] += H / 2 - py
    return M


def apply_M(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


def text_patch(s, scale, thick, font=cv2.FONT_HERSHEY_DUPLEX):
    (tw, th), bl = cv2.getTextSize(s, font, scale, thick)
    p = np.zeros((th + bl + 2 * thick + 8, tw + 2 * thick + 8), np.uint8)
    cv2.putText(p, s, (thick + 4, th + thick + 4), font, scale, 255, thick, cv2.LINE_AA)
    return p.astype(np.float32) / 255


_tcache = {}


def put_text(layer, s, cx, cy, scale, thick=2, val=1.0, font=cv2.FONT_HERSHEY_DUPLEX, anchor="c"):
    key = (s, scale, thick, font)
    if key not in _tcache:
        _tcache[key] = text_patch(s, scale, thick, font)
    p = _tcache[key]
    h, w = p.shape
    x0 = int(cx - w / 2) if anchor == "c" else int(cx)
    y0 = int(cy - h / 2)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, W), min(y0 + h, H)
    if xb <= xa or yb <= ya:
        return
    layer[ya:yb, xa:xb] = np.maximum(layer[ya:yb, xa:xb], p[ya - y0:yb - y0, xa - x0:xb - x0] * val)


def bolt(p0, p1, rng, rough=0.25, depth=6):
    pts = [np.array(p0, float), np.array(p1, float)]
    for _ in range(depth):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            mid = (a + b) / 2
            dvec = b - a
            nrm = np.array([-dvec[1], dvec[0]])
            mid = mid + nrm * rng.normal(0, rough * 0.5)
            new += [mid, b]
        pts = new
        rough *= 0.6
    return np.array(pts)


def draw_bolt(layer, p0, p1, rng, val=1.0, branches=3):
    pts = bolt(p0, p1, rng)
    cv2.polylines(layer, [pts.astype(np.int32)], False, val, 2, cv2.LINE_AA)
    for _ in range(branches):
        k = rng.integers(len(pts) // 4, len(pts) - 1)
        q0 = pts[k]
        q1 = q0 + (np.array(p1) - np.array(p0)) * rng.uniform(0.15, 0.4) + rng.normal(0, 60, 2)
        cv2.polylines(layer, [bolt(q0, q1, rng, 0.3, 4).astype(np.int32)], False, val * 0.7, 1, cv2.LINE_AA)


def glitch(a, rng, strength):
    out = a.copy()
    for _ in range(int(4 + 10 * strength)):
        y = rng.integers(0, H - 10); h = int(rng.integers(3, 50))
        out[y:y + h] = np.roll(out[y:y + h], int(rng.integers(-90, 90) * strength), axis=1)
    return out


# ====================================================================== Gesichter
UPPER = [159, 158, 160, 161, 157, 386, 385, 387, 384, 388]
LOWER = [145, 153, 144, 163, 154, 374, 380, 373, 390, 381]
DENSE = np.arange(0, 468, 3)


def blink_amount(i, seed):
    r = np.random.default_rng(seed)
    times = np.cumsum(r.uniform(45, 90, 200)).astype(int)
    k = np.searchsorted(times, i)
    best = 0.0
    for tt in (times[max(k - 1, 0)], times[min(k, len(times) - 1)]):
        dt = abs(i - tt)
        if dt < 3:
            best = max(best, 1 - dt / 3)
    return best


def expression(lm, smile=0.0, blink=0.0, mouth=0.0):
    out = lm.copy()
    mw = np.linalg.norm(lm[291] - lm[61]) + 1e-3
    for u, l in zip(UPPER, LOWER):
        out[u] = lm[u] + (lm[l] - lm[u]) * blink * 0.9
    if smile:
        up = -0.2 * mw * smile
        out[61] = lm[61] + np.array([-0.06 * mw * max(smile, 0), up])
        out[291] = lm[291] + np.array([0.06 * mw * max(smile, 0), up])
    if mouth:
        for i in (14, 17, 87, 317, 84, 314, 178, 402):
            out[i] = lm[i] + np.array([0, mw * 0.18 * mouth])
        for i in (152, 148, 377, 176, 400):
            out[i] = lm[i] + np.array([0, mw * 0.11 * mouth])
    return out


def warp_faces(img, src_list, dst_list):
    h, w = img.shape
    step = 6
    gy, gx = np.mgrid[0:h:step, 0:w:step].astype(np.float32)
    nx_ = np.zeros_like(gx); ny_ = np.zeros_like(gx); den = np.full_like(gx, 0.03)
    used = False
    for src, dst in zip(src_list, dst_list):
        fw = np.linalg.norm(src[454] - src[234]) + 1e-3
        d = dst - src
        moved = np.linalg.norm(d, axis=1) > 0.3
        if not moved.any():
            continue
        idx = np.array(sorted(set(DENSE.tolist()) | set(np.nonzero(moved)[0].tolist())))
        q, dd, mv = dst[idx], d[idx], moved[idx]
        keep = mv | (np.min(np.linalg.norm(q[:, None] - q[mv][None], axis=2), axis=1) > fw * 0.045)
        q, dd = q[keep], dd[keep]
        sig = fw * 0.055
        x0, y0 = q[:, 0].min() - 4 * sig, q[:, 1].min() - 4 * sig
        x1, y1 = q[:, 0].max() + 4 * sig, q[:, 1].max() + 4 * sig
        sel = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
        g = np.exp(-(((gx[sel][:, None] - q[None, :, 0]) ** 2 + (gy[sel][:, None] - q[None, :, 1]) ** 2) / (2 * sig * sig)))
        nx_[sel] += g @ dd[:, 0]; ny_[sel] += g @ dd[:, 1]; den[sel] += g.sum(1)
        used = True
    if not used:
        return img
    fx = cv2.resize(nx_ / den, (w, h)); fy = cv2.resize(ny_ / den, (w, h))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xx - fx, yy - fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def draw_mesh(layer, pts, val, every=2):
    p = pts.astype(np.int32)
    for a, b in EDGES[::every]:
        cv2.line(layer, tuple(p[a]), tuple(p[b]), val * 0.6, 1, cv2.LINE_AA)
    for a, b in CONTOURS:
        cv2.line(layer, tuple(p[a]), tuple(p[b]), val, 1, cv2.LINE_AA)


# ====================================================================== Generierte Elemente
yy0, xx0 = np.mgrid[0:H, 0:W].astype(np.float32)


def drones(gray, red, i, n=14, seed=5, targets=None):
    r = np.random.default_rng(seed)
    tt = i / FPS
    for k in range(n):
        z = r.uniform(0.3, 1.0)                 # Naehe
        sp = r.uniform(0.05, 0.12) * (0.5 + z)
        ph = r.uniform(0, 1)
        x = ((ph + tt * sp) % 1.3 - 0.15) * W
        y = H * (0.15 + 0.5 * r.uniform()) + 30 * np.sin(tt * 1.5 + k) * z
        s = 18 + 55 * z
        col = 0.05
        # Arme
        for ang in (35, 145):
            dx, dy = np.cos(np.radians(ang)) * s, np.sin(np.radians(ang)) * s * 0.35
            cv2.line(gray, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), col, max(1, int(s / 12)), cv2.LINE_AA)
        cv2.ellipse(gray, (int(x), int(y)), (int(s * 0.32), int(s * 0.16)), 0, 0, 360, col, -1, cv2.LINE_AA)
        for sx in (-1, 1):
            for sy in (-1, 1):
                rx, ry = int(x + sx * s * np.cos(np.radians(35))), int(y + sy * s * 0.35 * np.sin(np.radians(35)))
                spin = 0.35 + 0.15 * np.sin(i * 2.7 + k)
                cv2.ellipse(gray, (rx, ry), (int(s * 0.45), int(s * 0.08)), 0, 0, 360, spin, 1, cv2.LINE_AA)
        if (i + k * 3) % 12 < 3:
            cv2.circle(red, (int(x), int(y + s * 0.12)), max(2, int(s / 10)), 1.0, -1, cv2.LINE_AA)


def clones(a, m, i, li, count, spacing, infinite=False):
    """Klone der Person: verschobene, verkleinerte Kopien hinter dem Original."""
    out = a.copy()
    ys, xs = np.nonzero(m > 0.5)
    if len(xs) < 50:
        return out
    ax, ay = xs.mean(), ys.max()
    order = []
    for k in range(1, count + 1):
        side = 1 if k % 2 else -1
        rank = (k + 1) // 2
        order.append((rank, side))
    for rank, side in sorted(order, key=lambda o: -o[0]):
        sc = 0.88 ** rank if not infinite else 0.8 ** rank
        dx = side * rank * spacing * (0.9 if not infinite else 0.6)
        dy = -rank * (8 if not infinite else 20)
        lag = rank * 3
        sway = 2.5 * np.sin((i - lag) / 7)
        M = cv2.getRotationMatrix2D((float(ax), float(ay)), float(sway * side), float(sc))
        M[0, 2] += dx; M[1, 2] += dy
        ca = cv2.warpAffine(a, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        cm = cv2.warpAffine(m, M, (W, H))
        shade = 0.85 ** rank
        out = out * (1 - cm) + ca * shade * cm
    return out * (1 - m) + a * m


def ecg(red, i, li, mode):
    """EKG-Linie unten. mode: 'beat' (langsam werdend), 'flat', 'restart'."""
    xs = np.arange(0, W, 2)
    base = H - 90
    ys = np.full(len(xs), float(base))
    head = (li * 9) % W
    period = {0: 26, 1: 40, 2: 70}
    for k, x in enumerate(xs):
        dt = (head - x) % W
        if mode == "flat":
            continue
        per = 30 if mode == "restart" else period[min(2, li // 220)]
        ph = ((x // 2) % per) / per
        if 0.4 < ph < 0.45:
            ys[k] = base - 70 * np.sin((ph - 0.4) / 0.05 * np.pi)
        elif 0.45 < ph < 0.5:
            ys[k] = base + 25 * np.sin((ph - 0.45) / 0.05 * np.pi)
    fade = np.clip(1 - ((head - xs) % W) / (W * 0.8), 0, 1)
    for k in range(len(xs) - 1):
        if fade[k] > 0.02:
            cv2.line(red, (int(xs[k]), int(ys[k])), (int(xs[k + 1]), int(ys[k + 1])), float(fade[k]), 2, cv2.LINE_AA)
    cv2.circle(red, (int(head), int(ys[min(len(ys) - 1, head // 2)])), 5, 1.0, -1, cv2.LINE_AA)


# ====================================================================== Szenen
def keys_for(shot, d, t, i):
    cx, cy = d["center"]
    z0, z1 = shot.get("z", (1.0, 1.3))
    u = t * t * (3 - 2 * t)
    z = z0 + (z1 - z0) * u
    cx += 0.03 * (t - 0.5) * (1 if (shot["start"] // FPBAR) % 2 else -1)
    cx += 0.004 * snoise(i / 20, 1); cy += 0.004 * snoise(i / 25, 2)
    return cx, cy, z, 0.5 * snoise(i / 40, 3)


def photo(name, shot, i, li, t, E, R, punch=1.0):
    d = load(name)
    img = d["img"]
    ih, iw = img.shape
    if d["faces"]:
        dst = [expression(lm, smile=0.3 * np.sin(i / 30 + k), blink=blink_amount(i, 50 + k),
                          mouth=0.3 * max(0, np.sin(np.pi * (i % FPB) / FPB)) * (1 if sec(i // FPBAR) in ("rave", "life") else 0))
               for k, lm in enumerate(d["faces"])]
        img = warp_faces(img, d["faces"], dst)
    else:
        dst = []
    cx, cy, z, rot = keys_for(shot, d, t, i)
    M = cam_matrix(iw, ih, cx, cy, z * punch, rot)
    a = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = cv2.warpAffine(d["person"], M, (W, H)) if name in PERSON or d["faces"] else None
    faces_out = [apply_M(M, f) for f in dst]
    return a, m, faces_out


def render_photo(shot, i, li, t):
    E = np.zeros((H, W), np.float32); R = np.zeros((H, W), np.float32)
    bar = i // FPBAR
    beat = (i % FPB) / FPB
    fx = shot.get("fx", set())
    punch = 1 + 0.03 * np.exp(-beat * 6) * ENERGY[sec(bar)]
    a, m, faces = photo(shot["f"], shot, i, li, t, E, R, punch)
    rng = np.random.default_rng(i)
    if "clones" in fx and m is not None:
        cnt = min(8, 1 + li // FPB) if "infinite" not in fx else min(16, 2 + li // 12)
        a = clones(a, m, i, li, cnt, W * 0.11, "infinite" in fx)
    if "flow" in fx:
        g8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        ed = cv2.Canny(cv2.GaussianBlur(g8, (5, 5), 0), 60, 140).astype(np.float32) / 255
        rr = np.random.default_rng(shot["start"])
        ox, oy = rr.uniform(0.1, 0.9) * W, rr.uniform(0.1, 0.9) * H
        dist = np.sqrt((xx0 - ox) ** 2 + (yy0 - oy) ** 2)
        front = li * 22
        trail = np.clip(1 - (front - dist) / 500, 0.15, 1) * (dist < front)
        tip = np.exp(-((dist - front) / 25) ** 2)
        E = np.maximum(E, cv2.dilate(ed, None) * (trail * 0.8 + tip * 2.0))
        a = a * (0.75 + 0.25 * (dist < front))
    if "bolts" in fx and (i % FPB) < 3:
        for _ in range(1 + int(2 * ENERGY[sec(bar)])):
            p0 = (rng.uniform(0, W), rng.uniform(-50, H * 0.3))
            p1 = (rng.uniform(0, W), rng.uniform(H * 0.5, H))
            if m is not None and m.max() > 0.5:
                ys, xs = np.nonzero(m > 0.5)
                j = rng.integers(len(xs)); p1 = (xs[j], ys[j])
            draw_bolt(E, p0, p1, rng)
        a = a * 1.15
    if "drones" in fx:
        drones(a, R, i, 10 + (shot["start"] // FPBAR) % 8, seed=shot["start"])
    if "target" in fx and m is not None and m.max() > 0.5:
        ys, xs = np.nonzero(m > 0.5)
        x0, x1, y0, y1 = np.percentile(xs, 2), np.percentile(xs, 98), np.percentile(ys, 2), np.percentile(ys, 98)
        k = min(1, li / 25)
        c = ((x0 + x1) / 2, (y0 + y1) / 2)
        hw, hh = (x1 - x0) / 2 * (1 + 2 * (1 - k)) + 12, (y1 - y0) / 2 * (1 + 2 * (1 - k)) + 12
        cv2.rectangle(R, (int(c[0] - hw), int(c[1] - hh)), (int(c[0] + hw), int(c[1] + hh)), 1.0, 2, cv2.LINE_AA)
        cv2.line(R, (int(c[0]), 0), (int(c[0]), H), 0.5, 1); cv2.line(R, (0, int(c[1])), (W, int(c[1])), 0.5, 1)
        if k >= 1 and (li // 4) % 2:
            put_text(R, "ZIEL ERFASST", c[0] + hw + 10, c[1] - hh, 0.8, 2, 1.0, anchor="l")
    if "wire" in fx and m is not None:
        k = np.clip(t * 1.3, 0, 1)
        g8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        ed = cv2.Canny(g8, 40, 120).astype(np.float32) / 255
        ys, xs = np.nonzero(m > 0.5)
        ytop, ybot = (ys.min(), ys.max()) if len(ys) else (0, H)
        sweep = ybot - (ybot - ytop) * k
        conv = m * (yy0 > sweep)
        a = a * (1 - conv) + 0.03 * conv
        E = np.maximum(E, ed * conv * 1.5)
        E = np.maximum(E, np.exp(-((yy0 - sweep) / 6) ** 2) * m * 1.5)
    if "death" in fx:
        a = 1 - a                                   # Negativ
        a = a * (0.5 + 0.5 * (1 - t))
        rr = np.random.default_rng(shot["start"])
        noise = cv2.resize(rr.random((H // 16, W // 16)).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        diss = (noise < t * 0.9 * (bar >= 88 or t > 0.5))
        a = np.where(diss, 0.0, a)
        bar_ = i // FPBAR
        if bar_ < 91:
            ecg(R, i, i - 80 * FPBAR, "beat")
        else:
            ecg(R, i, i - 80 * FPBAR, "flat")
    for pts in faces:
        draw_mesh(E, pts, 0.4 * ENERGY[sec(bar)])
    return a, E, R


def render_intro(shot, i, li, t):
    a = np.zeros((H, W), np.float32); E = np.zeros_like(a); R = np.zeros_like(a)
    rng = np.random.default_rng(i)
    beat_i = (i % FPBAR) // FPB
    if (i % FPB) < 4:
        for _ in range(2):
            draw_bolt(E, (rng.uniform(0, W), -20), (rng.uniform(0, W), H + 20), rng, 1.0, 4)
        a += 0.15
    return a, E, R


def render_charge(shot, i, li, t):
    a = np.zeros((H, W), np.float32); E = np.zeros_like(a); R = np.zeros_like(a)
    rng = np.random.default_rng(i)
    ecg(R, i, li, "restart")
    k = t
    if rng.random() < 0.15 + 0.8 * k:
        for _ in range(1 + int(4 * k)):
            c = (W / 2, H / 2 - 40)
            ang = rng.uniform(0, 2 * np.pi); rr = rng.uniform(200, 700)
            draw_bolt(E, (c[0] + np.cos(ang) * rr, c[1] + np.sin(ang) * rr), c, rng, 0.9, 2)
    sz = 1.5 + 2.5 * k
    put_text(E, "NEUSTART", W / 2, H / 2 - 40, sz, int(2 + 4 * k), 0.4 + 0.6 * k)
    if li > shot["n"] - 6:
        a += 1.0
    return a, E, R


def render_robot(shot, i, li, t):
    E = np.zeros((H, W), np.float32); R = np.zeros((H, W), np.float32)
    seq = shot["seq"]
    seg = shot["n"] / len(seq)
    k = min(int(li / seg), len(seq) - 1)
    name = seq[k]
    lt = (li - k * seg) / seg
    beat = (i % FPB) / FPB
    d = load(name)
    img = d["img"]; ih, iw = img.shape
    dst = [expression(lm, smile=0.2 * np.sin(i / 20), blink=blink_amount(i, 70 + j),
                      mouth=0.6 * max(0, np.sin(np.pi * (i % FPB) / FPB)))
           for j, lm in enumerate(d["faces"])]
    img = warp_faces(img, d["faces"], dst)
    cx, cy = d["center"]
    z = 1.0 + 0.25 * lt
    M = cam_matrix(iw, ih, cx, cy + 0.03, z * (1 + 0.03 * np.exp(-beat * 6)), 0.6 * snoise(i / 30, 9))
    a = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    prog = (li / shot["n"])           # Verwandlungsgrad
    for f in dst:
        pts = apply_M(M, f)
        draw_mesh(E, pts, 0.3 + 0.7 * prog, every=1 if prog > 0.5 else 2)
        hull = cv2.convexHull(pts.astype(np.int32))
        fm = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(fm, hull, 1)
        fm = fm.astype(np.float32)
        # Gesicht wird metallisch/pixelig
        pix = cv2.resize(cv2.resize(a, (W // 14, H // 14), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_NEAREST)
        a = a * (1 - fm * prog * 0.6) + (np.round(pix * 4) / 4) * fm * prog * 0.6
        for c in (468, 473):
            p = pts[c]
            rad = int(np.linalg.norm(pts[33] - pts[133]) * 0.25 * (1 + 0.4 * np.exp(-beat * 5)))
            cv2.circle(R, (int(p[0]), int(p[1])), rad, 1.0, -1, cv2.LINE_AA)
            cv2.circle(R, (int(p[0]), int(p[1])), rad * 3, 0.5 * prog + 0.2, 1, cv2.LINE_AA)
    pct = int(min(100, prog * 110))
    put_text(E, f"VERWANDLUNG {pct:3d} %", 40, 50, 0.9, 2, 1.0, anchor="l")
    return a, E, R


def render_grid(shot, i, li, t):
    E = np.zeros((H, W), np.float32); R = np.zeros((H, W), np.float32)
    beat_n = i // FPB
    n = 2 if li < shot["n"] * 0.25 else (3 if li < shot["n"] * 0.5 else 4)
    cw, ch = W // n, H // n
    a = np.zeros((H, W), np.float32)
    rng = np.random.default_rng(beat_n)
    for gy in range(n):
        for gx in range(n):
            name = FACES[(gx + gy * n + beat_n) % len(FACES)]
            d = load(name)
            key = (name, cw, ch)
            if key not in _cache:
                img = d["img"]; ih, iw = img.shape
                cx, cy = d["center"]
                cwid = iw * 0.9
                chei = cwid * ch / cw
                x0 = int(np.clip(cx * iw - cwid / 2, 0, iw - cwid)); y0 = int(np.clip(cy * ih - chei / 2, 0, ih - chei))
                _cache[key] = cv2.resize(img[y0:y0 + int(chei), x0:x0 + int(cwid)], (cw, ch), interpolation=cv2.INTER_AREA)
            tile = _cache[key]
            if (gx + gy + beat_n) % 5 == 0:
                tile = 1 - tile
            a[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw] = tile
            cv2.rectangle(E, (gx * cw, gy * ch), ((gx + 1) * cw - 1, (gy + 1) * ch - 1), 0.6, 2)
    put_text(E, "KLON %03d" % (beat_n % 1000), 40, 50, 0.9, 2, 1.0, anchor="l")
    return a, E, R


def render_rave(shot, i, li, t):
    beat_n = li // FPB
    bar = i // FPBAR
    name = ALL[(beat_n * 7) % len(ALL)]
    sub = dict(start=shot["start"] + beat_n * FPB, f=name, z=(1.15, 1.0), fx=set())
    bt = (li % FPB) / FPB
    if bar % 8 in (6, 7):
        return render_grid(dict(n=shot["n"], start=shot["start"]), i, li + shot["n"] // 2, t)
    a, E, R = render_photo(sub | {"n": FPB}, i, li % FPB, bt)
    d = load(name)
    if name in PERSON and beat_n % 2 == 0:
        m = cv2.warpAffine(d["person"], cam_matrix(*d["img"].shape[::-1], *d["center"], 1.0), (W, H))
        a = clones(a, m, i, li, 6, W * 0.12)
    rng = np.random.default_rng(i)
    if beat_n % 4 == 3 and bt < 0.3:
        a = 1 - a
    if beat_n % 3 == 1 and (i % FPB) < 3:
        draw_bolt(E, (rng.uniform(0, W), 0), (rng.uniform(0, W), H), rng, 1.0)
    if bar % 4 == 1:
        drones(a, R, i, 8, seed=bar)
    return a, E, R


def render_outro(shot, i, li, t):
    a = np.zeros((H, W), np.float32); E = np.zeros_like(a); R = np.zeros_like(a)
    rng = np.random.default_rng(i)
    if (i % FPB) < 3:
        draw_bolt(E, (rng.uniform(0, W), -20), (rng.uniform(0, W), H + 20), rng, 0.8 * (1 - t), 2)
    put_text(E, "REPEAT", W / 2, H / 2, 3.0, 6, 1.0 - 0.8 * t)
    return a, E, R


RENDER = dict(intro=render_intro, charge=render_charge, robot=render_robot, grid=render_grid, rave=render_rave,
              outro=render_outro)

TINT = dict(intro=(0.9, 0.95, 1.0), life=(1.0, 0.98, 0.94), current=(0.88, 0.96, 1.05), drones=(1.0, 1.0, 1.0),
            death=(1.0, 0.75, 0.75), charge=(1.0, 1.0, 1.0), trans=(0.9, 1.0, 1.05), rave=(1.0, 1.0, 1.0),
            repeat=(0.95, 0.97, 1.0))
CYAN = np.array([0.45, 0.9, 1.0], np.float32)
REDC = np.array([1.0, 0.12, 0.08], np.float32)


def render(i):
    shot = next(s for s in reversed(SHOTS) if s["start"] <= i)
    li = i - shot["start"]
    t = li / max(shot["n"] - 1, 1)
    bar = i // FPBAR
    s = sec(bar)
    beat = (i % FPB) / FPB
    beat_in_bar = (i % FPBAR) // FPB
    rng = np.random.default_rng(i * 13 + 5)
    a, E, R = RENDER.get(shot.get("kind", ""), render_photo)(shot, i, li, t)

    # Schrift-Blitze zum Chant
    if bar in CHANT_BARS and (i % FPB) < 7:
        w = WORDS[beat_in_bar]
        lay = np.zeros((H, W), np.float32)
        put_text(lay, w, W / 2, H / 2, 4.2 if w != "REPEAT" else 3.6, 12, 1.0)
        if s in ("rave", "intro", "repeat"):
            a = np.where(lay > 0.5, 1 - a, a * 0.55)        # invertierte Schrift
            E = np.maximum(E, cv2.GaussianBlur(lay, (0, 0), 6) * 0.5)
        else:
            E = np.maximum(E, lay)
    # Beat-Strobe
    e = ENERGY[s]
    a = np.clip(a, 0, 1)
    a = a * (1 + 0.18 * e * np.exp(-beat * 9))
    if s == "rave" and (i % (FPB // 2)) == 0 and rng.random() < 0.2:
        a = np.minimum(1, a + 0.3)
    # Schnitt-Blitz + Glitch
    if li < 2 and shot["start"] > 0:
        a = a * 0.4 + 0.7
    gl = 0.0
    if li < 3 and shot["start"] > 0:
        gl = 0.7
    if e > 0.75 and (i % FPBAR) >= FPBAR - 3:
        gl = max(gl, 0.6)
    if s == "rave" and (i % FPB) < 2:
        gl = max(gl, 0.5)
    if gl:
        a = glitch(a, rng, gl)
        E = glitch(E, rng, gl)
    # Komposit RGB
    glowE = cv2.GaussianBlur(E, (0, 0), 5)
    glowR = cv2.GaussianBlur(R, (0, 0), 6)
    tint = np.array(TINT[s], np.float32)
    rgb = a[..., None] * tint + (E + glowE * 0.9)[..., None] * CYAN + (R + glowR * 0.8)[..., None] * REDC
    # RGB-Split auf Beats
    if e > 0.6 and beat < 0.15:
        sh = int(6 * e)
        rgb[..., 0] = np.roll(rgb[..., 0], sh, axis=1); rgb[..., 2] = np.roll(rgb[..., 2], -sh, axis=1)
    # Korn
    g = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
    rgb += cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)[..., None] * 0.035
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8).tobytes()


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
