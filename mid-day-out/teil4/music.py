"""Soundtrack "mid day out IV - EAT SLEEP FILM REPEAT": dunkler Hard-Techno / Rave, 144 BPM, f-Moll.

180 Takte = exakt 300 s. Alles synthetisiert: Rumble-Kick, Acid-Linie (resonanter Filter), Industrial-Hits,
Herzschlag/EKG im Tod-Teil, Roboter-Chant "EAT SLEEP FILM REPEAT" (Formant-Synthese).
Abschnitte: 0-8 Intro | 8-32 Leben | 32-56 Strom | 56-80 Drohnen | 80-96 Tod | 96-104 Aufladen
            104-136 Verwandlung | 136-168 Rave | 168-180 Repeat
"""
import sys
import wave
import numpy as np

SR = 44100
BPM = 144
BEAT = SR * 60 // BPM            # 18375
BAR = BEAT * 4
BARS = 180
TOTAL = BAR * BARS
TAIL = SR * 4
N = TOTAL + TAIL
S16, E8 = BEAT // 4, BEAT // 2
rng = np.random.default_rng(666)
dL = np.zeros(N); dR = np.zeros(N); rev = np.zeros(N)
duck = np.ones(N)                 # Sidechain-Huellkurve


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(sig, start, pan=0.0, gain=1.0, send=0.0, duckable=True):
    start = int(start)
    n = min(len(sig), N - start)
    if n <= 0:
        return
    s = sig[:n] * gain
    if duckable:
        s = s * duck[start:start + n]
    dL[start:start + n] += s * np.sqrt(0.5 * (1 - pan))
    dR[start:start + n] += s * np.sqrt(0.5 * (1 + pan))
    if send:
        rev[start:start + n] += s * send


def adsr(n, a=0.002, d=0.1, s=0.6, r=0.02):
    t = np.arange(n) / SR
    dur = n / SR
    e = np.where(t < a, t / max(a, 1e-6), np.where(t < a + d, 1 - (1 - s) * (t - a) / max(d, 1e-6), s))
    rs = max(dur - r, 0)
    return np.where(t > rs, e * np.clip(1 - (t - rs) / max(r, 1e-6), 0, 1), e)


def lp(x, cutoff):
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    a = 1 - np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x); acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc); y[i] = acc
    return y


def svf(x, cutoff, q):
    """Resonanter State-Variable-Tiefpass (Acid)."""
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    f = 2 * np.sin(np.pi * np.minimum(cutoff, SR / 6) / SR)
    damp = 1.0 / q
    low = band = 0.0
    y = np.empty_like(x)
    for i in range(len(x)):
        low += f[i] * band
        high = x[i] - low - damp * band
        band += f[i] * high
        y[i] = low
    return y


def fft_filter(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / SR) + 1e-9
    if hi:
        X *= 1 / np.sqrt(1 + (fr / hi) ** 4)
    if lo:
        X *= 1 / np.sqrt(1 + (lo / fr) ** 4)
    return np.fft.irfft(X, len(x))


def resonator(x, f, bw):
    r = np.exp(-np.pi * bw / SR)
    c1, c2 = 2 * r * np.cos(2 * np.pi * f / SR), -r * r
    y = np.zeros_like(x); y1 = y2 = 0.0; g = 1 - r
    for i in range(len(x)):
        y0 = g * x[i] + c1 * y1 + c2 * y2
        y[i] = y0; y2, y1 = y1, y0
    return y


def saw(f, n, ph=0.0):
    t = np.arange(n) / SR
    return 2 * ((np.asarray(f) * t + ph) % 1.0) - 1


def pulse(f, n, duty=0.5):
    t = np.arange(n) / SR
    return np.where((f * t) % 1.0 < duty, 1.0, -1.0)


# ------------------------------------------------------------------ Drums
def kick():
    n = int(0.5 * SR); t = np.arange(n) / SR
    f = 42 + 190 * np.exp(-t * 45) + 30 * np.exp(-t * 8)
    b = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5.5)
    c = rng.standard_normal(n) * np.exp(-t * 250) * 0.5
    return np.tanh((b + c) * 3.2) * 0.8


def rumble(root):
    """Hall-Rumble zwischen den Kicks (Hard-Techno)."""
    n = int(BEAT * 0.75); t = np.arange(n) / SR
    x = np.sin(2 * np.pi * mtof(root) * t) + 0.5 * np.sin(2 * np.pi * mtof(root) * 2.01 * t)
    x += fft_filter(rng.standard_normal(n), hi=180) * 3
    env = np.sin(np.pi * np.clip(t / (n / SR), 0, 1)) ** 0.8
    return np.tanh(x * env * 1.5) * 0.35


def clap():
    n = int(0.35 * SR); t = np.arange(n) / SR
    nz = rng.standard_normal(n); hp = nz - lp(nz, 1100)
    env = np.exp(-t * 14)
    for off in (0, .009, .018, .027):
        env += 0.7 * np.exp(-np.clip(t - off, 0, None) * 180) * (t >= off)
    return np.tanh(hp * env * 1.5) * 0.45


def hat(open_=False):
    n = int((0.16 if open_ else 0.035) * SR); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    return (nz - lp(nz, 9000)) * np.exp(-t * (16 if open_ else 90)) * (0.32 if open_ else 0.25)


def metal():
    n = int(0.6 * SR); t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, d in ((311, 6), (1033, 9), (1789, 12), (2950, 15), (4471, 20)))
    return np.tanh(x * 0.6 + rng.standard_normal(n) * np.exp(-t * 60) * 0.4) * 0.3


def heart():
    n = int(0.45 * SR); t = np.arange(n) / SR
    def thump(t0, g):
        tt = np.clip(t - t0, 0, None)
        return np.sin(2 * np.pi * (50 + 40 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 18) * (t >= t0) * g
    return np.tanh((thump(0, 1.0) + thump(0.16, 0.7)) * 2) * 0.7


def ecg_beep(n=int(0.09 * SR)):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 1000 * t) * adsr(n, 0.002, 0.01, 0.9, 0.005) * 0.2


def impact():
    n = int(3 * SR); t = np.arange(n) / SR
    f = 28 + 90 * np.exp(-t * 5)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.3)
    nz = fft_filter(rng.standard_normal(n), hi=4000) * np.exp(-t * 2.5)
    return np.tanh((boom + nz * 0.5) * 2.5) * 0.9


def riser(n):
    t = np.arange(n) / SR; k = t / t[-1]
    nz = fft_filter(rng.standard_normal(n), lo=800)
    sw = saw(150 + 1800 * k ** 2, n) * 0.0
    sweep = np.sin(2 * np.pi * np.cumsum(150 + 1800 * k ** 2) / SR)
    return (nz * 0.35 + sweep * 0.25 + sw) * k ** 2.5


def scream(n):
    """Detunte Saegezahn-Wand mit Pitch-Dive."""
    t = np.arange(n) / SR; k = t / t[-1]
    f = mtof(89) * (1 - 0.75 * k ** 1.5)
    ph = np.cumsum(f) / SR
    x = sum((2 * ((ph * d) % 1) - 1) for d in (1, 1.01, 0.99, 1.02))
    return fft_filter(np.tanh(x), hi=6000) * np.sin(np.pi * k) * 0.12


K, CL, HC, HO, MT = kick(), clap(), hat(), hat(True), metal()

# ------------------------------------------------------------------ Struktur
ROOTS = [41, 41, 44, 39]          # F F Ab Eb (Takt-weise, 4er Zyklus)


def sec(bar):
    for name, a, b in (("intro", 0, 8), ("life", 8, 32), ("current", 32, 56), ("drones", 56, 80), ("death", 80, 96),
                       ("charge", 96, 104), ("trans", 104, 136), ("rave", 136, 168), ("repeat", 168, 180)):
        if a <= bar < b:
            return name


ACID = [0, 12, 0, 0, 3, 0, 15, 0, 0, 12, 7, 0, 10, 0, 12, 3]
ACC = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
SLIDE = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]


def acid_bar(root, cut_base, q, dist):
    """Ein Takt Acid (16 Schritte) als zusammenhaengendes Signal mit Slides."""
    n = BAR
    freq = np.empty(n); env = np.empty(n); amp = np.empty(n)
    prev = mtof(root + ACID[-1] - 12)
    for st in range(16):
        a, b = st * S16, (st + 1) * S16
        f = mtof(root + ACID[st] - 12)
        tt = np.arange(b - a) / SR
        if SLIDE[st]:
            freq[a:b] = f + (prev - f) * np.exp(-tt * 60)
        else:
            freq[a:b] = f
        prev = f
        env[a:b] = np.exp(-tt * (18 if ACC[st] else 30)) * (1.6 if ACC[st] else 1.0)
        gate = (tt < (S16 / SR) * (1.0 if SLIDE[(st + 1) % 16] else 0.6)).astype(float)
        amp[a:b] = gate * (1.2 if ACC[st] else 0.8)
    amp = lp(amp, 300)
    ph = np.cumsum(freq) / SR
    x = 2 * (ph % 1) - 1
    y = svf(x, cut_base * (1 + 6 * env), q)
    return np.tanh(y * dist) * amp * 0.28


def pad(notes, n, cut=900):
    x = np.zeros(n)
    for m in notes:
        for det in (-0.15, 0, 0.15):
            x += saw(mtof(m) * 2 ** (det / 12), n, rng.uniform())
    return fft_filter(x / (len(notes) * 3), hi=cut) * adsr(n, 1.0, 0.3, 0.9, 1.0) * 0.45


SYL = {"EAT": [("i", 0.16, 0), ("t", 0.03, 1)], "SLEEP": [("s", 0.07, 1), ("i", 0.15, 0), ("p", 0.03, 1)],
       "FILM": [("f", 0.06, 1), ("I", 0.12, 0), ("m", 0.08, 0)], "RE": [("e", 0.11, 0)],
       "PEAT": [("p", 0.03, 1), ("i", 0.15, 0), ("t", 0.03, 1)]}
FORM = {"i": (290, 2300), "I": (420, 2000), "e": (520, 1850), "m": (260, 1100)}


def word(w, f0):
    out = []
    for ph, dur, cons in SYL[w]:
        n = int(dur * SR)
        if cons:
            nz = rng.standard_normal(n)
            if ph == "s":
                x = fft_filter(nz, lo=4000) * 0.5
            elif ph == "f":
                x = fft_filter(nz, lo=1500, hi=6000) * 0.35
            else:
                x = fft_filter(nz, lo=1000) * np.exp(-np.arange(n) / SR * 120) * 0.7
            out.append(x)
        else:
            f1, f2 = FORM[ph]
            src = pulse(f0, n, 0.1) + 0.15 * rng.standard_normal(n)
            y = resonator(src, f1, 80) * 3 + resonator(src, f2, 110) * 2.2 + resonator(src, 3000, 200)
            out.append(np.tanh(y * 2.5) * adsr(n, 0.004, 0.03, 0.85, 0.02))
    return np.concatenate(out) * 0.45


CHANT = {w: word(w, 82) for w in SYL}
CHANT_HI = {w: word(w, 164) for w in SYL}


def chant(b0, hi=False):
    c = CHANT_HI if hi else CHANT
    add(c["EAT"], b0, gain=1.0, send=0.4, duckable=False)
    add(c["SLEEP"], b0 + BEAT, gain=1.0, send=0.4, duckable=False)
    add(c["FILM"], b0 + 2 * BEAT, gain=1.0, send=0.4, duckable=False)
    add(c["RE"], b0 + 3 * BEAT, gain=1.0, send=0.4, duckable=False)
    add(c["PEAT"], b0 + 3 * BEAT + E8, gain=1.0, send=0.6, duckable=False)


# Sidechain-Huellkurve vorab (wo die Kick spielt)
def kick_on(bar):
    s = sec(bar)
    return s in ("life", "current", "drones", "trans", "rave") or (s == "intro" and bar >= 4) or (s == "repeat" and bar < 176)


for bar in range(BARS):
    if kick_on(bar):
        for b in range(4):
            p = bar * BAR + b * BEAT
            t = np.arange(BEAT) / SR
            duck[p:p + BEAT] = np.minimum(duck[p:p + BEAT], 1 - 0.7 * np.exp(-t * 14))

for bar in range(BARS):
    s = sec(bar)
    b0 = bar * BAR
    root = ROOTS[bar % 4]
    tones = [root + 24, root + 27, root + 31] if root != 44 else [root + 24, root + 28, root + 31]

    # ---------------- Kick + Rumble
    if kick_on(bar):
        for b in range(4):
            add(K, b0 + b * BEAT, gain=0.95, duckable=False)
            if s in ("current", "drones", "rave", "trans"):
                add(rumble(root - 12), b0 + b * BEAT + E8 // 2, gain=1.0, duckable=False)
    # ---------------- Hats / Clap
    if s in ("life", "current", "drones", "trans", "rave") or (s == "repeat" and bar < 176):
        for b in range(4):
            add(HO, b0 + b * BEAT + E8, pan=0.2, gain=0.9)
        for st in range(16):
            if s in ("current", "drones", "rave") and st % 2 == 1:
                add(HC, b0 + st * S16, pan=-0.3, gain=0.6)
        if s not in ("life",) or bar >= 16:
            add(CL, b0 + BEAT, gain=0.8, send=0.3); add(CL, b0 + 3 * BEAT, gain=0.8, send=0.3)
        if s in ("drones", "rave") and bar % 2:
            add(MT, b0 + 2 * BEAT + E8 + S16, pan=-0.6, gain=0.8, send=0.5)
            add(MT, b0 + 3 * BEAT + S16, pan=0.6, gain=0.6, send=0.5)
    # ---------------- Acid
    if s in ("current", "drones", "trans", "rave") or (s == "life" and bar >= 24) or (s == "charge"):
        prog = {"life": 0.0, "current": (bar - 32) / 24, "drones": 0.6, "charge": (bar - 96) / 8,
                "trans": 0.5 + 0.4 * np.sin((bar - 104) / 32 * np.pi), "rave": 0.9}[s]
        cut = 180 + 900 * prog + 250 * np.sin(bar * 0.7) ** 2
        add(acid_bar(root + 12, cut, 4 + 6 * prog, 2 + 3 * prog), b0, pan=0.0, gain=1.0, send=0.15)
    # ---------------- Drone / Pad
    if bar % 4 == 0 and s in ("intro", "death", "charge", "trans", "repeat", "life"):
        g = {"intro": 0.9, "death": 1.0, "charge": 0.6, "trans": 0.5, "repeat": 0.8, "life": 0.4}[s]
        add(pad(tones, 4 * BAR, 700 if s != "trans" else 1300), b0, gain=g, send=0.6)
        t = np.arange(4 * BAR) / SR
        sub = fft_filter(saw(mtof(root - 12), 4 * BAR) + saw(mtof(root - 12) * 1.004, 4 * BAR), hi=150)
        add(sub * adsr(4 * BAR, 1, .3, .9, 1) * 0.4, b0, gain=g, send=0.2)
    # ---------------- Tod: Herzschlag wird langsamer, dann Flatline
    if s == "death":
        k = (bar - 80) / 16
        if k < 0.7:
            nb = 2 if k < 0.35 else 1
            for j in range(nb):
                add(heart(), b0 + j * 2 * BEAT, gain=0.9, send=0.3, duckable=False)
                add(ecg_beep(), b0 + j * 2 * BEAT + int(0.05 * SR), gain=1.0, duckable=False)
        elif bar == 91:
            n = int(5 * BAR)
            t = np.arange(n) / SR
            add(np.sin(2 * np.pi * 1000 * t) * 0.16 * np.clip(1 - t / (n / SR), 0, 1) ** 0.3, b0, send=0.3, duckable=False)
    if s == "charge":
        add(heart(), b0, gain=0.8, duckable=False); add(heart(), b0 + 2 * BEAT, gain=0.8, duckable=False)
        if bar == 96:
            add(riser(8 * BAR), b0, gain=0.9, send=0.3, duckable=False)
        if bar >= 100:
            for j in range(4 * (2 if bar < 102 else 4)):
                add(CL, b0 + j * BEAT // (2 if bar < 102 else 4), gain=0.3 + 0.08 * (bar - 100), send=0.2, duckable=False)
    # ---------------- Effekte
    if bar in (8, 32, 56, 104, 136, 168):
        add(impact(), b0, gain=0.9, send=0.5, duckable=False)
    if bar in (31, 55, 79, 135, 167):
        add(riser(BAR), b0, gain=0.8, send=0.3, duckable=False)
    if s in ("drones",) and bar % 4 == 0:
        add(scream(2 * BAR), b0, pan=0.4 * (-1) ** (bar // 4), gain=0.8, send=0.4)
    if s == "rave" and bar % 8 == 4:
        add(scream(4 * BAR), b0, gain=0.9, send=0.4)
    # Drohnen-Surren
    if s == "drones" and bar % 2 == 0:
        n = 2 * BAR; t = np.arange(n) / SR
        buzz = saw(180 + 8 * np.sin(2 * np.pi * 0.3 * t), n) + saw(183, n)
        add(fft_filter(buzz, lo=150, hi=1800) * 0.06 * np.sin(np.pi * t / t[-1]), b0, pan=np.sin(bar) * 0.7, send=0.2)
    # ---------------- Chant
    if (s == "intro" and bar % 2 == 1) or (s == "rave" and bar % 2 == 0) or (s == "repeat" and bar % 2 == 0) \
            or (s == "life" and bar in (8, 16, 24)) or (s == "trans" and bar % 8 == 0):
        chant(b0, hi=(s == "rave" and bar % 4 == 2))

# ------------------------------------------------------------------ Tod-Teil dumpf (Filter auf Mix)
tb = np.arange(N) / BAR
muf = np.interp(tb, [0, 80, 81, 95, 96, 999], [0, 0, 1, 1, 0, 0])
for ch in (dL, dR):
    ch[:] = ch * (1 - muf) + fft_filter(ch, hi=600) * muf * 1.3

# Hall
ir_n = int(2.8 * SR); t_ir = np.arange(ir_n) / SR
size = 1 << int(np.ceil(np.log2(N + ir_n)))
RV = np.fft.rfft(rev, size)
wet = []
for seed in (5, 6):
    ir = np.random.default_rng(seed).standard_normal(ir_n) * np.exp(-t_ir * 2.3)
    ir[:int(0.02 * SR)] = 0
    wet.append(np.fft.irfft(RV * np.fft.rfft(fft_filter(ir, hi=5000), size), size)[:N])
L = dL + wet[0] * 0.03
R = dR + wet[1] * 0.03
L[:TAIL] += L[TOTAL:]; R[:TAIL] += R[TOTAL:]
L, R = L[:TOTAL], R[:TOTAL]
L = fft_filter(L, lo=28, hi=15000); R = fft_filter(R, lo=28, hi=15000)
pk = max(np.abs(L).max(), np.abs(R).max())
L = np.tanh(L / pk * 2.2) * 0.92; R = np.tanh(R / pk * 2.2) * 0.92

out = sys.argv[1] if len(sys.argv) > 1 else "build/soundtrack.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.stack([L, R], 1) * 32767).astype(np.int16).tobytes())
print("wrote", out, TOTAL / SR, "s")
