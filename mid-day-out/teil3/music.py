"""Soundtrack "mid day out III - Deus Ex Machina": Industrial / Cyberpunk / Electro, 140 BPM.

175 Takte = exakt 300 s. Alles synthetisiert (inkl. Roboter-"Stimme" per Formant-Synthese).
Story-Abschnitte (Takte):
  0-8 Boot | 8-24 Erwachen | 24-48 Reise | 48-80 Suche in Ruinen | 80-104 Tauchgang
  104-120 Gefunden | 120-130 Glueck | 130-140 Trauer | 140-152 Liebe | 152-171 Finale | 171-175 Ende
"""
import sys
import wave
import numpy as np

SR = 44100
BPM = 140
BEAT = SR * 60 // BPM            # 18900
BAR = BEAT * 4
BARS = 175
TOTAL = BAR * BARS
TAIL = SR * 5
N = TOTAL + TAIL
rng = np.random.default_rng(2052)
dL = np.zeros(N); dR = np.zeros(N); rev = np.zeros(N)
S16 = BEAT // 4
E8 = BEAT // 2


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(sig, start, pan=0.0, gain=1.0, send=0.0):
    start = int(start)
    n = min(len(sig), N - start)
    if n <= 0:
        return
    s = sig[:n] * gain
    dL[start:start + n] += s * np.sqrt(0.5 * (1 - pan))
    dR[start:start + n] += s * np.sqrt(0.5 * (1 + pan))
    if send:
        rev[start:start + n] += s * send


def adsr(n, a=0.003, d=0.1, s=0.6, r=0.03):
    t = np.arange(n) / SR
    dur = n / SR
    e = np.where(t < a, t / max(a, 1e-6), np.where(t < a + d, 1 - (1 - s) * (t - a) / max(d, 1e-6), s))
    rs = max(dur - r, 0)
    return np.where(t > rs, e * np.clip(1 - (t - rs) / max(r, 1e-6), 0, 1), e)


def lp(x, cutoff):
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    a = 1 - np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def fft_filter(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR) + 1e-9
    if hi:
        X *= 1 / np.sqrt(1 + (f / hi) ** 4)
    if lo:
        X *= 1 / np.sqrt(1 + (lo / f) ** 4)
    return np.fft.irfft(X, len(x))


def resonator(x, f, bw):
    """2-Pol-Bandpass (fuer Formanten)."""
    r = np.exp(-np.pi * bw / SR)
    c1, c2 = 2 * r * np.cos(2 * np.pi * f / SR), -r * r
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    g = 1 - r
    for i in range(len(x)):
        y0 = g * x[i] + c1 * y1 + c2 * y2
        y[i] = y0
        y2, y1 = y1, y0
    return y


def saw(f, n, ph=0.0):
    t = np.arange(n) / SR
    return 2 * ((f * t + ph) % 1.0) - 1


def pulse(f, n, duty=0.5):
    t = np.arange(n) / SR
    return np.where((f * t) % 1.0 < duty, 1.0, -1.0)


# --------------------------------------------------------------- Drums (industrial)
def kick(hard=1.0):
    n = int(0.4 * SR); t = np.arange(n) / SR
    f = 45 + 160 * np.exp(-t * 40)
    b = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    c = rng.standard_normal(n) * np.exp(-t * 300) * 0.4
    return np.tanh((b + c) * (1.8 + hard)) * 0.85


def snare():
    n = int(0.3 * SR); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    hp = nz - lp(nz, 1500)
    ring = np.sin(2 * np.pi * 210 * t) * np.exp(-t * 30) + 0.5 * np.sin(2 * np.pi * 330 * t) * np.exp(-t * 40)
    return np.tanh((hp * np.exp(-t * 16) + ring * 0.6) * 1.8) * 0.6


def clank():
    """Metallischer Industrial-Schlag (inharmonische Partialtoene)."""
    n = int(0.5 * SR); t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, d in ((523, 9), (1307, 12), (2213, 16), (3571, 22), (787, 10)))
    return x * 0.18 + rng.standard_normal(n) * np.exp(-t * 80) * 0.15


def hat(open_=False):
    n = int((0.2 if open_ else 0.04) * SR); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    return (nz - lp(nz, 8000)) * np.exp(-t * (13 if open_ else 80)) * 0.28


def impact():
    n = int(2.5 * SR); t = np.arange(n) / SR
    f = 30 + 80 * np.exp(-t * 6)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.6)
    nz = fft_filter(rng.standard_normal(n), hi=3000) * np.exp(-t * 3)
    return np.tanh((boom + nz * 0.4) * 2) * 0.9


def riser(n):
    t = np.arange(n) / SR
    k = t / t[-1]
    nz = rng.standard_normal(n)
    sweep = np.sin(2 * np.pi * np.cumsum(200 + 2500 * k ** 2) / SR)
    return (fft_filter(nz, lo=500) * 0.3 + sweep * 0.2) * k ** 2


def jet(n):
    nz = rng.standard_normal(n)
    t = np.arange(n) / SR
    k = t / t[-1]
    x = fft_filter(nz, lo=80, hi=900)
    return x * np.sin(np.pi * k) ** 0.7 * 0.5


def bubbles(n):
    x = np.zeros(n)
    for _ in range(int(n / SR * 12)):
        p = rng.integers(0, n - 4000)
        ln = 3000
        t = np.arange(ln) / SR
        f = rng.uniform(400, 1200) * (1 + 3 * t)
        x[p:p + ln] += np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 60) * 0.3
    return x


K1, K2, SN, CL, HC, HO = kick(0.6), kick(1.6), snare(), clank(), hat(), hat(True)

# --------------------------------------------------------------- Harmonie
DARK = [(40, [64, 67, 71]), (36, [64, 67, 72]), (45, [64, 69, 72]), (47, [63, 66, 71])]   # Em C Am B
LOVE = [(43, [67, 71, 74]), (38, [66, 69, 74]), (40, [64, 67, 71]), (36, [64, 67, 72])]   # G D Em C


def sec(bar):
    for name, a, b in (("boot", 0, 8), ("wake", 8, 24), ("travel", 24, 48), ("search", 48, 80), ("dive", 80, 104),
                       ("found", 104, 120), ("joy", 120, 130), ("sad", 130, 140), ("love", 140, 152),
                       ("finale", 152, 171), ("end", 171, 175)):
        if a <= bar < b:
            return name


def chord(bar):
    s = sec(bar)
    prog = LOVE if s in ("joy", "love", "finale", "found") else DARK
    return prog[bar % 4] if s in ("travel", "search", "finale") else prog[(bar // 2) % 4]


# --------------------------------------------------------------- Instrumente
def bass(m, n, acc, bright, dist=1.5):
    f = mtof(m)
    x = saw(f, n) * 0.6 + pulse(f * 0.5, n, 0.3) * 0.4
    t = np.arange(n) / SR
    cut = 100 + (400 + 3000 * bright) * acc * np.exp(-t * 25)
    y = lp(lp(x, cut), cut)
    return np.tanh(y * dist) * adsr(n, 0.002, 0.05, 0.7, 0.01) * 0.55


def arp(m, n, cut=3000):
    x = pulse(mtof(m), n, 0.22) * 0.6 + saw(mtof(m) * 1.004, n) * 0.4
    return fft_filter(x, hi=cut) * adsr(n, 0.001, 0.06, 0.3, 0.01) * 0.12


def lead(m, n, bright=0.6):
    t = np.arange(n) / SR
    f = mtof(m) * (1 + 0.005 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.3, 0, 1))
    ph = np.cumsum(f) / SR
    x = (2 * (ph % 1) - 1) * 0.5 + (2 * ((ph * 1.007) % 1) - 1) * 0.5 + np.sign(np.sin(2 * np.pi * ph * 0.5)) * 0.25
    return fft_filter(x, hi=1200 + 3000 * bright) * adsr(n, 0.01, 0.15, 0.8, 0.08) * 0.2


def pad(notes, n, cut=1500):
    x = np.zeros(n)
    for m in notes:
        for det in (-0.12, 0, 0.12):
            x += saw(mtof(m) * 2 ** (det / 12), n, rng.uniform())
    x = fft_filter(x / (len(notes) * 3), hi=cut)
    return x * adsr(n, 0.6, 0.3, 0.85, 0.7) * 0.4


def beep(f, n=int(0.06 * SR)):
    return pulse(f, n, 0.5) * adsr(n, 0.001, 0.02, 0.7, 0.005) * 0.08


VOWELS = {"a": (800, 1200), "e": (500, 1900), "i": (300, 2300), "o": (500, 900), "u": (320, 800), "m": (250, 1000)}


def robot(word, f0=110, syl=0.17):
    """Roboter-Silben: Pulswelle durch zwei Formant-Resonatoren."""
    out = []
    for ch in word:
        n = int(syl * SR)
        if ch == " ":
            out.append(np.zeros(n // 2)); continue
        f1, f2 = VOWELS.get(ch, (600, 1500))
        src = pulse(f0, n, 0.12) + 0.2 * rng.standard_normal(n)
        y = resonator(src, f1, 90) * 3 + resonator(src, f2, 120) * 2
        out.append(np.tanh(y * 2) * adsr(n, 0.005, 0.05, 0.8, 0.02))
    return np.concatenate(out) * 0.35


# --------------------------------------------------------------- Melodien (Achtel-Schritte innerhalb 2 Takte)
MEL_S = [[(0, 3, 76), (3, 3, 74), (6, 2, 71), (8, 6, 67), (14, 2, 71)],
         [(0, 3, 72), (3, 3, 71), (6, 2, 67), (8, 8, 64)],
         [(0, 2, 69), (2, 2, 72), (4, 4, 76), (8, 3, 74), (11, 5, 72)],
         [(0, 6, 71), (6, 2, 75), (8, 8, 78)]]
MEL_L = [[(0, 4, 74), (4, 4, 79), (8, 8, 83)],
         [(0, 4, 81), (4, 4, 78), (8, 8, 74)],
         [(0, 4, 76), (4, 4, 79), (8, 4, 83), (12, 4, 86)],
         [(0, 12, 84), (12, 4, 83)]]

ROBOT_LINES = {1: ("aiaa oe", 98), 9: ("i a e", 110), 25: ("u o e a", 92), 49: ("e i i a", 104),
               105: ("aaa ue", 98), 121: ("o a i", 116), 141: ("o e u i", 110), 171: ("e o a", 92)}

for bar in range(BARS):
    s = sec(bar)
    b0 = bar * BAR
    root, tones = chord(bar)
    first = bar % 2 == 0
    pi = (bar // 2) % 4

    # Boot-Beeps (Computer startet)
    if s == "boot":
        for st in range(16):
            if rng.random() < 0.45:
                add(beep(rng.choice([880, 1320, 1760, 2640])), b0 + st * S16, pan=rng.uniform(-.7, .7), send=0.2)
    # Drums
    drums = s in ("wake", "travel", "search", "found", "joy", "love", "finale") or (s == "dive" and bar < 84)
    if s == "wake" and bar < 16:
        add(K1, b0, gain=0.8); add(K1, b0 + 2 * BEAT, gain=0.8)
    elif drums:
        kk = K2 if s in ("search", "finale", "found") else K1
        for beat in range(4):
            add(kk, b0 + beat * BEAT, gain=0.85)
        if s in ("search", "finale"):          # Breakbeat-Kick-Ghost
            add(K1, b0 + 2 * BEAT + E8 + S16, gain=0.5)
        for beat in (1, 3):
            add(SN, b0 + beat * BEAT, gain=0.75, send=0.35)
        if s in ("search", "finale", "travel"):
            add(SN, b0 + 3 * BEAT + E8 + S16, gain=0.35, send=0.2)
        for st in range(16):
            if st % 4 == 2:
                add(HO, b0 + st * S16, pan=0.3, gain=0.5)
            else:
                add(HC, b0 + st * S16, pan=0.3, gain=0.8 if st % 2 == 0 else 0.4)
        if s in ("search", "finale") and bar % 2 == 1:
            add(CL, b0 + BEAT + E8, pan=-0.5, gain=0.8, send=0.5)
            add(CL, b0 + 3 * BEAT + S16, pan=0.5, gain=0.6, send=0.5)
    if s == "sad":                              # Halbtempo
        add(K1, b0, gain=0.7); add(SN, b0 + 2 * BEAT, gain=0.6, send=0.7)
        for st in range(0, 16, 2):
            add(HC, b0 + st * S16, pan=0.3, gain=0.4)
    if s == "dive" and bar >= 84 and bar < 100:
        add(K1, b0, gain=0.5); add(K1, b0 + 2 * BEAT + E8, gain=0.35)
    # Snare-Wirbel vor dem Fund
    if 100 <= bar < 104:
        div = 4 if bar < 102 else 8
        for i in range(4 * div):
            add(SN, b0 + i * BEAT // div, gain=0.25 + 0.5 * ((bar - 100) * 4 * div + i) / (16 * div), send=0.3)
    # Bass
    if s in ("wake", "travel", "search", "found", "joy", "love", "finale") and not (s == "wake" and bar < 12):
        bright = {"wake": 0.25, "travel": 0.5, "search": 0.65, "found": 0.7, "joy": 0.8, "love": 0.7, "finale": 0.9}[s]
        pat = [0, 0, 12, 0, 0, 12, 0, 7, 0, 0, 12, 0, 10, 12, 0, 7] if s in ("search", "finale") else \
              [0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 12, 7, 0]
        for st in range(16):
            add(bass(root + pat[st], S16 - 250, 1.0 if st % 4 == 0 else 0.6, bright), b0 + st * S16, gain=0.9)
    elif s in ("dive", "sad", "boot", "end") and first:
        add(fft_filter(saw(mtof(root), 2 * BAR) + saw(mtof(root) * 1.005, 2 * BAR), hi=220) * adsr(2 * BAR, 0.5, 0.3, 0.9, 0.8) * 0.4,
            b0, send=0.3)
    # Arpeggio (Deus-Ex-artig, 16tel, gefiltert)
    if s in ("travel", "search", "joy", "love", "finale") or (s == "wake" and bar >= 16) or (s == "dive" and bar < 88):
        notes = tones + [tones[0] + 12, tones[2], tones[1] + 12, tones[0] + 12, tones[1]]
        cut = {"wake": 1500, "travel": 2500, "search": 3500, "dive": 900, "joy": 4500, "love": 3000, "finale": 5000}[s]
        for st in range(16):
            add(arp(notes[st % len(notes)] + 12, S16 - 150, cut), b0 + st * S16, pan=-0.5 if st % 2 else 0.5, send=0.15)
    # Pads
    if first and s in ("boot", "wake", "dive", "found", "sad", "love", "finale", "end"):
        cut = {"boot": 600, "wake": 900, "dive": 500, "found": 2000, "sad": 900, "love": 2200, "finale": 2500, "end": 700}[s]
        add(pad(tones, 2 * BAR, cut), b0, gain=0.7, send=0.6)
    # Lead
    if first:
        mel = None
        if s == "search" and bar >= 56:
            mel = MEL_S
        elif s in ("found", "love") or (s == "finale" and bar < 168):
            mel = MEL_L
        elif s == "sad":
            mel = MEL_S
        if mel:
            for st, ln, m in mel[pi]:
                g = 0.7 if s == "sad" else 1.0
                add(lead(m - (12 if s == "sad" else 0), ln * E8, 0.3 if s == "sad" else 0.7), b0 + st * E8, pan=-0.1, gain=g, send=0.4)
    # Effekte
    if s == "travel" and bar % 8 == 0:
        add(jet(8 * BAR), b0, pan=0.2, gain=0.8, send=0.2)
    if s == "dive" and bar == 80:
        add(impact(), b0, gain=0.8, send=0.6)                       # Eintauchen
    if s == "dive":
        if first:
            add(bubbles(2 * BAR), b0, pan=rng.uniform(-.5, .5), gain=0.7, send=0.5)
        if bar % 4 == 0:
            t = np.arange(int(1.5 * SR)) / SR
            add(np.sin(2 * np.pi * 1250 * t) * np.exp(-t * 3) * 0.2, b0, pan=-0.3, send=0.9)   # Sonar
    if bar == 100:
        add(riser(4 * BAR), b0, gain=0.9, send=0.3)
    if bar in (104, 152):
        add(impact(), b0, gain=1.0, send=0.5)
    if bar in ROBOT_LINES:
        w, f0 = ROBOT_LINES[bar]
        add(robot(w, f0), b0, pan=0.0, gain=1.0, send=0.5)

# --------------------------------------------------------------- Unterwasser-Filter auf den Mix (Takte 80-104)
t_bar = np.arange(N) / BAR
uw = np.interp(t_bar, [0, 80, 81, 100, 104, 200], [0, 0, 1, 1, 0, 0])
for ch in (dL, dR):
    muff = fft_filter(ch, hi=450)
    ch[:] = ch * (1 - uw) + muff * uw * 1.4

# Echo (punktierte Achtel)
d = int(BEAT * 0.75)
for lvl, k in ((0.22, 1), (0.1, 2)):
    sh = d * k
    dL[sh:] += dR[:-sh] * lvl
    dR[sh:] += dL[:-sh] * lvl * 0.9

# Hall
ir_n = int(2.6 * SR); t_ir = np.arange(ir_n) / SR
size = 1 << int(np.ceil(np.log2(N + ir_n)))
RV = np.fft.rfft(rev, size)
wet = []
for seed in (3, 4):
    ir = np.random.default_rng(seed).standard_normal(ir_n) * np.exp(-t_ir * 2.6)
    ir[:int(0.015 * SR)] = 0
    wet.append(np.fft.irfft(RV * np.fft.rfft(fft_filter(ir, hi=6000), size), size)[:N])
L = dL + wet[0] * 0.03
R = dR + wet[1] * 0.03

L[:TAIL] += L[TOTAL:]; R[:TAIL] += R[TOTAL:]
L, R = L[:TOTAL], R[:TOTAL]
L = fft_filter(L, lo=25, hi=14000); R = fft_filter(R, lo=25, hi=14000)
pk = max(np.abs(L).max(), np.abs(R).max())
L = np.tanh(L / pk * 1.8) * 0.9; R = np.tanh(R / pk * 1.8) * 0.9

out = sys.argv[1] if len(sys.argv) > 1 else "build/soundtrack.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.stack([L, R], 1) * 32767).astype(np.int16).tobytes())
print("wrote", out, TOTAL / SR, "s")
