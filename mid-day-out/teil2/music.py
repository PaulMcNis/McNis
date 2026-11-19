"""Soundtrack "mid day out II": Darkwave / Synth / Kraftwerk / Retro-Game.

100 BPM, 125 Takte = exakt 300 s (5 min). Alles synthetisiert.
Dramaturgie: dunkle Welt (d-Moll) -> Suche -> Licht (F-Dur-Teil) -> Ausklang.
"""
import sys
import wave
import numpy as np

SR = 44100
BPM = 100
BEAT = SR * 60 // BPM            # 26460
BAR = BEAT * 4
BARS = 125
TOTAL = BAR * BARS               # 300 s
TAIL = SR * 6
N = TOTAL + TAIL
rng = np.random.default_rng(1983)

dry_L = np.zeros(N); dry_R = np.zeros(N)
rev = np.zeros(N)                # Hall-Send (mono)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(sig, start, pan=0.0, gain=1.0, send=0.0):
    start = int(start)
    n = min(len(sig), N - start)
    if n <= 0:
        return
    s = sig[:n] * gain
    dry_L[start:start + n] += s * np.sqrt(0.5 * (1 - pan))
    dry_R[start:start + n] += s * np.sqrt(0.5 * (1 + pan))
    if send:
        rev[start:start + n] += s * send


def adsr(n, a=0.005, d=0.1, s=0.6, r=0.05):
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


def fft_lp(x, cutoff):
    """Statischer Tiefpass per FFT (schnell fuer lange Signale)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (f / cutoff) ** 4)
    return np.fft.irfft(X, len(x))


def saw(f, n, ph=0.0):
    t = np.arange(n) / SR
    return 2 * ((f * t + ph) % 1.0) - 1


def pulse(f, n, duty=0.5):
    t = np.arange(n) / SR
    duty = np.broadcast_to(duty, (n,))
    return np.where((f * t) % 1.0 < duty, 1.0, -1.0)


# ------------------------------------------------------------------ Drums
def kick():
    n = int(0.45 * SR); t = np.arange(n) / SR
    f = 42 + 130 * np.exp(-t * 32)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6.5)
    click = rng.standard_normal(n) * np.exp(-t * 400) * 0.3
    return np.tanh((body + click) * 1.6) * 0.9


def snare():
    n = int(0.35 * SR); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    hp = nz - lp(nz, 1200)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25)
    return (hp * np.exp(-t * 14) * 0.7 + tone * 0.5) * 0.8


def hat(open_=False):
    n = int((0.25 if open_ else 0.05) * SR); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    return (nz - lp(nz, 7000)) * np.exp(-t * (10 if open_ else 65)) * 0.3


def tom(m):
    n = int(0.4 * SR); t = np.arange(n) / SR
    f = mtof(m) * (1 + 0.7 * np.exp(-t * 18))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8) * 0.6


K, SN, HC, HO = kick(), snare(), hat(), hat(True)

# ------------------------------------------------------------------ Harmonie
DARK = [(38, [62, 65, 69]), (34, [58, 62, 65]), (43, [62, 67, 70]), (45, [61, 64, 69])]      # Dm Bb Gm A
LIGHT = [(41, [60, 65, 69]), (36, [60, 64, 67]), (38, [62, 65, 69]), (34, [62, 65, 70])]     # F C Dm Bb


def chord(bar):
    prog = LIGHT if 96 <= bar < 121 else DARK
    return prog[(bar // 2) % 4]


def sec(bar):
    for name, a, b in (("intro", 0, 8), ("pulse", 8, 24), ("drive", 24, 40), ("leadA", 40, 56),
                       ("break", 56, 64), ("build", 64, 80), ("peak", 80, 96), ("light", 96, 121),
                       ("outro", 121, 125)):
        if a <= bar < b:
            return name


# ------------------------------------------------------------------ Instrumente
def bass(m, n, acc, bright):
    f = mtof(m)
    x = saw(f, n) * 0.7 + pulse(f * 0.5, n) * 0.3
    t = np.arange(n) / SR
    cut = 120 + (500 + 2200 * bright) * acc * np.exp(-t * 22)
    return lp(lp(x, cut), cut) * adsr(n, 0.002, 0.05, 0.6, 0.015) * 0.6


def lead(m, n, bright=0.5):
    f = mtof(m)
    t = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5 * t) * np.clip(t / 0.4, 0, 1)
    ph = np.cumsum(f * vib) / SR
    x = (2 * (ph % 1) - 1) * 0.5 + (2 * ((ph * 1.005) % 1) - 1) * 0.5
    x += pulse(f, n, 0.3 + 0.15 * np.sin(2 * np.pi * 0.7 * t)) * 0.3
    cut = 900 + 2500 * bright
    return lp(x, cut) * adsr(n, 0.02, 0.2, 0.75, 0.15) * 0.22


def chip(m, n):
    return pulse(mtof(m), n, 0.25) * adsr(n, 0.001, 0.05, 0.3, 0.01) * 0.09


def ping(m, n=int(1.5 * SR)):
    t = np.arange(n) / SR
    f = mtof(m)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2.01 * f * t)) * np.exp(-t * 3.5) * 0.25


def pad(notes, n, cut=1200):
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in notes:
        for det in (-0.1, 0.0, 0.1):
            x += saw(mtof(m) * 2 ** (det / 12), n, rng.uniform())
    x /= len(notes) * 3
    x = fft_lp(x, cut)
    return x * adsr(n, 0.8, 0.3, 0.85, 0.9) * 0.45


def drone(n, root=26):
    t = np.arange(n) / SR
    x = saw(mtof(root), n) * 0.5 + saw(mtof(root + 7) * 1.003, n) * 0.3 + saw(mtof(root + 12) * 0.997, n) * 0.3
    x = fft_lp(x, 300)
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * t / 9.6)
    return x * lfo * 0.5


def wind(n):
    nz = rng.standard_normal(n)
    t = np.arange(n) / SR
    x = fft_lp(nz, 700) - fft_lp(nz, 200)
    return x * (0.5 + 0.5 * np.sin(2 * np.pi * t / 7.3) ** 2) * 0.25


# ------------------------------------------------------------------ Melodien (16tel-Raster ueber 2 Takte = 32 Schritte; hier Achtel: 16 Schritte)
MEL_A = [[(0, 6, 74), (6, 2, 72), (8, 4, 69), (12, 4, 65)],
         [(0, 6, 70), (6, 2, 69), (8, 8, 65)],
         [(0, 4, 67), (4, 4, 70), (8, 4, 74), (12, 4, 72)],
         [(0, 8, 73), (8, 4, 76), (12, 4, 69)]]
MEL_B = [[(0, 4, 77), (4, 4, 76), (8, 8, 74)],
         [(0, 4, 74), (4, 2, 72), (6, 2, 70), (8, 8, 69)],
         [(0, 4, 70), (4, 4, 72), (8, 4, 74), (12, 4, 77)],
         [(0, 12, 76), (12, 4, 73)]]
MEL_L = [[(0, 4, 72), (4, 4, 77), (8, 8, 81)],
         [(0, 4, 79), (4, 4, 76), (8, 8, 72)],
         [(0, 4, 74), (4, 4, 77), (8, 4, 81), (12, 4, 84)],
         [(0, 12, 82), (12, 4, 81)]]

E8 = BEAT // 2
S16 = BEAT // 4

# Drone + Wind ueber die ganze Laenge (dynamisch per Huellkurve)
dn = drone(TOTAL)
wd = wind(TOTAL)
t_all = np.arange(TOTAL) / SR
bar_pos = t_all / (BAR / SR)
dr_env = np.interp(bar_pos, [0, 8, 24, 56, 64, 80, 96, 110, 121, 125], [1, 0.8, 0.4, 0.4, 1.0, 0.4, 0.3, 0.2, 0.8, 1.0])
wd_env = np.interp(bar_pos, [0, 8, 24, 56, 64, 72, 96, 121, 125], [1, 0.6, 0.2, 0.3, 0.9, 0.3, 0.1, 0.6, 1.0])
add(dn * dr_env, 0, gain=0.8, send=0.2)
add(wd * wd_env, 0, pan=0.3, gain=0.9, send=0.3)

for bar in range(BARS):
    s = sec(bar)
    b0 = bar * BAR
    root, tones = chord(bar)
    first = bar % 2 == 0          # erster Takt eines Akkords
    prog_i = (bar // 2) % 4

    # ---------------- Drums
    full = s in ("drive", "leadA", "build", "peak", "light")
    if s == "pulse" and bar >= 16 or full:
        for beat in range(4):
            add(K, b0 + beat * BEAT, gain=0.85)
    elif s == "pulse":
        add(K, b0, gain=0.7)
    if full:
        for beat in (1, 3):
            add(SN, b0 + beat * BEAT, pan=0.05, gain=0.7, send=0.5)
    if full or (s == "pulse" and bar >= 16):
        for st in range(16):
            if s in ("peak", "light") and st % 4 == 2:
                add(HO, b0 + st * S16, pan=0.35, gain=0.5, send=0.1)
            else:
                add(HC, b0 + st * S16, pan=0.35, gain=0.8 if st % 2 == 0 else 0.45)
    if full and bar % 8 == 7:
        for i, m in enumerate((50, 47, 43, 40, 38, 36)):
            add(tom(m), b0 + 2 * BEAT + i * (BEAT // 3), pan=-0.5 + 0.2 * i, gain=0.6, send=0.4)

    # ---------------- Bass (16tel-Puls, Kraftwerk/Darkwave)
    if s not in ("intro", "break", "outro"):
        bright = {"pulse": 0.1 + 0.3 * (bar - 8) / 16, "drive": 0.5, "leadA": 0.55, "build": 0.4 + 0.4 * (bar - 64) / 16,
                  "peak": 0.85, "light": 0.9}[s]
        pat = [0, 0, 12, 0, 0, 12, 0, 0, 0, 0, 12, 0, 0, 12, 0, 7]
        for st in range(16):
            acc = 1.0 if st % 4 == 0 else 0.6
            add(bass(root + pat[st], S16 - 300, acc, bright), b0 + st * S16, gain=0.9)
    elif s == "break" and first:
        add(bass(root, 2 * BAR - 3000, 0.2, 0.0), b0, gain=0.6, send=0.2)

    # ---------------- Pads
    if first and s in ("intro", "break", "peak", "light", "outro", "build"):
        cut = {"intro": 700, "break": 1000, "build": 900, "peak": 1500, "light": 2200, "outro": 700}[s]
        g = {"intro": 0.7, "break": 0.9, "build": 0.5, "peak": 0.6, "light": 0.8, "outro": 0.7}[s]
        add(pad(tones, 2 * BAR, cut), b0, gain=g, send=0.6)

    # ---------------- Retro-Game-Arpeggios
    if s in ("drive", "peak", "light") or (s == "pulse" and bar >= 20):
        arp = tones + [tones[0] + 12, tones[1] + 12, tones[2] + 12, tones[1] + 12, tones[0] + 12]
        for st in range(16):
            add(chip(arp[st % len(arp)] + 12, S16 - 200), b0 + st * S16, pan=-0.6 if st % 2 else 0.6,
                gain=1.0 if s == "drive" else 0.7, send=0.15)

    # ---------------- Sonar-Pings (Suche)
    if s in ("intro", "pulse", "break", "outro") and bar % 2 == 0:
        add(ping(tones[0] + 12), b0, pan=-0.4, gain=0.8, send=0.7)
        add(ping(tones[2] + 12), b0 + 3 * BEAT + E8, pan=0.4, gain=0.6, send=0.7)

    # ---------------- Lead
    mel = None
    if s == "leadA":
        mel, br = MEL_A, 0.4
    elif s == "build" and bar >= 72:
        mel, br = MEL_B, 0.5
    elif s == "peak":
        mel, br = (MEL_A if (bar // 8) % 2 == 0 else MEL_B), 0.7
    elif s == "light":
        mel, br = MEL_L, 0.9
    elif s == "break":
        mel, br = MEL_A, 0.2
    if mel is not None and first:
        for (st, ln, m) in mel[prog_i]:
            if s == "break":
                add(ping(m, int(ln * E8 + SR)), b0 + st * E8, pan=0.2, gain=0.9, send=0.8)
            else:
                add(lead(m, ln * E8), b0 + st * E8, pan=-0.1, gain=1.0, send=0.35)
                if s in ("peak", "light"):
                    add(lead(m - 12, ln * E8, 0.3), b0 + st * E8, pan=0.3, gain=0.5, send=0.3)

# ------------------------------------------------------------------ Echo (punktierte Achtel, Ping-Pong)
d = int(BEAT * 0.75)
for lvl, k in ((0.25, 1), (0.13, 2), (0.06, 3)):
    sh = d * k
    dry_L[sh:] += dry_R[:-sh] * lvl
    dry_R[sh:] += dry_L[:-sh] * lvl * 0.9

# ------------------------------------------------------------------ Hall per FFT-Faltung
ir_n = int(3.2 * SR)
t_ir = np.arange(ir_n) / SR
irs = []
for seed in (1, 2):
    r = np.random.default_rng(seed)
    ir = r.standard_normal(ir_n) * np.exp(-t_ir * 2.2)
    ir[: int(0.02 * SR)] = 0
    irs.append(fft_lp(ir, 5000))
size = 1 << int(np.ceil(np.log2(N + ir_n)))
R_sig = np.fft.rfft(rev, size)
wet = [np.fft.irfft(R_sig * np.fft.rfft(ir, size), size)[:N] for ir in irs]
L = dry_L + wet[0] * 0.035
R = dry_R + wet[1] * 0.035
del R_sig, wet

# Loop-Faltung
L[:TAIL] += L[TOTAL:]
R[:TAIL] += R[TOTAL:]
L, R = L[:TOTAL], R[:TOTAL]

# Tape-Wow (periodisch)
t = np.arange(TOTAL)
wob = 0.0012 * SR * np.sin(2 * np.pi * 60 * t / TOTAL)
src = (t + wob) % TOTAL
L = np.interp(src, t, L, period=TOTAL)
R = np.interp(src, t, R, period=TOTAL)

# Rauschen + Retro-Hoehenabsenkung + Saettigung
nz = rng.standard_normal(TOTAL) * 0.003
L = fft_lp(L + nz, 10000)
R = fft_lp(R + nz, 10000)
pk = max(np.abs(L).max(), np.abs(R).max())
L = np.tanh(L / pk * 1.6) * 0.9
R = np.tanh(R / pk * 1.6) * 0.9

out = sys.argv[1] if len(sys.argv) > 1 else "build/soundtrack.wav"
pcm = (np.stack([L, R], 1) * 32767).astype(np.int16)
with wave.open(out, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", out, TOTAL / SR, "s")
