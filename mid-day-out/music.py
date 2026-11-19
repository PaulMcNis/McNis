"""Soundtrack fuer "mid day out": Retro-Elektro x Spaghetti-Western x Sci-Fi.

120 BPM, 30 Takte = exakt 60 s, nahtlos loopbar (Hall-/Echo-Fahnen werden an den Anfang gefaltet).
Alles wird synthetisiert, keine Samples.
"""
import wave
import sys
import numpy as np

SR = 44100
BPM = 120
BEAT = SR * 60 // BPM          # 22050
BAR = BEAT * 4
BARS = 30
TOTAL = BAR * BARS             # 60 s
TAIL = SR * 4
rng = np.random.default_rng(7)

L = np.zeros(TOTAL + TAIL)
R = np.zeros(TOTAL + TAIL)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(sig, start, pan=0.0, gain=1.0):
    start = int(start)
    n = min(len(sig), len(L) - start)
    if n <= 0:
        return
    lg = gain * np.sqrt(0.5 * (1 - pan))
    rg = gain * np.sqrt(0.5 * (1 + pan))
    L[start:start + n] += sig[:n] * lg
    R[start:start + n] += sig[:n] * rg


def env_adsr(n, a=0.005, d=0.1, s=0.6, r=0.05):
    t = np.arange(n) / SR
    dur = n / SR
    e = np.where(t < a, t / max(a, 1e-6),
                 np.where(t < a + d, 1 - (1 - s) * (t - a) / max(d, 1e-6), s))
    rel_start = max(dur - r, 0)
    e = np.where(t > rel_start, e * np.clip(1 - (t - rel_start) / max(r, 1e-6), 0, 1), e)
    return e


def onepole_lp(x, cutoff):
    """Einpoliger Tiefpass, cutoff als Array oder Skalar (Hz)."""
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    a = 1 - np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def saw(freq, n, phase=0.0):
    t = np.arange(n) / SR
    return 2 * ((freq * t + phase) % 1.0) - 1


def square(freq, n, duty=0.5):
    t = np.arange(n) / SR
    return np.where((freq * t) % 1.0 < duty, 1.0, -1.0)


def tri(freq, n):
    t = np.arange(n) / SR
    return 2 * np.abs(2 * ((freq * t) % 1.0) - 1) - 1


# ---------------------------------------------------------------- Drums
def kick():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 9) * 1.0


def clap():
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    hp = np.diff(nz, prepend=0)
    bp = onepole_lp(hp, 3500)
    env = np.exp(-t * 18)
    # drei kurze Anschlaege wie bei alten Drumcomputern
    for off in (0.0, 0.011, 0.022):
        env += 0.6 * np.exp(-np.clip(t - off, 0, None) * 160) * (t >= off)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30) * 0.4
    return (bp * env * 0.9 + tone) * 0.8


def hat(open_=False):
    n = int((0.18 if open_ else 0.045) * SR)
    t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    hp = nz - onepole_lp(nz, 6000)
    return hp * np.exp(-t * (14 if open_ else 70)) * 0.35


def tom(m):
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    f = mtof(m) * (1 + 0.6 * np.exp(-t * 20))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 10) * 0.6


K, C, HC, HO = kick(), clap(), hat(), hat(True)

# ---------------------------------------------------------------- Harmonie
# Andalusische Kadenz = Western-Feeling: Am  G  F  E
CHORDS = [(45, [57, 60, 64]), (43, [55, 59, 62]), (41, [53, 57, 60]), (40, [52, 56, 59])]


def chord_of(bar):
    return CHORDS[bar % 4]


def section(bar):
    if bar < 2:
        return "intro"
    if bar < 10:
        return "A"
    if bar < 18:
        return "B"
    if bar < 26:
        return "C"
    return "outro"


# ---------------------------------------------------------------- Instrumente
def bass_note(m, n, accent=1.0):
    f = mtof(m)
    x = saw(f, n) * 0.6 + square(f * 0.5, n) * 0.4
    t = np.arange(n) / SR
    cut = 180 + 1800 * accent * np.exp(-t * 18)
    y = onepole_lp(onepole_lp(x, cut), cut)
    return y * env_adsr(n, 0.002, 0.08, 0.5, 0.02) * 0.55


def chip_note(m, n):
    return square(mtof(m), n, 0.25) * env_adsr(n, 0.001, 0.04, 0.4, 0.01) * 0.15


def whistle(m, n, glide_from=None):
    t = np.arange(n) / SR
    f0 = mtof(m)
    f = np.full(n, f0)
    if glide_from is not None:
        g = np.exp(-t * 25)
        f = f0 + (mtof(glide_from) - f0) * g
    vib = 1 + 0.012 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.25, 0, 1)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    breath = onepole_lp(rng.standard_normal(n), 2500) * 0.04
    x = np.sin(ph) + 0.08 * np.sin(2 * ph) + breath
    return x * env_adsr(n, 0.04, 0.1, 0.85, 0.08) * 0.32


def twang(m, n):
    """Karplus-Strong-Gitarre mit Tremolo (Western-Twang)."""
    f = mtof(m)
    period = int(SR / f)
    buf = rng.uniform(-1, 1, period)
    buf = onepole_lp(buf, 5000)
    out = np.empty(n)
    idx = 0
    prev = 0.0
    for i in range(n):
        cur = buf[idx]
        avg = 0.5 * (cur + prev) * 0.996
        prev = cur
        buf[idx] = avg
        out[i] = cur
        idx = (idx + 1) % period
    t = np.arange(n) / SR
    trem = 1 - 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * 6 * t))
    out = np.tanh(out * 2.2) * trem
    return out * env_adsr(n, 0.001, 0.05, 1.0, 0.05) * 0.30


def pad(notes, n):
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in notes:
        for det in (-0.12, 0.0, 0.12):
            x += saw(mtof(m + 12) * 2 ** (det / 12), n, rng.uniform())
    x /= len(notes) * 3
    # "Vocoder"-artige Formant-Bewegung
    cut = 900 + 600 * np.sin(2 * np.pi * 0.5 * t) ** 2
    return onepole_lp(onepole_lp(x, cut), cut) * env_adsr(n, 0.4, 0.2, 0.8, 0.4) * 0.5


def theremin(m_from, m_to, n):
    t = np.arange(n) / SR
    k = t / t[-1]
    m = m_from + (m_to - m_from) * (0.5 - 0.5 * np.cos(np.pi * k))
    f = mtof(m) * (1 + 0.02 * np.sin(2 * np.pi * 6.2 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (np.sin(ph) + 0.15 * np.sin(2 * ph)) * env_adsr(n, 0.3, 0.1, 0.9, 0.5) * 0.22


def zap(n=int(0.25 * SR)):
    t = np.arange(n) / SR
    f = 2600 * np.exp(-t * 18) + 120
    return square(1, n) * 0 + np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 12) * 0.18


# ---------------------------------------------------------------- Melodien (Achtel-Raster: start, laenge, midi)
WHISTLE = [
    [(0, 6, 76), (6, 1, 74), (7, 1, 72)],
    [(0, 4, 74), (4, 2, 71), (6, 2, 67)],
    [(0, 3, 72), (3, 1, 69), (4, 4, 65)],
    [(0, 6, 68), (6, 2, 71)],
    [(0, 2, 69), (2, 2, 72), (4, 4, 76)],
    [(0, 3, 79), (3, 1, 77), (4, 4, 74)],
    [(0, 2, 77), (2, 2, 76), (4, 4, 72)],
    [(0, 8, 76)],
]
TWANG = [
    [(0, 1, 57), (1, 1, 57), (2, 1, 60), (3, 2, 64), (5, 1, 62), (6, 2, 60)],
    [(0, 1, 55), (1, 1, 55), (2, 1, 59), (3, 2, 62), (5, 1, 60), (6, 2, 59)],
    [(0, 1, 53), (1, 1, 53), (2, 1, 57), (3, 2, 60), (5, 1, 59), (6, 2, 57)],
    [(0, 2, 56), (2, 2, 59), (4, 2, 62), (6, 2, 64)],
]

EIGHTH = BEAT // 2
SIXT = BEAT // 4

for bar in range(BARS):
    s = section(bar)
    b0 = bar * BAR
    root, tones = chord_of(bar)

    # Drums
    for beat in range(4):
        p = b0 + beat * BEAT
        if s == "intro":
            if beat in (0, 2):
                add(K, p, gain=0.8)
        elif s == "outro" and bar == 29:
            if beat in (0, 2):
                add(K, p, gain=0.8)
        else:
            add(K, p, gain=0.75)
            if beat in (1, 3):
                add(C, p, pan=0.1, gain=0.8)
    if s not in ("intro",):
        for st in range(16):
            p = b0 + st * SIXT
            if st % 4 == 2 and s in ("B", "C"):
                add(HO, p, pan=0.3, gain=0.6)
            else:
                add(HC, p, pan=0.3, gain=0.9 if st % 2 == 0 else 0.5)
    # Elektro-Tom-Fill am Ende jedes 4. Taktes
    if bar % 4 == 3 and s in ("A", "B", "C"):
        for i, m in enumerate((52, 48, 45, 40)):
            add(tom(m), b0 + 3 * BEAT + i * SIXT, pan=-0.4 + 0.25 * i, gain=0.7)

    # Bass-Sequenz (Kraftwerk-artig, 8tel)
    if s != "intro":
        pattern = [0, 0, 12, 0, 7, 0, 12, 7]
        for i, iv in enumerate(pattern):
            add(bass_note(root + iv, EIGHTH - 400, 1.0 if i % 2 == 0 else 0.6), b0 + i * EIGHTH, gain=1.0)
    else:
        add(bass_note(root, BAR - 2000, 0.3), b0, gain=0.7)

    # Konsolen-Arpeggio (16tel)
    if s in ("A", "C", "outro"):
        arp = tones + [tones[0] + 12, tones[1] + 12, tones[2] + 12, tones[0] + 24, tones[2] + 12]
        for st in range(16):
            add(chip_note(arp[st % len(arp)] + 12, SIXT - 200), b0 + st * SIXT, pan=-0.5 if st % 2 else 0.5)

    # Western-Pfeifen
    if s == "B" or (s == "C" and bar >= 22):
        mel = WHISTLE[(bar - 10) % 8]
        prev = None
        for (st, ln, m) in mel:
            add(whistle(m, ln * EIGHTH, prev), b0 + st * EIGHTH, pan=-0.2, gain=0.9 if s == "B" else 0.6)
            prev = m

    # Twang-Gitarre
    if s == "C":
        for (st, ln, m) in TWANG[bar % 4]:
            add(twang(m, int(ln * EIGHTH + SR * 0.3)), b0 + st * EIGHTH, pan=0.35)

    # Pads
    if s in ("C", "outro"):
        add(pad(tones, BAR), b0, gain=0.8 if s == "C" else 0.5)

    # Sci-Fi
    if s == "intro" and bar == 0:
        add(theremin(57, 76, BAR * 2 - 1000), b0, gain=1.0)
    if s == "outro" and bar == 28:
        add(theremin(64, 57, BAR * 2 - 1000), b0, gain=0.9)
    if s in ("A", "B", "C") and bar % 2 == 0:
        add(zap(), b0 + 3 * BEAT + EIGHTH, pan=0.6 if bar % 4 else -0.6)

# Einfaches Stereo-Echo (punktierte Achtel) fuer Raumgefuehl
d = int(BEAT * 0.75)
for lvl, k in ((0.28, 1), (0.14, 2), (0.07, 3)):
    sh = d * k
    L[sh:] += R[:-sh] * lvl
    R[sh:] += L[:-sh] * lvl * 0.9

# Loop-Faltung: Fahnen nach 60 s wieder an den Anfang
L[:TAIL] += L[TOTAL:]
R[:TAIL] += R[TOTAL:]
L, R = L[:TOTAL], R[:TOTAL]

# Tape-Wow/Flutter (periodisch in 60 s -> loopbar)
t = np.arange(TOTAL)
wob = 0.0016 * SR * np.sin(2 * np.pi * 15 * t / TOTAL) + 0.00025 * SR * np.sin(2 * np.pi * 420 * t / TOTAL)
src = (t + wob) % TOTAL
L = np.interp(src, t, L, period=TOTAL)
R = np.interp(src, t, R, period=TOTAL)

# Projektor-Rattern (18 Bilder/s) + Rauschen, ganz leise
nz = rng.standard_normal(TOTAL)
clatter = onepole_lp(nz, 1800) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 18 * t / SR))) * 0.012
hiss = (nz - onepole_lp(nz, 4000)) * 0.004
L += clatter + hiss
R += clatter + hiss

# Retro-Klang: Hoehen leicht abschneiden, sanft saettigen
L = onepole_lp(L, 9000)
R = onepole_lp(R, 9000)
peak = max(np.abs(L).max(), np.abs(R).max())
L, R = np.tanh(L / peak * 1.3) * 0.89, np.tanh(R / peak * 1.3) * 0.89

out = sys.argv[1] if len(sys.argv) > 1 else "build/soundtrack.wav"
pcm = (np.stack([L, R], axis=1) * 32767).astype(np.int16)
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", out, TOTAL / SR, "s")
