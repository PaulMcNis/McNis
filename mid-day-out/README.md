# mid day out – 60 s Super-8-Loop

- `music.py` – synthetisiert den Soundtrack (120 BPM, Elektro/Western/Sci-Fi, 60 s nahtlos loopbar)
- `video.py` – rendert 16 Bilder aus `bilder/` im Super-8/60er-TV-Look (4:3, 18 fps)

```bash
mkdir -p build
python3 music.py build/soundtrack.wav
python3 video.py | ffmpeg -f rawvideo -pix_fmt rgb24 -s 960x720 -r 18 -i - -i build/soundtrack.wav \
  -map 0:v -map 1:a -c:v libx264 -preset slow -crf 25 -tune grain -pix_fmt yuv420p -aspect 4:3 \
  -c:a aac -b:a 192k -shortest -movflags +faststart mid_day_out.mp4
```

Fotos und fertiges Video werden bewusst nicht ins Repository eingecheckt (`.gitignore`).
