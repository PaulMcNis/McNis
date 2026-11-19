"""Analysiert alle Bilder: Personen-Maske (Segmentierung) + Gesichts-Landmarken. Ergebnis: build/analysis/*.npz"""
import os, sys, glob
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

MODELS = sys.argv[1]
os.makedirs("build/analysis", exist_ok=True)
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=mpt.BaseOptions(model_asset_path=f"{MODELS}/selfie_multiclass_256x256.tflite"),
    output_category_mask=False, output_confidence_masks=True))
face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpt.BaseOptions(model_asset_path=f"{MODELS}/face_landmarker.task"),
    num_faces=4, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))

for path in sorted(glob.glob("bilder/a_*")):
    im = Image.open(path).convert("RGB")
    s = 1400 / max(im.size)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    arr = np.ascontiguousarray(np.asarray(im))
    mimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    r = seg.segment(mimg)
    person = 1 - r.confidence_masks[0].numpy_view().astype(np.float32)   # 0 = Hintergrund
    fr = face.detect(mimg)
    lms = [np.array([[p.x, p.y, p.z] for p in f], np.float32) for f in fr.face_landmarks]
    name = os.path.splitext(os.path.basename(path))[0]
    np.savez_compressed(f"build/analysis/{name}.npz", person=person.astype(np.float16),
                        faces=np.array(lms, np.float32) if lms else np.zeros((0, 478, 3), np.float32))
    print(f"{name:24s} faces={len(lms)} person={person.mean():.2f}", flush=True)
