import os
import cv2
import numpy as np
import shutil

PATIENT_DIR = "data/raw"
UNCERTAIN_DIR = "data/raw/uncertain"
CLEAN_DIR = "data/raw/normal_opacity_candidates"

os.makedirs(UNCERTAIN_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

def is_blurry(img, threshold=120):
    lap = cv2.Laplacian(img, cv2.CV_64F).var()
    return lap < threshold

def is_bad_exposure(img, low=40, high=210):
    mean = np.mean(img)
    return mean < low or mean > high

def is_too_small(img, min_size=400):
    h, w = img.shape[:2]
    return h < min_size or w < min_size

def classify_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "uncertain"

    if is_blurry(img) or is_bad_exposure(img) or is_too_small(img):
        return "uncertain"
    return "clean"

for patient in os.listdir(PATIENT_DIR):
    patient_path = os.path.join(PATIENT_DIR, patient)
    if not os.path.isdir(patient_path):
        continue

    study_path = os.path.join(patient_path, "study1")
    if not os.path.exists(study_path):
        continue

    for img_name in os.listdir(study_path):
        img_path = os.path.join(study_path, img_name)

        cls = classify_image(img_path)

        if cls == "uncertain":
            shutil.copy(img_path, os.path.join(UNCERTAIN_DIR, f"{patient}_{img_name}"))
        else:
            shutil.copy(img_path, os.path.join(CLEAN_DIR, f"{patient}_{img_name}"))

print("Étape 1 terminée : uncertain détecté automatiquement.")
