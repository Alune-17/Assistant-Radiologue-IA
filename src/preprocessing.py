from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 Mo : suffisant pour une démo, évite les uploads abusifs

# Seuils de qualité image (empiriques, documentés)
MIN_RESOLUTION = 100          # px — en dessous : poor
MIN_CONTRAST_STD = 15.0       # écart-type pixel — faible contraste = limited
BRIGHTNESS_LOW = 30.0         # luminosité moyenne trop sombre = limited
BRIGHTNESS_HIGH = 225.0       # luminosité moyenne trop claire = limited
MAX_ASPECT_RATIO = 5.0        # ratio w/h ou h/w extrême = limited


def validate_image_upload(filename: str, content: bytes) -> str:
    """Valide un upload avant écriture disque.

    Le contrôle combine suffixe, taille et vérification Pillow. Cela évite de
    traiter un fichier non-image renommé en .png/.jpg et sécurise l'API.

    Returns:
        Suffixe normalisé de l'image, par exemple `.png`.

    Raises:
        ValueError: si le fichier ne respecte pas les contraintes du prototype.
    """
    suffix = Path(filename or "image.png").suffix.lower() or ".png"
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Format non supporté : {suffix}. Formats autorisés : {sorted(ALLOWED_SUFFIXES)}")
    if not content:
        raise ValueError("Fichier vide.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Fichier trop volumineux : maximum {MAX_UPLOAD_BYTES // (1024 * 1024)} Mo.")

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("Le fichier uploadé n'est pas une image valide.") from exc

    return suffix


def image_quality_report(path: str | Path) -> dict[str, Any]:
    """Calcule un rapport de qualité image explicable et journalisable.

    Le but n'est pas de valider cliniquement une radiographie, mais de fournir
    des preuves simples pour expliquer pourquoi le prototype accepte, limite ou
    refuse une interprétation automatique.
    """
    path = Path(path)
    try:
        with Image.open(path) as img:
            w, h = img.size
            ratio = max(w, h) / max(min(w, h), 1)
            gray = img.convert("L")
            arr = np.array(gray, dtype=np.float32)

        brightness = float(arr.mean())
        contrast = float(arr.std())
        reasons: list[str] = []
        quality = "good"

        if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
            quality = "poor"
            reasons.append(f"résolution trop faible ({w}x{h}px)")

        if ratio > MAX_ASPECT_RATIO:
            quality = "limited" if quality != "poor" else quality
            reasons.append(f"ratio d'aspect extrême ({ratio:.2f}:1)")

        if brightness < BRIGHTNESS_LOW:
            quality = "limited" if quality != "poor" else quality
            reasons.append(f"image trop sombre (luminosité {brightness:.1f})")
        elif brightness > BRIGHTNESS_HIGH:
            quality = "limited" if quality != "poor" else quality
            reasons.append(f"image trop claire (luminosité {brightness:.1f})")

        if contrast < MIN_CONTRAST_STD:
            quality = "limited" if quality != "poor" else quality
            reasons.append(f"contraste faible (écart-type {contrast:.1f})")

        if not reasons:
            reasons.append("contrôles qualité basiques satisfaits")

        return {
            "width": int(w),
            "height": int(h),
            "aspect_ratio": round(float(ratio), 3),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "quality": quality,
            "reasons": reasons,
        }
    except Exception as exc:
        return {
            "width": 0,
            "height": 0,
            "aspect_ratio": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "quality": "poor",
            "reasons": [f"image illisible ou non ouverte ({type(exc).__name__})"],
        }


def image_quality_metadata(path: str | Path) -> dict[str, Any]:
    """Alias historique : retourne le rapport de qualité image complet."""
    return image_quality_report(path)


def load_image(path: str | Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Load an image safely for the educational prototype.

    This function intentionally keeps preprocessing minimal. For real CXR work,
    DICOM metadata, windowing, projection and acquisition details should be handled
    explicitly and documented.
    """
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    img = Image.open(path).convert("RGB")
    return img.resize(size)


def basic_quality_flag(path: str | Path) -> str:
    """Retourne uniquement le flag qualité : ``good`` | ``limited`` | ``poor``."""
    return str(image_quality_report(path)["quality"])
