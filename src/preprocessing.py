from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

# Seuils de qualité image (empiriques, documentés)
MIN_RESOLUTION = 100          # px — en dessous : poor
MIN_CONTRAST_STD = 15.0       # écart-type pixel — faible contraste = limited
BRIGHTNESS_LOW = 30.0         # luminosité moyenne trop sombre = limited
BRIGHTNESS_HIGH = 225.0       # luminosité moyenne trop claire = limited
MAX_ASPECT_RATIO = 5.0        # ratio w/h ou h/w extrême = limited


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
    """Analyse la qualité d'image basée sur les propriétés visuelles réelles.

    Critères évalués :
    - Résolution minimale (< 100×100 px → poor)
    - Luminosité moyenne (trop sombre ou trop claire → limited)
    - Contraste (écart-type des pixels en niveaux de gris → limited si faible)
    - Ratio d'aspect extrême (> 5:1 → limited)

    Returns:
        "good" | "limited" | "poor"
    """
    path = Path(path)
    try:
        img = Image.open(path)
        w, h = img.size

        # Résolution insuffisante → poor
        if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
            return "poor"

        # Ratio d'aspect extrême → limited
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio > MAX_ASPECT_RATIO:
            return "limited"

        # Analyse en niveaux de gris pour luminosité et contraste
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.float32)

        brightness = float(arr.mean())
        contrast = float(arr.std())

        # Luminosité hors plage → limited
        if brightness < BRIGHTNESS_LOW or brightness > BRIGHTNESS_HIGH:
            return "limited"

        # Faible contraste → limited
        if contrast < MIN_CONTRAST_STD:
            return "limited"

        return "good"

    except Exception:
        # En cas d'erreur d'ouverture ou d'analyse → poor (plus sûr)
        return "poor"
