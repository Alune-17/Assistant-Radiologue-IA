#!/usr/bin/env python3
"""
Script de téléchargement optimisé CheXpert depuis Kaggle.

Stratégie :
  - CSV des labels : amritpal333/chexpert-train-csv-modified (~20 Mo, 220K labels)
  - Images         : duong1589/chexpert (~52 Mo, ~1000 vraies radios disponibles)

Avantage : on ne télécharge PAS les 11 Go du dataset complet.

PRÉREQUIS :
  1. kaggle.json configuré dans ~/.kaggle/kaggle.json
  2. Accepter les conditions des datasets sur Kaggle :
     - https://www.kaggle.com/datasets/amritpal333/chexpert-train-csv-modified
     - https://www.kaggle.com/datasets/duong1589/chexpert
  3. Package kaggle installé : .venv\\Scripts\\pip install kaggle

UTILISATION :
  .venv\\Scripts\\python.exe data/download_chexpert.py
  .venv\\Scripts\\python.exe data/download_chexpert.py --n-per-class 15
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Configuration Kaggle ───────────────────────────────────────────────────────
DATASET_CSV    = "amritpal333/chexpert-train-csv-modified"   # labels CSV (20 Mo)
DATASET_IMAGES = "duong1589/chexpert"                        # images sample (52 Mo)

DEFAULT_OUT_DIR     = ROOT / "data" / "chexpert_subset"
DEFAULT_N_PER_CLASS = 10   # 10 × 3 classes = 30 cas

# Colonnes CheXpert indiquant une opacité pulmonaire
OPACITY_COLUMNS = ["Consolidation", "Pleural Effusion", "Edema", "Pneumonia"]


# ── Mapping de labels ──────────────────────────────────────────────────────────

def map_chexpert_label(row: dict) -> str:
    """Convertit une ligne du CSV CheXpert en l'une de nos 3 classes.

    Règles :
    1. No Finding = 1.0                                → normal
    2. Colonne opacité = 1.0 (au moins une)            → suspected_opacity
    3. Colonne opacité = -1.0 (incertain, aucune à 1)  → uncertain
    4. Sinon                                           → uncertain (garde-fou)
    """
    try:
        no_finding = float(row.get("No Finding", 0) or 0)
    except (ValueError, TypeError):
        no_finding = 0.0

    if no_finding == 1.0:
        return "normal"

    opacity_vals = []
    for col in OPACITY_COLUMNS:
        try:
            val = float(row.get(col, 0) or 0)
            opacity_vals.append(val)
        except (ValueError, TypeError):
            opacity_vals.append(0.0)

    if any(v == 1.0 for v in opacity_vals):
        return "suspected_opacity"
    if any(v == -1.0 for v in opacity_vals):
        return "uncertain"

    return "uncertain"


# ── Téléchargement Kaggle ──────────────────────────────────────────────────────

def kaggle_download(dataset: str, dest: Path) -> None:
    """Télécharge et décompresse un dataset Kaggle."""
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", dataset,
            "-p", str(dest),
            "--unzip",
        ],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERR] Echec du telechargement de '{dataset}'")
        print(f"      Verifiez que vous avez accepte les conditions sur :")
        print(f"      https://www.kaggle.com/datasets/{dataset}")
        sys.exit(1)


# ── Construction du CSV de cas ─────────────────────────────────────────────────

def build_cases_csv(
    labels_csv: Path,
    images_root: Path,
    out_dir: Path,
    n_per_class: int,
    seed: int = 42,
) -> Path:
    """Sélectionne n_per_class cas par classe et copie les images correspondantes.

    Args:
        labels_csv:  CSV CheXpert avec colonnes de labels (train.csv ou équivalent).
        images_root: Répertoire racine contenant les images .jpg.
        out_dir:     Répertoire de sortie.
        n_per_class: Nombre de cas à sélectionner par classe (défaut: 10).
        seed:        Graine aléatoire pour la reproductibilité.

    Returns:
        Chemin vers le CSV de cas généré (compatible run_evaluation.py).
    """
    random.seed(seed)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Indexer les images disponibles (depuis le sample duong1589)
    available_images: dict[str, Path] = {}
    for img in images_root.rglob("*.jpg"):
        # Clé : patient + study + filename (pour matcher avec le CSV)
        # Exemple de path CheXpert : patient00028/study2/view1_frontal.jpg
        parts = img.parts
        if len(parts) >= 3:
            key = "/".join(parts[-3:])  # patient/study/file
        else:
            key = img.name
        available_images[img.name] = img
        if len(parts) >= 3:
            available_images["/".join(parts[-3:])] = img

    print(f"    {len(available_images)} images indexees dans le sample Kaggle")

    # Lecture du CSV de labels
    print(f"[>>] Lecture des labels : {labels_csv.name}...")
    with labels_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    print(f"    {len(all_rows)} lignes de labels lues")

    # Construire les buckets par classe (vues frontales uniquement)
    buckets: dict[str, list[dict]] = {
        "normal": [],
        "suspected_opacity": [],
        "uncertain": [],
    }
    for row in all_rows:
        path_val = row.get("Path", "")
        if "lateral" in path_val.lower():
            continue
        label = map_chexpert_label(row)
        # Vérifier si une image correspondante est disponible dans notre sample
        filename = Path(path_val).name
        if filename in available_images:
            row["_img_path"] = available_images[filename]
            buckets[label].append(row)

    print()
    for cls, items in buckets.items():
        status = "[OK]" if len(items) >= n_per_class else "[WARN]"
        print(f"    {status} {cls:20s}: {len(items):4d} cas avec image disponible")

    # Sélection équilibrée et reproductible
    output_rows = []
    case_counter = 1

    for label in ["normal", "suspected_opacity", "uncertain"]:
        pool = buckets[label]
        if not pool:
            print(f"\n[WARN] Aucune image disponible pour la classe '{label}' !")
            print(f"       Le sample duong1589/chexpert ne couvre pas assez cette classe.")
            print(f"       Vous pouvez télécharger le dataset complet ashery/chexpert (~11 Go).")
            continue

        selected = random.sample(pool, min(n_per_class, len(pool)))

        for row in selected:
            src = row["_img_path"]
            case_id = f"CXR_CHEX_{case_counter:03d}"
            dest_name = f"{case_id}_{label}.jpg"
            dest_path = images_out / dest_name

            shutil.copy2(src, dest_path)

            output_rows.append({
                "case_id":    case_id,
                "image_path": f"data/chexpert_subset/images/{dest_name}",
                "source":     "chexpert_v1",
                "label":      label,
                "split":      "final",
                "quality":    "good",
                "notes": (
                    f"CheXpert frontal view. "
                    f"Original: {row.get('Path', 'unknown')}. "
                    f"License: Stanford AIMI — research and education only."
                ),
            })
            case_counter += 1

    # Écriture du CSV de sortie
    out_csv = out_dir / "chexpert_cases.csv"
    fieldnames = ["case_id", "image_path", "source", "label", "split", "quality", "notes"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(output_rows)

    return out_csv


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Télécharge et prépare un sous-ensemble CheXpert depuis Kaggle."
    )
    parser.add_argument(
        "--n-per-class", type=int, default=DEFAULT_N_PER_CLASS,
        help=f"Nombre d'images par classe (défaut: {DEFAULT_N_PER_CLASS} → {DEFAULT_N_PER_CLASS*3} cas).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Répertoire de sortie (défaut: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Graine aléatoire pour la reproductibilité (défaut: 42).",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Réutilise les fichiers déjà téléchargés (évite de re-télécharger).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_cache"

    print("=" * 65)
    print("  CheXpert Subset Builder — Assistant Radiologue Virtuel EFREI")
    print("=" * 65)
    print(f"  Objectif : {args.n_per_class} images x 3 classes = {args.n_per_class * 3} cas")
    print(f"  Sortie   : {out_dir}")
    print()

    # Étape 1 : CSV des labels (~20 Mo)
    csv_cache = cache_dir / "csv"
    labels_csv_path = None

    if not args.skip_download or not any(csv_cache.rglob("*.csv")):
        print("[1/2] Telechargement du CSV de labels (amritpal333, ~20 Mo)...")
        kaggle_download(DATASET_CSV, csv_cache)
    else:
        print("[1/2] CSV deja telecharge — utilisation du cache.")

    # Trouver le CSV téléchargé
    csv_candidates = list(csv_cache.rglob("*.csv"))
    if not csv_candidates:
        print("[ERR] Aucun CSV trouve apres telechargement.")
        sys.exit(1)
    labels_csv_path = csv_candidates[0]
    print(f"      CSV : {labels_csv_path.name} ({labels_csv_path.stat().st_size // 1024} Ko)")

    # Étape 2 : Images sample (~52 Mo)
    img_cache = cache_dir / "images"

    if not args.skip_download or not any(img_cache.rglob("*.jpg")):
        print("[2/2] Telechargement des images sample (duong1589, ~52 Mo)...")
        kaggle_download(DATASET_IMAGES, img_cache)
    else:
        n_cached = len(list(img_cache.rglob("*.jpg")))
        print(f"[2/2] Images deja telechargees — {n_cached} images en cache.")

    # Étape 3 : Construire le subset équilibré
    print()
    print("[3/3] Construction du subset equilibre...")
    out_csv = build_cases_csv(
        labels_csv  = labels_csv_path,
        images_root = img_cache,
        out_dir     = out_dir,
        n_per_class = args.n_per_class,
        seed        = args.seed,
    )

    # Résumé
    with out_csv.open(encoding="utf-8") as f:
        n_total = sum(1 for _ in f) - 1  # header exclus

    print()
    print("=" * 65)
    print(f"  [DONE] {n_total} cas selectionnes dans :")
    print(f"         {out_csv}")
    print()
    print("  Lancer l'evaluation sur les vraies radios :")
    print()
    print("    .venv\\Scripts\\python.exe eval/run_evaluation.py \\")
    print("      --mode gemini-baseline \\")
    print(f"      --cases-csv {out_csv}")
    print()
    print("  Ou comparer baseline vs improved :")
    print()
    print("    .venv\\Scripts\\python.exe eval/run_evaluation.py \\")
    print("      --mode all-gemini \\")
    print(f"      --cases-csv {out_csv}")
    print()
    print("  RAPPEL LICENCE :")
    print("  CheXpert — Stanford AIMI — recherche et education uniquement.")
    print("  Citation : Irvin et al. (2019), AAAI. DOI: 10.1609/aaai.v33i01.3301590")
    print("=" * 65)


if __name__ == "__main__":
    main()
