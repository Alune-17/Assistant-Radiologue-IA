#!/usr/bin/env python3
"""
Script de téléchargement CheXpert depuis Kaggle (source officielle).

Source unique : ashery/chexpert
  - Dataset officiel CheXpert v1.0 de Stanford AIMI (~11 Go)
  - Inclut les labels (train.csv) ET les images (train/ + valid/)
  - Seul dataset Kaggle avec la licence Stanford correctement mentionnée
  - URL : https://www.kaggle.com/datasets/ashery/chexpert

PRÉREQUIS :
  1. kaggle.json configuré dans ~/.kaggle/kaggle.json
  2. Accepter les conditions du dataset sur Kaggle :
     - https://www.kaggle.com/datasets/ashery/chexpert
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

# ── Configuration Kaggle ───────────────────────────────────────────────────────────────────────────────
DATASET = "ashery/chexpert"   # Source officielle Stanford AIMI (train + labels)

DEFAULT_OUT_DIR     = ROOT / "data" / "chexpert_eval"
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
    # Nettoyer le dossier images existant pour repartir d'un état propre
    # (évite de mélanger les images d'un ancien run bugué avec les nouvelles)
    if images_out.exists():
        shutil.rmtree(images_out)
    images_out.mkdir(parents=True, exist_ok=True)

    # Indexer les images disponibles (depuis le sample duong1589).
    # On construit deux index :
    #   1. available_3 : patient/study/filename  → Path  (match exact)
    #   2. available_p : patient                → list[Path]  (fallback si étude absente)
    #
    # ATTENTION : ne pas utiliser img.name seul comme clé, car tous les fichiers
    # s'appellent "view1_frontal.jpg" → collision, une seule image conservée !
    available_3: dict[str, Path]       = {}
    available_p: dict[str, list[Path]] = {}
    for img in images_root.rglob("*.jpg"):
        parts = img.parts
        if len(parts) >= 3:
            key3 = "/".join(parts[-3:])   # patient/study/file
            available_3[key3] = img
        if len(parts) >= 4:
            patient_id = parts[-4]        # patientXXXXX
            available_p.setdefault(patient_id, []).append(img)

    print(f"    {len(available_3)} images indexées (patient/study/file) dans le sample Kaggle")

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
    # Pour dédupliquer : on garde trace des chemins physiques déjà ajoutés
    seen_img_paths: set[Path] = set()

    for row in all_rows:
        path_val = row.get("Path", "")
        if "lateral" in path_val.lower():
            continue
        label = map_chexpert_label(row)

        # --- Matching robuste (2 niveaux) ---
        # Le CSV amritpal333 : "CheXpert-v1.0-small/train/patientXXXXX/studyY/file.jpg"
        # Le cache duong1589 : "CheXpert-v1.0/train/patientXXXXX/studyY/file.jpg"
        # → 1) normaliser le préfixe et tenter le match exact patient/study/file
        # → 2) si l'étude exacte est absente du cache, accepter n'importe quelle
        #      étude du même patient (même type de vue, frontal uniquement).
        norm_path = path_val.replace("CheXpert-v1.0-small/", "CheXpert-v1.0/")
        parts_csv = Path(norm_path).parts
        if len(parts_csv) < 4:
            continue

        patient_id = parts_csv[-4]        # patientXXXXX
        filename   = parts_csv[-1]        # view1_frontal.jpg
        key_3      = "/".join(parts_csv[-3:])

        img_path = available_3.get(key_3)  # match exact étude

        if img_path is None:
            # Fallback : chercher n'importe quelle image du même patient + même nom de fichier
            candidates = [
                p for p in available_p.get(patient_id, [])
                if p.name == filename
            ]
            if candidates:
                img_path = candidates[0]

        if img_path is not None and img_path not in seen_img_paths:
            seen_img_paths.add(img_path)
            row["_img_path"] = img_path
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
        description="Télécharge et prépare un sous-ensemble CheXpert depuis Kaggle (ashery/chexpert)."
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
        "--raw-dir", type=Path, default=None,
        help=(
            "Chemin vers le dataset CheXpert déjà téléchargé localement  "
            "(dossier contenant train/ et train.csv). "
            "Si renseigné, le téléchargement Kaggle est ignoré. "
            "Exemple : data/chexpert_raw"
        ),
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  CheXpert Subset Builder — Assistant Radiologue Virtuel EFREI")
    print("=" * 65)
    print(f"  Source  : {DATASET}")
    print(f"  Objectif : {args.n_per_class} images x 3 classes = {args.n_per_class * 3} cas")
    print(f"  Sortie   : {out_dir}")
    print()

    # Étape 1 : Déterminer l'emplacement du dataset brut
    if args.raw_dir is not None:
        # Utiliser un dataset déjà téléchargé localement (ex: data/chexpert_raw)
        raw_dir = args.raw_dir.resolve()
        print(f"[1/2] Dataset local utilisé : {raw_dir}")
    else:
        # Télécharger ashery/chexpert depuis Kaggle
        cache_dir = out_dir / "_cache"
        raw_dir = cache_dir / "ashery_chexpert"
        if any(raw_dir.rglob("*.jpg")):
            n_cached = len(list(raw_dir.rglob("*.jpg")))
            print(f"[1/2] Dataset deja téléchargé — {n_cached} images en cache.")
        else:
            print(f"[1/2] Téléchargement depuis Kaggle ({DATASET}, ~11 Go)...")
            print(f"      ⚠️  Assure-toi d'avoir accepté les conditions sur :")
            print(f"          https://www.kaggle.com/datasets/{DATASET}")
            kaggle_download(DATASET, raw_dir)

    # Localiser train.csv et le dossier train/
    labels_csv_path = None
    for candidate in [raw_dir / "train.csv", raw_dir / "CheXpert-v1.0" / "train.csv",
                      raw_dir / "CheXpert-v1.0-small" / "train.csv"]:
        if candidate.exists():
            labels_csv_path = candidate
            break
    if labels_csv_path is None:
        csv_found = list(raw_dir.rglob("train.csv"))
        if csv_found:
            labels_csv_path = csv_found[0]
    if labels_csv_path is None:
        print("[ERR] train.csv introuvable dans le dataset. Vérifie le contenu de raw_dir.")
        sys.exit(1)
    print(f"      Labels CSV : {labels_csv_path.relative_to(raw_dir) if raw_dir in labels_csv_path.parents else labels_csv_path.name}")

    # Le dossier images = dossier parent de train.csv / train/
    images_root = labels_csv_path.parent

    # Étape 2 : Construire le subset équilibré
    print()
    print("[2/2] Construction du subset équilibré...")
    out_csv = build_cases_csv(
        labels_csv  = labels_csv_path,
        images_root = images_root,
        out_dir     = out_dir,
        n_per_class = args.n_per_class,
        seed        = args.seed,
    )

    # Résumé
    with out_csv.open(encoding="utf-8") as f:
        n_total = sum(1 for _ in f) - 1  # header exclus

    print()
    print("=" * 65)
    print(f"  [DONE] {n_total} cas sélectionnés dans :")
    print(f"         {out_csv}")
    print()
    print("  Si le dataset est déjà local (data/chexpert_raw) :")
    print()
    print("    .venv\\Scripts\\python.exe data/download_chexpert.py \\")
    print(f"      --raw-dir data/chexpert_raw")
    print()
    print("  Lancer l'évaluation sur les vraies radios :")
    print()
    print("    .venv\\Scripts\\python.exe eval/run_evaluation.py \\")
    print("      --mode gemini-baseline \\")
    print(f"      --cases-csv {out_csv}")
    print()
    print("  RAPPEL LICENCE :")
    print("  CheXpert — Stanford AIMI — recherche et éducation uniquement.")
    print("  Citation : Irvin et al. (2019), AAAI. DOI: 10.1609/aaai.v33i01.3301590")
    print("  Dataset  : https://www.kaggle.com/datasets/ashery/chexpert")
    print("=" * 65)


if __name__ == "__main__":
    main()
