# Données — Assistant Radiologue Virtuel EFREI

Ce dossier contient **3 zones de données clairement séparées**, chacune avec un rôle distinct.

---

## Structure

```
data/
├── synthetic/                    ← images de test (pas des vraies radios)
│   ├── images/                   ← 30 PNG générés programmatiquement
│   └── cases.csv                 ← index des 30 cas de test
│
├── chexpert_raw/                 ← dataset CheXpert brut complet [❌ gitignored]
│   ├── train/                    ← 50 594 patients, ~8 Go
│   ├── valid/                    ← 200 patients, ~12 Mo
│   └── valid.csv                 ← labels du valid set officiel Stanford
│
├── chexpert_eval/                ← sous-ensemble prêt pour l'évaluation [❌ gitignored]
│   ├── images/                   ← radios sélectionnées (21 actuellement)
│   ├── cases.csv                 ← index pour run_evaluation.py
│   └── _cache/                   ← cache temporaire (ne pas modifier)
│
├── download_chexpert.py          ← script de préparation du subset
└── README.md                     ← ce fichier
```

---

## Zone 1 — `synthetic/` — Images de test

> **Rôle :** Valider l'architecture, les logs, les métriques et les smoke tests CI/CD.
> **⚠️ Ne pas utiliser pour évaluer les performances médicales du modèle.**

| Propriété | Valeur |
|-----------|--------|
| Format | PNG synthétique (~27 Ko/image) |
| Taille totale | 30 images (~800 Ko) |
| Répartition | 10 normal · 10 suspected_opacity · 10 uncertain |
| Génération | Programmatique (script Python — gribouillis colorés) |
| Licence | Libre — usage interne projet EFREI |
| Git | ✅ Commité (léger, utile pour le CI) |

Ces images **ne ressemblent pas à de vraies radiographies**. Elles servent uniquement à s'assurer que le pipeline Python tourne de bout en bout.

### Utilisation

```python
# Dans le code Python, accéder aux images synthétiques :
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
image = ROOT / "data" / "synthetic" / "images" / "CXR_SYN_002_suspected_opacity.png"
```

```bash
# Lancer l'évaluation sur les images synthétiques (défaut) :
.venv\Scripts\python.exe eval/run_evaluation.py
# équivaut à :
.venv\Scripts\python.exe eval/run_evaluation.py --cases-csv data/synthetic/cases.csv
```

---

## Zone 2 — `chexpert_raw/` — Dataset CheXpert complet

> **Rôle :** Données sources brutes de référence. Utilisées par `download_chexpert.py`
> pour construire le subset d'évaluation. Très volumineuses — ne pas commiter.

| Propriété | Valeur |
|-----------|--------|
| Nom officiel | **CheXpert v1.0** (version Small) |
| Auteurs | Irvin et al. (2019) — Stanford ML Group / Stanford AIMI |
| Source Kaggle | https://www.kaggle.com/datasets/ashery/chexpert |
| Source officielle Stanford | https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2 |
| **Licence** | **Recherche et éducation uniquement — usage non commercial** |
| Taille | ~8,1 Go (train) + ~12 Mo (valid) |
| Contenu train | 50 594 patients · ~220 000 radiographies |
| Contenu valid | 200 patients · 234 radiographies |
| Format images | JPEG, résolution variable (≈ 320×390 px après resize) |
| Git | ❌ Exclu du repo (`.gitignore`) |

### ⚠️ Fiabilité et limites

- **Labels auto-générés par NLP** sur des rapports radiologiques (aucune relecture manuelle systématique)
- Taux d'incertitude élevé : ~35 % des labels sont `-1.0` (incertain)
- Biais de population : adultes hospitalisés à Stanford — non représentatif de la population générale
- Résolution réduite dans la version "Small" — certains détails fins peuvent être perdus

### Comment re-télécharger

1. Créer un compte Kaggle et accepter les conditions du dataset :
   👉 https://www.kaggle.com/datasets/ashery/chexpert
2. Générer un token API : https://www.kaggle.com/settings → **"Create New Token"**
3. Placer `kaggle.json` dans `C:\Users\<vous>\.kaggle\kaggle.json`

```bash
# Préparer le subset d'évaluation (utilise chexpert_raw/ si déjà présent)
.venv\Scripts\python.exe data/download_chexpert.py --raw-dir data/chexpert_raw

# Ou télécharger depuis Kaggle (si chexpert_raw/ absent) :
.venv\Scripts\python.exe data/download_chexpert.py
# → télécharge ashery/chexpert (~11 Go) dans chexpert_eval/_cache/
```

---

## Zone 3 — `chexpert_eval/` — Subset d'évaluation

> **Rôle :** Sous-ensemble prêt à l'emploi pour `run_evaluation.py`.
> Construit automatiquement par `download_chexpert.py` depuis `chexpert_raw/`.

| Propriété | Valeur |
|-----------|--------|
| Source | Extrait de `chexpert_raw/train/` (voir Zone 2) |
| Taille actuelle | ~21 images (~1 Mo) |
| Répartition cible | 10 normal · 10 suspected_opacity · 10 uncertain |
| Sélection | Aléatoire avec `seed=42` (reproductible) |
| Licence | Identique à CheXpert — recherche/éducation uniquement |
| Git | ❌ Exclu du repo (`.gitignore`) |

### Mapping des labels CheXpert → nos 3 classes

| Condition CheXpert | Classe projet |
|---|---|
| `No Finding = 1.0` | `normal` |
| `Consolidation = 1.0` OU `Pleural Effusion = 1.0` OU `Edema = 1.0` OU `Pneumonia = 1.0` | `suspected_opacity` |
| Colonne opacité `= -1.0` (incertain, aucune à 1.0) | `uncertain` |
| Autre / non classifiable | `uncertain` (garde-fou) |

### Limitation connue

Le dataset source (`ashery/chexpert`) contient très peu de patients avec `No Finding = 1.0`
dans le train set → la classe `normal` est souvent sous-représentée dans le subset.

### Lancer l'évaluation

```bash
# Évaluation sur les radios réelles :
.venv\Scripts\python.exe eval/run_evaluation.py \
  --mode gemini-baseline \
  --cases-csv data/chexpert_eval/cases.csv
```

---

## Référence et citation obligatoire

```
Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciosi, S., Chute, C., ... & Ng, A. Y. (2019).
CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 590-597.
https://doi.org/10.1609/aaai.v33i01.3301590
```

> ⚠️ **Rappel éthique** : CheXpert contient des radiographies de patients réels dé-identifiées.
> Usage **strictement limité** à la recherche et l'éducation. Ne jamais utiliser pour le diagnostic.
> Ne jamais redistribuer les images.
