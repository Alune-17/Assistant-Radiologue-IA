# Données — Assistant Radiologue Virtuel

Ce projet utilise **deux datasets** avec des rôles distincts.

---

## 1. Dataset synthétique (pipeline de tests) — `data/sample_images/`

> **Rôle :** Valider l'architecture, les logs, les métriques et les smoke tests.  
> **Ne pas utiliser pour évaluer la performance médicale.**

| Propriété | Valeur |
|-----------|--------|
| Format | PNG synthétique (~27 Ko/image) |
| Taille | 30 images (~800 Ko total) |
| Classes | 10 normal, 10 suspected_opacity, 10 uncertain |
| Source | Généré programmatiquement (gribouillis Python) |
| Licence | Libre — usage interne uniquement |
| Git | ✅ Commité (petit, utile pour le CI) |

Ces images **ne ressemblent pas à de vraies radiographies**. Elles servent uniquement à s'assurer que le pipeline Python tourne de bout en bout.

---

## 2. Dataset CheXpert (évaluation réelle) — `data/chexpert_subset/`

> **Rôle :** Évaluation réelle du modèle sur des vraies radiographies thoraciques.  
> **Ce dossier n'est PAS commité dans Git** (voir `.gitignore`).

| Propriété | Valeur |
|-----------|--------|
| Nom | CheXpert v1.0 Small |
| Auteurs | Irvin et al. (2019), Stanford ML Group |
| Source Kaggle | https://www.kaggle.com/datasets/willarevalo/chexpert-v10-small |
| Source officielle | https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2 |
| Licence | **Recherche et éducation uniquement** — non commercial |
| Taille totale | ~11 Go (compressé) — 224 316 radiographies |
| Subset utilisé | **30 images** (10 par classe) — sélection aléatoire seed=42 |
| Git | ❌ Exclu du repo (trop volumineux, licence restrictive) |

### Mapping des labels CheXpert → nos 3 classes

| Condition CheXpert | Classe projet |
|-------------------|---------------|
| `No Finding = 1.0` | `normal` |
| `Consolidation = 1.0` OU `Pleural Effusion = 1.0` OU `Edema = 1.0` OU `Pneumonia = 1.0` | `suspected_opacity` |
| Label `-1.0` (incertain) sur les colonnes d'opacité | `uncertain` |
| Autre / non classifiable | `uncertain` |

### Comment télécharger le subset CheXpert

#### Prérequis

1. Créer un compte Kaggle : https://www.kaggle.com
2. Accepter les conditions du dataset sur la page Kaggle
3. Générer un token API : https://www.kaggle.com/settings → **"Create New Token"**
4. Placer `kaggle.json` dans `C:\Users\<vous>\.kaggle\kaggle.json`
5. Installer le package kaggle :

```bash
.venv\Scripts\pip install kaggle
```

#### Téléchargement (30 cas, ~50 Mo d'images)

```bash
# Télécharge depuis Kaggle et prépare 30 images (10 par classe)
.venv\Scripts\python.exe data/download_chexpert.py

# Options avancées
.venv\Scripts\python.exe data/download_chexpert.py --n-per-class 15 --seed 42
# → 45 cas (15 par classe)
```

#### Si CheXpert est déjà téléchargé localement

```bash
.venv\Scripts\python.exe data/download_chexpert.py \
  --chexpert-dir "C:\chemin\vers\CheXpert-v1.0-small" \
  --n-per-class 10
```

#### Lancer l'évaluation sur les images réelles

```bash
.venv\Scripts\python.exe eval/run_evaluation.py \
  --mode gemini-baseline \
  --cases-csv data/chexpert_subset/chexpert_cases.csv
```

---

## 3. Structure attendue après téléchargement

```
data/
├── sample_images/           ✅ commité — 30 images synthétiques (tests)
├── synthetic_cases.csv      ✅ commité — 30 cas synthétiques
├── download_chexpert.py     ✅ commité — script de téléchargement
├── README.md                ✅ commité — cette documentation
└── chexpert_subset/         ❌ non commité (.gitignore)
    ├── chexpert_cases.csv   ← généré par download_chexpert.py
    └── images/
        ├── CXR_CHEX_001_normal.jpg
        ├── CXR_CHEX_002_suspected_opacity.jpg
        └── ...
```

---

## 4. Références et citations

```
Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciosi, S., Chute, C., ... & Ng, A. Y. (2019).
CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 590-597.
https://doi.org/10.1609/aaai.v33i01.3301590
```

> ⚠️ **Rappel éthique** : CheXpert contient des radiographies de patients réels dé-identifiées.  
> Usage strictement limité à la recherche et l'éducation. Ne jamais utiliser pour le diagnostic.
