# Données — Assistant Radiologue Virtuel

## Structure locale

```
data/
├── synthetic/              ← 30 images de test (PNG synthétiques, commitées)
│   ├── images/
│   └── cases.csv
│
├── chexpert_raw/           ← Dataset CheXpert brut (gitignored, ~8 Go)
│   ├── train/
│   ├── valid/
│   └── valid.csv
│
├── chexpert_eval/          ← 21 radios réelles prêtes pour l'évaluation (gitignored)
│   ├── images/
│   ├── cases.csv
│   └── _cache/
│
├── download_chexpert.py    ← Script pour recréer chexpert_raw/ et chexpert_eval/
└── README.md
```

---

## `synthetic/` — Tests et CI

30 images PNG générées par code (pas de vraies radios). Répartition : 10 normal, 10 suspected_opacity, 10 uncertain.
Servent uniquement à valider le pipeline et les tests automatiques. **Commitées dans le repo.**

```bash
# Évaluation sur les images synthétiques (défaut) :
.venv\Scripts\python eval/run_evaluation.py
```

---

## `chexpert_eval/` — Évaluation réelle

21 radiographies CheXpert sélectionnées pour évaluer les prompts. Répartition : 1 normal, 10 suspected_opacity, 10 uncertain.

```bash
# Évaluation sur les radios réelles (les 3 prompts d'un coup) :
.venv\Scripts\python eval/run_evaluation.py --mode all-groq --cases-csv data/chexpert_eval/cases.csv
```

---

## `chexpert_raw/` — Dataset source brut

Dataset CheXpert v1.0 Small (~8 Go). Utilisé par `download_chexpert.py` pour construire `chexpert_eval/`.

---

## Recréer les dossiers CheXpert depuis zéro

Si vous clonez le repo, `chexpert_raw/` et `chexpert_eval/` seront vides (gitignored). Voici comment les recréer :

### 1. Configurer Kaggle

```powershell
# Créer le dossier .kaggle
mkdir "$env:USERPROFILE\.kaggle" -Force

# Y copier le fichier kaggle.json téléchargé depuis https://www.kaggle.com/settings > API > Create New Token
Copy-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

### 2. Accepter les conditions du dataset

Allez sur https://www.kaggle.com/datasets/ashery/chexpert et cliquez sur **Download** une première fois pour accepter la licence.

### 3. Lancer le script

```powershell
.venv\Scripts\pip install kaggle
.venv\Scripts\python data/download_chexpert.py
```

Le script va :
1. Télécharger CheXpert depuis Kaggle → `chexpert_raw/`
2. Sélectionner 10 images par classe (normal, suspected_opacity, uncertain) → `chexpert_eval/`
3. Générer le fichier `chexpert_eval/cases.csv`

---

## Licence CheXpert

> Irvin, J., et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. AAAI 2019.

**Usage recherche et éducation uniquement.** Ne jamais redistribuer les images ni les utiliser pour du diagnostic.
