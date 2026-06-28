# Assistant radiologue virtuel responsable

> **Prototype pédagogique** — Non destiné au diagnostic. Validation par un professionnel qualifié requise.

Projet EFREI Solution Delivery — Filière Data (2025-2026).  
Prototype d'IA médicale multimodale pour l'analyse **prudente, traçable et évaluée** de radiographies thoraciques frontales.

---

## Démarrage rapide

### 1. Installation

```bash
python -m venv .venv
# Windows :
.venv\Scripts\activate
# Mac/Linux :
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration de la clé API

```bash
cp .env.example .env
# Éditez .env et collez votre clé API Groq (gratuite : https://console.groq.com/keys)
```

> **Ne pas commit le fichier `.env` sur Git**

### 3. Lancement de l'interface (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

### 4. Évaluation sur les images synthétiques

```bash
python eval/run_evaluation.py --mode toy
```

Cette commande génère dans `eval/outputs/` :

- `before_after_summary.csv` : comparaison baseline vs amélioration ;
- `*_predictions.csv` : prédictions détaillées ;
- `*_error_register.csv` : registre d'erreurs ;
- `evaluation_report.md` : rapport automatique ;
- `case_review_template.csv` : tableau à compléter pour les 20 à 30 cas commentés.

Vous pouvez aussi régénérer uniquement le template de revue humaine :

```bash
python eval/generate_case_review.py --out-dir eval/outputs --limit 30
```

### 5. Évaluation sur les vraies radios (CheXpert)

```bash
# Télécharger le sous-ensemble CheXpert (voir data/README.md pour les prérequis Kaggle)
python data/download_chexpert.py

# Lancer l'évaluation avec l'API Groq
python eval/run_evaluation.py --mode groq-baseline --cases-csv data/chexpert_eval/cases.csv
```

---

## Organisation du dépôt

```text
Assistant-Radiologue-IA/
├── README.md                       ← Ce fichier
├── .env.example                    ← Template de configuration (clé API Groq)
├── requirements.txt                ← Dépendances minimales
├── requirements-test.txt           ← Dépendances pour les tests
├── requirements-gpu.txt            ← Dépendances optionnelles (fine-tuning, GPU)
│
├── src/                            ← Code source principal
│   ├── inference.py                    Inférence toy + API Groq (Llama 4 Scout)
│   ├── guardrails.py                   Validation JSON, warning, fallback uncertain
│   ├── preprocessing.py                Qualité image (luminosité, contraste, résolution)
│   ├── metrics.py                      Accuracy, macro-F1, sensibilité, spécificité
│   └── database.py                     Connecteur SQLite
│
├── api/                            ← API REST
│   └── main.py                         FastAPI : GET / + POST /predict
│
├── app/                            ← Interfaces web
│   ├── streamlit_app.py                Interface principale (4 modes d'analyse)
│   └── gradio_app.py                   Interface alternative (mode toy)
│
├── eval/                           ← Évaluation, rapports et registre d'erreurs
│   ├── run_evaluation.py               Script d'évaluation batch
│   ├── reporting.py                    Rapport Markdown automatique
│   ├── case_review.py                  Génération du tableau des cas commentés
│   ├── generate_report.py              CLI pour régénérer le rapport
│   ├── generate_case_review.py         CLI pour régénérer la revue de cas
│   └── error_register_template.csv     Template du registre d'erreurs
│
├── data/                           ← Données
│   ├── synthetic/                      30 images synthétiques + cases.csv (CI/CD)
│   ├── download_chexpert.py            Script de téléchargement CheXpert via Kaggle
│   └── README.md                       Documentation des datasets et licences
│
├── prompts/                        ← Prompts système
│   ├── baseline_prompt.txt             Prompt anglais minimaliste
│   ├── improved_prompt.txt             Prompt anglais avec garde-fous renforcés
│   ├── french_prompt.txt               Prompt français
│   └── json_schema.md                  Spécification du schéma JSON de sortie
│
├── docs/                           ← Documentation
│   ├── appel_offre.pdf                 Appel d'offre original (PDF du professeur)
│   ├── architecture.md                 Architecture et pipeline du prototype
│   ├── ethique_et_limites.md           Guide éthique et garde-fous
│   ├── evaluation_protocol.md          Protocole d'évaluation et taxonomie d'erreurs
│   └── rapport_evaluation.md           Résultats Llama 4 Scout sur CheXpert
│
├── tests/                          ← Tests automatisés
│   └── test_repository_smoke.py        8 smoke tests (structure, schéma, guardrails, API)
│
├── notebooks/                      ← Notebooks pédagogiques (squelettes)
│   ├── 01_baseline_vlm.ipynb
│   ├── 02_prompt_comparison.ipynb
│   └── 03_optional_finetuning_lora.ipynb
│
├── finetuning/                     ← Stubs fine-tuning (Phase COULD)
│   ├── gemma4_unsloth_lora_stub.py
│   └── medgemma_peft_qlora_stub.py
│
└── sql/                            ← Schéma de base de données
    └── schema.sql                      4 tables : cases, prompts, runs, evaluations
```

---

## Contrat du projet

| Élément | Valeur |
|---|---|
| **Entrée** | Une radiographie thoracique frontale (PNG, JPG, JPEG, BMP) |
| **Sorties** | `normal`, `suspected_opacity`, `uncertain` |
| **Format** | JSON structuré avec confiance, observations, justification, limites et warning |
| **Données** | Synthétiques ou publiques, autorisées et dé-identifiées |
| **Finalité** | Prototype éducatif — pas d'aide au diagnostic réelle |

---

## API

```bash
# Lancer le serveur
uvicorn api.main:app --reload

# Tester une prédiction
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@data/synthetic/images/CXR_SYN_002_suspected_opacity.png"
```

Modèles disponibles via `?model=` : `toy` (défaut), `groq-baseline`, `groq-improved`.

---

## Smoke tests

```bash
pip install -r requirements-test.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Vérifie : structure du dépôt, contrat dataset, schéma JSON, guardrails, compilation Python, API, évaluation, rapport automatique et template des cas commentés.

---

## Livrables attendus

| Niveau | Attendu |
|---|---|
| **MUST** | Baseline reproductible, sortie JSON valide, warning obligatoire, logs, métriques, mini-rapport |
| **SHOULD** | Prompt amélioré, comparaison baseline/amélioration, dashboard, registre d'erreurs, template 20-30 cas commentés |
| **COULD** | LoRA expérimental, MedGemma/PEFT, localisation visuelle, ablations de prompts |

---

## Licence

Le code pédagogique est publié sous licence MIT.  
**Les datasets externes et modèles conservent leurs licences propres** — voir `data/README.md` pour les détails.
