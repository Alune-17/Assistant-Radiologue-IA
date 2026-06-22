# Cahier des Charges — Assistant Radiologue Virtuel Responsable

> **Projet :** Solution Delivery — Filière Data, EFREI  
> **Année académique :** 2025-2026  
> **Date du document :** 15 juin 2026 — _mis à jour le 22 juin 2026_  
> **Source :** Analyse du repository GitHub `assistant-radiologue-virtuel-main`

---

## Table des matières

1. [Contexte et objectif du projet](#1-contexte-et-objectif-du-projet)
2. [Audit de l'existant — ce qui est à disposition](#2-audit-de-lexistant--ce-qui-est-à-disposition)
3. [Problèmes de propreté du repository](#3-problèmes-de-propreté-du-repository)
4. [Exigences fonctionnelles (MUST / SHOULD / COULD)](#4-exigences-fonctionnelles-must--should--could)
5. [Analyse des écarts — ce qui est à faire](#5-analyse-des-écarts--ce-qui-est-à-faire)
6. [Plan de réalisation par phases](#6-plan-de-réalisation-par-phases)
7. [Critères d'évaluation et livrables](#7-critères-dévaluation-et-livrables)
8. [Contraintes éthiques et réglementaires](#8-contraintes-éthiques-et-réglementaires)
9. [Stack technique et dépendances](#9-stack-technique-et-dépendances)
10. [Stratégie dataset — quantité et hébergement](#10-stratégie-dataset--quantité-et-hébergement)
11. [Journal des sessions de développement](#11-journal-des-sessions-de-développement)

---

## 1. Contexte et objectif du projet

### 1.1 Description

Prototype pédagogique d'IA médicale multimodale pour l'analyse **prudente, traçable et évaluée** de radiographies thoraciques frontales. Le projet n'est **pas** un dispositif médical et ne doit jamais être utilisé pour le diagnostic.

### 1.2 Entrée / Sortie

| Élément | Description |
|---------|-------------|
| **Entrée** | Une radiographie thoracique frontale (PNG, JPG, JPEG, BMP) |
| **Sorties** | `normal`, `suspected_opacity`, `uncertain` |
| **Format** | JSON structuré avec confiance, observations, justification, limites et warning |
| **Données** | Synthétiques ou publiques, autorisées et dé-identifiées |
| **Finalité** | Prototype éducatif — pas d'aide au diagnostic réelle |

### 1.3 Philosophie

> *"Un prototype fiable, prudent et documenté vaut mieux qu'une solution spectaculaire mais impossible à défendre."*

Le bon rendu démontre une **méthode** : périmètre limité, baseline reproductible, garde-fous, évaluation, analyse d'erreurs et limites explicites.

---

## 2. Audit de l'existant — ce qui est à disposition

### 2.1 Arborescence du repository

```text
assistant-radiologue-virtuel-main/

├── .github/workflows/ci.yml      ✅ CI basique (pytest + eval)
├── .gitignore                     ⚠️ Incomplet (ne bloque pas tout)
├── LICENSE                        ✅ MIT
├── README.md                      ✅ Documenté
├── api/
│   └── main.py                    ✅ FastAPI avec /predict
├── app/
│   ├── streamlit_app.py           ✅ Interface Streamlit
│   └── gradio_app.py              ✅ Interface Gradio
├── data/
│   ├── README.md                  ✅ Documentation données
│   ├── synthetic_cases.csv        ✅ 30 cas synthétiques
│   └── sample_images/             ✅ 30 images PNG synthétiques
├── docs/
│   ├── appel_offre.md             ✅ Appel d'offre complet
│   ├── appel_offre.pdf            ✅ Version PDF
│   ├── architecture.md            ✅ Architecture cible
│   ├── ethique_et_limites.md      ✅ Guide éthique
│   └── evaluation_protocol.md     ✅ Protocole d'évaluation
├── eval/
│   ├── run_evaluation.py          ✅ Script d'évaluation
│   ├── error_register_template.csv ✅ Template registre d'erreurs
│   └── outputs/                   ⚠️ Devrait être dans .gitignore mais présent
│       ├── baseline_metrics.json
│       ├── improved_metrics.json
│       ├── baseline_predictions.csv
│       ├── improved_predictions.csv
│       └── before_after_summary.csv
├── finetuning/
│   ├── gemma4_unsloth_lora_stub.py   ✅ Squelette uniquement
│   └── medgemma_peft_qlora_stub.py   ✅ Squelette uniquement
├── notebooks/
│   ├── 01_baseline_vlm.ipynb         ✅ Notebook démarrage
│   ├── 02_prompt_comparison.ipynb    ✅ Notebook comparaison
│   └── 03_optional_finetuning_lora.ipynb ✅ Notebook LoRA
├── prompts/
│   ├── baseline_prompt.txt        ✅ Prompt baseline anglais
│   ├── improved_prompt.txt        ✅ Prompt amélioré anglais
│   ├── gemini_prompt.txt          ✅ Prompt Gemini français
│   └── json_schema.md            ✅ Schéma JSON documenté
├── sql/
│   └── schema.sql                 ✅ Schéma SQLite 4 tables
├── src/
│   ├── __init__.py                ✅ Package Python
│   ├── database.py                ✅ Connecteur SQLite
│   ├── guardrails.py              ✅ Garde-fous et validation
│   ├── inference.py               ✅ Inférence toy + Gemini
│   ├── metrics.py                 ✅ Accuracy, macro-F1, etc.
│   └── preprocessing.py          ✅ Chargement et qualité image
├── tests/
│   └── test_repository_smoke.py   ✅ 7 tests de contrat
├── medical_ai_evidence.sqlite     ⚠️ DB présente (devrait être ignorée)
├── pyproject.toml                 ✅ Config pytest
├── requirements.txt               ✅ Dépendances principales
└── requirements-test.txt          ✅ Dépendances de test
```

### 2.2 Détail des composants existants

#### Source (`src/`)

| Fichier | Lignes | État | Description |
|---------|--------|------|-------------|
| [inference.py](file:///c:/Users/cleme/Desktop/EFREI/ING1-NEW/Mastercamp/SOLUTION%20DELIVERY/assistant-radiologue-virtuel-main/src/inference.py) | 199 | ✅ **Mis à jour** | SDK migré vers `google.genai`. Modèle `gemini-2.0-flash`. Fonctions : `toy_predict()`, `gemini_predict()`, `gemini_predict_baseline()`, `gemini_predict_improved()`. Client singleton `_client`. |
| [guardrails.py](file:///c:/Users/cleme/Desktop/EFREI/ING1-NEW/Mastercamp/SOLUTION%20DELIVERY/assistant-radiologue-virtuel-main/src/guardrails.py) | 39 | **Fonctionnel** | Validation JSON, warning obligatoire, fallback `uncertain` |
| [preprocessing.py](file:///c:/Users/cleme/Desktop/EFREI/ING1-NEW/Mastercamp/SOLUTION%20DELIVERY/assistant-radiologue-virtuel-main/src/preprocessing.py) | 65 | ✅ **Mis à jour** | Analyse réelle : luminosité moyenne, contraste (std), résolution minimale, ratio d'aspect. Remplace le flag basé sur le nom de fichier. |
| [metrics.py](file:///c:/Users/cleme/Desktop/EFREI/ING1-NEW/Mastercamp/SOLUTION%20DELIVERY/assistant-radiologue-virtuel-main/src/metrics.py) | 86 | ✅ **Mis à jour** | + `sensitivity()`, `specificity()`, `median_latency()`. `summarize_metrics()` enrichi. |
| [database.py](file:///c:/Users/cleme/Desktop/EFREI/ING1-NEW/Mastercamp/SOLUTION%20DELIVERY/assistant-radiologue-virtuel-main/src/database.py) | 42 | **Fonctionnel** | Connexion SQLite, `init_db()`, `insert_run()` |

#### API (`api/`)

| Fichier | État | Description |
|---------|------|-------------|
| [main.py](file:///c:/Users/cleme/Desktop/EFREI/ING1-NEW/Mastercamp/SOLUTION%20DELIVERY/assistant-radiologue-virtuel-main/api/main.py) | ✅ **Mis à jour** | FastAPI avec `GET /` (health + liste modèles) et `POST /predict?model=toy\|gemini-baseline\|gemini-improved`. |

#### Applications web (`app/`)

| Fichier | État | Description |
|---------|------|-------------|
| [streamlit_app.py](file:///c:/Users/cleme/Desktop/EFREI/ING1-NEW/Mastercamp/SOLUTION%20DELIVERY/assistant-radiologue-virtuel-main/app/streamlit_app.py) | ✅ **Mis à jour** | 4 modes : Toy baseline, Gemini baseline, Gemini improved, Gemini prompt français. UI enrichie (3 colonnes, latence, spinner). |
| [gradio_app.py](file:///c:/Users/cleme/Desktop/EFREI/ING1-NEW/Mastercamp/SOLUTION%20DELIVERY/assistant-radiologue-virtuel-main/app/gradio_app.py) | **Fonctionnel** | Interface Gradio minimale, mode baseline/improved uniquement |

#### Données (`data/`)

- **30 images synthétiques** PNG (~27 Ko chacune) — simulacres grossiers de radiographies
- **30 cas** dans `synthetic_cases.csv` : 10 normal, 10 suspected_opacity, 10 uncertain
- Split : 20 `smoke`, 10 `final`

#### Tests (`tests/`)

| Test | Vérifie |
|------|---------|
| `test_repository_student_contract_is_present` | Présence des fichiers obligatoires + absence de fichiers interdits |
| `test_synthetic_dataset_contract_is_valid` | Contrat du dataset (≥20 cas, colonnes, labels, images existantes) |
| `test_prediction_schema_warning_and_guardrails` | Schéma JSON, warning, guardrails |
| `test_python_source_tree_compiles` | Compilation Python |
| `test_invalid_model_output_falls_back_to_uncertain` | Fallback en cas de sortie invalide |
| `test_metrics_and_api_health_contract` | Métriques + endpoint health |
| `test_api_predict_preserves_uploaded_case_signal` | Endpoint /predict fonctionnel |
| `test_evaluation_command_runs` | Commande d'évaluation end-to-end |

#### Prompts (`prompts/`)

| Fichier | Langue | Utilisation |
|---------|--------|-------------|
| `baseline_prompt.txt` | Anglais | Prompt simple — analyse basique |
| `improved_prompt.txt` | Anglais | Prompt renforcé — règles d'incertitude strictes |
| `gemini_prompt.txt` | Français | Prompt spécifique API Gemini |
| `json_schema.md` | N/A | Spécification du schéma JSON attendu |

---

## 3. Problèmes de propreté du repository

> [!CAUTION]
> Le repository contient plusieurs problèmes critiques qui doivent être résolus avant toute livraison.

### 3.1 Problèmes critiques

| # | Problème | Fichier | Gravité |
|---|----------|---------|---------|
| 1 | **Base SQLite commitée** | `medical_ai_evidence.sqlite` (90 Ko) | ✅ **RÉSOLU** — Supprimé du repo |
| 2 | **eval/outputs/ commité** | `eval/outputs/` (5 fichiers) | ✅ **RÉSOLU** — Supprimé du repo |
| 3 | **`.venv/` commité** | Répertoire `.venv/` | 🟠 **À nettoyer de l'historique Git** |

### 3.2 Problèmes modérés

| # | Problème | Impact |
|---|----------|--------|
| 5 | Le `toy_predict` lit les labels **à partir du nom de fichier** → accuracy 100% artificielle | ⚠️ Toujours présent — c'est le comportement attendu du toy |
| 6 | `vlm_predict_placeholder()` est un **stub** qui renvoie simplement `toy_predict` | ⚠️ Toujours présent — stub conservé pour HuggingFace |
| 7 | Le `basic_quality_flag()` était basé sur le nom du fichier | ✅ **RÉSOLU** — Analyse pixel réelle implementée |
| 8 | Baseline et improved avaient des **métriques identiques** (1.0 partout) | 🔄 En attente clé API — test réel à faire |
| 9 | Notebooks probablement vides | 🔄 À faire Phase 2 |
| 10 | `gemini_predict()` référençait `gemini-3.5-flash` — inexistant | ✅ **RÉSOLU** — Corrigé vers `gemini-2.0-flash` |

### 3.3 Problèmes mineurs

| # | Problème |
|---|----------|
| 11 | Pas de `__init__.py` dans `api/`, `app/`, `eval/`, `finetuning/` |
| 12 | Le schéma SQL a 4 tables mais seule `runs` est utilisée dans le code |
| 13 | Pas de sensibilité/spécificité dans `metrics.py` (mentionné dans le protocole d'évaluation) |
| 14 | Pas de calcul de latence médiane dans les métriques |
| 15 | Le `.gitignore` liste `eval/outputs/` et `medical_ai_evidence.sqlite` mais ils sont quand même présents |

---

## 4. Exigences fonctionnelles (MUST / SHOULD / COULD)

### 4.1 MUST — Socle obligatoire

> [!IMPORTANT]
> Toutes ces exigences **doivent** être livrées pour une note de passage.

| # | Exigence | État actuel | Action |
|---|----------|-------------|--------|
| M1 | Application web fonctionnelle avec upload d'image | ✅ Streamlit + Gradio présents | Vérifier exécution réelle |
| M2 | Dépôt d'image et traitement | ✅ Upload fonctionnel | OK |
| M3 | Sortie JSON avec les 3 classes | ✅ Schéma validé | OK |
| M4 | Warning obligatoire dans toutes les sorties | ✅ Guardrails en place | OK |
| M5 | Baseline reproductible | ⚠️ Toy uniquement (labels dans noms de fichiers) | **À refaire avec un vrai modèle** |
| M6 | Logs des résultats | ✅ SQLite `insert_run()` | OK |
| M7 | CSV de résultats | ✅ Généré par `run_evaluation.py` | OK |
| M8 | Mini-rapport | ❌ **Absent** | **À rédiger** |
| M9 | Analyse minimale des limites | ✅ Document `ethique_et_limites.md` | À enrichir |

### 4.2 SHOULD — Niveau attendu

| # | Exigence | État actuel | Action |
|---|----------|-------------|--------|
| S1 | Prompt amélioré | ✅ `improved_prompt.txt` existe | Tester réellement avec un VLM |
| S2 | Règle d'incertitude | ✅ Implémentée dans guardrails | OK |
| S3 | Comparaison baseline vs amélioration | ❌ **Résultats identiques (1.0/1.0)** | **Comparaison réelle à réaliser** |
| S4 | Dashboard de métriques | ❌ **Absent** | **À créer** |
| S5 | SQLite fonctionnel | ✅ En place | OK (compléter les 4 tables) |
| S6 | Smoke test automatisé | ✅ 7 tests + CI GitHub Actions | OK |
| S7 | Analyse d'erreurs sur cas commentés | ⚠️ Template existe mais 3 exemples seulement | **À compléter (20-30 cas)** |

### 4.3 COULD — Approfondissements

| # | Exigence | État actuel | Action |
|---|----------|-------------|--------|
| C1 | LoRA Gemma 4 via Unsloth | ❌ Squelette vide | Optionnel — nécessite GPU |
| C2 | Adaptation MedGemma via PEFT/QLoRA | ❌ Squelette vide | Optionnel — nécessite GPU + accès modèle |
| C3 | Localisation visuelle | ❌ Non implémenté | Optionnel |
| C4 | Ablation systématique de prompts | ❌ Non implémenté | Optionnel |

---

## 5. Analyse des écarts — ce qui est à faire

### 5.1 Écarts critiques à combler

> [!WARNING]
> Ces points représentent le travail principal à réaliser pour satisfaire les exigences de l'appel d'offre.

#### 🔴 1. Implémenter une vraie inférence VLM (baseline réelle)

**Situation :** Le `toy_predict()` actuel lit les labels depuis les noms de fichiers — ce n'est pas de l'inférence. Le `gemini_predict()` existe mais n'a pas été testé/évalué systématiquement.

**À faire :**
- Connecter un vrai modèle VLM (Gemini API ou MedGemma en local)
- Évaluer la baseline sur les 30 cas synthétiques + si possible sur des cas réels publics
- Produire des métriques réalistes (≠ 1.0 partout)

#### 🔴 2. Réaliser une comparaison baseline vs prompt amélioré

**Situation :** Les résultats `before_after_summary.csv` montrent des métriques identiques (accuracy 1.0, macro_f1 1.0) pour baseline et improved — pas de différence observable.

**À faire :**
- Exécuter les deux modes avec un vrai modèle
- Documenter les différences mesurées
- Produire un rapport de comparaison

#### 🔴 3. Rédiger le rapport final / mini-rapport

**Situation :** Aucun rapport n'est présent dans le repository.

**À faire :**
- Rédiger un rapport couvrant : dataset, prompts, métriques, limites, risques
- Documenter les choix techniques et les résultats

#### 🟠 4. Compléter le registre d'erreurs (20-30 cas)

**Situation :** Le template [error_register_template.csv](file:///c:/Users/cleme/Desktop/EFREI/ING1-NEW/Mastercamp/SOLUTION%20DELIVERY/assistant-radiologue-virtuel-main/eval/error_register_template.csv) ne contient que 3 exemples.

**À faire :**
- Analyser 20-30 cas avec le modèle réel
- Classifier les erreurs (FN, FP, UA, JF, HT)
- Commenter chaque cas

#### 🟠 5. Créer un dashboard de métriques

**Situation :** Aucun dashboard n'existe.

**À faire :**
- Créer une page de visualisation (dans Streamlit ou séparée)
- Afficher : accuracy, macro-F1, sensibilité, spécificité, taux JSON valide, taux warning, latence, matrice de confusion

#### 🟠 6. Nettoyer le repository

**Situation :** Fichiers interdits présents, artefacts générés commités.

**À faire :**
- Supprimer `medical_ai_evidence.sqlite`, `eval/outputs/`, `.venv/`
- Révoquer et renouveler la clé API Google
- Remplacer `.env` par `.env.example` avec placeholder
- Vérifier l'historique Git pour les secrets

### 5.2 Écarts modérés

#### 🟡 7. Enrichir les métriques

**À faire dans** [metrics.py] :
- Ajouter sensibilité (rappel sur `suspected_opacity`)
- Ajouter spécificité (rappel sur `normal`)
- Ajouter latence médiane
- Ajouter détection d'hallucinations textuelles (colonne manuelle)

#### 🟡 8. Implémenter une vraie analyse de qualité image

**À faire dans** [preprocessing.py] :
- Remplacer le flag basé sur le nom de fichier
- Analyser la luminosité, le contraste, la résolution
- Détecter les images trop sombres/claires

#### 🟡 9. Compléter l'utilisation des tables SQLite

**Situation :** Le [schéma SQL] définit 4 tables (`cases`, `prompts`, `runs`, `evaluations`) mais seule `runs` est alimentée.

**À faire :**
- Peupler la table `cases` depuis le CSV
- Stocker les prompts dans la table `prompts`
- Remplir `evaluations` avec les résultats du registre d'erreurs

#### 🟡 10. Compléter les notebooks

**À faire :**
- `01_baseline_vlm.ipynb` : exécuter et documenter la baseline avec un vrai modèle
- `02_prompt_comparison.ipynb` : comparer baseline vs improved avec métriques
- `03_optional_finetuning_lora.ipynb` : documenter une tentative LoRA (optionnel)

---

## 6. Plan de réalisation par phases

### Phase 1 — Nettoyage et mise en conformité ✅ TERMINÉE

| # | Tâche | Effort estimé | Statut |
|---|-------|---------------|--------|
| 1.1 | Générer une nouvelle Clé API Google et créer un `.env.example` | 30 min | ✅ `.env.example` créé, `.env` avec placeholder |
| 1.2 | Supprimer `medical_ai_evidence.sqlite` du repo | 15 min | ✅ Supprimé |
| 1.3 | Supprimer `eval/outputs/` du repo | 15 min | ✅ Supprimé |
| 1.4 | Supprimer `.venv/` du repo | 15 min | 🔄 À nettoyer de l'historique Git |
| 1.5 | Vérifier et nettoyer l'historique Git | 1h | 🔄 À faire |
| 1.6 | Vérifier que le CI passe après nettoyage | 30 min | ✅ **8/8 smoke tests passent** |

**Durée totale Phase 1 : ~2h30** — ✅ Essentiellement complète

---

### Phase 2 — Baseline fonctionnelle avec vrai modèle 🔄 EN COURS

| # | Tâche | Effort estimé | Statut |
|---|-------|---------------|--------|
| 2.1 | Configurer et tester l'accès API Gemini (nouvelle clé) | 1h | 🔄 **Clé API à renseigner dans `.env`** |
| 2.2 | Adapter `gemini_predict()` — SDK `google.genai`, modèle `gemini-2.0-flash` | 2h | ✅ Fait — `_gemini_call()`, `gemini_predict_baseline()`, `gemini_predict_improved()` |
| 2.3 | Exécuter la baseline sur les 30 cas synthétiques avec Gemini | 1h | 🔄 **En attente de la clé API** |
| 2.4 | Implémenter une vraie analyse de qualité image dans `preprocessing.py` | 2h | ✅ Fait — luminosité, contraste (std), résolution, ratio |
| 2.5 | Ajouter sensibilité, spécificité, latence médiane dans `metrics.py` | 1h | ✅ Fait |
| 2.6 | Documenter le notebook `01_baseline_vlm.ipynb` | 2h | 🔄 À faire après obtention des résultats Gemini |

**Durée totale Phase 2 : ~9h** — 🔄 60% complète, bloquée sur clé API

---

### Phase 3 — Comparaison et amélioration (Priorité : haute)

| # | Tâche | Effort estimé |
|---|-------|---------------|
| 3.1 | Exécuter le prompt amélioré (`improved_prompt.txt`) avec Gemini | 1h |
| 3.2 | Comparer les métriques baseline vs improved | 1h |
| 3.3 | Ajuster les seuils d'incertitude si nécessaire | 1h |
| 3.4 | Produire le CSV et JSON de comparaison | 30 min |
| 3.5 | Documenter le notebook `02_prompt_comparison.ipynb` | 2h |
| 3.6 | Itérer sur les prompts si les résultats ne sont pas différenciés | 2h |

**Durée totale Phase 3 : ~7h30**

---

### Phase 4 — Évaluation et registre d'erreurs (Priorité : haute)

| # | Tâche | Effort estimé |
|---|-------|---------------|
| 4.1 | Analyser 20-30 cas un par un avec le modèle réel | 3h |
| 4.2 | Remplir le registre d'erreurs avec classification (FN/FP/UA/JF/HT) | 2h |
| 4.3 | Compléter les tables SQLite (`cases`, `prompts`, `evaluations`) | 2h |
| 4.4 | Générer la matrice de confusion et le résumé d'erreurs | 1h |

**Durée totale Phase 4 : ~8h**

---

### Phase 5 — Dashboard et interface (Priorité : moyenne)

| # | Tâche | Effort estimé |
|---|-------|---------------|
| 5.1 | Créer un dashboard Streamlit de métriques (graphiques, matrice de confusion) | 3h |
| 5.2 | Améliorer l'interface Streamlit (UX, affichage des erreurs, historique) | 2h |
| 5.3 | Tester l'app Gradio et s'assurer de la cohérence | 1h |
| 5.4 | Intégrer le modèle Gemini dans l'API FastAPI `/predict` (option au lieu de toy) | 1h |

**Durée totale Phase 5 : ~7h**

---

### Phase 6 — Rapport et documentation finale (Priorité : haute)

| # | Tâche | Effort estimé |
|---|-------|---------------|
| 6.1 | Rédiger le mini-rapport (dataset, prompts, métriques, limites, risques) | 4h |
| 6.2 | Documenter les choix techniques et architecture dans `docs/` | 2h |
| 6.3 | Mettre à jour le README avec les résultats réels | 1h |
| 6.4 | Préparer les éléments de soutenance (commandes exécutables, preuves) | 2h |

**Durée totale Phase 6 : ~9h**

---

### Phase 7 — Approfondissements optionnels (COULD) (Priorité : basse)

| # | Tâche | Effort estimé |
|---|-------|---------------|
| 7.1 | Tenter un fine-tuning LoRA avec Gemma 4 / Unsloth (si GPU disponible) | 8-16h |
| 7.2 | Tester MedGemma comme alternative (si accès autorisé) | 4-8h |
| 7.3 | Implémenter une localisation visuelle (heatmap / attention) | 4h |
| 7.4 | Ablation systématique de prompts | 3h |

**Durée totale Phase 7 : 19-31h (optionnel)**

---

### Résumé des efforts

| Phase | Durée estimée | Priorité |
|-------|---------------|----------|
| Phase 1 — Nettoyage | ~2h30 | 🔴 Immédiate |
| Phase 2 — Baseline réelle | ~9h | 🔴 Haute |
| Phase 3 — Comparaison | ~7h30 | 🔴 Haute |
| Phase 4 — Évaluation | ~8h | 🔴 Haute |
| Phase 5 — Dashboard | ~7h | 🟠 Moyenne |
| Phase 6 — Rapport | ~9h | 🔴 Haute |
| **Total obligatoire** | **~43h** | |
| Phase 7 — Optionnel | ~19-31h | 🟡 Basse |

---

## 7. Critères d'évaluation et livrables

### 7.1 Barème (tiré de l'appel d'offre)

| Critère | Poids | Ce qu'il faut démontrer |
|---------|-------|------------------------|
| **Périmètre + dataset** | 15% | Jeu synthétique documenté, sources citées, limitations connues |
| **Baseline fonctionnelle** | 15% | Modèle réel fonctionnel, résultats reproductibles, JSON valide |
| **Amélioration mesurée** | 20% | Comparaison baseline/improved avec métriques différenciées |
| **Intégration application** | 15% | App web fonctionnelle, upload, warning, logs, API |
| **Évaluation + erreurs** | 20% | Registre 20-30 cas, matrice de confusion, analyse qualitative |
| **Éthique + limites** | 10% | Document éthique, warning partout, limites explicites |
| **Oral professionnel** | 5% | Commandes exécutables, preuves visuelles, défense des erreurs |

### 7.2 Liste de contrôle des livrables

- [ ] Dépôt GitHub propre et documenté
- [ ] Baseline reproductible avec métriques réalistes
- [ ] Comparaison baseline vs prompt amélioré
- [ ] Application web avec upload, warning, logs
- [ ] Dashboard / CSV de métriques
- [ ] Registre d'erreurs sur 20-30 cas commentés
- [ ] Rapport final (dataset, prompts, limites, risques)
- [ ] Smoke tests passant (CI green)
- [ ] Toutes les sorties contiennent le warning
- [ ] Aucune clé API en clair dans le repo

---

## 8. Contraintes éthiques et réglementaires

> [!CAUTION]
> Non-respect = risque de note éliminatoire.

| Contrainte | Détail |
|-----------|--------|
| **Pas de diagnostic** | Le prototype ne doit jamais donner de conclusion clinique |
| **Warning obligatoire** | *"Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."* — dans l'UI, le JSON, le README, la soutenance et le rapport |
| **Données autorisées** | Synthétiques ou publiques dé-identifiées uniquement |
| **Pas de données patient** | Jamais de nom, prénom, date de naissance, ID patient |
| **Classe `uncertain`** | Garde-fou obligatoire — ne pas la supprimer |
| **Erreurs en soutenance** | Montrer les FP, FN, incertitudes — pas seulement les réussites |
| **Licences** | Citer source, version, licence, conditions d'accès pour chaque ressource externe |

---

## 9. Stack technique et dépendances

### 9.1 Dépendances principales

| Package | Version min | Usage |
|---------|-------------|-------|
| `fastapi` | ≥0.115 | API REST `/predict` |
| `uvicorn` | ≥0.30 | Serveur ASGI |
| `streamlit` | ≥1.37 | Interface web principale |
| `gradio` | ≥4.44 | Interface web alternative |
| `pillow` | ≥10.4 | Traitement d'image |
| `pandas` | ≥2.2 | Manipulation de données |
| `scikit-learn` | ≥1.5 | Métriques (non utilisé dans le code actuel) |
| `numpy` | ≥1.26 | Calcul numérique — utilisé dans `preprocessing.py` |
| `opencv-python` | ≥4.10 | Vision (non utilisé dans le code actuel) |
| `pydicom` | ≥2.4 | DICOM (non utilisé dans le code actuel) |
| `transformers` | ≥4.56 | Modèles HuggingFace (pour COULD) |
| `accelerate` | ≥0.34 | Accélération GPU (pour COULD) |
| `torch` | Latest | PyTorch (pour COULD) |
| `google-genai` | ≥1.0.0 | ✅ API Gemini — remplace `google-generativeai` (déprécié) |
| `python-dotenv` | ≥1.0.0 | Chargement `.env` |

### 9.2 Dépendances de test

| Package | Version min |
|---------|-------------|
| `pytest` | ≥8.0 |
| `httpx` | ≥0.27 |

### 9.3 Infrastructure

| Composant | Technologie |
|-----------|-------------|
| Base de données | SQLite (fichier local) |
| CI/CD | GitHub Actions |
| Python | 3.11 |
| Licence | MIT |

> [!NOTE]
> Plusieurs dépendances sont déclarées mais **pas utilisées** dans le code actuel : `scikit-learn`, `opencv-python`, `pydicom`, `transformers`, `accelerate`, `torch`. Elles sont là pour les phases COULD (fine-tuning LoRA). Envisager de les séparer dans un `requirements-optional.txt` pour alléger l'installation de base.

---

## 10. Stratégie dataset — quantité et hébergement

> [!CAUTION]
> **GitHub refuse les dépôts > 1 Go et les fichiers > 100 Mo.** Un dataset de 10 Go ne peut pas être poussé sur GitHub directement.

### 10.1 Ce que l'appel d'offre exige réellement

L'appel d'offre demande :
- Un **registre d'erreurs sur 20 à 30 cas commentés** (pas 10 Go)
- Un jeu de données **documenté, cité, avec licence** (pas nécessairement téléchargé)
- Des images **synthétiques ou publiques dé-identifiées**

Le dataset synthétique existant (**30 images, ~800 Ko au total**) suffit pour valider le pipeline. Les 30 cas couvrent les 3 classes (10 normaux, 10 opacités, 10 incertains) et les splits smoke/final.

### 10.2 Que faire d'un dataset de 10 Go

| Option | Recommandation | Détail |
|--------|---------------|--------|
| **Ne pas pousser 10 Go sur GitHub** | ✅ Recommandé | GitHub limite à 1 Go par repo, 100 Mo par fichier |
| **Git LFS** | ⚠️ Partiel | 1 Go gratuit seulement — insuffisant pour 10 Go |
| **Hugging Face Datasets** | ✅ Solution idéale | Hébergement gratuit, versioning, citation DOI — `datasets` library |
| **Google Drive / lien externe** | ✅ Acceptable | Documenter l'URL et la licence dans `data/README.md` |
| **Kaggle Datasets** | ✅ Acceptable | Hébergement gratuit avec versioning |
| **Zenodo** | ✅ Pour publication | DOI citable, idéal pour rapport académique |

### 10.3 Recommandation pour ce projet

Conserver dans le repo :
```
data/
  synthetic_cases.csv          # 30 cas synthétiques — pipeline de tests
  sample_images/               # 30 images PNG synthétiques (~800 Ko total)
  download_chexpert.py         # Script de téléchargement CheXpert via Kaggle API
  README.md                    # Documentation + sources + licences
```

Pour les images publiques réelles :
1. **NE PAS les commiter** — `data/chexpert_subset/` est dans `.gitignore`
2. Documenter dans `data/README.md` : source, licence, version, URL
3. Utiliser `data/download_chexpert.py` pour télécharger le sous-ensemble de 30 images
4. Passer `--cases-csv data/chexpert_subset/chexpert_cases.csv` à `run_evaluation.py`

> [!IMPORTANT]
> L'évaluation porte sur la **méthode et la documentation**, pas sur la taille du dataset. 30 cas bien annotés et commentés valent plus que 10 Go de données non documentées.

### 10.4 État du dataset CheXpert ✅ TÉLÉCHARGÉ

| Élément | Détail |
|---------|--------|
| CSV labels | `amritpal333/chexpert-train-csv-modified` — 223 414 radios labelisées |
| Images sample | `duong1589/chexpert` — 1 002 vraies radios JPEG (~52 Mo) |
| Subset généré | 30 cas équilibrés : 10 normal / 10 suspected_opacity / 10 uncertain |
| CSV de cas | `data/chexpert_subset/chexpert_cases.csv` — compatible `run_evaluation.py` |
| Reproductibilité | Graine seed=42, script versionné, labels appariés aux images |
| Prochaine étape | Fournir la clé API Gemini pour lancer l'évaluation réelle |

---

## 11. Journal des sessions de développement

### Session 1 — 22 juin 2026

**Objectif :** Phases 1 & 2 — Nettoyage repo + infrastructure API Gemini + dataset réel

#### Nettoyage et conformité (Phase 1)

| Action | Résultat |
|--------|----------|
| Migration SDK `google.generativeai` → `google.genai` v2.9.0 | ✅ Terminé |
| Correction modèle : `gemini-3.5-flash` → `gemini-2.0-flash` | ✅ Terminé |
| Suppression fichiers interdits : `eval/outputs/`, `medical_ai_evidence.sqlite` | ✅ Terminé |
| `.env.example` créé, `.env` avec placeholder (clé non commitée) | ✅ Terminé |
| `.gitignore` enrichi : `data/chexpert_subset/`, `data/*.zip` | ✅ Terminé |
| Smoke tests : **8/8 passent** | ✅ Terminé |

#### Infrastructure API Gemini (Phase 2 — partiel)

| Action | Résultat |
|--------|----------|
| `inference.py` : `_gemini_call()`, `gemini_predict_baseline()`, `gemini_predict_improved()` | ✅ Terminé |
| `preprocessing.py` : analyse qualité pixel réelle (luminosité, contraste, résolution) | ✅ Terminé |
| `metrics.py` : sensibilité, spécificité, latence médiane | ✅ Terminé |
| `run_evaluation.py` : modes `gemini-baseline`, `gemini-improved`, `all-gemini`, `--cases-csv` | ✅ Terminé |
| `api/main.py` : paramètre `?model=toy\|gemini-baseline\|gemini-improved` | ✅ Terminé |
| `streamlit_app.py` : 4 modes, UI enrichie (3 colonnes, spinner, latence) | ✅ Terminé |
| Évaluation Gemini sur vraies radios | 🔄 **En attente clé API Google** |

#### Dataset réel CheXpert (Phase 2 — données)

| Action | Résultat |
|--------|----------|
| Stratégie dataset documentée dans `data/README.md` | ✅ Terminé |
| `data/download_chexpert.py` : script Kaggle API, mapping labels, sélection équilibrée | ✅ Terminé |
| Authentification Kaggle CLI v2.2.2 configurée | ✅ Terminé |
| Téléchargement CSV labels (223 414 radios, ~20 Mo) | ✅ Terminé |
| Téléchargement images sample (1 002 vraies radios, ~52 Mo) | ✅ Terminé |
| Génération `chexpert_cases.csv` — 30 cas (10/classe, seed=42) | ✅ Terminé |

**Blocage restant :** clé API Gemini à renseigner dans `.env` → relancer l'évaluation.
