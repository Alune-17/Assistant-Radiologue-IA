# Changelog — Assistant Radiologue Virtuel Responsable

> Format : [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/)  
> Versioning : [Semantic Versioning](https://semver.org/lang/fr/)  
> Chaque section correspond à un **push** ou un **sprint de travail**.

---

## [Unreleased] — À venir

### Bloqué en attente
- Évaluation complète `groq-baseline` vs `groq-improved` sur 30 radios CheXpert
- Notebook `01_baseline_vlm.ipynb` — documentation résultats
- Registre d'erreurs commenté (Phase 4)
- Dashboard de métriques Streamlit (Phase 5)
- Rapport final (Phase 6)

---

## [0.3.0] — 2026-06-22 — Migration Groq (Llama 4 Scout) et correctifs

### Ajouté
- Migration totale vers l'API **Groq** pour pallier les erreurs de quota/billing de Google Gemini en Europe.
- Intégration du modèle **Llama 4 Scout** (`meta-llama/llama-4-scout-17b-16e-instruct`) en remplacement de Llama 3.2 Vision (décommissionné par Groq).
- Parsing robuste du JSON via `re.search` (expressions régulières) pour contourner les erreurs strictes de l'API Groq sur les modèles de Vision qui incluent des balises Markdown.

### Modifié
- `src/inference.py` — Remplacement complet du client Gemini par le SDK `groq`.
- `app/streamlit_app.py` — Mise à jour des labels de l'interface graphique pour refléter les modèles Groq.
- `eval/run_evaluation.py` — Remplacement des arguments `--mode gemini-*` par `--mode groq-*`.
- `api/main.py` — Mise à jour des modèles dans le dictionnaire de routage de l'API.
- `requirements.txt` — Suppression de `google-genai`, ajout de `groq`.
- `.env.example` — Remplacement de `GOOGLE_API_KEY` par `GROQ_API_KEY`.

---

## [0.2.0] — 2026-06-22 — Dataset réel CheXpert + Infrastructure Kaggle

### Ajouté
- `data/download_chexpert.py` — script de téléchargement optimisé via Kaggle API
  - Télécharge uniquement le CSV de labels (~20 Mo) et un sample de 1 002 images (~52 Mo)
  - Évite le téléchargement des 11 Go du dataset complet
  - Mapping automatique des labels CheXpert → 3 classes du projet :
    - `No Finding = 1.0` → `normal`
    - `Consolidation / Pleural Effusion / Edema / Pneumonia = 1.0` → `suspected_opacity`
    - Label `-1.0` (incertain) → `uncertain`
  - Sélection équilibrée et reproductible (graine seed=42)
  - Option `--skip-download` pour réutiliser le cache
- `data/README.md` — documentation complète des deux datasets :
  - Dataset synthétique (tests CI) vs CheXpert (évaluation réelle)
  - Prérequis Kaggle, instructions d'installation, citation académique
- `.gitignore` — exclusion de `data/chexpert_subset/` et `data/_kaggle_download/`

### Dataset téléchargé (non commité)
- `data/chexpert_subset/chexpert_cases.csv` — 30 cas réels :
  - 10 `normal` (No Finding confirmé)
  - 10 `suspected_opacity` (Consolidation / Pleural Effusion / Edema / Pneumonia)
  - 10 `uncertain` (labels -1.0 dans CheXpert)
- Source : CheXpert v1.0, Stanford AIMI — Irvin et al. (2019), AAAI

### Modifié
- `eval/run_evaluation.py` — nouveau paramètre `--cases-csv` :
  - Défaut : `data/synthetic_cases.csv` (comportement précédent inchangé)
  - Option : `data/chexpert_subset/chexpert_cases.csv` (vraies radios)
  - Suffixe automatique dans les noms de fichiers de sortie

---

## [0.1.0] — 2026-06-22 — Migration SDK Gemini + Baseline API + Nettoyage repo

### Ajouté
- `.env.example` — template de configuration avec `GOOGLE_API_KEY` (clé non commitée)
- `.env` — fichier local avec placeholder (exclu par `.gitignore`)
- `src/inference.py` — nouvelles fonctions :
  - `gemini_predict_baseline()` — prompt minimaliste, JSON structuré
  - `gemini_predict_improved()` — prompt renforcé avec raisonnement étape par étape
  - `_gemini_call()` — fonction interne partagée, gestion d'erreur, fallback `uncertain`
  - Garde-fou : confiance < 0.60 → classe forcée à `uncertain`
  - Client singleton `_client` initialisé une fois au démarrage

### Modifié
- `src/inference.py` — migration SDK :
  - `google.generativeai` (déprécié) → `google.genai` v2.9.0
  - `genai.configure()` → `genai.Client(api_key=...)`
  - `GenerativeModel.generate_content()` → `client.models.generate_content()`
  - `GenerationConfig` → `types.GenerateContentConfig`
  - Modèle : `gemini-3.5-flash` (inexistant) → **`gemini-2.0-flash`**
- `src/preprocessing.py` — analyse de qualité image réelle :
  - Avant : flag basé sur le nom du fichier (non fonctionnel)
  - Après : luminosité moyenne (PIL), contraste (std numpy), résolution minimale, ratio d'aspect
- `src/metrics.py` — métriques cliniques ajoutées :
  - `sensitivity()` — rappel sur la classe `suspected_opacity`
  - `specificity()` — vrai négatif / (vrai négatif + faux positif)
  - `median_latency_ms()` — latence médiane des appels API
  - `summarize_metrics()` enrichi avec ces 3 nouvelles métriques
- `eval/run_evaluation.py` — améliorations :
  - Modes `gemini-baseline`, `gemini-improved`, `all-gemini` ajoutés
  - Sortie `stdout` = JSON pur (pour `json.loads()` dans les tests)
  - Sortie `stderr` = logs de progression (fix UnicodeEncodeError Windows cp1252)
- `api/main.py` — paramètre `?model=toy|gemini-baseline|gemini-improved`
- `app/streamlit_app.py` — interface 4 modes, 3 colonnes, spinner, affichage latence
- `requirements.txt` — `google-generativeai>=0.8.2` → `google-genai>=1.0.0`

### Supprimé
- `medical_ai_evidence.sqlite` — base SQLite commitée (interdit par les smoke tests)
- `eval/outputs/` — répertoire de résultats commité (interdit par les smoke tests)

### Corrigé
- `eval/run_evaluation.py` — `UnicodeEncodeError` sous Windows (cp1252) :
  - Remplacement des emojis (`▶`, `✅`, `❌`, `📊`) par des équivalents ASCII
  - Séparation stdout (JSON) / stderr (logs) pour compatibilité avec `subprocess.run(..., capture_output=True)`

### Tests
- `tests/test_repository_smoke.py` — **8/8 passent** ✅
  - `test_repository_student_contract_is_present`
  - `test_synthetic_dataset_contract_is_valid`
  - `test_prediction_schema_warning_and_guardrails`
  - `test_python_source_tree_compiles`
  - `test_invalid_model_output_falls_back_to_uncertain`
  - `test_metrics_and_api_health_contract`
  - `test_api_predict_preserves_uploaded_case_signal`
  - `test_evaluation_command_runs_and_preserves_warning_contract`

---

## [0.0.1] — Avant session 1 — État initial du repository

### État initial (hérité)
- Architecture de base : `src/`, `api/`, `app/`, `eval/`, `data/`, `tests/`, `docs/`
- `src/inference.py` — `toy_predict()` déterministe + stub `gemini_predict()` (SDK déprécié)
- `src/guardrails.py` — validation JSON, warning obligatoire, fallback `uncertain`
- `src/database.py` — SQLite `init_db()`, `insert_run()`
- `api/main.py` — FastAPI `/predict` (mode `improved` uniquement)
- `app/streamlit_app.py` — interface upload + résultat JSON
- `data/synthetic_cases.csv` — 30 cas synthétiques (20 smoke + 10 final)
- `data/sample_images/` — 30 images PNG simulées (gribouillis, non médicales)
- `tests/test_repository_smoke.py` — 7/8 tests passaient (1 échec UnicodeEncodeError)

### Problèmes connus
- `medical_ai_evidence.sqlite` et `eval/outputs/` committés malgré `.gitignore`
- SDK `google.generativeai` déprécié, modèle `gemini-3.5-flash` inexistant
- `basic_quality_flag()` basé sur le nom de fichier, non fonctionnel
- Métriques identiques entre baseline et improved (toutes à 1.0)
- Emoji Unicode dans logs → `UnicodeEncodeError` sous Windows
