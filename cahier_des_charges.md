# Suivi personnel — Assistant Radiologue Virtuel

> ⚠️ **Ce fichier est dans `.gitignore` — il ne sera pas poussé sur GitHub.**  
> C'est un document de travail personnel pour suivre l'avancement du projet.  
> Les livrables officiels sont dans `docs/`, `README.md` et le code source.

> **Dernière mise à jour :** 27 juin 2026

---

## État d'avancement global

| Phase | Description | Statut |
|---|---|---|
| Phase 1 | Nettoyage et mise en conformité | ✅ Terminée |
| Phase 2 | Baseline fonctionnelle avec vrai modèle | 🔄 ~60% — bloquée sur clé API |
| Phase 3 | Comparaison baseline vs prompt amélioré | ❌ À faire |
| Phase 4 | Évaluation et registre d'erreurs | ❌ À faire |
| Phase 5 | Dashboard de métriques | ❌ À faire |
| Phase 6 | Rapport final et documentation | ❌ À faire |
| Phase 7 | Fine-tuning optionnel (COULD) | ❌ Optionnel |

---

## Ce qui est fait ✅

### Infrastructure technique
- [x] Migration SDK : Gemini → Groq (Llama 4 Scout)
- [x] `src/inference.py` : `toy_predict()`, `groq_predict_baseline()`, `groq_predict_improved()`
- [x] `src/preprocessing.py` : analyse qualité pixel réelle (luminosité, contraste, résolution)
- [x] `src/metrics.py` : sensibilité, spécificité, latence médiane
- [x] `src/guardrails.py` : validation JSON, warning, fallback uncertain
- [x] `api/main.py` : `GET /` + `POST /predict?model=toy|groq-baseline|groq-improved`
- [x] `app/streamlit_app.py` : 4 modes, UI enrichie (3 colonnes, spinner, latence)
- [x] Smoke tests : **8/8 passent**
- [x] CI GitHub Actions fonctionnel

### Données
- [x] 30 images synthétiques + `cases.csv` (CI/CD)
- [x] Script `data/download_chexpert.py` pour télécharger CheXpert via Kaggle
- [x] Sous-ensemble CheXpert téléchargé (21 radios réelles)
- [x] `data/README.md` documenté avec licences et sources

### Nettoyage du dépôt
- [x] Suppression `medical_ai_evidence.sqlite` et `eval/outputs/` du repo
- [x] `.gitignore` enrichi
- [x] `.env.example` avec placeholder (clé non commitée)
- [x] Réorganisation complète de l'arborescence (v0.4.0)
- [x] Séparation `requirements.txt` (core) / `requirements-gpu.txt` (optionnel)

### Évaluation réelle (avec clé API Groq)
- [x] Évaluation baseline sur 30 radios CheXpert → **accuracy 33%, sensibilité 100%, spécificité 0%**
- [x] Évaluation improved sur 30 radios CheXpert → **accuracy 33%, 100% uncertain**
- [x] Rapport d'évaluation rédigé → `docs/rapport_evaluation.md`

---

## Ce qu'il reste à faire 🔴

### Priorité HAUTE — Obligatoire pour la soutenance

#### 1. Rédiger le rapport final (M8)
- [ ] Document couvrant : dataset, prompts, métriques, limites, risques
- [ ] Documenter les choix techniques
- [ ] Résultats réels de l'évaluation avec analyse
- **Où :** Créer `docs/rapport_final.md` ou un PDF

#### 2. Compléter le registre d'erreurs (S7) — 20 à 30 cas commentés
- [ ] Analyser 20-30 cas un par un avec les résultats Groq
- [ ] Classifier chaque erreur avec la taxonomie (FN, FP, UA, JF, HT)
- [ ] Commenter chaque cas (pourquoi l'erreur, gravité, action corrective)
- **Où :** Compléter `eval/error_register_template.csv` (actuellement 3 exemples seulement)

#### 3. Créer un dashboard de métriques (S4)
- [ ] Page Streamlit dédiée ou section dans l'app existante
- [ ] Afficher : accuracy, macro-F1, sensibilité, spécificité, matrice de confusion
- [ ] Taux JSON valide, taux warning, latence médiane
- [ ] Graphiques comparatifs baseline vs improved

#### 4. Documenter les notebooks
- [ ] `01_baseline_vlm.ipynb` — exécuter et documenter la baseline avec résultats Groq
- [ ] `02_prompt_comparison.ipynb` — comparer baseline vs improved avec les métriques réelles
- [ ] `03_optional_finetuning_lora.ipynb` — documenter la tentative LoRA (optionnel)

### Priorité MOYENNE — Améliorations attendues

#### 5. Compléter les tables SQLite (S5)
- [ ] Peupler la table `cases` depuis le CSV
- [ ] Stocker les prompts dans la table `prompts`
- [ ] Remplir `evaluations` avec les résultats du registre d'erreurs
- **Fichier :** `sql/schema.sql` définit les 4 tables, seule `runs` est alimentée

#### 6. Préparer la soutenance
- [ ] Commandes exécutables prêtes à démontrer
- [ ] Preuves visuelles (screenshots, JSON de sortie)
- [ ] Montrer les erreurs (FP, FN, incertitudes) — pas seulement les réussites
- [ ] Défense des choix techniques et des limites

### Priorité BASSE — Optionnel (Phase COULD)

- [ ] Fine-tuning LoRA Gemma 4 / Unsloth (`finetuning/gemma4_unsloth_lora_stub.py`)
- [ ] MedGemma via PEFT/QLoRA (`finetuning/medgemma_peft_qlora_stub.py`)
- [ ] Localisation visuelle (heatmap / attention)
- [ ] Ablation systématique de prompts

---

## Barème de notation (rappel)

| Critère | Poids | Mon état |
|---|---|---|
| Périmètre + dataset | 15% | ✅ OK — synthétique + CheXpert documentés |
| Baseline fonctionnelle | 15% | ✅ OK — Groq fonctionne, résultats réels obtenus |
| Amélioration mesurée | 20% | ⚠️ Résultats obtenus mais à mieux documenter |
| Intégration application | 15% | ✅ OK — Streamlit + API + Gradio |
| Évaluation + erreurs | 20% | 🔴 Registre d'erreurs à compléter (3/30 cas) |
| Éthique + limites | 10% | ✅ OK — doc éthique + warning partout |
| Oral professionnel | 5% | 🔴 Préparation à faire |

---

## Résultats clés à retenir pour la soutenance

### Le paradoxe de la prudence (résultat principal)

- **Baseline** : Le modèle diagnostique TOUT comme `suspected_opacity` → sensibilité 100%, spécificité 0%
- **Improved** : Le modèle répond TOUT comme `uncertain` → sensibilité 0%, spécificité 0%
- **Conclusion** : Un VLM généraliste ne sait pas lire une radio. Le prompt engineering ne remplace pas le savoir médical.

C'est un **résultat scientifiquement intéressant** à défendre en soutenance, pas un échec.

---

## Fichiers clés — aide-mémoire

| Pour... | Ouvrir... |
|---|---|
| Comprendre le projet | `README.md` |
| Voir l'appel d'offre original | `docs/appel_offre.pdf` |
| Voir l'architecture | `docs/architecture.md` |
| Voir les résultats d'évaluation | `docs/rapport_evaluation.md` |
| Voir les contraintes éthiques | `docs/ethique_et_limites.md` |
| Voir le protocole d'évaluation | `docs/evaluation_protocol.md` |
| Lancer l'interface | `streamlit run app/streamlit_app.py` |
| Lancer les tests | `python -m pytest -q` |
| Lancer l'évaluation | `python eval/run_evaluation.py --mode toy` |
