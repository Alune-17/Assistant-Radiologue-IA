# Assistant radiologue virtuel responsable

> **Auteur :** Badr Tajini  
> **Solution Delivery - Filière Data**  
> **École :** EFREI  
> **Année académique :** 2025-2026

## Contexte

Prototype pédagogique d'IA médicale multimodale pour apprendre à construire une chaîne **prudente, traçable et évaluée** autour d'une radiographie thoracique frontale.

---

>  **Position non clinique.** Ce dépôt n'est pas un dispositif médical. Il ne doit jamais être utilisé pour diagnostiquer, trier ou orienter un patient. Toute sortie doit rester un résultat expérimental, vérifié par un professionnel qualifié.

---

## Contrat du projet

| Élément | Cadrage |
|---|---|
| Entrée | Une radiographie thoracique frontale |
| Sorties | `normal`, `suspected_opacity`, `uncertain` |
| Preuve minimale | JSON valide, warning, logs, métriques, cas d'erreur |
| Données | Synthétiques ou publiques, autorisées et dé-identifiées |
| Finalité | Prototype éducatif de data/IA, pas aide au diagnostic réelle |

Le bon rendu ne cherche pas à impressionner par un modèle spectaculaire. Il démontre une méthode : périmètre limité, baseline reproductible, garde-fous, évaluation, analyse d'erreurs et limites explicites.

## Prérequis

Pour commencer à travailler sur ce projet à partir de zéro, vous aurez besoin de :
1. **Python 3.10+** installé sur votre machine.
2. **Clé API Groq (Gratuite)** : Utilisée pour exécuter le modèle de vision (Llama 4 Scout). Obtenez-la sur [console.groq.com](https://console.groq.com/keys).
3. **Clé API Kaggle (Gratuite)** : Nécessaire pour télécharger les vraies radios du dataset CheXpert. Obtenez-la sur [kaggle.com](https://www.kaggle.com/settings) (créez un token API `kaggle.json`).

## Démarrage rapide étape par étape

### 1. Installation de l'environnement
```bash
python -m venv .venv
# Sur Windows :
.venv\Scripts\activate
# Sur Mac/Linux :
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration des clés API
Dupliquez le fichier `.env.example` pour créer un fichier `.env` à la racine du projet :
```bash
cp .env.example .env
```
Ouvrez le fichier `.env` et collez votre clé API Groq à l'intérieur. **Ne commitez jamais le fichier `.env` sur Git !**

### 3. Téléchargement des données réelles (CheXpert)
Assurez-vous d'avoir configuré vos credentials Kaggle (voir `data/README.md`), puis lancez le script :
```bash
python data/download_chexpert.py
```
*Cela téléchargera uniquement un sous-ensemble pertinent (30 cas, ~50 Mo) sans polluer votre disque dur avec les 11 Go complets.*

### 4. Lancement de l'interface visuelle (Streamlit)
Pour tester l'assistant radiologue dans votre navigateur avec vos propres images ou celles téléchargées :
```bash
streamlit run app/streamlit_app.py
```

### 5. Évaluation du modèle
Pour lancer une évaluation sur les 30 radios téléchargées et générer des métriques complètes :
```bash
python eval/run_evaluation.py --mode groq-baseline --cases-csv data/chexpert_subset/chexpert_cases.csv
```

## Smoke test du dépôt

Avant une soutenance, un push ou une livraison, lancer le contrôle court :

```bash
pip install -r requirements-test.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q src api app eval finetuning tests
python eval/run_evaluation.py --mode toy \
  --out-dir /tmp/assistant-radio-eval \
  --db-path /tmp/assistant-radio-evidence.sqlite
```

Ce smoke test vérifie la structure du dépôt, le contrat du dataset synthétique, le schéma de sortie, les garde-fous, l'API de démonstration, la compilation Python et l'évaluation jouet.

## API de démonstration

```bash
uvicorn api.main:app --reload
```

Exemple :

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/sample_images/CXR_SYN_002_suspected_opacity.png"
```

La réponse doit contenir une classe, une confiance, des observations visuelles, une justification, des limites et l'avertissement non clinique.

## Organisation

```text
assistant-radiologue-virtuel/
├── README.md
├── docs/          # appel d'offre, architecture, éthique, évaluation
├── data/          # cas synthétiques et images jouet
├── prompts/       # prompt baseline, prompt amélioré, schéma JSON
├── src/           # inférence jouet, garde-fous, métriques, SQLite
├── api/           # FastAPI
├── app/           # Streamlit / Gradio
├── eval/          # évaluation, sorties CSV/JSON, registre d'erreurs
├── tests/         # smoke tests et contrat minimal
├── notebooks/     # notebooks de démarrage
└── finetuning/    # stubs expérimentaux, non obligatoires
```

## Livrables attendus

| Niveau | Attendu |
|---|---|
| **MUST** | Baseline reproductible, sortie JSON valide, warning obligatoire, logs, métriques, mini-rapport |
| **SHOULD** | Prompt amélioré, règle d'incertitude, comparaison baseline/amélioration, analyse d'erreurs |
| **COULD** | LoRA expérimental, MedGemma/PEFT, localisation visuelle, ablations de prompts |

## Références techniques

Les pistes avancées doivent rester expérimentales, traçables et justifiées. L'architecture repose actuellement sur l'API **Groq** avec le modèle **Llama 4 Scout** (rapide et gratuit), mais peut être étendue à d'autres solutions.

| Ressource | Usage possible | Référence à citer |
|---|---|---|
| Llama 4 Scout / Groq | Inférence Vision multimodale (Baseline actuelle) | [Groq API](https://console.groq.com/docs/models) |
| Unsloth - Llama Vision | Fine-tuning LoRA expérimental | [Unsloth AI](https://unsloth.ai/) |
| MIMIC-CXR / MIMIC-CXR-JPG | Jeu de données de radiographies thoraciques, accès contrôlé et non redistribuable | [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/), [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) |
| CheXpert | Jeu de données public de radiographies thoraciques avec rapports associés | [Stanford AIMI - CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) |

## Points de vigilance

- Ne pas inventer d'information clinique absente de l'image.
- Ne pas supprimer la classe `uncertain`; elle est un garde-fou, pas un échec.
- Ne pas afficher uniquement des réussites en soutenance.
- Ne jamais commiter de données patient réelles, identifiantes ou ambiguës.
- Ne pas présenter le prototype comme validé médicalement.

## Licence et sources externes

Le code pédagogique du dépôt est publié sous licence MIT. **Les datasets externes, modèles et bibliothèques utilisés conservent leurs licences propres** : les étudiants doivent vérifier et documenter les droits d'usage avant toute expérimentation.

Exigence minimale : indiquer dans le rapport la source, la version, la licence ou les conditions d'accès, les restrictions de redistribution, les traitements d'anonymisation et les limites d'interprétation. Aucun fichier patient réel, même pseudonymisé, ne doit être ajouté au dépôt sans autorisation explicite et traçable.
