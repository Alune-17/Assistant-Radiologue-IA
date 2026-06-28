from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd

# Permet d'importer le dossier 'src' même si le script est lancé depuis un autre dossier
sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.inference import toy_predict, groq_predict, groq_predict_baseline, groq_predict_improved
from src.guardrails import apply_safety_guardrails
from src.database import fetch_error_counts, fetch_recent_runs, insert_run

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("ARVIRX_DB_PATH", ROOT / "medical_ai_evidence.sqlite"))
OUTPUT_DIR = ROOT / "eval" / "outputs"

# ── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistant Radiologue Virtuel — EFREI",
    page_icon="🩺",
    layout="wide",
)

st.title("Assistant Radiologue Virtuel — prototype pédagogique")
st.warning(
    "⚠️ **Prototype pédagogique. Non destiné au diagnostic. "
    "Validation par un professionnel qualifié requise.**"
)


def render_analysis_tab() -> None:
    # ── Sélecteur de modèle ───────────────────────────────────────────────────────
    model_options = {
        "🎲 Toy Baseline (local, sans API)": ("toy", "baseline"),
        "🤖 Groq Llama 4 Scout — Prompt Baseline": ("groq", "baseline"),
        "🤖 Groq Llama 4 Scout — Prompt Amélioré": ("groq", "improved"),
        "🌐 Groq Llama 4 Scout — Prompt Français": ("groq", "fr"),
    }

    mode_label = st.selectbox("Modèle d'analyse", list(model_options.keys()))
    model_type, prompt_type = model_options[mode_label]

    uploaded = st.file_uploader(
        "Déposer une radiographie thoracique frontale (PNG, JPG, JPEG, BMP)",
        type=["png", "jpg", "jpeg", "bmp"],
    )

    if not uploaded:
        st.info(
            "💡 Utilisez les images synthétiques dans `data/synthetic/images/` pour tester le flux.\n\n"
            "Exemple : `CXR_SYN_001_normal.png`, `CXR_SYN_002_suspected_opacity.png`, etc."
        )
        return

    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Image uploadée")
        st.image(Image.open(tmp_path), caption=uploaded.name, use_container_width=True)

    with col2:
        st.subheader("Résultats de l'analyse")

        with st.spinner("Analyse en cours..."):
            if model_type == "toy":
                pred = apply_safety_guardrails(toy_predict(tmp_path, mode="baseline"))
            elif prompt_type == "baseline":
                pred = apply_safety_guardrails(groq_predict_baseline(tmp_path))
            elif prompt_type == "improved":
                pred = apply_safety_guardrails(groq_predict_improved(tmp_path))
            else:  # fr
                pred = apply_safety_guardrails(groq_predict(tmp_path))

        try:
            run_id = insert_run(DB_PATH, f"streamlit:{Path(uploaded.name).stem}", str(tmp_path), pred)
            st.caption(f"Run journalisé dans SQLite : #{run_id}")
        except Exception as exc:
            st.caption(f"Log SQLite non enregistré : {type(exc).__name__}")

        col_class, col_conf, col_quality = st.columns(3)
        with col_class:
            st.metric("Classe prédite", pred["predicted_class"])
        with col_conf:
            st.metric("Confiance", f"{pred['confidence']:.0%}")
        with col_quality:
            st.metric("Qualité image", pred.get("image_quality", "—"))

        st.divider()

        st.markdown("**📋 Observations visuelles**")
        for obs in pred.get("visual_evidence", []):
            st.markdown(f"- {obs}")

        st.markdown("**🔍 Justification**")
        st.info(pred.get("justification", ""))

        st.markdown("**⚠️ Limites**")
        for lim in pred.get("limitations", []):
            st.markdown(f"- {lim}")

        st.markdown(f"**🕐 Latence** : `{pred.get('latency_ms', 0)} ms`")
        st.markdown(f"**🤖 Modèle** : `{pred.get('model_name', '—')}`")
        st.markdown(f"**📝 Prompt** : `{pred.get('prompt_version', '—')}`")

        st.divider()
        st.markdown("**📦 JSON complet**")
        st.json(pred)


def render_dashboard_tab() -> None:
    st.subheader("Dashboard de métriques et de traçabilité")
    st.caption(
        "Ce dashboard exploite les sorties `eval/outputs/` et les runs SQLite générés par l'API, "
        "Streamlit ou `eval/run_evaluation.py`."
    )

    summary_path = OUTPUT_DIR / "before_after_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        st.markdown("### Comparaison baseline vs amélioration")
        st.dataframe(summary_df, use_container_width=True)

        metric_cols = st.columns(4)
        latest = summary_df.iloc[-1]
        metric_cols[0].metric("Accuracy", f"{latest.get('accuracy', 0):.1%}")
        metric_cols[1].metric("Macro-F1", f"{latest.get('macro_f1', 0):.1%}")
        metric_cols[2].metric("Taux incertain", f"{latest.get('uncertain_rate', 0):.1%}")
        metric_cols[3].metric("JSON valide", f"{latest.get('json_valid_rate', 0):.1%}")

        chart_columns = [
            col
            for col in ["accuracy", "macro_f1", "sensitivity", "specificity", "uncertain_rate"]
            if col in summary_df.columns
        ]
        chart_df = summary_df.set_index("mode")[chart_columns]
        if not chart_df.empty:
            st.bar_chart(chart_df)
    else:
        st.info("Aucune synthèse d'évaluation trouvée. Lancez : `python eval/run_evaluation.py --mode toy`.")

    error_files = sorted(OUTPUT_DIR.glob("*_error_register.csv"))
    if error_files:
        st.markdown("### Registre d'erreurs")
        selected = st.selectbox("Fichier de registre", [p.name for p in error_files])
        error_df = pd.read_csv(OUTPUT_DIR / selected)
        st.dataframe(error_df, use_container_width=True)
        if "error_type" in error_df.columns:
            st.bar_chart(error_df["error_type"].value_counts())


    case_review_path = OUTPUT_DIR / "case_review_template.csv"
    if case_review_path.exists():
        st.markdown("### Template des cas à commenter")
        case_review_df = pd.read_csv(case_review_path)
        st.caption(
            "Ce CSV pré-sélectionne jusqu'à 30 cas pour le rapport critique. "
            "Complétez manuellement les colonnes d'observation humaine avant le rendu."
        )
        st.dataframe(case_review_df, use_container_width=True)
        st.download_button(
            "Télécharger le template de revue de cas",
            data=case_review_path.read_text(encoding="utf-8"),
            file_name="case_review_template.csv",
            mime="text/csv",
        )

    report_path = OUTPUT_DIR / "evaluation_report.md"
    if report_path.exists():
        st.markdown("### Rapport automatique")
        report_text = report_path.read_text(encoding="utf-8")
        with st.expander("Voir le rapport Markdown généré", expanded=False):
            st.markdown(report_text)
        st.download_button(
            "Télécharger le rapport Markdown",
            data=report_text,
            file_name="rapport_evaluation_automatique.md",
            mime="text/markdown",
        )

    recent_runs = fetch_recent_runs(DB_PATH, limit=50)
    st.markdown("### Derniers runs SQLite")
    if recent_runs:
        runs_df = pd.DataFrame(recent_runs)
        st.dataframe(runs_df, use_container_width=True)
        if "predicted_class" in runs_df.columns:
            st.bar_chart(runs_df["predicted_class"].value_counts())

        error_counts = fetch_error_counts(DB_PATH)
        if error_counts:
            st.markdown("### Types d'erreurs annotés dans SQLite")
            st.bar_chart(pd.Series(error_counts))
    else:
        st.info("Aucun run SQLite encore enregistré.")


tab_analysis, tab_dashboard = st.tabs(["Analyse IA", "Dashboard"])
with tab_analysis:
    render_analysis_tab()
with tab_dashboard:
    render_dashboard_tab()
