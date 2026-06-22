from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import streamlit as st

# Permet d'importer le dossier 'src' même si le script est lancé depuis un autre dossier
sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.inference import toy_predict, groq_predict, groq_predict_baseline, groq_predict_improved
from src.guardrails import apply_safety_guardrails

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

# ── Sélecteur de modèle ───────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "🎲 Toy Baseline (local, sans API)": ("toy", "baseline"),
    "🤖 Groq Llama 4 Scout — Prompt Baseline": ("groq", "baseline"),
    "🤖 Groq Llama 4 Scout — Prompt Amélioré": ("groq", "improved"),
    "🌐 Groq Llama 4 Scout — Prompt Français": ("groq", "fr"),
}

mode_label = st.selectbox("Modèle d'analyse", list(MODEL_OPTIONS.keys()))
model_type, prompt_type = MODEL_OPTIONS[mode_label]

# ── Upload de l'image ─────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Déposer une radiographie thoracique frontale (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
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

        # Affichage des métriques principales
        col_class, col_conf, col_quality = st.columns(3)
        with col_class:
            st.metric("Classe prédite", pred["predicted_class"])
        with col_conf:
            st.metric("Confiance", f"{pred['confidence']:.0%}")
        with col_quality:
            st.metric("Qualité image", pred.get("image_quality", "—"))

        st.divider()

        # Observations et justification
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

else:
    st.info(
        "💡 Utilisez les images synthétiques dans `data/sample_images/` pour tester le flux.\n\n"
        "Exemple : `CXR_SYN_001_normal.png`, `CXR_SYN_002_suspected_opacity.png`, etc."
    )
