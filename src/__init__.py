"""Package source de l'assistant radiologue virtuel.

Modules :
    inference       — Fonctions d'inférence (toy déterministe + API Groq / Llama 4 Scout)
    guardrails      — Validation JSON, warning obligatoire, fallback incertitude
    preprocessing   — Chargement d'image et analyse de qualité (luminosité, contraste, résolution)
    metrics         — Accuracy, macro-F1, sensibilité, spécificité, latence médiane
    database        — Connecteur SQLite pour journaliser les runs d'évaluation
"""
