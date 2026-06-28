from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
DEFAULT_DB_PATH = Path("medical_ai_evidence.sqlite")


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    if db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()


def insert_run(db_path: str | Path, case_id: str, image_path: str, prediction: dict[str, Any]) -> int:
    """Insère un run d'inférence et retourne son identifiant SQLite."""
    init_db(db_path)
    conn = connect(db_path)
    cursor = conn.execute(
        """
        INSERT INTO runs(case_id, image_path, model_name, prompt_version, prediction_json, predicted_class, confidence, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            image_path,
            prediction.get("model_name"),
            prediction.get("prompt_version"),
            json.dumps(prediction, ensure_ascii=False),
            prediction.get("predicted_class"),
            float(prediction.get("confidence", 0.0) or 0.0),
            int(prediction.get("latency_ms", 0) or 0),
        ),
    )
    conn.commit()
    run_id = int(cursor.lastrowid)
    conn.close()
    return run_id


def insert_evaluation(
    db_path: str | Path,
    run_id: int,
    ground_truth_label: str,
    correct: bool,
    error_type: str,
    reviewer_comment: str = "",
) -> None:
    """Journalise l'évaluation d'un run dans la table evaluations."""
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        """
        INSERT INTO evaluations(run_id, ground_truth_label, correct, error_type, reviewer_comment)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, ground_truth_label, int(correct), error_type, reviewer_comment),
    )
    conn.commit()
    conn.close()


def fetch_recent_runs(db_path: str | Path = DEFAULT_DB_PATH, limit: int = 50) -> list[dict[str, Any]]:
    """Retourne les derniers runs pour alimenter le dashboard Streamlit."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    conn = connect(db_path)
    rows = conn.execute(
        """
        SELECT id, case_id, image_path, model_name, prompt_version, predicted_class, confidence, latency_ms, created_at
        FROM runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_error_counts(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, int]:
    """Compte les types d'erreurs déjà annotés dans SQLite."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {}
    conn = connect(db_path)
    rows = conn.execute(
        """
        SELECT error_type, COUNT(*) AS n
        FROM evaluations
        GROUP BY error_type
        ORDER BY n DESC
        """
    ).fetchall()
    conn.close()
    return {str(row["error_type"]): int(row["n"]) for row in rows}
