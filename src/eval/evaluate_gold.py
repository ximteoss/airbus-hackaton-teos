from __future__ import annotations

import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support


def _norm_clause(clause_id: str) -> str:
    return " ".join(clause_id.upper().replace("PART-IS", "").split())


def evaluate_against_gold(pred_report_path: str | Path, gold_path: str | Path) -> dict:
    """
    Formato esperado de gold_path (JSON):
    [
      {"req_id": "REQ-021", "verdict": "No cumple", "applicable_clauses": ["IS.I.OR.245"]},
      ...
    ]
    """
    pred_data = json.loads(Path(pred_report_path).read_text(encoding="utf-8"))
    gold_items = json.loads(Path(gold_path).read_text(encoding="utf-8"))

    pred_by_id = {f["req_id"]: f for f in pred_data["findings"]}

    y_true_non_compliant: list[int] = []
    y_pred_non_compliant: list[int] = []

    trace_hits = 0
    trace_total = 0
    errors: list[dict] = []

    for g in gold_items:
        req_id = g["req_id"]
        pred = pred_by_id.get(req_id)

        true_is_nc = 1 if g["verdict"].lower() == "no cumple" else 0
        pred_is_nc = 1 if (pred and pred["verdict"].lower() == "no cumple") else 0

        y_true_non_compliant.append(true_is_nc)
        y_pred_non_compliant.append(pred_is_nc)

        if true_is_nc != pred_is_nc:
            errors.append({
                "req_id": req_id,
                "expected_verdict": g["verdict"],
                "predicted_verdict": pred["verdict"] if pred else "MISSING",
            })

        gold_clauses = {_norm_clause(c) for c in g.get("applicable_clauses", [])}
        if gold_clauses:
            trace_total += 1
            pred_clauses = {_norm_clause(c) for c in (pred.get("applicable_clauses", []) if pred else [])}
            if gold_clauses.intersection(pred_clauses):
                trace_hits += 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_non_compliant,
        y_pred_non_compliant,
        average="binary",
        zero_division=0,
    )
    traceability_acc = (trace_hits / trace_total) if trace_total else 0.0

    return {
        "non_compliance_precision": round(float(precision), 4),
        "non_compliance_recall": round(float(recall), 4),
        "non_compliance_f1": round(float(f1), 4),
        "meets_f1_target_0_80": f1 >= 0.80,
        "traceability_accuracy": round(traceability_acc, 4),
        "meets_traceability_target_0_90": traceability_acc >= 0.90,
        "error_cases": errors,
    }