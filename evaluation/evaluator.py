"""
Evaluator — Metrics computation cho ASTRA.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Optional

import numpy as np

from config.config import RELATIONS, AXIS_MAP


def evaluate_predictions(predictions: list[dict]) -> dict:
    n = len(predictions)
    if n == 0:
        return {}

    correct = sum(1 for p in predictions if _is_correct(p))
    overall_acc = correct / n

    # Per-relation
    per_relation = {}
    for rel in RELATIONS:
        rel_preds = [p for p in predictions if _get_relation(p) == rel]
        if rel_preds:
            rc = sum(1 for p in rel_preds if _is_correct(p))
            per_relation[rel] = {"acc": rc / len(rel_preds), "count": len(rel_preds), "correct": rc}

    # Per-axis
    per_axis = {}
    for axis, rels in AXIS_MAP.items():
        axis_preds = [p for p in predictions if _get_relation(p) in rels]
        if axis_preds:
            ac = sum(1 for p in axis_preds if _is_correct(p))
            per_axis[axis] = ac / len(axis_preds)

    # Macro P/R/F1
    ps, rs, fs = [], [], []
    for rel in RELATIONS:
        if rel in per_relation and per_relation[rel]["count"] > 0:
            p, r, f = _prf1(predictions, rel)
            ps.append(p); rs.append(r); fs.append(f)

    mp = np.mean(ps) if ps else 0.0
    mr = np.mean(rs) if rs else 0.0
    mf = np.mean(fs) if fs else 0.0

    bstf = _bstf_rate(predictions)
    pcr = _pcr(predictions)

    return {
        "overall_acc": overall_acc,
        "per_relation": per_relation,
        "per_axis": per_axis,
        "macro_precision": mp, "macro_recall": mr, "macro_f1": mf,
        "depth_acc": per_axis.get("z", 0.0),
        "horiz_acc": per_axis.get("x", 0.0),
        "vert_acc": per_axis.get("y", 0.0),
        "pcr": pcr, "bstf_rate": bstf,
        "total": n, "correct": correct,
    }


def _is_correct(pred: dict) -> bool:
    pv = pred.get("predicted", "")
    av = pred.get("answer", "")
    return bool(pv and av and pv.lower().strip() == av.lower().strip())


def _get_relation(pred: dict) -> Optional[str]:
    ans = pred.get("answer", "")
    opts = pred.get("options", [])
    for opt in opts:
        if opt.lower() in ans.lower() or ans.lower() in opt.lower():
            return opt
    return ans


def _prf1(predictions: list[dict], target: str) -> tuple:
    tp = fp = fn = 0
    for p in predictions:
        gt = _get_relation(p)
        pred = p.get("predicted", "")
        is_t = gt == target
        if pred.lower().strip() == gt.lower().strip():
            if is_t: tp += 1
        else:
            if is_t: fn += 1
            if pred.lower().strip() == target.lower().strip(): fp += 1
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    return pre, rec, f1


def _bstf_rate(predictions: list[dict]) -> float:
    bc, hw = 0, 0
    for p in predictions:
        votes = p.get("votes", [])
        ans = p.get("answer", "")
        if not votes or len(votes) < 2:
            continue
        hw += 1
        vc = Counter(v for v in votes if v)
        winner = vc.most_common(1)[0][0]
        if winner.lower().strip() != ans.lower().strip():
            bc += 1
    return bc / hw if hw > 0 else 0.0


def _pcr(predictions: list[dict]) -> float:
    cons, hw = 0, 0
    for p in predictions:
        votes = p.get("votes", [])
        if not votes or len(votes) < 2:
            continue
        hw += 1
        vc = Counter(v for v in votes if v)
        if vc.most_common(1)[0][1] == len(votes):
            cons += 1
    return cons / hw if hw > 0 else 0.0


def evaluate_file(pred_file: str) -> dict:
    if not os.path.exists(pred_file):
        return {}
    preds = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    return evaluate_predictions(preds)


def save_metrics(metrics: dict, output_file: str):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def print_eval_report(metrics: dict, scenario: str = ""):
    if not metrics:
        return
    if scenario:
        print(f"\n{'=' * 60}\n  {scenario}\n{'=' * 60}")
    print(f"\n  Overall Accuracy: {metrics.get('overall_acc', 0.0):.4f} "
          f"({metrics.get('correct', 0)}/{metrics.get('total', 0)})")
    pa = metrics.get("per_axis", {})
    print(f"\n  Per-Axis:  X={pa.get('x', 0.0):.4f}  Y={pa.get('y', 0.0):.4f}  Z={pa.get('z', 0.0):.4f}")
    pr = metrics.get("per_relation", {})
    for rel in RELATIONS:
        if rel in pr:
            d = pr[rel]
            print(f"    {rel:<15} {d['acc']:.4f}  (n={d['count']})")
    print(f"\n  Macro P/R/F1: {metrics.get('macro_precision', 0.0):.4f} / "
          f"{metrics.get('macro_recall', 0.0):.4f} / {metrics.get('macro_f1', 0.0):.4f}")
    print(f"  PCR: {metrics.get('pcr', 0.0):.4f}  |  BSTF: {metrics.get('bstf_rate', 0.0):.4f}\n")


def build_ablation_summary(results_dir: str) -> dict:
    summary = {}
    if not os.path.exists(results_dir):
        return summary
    for model_dir in os.listdir(results_dir):
        mp = os.path.join(results_dir, model_dir)
        if not os.path.isdir(mp):
            continue
        mr = {}
        for vd in os.listdir(mp):
            vp = os.path.join(mp, vd)
            if not os.path.isdir(vp):
                continue
            lr = os.path.join(vp, "last_results.json")
            if os.path.exists(lr):
                with open(lr, "r", encoding="utf-8") as f:
                    mr[vd] = json.load(f)
        if mr:
            summary[model_dir] = mr
    return summary


def print_ablation_summary(summary: dict):
    if not summary:
        return
    for model, variants in sorted(summary.items()):
        print(f"\n{'=' * 70}\n  Model: {model}\n{'=' * 70}")
        for variant, m in sorted(variants.items()):
            print(f"\n  [{variant}]  "
                  f"Overall={m.get('overall_acc', 0.0):.4f}  "
                  f"Depth(Z)={m.get('depth_acc', 0.0):.4f}  "
                  f"BSTF={m.get('bstf_rate', 0.0):.4f}")


def export_ablation_csv(summary: dict, output_file: str):
    import csv
    rows = []
    for model, variants in sorted(summary.items()):
        for variant, m in sorted(variants.items()):
            rows.append({
                "model": model, "variant": variant,
                "overall_acc": f"{m.get('overall_acc', 0.0):.4f}",
                "depth_acc": f"{m.get('depth_acc', 0.0):.4f}",
                "horiz_acc": f"{m.get('horiz_acc', 0.0):.4f}",
                "vert_acc": f"{m.get('vert_acc', 0.0):.4f}",
                "macro_f1": f"{m.get('macro_f1', 0.0):.4f}",
                "pcr": f"{m.get('pcr', 0.0):.4f}",
                "bstf_rate": f"{m.get('bstf_rate', 0.0):.4f}",
            })
    if not rows:
        return
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Evaluator] CSV saved: {output_file}")
