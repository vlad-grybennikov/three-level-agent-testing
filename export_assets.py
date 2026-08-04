#!/usr/bin/env python
"""Export tables and figures from suite reports for reuse (paper, slides).

    .venv/bin/python export_assets.py [reports ...] [--out exports]

Defaults to every report-*.json in the working directory. Per-report assets
land in <out>/<report-stem>/, cross-report comparisons in <out>/comparison/
when more than one report is given. Tables are written as CSV (and Markdown
when tabulate is installed), figures as 300-dpi PNG plus vector PDF.
Everything reads from the saved reports, no model calls.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def save_fig(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, out: Path, name: str, index=True) -> None:
    df.to_csv(out / f"{name}.csv", index=index)
    try:
        (out / f"{name}.md").write_text(df.to_markdown(index=index) + "\n",
                                        encoding="utf-8")
    except ImportError:  # tabulate not installed, CSV alone is fine
        pass


def task_matrix(rep: dict) -> pd.DataFrame:
    rows = []
    for t in rep["tasks"]:
        ok = [ep for ep in t["episodes"] if ep["status"] != "error"]
        l3 = t.get("level3") or {}
        mean2 = lambda key: st.mean(ep["level2"][key] for ep in ok) if ok else None
        rows.append({
            "task": t["task_id"], "capability": t["capability"],
            "L1": f"{t['level1_successes']}/{t['k']}",
            "judge": (f"{t['judge_successes']}/{t['k']}"
                      if t.get("judge_successes") is not None else "-"),
            "pass^1": t["pass_hat_k"]["1"],
            f"pass^{t['k']}": t["pass_hat_k"][str(t["k"])],
            "node_f1": mean2("node_f1"), "edge_f1": mean2("edge_f1"),
            "order": mean2("order_conformance"),
            "redundancy": mean2("redundancy_ratio"),
            "policies": st.mean(ep["policies_passed"] for ep in ok) if ok else None,
            "intent": ("ok" if l3.get("intent_correct") else "WRONG") if l3 else "-",
            "slots": l3.get("slot_accuracy"),
            "recall@k": l3.get("selection_recall_at_k"),
            "binding": l3.get("binding_accuracy"),
        })
    return pd.DataFrame(rows).set_index("task")


def export_report(name: str, rep: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tasks, k = rep["tasks"], rep["k"]
    task_ids = [t["task_id"] for t in tasks]
    ok = [(t, ep) for t in tasks for ep in t["episodes"]
          if ep["status"] != "error"]
    judged = [(t, ep) for t, ep in ok if ep.get("judge")]

    matrix = task_matrix(rep)
    save_table(matrix, out, "task_matrix")

    # Three-level summary figure.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    l1 = {"pass^1": st.mean(t["pass_hat_k"]["1"] for t in tasks),
          f"pass^{k}": st.mean(t["pass_hat_k"][str(k)] for t in tasks)}
    if judged:
        l1["judge pass^1"] = st.mean(t["judge_pass_hat_k"]["1"] for t in tasks
                                     if t.get("judge_pass_hat_k"))
    axes[0].bar(l1.keys(), l1.values(), color="#3a7d44")
    axes[0].set_title("Level 1: outcome")
    l2 = {"node F1": st.mean(ep["level2"]["node_f1"] for _, ep in ok),
          "edge F1": st.mean(ep["level2"]["edge_f1"] for _, ep in ok),
          "order": st.mean(ep["level2"]["order_conformance"] for _, ep in ok),
          "policy-clean": st.mean(ep["policies_passed"] for _, ep in ok)}
    axes[1].bar(l2.keys(), l2.values(), color="#2f6690")
    axes[1].set_title("Level 2: path")
    l3s = [t["level3"] for t in tasks if t.get("level3")]
    if l3s:
        l3 = {"intent": st.mean(x["intent_correct"] for x in l3s),
              "slots": st.mean(x["slot_accuracy"] for x in l3s),
              "recall@k": st.mean(x["selection_recall_at_k"] for x in l3s),
              "binding": st.mean(x["binding_accuracy"] for x in l3s)}
        axes[2].bar(l3.keys(), l3.values(), color="#8e5572")
    axes[2].set_title("Level 3: components")
    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"{rep['model_id']}: three-level summary", y=1.04)
    fig.tight_layout()
    save_fig(fig, out, "three_level_summary")

    # Outcome grid: oracle vs judge, per episode.
    def grid(kind):
        g = np.full((len(tasks), k), np.nan)
        for i, t in enumerate(tasks):
            for ep in t["episodes"]:
                if ep["status"] == "error":
                    continue
                if kind == "L1":
                    g[i, ep["run"]] = ep["level1_passed"]
                elif ep.get("judge"):
                    g[i, ep["run"]] = ep["judge"]["passed"]
        return g

    grids = [("Level 1 (oracle)", grid("L1"))]
    if judged:
        grids.append(("LLM judge", grid("judge")))
    fig, axes = plt.subplots(1, len(grids),
                             figsize=(3.0 * len(grids) + 3.5,
                                      0.45 * len(tasks) + 1.4),
                             squeeze=False)
    for ax, (title, g) in zip(axes[0], grids):
        ax.imshow(g, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(k), [f"run {r}" for r in range(k)])
        ax.set_yticks(range(len(tasks)), task_ids)
        ax.set_title(title)
    for ax in axes[0][1:]:
        ax.set_yticklabels([])
    fig.tight_layout()
    save_fig(fig, out, "outcome_grid")

    # Level 2 per task.
    l2df = pd.DataFrame([
        {"task": t["task_id"], "node_f1": ep["level2"]["node_f1"],
         "edge_f1": ep["level2"]["edge_f1"],
         "order": ep["level2"]["order_conformance"]}
        for t, ep in ok]).groupby("task").mean().reindex(task_ids)
    save_table(l2df, out, "level2_per_task")
    ax = l2df.plot.bar(figsize=(10, 3.5), rot=30,
                       title="Level 2 path quality per task (mean over runs)")
    ax.set_ylim(0, 1.05)
    ax.legend(title=None)
    ax.figure.tight_layout()
    save_fig(ax.figure, out, "level2_per_task")

    # Level 3 components per task.
    comp = matrix[["slots", "recall@k", "binding"]].dropna(how="all")
    if not comp.empty:
        save_table(comp, out, "level3_components")
        ax = comp.plot.bar(figsize=(10, 3.5), rot=30,
                           title="Level 3 components per task")
        ax.set_ylim(0, 1.05)
        ax.legend(title=None)
        ax.figure.tight_layout()
        save_fig(ax.figure, out, "level3_components")

    # Judge vs oracle.
    if judged:
        pairs = [(ep["level1_passed"], ep["judge"]["passed"])
                 for _, ep in judged]
        conf = pd.DataFrame(
            [[sum(l and j for l, j in pairs),
              sum(l and not j for l, j in pairs)],
             [sum(j and not l for l, j in pairs),
              sum(not (l or j) for l, j in pairs)]],
            index=["L1 pass", "L1 fail"], columns=["judge pass", "judge fail"])
        save_table(conf, out, "judge_confusion")
        dis = pd.DataFrame([
            {"task": t["task_id"], "run": ep["run"],
             "L1": ep["level1_passed"], "judge": ep["judge"]["passed"],
             "judge_reasoning": ep["judge"]["reasoning"]}
            for t, ep in judged
            if ep["judge"]["passed"] != ep["level1_passed"]])
        save_table(dis, out, "judge_disagreements", index=False)

    # Intent classification.
    ic = rep.get("intent_classification")
    if ic:
        per_class = pd.DataFrame(ic["per_class"]).T
        save_table(per_class, out, "intent_per_class")
        ax = per_class["f1"].plot.bar(figsize=(8, 3), rot=20,
                                      title="Intent per-class F1")
        ax.set_ylim(0, 1.05)
        ax.figure.tight_layout()
        save_fig(ax.figure, out, "intent_per_class")
        if ic["errors"]:
            save_table(pd.DataFrame(ic["errors"]), out, "intent_misses",
                       index=False)

    # Failure evidence, human-readable.
    lines = []
    for t in tasks:
        for ep in t["episodes"]:
            if ep["status"] == "error":
                lines.append(f"{t['task_id']} run {ep['run']}: ERROR {ep['error']}\n")
                continue
            if ep["level1_passed"]:
                continue
            lines.append(f"{t['task_id']} run {ep['run']} – L1 FAIL")
            lines.append(f"  instruction: {ep['instruction']}")
            lines.extend(f"  diff: {d}" for d in ep["level1"]["diffs"])
            if ep.get("judge"):
                j = ep["judge"]
                lines.append(f"  judge: {'PASS' if j['passed'] else 'FAIL'}"
                             f" – {j['reasoning']}")
            lines.append(f"  reply: {ep.get('summary')}\n")
    (out / "failures.txt").write_text(
        "\n".join(lines) or "no Level-1 failures\n", encoding="utf-8")

    (out / "manifest.json").write_text(json.dumps({
        "source": name, "model_id": rep["model_id"], "k": k,
        "config_hash": rep["config_hash"],
        "tasks": len(tasks), "episodes": sum(len(t["episodes"]) for t in tasks),
    }, indent=2), encoding="utf-8")


def export_comparison(reports: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # Scoreboard.
    rows = []
    for name, rep in reports.items():
        tasks = rep["tasks"]
        k = rep["k"]
        eps = [ep for t in tasks for ep in t["episodes"]]
        ok = [ep for ep in eps if ep["status"] != "error"]
        judged = [ep for ep in ok if ep.get("judge")]
        row = {"report": name, "model": rep["model_id"], "k": k,
               "episodes": len(eps), "errors": len(eps) - len(ok),
               "pass^1 (L1)": st.mean(t["pass_hat_k"]["1"] for t in tasks),
               f"pass^k (L1)": st.mean(t["pass_hat_k"][str(k)] for t in tasks),
               "policy-clean": st.mean(ep["policies_passed"] for ep in ok)}
        if judged:
            row["pass^1 (judge)"] = st.mean(
                t["judge_pass_hat_k"]["1"] for t in tasks
                if t.get("judge_pass_hat_k"))
            row["judge=L1 agreement"] = st.mean(
                ep["judge"]["passed"] == ep["level1_passed"] for ep in judged)
        ic = rep.get("intent_classification")
        if ic:
            row["intent macro-F1"] = ic["macro_f1"]
        rows.append(row)
    save_table(pd.DataFrame(rows).set_index("report"), out, "scoreboard")

    # pass^1 / pass^k per task, per model.
    long = pd.DataFrame([
        {"task": t["task_id"], "model": name,
         "pass^1": t["pass_hat_k"]["1"],
         "pass^k": t["pass_hat_k"][str(rep["k"])]}
        for name, rep in reports.items() for t in rep["tasks"]])
    for metric in ("pass^1", "pass^k"):
        piv = long.pivot(index="task", columns="model", values=metric)
        save_table(piv, out, metric.replace("^", "_hat_") + "_by_task")
        ax = piv.plot.bar(figsize=(10, 3.5), rot=30,
                          title=f"{metric} (Level 1) per task")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.legend(title=None)
        ax.figure.tight_layout()
        save_fig(ax.figure, out, metric.replace("^", "_hat_") + "_by_task")

    # Level 2 means per model.
    l2 = pd.DataFrame([
        {"model": name, "node_f1": ep["level2"]["node_f1"],
         "edge_f1": ep["level2"]["edge_f1"],
         "order": ep["level2"]["order_conformance"],
         "redundancy": ep["level2"]["redundancy_ratio"],
         "policies": ep["policies_passed"]}
        for name, rep in reports.items()
        for t in rep["tasks"] for ep in t["episodes"]
        if ep["status"] != "error"]).groupby("model").mean()
    save_table(l2, out, "level2_means")
    ax = l2[["node_f1", "edge_f1", "order"]].plot.bar(
        figsize=(7, 3.5), rot=0,
        title="Level 2 path quality (mean over episodes)")
    ax.set_ylim(0, 1.05)
    ax.figure.tight_layout()
    save_fig(ax.figure, out, "level2_means")

    # Intent per-class F1 across models.
    per_class = {name: {label: m["f1"]
                        for label, m in rep["intent_classification"]["per_class"].items()}
                 for name, rep in reports.items()
                 if rep.get("intent_classification")}
    if per_class:
        df = pd.DataFrame(per_class)
        save_table(df, out, "intent_f1_by_model")
        ax = df.plot.bar(figsize=(8, 3.5), rot=20, title="Intent per-class F1")
        ax.set_ylim(0, 1.05)
        ax.figure.tight_layout()
        save_fig(ax.figure, out, "intent_f1_by_model")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="*",
                        help="report JSON files (default: ./report-*.json)")
    parser.add_argument("--out", default="exports")
    args = parser.parse_args()

    paths = ([Path(p) for p in args.reports]
             or sorted(Path(".").glob("report-*.json")))
    if not paths:
        print("no reports found: run `make suite MODEL=...` first")
        return 1
    reports = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in paths}

    out = Path(args.out)
    for name, rep in reports.items():
        export_report(name, rep, out / name)
        print(f"exported {name} -> {out / name}/")
    if len(reports) > 1:
        export_comparison(reports, out / "comparison")
        print(f"exported comparison -> {out / 'comparison'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
