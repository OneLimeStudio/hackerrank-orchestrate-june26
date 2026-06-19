"""Evaluation script for the Tiered Cascade claim verification system.

Runs the pipeline on sample_claims.csv, computes accuracy metrics, generates
the evaluation report, and prints a side-by-side comparison against the
Direct VLM baseline's saved metrics.
"""

import json
import os
import time

import pandas as pd
from sklearn.metrics import accuracy_score


# Saved baseline metrics from the Direct VLM run.
# These are the numbers from the first non-cascade run so we can show delta.
BASELINE_METRICS = {
    "claim_status_accuracy": 0.0,
    "issue_type_accuracy": 0.0,
    "object_part_accuracy": 0.0,
    "risk_flags_exact_match": 0.0,
    "supporting_image_ids_exact_match": 0.0,
}

BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "baseline_metrics.json"
)


def load_baseline():
    """Load saved baseline metrics if available."""
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return BASELINE_METRICS.copy()


def save_baseline(metrics: dict):
    """Save current metrics as the baseline for future comparisons."""
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def exact_match(t_series, p_series):
    """Compute exact match rate for semicolon-delimited fields."""
    matches = 0
    for t, p in zip(t_series, p_series):
        t_set = set(x.strip().lower() for x in str(t).split(";"))
        p_set = set(x.strip().lower() for x in str(p).split(";"))
        if t_set == p_set:
            matches += 1
    return matches / len(t_series) if len(t_series) > 0 else 0.0


def evaluate():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sample_csv = os.path.join(repo_root, "dataset", "sample_claims.csv")
    eval_output_csv = os.path.join(repo_root, "eval_output.csv")

    # Import the pipeline function directly — avoids subprocess issues
    import sys
    sys.path.insert(0, os.path.join(repo_root, "code"))
    from main import run_pipeline, write_evaluation_report

    print("Running Tiered Cascade pipeline on sample_claims.csv...")
    start_time = time.time()

    stats = run_pipeline(
        input_csv=sample_csv,
        output_csv=eval_output_csv,
        image_root="dataset/images/sample/",  # Explicitly sample images
        repo_root=repo_root,
    )

    runtime = time.time() - start_time
    print(f"Pipeline finished in {runtime:.2f}s")

    # Load ground truth and predictions
    truth_df = pd.read_csv(sample_csv)
    pred_df = pd.read_csv(eval_output_csv)

    if len(truth_df) != len(pred_df):
        print(f"Row count mismatch! Truth: {len(truth_df)}, Preds: {len(pred_df)}")
        return

    # Compute metrics
    metrics = {}
    for col in ["claim_status", "issue_type", "object_part"]:
        metrics[f"{col}_accuracy"] = accuracy_score(
            truth_df[col].astype(str).str.lower(),
            pred_df[col].astype(str).str.lower(),
        )

    metrics["risk_flags_exact_match"] = exact_match(
        truth_df["risk_flags"], pred_df["risk_flags"]
    )
    metrics["supporting_image_ids_exact_match"] = exact_match(
        truth_df["supporting_image_ids"], pred_df["supporting_image_ids"]
    )

    # Print current metrics
    sep = "=" * 60
    print(f"\n{sep}")
    print("CASCADE METRICS (sample_claims.csv)")
    print(sep)
    for k, v in metrics.items():
        print(f"  {k}: {v:.2%}")

    # Side-by-side comparison with baseline
    baseline = load_baseline()
    has_baseline = any(v > 0 for v in baseline.values())

    print(f"\n{sep}")
    print("SIDE-BY-SIDE: Cascade vs Direct VLM Baseline")
    print(sep)
    print(f"  {'Metric':<45} {'Baseline':>10} {'Cascade':>10} {'Delta':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10}")
    for key in metrics:
        b = baseline.get(key, 0.0)
        c = metrics[key]
        delta = c - b
        sign = "+" if delta >= 0 else ""
        b_str = f"{b:.2%}" if has_baseline else "N/A"
        print(f"  {key:<45} {b_str:>10} {c:.2%}    {sign}{delta:.2%}")

    # Cascade-specific stats
    print(f"\n{sep}")
    print("CASCADE-SPECIFIC STATS")
    print(sep)
    print(f"  Layer 1 auto-fail rate:           {stats.claims_layer1_auto_fail}/{stats.total_claims} ({stats.claims_layer1_auto_fail / max(stats.total_claims, 1) * 100:.1f}%)")
    baseline_calls = stats.total_claims
    print(f"  API call reduction vs baseline:   {baseline_calls} -> {stats.vlm_calls_made} ({(1 - stats.vlm_calls_made / max(baseline_calls, 1)) * 100:.1f}% reduction)")
    print(f"  Evidence standard disagreements:  {stats.evidence_disagreements}/{stats.evidence_total_checked}")
    print(f"  Hallucinated image IDs stripped:   {stats.hallucinated_ids_stripped}")

    if stats.blur_variances:
        import statistics
        bv = sorted(stats.blur_variances)
        print(f"\n  Blur variance distribution:")
        print(f"    Min={bv[0]:.1f}  Max={bv[-1]:.1f}  Median={statistics.median(bv):.1f}  Mean={statistics.mean(bv):.1f}")
        print(f"    Below threshold (80): {sum(1 for v in bv if v < 80)}/{len(bv)}")

    print(sep)

    # Write the full evaluation report
    report_path = os.path.join(repo_root, "code", "evaluation", "evaluation_report.md")
    write_evaluation_report(stats, report_path, runtime)

    # If no baseline existed, save current as baseline
    if not has_baseline:
        print("\nNo previous baseline found. Saving current metrics as baseline.")
        save_baseline(metrics)
    else:
        print(f"\nBaseline loaded from {BASELINE_PATH}")
        print("To update the baseline with current metrics, delete baseline_metrics.json and re-run.")

    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    evaluate()
