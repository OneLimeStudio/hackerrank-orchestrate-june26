"""Evaluation script for the claim verification system."""

import os
import subprocess
import time
import pandas as pd
from sklearn.metrics import accuracy_score


def evaluate():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_csv = os.path.join(repo_root, "dataset/sample_claims.csv")
    eval_output_csv = os.path.join(repo_root, "eval_output.csv")
    main_script = os.path.join(repo_root, "code/main.py")
    
    print("Running pipeline on sample claims...")
    start_time = time.time()
    
    # Run the main pipeline
    result = subprocess.run(
        ["python", main_script, "--input", "dataset/sample_claims.csv", "--output", "eval_output.csv"],
        cwd=repo_root,
        capture_output=True,
        text=True
    )
    
    runtime = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Pipeline failed:\n{result.stderr}")
        return
        
    print(f"Pipeline finished in {runtime:.2f}s")
    
    # Load ground truth and predictions
    truth_df = pd.read_csv(sample_csv)
    pred_df = pd.read_csv(eval_output_csv)
    
    # Ensure they align
    if len(truth_df) != len(pred_df):
        print(f"Row count mismatch! Truth: {len(truth_df)}, Preds: {len(pred_df)}")
        return
        
    # Metrics
    metrics = {}
    for col in ["claim_status", "issue_type", "object_part"]:
        metrics[f"{col}_accuracy"] = accuracy_score(
            truth_df[col].astype(str).str.lower(), 
            pred_df[col].astype(str).str.lower()
        )
        
    # Exact match for lists
    def exact_match(t_series, p_series):
        matches = 0
        for t, p in zip(t_series, p_series):
            # Sort delimited strings to ignore order
            t_set = set([x.strip().lower() for x in str(t).split(";")])
            p_set = set([x.strip().lower() for x in str(p).split(";")])
            if t_set == p_set:
                matches += 1
        return matches / len(t_series)
        
    metrics["risk_flags_exact_match"] = exact_match(truth_df["risk_flags"], pred_df["risk_flags"])
    metrics["supporting_image_ids_exact_match"] = exact_match(truth_df["supporting_image_ids"], pred_df["supporting_image_ids"])
    
    # Operational Heuristics
    num_claims = len(truth_df)
    total_images = sum(len(str(p).split(";")) for p in truth_df["image_paths"])
    avg_tokens_per_image = 258 # Gemini standard base
    avg_text_tokens = 500
    total_input_tokens = num_claims * avg_text_tokens + total_images * avg_tokens_per_image
    total_output_tokens = num_claims * 150 # approx JSON response size
    
    # Gemini 2.5 flash pricing (approx)
    cost_per_1m_in = 0.075
    cost_per_1m_out = 0.30
    est_sample_cost = (total_input_tokens / 1_000_000 * cost_per_1m_in) + (total_output_tokens / 1_000_000 * cost_per_1m_out)
    
    # Extrapolate to full claims.csv (assume ~45 rows based on wc)
    extrapolate_factor = 45 / num_claims
    est_full_cost = est_sample_cost * extrapolate_factor
    
    report = f"""# Evaluation Report

## Performance Metrics (Sample Set)
- **claim_status Accuracy**: {metrics['claim_status_accuracy']:.2%}
- **issue_type Accuracy**: {metrics['issue_type_accuracy']:.2%}
- **object_part Accuracy**: {metrics['object_part_accuracy']:.2%}
- **risk_flags Exact Match**: {metrics['risk_flags_exact_match']:.2%}
- **supporting_image_ids Exact Match**: {metrics['supporting_image_ids_exact_match']:.2%}

## Operational Analysis
- **Runtime (Sample Set)**: {runtime:.2f} seconds
- **Images Processed**: {total_images}
- **Model Calls**: ~{num_claims} (assuming no retries)
- **Approx Input Tokens**: {total_input_tokens:,}
- **Approx Output Tokens**: {total_output_tokens:,}
- **Estimated API Cost (Sample)**: ${est_sample_cost:.5f}
- **Estimated Full Run Cost (45 claims)**: ${est_full_cost:.5f}

## Rate-Limit Handling
The system uses an exponential backoff strategy for HTTP 429 Resource Exhausted errors. 
It sleeps for 1s, doubling the delay up to 5 times. Pydantic validation failures trigger a 1-time retry loop passing the validation error back to the model as feedback. 
"""
    
    report_path = os.path.join(repo_root, "code", "evaluation", "evaluation_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    evaluate()
