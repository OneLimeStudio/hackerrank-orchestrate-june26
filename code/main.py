"""Tiered Cascade pipeline for damage claim verification.

Layer 1 (Gatekeeper): local CPU checks — PIL, OpenCV blur/lighting, optional CLIP.
Layer 2 (VLM): Gemini API call with domain-isolated Pydantic schemas.
Post-processing: deterministic evidence_standard_met, anti-hallucination checks.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List

import pandas as pd
from PIL import Image
from dotenv import load_dotenv

from google import genai
from google.genai import types

from schemas import (
    DOMAIN_MODEL_MAP,
    OBJECT_PART_ENUMS,
    BaseClaimOutput,
)
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from evidence_lookup import EvidenceLookup
from evidence_check import EvidenceChecker
from gatekeeper import Gatekeeper, GatekeeperResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ──────────────────────────────────────────────────────────────────────
# Run stats — collected during the run, written to evaluation_report.md
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RunStats:
    total_claims: int = 0
    claims_zero_images: int = 0
    claims_fallback_default: int = 0
    claims_pydantic_retry: int = 0
    claims_layer1_auto_fail: int = 0
    vlm_calls_made: int = 0
    hallucinated_ids_stripped: int = 0
    anti_hallucination_downgrades: int = 0
    evidence_disagreements: int = 0
    evidence_total_checked: int = 0
    blur_variances: List[float] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# API key pool
# ──────────────────────────────────────────────────────────────────────
class ClientPool:
    """Manages a pool of Gemini clients to rotate API keys on quota exhaustion."""
    def __init__(self, api_keys: List[str]):
        self.clients = [genai.Client(api_key=key) for key in api_keys]
        self.current_idx = 0
        logging.info(f"Initialized ClientPool with {len(self.clients)} API keys.")

    def get_client(self) -> genai.Client:
        return self.clients[self.current_idx]

    def rotate_client(self):
        self.current_idx = (self.current_idx + 1) % len(self.clients)
        logging.info(f"Rotated to API key index {self.current_idx}")


# ──────────────────────────────────────────────────────────────────────
# Env setup
# ──────────────────────────────────────────────────────────────────────
def ensure_env() -> None:
    """Ensure at least one GEMINI_API_KEY exists, or create a template and exit."""
    load_dotenv()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(repo_root, ".env"))
    load_dotenv(os.path.join(repo_root, "code", ".env"))

    has_key = any(k.startswith("GEMINI_API_KEY") and v.strip() for k, v in os.environ.items())
    if not has_key:
        env_path = ".env"
        if not os.path.exists(env_path):
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("GEMINI_API_KEY=\nGEMINI_MODEL=gemini-2.5-flash\n")
        logging.error(
            "GEMINI_API_KEY is missing or empty! "
            "Please add your Gemini API key to .env and run again."
        )
        sys.exit(1)


def load_user_history(path: str) -> Dict[str, Dict[str, Any]]:
    """Load user history into a dictionary keyed by user_id."""
    df = pd.read_csv(path)
    return df.set_index("user_id").to_dict(orient="index")


# ──────────────────────────────────────────────────────────────────────
# API call with backoff + key rotation
# ──────────────────────────────────────────────────────────────────────
def generate_with_backoff(
    client_pool: ClientPool,
    model_name: str,
    contents: list,
    config: types.GenerateContentConfig,
    max_retries: int = 5,
) -> Any:
    """Call Gemini API with exponential backoff and API key rotation."""
    # ponytail: simple exponential backoff with key rotation.
    delay = 1.0
    for attempt in range(max_retries + 1):
        client = client_pool.get_client()
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if attempt < max_retries and ("429" in err_msg or "quota" in err_msg or "rate limit" in err_msg):
                logging.warning(
                    f"Rate limit/quota hit. Rotating key, retrying in {delay}s "
                    f"(attempt {attempt+1}/{max_retries})..."
                )
                client_pool.rotate_client()
                time.sleep(delay)
                delay *= 2
            else:
                raise


# ──────────────────────────────────────────────────────────────────────
# Anti-hallucination post-processing
# ──────────────────────────────────────────────────────────────────────
def strip_invalid_image_ids(predicted_ids_str: str, actual_ids: List[str]) -> tuple:
    """Ensure supporting_image_ids ⊆ actual_ids. Returns (cleaned_str, num_stripped)."""
    if not predicted_ids_str or predicted_ids_str.lower() == "none":
        return "none", 0

    predicted = [i.strip() for i in predicted_ids_str.split(";")]
    valid = [i for i in predicted if i in actual_ids]
    stripped = len(predicted) - len(valid)
    return (";".join(valid) if valid else "none"), stripped


def anti_hallucination_checks(
    result: Dict[str, Any],
    actual_image_ids: List[str],
    claim_object: str,
    user_id: str,
    stats: RunStats,
) -> Dict[str, Any]:
    """Apply post-VLM anti-hallucination rules. Mutates and returns result."""

    # 1. Strip invalid image IDs
    cleaned_ids, num_stripped = strip_invalid_image_ids(
        result.get("supporting_image_ids", "none"), actual_image_ids
    )
    result["supporting_image_ids"] = cleaned_ids
    if num_stripped > 0:
        stats.hallucinated_ids_stripped += num_stripped
        logging.warning(
            f"[claim={user_id}] Stripped {num_stripped} hallucinated image IDs"
        )

    # 2. Contradiction: claim_status="supported" but no supporting images
    if result.get("claim_status") == "supported" and cleaned_ids == "none":
        logging.warning(
            f"[claim={user_id}] ANTI-HALLUCINATION: supported with no supporting_image_ids → "
            "downgrading to not_enough_information"
        )
        result["claim_status"] = "not_enough_information"
        result["claim_status_justification"] = (
            result.get("claim_status_justification", "") +
            " [Downgraded: model said supported but cited no images.]"
        )
        _merge_flag(result, "manual_review_required")
        stats.anti_hallucination_downgrades += 1

    # 3. Contradiction: contradicted with no images and no evaluable evidence
    if result.get("claim_status") == "contradicted" and cleaned_ids == "none" and not actual_image_ids:
        logging.warning(
            f"[claim={user_id}] ANTI-HALLUCINATION: contradicted with no images → "
            "downgrading to not_enough_information"
        )
        result["claim_status"] = "not_enough_information"
        _merge_flag(result, "manual_review_required")
        stats.anti_hallucination_downgrades += 1

    # 4. Deterministic consistency override for wrong_object (Fix 2)
    flags = set(result.get("risk_flags", "none").split(";"))
    if "wrong_object" in flags:
        logging.warning(
            f"[claim={user_id}] ANTI-HALLUCINATION: wrong_object flag present -> forcing contradicted"
        )
        result["claim_status"] = "contradicted"
        result["valid_image"] = False
        result["evidence_standard_met"] = False
        if "claim_mismatch" not in flags:
            _merge_flag(result, "claim_mismatch")

    # 5. Defensive object_part enum check
    allowed_parts = OBJECT_PART_ENUMS.get(claim_object, "")
    if result.get("object_part") and result["object_part"] not in allowed_parts.split(", ") and result["object_part"] != "unknown":
        logging.warning(
            f"[claim={user_id}] object_part '{result['object_part']}' not in allowed enum for {claim_object} → forcing 'unknown'"
        )
        result["object_part"] = "unknown"

    # 5. Ensure enum strings (not Pydantic objects)
    for key in ["issue_type", "claim_status", "severity", "object_part"]:
        if key in result and hasattr(result[key], "value"):
            result[key] = result[key].value

    return result


def _merge_flag(result: Dict[str, Any], flag: str) -> None:
    """Add a flag to risk_flags if not already present."""
    current = result.get("risk_flags", "none")
    if current == "none":
        result["risk_flags"] = flag
    elif flag not in current:
        result["risk_flags"] = current + ";" + flag


# ──────────────────────────────────────────────────────────────────────
# Layer 2: VLM call
# ──────────────────────────────────────────────────────────────────────
def process_claim_vlm(
    client_pool: ClientPool,
    model_name: str,
    row: pd.Series,
    history: Dict[str, Any],
    evidence_req: str,
    gk_result: GatekeeperResult,
    evidence_checker: EvidenceChecker,
    stats: RunStats,
    repo_root: str,
) -> Dict[str, Any]:
    """Layer 2: call the VLM with validated images from the gatekeeper."""
    user_id = str(row["user_id"])
    claim_object = str(row["claim_object"]).lower()
    user_claim = str(row["user_claim"])

    image_parts = gk_result.valid_image_parts
    actual_image_ids = gk_result.valid_image_ids

    # ── INSTRUMENTATION: log image payload before every API call ──
    total_bytes = sum(len(p.inline_data.data) for p in image_parts) if image_parts else 0
    logging.info(
        f"[claim={user_id}] images_attached={len(image_parts)} total_bytes={total_bytes}"
    )

    has_csv_images = bool(str(row.get("image_paths", "")).strip())
    if len(image_parts) == 0 and has_csv_images:
        logging.error(
            f"[claim={user_id}] *** IMAGE_LOAD_FAILURE *** "
            f"image_paths is non-empty but 0 images attached after gatekeeper. "
            f"Setting manual_review_required."
        )

    # Pick Pydantic model and enum
    pydantic_model = DOMAIN_MODEL_MAP.get(claim_object, DOMAIN_MODEL_MAP["car"])
    enum_str = OBJECT_PART_ENUMS.get(claim_object, OBJECT_PART_ENUMS["car"])

    # Build prompts
    sys_prompt = SYSTEM_PROMPT.format(object_part_enum=enum_str, claim_object=claim_object)

    # Inject Layer 1 flags into the prompt so the VLM has context
    layer1_note = ""
    if gk_result.all_flags:
        layer1_note = (
            "\n\nPRE-SCREENING FLAGS (from automated image quality checks): "
            + "; ".join(gk_result.all_flags)
            + "\nConsider these flags when assessing risk, but make your own visual judgment."
        )

    hist_flags = history.get("history_flags", "none")
    hist_summary = history.get("history_summary", "")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        claim_object=claim_object,
        user_claim=user_claim,
        minimum_image_evidence=evidence_req,
        past_claim_count=history.get("past_claim_count", 0),
        accept_claim=history.get("accept_claim", 0),
        rejected_claim=history.get("rejected_claim", 0),
        last_90_days_claim_count=history.get("last_90_days_claim_count", 0),
        history_flags=hist_flags if pd.notna(hist_flags) else "none",
        history_summary=hist_summary if pd.notna(hist_summary) else "",
        image_id_list=", ".join(actual_image_ids) if actual_image_ids else "none",
    ) + layer1_note

    contents = [user_prompt] + list(image_parts)

    config = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        response_mime_type="application/json",
        response_schema=pydantic_model,
        temperature=0.0,
    )

    # Call API with correction loop (max 1 retry)
    error_feedback = ""
    for attempt in range(2):
        if attempt > 0:
            stats.claims_pydantic_retry += 1
            retry_contents = [
                user_prompt + f"\n\nERROR IN PREVIOUS RESPONSE: {error_feedback}\n"
                "Please correct the JSON and adhere to the schema." + layer1_note
            ] + list(image_parts)
            call_contents = retry_contents
        else:
            call_contents = contents

        try:
            cache_file = os.path.join(repo_root, "code", "vlm_cache", f"{user_id}_attempt_{attempt}.json")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            raw_text = None
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                    logging.info(f"Loaded cached VLM response for {user_id} attempt {attempt}")
                except Exception as e:
                    logging.warning(f"Failed to read cache: {e}")
            
            if raw_text is None:
                stats.vlm_calls_made += 1
                response = generate_with_backoff(client_pool, model_name, call_contents, config)
                raw_text = response.text
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(raw_text)
                except Exception as e:
                    logging.warning(f"Failed to write cache: {e}")

            parsed_json = json.loads(raw_text)
            validated = pydantic_model(**parsed_json)
            result_dict = validated.model_dump()

            # Anti-hallucination
            result_dict = anti_hallucination_checks(
                result_dict, actual_image_ids, claim_object, user_id, stats
            )

            # Deterministic evidence_standard_met
            computed_esm = evidence_checker.check(
                claim_object=claim_object,
                user_claim=user_claim,
                issue_type=result_dict.get("issue_type", "unknown"),
                valid_image_count=len(image_parts),
                model_self_report=result_dict.get("evidence_standard_met", False),
                user_id=user_id,
            )
            result_dict["evidence_standard_met"] = computed_esm

            # Merge Layer 1 flags into risk_flags
            for flag in gk_result.all_flags:
                _merge_flag(result_dict, flag)

            return result_dict

        except Exception as e:
            error_feedback = str(e)
            logging.warning(
                f"Validation failed on attempt {attempt+1} for {user_id}: {e}"
            )

    # Total failure → safe defaults
    stats.claims_fallback_default += 1
    logging.error(f"Total failure for {user_id}, returning safe defaults.")
    return _safe_defaults("System error during validation.", gk_result.all_flags)


def _safe_defaults(reason: str, extra_flags: List[str] = None) -> Dict[str, Any]:
    flags = ["manual_review_required"]
    if extra_flags:
        flags.extend(f for f in extra_flags if f not in flags)
    return {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": reason,
        "risk_flags": ";".join(flags),
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": reason,
        "supporting_image_ids": "none",
        "valid_image": False,
        "severity": "unknown",
    }


# ──────────────────────────────────────────────────────────────────────
# Pipeline orchestrator (called by main() and by evaluation/main.py)
# ──────────────────────────────────────────────────────────────────────
def run_pipeline(
    input_csv: str,
    output_csv: str,
    image_root: str,
    repo_root: str,
) -> RunStats:
    """Run the full tiered cascade pipeline. Returns stats for reporting."""
    ensure_env()

    history_path = os.path.join(repo_root, "dataset", "user_history.csv")
    reqs_path = os.path.join(repo_root, "dataset", "evidence_requirements.csv")

    logging.info(f"Loading data from {input_csv}...")
    claims_df = pd.read_csv(input_csv)
    user_history = load_user_history(history_path)
    evidence_lookup = EvidenceLookup(reqs_path)
    evidence_checker = EvidenceChecker(reqs_path)

    # API client pool
    api_keys = [v.strip() for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY") and v.strip()]
    if not api_keys:
        logging.error("No valid GEMINI_API_KEY found.")
        sys.exit(1)
    client_pool = ClientPool(api_keys)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    gatekeeper = Gatekeeper()
    stats = RunStats()
    results = []

    output_cols = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
        "issue_type", "object_part", "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity",
    ]

    for idx, row in claims_df.iterrows():
        user_id = str(row["user_id"])
        stats.total_claims += 1
        logging.info(f"Processing claim {idx+1}/{len(claims_df)} for user {user_id}...")

        history = user_history.get(user_id, {})
        evidence_req = evidence_lookup.get_evidence_requirement(
            row["claim_object"], row["user_claim"]
        )

        # Resolve image paths
        raw_paths = str(row.get("image_paths", "")).split(";")
        resolved_paths = []
        for p in raw_paths:
            p = p.strip()
            if not p:
                continue
            # Strip any leading "images/" and re-root under image_root
            # CSV paths look like "images/test/case_001/img_1.jpg" or "images/sample/..."
            # image_root is e.g. "dataset/images/test/" or "dataset/images/sample/"
            basename_parts = p.replace("\\", "/")
            # Remove "images/test/" or "images/sample/" prefix if present
            for prefix in ["images/test/", "images/sample/"]:
                if basename_parts.startswith(prefix):
                    basename_parts = basename_parts[len(prefix):]
                    break
            full = os.path.join(repo_root, image_root, basename_parts)
            resolved_paths.append(os.path.normpath(full))

        # ── Layer 1: Gatekeeper ──
        gk_result = gatekeeper.check_claim_images(
            image_paths=resolved_paths,
            claim_object=str(row["claim_object"]).lower(),
            user_id=user_id,
        )

        # Collect blur variances for report
        for v in gk_result.verdicts:
            if v.blur_variance >= 0:
                stats.blur_variances.append(v.blur_variance)

        if not resolved_paths or (not gk_result.valid_image_parts and not gk_result.verdicts):
            stats.claims_zero_images += 1

        # ── Layer 1 Auto-fail ──
        if gk_result.auto_fail:
            stats.claims_layer1_auto_fail += 1
            logging.warning(
                f"[claim={user_id}] LAYER1_AUTO_FAIL: no valid images passed gatekeeper."
            )
            result_dict = _safe_defaults(
                "No images passed automated quality checks (Layer 1 auto-fail).",
                gk_result.all_flags,
            )
            result_dict["evidence_standard_met"] = False
            result_dict["valid_image"] = False
        else:
            # ── Layer 2: VLM ──
            try:
                result_dict = process_claim_vlm(
                    client_pool=client_pool,
                    model_name=model_name,
                    row=row,
                    history=history,
                    evidence_req=evidence_req,
                    gk_result=gk_result,
                    evidence_checker=evidence_checker,
                    stats=stats,
                    repo_root=repo_root,
                )
            except Exception as e:
                logging.error(f"Unexpected error processing claim {user_id}: {e}")
                stats.claims_fallback_default += 1
                result_dict = _safe_defaults(
                    f"Fatal exception: {e}", gk_result.all_flags
                )

        out_row = {
            "user_id": user_id,
            "image_paths": row["image_paths"],
            "user_claim": row["user_claim"],
            "claim_object": row["claim_object"],
            **result_dict,
        }
        results.append(out_row)

    # Write output
    logging.info(f"Writing {len(results)} rows to {output_csv}...")
    out_df = pd.DataFrame(results)
    for col in output_cols:
        if col not in out_df.columns:
            out_df[col] = ""
    out_df = out_df[output_cols]
    out_df.to_csv(output_csv, index=False)

    # Finalize stats from evidence checker
    stats.evidence_disagreements = evidence_checker.disagreement_count
    stats.evidence_total_checked = evidence_checker.total_checked

    # ── Print run summary ──
    _print_summary(stats)

    logging.info("Done.")
    return stats


def _print_summary(stats: RunStats) -> None:
    """Print and return the run summary block."""
    sep = "=" * 60
    lines = [
        sep,
        "RUN SUMMARY",
        sep,
        f"Total claims processed:          {stats.total_claims}",
        f"Claims with 0 images attached:   {stats.claims_zero_images}",
        f"Layer 1 auto-fail (no VLM call): {stats.claims_layer1_auto_fail}",
        f"Fallback-default rows:           {stats.claims_fallback_default}",
        f"Pydantic retries:                {stats.claims_pydantic_retry}",
        f"VLM API calls made:              {stats.vlm_calls_made}",
        f"Hallucinated image IDs stripped:  {stats.hallucinated_ids_stripped}",
        f"Anti-hallucination downgrades:    {stats.anti_hallucination_downgrades}",
        f"Evidence standard disagreements:  {stats.evidence_disagreements}/{stats.evidence_total_checked}",
        sep,
    ]
    for line in lines:
        logging.info(line)


def write_evaluation_report(stats: RunStats, report_path: str, runtime: float = 0.0) -> None:
    """Write evaluation_report.md from run stats."""
    total_images = len(stats.blur_variances)
    blur_vals = sorted(stats.blur_variances)
    blur_dist = ""
    if blur_vals:
        import statistics
        from gatekeeper import BLUR_THRESHOLD
        blur_dist = (
            f"  - Min: {blur_vals[0]:.1f}\n"
            f"  - Max: {blur_vals[-1]:.1f}\n"
            f"  - Median: {statistics.median(blur_vals):.1f}\n"
            f"  - Mean: {statistics.mean(blur_vals):.1f}\n"
            f"  - Threshold: {BLUR_THRESHOLD}\n"
            f"  - Below threshold: {sum(1 for v in blur_vals if v < BLUR_THRESHOLD)}/{len(blur_vals)}\n"
        )

    baseline_calls = stats.total_claims  # Direct VLM baseline = 1 call per claim
    cascade_calls = stats.vlm_calls_made
    if baseline_calls > 0:
        reduction = (1 - cascade_calls / baseline_calls) * 100
    else:
        reduction = 0.0

    ev_rate = (
        f"{stats.evidence_disagreements}/{stats.evidence_total_checked} "
        f"({stats.evidence_disagreements / max(stats.evidence_total_checked, 1) * 100:.1f}%)"
    )

    # Token estimates
    avg_text_tokens = 500
    avg_tokens_per_image = 258
    total_input_tokens = cascade_calls * avg_text_tokens + total_images * avg_tokens_per_image
    total_output_tokens = cascade_calls * 150

    # Gemini 2.5 flash pricing
    cost_in = 0.075  # per 1M input tokens
    cost_out = 0.30   # per 1M output tokens
    est_cost = (total_input_tokens / 1_000_000 * cost_in) + (total_output_tokens / 1_000_000 * cost_out)

    report = f"""# Evaluation Report — Tiered Cascade Pipeline

## Run Summary
- **Total claims processed**: {stats.total_claims}
- **Claims with 0 images attached**: {stats.claims_zero_images}
- **Layer 1 auto-fail (skipped VLM)**: {stats.claims_layer1_auto_fail}
- **Layer 1 auto-fail rate**: {stats.claims_layer1_auto_fail / max(stats.total_claims, 1) * 100:.1f}%
- **Fallback-default rows**: {stats.claims_fallback_default}
- **Pydantic retries**: {stats.claims_pydantic_retry}

## API Call Reduction
- **Direct VLM baseline calls**: {baseline_calls} (1 per claim)
- **Cascade VLM calls**: {cascade_calls}
- **API call reduction**: {reduction:.1f}%

## Anti-Hallucination
- **Hallucinated image IDs stripped**: {stats.hallucinated_ids_stripped}
- **Anti-hallucination claim downgrades**: {stats.anti_hallucination_downgrades}

## Evidence Standard Disagreement
- **Model vs computed disagreements**: {ev_rate}

## Blur Threshold Distribution
{blur_dist if blur_dist else "No blur data collected."}

## Operational Analysis
- **Runtime**: {runtime:.2f}s
- **Images processed (Layer 1)**: {total_images}
- **Model calls (Layer 2)**: {cascade_calls}
- **Approx input tokens**: {total_input_tokens:,}
- **Approx output tokens**: {total_output_tokens:,}
- **Estimated API cost**: ${est_cost:.5f}
- **Pricing**: Gemini 2.5 Flash — $0.075/1M input, $0.30/1M output

## Rate-Limit Handling
Exponential backoff (1s → 2s → 4s → …) with automatic API key rotation across
a pool of keys. Pydantic validation failures trigger a 1-time retry with the
validation error fed back to the model. Layer 1 gatekeeper pre-filters save
API calls for claims with unusable images.
"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logging.info(f"Evaluation report written to {report_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Tiered Cascade Damage Claim Pipeline")
    parser.add_argument("--input", default="dataset/claims.csv", help="Path to input claims CSV (relative to repo root)")
    parser.add_argument("--output", default="output.csv", help="Path to output CSV (relative to repo root)")
    parser.add_argument("--image-root", default="dataset/images/test/", help="Image root directory (relative to repo root)")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(repo_root, args.input)
    if not os.path.exists(input_path):
        input_path = args.input
        repo_root = os.getcwd()

    output_path = os.path.join(repo_root, args.output)

    start = time.time()
    stats = run_pipeline(
        input_csv=input_path,
        output_csv=output_path,
        image_root=args.image_root,
        repo_root=repo_root,
    )
    runtime = time.time() - start

    report_path = os.path.join(repo_root, "code", "evaluation", "evaluation_report.md")
    write_evaluation_report(stats, report_path, runtime)


if __name__ == "__main__":
    main()
