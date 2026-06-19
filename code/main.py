"""Main pipeline for damage claim verification."""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Type, List

import pandas as pd
from PIL import Image
from dotenv import load_dotenv

# Import the new google-genai SDK
from google import genai
from google.genai import types

from schemas import (
    DOMAIN_MODEL_MAP,
    OBJECT_PART_ENUMS,
    BaseClaimOutput,
)
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from evidence_lookup import EvidenceLookup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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


def ensure_env() -> None:
    """Ensure at least one GEMINI_API_KEY exists, or create a template and exit."""
    # Try multiple common .env locations
    load_dotenv()
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(repo_root, ".env"))
    load_dotenv(os.path.join(repo_root, "code", ".env"))
    
    # Check if any GEMINI_API_KEY exists
    has_key = any(k.startswith("GEMINI_API_KEY") and v.strip() for k, v in os.environ.items())
    
    if not has_key:
        env_path = ".env"
        if not os.path.exists(env_path):
            with open(env_path, "w") as f:
                f.write("GEMINI_API_KEY=\nGEMINI_MODEL=gemini-2.5-flash\n")
        
        logging.error(
            "GEMINI_API_KEY is missing or empty! "
            "A .env template has been created in the repository root. "
            "Please add your Gemini API key to it and run again."
        )
        sys.exit(1)


def load_user_history(path: str) -> Dict[str, Dict[str, Any]]:
    """Load user history into a dictionary keyed by user_id."""
    df = pd.read_csv(path)
    return df.set_index("user_id").to_dict(orient="index")


def generate_with_backoff(
    client_pool: ClientPool,
    model_name: str,
    contents: list,
    config: types.GenerateContentConfig,
    max_retries: int = 5
) -> Any:
    """Call Gemini API with exponential backoff and API key rotation for rate limits."""
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
                logging.warning(f"Rate limit or quota hit ({e}). Rotating API key and retrying in {delay}s (attempt {attempt+1}/{max_retries})...")
                client_pool.rotate_client()
                time.sleep(delay)
                delay *= 2
            else:
                raise e


def strip_invalid_image_ids(predicted_ids_str: str, actual_ids: List[str]) -> str:
    """Ensure supporting_image_ids only contains valid IDs."""
    if not predicted_ids_str or predicted_ids_str.lower() == "none":
        return "none"
    
    predicted_ids = [id.strip() for id in predicted_ids_str.split(";")]
    valid_ids = [id for id in predicted_ids if id in actual_ids]
    
    return ";".join(valid_ids) if valid_ids else "none"


def process_claim(
    client_pool: ClientPool,
    model_name: str,
    row: pd.Series,
    history: Dict[str, Any],
    evidence_req: str,
    repo_root: str
) -> Dict[str, Any]:
    """Process a single claim through the VLM."""
    claim_object = str(row["claim_object"]).lower()
    user_claim = str(row["user_claim"])
    
    # 1. Load images and extract IDs
    image_paths = str(row["image_paths"]).split(";")
    images = []
    actual_image_ids = []
    
    for path in image_paths:
        path = path.strip()
        if not path: continue
        
        # Paths in CSV are like "images/test/case_001/img_1.jpg", but they live in "dataset/"
        if not path.startswith("dataset/") and not path.startswith("dataset\\"):
            path = os.path.join("dataset", path)
            
        full_path = os.path.join(repo_root, path)
        try:
            # Load with Pillow to ensure it's a valid image
            img = Image.open(full_path)
            # Make sure we don't hold the file open unnecessarily
            img.load()
            images.append(img)
            
            # ID is filename without extension
            img_id = os.path.splitext(os.path.basename(path))[0]
            actual_image_ids.append(img_id)
        except Exception as e:
            logging.warning(f"Failed to load image {full_path}: {e}")
    
    if not images:
        logging.warning(f"No valid images loaded for {row['user_id']}")
        # We can still proceed, the model should output not_enough_information
    
    # 2. Pick Pydantic model and enum
    pydantic_model = DOMAIN_MODEL_MAP.get(claim_object, DOMAIN_MODEL_MAP["car"])
    enum_str = OBJECT_PART_ENUMS.get(claim_object, OBJECT_PART_ENUMS["car"])
    
    # 3. Build prompts
    sys_prompt = SYSTEM_PROMPT.format(object_part_enum=enum_str)
    
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
        image_id_list=", ".join(actual_image_ids) if actual_image_ids else "none"
    )
    
    # 4. Content payload
    contents = [user_prompt] + images
    
    config = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        response_mime_type="application/json",
        response_schema=pydantic_model,
        temperature=0.0,
    )
    
    # 5. Call API with correction loop
    error_feedback = ""
    for attempt in range(2): # Max 1 retry
        if attempt > 0:
            retry_contents = contents.copy()
            retry_contents[0] = user_prompt + f"\n\nERROR IN PREVIOUS RESPONSE: {error_feedback}\nPlease correct the JSON and adhere to the schema."
            call_contents = retry_contents
        else:
            call_contents = contents
            
        try:
            response = generate_with_backoff(client_pool, model_name, call_contents, config)
            raw_text = response.text
            
            # Attempt to parse json and pydantic model
            parsed_json = json.loads(raw_text)
            validated_model = pydantic_model(**parsed_json)
            
            result_dict = validated_model.model_dump()
            
            # Post-validation: strip invalid image IDs
            result_dict["supporting_image_ids"] = strip_invalid_image_ids(
                result_dict["supporting_image_ids"], 
                actual_image_ids
            )
            
            # Ensure enums are extracted to strings
            for key in ["issue_type", "claim_status", "severity", "object_part"]:
                if key in result_dict and hasattr(result_dict[key], "value"):
                    result_dict[key] = result_dict[key].value
                    
            return result_dict
            
        except Exception as e:
            error_feedback = str(e)
            logging.warning(f"Validation failed on attempt {attempt+1} for {row['user_id']}: {e}")
    
    # 6. Safe defaults on total failure
    logging.error(f"Total failure for {row['user_id']}, returning safe defaults.")
    return {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": "System error during validation.",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Failed to generate valid analysis.",
        "supporting_image_ids": "none",
        "valid_image": False,
        "severity": "unknown"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/claims.csv", help="Path to input claims CSV")
    parser.add_argument("--output", default="output.csv", help="Path to output CSV")
    args = parser.parse_args()
    
    ensure_env()
    
    # Paths are relative to the repo root where the script should be run from
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    input_path = os.path.join(repo_root, args.input)
    if not os.path.exists(input_path):
        # Fallback if run directly from repo root
        input_path = args.input
        repo_root = os.getcwd()
        
    history_path = os.path.join(repo_root, "dataset/user_history.csv")
    reqs_path = os.path.join(repo_root, "dataset/evidence_requirements.csv")
    output_path = os.path.join(repo_root, args.output)
    
    # Load data
    logging.info(f"Loading data from {input_path}...")
    claims_df = pd.read_csv(input_path)
    user_history = load_user_history(history_path)
    evidence_lookup = EvidenceLookup(reqs_path)
    
    # Setup API client pool
    api_keys = []
    for k, v in os.environ.items():
        if k.startswith("GEMINI_API_KEY") and v.strip():
            api_keys.append(v.strip())
            
    if not api_keys:
        logging.error("No valid GEMINI_API_KEY found.")
        sys.exit(1)
        
    client_pool = ClientPool(api_keys)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    
    results = []
    
    # Required output columns in exact order
    output_cols = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
        "issue_type", "object_part", "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity"
    ]
    
    # Process rows
    for idx, row in claims_df.iterrows():
        user_id = row["user_id"]
        logging.info(f"Processing claim {idx+1}/{len(claims_df)} for user {user_id}...")
        
        history = user_history.get(user_id, {})
        evidence_req = evidence_lookup.get_evidence_requirement(
            row["claim_object"], 
            row["user_claim"]
        )
        
        try:
            result_dict = process_claim(
                client_pool=client_pool,
                model_name=model_name,
                row=row,
                history=history,
                evidence_req=evidence_req,
                repo_root=repo_root
            )
            
            # Merge base info
            out_row = {
                "user_id": user_id,
                "image_paths": row["image_paths"],
                "user_claim": row["user_claim"],
                "claim_object": row["claim_object"],
                **result_dict
            }
            results.append(out_row)
            
        except Exception as e:
            logging.error(f"Unexpected error processing row {idx}: {e}")
            out_row = {
                "user_id": user_id,
                "image_paths": row["image_paths"],
                "user_claim": row["user_claim"],
                "claim_object": row["claim_object"],
                "evidence_standard_met": False,
                "evidence_standard_met_reason": "Fatal exception.",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": f"Exception: {str(e)}",
                "supporting_image_ids": "none",
                "valid_image": False,
                "severity": "unknown"
            }
            results.append(out_row)
            
    # Write output preserving order
    logging.info(f"Writing {len(results)} rows to {output_path}...")
    out_df = pd.DataFrame(results)
    
    # Ensure all required columns exist (fill missing with defaults just in case)
    for col in output_cols:
        if col not in out_df.columns:
            out_df[col] = ""
            
    out_df = out_df[output_cols]
    out_df.to_csv(output_path, index=False)
    logging.info("Done.")


if __name__ == "__main__":
    main()
