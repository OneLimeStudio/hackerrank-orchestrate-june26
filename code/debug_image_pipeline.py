import os
import sys
import pandas as pd
from PIL import Image

def run_diagnostic():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_csv = os.path.join(repo_root, "dataset", "sample_claims.csv")
    
    report_lines = []
    def log(msg):
        print(msg)
        report_lines.append(msg)
        
    log("=== DIAGNOSTIC START ===")
    
    if not os.path.exists(sample_csv):
        log(f"ERROR: Cannot find {sample_csv}")
        return
        
    df = pd.read_csv(sample_csv)
    
    total_referenced = 0
    total_resolved = 0
    total_loaded = 0
    
    log("\n--- IMAGE PATH RESOLUTION & LOAD TEST ---")
    for idx, row in df.iterrows():
        image_paths_str = str(row.get("image_paths", ""))
        if not image_paths_str or image_paths_str.lower() == "nan":
            continue
            
        paths = image_paths_str.split(";")
        for p in paths:
            p = p.strip()
            if not p: continue
            
            total_referenced += 1
            
            # Reconstruct what the user probably wants to see based on the prompt
            # "resolving against the actual image root used during the last evaluation run"
            # Note: The raw paths in CSV are like "images/sample/case_001/img_1.jpg"
            # But they should be relative to dataset/ OR repo_root/ depending on the run.
            # We will test repo_root + p, and repo_root + dataset + p.
            
            if not p.startswith("dataset/") and not p.startswith("dataset\\"):
                test_p = os.path.join("dataset", p)
            else:
                test_p = p
                
            abs_path = os.path.join(repo_root, test_p)
            exists = os.path.exists(abs_path)
            
            log(f"Resolved Path: {abs_path}")
            log(f"  Exists: {exists}")
            
            if exists:
                total_resolved += 1
                try:
                    img = Image.open(abs_path)
                    img.verify()  # verify first
                    
                    # Must reopen to load after verify
                    img = Image.open(abs_path)
                    img.load()
                    
                    # File size
                    file_size = os.path.getsize(abs_path)
                    # First 8 bytes of raw file
                    with open(abs_path, "rb") as f:
                        first_8_bytes = f.read(8)
                        
                    log(f"  Loaded successfully: {img.width}x{img.height}, {file_size} bytes, mode={img.mode}")
                    log(f"  Raw file first 8 bytes: {first_8_bytes}")
                    total_loaded += 1
                except Exception as e:
                    log(f"  LOAD FAILED: {str(e)}")
                    
    log("\n--- COUNTS ---")
    log(f"Total Referenced: {total_referenced}")
    log(f"Total Resolved: {total_resolved}")
    log(f"Total Loaded: {total_loaded}")
    
    if total_referenced == total_resolved == total_loaded:
        log("COUNTS MATCH: Pipeline is healthy at the I/O layer.")
    else:
        if total_resolved < total_referenced:
            log(f"MISMATCH: {total_referenced} referenced, {total_resolved} resolved, {total_loaded} loaded — likely a path/root mismatch, not a Pillow issue.")
        else:
            log(f"MISMATCH: {total_referenced} referenced, {total_resolved} resolved, {total_loaded} loaded — likely a Pillow/format issue, not a path issue.")

    log("\n--- MAIN.PY PAYLOAD ASSEMBLY TEST (All claims) ---")
    
    # Import main module to intercept
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import main
    
    # Mock the backoff function
    intercepted_payloads = []
    
    def mock_generate_with_backoff(client_pool, model_name, contents, config, max_retries=5):
        # We just want to inspect the contents!
        image_parts = []
        for item in contents:
            if isinstance(item, Image.Image):
                image_parts.append({
                    "type": str(type(item)),
                    "first_8_bytes": item.tobytes()[:8],
                    "mime_type": getattr(item, "mime_type", "None (PIL Image object)")
                })
            elif not isinstance(item, str):
                # If it's a Part or other object
                image_parts.append({
                    "type": str(type(item)),
                    "first_8_bytes": "N/A",
                    "mime_type": getattr(item, "mime_type", "Unknown")
                })
                
        intercepted_payloads.append(image_parts)
        # Raise an exception so process_claim catches it and moves on
        raise Exception("MOCK_INTERCEPT_SUCCESS")
        
    # Monkey-patch
    original_generate = main.generate_with_backoff
    main.generate_with_backoff = mock_generate_with_backoff
    
    # Run all claims
    from evidence_lookup import EvidenceLookup
    history_path = os.path.join(repo_root, "dataset", "user_history.csv")
    req_path = os.path.join(repo_root, "dataset", "evidence_requirements.csv")
    
    user_history = main.load_user_history(history_path)
    lookup = EvidenceLookup(req_path)
    
    for i in range(len(df)):
        row = df.iloc[i]
        user_id = row["user_id"]
        history = user_history.get(user_id, {})
        evidence_req = lookup.get_evidence_requirement(row["claim_object"], row["user_claim"])
        
        # We can pass None for client_pool because our mock doesn't use it
        _ = main.process_claim(None, "mock-model", row, history, evidence_req, repo_root)
        
        if len(intercepted_payloads) > i:
            parts = intercepted_payloads[i]
            log(f"Claim {i+1} ({user_id}): {len(parts)} images passed to generate_content.")
            for j, p in enumerate(parts):
                log(f"  Image {j+1}: type={p['type']}, mime_type={p['mime_type']}")
                log(f"  Image {j+1} payload first 8 bytes: {p['first_8_bytes']}")
        else:
            log(f"Claim {i+1} ({user_id}): 0 images attached to API call (or call never reached).")

    # Restore
    main.generate_with_backoff = original_generate
    
    log("\n--- VERDICT ---")
    if total_resolved < total_referenced:
        verdict = "VERDICT: PATH RESOLUTION BUG — images never found on disk"
    elif total_loaded < total_resolved:
        verdict = "VERDICT: IMAGE DECODE BUG — found but failed to load as valid images"
    elif any(len(p) == 0 for p in intercepted_payloads if total_loaded > 0):
        # We expect some images to be attached since they loaded
        verdict = "VERDICT: PAYLOAD ASSEMBLY BUG — images load fine but aren't reaching the API call"
    else:
        verdict = "VERDICT: IMAGES LOADING CORRECTLY — bug is not image I/O"
        
    log(verdict)
    
    with open(os.path.join(repo_root, "debug_image_pipeline_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    run_diagnostic()
