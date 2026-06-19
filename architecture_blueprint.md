# Hybrid Cascade Architecture Blueprint
## Multi-Modal Insurance Damage Claim Verification System

---

## Pillar 1: Gatekeeper Blueprint (Layer 1)

### Objective
Filter out unprocessable or trivially invalid images **locally** before touching the VLM API. This is the single biggest cost and latency lever.

### Recommended Local Libraries & Models

| Check | Library / Model | Rationale |
|---|---|---|
| **Blurriness** | `OpenCV` — `cv2.Laplacian` variance | Fast, deterministic, zero-cost. Variance < 80 = blurry. |
| **File validity / corruption** | `Pillow (PIL)` — `Image.verify()` | Catch truncated/corrupted JPEG/PNG before decode. |
| **Image size** | `Pillow` — width/height check | Reject images < 100×100 px as non-informative. |
| **Object-type mismatch** | `CLIP` (ViT-B/32, via `open_clip`) | Compare image embedding to text labels: `"a damaged car"`, `"a damaged laptop"`, `"a damaged package"`. cosine_sim < 0.20 = wrong object. |
| **Over-cropping / obstruction** | `CLIP` + `edge density` (OpenCV) | Low CLIP score AND low edge density = likely a blank/cropped shot. |
| **Lighting / glare** | `OpenCV` — mean pixel intensity | Mean > 240 = severe glare. Mean < 30 = low-light. |

> **Why CLIP ViT-B/32?** It runs in ~10–50ms CPU-locally per image, needs no GPU for inference on small batches, and zero API cost. It is well-calibrated for coarse object-type classification.

---

### Layer 1: Step-by-Step Triage Logic Gate

```text
FOR each image_path in claim.image_paths:

  STEP 1 — File Guard
    └── Can PIL open and verify the file?
        NO  → flag: blurry_image (or use a generic "invalid_file" note)
            → mark image as INVALID; set valid_image=false
        YES → proceed

  STEP 2 — Resolution Guard
    └── width < 100 OR height < 100?
        YES → flag: cropped_or_obstructed
            → mark image as INVALID
        NO  → proceed

  STEP 3 — Blur Guard  (cv2.Laplacian variance)
    └── variance < threshold (default: 80)?
        YES → flag: blurry_image
            → mark image as INVALID
        NO  → proceed

  STEP 4 — Lighting Guard  (cv2.mean)
    └── mean > 240 → flag: low_light_or_glare
    └── mean < 30  → flag: low_light_or_glare
        EITHER → mark image as INVALID
        NO     → proceed

  STEP 5 — Object-Type Guard  (CLIP cosine similarity)
    └── load text labels for claim.claim_object:
          car     → ["a photo of a car", "damaged car exterior"]
          laptop  → ["a photo of a laptop", "damaged laptop"]
          package → ["a photo of a package", "shipping box"]
    └── cosine_sim(image_emb, best_label_emb) < 0.20?
        YES → flag: wrong_object
            → mark image as INVALID
        NO  → proceed

  STEP 6 — Cropping Guard  (edge density heuristic)
    └── edge_pixel_ratio < 0.01 (nearly blank image)?
        YES → flag: cropped_or_obstructed
            → mark image as INVALID
        NO  → PASS image to Layer 2

AFTER all images evaluated:
  └── valid_images = [images that passed all 6 steps]
  └── Are ALL images INVALID?
        YES → set claim_status = "not_enough_information"
              set valid_image = false
              set evidence_standard_met = false
              set evidence_standard_met_reason = "No valid images passed quality checks"
              AUTO-FAIL: skip Layer 2, write row to output.csv directly
        NO  → PASS claim (with only valid_images) to Layer 2
```

---

## Pillar 2: SOTA Multimodal Reasoning & Prompt Engineering (Layer 2)

### Recommended Model
**`gpt-4o`** (OpenAI) or **`claude-3-5-sonnet`** (Anthropic) with native vision support.

Both support structured JSON output mode and accept base64-encoded images or URLs in a single API call. `gpt-4o` is preferred for its strong object-part detection and schema adherence in JSON mode.

---

### Visual Supremacy Enforcement

The core risk is the model reading a convincing user transcript ("my door is completely crushed") and hallucinating damage confirmation even when the image is ambiguous. Prevent this with a **separation-of-context prompt structure**:

#### System Instruction (set once per domain)

```text
You are a forensic visual evidence analyst for an insurance claim system.
Your ONLY source of truth is the image(s) provided.

VISUAL SUPREMACY RULE:
- The user's chat transcript tells you WHAT to look for.
- The image tells you WHAT IS ACTUALLY THERE.
- These are two different things. Never merge them.

If the user says "my door is scratched" but the image shows no scratch,
the correct claim_status is "contradicted", not "supported".

If the image is ambiguous, the correct claim_status is "not_enough_information".
You must NEVER infer, assume, or guess damage that is not visually present.

Domain context: You are evaluating claims about {claim_object} only.
Object parts valid for this domain: {domain_parts_list}
```

#### User Message Structure

Construct the message as a strict 4-block payload:

```
BLOCK 1 — CLAIM CONTEXT (what to look for, NOT evidence)
---
The claimant describes: "{user_claim}"
Extract the claimed damage type. This is your search hypothesis only.

BLOCK 2 — EVIDENCE REQUIREMENTS (deterministic rule)
---
For a "{claim_object}" with issue family "{matched_issue_family}", the minimum
required visual evidence is:
{evidence_requirements.minimum_image_evidence}

BLOCK 3 — USER RISK PROFILE (contextual signal, not override)
---
This user's historical profile:
- Past claims: {past_claim_count} | Accepted: {accept_claim} | Rejected: {rejected_claim}
- Last 90 days: {last_90_days_claim_count} claims
- History flags: {history_flags}
- Summary: {history_summary}

INSTRUCTION: Use this profile ONLY to populate risk_flags (e.g., user_history_risk).
Do NOT use it to change claim_status. A high-risk user with a valid image still
gets claim_status=supported. A low-risk user with a blank image still gets
claim_status=not_enough_information.

BLOCK 4 — IMAGES (primary evidence, analysed last)
---
[image_1], [image_2], ... (attached as base64 or URL in the message content)

Now analyse the images and produce the JSON output as specified.
```

> **Why this ordering?** By presenting the user history *before* the images but explicitly flagging it as "contextual signal only," and placing images *last* as the explicit "primary evidence," you architecturally prevent the model from rationalizing the transcript using the images.

---

## Pillar 3: Strict Ontological Schema Enforcement

### Design Pattern: Pydantic + OpenAI JSON Mode

Use a **two-level Pydantic schema** — a base output model plus domain-specific discriminated validators.

#### Step 1: Define the Canonical Allowed Values

```python
from enum import Enum
from typing import Literal

class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"

class Severity(str, Enum):
    NONE = "none"; LOW = "low"; MEDIUM = "medium"; HIGH = "high"; UNKNOWN = "unknown"

class IssueType(str, Enum):
    DENT = "dent"; SCRATCH = "scratch"; CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"; BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"; TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"; WATER_DAMAGE = "water_damage"
    STAIN = "stain"; NONE = "none"; UNKNOWN = "unknown"

# Domain-specific object_part enums
CarPart = Literal["front_bumper","rear_bumper","door","hood","windshield",
                  "side_mirror","headlight","taillight","fender",
                  "quarter_panel","body","unknown"]
LaptopPart = Literal["screen","keyboard","trackpad","hinge","lid",
                     "corner","port","base","body","unknown"]
PackagePart = Literal["box","package_corner","package_side","seal",
                      "label","contents","item","unknown"]

VALID_RISK_FLAGS = {
    "none","blurry_image","cropped_or_obstructed","low_light_or_glare",
    "wrong_angle","wrong_object","wrong_object_part","damage_not_visible",
    "claim_mismatch","possible_manipulation","non_original_image",
    "text_instruction_present","user_history_risk","manual_review_required"
}
```

#### Step 2: Domain-Isolated Pydantic Output Models

```python
from pydantic import BaseModel, field_validator, model_validator
from typing import Union

class BaseClaimOutput(BaseModel):
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str                   # semicolon-separated or "none"
    issue_type: IssueType
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: str         # semicolon-separated or "none"
    valid_image: bool
    severity: Severity

    @field_validator("risk_flags")
    @classmethod
    def validate_risk_flags(cls, v):
        flags = [f.strip() for f in v.split(";")]
        invalid = [f for f in flags if f not in VALID_RISK_FLAGS]
        if invalid:
            raise ValueError(f"Invalid risk_flags: {invalid}")
        return v

class CarClaimOutput(BaseClaimOutput):
    object_part: CarPart

class LaptopClaimOutput(BaseClaimOutput):
    object_part: LaptopPart

class PackageClaimOutput(BaseClaimOutput):
    object_part: PackagePart

DOMAIN_MODEL_MAP = {
    "car": CarClaimOutput,
    "laptop": LaptopClaimOutput,
    "package": PackageClaimOutput
}
```

#### Step 3: Enforcement at Call Time

```python
import json

def parse_and_validate(raw_json: str, claim_object: str) -> BaseClaimOutput:
    model_cls = DOMAIN_MODEL_MAP[claim_object]
    data = json.loads(raw_json)
    # Pydantic raises ValidationError on any enum mismatch — catch and retry
    return model_cls(**data)
```

#### Step 4: Retry-on-Validation-Failure Strategy

```python
for attempt in range(3):
    raw = call_vlm_api(prompt, images)
    try:
        result = parse_and_validate(raw, claim.claim_object)
        break
    except ValidationError as e:
        # Append the error back to the prompt: "Your last response had errors: {e}. Correct and retry."
        prompt = inject_correction_feedback(prompt, str(e))
```

> This "correction loop" fixes ~95% of schema violations on the first retry without wasting a full new context.

---

### Domain Isolation via JSON Schema in the Prompt

Inject the domain-specific JSON schema **directly into the system instruction** for each call:

```text
You MUST return a JSON object matching this exact schema:
{
  "evidence_standard_met": boolean,
  "evidence_standard_met_reason": string,
  "risk_flags": "none" | "<flag1>;<flag2>",     # from allowed list only
  "issue_type": "dent" | "scratch" | ... ,       # from allowed list only
  "object_part": "door" | "hood" | "fender" | ..., # CAR PARTS ONLY
  ...
}
Allowed object_part values for claim_object="car":
  front_bumper, rear_bumper, door, hood, windshield, side_mirror,
  headlight, taillight, fender, quarter_panel, body, unknown
DO NOT use values from other domains (e.g., "screen", "keyboard" are INVALID here).
```

---

## Pillar 4: Operational & ROI Pipeline Flow

### End-to-End Claim Lifecycle

```text
[INGESTION]
  1. Load claims.csv into a pandas DataFrame.
  2. Load user_history.csv into a dict keyed by user_id (O(1) lookup).
  3. Load evidence_requirements.csv into a nested dict:
       {claim_object: {issue_family: minimum_image_evidence}}.

[LAYER 1 — GATEKEEPER] (Parallelized via ThreadPoolExecutor, 8–16 workers)
  4. For each claim row:
     a. Split image_paths on ";" → list of local file paths.
     b. Run all 6 Triage Gates (PIL, OpenCV, CLIP) on each image.
     c. Collect: valid_images[], failed_flags[].
  5. Decision:
     • All images failed → AUTO-FAIL: write output row directly. ✓ (No API call)
     • ≥1 valid image    → Forward claim to Layer 2 queue.

[LAYER 2 — VLM CORE] (Rate-limited async queue, see below)
  6. For each queued claim:
     a. Encode valid_images as base64.
     b. Look up user_history by user_id.
     c. Match claim intent → evidence_requirements entry.
     d. Construct 4-Block Prompt (Context / Evidence Req / Risk Profile / Images).
     e. Call VLM API (gpt-4o, JSON mode, max_tokens=800).
     f. Parse raw JSON → parse_and_validate() → Pydantic model.
     g. On ValidationError → inject correction feedback → retry (max 2 retries).
  7. Merge Layer 1 flags (from gatekeeper) into risk_flags of the VLM output.

[OUTPUT ASSEMBLY]
  8. Reconstruct output row with all 14 required columns in exact schema order.
  9. Append to output buffer (list of dicts).
 10. After all claims: pd.DataFrame(buffer).to_csv("output.csv", index=False).

[EVALUATION] (separate run on sample_claims.csv)
 11. Run identical pipeline on sample set.
 12. Compare output vs. expected columns: compute F1, accuracy, exact-match.
 13. Write evaluation/evaluation_report.md.
```

---

### Cost, Latency, and Rate-Limit Optimization

#### Cost Model (per claim)

| Stage | Cost Driver | Est. Cost |
|---|---|---|
| Layer 1 (CLIP + OpenCV) | Local CPU only | **$0.00** |
| Layer 2 (gpt-4o) | ~1000 tok input + 2 images + 300 tok output | **~$0.015–0.025** |
| Auto-failed claims | No API call | **$0.00** |

> If 20–30% of claims auto-fail at Layer 1, you reduce VLM spend by 20–30% directly.

#### Rate-Limit Strategy (TPM/RPM)

```python
import asyncio
from asyncio import Semaphore

# gpt-4o default tier: 500 RPM, 30,000 TPM
# Set conservative limits with headroom:
MAX_CONCURRENT_REQUESTS = 10      # Controls RPM burst
TPM_BUDGET_PER_MINUTE = 25_000   # Leaves 5k buffer

semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

async def safe_vlm_call(prompt, images):
    async with semaphore:
        return await call_vlm_api_async(prompt, images)
```

**Additional strategies:**
- **Token budgeting**: Cap `max_tokens=800` (sufficient for full JSON output). Never use 4096 if 800 works.
- **Image resizing**: Resize images to `≤ 768px` longest-side before base64 encoding. gpt-4o's "low" detail mode halves image tokens while retaining enough resolution for damage classification.
- **Exponential backoff on 429**: Catch `RateLimitError`, wait `2^attempt * 1s`, retry up to 5 times.
- **Result caching**: Hash `(image_path + user_claim + claim_object)` → cache VLM response to disk. Avoids re-processing identical claims during evaluation runs.
- **Batching Layer 1**: Process all images for a claim in a single CLIP batch call (`encode(images_list)`) rather than one-by-one.

#### Latency Profile (per claim)

| Stage | Typical Latency |
|---|---|
| Layer 1 (6 gates, 2 images) | 100–300ms |
| VLM API call (gpt-4o, 2 images) | 2,000–4,000ms |
| Pydantic validation | < 5ms |
| **Total (happy path)** | **~2.5–4.5 seconds** |

With `MAX_CONCURRENT_REQUESTS=10`, a 50-claim test set completes in approximately **15–25 seconds wall-clock** while staying within rate limits.

---

## Summary Decision Table

| Scenario | Layer 1 Result | Layer 2 Called? | Output |
|---|---|---|---|
| All images blurry/invalid | AUTO-FAIL | No | `claim_status=not_enough_information`, `valid_image=false` |
| Wrong object in all images | AUTO-FAIL | No | `claim_status=not_enough_information`, `risk_flags=wrong_object` |
| ≥1 valid image, low-risk user | PASS | Yes | Full VLM-determined output |
| ≥1 valid image, high-risk user | PASS | Yes | Full VLM output + `risk_flags=user_history_risk` |
| VLM returns invalid schema | PASS | Retry (max 2×) | Corrected output or fallback to `unknown` |
