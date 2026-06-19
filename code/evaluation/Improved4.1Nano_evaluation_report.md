# Evaluation Report — Tiered Cascade Pipeline

## Run Summary
- **Total claims processed**: 20
- **Claims with 0 images attached**: 0
- **Layer 1 auto-fail (skipped VLM)**: 3
- **Layer 1 auto-fail rate**: 15.0%
- **Fallback-default rows**: 0
- **Pydantic retries**: 0

## API Call Reduction
- **Direct VLM baseline calls**: 20 (1 per claim)
- **Cascade VLM calls**: 3
- **API call reduction**: 85.0%

## Anti-Hallucination
- **Hallucinated image IDs stripped**: 5
- **Anti-hallucination claim downgrades**: 6

## Evidence Standard Disagreement
- **Model vs computed disagreements**: 1/15 (6.7%)

## Blur Threshold Distribution
  - Min: 7.8
  - Max: 3306.4
  - Median: 667.2
  - Mean: 822.8
  - Threshold: 150.0
  - Below threshold: 5/29


## Operational Analysis
- **Runtime**: 33.37s
- **Images processed (Layer 1)**: 29
- **Model calls (Layer 2)**: 3
- **Approx input tokens**: 8,982
- **Approx output tokens**: 450
- **Estimated API cost**: $0.00081
- **Pricing**: Gemini 2.5 Flash — $0.075/1M input, $0.30/1M output

## Rate-Limit Handling
Exponential backoff (1s → 2s → 4s → …) with automatic API key rotation across
a pool of keys. Pydantic validation failures trigger a 1-time retry with the
validation error fed back to the model. Layer 1 gatekeeper pre-filters save
API calls for claims with unusable images.
{
  "claim_status_accuracy": 0.35,
  "issue_type_accuracy": 0.5,
  "object_part_accuracy": 0.6,
  "risk_flags_exact_match": 0.3,
  "supporting_image_ids_exact_match": 0.5
}