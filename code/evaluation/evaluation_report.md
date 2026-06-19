# Evaluation Report — Tiered Cascade Pipeline
# Run: 2026-06-19 20:26:46

## Run Summary
- **Total claims processed**: 44
- **Claims with 0 images attached**: 0
- **Layer 1 auto-fail (skipped VLM)**: 0
- **Layer 1 auto-fail rate**: 0.0%
- **Fallback-default rows**: 0
- **Pydantic retries**: 1

## API Call Reduction
- **Direct VLM baseline calls**: 44 (1 per claim)
- **Cascade VLM calls**: 35
- **API call reduction**: 20.5%

## Anti-Hallucination
- **Hallucinated image IDs stripped**: 18
- **Anti-hallucination claim downgrades**: 16

## Evidence Standard Disagreement
- **Model vs computed disagreements**: 3/38 (7.9%)

## Blur Threshold Distribution
  - Min: 6.9
  - Max: 5748.0
  - Median: 718.1
  - Mean: 1189.3
  - Threshold: 150.0
  - Below threshold: 15/81


## Operational Analysis
- **Runtime**: 224.65s
- **Images processed (Layer 1)**: 81
- **Model calls (Layer 2)**: 35
- **Approx input tokens**: 38,398
- **Approx output tokens**: 5,250
- **Estimated API cost**: $0.00594
- **Pricing**: OpenAI gpt-4.1-nano — $0.10/1M input, $0.40/1M output

## Rate-Limit Handling
Exponential backoff (1s → 2s → 4s → …) with automatic API key rotation across
a pool of keys. Pydantic validation failures trigger a 1-time retry with the
validation error fed back to the model. Layer 1 gatekeeper pre-filters save
API calls for claims with unusable images.

