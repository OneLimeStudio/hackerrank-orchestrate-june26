# Evaluation Report — Tiered Cascade Pipeline

## Run Summary
- **Total claims processed**: 20
- **Claims with 0 images attached**: 0
- **Layer 1 auto-fail (skipped VLM)**: 3
- **Layer 1 auto-fail rate**: 15.0%
- **Fallback-default rows**: 2
- **Pydantic retries**: 2

## API Call Reduction
- **Direct VLM baseline calls**: 20 (1 per claim)
- **Cascade VLM calls**: 12
- **API call reduction**: 40.0%

## Anti-Hallucination
- **Hallucinated image IDs stripped**: 3
- **Anti-hallucination claim downgrades**: 2

## Evidence Standard Disagreement
- **Model vs computed disagreements**: 0/15 (0.0%)

## Blur Threshold Distribution
  - Min: 7.8
  - Max: 3306.4
  - Median: 667.2
  - Mean: 822.8
  - Threshold: 80.0
  - Below threshold: 3/29


## Operational Analysis
- **Runtime**: 178.74s
- **Images processed (Layer 1)**: 29
- **Model calls (Layer 2)**: 12
- **Approx input tokens**: 13,482
- **Approx output tokens**: 1,800
- **Estimated API cost**: $0.00155
- **Pricing**: Gemini 2.5 Flash — $0.075/1M input, $0.30/1M output

## Rate-Limit Handling
Exponential backoff (1s → 2s → 4s → …) with automatic API key rotation across
a pool of keys. Pydantic validation failures trigger a 1-time retry with the
validation error fed back to the model. Layer 1 gatekeeper pre-filters save
API calls for claims with unusable images.
