# Evaluation Report

## Performance Metrics (Sample Set)
- **claim_status Accuracy**: 10.00%
- **issue_type Accuracy**: 15.00%
- **object_part Accuracy**: 5.00%
- **risk_flags Exact Match**: 0.00%
- **supporting_image_ids Exact Match**: 10.00%

## Operational Analysis
- **Runtime (Sample Set)**: 1695.31 seconds
- **Images Processed**: 29
- **Model Calls**: ~20 (assuming no retries)
- **Approx Input Tokens**: 17,482
- **Approx Output Tokens**: 3,000
- **Estimated API Cost (Sample)**: $0.00221
- **Estimated Full Run Cost (45 claims)**: $0.00498

## Rate-Limit Handling
The system uses an exponential backoff strategy for HTTP 429 Resource Exhausted errors. 
It sleeps for 1s, doubling the delay up to 5 times. Pydantic validation failures trigger a 1-time retry loop passing the validation error back to the model as feedback. 
