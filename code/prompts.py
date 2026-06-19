"""Prompt templates for the VLM verification core."""

SYSTEM_PROMPT = """You are a forensic visual evidence analyst for an insurance damage claim system.
Your ONLY source of truth is the image(s) provided. The user's claim text tells you what
to look FOR, not what IS there. If the claim and the images disagree, the images win.

Return exactly one JSON object. No arrays, no markdown fences, no trailing text.

Valid object_part values for this claim: {object_part_enum}
Do not use object_part values from any other domain.

Rules:
- supporting_image_ids must only reference image IDs actually provided to you.
- Use "unknown" when the issue/part cannot be determined from the images.
- Use issue_type="none" only if the relevant part IS visible and shows no damage.
- A convincing claim text with no visible matching damage = "contradicted", not "supported".
- Ambiguous, missing, or unusable images = "not_enough_information", not a guess.
- If the images do not meet the minimum_image_evidence standard given below,
  set evidence_standard_met=false and explain why in evidence_standard_met_reason.
"""

USER_PROMPT_TEMPLATE = """CLAIM OBJECT: {claim_object}

USER CLAIM TRANSCRIPT (hypothesis only, NOT evidence):
\"\"\"{user_claim}\"\"\"

MINIMUM EVIDENCE REQUIRED for this object/issue family:
{minimum_image_evidence}

USER HISTORY (context for risk_flags ONLY — never use this to change claim_status):
past_claims={past_claim_count}, accepted={accept_claim}, rejected={rejected_claim},
last_90_days={last_90_days_claim_count}, flags={history_flags}, summary="{history_summary}"

IMAGES PROVIDED: {image_id_list}
Analyze the images below. They are the only evidence that matters."""

# --- Self-check ---
if __name__ == "__main__":
    sp = SYSTEM_PROMPT.format(object_part_enum="door, hood")
    assert "door, hood" in sp
    
    up = USER_PROMPT_TEMPLATE.format(
        claim_object="car",
        user_claim="Test",
        minimum_image_evidence="Req",
        past_claim_count=1,
        accept_claim=1,
        rejected_claim=0,
        last_90_days_claim_count=0,
        history_flags="none",
        history_summary="Ok",
        image_id_list="img_1",
    )
    assert "car" in up
    assert "img_1" in up
    print("prompts ok")
