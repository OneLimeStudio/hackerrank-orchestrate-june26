"""Pydantic output models for claim verification — domain-isolated."""

from typing import Literal
from pydantic import BaseModel, field_validator

ClaimStatus = Literal[
    "supported",
    "contradicted",
    "not_enough_information"
]

Severity = Literal[
    "none",
    "low",
    "medium",
    "high",
    "unknown"
]

IssueType = Literal[
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown"
]

# Domain-specific object_part types
CarPart = Literal[
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender",
    "quarter_panel", "body", "unknown"
]
LaptopPart = Literal[
    "screen", "keyboard", "trackpad", "hinge", "lid",
    "corner", "port", "base", "body", "unknown"
]
PackagePart = Literal[
    "box", "package_corner", "package_side", "seal",
    "label", "contents", "item", "unknown"
]

VALID_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required"
}

# Enum string lists for prompt injection
OBJECT_PART_ENUMS = {
    "car": ", ".join(CarPart.__args__),
    "laptop": ", ".join(LaptopPart.__args__),
    "package": ", ".join(PackagePart.__args__),
}

class BaseClaimOutput(BaseModel):
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: IssueType
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: Severity

    @field_validator("risk_flags")
    @classmethod
    def validate_risk_flags(cls, v: str) -> str:
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
    "package": PackageClaimOutput,
}

# --- Self-check ---
if __name__ == "__main__":
    raw = {
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "bumper visible",
        "risk_flags": "none",
        "issue_type": "dent",
        "claim_status": "supported",
        "claim_status_justification": "dent visible in img_1",
        "supporting_image_ids": "img_1",
        "valid_image": True,
        "severity": "medium",
        "object_part": "rear_bumper",
    }
    result = CarClaimOutput(**raw)
    assert result.object_part == "rear_bumper"
    assert result.claim_status == "supported"

    # Cross-domain must fail
    raw["object_part"] = "screen"
    try:
        CarClaimOutput(**raw)
        assert False, "should have rejected laptop part for car"
    except Exception:
        pass

    print("schemas ok")
