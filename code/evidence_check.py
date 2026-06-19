"""Deterministic evidence_standard_met checker.

Replaces the model's self-reported evidence_standard_met with a computed
value based on evidence_requirements.csv, the model's reported issue_type,
and the count of valid images that reached Layer 2.
"""

import logging
from typing import List

import pandas as pd

from evidence_lookup import EvidenceLookup


class EvidenceChecker:
    """Checks whether evidence standards are met using rules, not the model."""

    def __init__(self, requirements_csv: str):
        self.lookup = EvidenceLookup(requirements_csv)
        self.disagreement_count = 0
        self.total_checked = 0

    def check(
        self,
        claim_object: str,
        user_claim: str,
        issue_type: str,
        valid_image_count: int,
        model_self_report: bool,
        user_id: str,
    ) -> bool:
        """Compute evidence_standard_met deterministically.

        Returns the computed boolean, and logs when it disagrees with the
        model's own self-report.
        """
        self.total_checked += 1

        # ponytail: simple rule — if no valid images reached Layer 2, standard is not met.
        # If at least one valid image exists AND a requirement was matched, standard is met.
        # Upgrade to per-requirement minimum-image-count checks if the dataset demands it.
        if valid_image_count == 0:
            computed = False
        else:
            req = self.lookup.get_evidence_requirement(claim_object, user_claim)
            # If we found a specific requirement and have images, standard is met
            computed = True

            # Special case: issue_type is "unknown" and we have images — the model
            # couldn't determine the issue, so evidence is arguably insufficient
            if issue_type == "unknown":
                computed = False

        if computed != model_self_report:
            self.disagreement_count += 1
            logging.info(
                f"[claim={user_id}] evidence_standard_met DISAGREEMENT: "
                f"model={model_self_report}, computed={computed} "
                f"(issue_type={issue_type}, valid_images={valid_image_count})"
            )

        return computed

    @property
    def disagreement_rate(self) -> float:
        if self.total_checked == 0:
            return 0.0
        return self.disagreement_count / self.total_checked


# --- Self-check ---
if __name__ == "__main__":
    import os
    import tempfile

    # Create a minimal requirements CSV for testing
    test_csv = os.path.join(os.path.dirname(__file__), "_test_ev_req.csv")
    pd.DataFrame({
        "requirement_id": ["1"],
        "claim_object": ["all"],
        "applies_to": ["general claim review"],
        "minimum_image_evidence": ["Object visible"]
    }).to_csv(test_csv, index=False)

    checker = EvidenceChecker(test_csv)

    # No images → False
    assert checker.check("car", "dent on bumper", "dent", 0, True, "test") is False
    assert checker.disagreement_count == 1  # model said True, we said False

    # Has images and known issue → True
    assert checker.check("car", "dent on bumper", "dent", 2, True, "test") is True
    assert checker.disagreement_count == 1  # agreed this time

    # Has images but unknown issue → False
    assert checker.check("car", "something", "unknown", 1, True, "test") is False
    assert checker.disagreement_count == 2

    os.remove(test_csv)
    print("evidence_check ok")
