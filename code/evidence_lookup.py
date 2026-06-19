"""Lookup logic for matching user claims to minimum image evidence requirements."""

import pandas as pd


class EvidenceLookup:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def get_evidence_requirement(self, claim_object: str, user_claim: str) -> str:
        """Find the closest evidence requirement for the claim object and user claim."""
        user_claim_lower = str(user_claim).lower()
        
        # Filter for the specific object or 'all'
        object_df = self.df[self.df["claim_object"].isin([claim_object, "all"])]
        
        best_match = None
        
        # Simple keyword matching heuristic
        for _, row in object_df.iterrows():
            applies_to = str(row["applies_to"]).lower()
            
            # General requirements (fallback)
            if applies_to == "general claim review" and best_match is None:
                best_match = row["minimum_image_evidence"]
            
            # Specific domain matching by checking if any word in applies_to is in user_claim
            # Split applies_to into keywords, ignoring filler words
            keywords = [k.strip() for k in applies_to.replace(" or ", ",").replace(" and ", ",").split(",")]
            for keyword in keywords:
                if keyword and keyword in user_claim_lower:
                    # Found a specific match, return immediately
                    return row["minimum_image_evidence"]
        
        # Fallback to general requirement if specific match fails
        return best_match if best_match else "The claimed object and relevant part should be visible clearly enough to inspect the claimed condition."

# --- Self-check ---
if __name__ == "__main__":
    import os
    # Create a dummy csv for self-check
    test_csv = "test_evidence.csv"
    pd.DataFrame({
        "requirement_id": ["1", "2"],
        "claim_object": ["all", "car"],
        "applies_to": ["general claim review", "dent or scratch"],
        "minimum_image_evidence": ["General req", "Dent req"]
    }).to_csv(test_csv, index=False)
    
    lookup = EvidenceLookup(test_csv)
    assert lookup.get_evidence_requirement("car", "I have a big dent") == "Dent req"
    assert lookup.get_evidence_requirement("laptop", "Screen is blank") == "General req"
    
    os.remove(test_csv)
    print("evidence lookup ok")
