"""Layer 1: Local CPU gatekeeper — zero API cost pre-filter.

Runs PIL validation, resolution, blur, lighting, and (optional) CLIP checks
BEFORE images reach the VLM. Produces per-image verdicts and aggregated
risk flags for the claim.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

# TUNE AGAINST sample_claims.csv BEFORE TRUSTING THIS THRESHOLD.
# Laplacian variance below this → image flagged as blurry.
BLUR_THRESHOLD = 150.0

# Mean pixel intensity outside this range → lighting flag (soft, not auto-fail).
LIGHTING_LOW = 30
LIGHTING_HIGH = 240

# Minimum resolution: width or height below this → invalid.
MIN_RESOLUTION = 100

# CLIP similarity below this → hard flag "wrong_object".
CLIP_THRESHOLD = 0.20

# Attempt to import open_clip for optional object-match check.
_clip_available = False
try:
    import open_clip
    import torch
    _clip_available = True
except ImportError:
    pass


@dataclass
class ImageVerdict:
    """Result of gatekeeper checks for a single image."""
    image_id: str
    path: str
    valid: bool = True
    flags: List[str] = field(default_factory=list)
    blur_variance: float = -1.0
    mean_brightness: float = -1.0
    clip_score: float = -1.0
    raw_bytes: bytes = b""
    mime_type: str = "image/jpeg"
    error_message: str = ""


@dataclass
class GatekeeperResult:
    """Aggregated gatekeeper result for all images in a claim."""
    verdicts: List[ImageVerdict] = field(default_factory=list)
    all_flags: List[str] = field(default_factory=list)
    valid_image_parts: list = field(default_factory=list)  # types.Part objects for Layer 2
    valid_image_ids: List[str] = field(default_factory=list)
    auto_fail: bool = False


class Gatekeeper:
    """Runs local image quality checks before VLM calls."""

    def __init__(self):
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        if _clip_available:
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k"
                )
                self._clip_model = model.eval()
                self._clip_preprocess = preprocess
                self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
                logging.info("CLIP model loaded for object-match gating.")
            except Exception as e:
                logging.warning(f"CLIP init failed, skipping object-match: {e}")

    def check_claim_images(
        self,
        image_paths: List[str],
        claim_object: str,
        user_id: str,
    ) -> GatekeeperResult:
        """Run all Layer 1 checks on a claim's images."""
        from google.genai import types  # deferred to avoid circular import at module level

        result = GatekeeperResult()
        seen_flags = set()

        for path in image_paths:
            path = path.strip()
            if not path:
                continue

            img_id = os.path.splitext(os.path.basename(path))[0]
            verdict = ImageVerdict(image_id=img_id, path=path)

            # --- (a) PIL verify ---
            try:
                img = Image.open(path)
                img.verify()
            except Exception as e:
                verdict.valid = False
                verdict.flags.append("cropped_or_obstructed")
                verdict.error_message = f"PIL.verify failed: {e}"
                logging.warning(f"[claim={user_id}] IMAGE_INVALID {img_id}: {verdict.error_message}")
                result.verdicts.append(verdict)
                seen_flags.update(verdict.flags)
                continue

            # Re-open after verify (verify leaves the file in an unusable state)
            try:
                img = Image.open(path)
                img.load()
            except Exception as e:
                verdict.valid = False
                verdict.flags.append("cropped_or_obstructed")
                verdict.error_message = f"PIL.load failed: {e}"
                logging.warning(f"[claim={user_id}] IMAGE_INVALID {img_id}: {verdict.error_message}")
                result.verdicts.append(verdict)
                seen_flags.update(verdict.flags)
                continue

            # --- (b) Resolution check ---
            w, h = img.size
            if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
                verdict.valid = False
                verdict.flags.append("cropped_or_obstructed")
                logging.warning(f"[claim={user_id}] IMAGE_INVALID {img_id}: resolution {w}x{h} below minimum {MIN_RESOLUTION}")

            # --- (c) Blur check ---
            try:
                cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                verdict.blur_variance = lap_var
                logging.info(f"[claim={user_id}] {img_id} blur_variance={lap_var:.2f}")
                if lap_var < BLUR_THRESHOLD:
                    verdict.valid = False
                    verdict.flags.append("blurry_image")
                    logging.warning(f"[claim={user_id}] IMAGE_INVALID {img_id}: blur variance {lap_var:.2f} < {BLUR_THRESHOLD}")
            except Exception as e:
                logging.warning(f"[claim={user_id}] Blur check failed for {img_id}: {e}")

            # --- (d) Lighting check (SOFT flag, never auto-fail) ---
            try:
                mean_val = cv2.mean(gray)[0]
                verdict.mean_brightness = mean_val
                if mean_val < LIGHTING_LOW or mean_val > LIGHTING_HIGH:
                    verdict.flags.append("low_light_or_glare")
                    logging.info(f"[claim={user_id}] {img_id} lighting_flag mean={mean_val:.1f}")
            except Exception as e:
                logging.warning(f"[claim={user_id}] Lighting check failed for {img_id}: {e}")

            # --- (e) CLIP object-match check (HARD flag) ---
            if self._clip_model is not None:
                try:
                    clip_img = self._clip_preprocess(img.convert("RGB")).unsqueeze(0)
                    prompts = [
                        f"a photo of a {claim_object}",
                        f"a photo of a damaged {claim_object}",
                        f"a close-up of {claim_object} damage",
                    ]
                    text_tokens = self._clip_tokenizer(prompts)
                    with torch.no_grad():
                        img_features = self._clip_model.encode_image(clip_img)
                        text_features = self._clip_model.encode_text(text_tokens)
                        img_features /= img_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        similarity = (img_features @ text_features.T).max().item()
                    verdict.clip_score = similarity
                    logging.info(f"[claim={user_id}] {img_id} clip_score={similarity:.3f}")
                    if similarity < CLIP_THRESHOLD:
                        verdict.valid = False
                        verdict.flags.append("wrong_object")
                        logging.warning(f"[claim={user_id}] IMAGE_INVALID {img_id}: CLIP score {similarity:.3f} < {CLIP_THRESHOLD} (wrong object)")
                except Exception as e:
                    logging.warning(f"[claim={user_id}] CLIP check failed for {img_id}: {e}")
            else:
                logging.debug(f"[claim={user_id}] {img_id} CLIP check skipped (open_clip not installed)")

            # --- Read raw encoded bytes for Layer 2 ---
            try:
                with open(path, "rb") as f:
                    verdict.raw_bytes = f.read()
                ext = path.lower()
                if ext.endswith(".png"):
                    verdict.mime_type = "image/png"
                elif ext.endswith(".webp"):
                    verdict.mime_type = "image/webp"
                else:
                    verdict.mime_type = "image/jpeg"
            except Exception as e:
                verdict.valid = False
                verdict.error_message = f"Failed to read bytes: {e}"
                logging.warning(f"[claim={user_id}] IMAGE_INVALID {img_id}: {verdict.error_message}")

            result.verdicts.append(verdict)
            seen_flags.update(verdict.flags)

        # --- Aggregate ---
        result.all_flags = sorted(seen_flags)

        for v in result.verdicts:
            if v.valid and v.raw_bytes:
                part = types.Part.from_bytes(data=v.raw_bytes, mime_type=v.mime_type)
                result.valid_image_parts.append(part)
                result.valid_image_ids.append(v.image_id)

        if not result.valid_image_parts and any(v.path for v in result.verdicts):
            result.auto_fail = True

        return result


# --- Self-check ---
if __name__ == "__main__":
    gk = Gatekeeper()
    # Verify the dataclasses instantiate correctly
    v = ImageVerdict(image_id="test", path="fake.jpg")
    assert v.valid is True
    r = GatekeeperResult()
    assert r.auto_fail is False
    print("gatekeeper ok")
