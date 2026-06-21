"""Emotion inference from MediaPipe Face Landmarker blendshapes.

MediaPipe's Face Landmarker outputs 52 ARKit-style blendshape scores
(each in the range 0..1). There is no single "emotion" output, so we
derive a small set of coarse emotions from combinations of blendshapes
using transparent, tunable heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass


# Emoji shown in the menu bar for each emotion label.
EMOTION_EMOJI = {
    "happy": "😀",
    "sad": "😢",
    "surprised": "😮",
    "angry": "😠",
    "neutral": "😐",
    "no face": "🫥",
}


@dataclass
class EmotionResult:
    label: str
    score: float  # confidence of the winning emotion, 0..1

    @property
    def emoji(self) -> str:
        return EMOTION_EMOJI.get(self.label, "🙂")

    @property
    def title(self) -> str:
        return f"{self.emoji} {self.label}"


def _get(scores: dict[str, float], *names: str) -> float:
    """Return the max score among the given blendshape names (0 if absent)."""
    return max((scores.get(name, 0.0) for name in names), default=0.0)


def _deadzone(value: float, floor: float) -> float:
    """Subtract a baseline `floor` and rescale the remainder back to 0..1.

    Used for noisy blendshapes that report non-zero values on neutral faces,
    so only activation clearly above the resting baseline counts.
    """
    if value <= floor:
        return 0.0
    return (value - floor) / (1.0 - floor)


def infer_emotion(blendshapes: dict[str, float]) -> EmotionResult:
    """Map a dict of {blendshape_name: score} to a coarse emotion.

    The weights below are simple, hand-tuned linear combinations. They are
    intentionally easy to read and adjust.
    """
    if not blendshapes:
        return EmotionResult("neutral", 0.0)

    smile = _get(blendshapes, "mouthSmileLeft", "mouthSmileRight")
    frown = _get(blendshapes, "mouthFrownLeft", "mouthFrownRight")
    # browDown fires strongly on many neutral faces (brow shape, camera
    # angle, lighting), so subtract a deadzone before using it.
    brow_down = _deadzone(_get(blendshapes, "browDownLeft", "browDownRight"), 0.45)
    brow_inner_up = _get(blendshapes, "browInnerUp")
    jaw_open = _get(blendshapes, "jawOpen")
    eye_wide = _get(blendshapes, "eyeWideLeft", "eyeWideRight")
    sneer = _deadzone(_get(blendshapes, "noseSneerLeft", "noseSneerRight"), 0.20)
    mouth_press = _get(blendshapes, "mouthPressLeft", "mouthPressRight")
    cheek_squint = _get(blendshapes, "cheekSquintLeft", "cheekSquintRight")

    candidates = {
        "happy": 1.10 * smile + 0.30 * cheek_squint,
        "surprised": 0.80 * jaw_open + 0.70 * brow_inner_up + 0.60 * eye_wide,
        "angry": 0.90 * brow_down + 0.60 * sneer + 0.30 * mouth_press,
        "sad": 0.90 * frown + 0.40 * brow_inner_up,
    }

    # Per-emotion thresholds: anger needs strong evidence because its inputs
    # are noisy, while a smile is a reliable signal at lower values.
    min_score = {
        "happy": 0.25,
        "surprised": 0.35,
        "angry": 0.45,
        "sad": 0.30,
    }

    label, raw = max(candidates.items(), key=lambda kv: kv[1])
    score = min(raw, 1.0)

    # Not expressive enough for its category → treat as neutral.
    if score < min_score[label]:
        return EmotionResult("neutral", 1.0 - score)

    return EmotionResult(label, score)
