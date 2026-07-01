"""Unit tests for emotion.py — blendshape → emotion inference."""
from __future__ import annotations

import pytest

from emotion import (
    EMOTION_EMOJI,
    EmotionResult,
    _deadzone,
    _get,
    infer_emotion,
)


class TestEmotionResult:
    def test_emoji_known_label(self) -> None:
        assert EmotionResult("happy", 0.9).emoji == EMOTION_EMOJI["happy"]

    def test_emoji_unknown_label_falls_back(self) -> None:
        assert EmotionResult("confused", 0.5).emoji == "🙂"

    def test_title_combines_emoji_and_label(self) -> None:
        assert EmotionResult("sad", 0.4).title == "😢 sad"


class TestGet:
    def test_returns_max_of_named_scores(self) -> None:
        scores = {"a": 0.2, "b": 0.7, "c": 0.5}
        assert _get(scores, "a", "b") == 0.7

    def test_missing_names_default_to_zero(self) -> None:
        assert _get({"a": 0.3}, "x", "y") == 0.0

    def test_empty_names_returns_zero(self) -> None:
        assert _get({"a": 0.3}) == 0.0


class TestDeadzone:
    def test_value_below_floor_is_zero(self) -> None:
        assert _deadzone(0.3, 0.45) == 0.0

    def test_value_at_floor_is_zero(self) -> None:
        assert _deadzone(0.45, 0.45) == 0.0

    def test_value_above_floor_is_rescaled(self) -> None:
        # (0.6 - 0.2) / (1 - 0.2) == 0.5
        assert _deadzone(0.6, 0.2) == pytest.approx(0.5)

    def test_full_value_maps_to_one(self) -> None:
        assert _deadzone(1.0, 0.45) == pytest.approx(1.0)


class TestInferEmotion:
    def test_empty_blendshapes_is_neutral(self) -> None:
        result = infer_emotion({})
        assert result.label == "neutral"
        assert result.score == 0.0

    def test_strong_smile_is_happy(self) -> None:
        result = infer_emotion({"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9})
        assert result.label == "happy"
        assert result.score > 0.25

    def test_score_is_clamped_to_one(self) -> None:
        # 1.10 * 1.0 would exceed 1.0 without clamping.
        result = infer_emotion({"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0})
        assert result.label == "happy"
        assert result.score == 1.0

    def test_open_mouth_and_wide_eyes_is_surprised(self) -> None:
        result = infer_emotion(
            {"jawOpen": 0.8, "browInnerUp": 0.6, "eyeWideLeft": 0.7, "eyeWideRight": 0.7}
        )
        assert result.label == "surprised"

    def test_brow_down_and_sneer_is_angry(self) -> None:
        result = infer_emotion(
            {
                "browDownLeft": 0.95,
                "browDownRight": 0.95,
                "noseSneerLeft": 0.6,
                "mouthPressLeft": 0.5,
            }
        )
        assert result.label == "angry"

    def test_strong_frown_is_sad(self) -> None:
        result = infer_emotion({"mouthFrownLeft": 0.8, "mouthFrownRight": 0.8})
        assert result.label == "sad"

    def test_weak_signal_falls_back_to_neutral(self) -> None:
        # A tiny smile is below the happy threshold (0.25) → neutral.
        result = infer_emotion({"mouthSmileLeft": 0.05})
        assert result.label == "neutral"
        # Neutral confidence is 1 - winning score.
        assert 0.0 < result.score <= 1.0

    def test_resting_brow_down_does_not_trigger_anger(self) -> None:
        # browDown below the 0.45 deadzone should be ignored.
        result = infer_emotion({"browDownLeft": 0.4, "browDownRight": 0.4})
        assert result.label == "neutral"


class TestSensitivity:
    # A faint smile whose happy score (~0.198) sits between the "high"
    # threshold (0.15) and the "normal" threshold (0.25).
    FAINT_SMILE = {"mouthSmileLeft": 0.18}
    # A modest smile whose happy score (~0.275) clears "normal" (0.25) but not
    # "low" (0.35).
    MODEST_SMILE = {"mouthSmileLeft": 0.25}

    def test_high_sensitivity_detects_faint_smile(self) -> None:
        assert infer_emotion(self.FAINT_SMILE).label == "neutral"
        result = infer_emotion(self.FAINT_SMILE, {"happy": "high"})
        assert result.label == "happy"

    def test_low_sensitivity_rejects_modest_smile(self) -> None:
        assert infer_emotion(self.MODEST_SMILE).label == "happy"
        result = infer_emotion(self.MODEST_SMILE, {"happy": "low"})
        assert result.label == "neutral"

    def test_normal_sensitivity_matches_default(self) -> None:
        assert (
            infer_emotion(self.MODEST_SMILE, {"happy": "normal"}).label
            == infer_emotion(self.MODEST_SMILE).label
        )

    def test_unknown_level_falls_back_to_default(self) -> None:
        result = infer_emotion(self.MODEST_SMILE, {"happy": "bogus"})
        assert result.label == "happy"

    def test_missing_emotion_uses_default(self) -> None:
        # Sensitivity only set for another emotion → happy uses the default.
        result = infer_emotion(self.MODEST_SMILE, {"angry": "low"})
        assert result.label == "happy"

    def test_empty_sensitivity_is_ignored(self) -> None:
        result = infer_emotion(self.MODEST_SMILE, {})
        assert result.label == "happy"

    def test_off_sensitivity_never_detects_emotion(self) -> None:
        # Even a strong smile should be suppressed when "off".
        strong_smile = {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9}
        assert infer_emotion(strong_smile).label == "happy"
        result = infer_emotion(strong_smile, {"happy": "off"})
        assert result.label == "neutral"

    def test_exact_sensitivity_requires_perfect_score(self) -> None:
        # A modest smile clears normal threshold but not exact (1.0).
        assert infer_emotion(self.MODEST_SMILE).label == "happy"
        result = infer_emotion(self.MODEST_SMILE, {"happy": "exact"})
        assert result.label == "neutral"

