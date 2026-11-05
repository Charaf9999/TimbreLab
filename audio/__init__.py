"""Entry point for extracting audio features."""

from timbrelab.contracts import AUDIO_FEATURES, AUDIO_FEATURE_VERSION


def extract_audio_features(wav_path):
    """Return exactly the 16 AUDIO_FEATURES; do not include IDs or the genre."""
    raise NotImplementedError("Audio extraction is not connected yet. Import a CSV of audio features to explore them.")
