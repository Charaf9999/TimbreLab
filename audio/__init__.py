"""Entry point for extracting audio features."""

from audio.schema import FEATURE_COLUMNS as AUDIO_FEATURES, FEATURE_VERSION as AUDIO_FEATURE_VERSION


def extract_audio_features(wav_path):
    """Return exactly the 16 AUDIO_FEATURES; do not include IDs or the genre."""
    # Load signal-processing dependencies only when a WAV is analysed.
    from audio.features import extract_audio_features as extract
    return extract(wav_path)
