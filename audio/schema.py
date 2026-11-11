"""Charaf's ordered audio_v1 schema, shared by extraction and the interface."""

FEATURE_VERSION = "audio_v1"
FEATURE_COLUMNS = (
    "tempo_bpm", "rms_mean", "centroid_hz_mean",
    *(f"mfcc_{number:02d}_mean" for number in range(1, 14)),
)
