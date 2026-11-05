"""Convert a WAV clip into the agreed audio_v1 feature dictionary."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

FEATURE_VERSION = "audio_v1"
SAMPLE_RATE = 22050
MAX_SECONDS = 30
MIN_SECONDS = 1
N_FFT = 2048
HOP_LENGTH = 512
FEATURE_COLUMNS = [
    "tempo_bpm", "rms_mean", "centroid_hz_mean",
    *[f"mfcc_{number:02d}_mean" for number in range(1, 14)],
]


def extract_audio_features(wav_path):
    """Read up to 30 seconds and return 16 floats; unavailable tempo is NaN.

    WAV clips must contain at least one second of usable audio. Metadata such
    as the genre and recording ID is added by the caller, never inferred here.
    """
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        with sf.SoundFile(path) as audio_file:
            if audio_file.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError(f"Choose a WAV file: {path.name}")
            original_rate = audio_file.samplerate
            samples = audio_file.read(
                frames=original_rate * MAX_SECONDS,
                dtype="float32", always_2d=True,
            )
    except sf.LibsndfileError as error:
        raise ValueError(f"Cannot read WAV file {path.name}: {error}") from error

    if len(samples) < original_rate * MIN_SECONDS:
        raise ValueError(f"{path.name} must contain at least {MIN_SECONDS} second of audio.")
    if not np.isfinite(samples).all():
        raise ValueError(f"{path.name} contains invalid sample values.")

    # Average channels first so every file follows the same mono convention.
    signal = samples.mean(axis=1)
    signal = librosa.resample(signal, orig_sr=original_rate, target_sr=SAMPLE_RATE)
    if np.max(np.abs(signal)) <= 1e-8:
        raise ValueError(f"{path.name} is silent after conversion to mono.")

    rms = librosa.feature.rms(y=signal, frame_length=N_FFT, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(
        y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    mfccs = librosa.feature.mfcc(
        y=signal, sr=SAMPLE_RATE, n_mfcc=13,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )

    # Onset strength measures changes in sound energy across successive frames.
    onsets = librosa.onset.onset_strength(
        y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    tempo = float("nan")
    if np.any(onsets > 1e-6):
        tempo = float(librosa.feature.tempo(
            onset_envelope=onsets, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        )[0])

    values = [tempo, float(rms.mean()), float(centroid.mean())]
    values.extend(float(value) for value in mfccs.mean(axis=1))
    return dict(zip(FEATURE_COLUMNS, values))
