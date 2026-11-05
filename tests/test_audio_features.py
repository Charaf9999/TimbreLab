"""Checks with generated sounds; no music dataset download is needed."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio.features import FEATURE_COLUMNS, extract_audio_features


class AudioFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def write_tone(self, name, frequency=440, rate=22050, seconds=2, stereo=False):
        time = np.arange(int(rate * seconds)) / rate
        signal = 0.25 * np.sin(2 * np.pi * frequency * time)
        if stereo:
            signal = np.column_stack([signal, signal])
        path = self.root / name
        sf.write(path, signal, rate, subtype="FLOAT")
        return path

    def test_schema_and_known_tone_measurements(self):
        values = extract_audio_features(self.write_tone("tone.wav"))
        self.assertEqual(list(values), FEATURE_COLUMNS)
        self.assertEqual(len(values), 16)
        self.assertTrue(all(np.isfinite(values[key]) for key in FEATURE_COLUMNS[1:]))
        self.assertAlmostEqual(values["rms_mean"], 0.25 / np.sqrt(2), delta=0.01)
        self.assertAlmostEqual(values["centroid_hz_mean"], 440, delta=25)

    def test_stereo_resampling_preserves_measurements(self):
        mono = extract_audio_features(self.write_tone("mono.wav"))
        stereo = extract_audio_features(self.write_tone("stereo.wav", rate=44100, stereo=True))
        self.assertAlmostEqual(mono["rms_mean"], stereo["rms_mean"], delta=0.001)
        self.assertAlmostEqual(mono["centroid_hz_mean"], stereo["centroid_hz_mean"], delta=3)

    def test_brighter_tone_has_higher_centroid(self):
        low = extract_audio_features(self.write_tone("low.wav", frequency=220))
        high = extract_audio_features(self.write_tone("high.wav", frequency=2200))
        self.assertGreater(high["centroid_hz_mean"], low["centroid_hz_mean"])

    def test_only_first_thirty_seconds_are_analysed(self):
        rate = 22050
        time = np.arange(rate * 30) / rate
        first = 0.25 * np.sin(2 * np.pi * 440 * time)
        short = self.root / "thirty.wav"
        long = self.root / "long.wav"
        sf.write(short, first, rate, subtype="FLOAT")
        sf.write(long, np.concatenate([first, np.ones(rate)]), rate, subtype="FLOAT")
        np.testing.assert_allclose(
            list(extract_audio_features(short).values()),
            list(extract_audio_features(long).values()), equal_nan=True,
        )

    def test_bad_inputs_have_clear_errors(self):
        silent = self.root / "silent.wav"
        empty = self.root / "empty.wav"
        corrupt = self.root / "broken.wav"
        nonfinite = self.root / "invalid.wav"
        sf.write(silent, np.zeros(22050), 22050)
        sf.write(empty, np.zeros(0), 22050)
        sf.write(nonfinite, np.full(22050, np.nan), 22050, subtype="FLOAT")
        corrupt.write_text("This is not audio.")
        for path in [silent, empty, corrupt, nonfinite, self.write_tone("short.wav", seconds=0.1)]:
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                extract_audio_features(path)
        with self.assertRaises(FileNotFoundError):
            extract_audio_features(self.root / "missing.wav")


if __name__ == "__main__":
    unittest.main()
