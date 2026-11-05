import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio.batch import build_dataset
from audio.features import FEATURE_COLUMNS


class BatchTests(unittest.TestCase):
    def test_duplicate_grouping_failures_and_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genre = root / "input" / "classical"
            genre.mkdir(parents=True)
            tone = np.sin(np.arange(22050) * 0.1)
            sf.write(genre / "one.wav", tone, 22050)
            sf.write(genre / "duplicate.wav", tone, 22050)
            (genre / "broken.wav").write_text("broken")
            (genre / "._one.wav").write_text("macOS resource metadata")
            features = dict.fromkeys(FEATURE_COLUMNS, 1.0)
            features["tempo_bpm"] = float("nan")
            # Exercise file hashing, grouping and serialization with a cheap extractor.
            with patch("audio.batch.extract_audio_features", return_value=features):
                audit = build_dataset(root / "input", root / "output")
            self.assertEqual(audit["usable_rows"], 2)
            self.assertEqual(audit["files_found"], 3)
            self.assertEqual(len(audit["failed_files"]), 1)
            self.assertEqual(len(audit["exact_duplicate_pairs"]), 1)
            self.assertEqual(audit["missing_features"]["tempo_bpm"], 2)
            with (root / "output" / "gtzan_features.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["recording_id"], rows[1]["recording_id"])
            self.assertEqual(rows[0]["tempo_bpm"], "")

    def test_empty_folder_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No WAV"):
                build_dataset(directory, Path(directory) / "output")
