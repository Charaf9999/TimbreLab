"""Real handoff upload checks and UI/model-boundary checks (no invented scores)."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from timbrelab.audio_upload import extract_upload, predict_audio, validate_audio_model
from timbrelab.contracts import AUDIO_FEATURES, AUDIO_FEATURE_VERSION
from timbrelab.dataset_audit import matching_audit
from timbrelab.data import read_csv_bytes
from timbrelab.results import make_export_bundle
from test_foundation import TestPipeline, fixture_result

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/examples/gtzan_features.csv"


class RealDatasetTests(unittest.TestCase):
    def make_app(self, content=None):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.file_uploader(key="csv_upload").set_value(("gtzan_features.csv", content or CSV.read_bytes(), "text/csv")).run()
        self.assertEqual(list(app.exception), [])
        return app

    def select(self, app, label):
        return next(element for element in app.selectbox if element.label == label)

    def test_real_upload_counts_audit_and_exact_duplicate_groups(self):
        app = self.make_app()
        self.assertEqual(app.radio(key="data_source").value, "Import a CSV")
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual((metrics["Rows"], metrics["Columns"], metrics["Genres"]), ("999", "19", "10"))
        self.assertEqual(self.select(app, "Explore a column").value, "genre")
        self.assertTrue(any("bad data offset" in text.value for text in app.markdown))
        table = read_csv_bytes(CSV.read_bytes())
        self.assertEqual(table.genre.value_counts().to_dict(), {**dict.fromkeys(
            ["blues", "classical", "country", "disco", "hiphop", "metal", "pop", "reggae", "rock"], 100), "jazz": 99})
        by_id = table.set_index("clip_id")
        audit = matching_audit(CSV.read_bytes())
        self.assertEqual(len(audit["exact_duplicate_pairs"]), 14)
        for pair in audit["exact_duplicate_pairs"]:
            self.assertEqual(by_id.loc[pair["clip_id"], "recording_id"], by_id.loc[pair["same_audio_as"], "recording_id"])
        self.assertIsNone(matching_audit(b"unrelated CSV"))

    def test_real_configuration_handoff_and_pending_workflows(self):
        app = self.make_app()
        self.assertEqual(self.select(app, "Target column").value, "genre")
        self.assertEqual(next(e for e in app.multiselect if e.label == "Numeric predictors").value, list(AUDIO_FEATURES))
        self.assertEqual(self.select(app, "Keep rows from the same recording/group together").value, "recording_id")
        with patch("models.train_supervised", side_effect=NotImplementedError("Basma pending")) as trainer:
            next(b for b in app.button if b.label == "Run analysis").click().run()
            self.assertEqual(trainer.call_args.args[1:], ("classification", "genre", list(AUDIO_FEATURES), "logistic_regression", "recording_id"))
        self.assertNotIn("result", app.session_state)
        next(e for e in app.radio if e.label == "Task").set_value("Regression").run()
        self.assertEqual(self.select(app, "Target column").value, "tempo_bpm")
        next(b for b in app.button if b.label == "Run analysis").click().run()
        self.assertTrue(any("Basma" in msg.value for msg in app.info))
        next(e for e in app.radio if e.label == "Task").set_value("Clustering").run()
        next(b for b in app.button if b.label == "Run analysis").click().run()
        self.assertTrue(any("Layla" in msg.value for msg in app.info))
        self.assertEqual(list(app.exception), [])

    def test_csv_column_order_does_not_change_audio_defaults(self):
        table = read_csv_bytes(CSV.read_bytes())
        shuffled = table.loc[:, list(reversed(table.columns))].to_csv(index=False).encode()
        app = self.make_app(shuffled)
        self.assertEqual(next(e for e in app.multiselect if e.label == "Numeric predictors").value, list(AUDIO_FEATURES))

    def test_upload_replacement_clears_old_training_result(self):
        app = self.make_app()
        with patch("models.train_supervised", side_effect=fixture_result):
            next(b for b in app.button if b.label == "Run analysis").click().run()
        self.assertIn("result", app.session_state)
        app.file_uploader(key="csv_upload").set_value(("changed.csv", b"x,genre\n1,a\n2,a\n3,b\n4,b", "text/csv")).run()
        self.assertNotIn("result", app.session_state)
        self.assertEqual(list(app.exception), [])


class AudioBoundaryTests(unittest.TestCase):
    def result(self):
        return {"pipeline": TestPipeline(), "metrics": {},
                "predictions": pd.DataFrame({"row_id": ["test-only"], "actual": ["jazz"], "predicted": ["jazz"]}),
                "metadata": {"task": "classification", "target": "genre", "model_name": "test_only",
                             "features": list(AUDIO_FEATURES), "settings": {}, "feature_version": AUDIO_FEATURE_VERSION}}

    def test_schema_version_order_and_fitted_pipeline_are_checked(self):
        result = self.result()
        validate_audio_model(result)
        for change in [{"feature_version": "audio_v2"}, {"feature_version": None},
                       {"features": list(reversed(AUDIO_FEATURES))}, {"target": "other"},
                       {"task": "regression"}, {"input": {"synthetic_example": True}}]:
            bad = deepcopy(result)
            bad["metadata"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_audio_model(bad)
        result["pipeline"].feature_names_in_ = list(reversed(AUDIO_FEATURES))
        with self.assertRaisesRegex(ValueError, "pipeline's feature names"):
            validate_audio_model(result)

    def test_export_reload_preserves_prediction_with_test_pipeline(self):
        result = self.result()
        row = read_csv_bytes(CSV.read_bytes()).loc[:0, list(AUDIO_FEATURES)]
        _, files, error = make_export_bundle(result, None)
        self.assertFalse(error)
        restored = joblib.load(BytesIO(files["06-fitted-model.joblib"]))
        self.assertEqual(restored["metadata"], result["metadata"])
        self.assertEqual(predict_audio(restored, row), predict_audio(result, row))
        pd.testing.assert_frame_equal(pd.read_csv(BytesIO(files["02-predictions.csv"])), result["predictions"])

    def test_temp_file_cleanup_on_success_and_failure(self):
        paths = []
        def extractor(path):
            paths.append(path)
            self.assertEqual(path.read_bytes(), b"test-only audio bytes")
            return dict.fromkeys(AUDIO_FEATURES, 1.0)
        with patch("audio.extract_audio_features", side_effect=extractor):
            row = extract_upload(b"test-only audio bytes", "clip.wav")
        self.assertEqual(list(row), list(AUDIO_FEATURES))
        self.assertFalse(paths[0].exists())
        def broken(path):
            paths.append(path)
            raise ValueError("Unreadable WAV")
        with patch("audio.extract_audio_features", side_effect=broken), self.assertRaisesRegex(ValueError, "clip.wav"):
            extract_upload(b"broken", "clip.wav")
        self.assertFalse(paths[-1].exists())

    def test_real_wav_uploader_and_unreadable_file(self):
        import soundfile as sf
        samples = .25 * np.sin(2 * np.pi * 440 * np.arange(44100) / 22050)
        stream = BytesIO()
        sf.write(stream, samples, 22050, format="WAV", subtype="FLOAT")
        content = stream.getvalue()
        expected = extract_upload(content, "generated-tone.wav")
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        app.file_uploader(key="wav_upload").set_value(("generated-tone.wav", content, "audio/wav")).run()
        next(b for b in app.button if b.label == "Extract WAV features").click().run()
        self.assertEqual(list(app.exception), [])
        pd.testing.assert_frame_equal(app.session_state["wav_features"], expected)
        self.assertTrue(next(b for b in app.button if b.label == "Predict WAV genre").disabled)
        app.file_uploader(key="wav_upload").set_value(("unreadable.wav", b"not a WAV", "audio/wav")).run()
        self.assertNotIn("wav_features", app.session_state)
        next(b for b in app.button if b.label == "Extract WAV features").click().run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any("unreadable.wav" in error.value for error in app.error))


if __name__ == "__main__":
    unittest.main()
