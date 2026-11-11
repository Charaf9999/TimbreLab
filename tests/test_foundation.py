"""Boundary and integration checks; no test model is shipped as an app result."""

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from timbrelab.charts import exploration_chart, figure_png
from timbrelab.contracts import AUDIO_FEATURES, validate_result
from timbrelab.data import (clear_stale_result, feature_exclusions, read_csv_bytes,
                            run_fingerprint, validate_selection)
from timbrelab.results import make_export_bundle, sample_session

ROOT = Path(__file__).resolve().parents[1]


class TestPipeline:
    """Only used to check the UI handoff and bundle serialization."""
    def predict(self, table):
        return np.repeat("jazz", len(table))


def fixture_result(table, task, target, features, model_name, group_column=None):
    actual = ["jazz", "jazz"] if task == "classification" else table[target].iloc[:2].tolist()
    predicted = ["jazz", "rock"] if task == "classification" else [value + 1 for value in actual]
    return {
        "pipeline": TestPipeline(), "metrics": {"accuracy": 0.5} if task == "classification" else {"mae": 1.0},
        "predictions": pd.DataFrame({"row_id": table["clip_id"].iloc[:2].tolist(),
                                      "actual": actual, "predicted": predicted}),
        "metadata": {"task": task, "model_name": model_name, "features": features,
                     "settings": {"random_state": 42}},
    }


class DataTests(unittest.TestCase):
    def test_csv_formats_and_blank_cells(self):
        table = read_csv_bytes("nom;valeur\néchantillon;3\nligne; \n".encode("utf-8-sig"))
        self.assertEqual(table.columns.tolist(), ["nom", "valeur"])
        self.assertTrue(pd.isna(table.iloc[1, 1]))
        western = read_csv_bytes("nom,valeur\ncafé,2\n".encode("cp1252"), encoding="cp1252")
        self.assertEqual(western.iloc[0, 0], "café")

    def test_reject_bad_csv_without_silent_header_repair(self):
        for content in [b"", b"a,a\n1,2", b"a, a \n1,2", b",b\n1,2", b"a,b\n", b"a,b\n1,2,3", b"a,b\n1", b"\x00binary"]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_csv_bytes(content)
        with self.assertRaisesRegex(ValueError, "encoding"):
            read_csv_bytes(b"a\ncaf\xe9")

    def test_leakage_and_invalid_inputs(self):
        table = pd.DataFrame({"row_id": [1, 2, 3, 4], "x": [1., 2., 3., 4.],
                              "empty": [np.nan] * 4, "label": ["a", "a", "b", "b"]})
        excluded = feature_exclusions(table, "label", ["row_id"])
        self.assertEqual(set(excluded), {"label", "row_id", "empty"})
        errors, _ = validate_selection(table, "classification", "label", ["x"], "svm", ["row_id"])
        self.assertEqual(errors, [])
        for features in [["row_id"], ["label"], ["empty"]]:
            self.assertTrue(validate_selection(table, "classification", "label", features, "svm", ["row_id"])[0])
        self.assertTrue(validate_selection(table, "regression", "label", ["x"], "simple_linear", ["row_id"])[0])
        table.loc[0, "x"] = np.inf
        self.assertTrue(validate_selection(table, "clustering", None, ["x"], "kmeans", ["row_id"], k=2)[0])

    def test_regression_predictor_count_and_missing_target(self):
        # Shared backend validation stays covered without a non-music example file.
        table = pd.DataFrame({"clip_id": [1, 2, 3, 4, 5], "tempo_bpm": [80., 90., 100., 110., 120.],
                              "rms_mean": [.1, .2, .3, .4, np.nan], "genre": ["jazz"] * 5})
        errors, notices = validate_selection(table, "regression", "rms_mean", ["tempo_bpm"], "simple_linear", ["clip_id"])
        self.assertEqual(errors, [])
        self.assertTrue(any("no target" in n for n in notices))
        errors, _ = validate_selection(table, "regression", "rms_mean", ["tempo_bpm"], "multiple_linear", ["clip_id"])
        self.assertTrue(errors)
        self.assertTrue(validate_selection(table, "clustering", "genre", ["tempo_bpm"], "kmeans", ["clip_id"], k=len(table))[0])

    def test_fingerprints_and_clearing(self):
        config = {"task": "classification", "features": ["x", "y"]}
        fingerprint = run_fingerprint(b"data", config)
        state = {"result": {"kept": True}, "run_fingerprint": fingerprint}
        clear_stale_result(state, fingerprint)
        self.assertIn("result", state)
        for data, changed in [(b"new data", config), (b"data", {**config, "features": ["y", "x"]}),
                              (b"data", {**config, "task": "regression"})]:
            self.assertNotEqual(fingerprint, run_fingerprint(data, changed))
        clear_stale_result(state, run_fingerprint(b"new data", config))
        self.assertNotIn("result", state)

    def test_audio_fixture_contract(self):
        table = read_csv_bytes((ROOT / "data/examples/audio_fixture.csv").read_bytes())
        self.assertEqual(list(table.columns[3:]), list(AUDIO_FEATURES))
        self.assertEqual(len(AUDIO_FEATURES), 16)
        self.assertTrue(table.clip_id.is_unique)
        self.assertTrue((table.groupby("recording_id").size() == 2).all())

    def test_result_contract_and_saved_bundle(self):
        table = read_csv_bytes((ROOT / "data/examples/audio_fixture.csv").read_bytes())
        result = fixture_result(table, "classification", "genre", list(AUDIO_FEATURES), "svm")
        validate_result(result, "classification", list(AUDIO_FEATURES), "svm")
        with self.assertRaises(ValueError):
            validate_result(result, "classification", list(AUDIO_FEATURES), "knn")
        buffer = BytesIO()
        joblib.dump({"pipeline": result["pipeline"], "metadata": result["metadata"]}, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_array_equal(restored["pipeline"].predict(table), result["pipeline"].predict(table))
        result["metadata"]["features"] = ["wrong"]
        with self.assertRaises(ValueError):
            validate_result(result, "classification", list(AUDIO_FEATURES))

    def test_png_export_opens(self):
        from PIL import Image
        import matplotlib.pyplot as plt
        figure = exploration_chart(pd.DataFrame({"genre": ["jazz", "rock", "jazz"]}), "genre")
        png = figure_png(figure)
        plt.close(figure)
        with Image.open(BytesIO(png)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, 500)


class AppTests(unittest.TestCase):
    def make_app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def select(self, app, label):
        return next(element for element in app.selectbox if element.label == label)

    def test_default_and_training(self):
        app = self.make_app()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(self.select(app, "Target column").value, "genre")
        features = next(e for e in app.multiselect if e.label == "Numeric predictors")
        self.assertEqual(features.value, list(AUDIO_FEATURES))
        self.assertFalse(next(b for b in app.button if b.label == "Run analysis").disabled)
        next(b for b in app.button if b.label == "Run analysis").click().run()
        self.assertEqual(list(app.exception), [])
        self.assertIn("result", app.session_state)

    def test_music_data_sources_and_classifiers(self):
        app = self.make_app()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.radio(key="data_source").options, ["Music genres", "Import a CSV"])
        self.assertEqual(self.select(app, "Target column").value, "genre")
        self.assertEqual(self.select(app, "Model").options, [
            "Logistic regression", "Decision tree", "Gaussian Naive Bayes", "Support vector machine",
            "K-nearest neighbours", "Random Forest", "Neural network (MLP)",
        ])
        self.select(app, "Model").set_value("Random Forest").run()
        with patch("models.train_supervised", side_effect=fixture_result) as trainer:
            next(b for b in app.button if b.label == "Run analysis").click().run()
            self.assertEqual(trainer.call_args.args[1:3], ("classification", "genre"))
            self.assertEqual(trainer.call_args.args[4], "random_forest")
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.session_state["result"]["metadata"]["input"]["task"], "classification")

    def test_music_regression_and_clustering_navigation(self):
        app = self.make_app()
        next(e for e in app.radio if e.label == "Task").set_value("Regression").run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(self.select(app, "Target column").value, "tempo_bpm")
        self.assertNotIn("genre", self.select(app, "Target column").options)
        self.assertEqual(self.select(app, "Model").options, [
            "Simple linear regression", "Multiple linear regression", "Polynomial regression",
        ])
        features = next(e for e in app.multiselect if e.label == "Numeric predictors")
        self.assertEqual(features.value, ["rms_mean"])
        for excluded in ["tempo_bpm", "genre", "clip_id", "recording_id"]:
            self.assertNotIn(excluded, features.options)
        self.select(app, "Model").set_value("Multiple linear regression").run()
        self.assertEqual(next(e for e in app.multiselect if e.label == "Numeric predictors").value,
                         list(AUDIO_FEATURES[1:]))
        self.assertEqual(self.select(app, "Keep rows from the same recording/group together").value, "recording_id")
        next(e for e in app.radio if e.label == "Task").set_value("Clustering").run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(self.select(app, "Model").value, "K-means")
        self.assertEqual(self.select(app, "Known label to exclude (optional)").value, "genre")
        self.assertEqual(next(e for e in app.multiselect if e.label == "Numeric predictors").value, list(AUDIO_FEATURES))

    def test_music_regression_backend_export_and_target_change(self):
        app = self.make_app()
        next(e for e in app.radio if e.label == "Task").set_value("Regression").run()
        with patch("models.train_supervised", side_effect=fixture_result) as trainer:
            next(b for b in app.button if b.label == "Run analysis").click().run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(trainer.call_args.args[1:],
                             ("regression", "tempo_bpm", ["rms_mean"], "simple_linear", "recording_id"))
        result = app.session_state["result"]
        self.assertEqual(result["metadata"]["input"]["target"], "tempo_bpm")
        self.assertTrue(any(b.label == "Download session pack · ZIP" for b in app.get("download_button")))
        bundle, _, _ = make_export_bundle(result, None)
        with ZipFile(BytesIO(bundle)) as archive:
            rows = pd.read_csv(BytesIO(archive.read("02-predictions.csv")))
            self.assertEqual(rows["actual"].tolist(), [75, 76])
        self.select(app, "Target column").set_value("centroid_hz_mean").run()
        self.assertEqual(list(app.exception), [])
        self.assertNotIn("result", app.session_state)
        self.assertNotIn("centroid_hz_mean", next(e for e in app.multiselect if e.label == "Numeric predictors").options)

    def test_regression_requires_a_numeric_audio_target(self):
        table = pd.DataFrame({"clip_id": [1, 2, 3, 4], "genre": [0, 0, 1, 1]})
        with patch("timbrelab.data.read_csv_bytes", return_value=table):
            app = self.make_app()
            next(e for e in app.radio if e.label == "Task").set_value("Regression").run()
        self.assertEqual(list(app.exception), [])
        self.assertFalse(any(e.label == "Target column" for e in app.selectbox))
        self.assertTrue(next(b for b in app.button if b.label == "Run analysis").disabled)
        self.assertTrue(any("no numeric target columns" in e.value for e in app.warning))

    def test_connected_backend_and_clear_on_model_change(self):
        app = self.make_app()
        with patch("models.train_supervised", side_effect=fixture_result) as trainer:
            next(b for b in app.button if b.label == "Run analysis").click().run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(trainer.call_args.args[-1], "recording_id")
            self.assertIn("result", app.session_state)
            self.assertTrue(any(e.label == "Download predictions · CSV" for e in app.get("download_button")))
        self.select(app, "Model").set_value("Decision tree").run()
        self.assertEqual(list(app.exception), [])
        self.assertNotIn("result", app.session_state)

    def test_import_empty_state_clears_results(self):
        app = self.make_app()
        app.session_state["result"] = {"old": True}
        app.radio(key="data_source").set_value("Import a CSV").run()
        self.assertEqual(list(app.exception), [])
        self.assertNotIn("result", app.session_state)
        self.assertTrue(any("Choose a CSV to import" in e.value for e in app.info))

    def test_sample_session_and_zip_keep_preview_separate(self):
        app = self.make_app()
        next(b for b in app.button if b.label == "Preview sample results").click().run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.session_state["studio_stage"], "Results")
        self.assertNotIn("result", app.session_state)
        self.assertTrue(any(b.label == "Download session pack · ZIP" for b in app.get("download_button")))

        bundle, _, _ = make_export_bundle(sample_session(), None)
        with ZipFile(BytesIO(bundle)) as archive:
            self.assertIn(b"Illustrative", archive.read("01-start-here.txt"))
            self.assertNotIn("06-fitted-model.joblib", archive.namelist())
            rows = pd.read_csv(BytesIO(archive.read("02-predictions.csv")))
            self.assertEqual(len(rows), 9)


if __name__ == "__main__":
    unittest.main()
