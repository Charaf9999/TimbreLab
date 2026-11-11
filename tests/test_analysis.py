
import unittest
import numpy as np
import pandas as pd

from analysis.clustering import run_clustering


class TestClustering(unittest.TestCase):

    def setUp(self):
        # Petites données de test
        self.df = pd.DataFrame({
            "clip_id": [f"clip_{i}" for i in range(10)],
            "genre": ["classical"] * 5 + ["rock"] * 5,
            "tempo_bpm": [
                60, 62, 65, 67, 70,
                140, 145, 150, 155, 160
            ],
            "rms_mean": [
                0.03, 0.04, 0.05, 0.06, 0.07,
                0.20, 0.22, 0.24, 0.26, 0.28
            ]
        })

        self.features = ["tempo_bpm", "rms_mean"]

    def test_kmeans_execution(self):
        result = run_clustering(self.df, self.features, 2)

        self.assertEqual(len(result["predictions"]), 10)
        self.assertEqual(
            result["predictions"]["cluster"].nunique(), 2
        )

    def test_result_structure(self):
        result = run_clustering(self.df, self.features, 2)

        self.assertEqual(
            set(result.keys()),
            {"pipeline", "metrics", "predictions", "metadata"}
        )

        self.assertIn("silhouette_score", result["metrics"])
        self.assertEqual(
            list(result["predictions"].columns),
            ["row_id", "cluster"]
        )

    def test_missing_values(self):
        df = self.df.copy()
        df.loc[0, "tempo_bpm"] = np.nan

        result = run_clustering(df, self.features, 2)

        self.assertEqual(len(result["predictions"]), 10)

    def test_exclude_genre(self):
        with self.assertRaises(ValueError):
            run_clustering(self.df, ["tempo_bpm", "genre"], 2)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            run_clustering(self.df, self.features, 20)

    def test_unknown_feature(self):
        with self.assertRaises(ValueError):
            run_clustering(self.df, ["unknown_column"], 2)

    def test_reproducibility(self):
        result1 = run_clustering(self.df, self.features, 2)
        result2 = run_clustering(self.df, self.features, 2)

        self.assertEqual(
            result1["predictions"]["cluster"].tolist(),
            result2["predictions"]["cluster"].tolist()
        )


if __name__ == "__main__":
    unittest.main()
