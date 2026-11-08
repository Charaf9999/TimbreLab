
import unittest
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from analysis.visualizations import (
    plot_confusion_matrix,
    plot_regression_predictions
)


class TestSupervisedVisualizations(unittest.TestCase):

    def test_confusion_matrix(self):
        df = pd.DataFrame({
            "actual": ["rock", "jazz", "pop", "rock"],
            "predicted": ["rock", "pop", "pop", "jazz"]
        })

        fig = plot_confusion_matrix(df)

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_regression_plot(self):
        df = pd.DataFrame({
            "actual": [100, 200, 300, 400],
            "predicted": [110, 190, 310, 390]
        })

        fig = plot_regression_predictions(df)

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_columns(self):
        df = pd.DataFrame({
            "actual": ["rock", "jazz"]
        })

        with self.assertRaises(ValueError):
            plot_confusion_matrix(df)

    def test_invalid_regression_values(self):
        df = pd.DataFrame({
            "actual": ["abc", "def"],
            "predicted": [100, 200]
        })

        with self.assertRaises(ValueError):
            plot_regression_predictions(df)


if __name__ == "__main__":
    unittest.main()
