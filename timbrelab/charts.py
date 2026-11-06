"""Initial UI exploration chart, independent of the future model figures."""

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ACCENT = "#bdcfa7"
GOLD = "#ebc183"
SURFACE = "#141c18"
HEATMAP = LinearSegmentedColormap.from_list("timbrelab", ["#202c23", "#718e65", "#edc58e"])


def studio_axes():
    fig, ax = plt.subplots(figsize=(7.5, 3.5), layout="constrained", facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors="#c7d1bf")
    ax.xaxis.label.set_color("#c7d1bf")
    ax.yaxis.label.set_color("#c7d1bf")
    ax.title.set_color("#f4efe5")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def exploration_chart(table, column):
    fig, ax = studio_axes()
    values = table[column].dropna()
    if values.empty:
        ax.text(0.5, 0.5, "No values to display", ha="center", color="#f4efe5", transform=ax.transAxes)
    elif values.dtype.kind in "iuf":
        values = values[np.isfinite(values)]
        ax.hist(values, bins=min(20, max(1, values.nunique())), color=ACCENT, edgecolor=SURFACE)
        ax.set_ylabel("Rows")
    else:
        counts = values.astype(str).value_counts().head(15)
        ax.barh(counts.index[::-1], counts.values[::-1], color=ACCENT)
        ax.set_ylabel(column)
        ax.set_xlabel("Rows")
    if values.dtype.kind in "iuf":
        ax.set_xlabel(column)
    ax.set_title(f"{column} · distribution", loc="left", pad=12, color="#f4efe5")
    return fig


def result_chart(task, predictions):
    """Plot the rows returned by the backend for the current task."""
    fig, ax = studio_axes()
    if task == "classification":
        actual = predictions["actual"].astype("string").fillna("(missing)")
        predicted = predictions["predicted"].astype("string").fillna("(missing)")
        counts = pd.concat([actual, predicted]).value_counts()
        labels = counts.index.tolist()
        if len(labels) <= 10:
            matrix = pd.crosstab(actual, predicted).reindex(index=labels, columns=labels, fill_value=0)
            values = matrix.to_numpy()
            fig.set_size_inches(7.5, max(3.5, len(labels) * .46 + 1.5))
            ax.imshow(values, cmap=HEATMAP, aspect="auto", interpolation="nearest")
            ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
            ax.set_yticks(range(len(labels)), labels)
            for row, col in np.ndindex(values.shape):
                color = "#172019" if values[row, col] > values.max() * .57 else "#f4efe5"
                ax.text(col, row, str(values[row, col]), ha="center", va="center", color=color)
            ax.set(xlabel="Predicted label", ylabel="Actual label", title="Where the labels matched")
            explanation = "Read across an actual label to see what the model predicted. The diagonal shows correct predictions."
        else:
            correct = int((actual == predicted).sum())
            ax.barh(["Matched", "Different"], [correct, len(actual) - correct], color=[GOLD, ACCENT])
            ax.set(xlabel="Rows", title="Label matches across all classes")
            explanation = "This summary combines all classes. See the prediction table for individual labels."
    elif task == "regression":
        actual = pd.to_numeric(predictions["actual"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        predicted = pd.to_numeric(predictions["predicted"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(actual) & np.isfinite(predicted)
        if not finite.any():
            plt.close(fig)
            return None, "The chart needs at least one row with two finite numeric values."
        actual, predicted = actual[finite], predicted[finite]
        low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax.plot([low, high], [low, high], color=GOLD, linestyle="--")
        ax.scatter(actual, predicted, color=ACCENT, alpha=.75, s=23)
        ax.set(xlabel="Actual value", ylabel="Predicted value", title="Actual vs. predicted")
        explanation = "Points closer to the gold line are closer to the actual values."
    else:
        counts = predictions["cluster"].astype("string").fillna("(missing)").value_counts()
        shown = counts.head(15).iloc[::-1]
        ax.barh(shown.index, shown.values, color=ACCENT)
        ax.set(xlabel="Rows", ylabel="Cluster", title="Rows per cluster")
        explanation = "The bars show how many rows landed in each group."
        if len(counts) > 15:
            explanation += " The chart shows the 15 largest groups."
    return fig, explanation


def figure_png(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    return buffer.getvalue()
