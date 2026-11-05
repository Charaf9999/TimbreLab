"""Present a run and build an organized set of files from the same result."""

from io import BytesIO
import json
import pickle
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from timbrelab.charts import figure_png, plt, result_chart
from timbrelab.contracts import MODELS
from timbrelab.studio import section_heading

METRIC_DETAILS = {
    "accuracy": ("Accuracy", "Share of predictions that match the actual label."),
    "macro_f1": ("Macro F1", "Balances precision and recall, giving each class equal weight."),
    "mae": ("Mean absolute error", "Average prediction error in the target's units. Lower is better."),
    "r2": ("R²", "Measures fit relative to predicting the average. Higher is better."),
    "silhouette": ("Silhouette score", "Measures how clearly separated the groups are. Higher is better."),
    "silhouette_score": ("Silhouette score", "Measures how clearly separated the groups are. Higher is better."),
}


def csv_bytes(table):
    return table.to_csv(index=False).encode("utf-8-sig")


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def sample_session():
    """Illustrative rows for a walkthrough of the results and downloads."""
    actual = ["jazz"] * 3 + ["rock"] * 3 + ["classical"] * 3
    predicted = ["jazz", "rock", "jazz", "rock", "rock", "jazz", "classical", "classical", "classical"]
    table = pd.DataFrame({"row_id": [f"sample-{i:02}" for i in range(1, 10)],
                          "actual": actual, "predicted": predicted})
    return {
        "predictions": table,
        "metrics": {"accuracy": float(table["actual"].eq(table["predicted"]).mean())},
        "metadata": {"task": "classification", "model_name": "illustrative_preview",
                     "settings": {}, "sample_preview": True},
    }


def show_results(result):
    section_heading("04 / RESULTS", "Let the data speak.",
                    "The scores, the patterns, and the details behind your session.")
    if result is None:
        st.info("Your results will appear here after you run a session in Configure. "
                "You can open a sample session from Import.")
        return None
    metadata, predictions = result["metadata"], result["predictions"]
    task = metadata["task"]
    if metadata.get("sample_preview"):
        st.info("Sample session · These example predictions are illustrative and separate from your imported data.")
    else:
        model = next((label for label, name in MODELS[task].items() if name == metadata["model_name"]),
                     metadata["model_name"])
        st.success(f"{task.title()} session complete · {model}")
        st.caption(metadata.get("input", {}).get("dataset_name", "Current dataset"))
    metrics = list(result["metrics"].items())
    if metrics:
        for offset in range(0, len(metrics), 3):
            for column, (key, value) in zip(st.columns(3), metrics[offset:offset + 3]):
                label, meaning = METRIC_DETAILS.get(key, (key.replace("_", " ").capitalize(), None))
                if isinstance(value, (float, int, np.number)):
                    display = f"{value:.1%}" if key == "accuracy" else f"{value:.3g}"
                else:
                    display = "—" if value is None else str(value)
                column.metric(label, display, help=meaning)
    figure, explanation = result_chart(task, predictions)
    chart = None
    if figure is not None:
        try:
            chart = figure_png(figure)
        finally:
            plt.close(figure)
        st.image(chart, width="stretch")
    st.caption(explanation)
    st.markdown("**Cluster assignments**" if task == "clustering" else "**Row-by-row predictions**")
    st.caption(f"{len(predictions):,} rows · The CSV contains the full result.")
    st.dataframe(predictions.head(200), hide_index=True, width="stretch", column_config={
        "row_id": "Row ID", "actual": "Actual", "predicted": "Predicted", "cluster": "Cluster",
    })
    prefix = "timbrelab_sample" if metadata.get("sample_preview") else "timbrelab"
    st.download_button("Download predictions · CSV", csv_bytes(predictions),
                       f"{prefix}_predictions.csv", "text/csv", on_click="ignore")
    st.button("Open Export", on_click=lambda: st.session_state.update(studio_stage="Export"),
              icon=":material/arrow_forward:")
    return chart


def make_export_bundle(result, chart):
    metadata = result["metadata"]
    preview = metadata.get("sample_preview", False)
    metrics = pd.DataFrame(list(result["metrics"].items()), columns=["metric", "value"])
    files = {
        "02-predictions.csv": csv_bytes(result["predictions"]),
        "03-metrics.csv": csv_bytes(metrics),
        "05-session-details.json": json_text(metadata).encode("utf-8"),
    }
    if chart:
        files["04-result-chart.png"] = chart
    model_error = False
    if "pipeline" in result:
        try:
            buffer = BytesIO()
            joblib.dump({"pipeline": result["pipeline"], "metadata": metadata}, buffer)
            files["06-fitted-model.joblib"] = buffer.getvalue()
        except (TypeError, ValueError, AttributeError, pickle.PicklingError):
            model_error = True
    summary = [
        "TimbreLab | Sample session" if preview else "TimbreLab | Analysis session",
        "",
        "Illustrative example predictions for learning the interface." if preview
        else f"Dataset: {metadata.get('input', {}).get('dataset_name', 'Current dataset')}",
        f"Task: {metadata['task'].title()}",
        f"Result rows: {len(result['predictions']):,}",
        "",
        "Start with 02-predictions.csv to see the results row by row.",
        "03-metrics.csv contains the scores returned for this session.",
    ]
    if chart:
        summary.append("04-result-chart.png is ready to insert into a report or slides.")
    summary.append("05-session-details.json records the model and its settings.")
    if "06-fitted-model.joblib" in files:
        summary.append("06-fitted-model.joblib contains the fitted Python pipeline and metadata. Only load trusted model files.")
    if model_error:
        summary.append("The fitted model could not be serialized by this backend.")
    files["01-start-here.txt"] = ("\n".join(summary) + "\n").encode("utf-8")
    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zip_file:
        for name, data in sorted(files.items()):
            zip_file.writestr(name, data)
    return archive.getvalue(), files, model_error


def show_exports(result, chart, configuration):
    section_heading("05 / EXPORT", "Take your discoveries with you.",
                    "Save the story of your session, ready for your next idea.")
    if result is None:
        st.info("Run an analysis or preview the sample session to unlock a result pack.")
    else:
        preview = result["metadata"].get("sample_preview", False)
        archive, files, model_error = make_export_bundle(result, chart)
        prefix = "timbrelab_sample_session" if preview else "timbrelab_session"
        with st.container(border=True):
            st.markdown("**Sample session pack**" if preview else "**Complete session pack**")
            st.caption("One ZIP with a Start Here guide, predictions, scores, session details and any available chart or fitted model.")
            st.download_button("Download session pack · ZIP", archive, f"{prefix}.zip", "application/zip",
                               type="primary", icon=":material/download:", on_click="ignore")
        st.markdown("**Download individual files**")
        options = [
            ("02-predictions.csv", "Predictions", "Open the rows in Excel or another spreadsheet.", "text/csv"),
            ("03-metrics.csv", "Scores", "Keep the evaluation numbers for your report.", "text/csv"),
            ("04-result-chart.png", "Chart", "Insert the visualization into slides or a report.", "image/png"),
            ("05-session-details.json", "Session details", "Keep a record of what ran.", "application/json"),
            ("06-fitted-model.joblib", "Fitted model", "Reuse the trained pipeline in Python.", "application/octet-stream"),
        ]
        available = [option for option in options if option[0] in files]
        for offset in range(0, len(available), 3):
            for column, (name, label, description, mime) in zip(st.columns(3), available[offset:offset + 3]):
                with column, st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.caption(description)
                    st.download_button(f"Get {name.rsplit('.', 1)[-1].upper()}", files[name],
                                       f"{prefix}_{name[3:]}", mime, key=f"export_{name}", on_click="ignore")
        if model_error:
            st.warning("This fitted model could not be saved. You can still download the results and scores.")
    if configuration:
        with st.expander("Current analysis setup"):
            st.caption("Save your current choices, including the target and predictors, before running an analysis.")
            st.download_button("Download analysis configuration · JSON", json_text(configuration),
                               "timbrelab_configuration.json", "application/json", on_click="ignore")
