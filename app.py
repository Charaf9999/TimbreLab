from hashlib import sha256
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import analysis
import models
from timbrelab.charts import exploration_chart, figure_png, plt
from timbrelab.contracts import AUDIO_FEATURES, MODELS, validate_result
from timbrelab.data import (
    clear_stale_result, feature_exclusions, identifier_columns,
    numeric_columns, read_csv_bytes, run_fingerprint, validate_selection,
)
from timbrelab.results import csv_bytes, sample_session, show_exports, show_results
from timbrelab.studio import render_studio_header, section_heading, show_footer

ROOT = Path(__file__).resolve().parent
EXAMPLES = {
    "Music genres": ("audio_fixture.csv", "genre"),
}
TASKS = {"Classification": "classification", "Regression": "regression", "Clustering": "clustering"}
IMPORT_CSV = "Import a CSV"
STAGES = ["Import", "Explore", "Configure", "Results", "Export"]


def go_to(stage):
    """Select a workspace tab through Streamlit's per-session state."""
    st.session_state["studio_stage"] = stage


def preview_sample():
    """Open illustrative music predictions without replacing a trained result."""
    st.session_state["preview_sample"] = True
    go_to(STAGES[3])


def show_header():
    """Display the illustrated studio header and ambient-motion preference."""
    render_studio_header()


def select_imported_file():
    """Make an uploaded CSV the active data source."""
    st.session_state["data_source"] = IMPORT_CSV


def import_dataset():
    """Render CSV controls and return the parsed table and source details, or None."""
    section_heading("01 / IMPORT", "Your next discovery starts here.",
                    "Bring your music feature table, or take the studio for a spin with our example.")
    left, right = st.columns([1.5, 1], gap="large")
    with left, st.container(border=True, key="import-upload"):
        st.html('<p class="card-kicker">YOUR COLLECTION</p>')
        st.markdown("**CSV from your computer**")
        st.caption("Up to 20 MB. Include column names in the first row.")
        imported = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload",
                                    on_change=select_imported_file)
        with st.expander("CSV import options"):
            separator = st.selectbox("Separator", ["Auto", "Comma", "Semicolon", "Tab"])
            encoding = st.selectbox("Encoding", ["utf-8-sig", "cp1252"],
                                    format_func={"utf-8-sig": "UTF-8", "cp1252": "Windows-1252"}.get)
    with right, st.container(border=True, key="import-example"):
        st.html('<p class="card-kicker">TAKE IT FOR A SPIN</p>')
        st.markdown("**Data source**")
        source = st.radio("Data source", [*EXAMPLES, IMPORT_CSV], key="data_source", label_visibility="collapsed")
        st.caption("Music genres is a synthetic example of music feature data.")
        st.button("Preview sample results", on_click=preview_sample, width="stretch",
                  icon=":material/play_arrow:")
    suggested_target = None
    if source in EXAMPLES:
        filename, suggested_target = EXAMPLES[source]
        content = (ROOT / "data" / "examples" / filename).read_bytes()
        separator, encoding = "Auto", "utf-8-sig"
    else:
        filename = imported.name if imported else ""
        content = imported.getvalue() if imported else None
    if content is None:
        st.info("Choose a CSV to import, or select a built-in example.")
        return None
    try:
        table = read_csv_bytes(content, separator, encoding)
    except ValueError as exc:
        st.error(f"Could not import {filename}. {exc}")
        return None
    st.caption(f"{'Example dataset' if source in EXAMPLES else 'Imported'} · {filename} · "
               f"{len(table):,} rows · {len(table.columns):,} columns")
    st.button("Explore dataset", type="primary", on_click=go_to, args=(STAGES[1],),
              icon=":material/arrow_forward:")
    return table, content, filename, source, suggested_target, separator, encoding


def main():
    """Build the five-step music workspace and hand validated inputs to a backend."""
    st.set_page_config(page_title="TimbreLab Studio", page_icon="🎛️", layout="wide")
    show_header()
    next_stage = st.session_state.pop("next_stage", None)
    if next_stage:
        go_to(next_stage)
    import_tab, explore, configure, results, exports = st.tabs(STAGES, key="studio_stage", on_change="rerun")
    with import_tab:
        dataset = import_dataset()
    if dataset is None:
        st.session_state.pop("result", None)
        with explore:
            st.info("Import a dataset first to see its rows and charts.")
        with configure:
            st.info("Import a dataset first to choose an analysis.")
        show_session(results, exports, None)
        return
    table, content, filename, source, suggested_target, separator, encoding = dataset
    missing = int(table.isna().sum().sum())
    # Dataset-specific keys prevent selections from a previous CSV leaking into a new one.
    dataset_key = run_fingerprint(content, {"separator": separator, "encoding": encoding})[:16]

    with explore:
        section_heading("02 / EXPLORE", "Look a little closer.",
                        "Every feature has a story. Explore the shape of your music.")
        st.caption(filename)
        for col, label, value in zip(st.columns(4), ["Rows", "Columns", "Missing cells", "Duplicate rows"],
                                     [len(table), len(table.columns), missing, int(table.duplicated().sum())]):
            col.metric(label, f"{value:,}")
        st.dataframe(table.head(200), hide_index=True, width="stretch")
        st.caption(f"Showing the first {min(200, len(table)):,} of {len(table):,} rows. Downloads include the full table.")
        st.download_button("Download current table · CSV", csv_bytes(table),
                           "timbrelab_data.csv", "text/csv")
        left, right = st.columns([1, 1.6])
        with left:
            st.markdown("**Column health**")
            profile = pd.DataFrame({"Column": table.columns,
                                    "Type": [str(table[c].dtype) for c in table],
                                    "Missing": table.isna().sum().values,
                                    "Distinct": table.nunique().values})
            st.dataframe(profile, hide_index=True, width="stretch")
            if missing:
                st.caption("Missing values are included in the counts above.")
        with right:
            chart_default = list(table.columns).index(suggested_target) if suggested_target in table.columns else 0
            column = st.selectbox("Explore a column", list(table.columns), index=chart_default, key=f"chart_{dataset_key}")
            fig = exploration_chart(table, column)
            try:
                chart = figure_png(fig)
            finally:
                plt.close(fig)
            st.image(chart, width="stretch")
            st.download_button("Download this chart · PNG", chart, "timbrelab_distribution.png", "image/png", on_click="ignore")
            if not pd.api.types.is_numeric_dtype(table[column]) and table[column].nunique() > 15:
                st.caption("Showing the 15 most frequent values.")
        with st.expander("Numeric summary"):
            numeric = table.select_dtypes(include="number")
            if numeric.empty:
                st.info("This table has no numeric columns. Import numeric predictors to configure an analysis.")
            else:
                st.dataframe(numeric.replace([np.inf, -np.inf], np.nan).describe().T, width="stretch")
        st.button("Configure analysis", on_click=go_to, args=(STAGES[2],), icon=":material/tune:")

    with configure:
        section_heading("03 / CONFIGURE", "Shape your experiment.")
        st.caption("Choose a music analysis task, then select the audio features to use.")
        task_label = st.radio("Task", list(TASKS), horizontal=True, key=f"task_{dataset_key}")
        task = TASKS[task_label]
        st.caption({"classification": "Predict a music genre from numeric audio features.",
                    "regression": "Predict a numeric audio measurement, such as tempo in beats per minute (BPM).",
                    "clustering": "Find groups of clips with similar audio features, without using genre labels."}[task])
        detected_ids = identifier_columns(table)
        identifiers = st.multiselect("Identifier columns to exclude", list(table.columns), default=detected_ids,
                                     key=f"ids_{dataset_key}",
                                     help="Keep IDs out of the model. Add any identifier whose name does not end in _id.")
        # Recognised IDs are always excluded, even if removed from the extra-ID selector.
        identifiers = sorted(set(identifiers) | set(detected_ids))
        if detected_ids:
            st.caption("Recognised IDs are always excluded: " + ", ".join(detected_ids))
        possible_targets = [c for c in table if c not in identifiers]
        if task == "regression":
            possible_targets = [c for c in numeric_columns(table) if c not in identifiers and c != "genre"]
        preferred_target = "tempo_bpm" if task == "regression" else suggested_target or "genre"
        if task == "clustering":
            targets = ["(No label)"] + possible_targets
            default = targets.index(preferred_target) if preferred_target in targets else 0
            label = st.selectbox("Known label to exclude (optional)", targets, index=default,
                                 key=f"label_{dataset_key}",
                                 help="This column will not be used to form clusters.")
            target = None if label == "(No label)" else label
        elif possible_targets:
            default = possible_targets.index(preferred_target) if preferred_target in possible_targets else 0
            target = st.selectbox("Target column", possible_targets, index=default, key=f"target_{dataset_key}_{task}")
        else:
            target = None
            st.warning("There are no numeric target columns after excluding IDs." if task == "regression"
                       else "There are no target columns after excluding IDs.")
        if task == "regression":
            st.caption("The selected target is excluded from predictors. Genre labels and IDs are also excluded.")
        exclusions = feature_exclusions(table, target, identifiers)
        if task == "regression" and "genre" in table:
            exclusions["genre"] = "Genre label: excluded from audio regression"
        available = [c for c in table if c not in exclusions]
        model_label = st.selectbox("Model", list(MODELS[task]), key=f"model_{dataset_key}_{task}")
        model_name = MODELS[task][model_label]
        # Target changes also reset feature defaults so an old target is never carried over.
        feature_key = f"features_{dataset_key}_{task}_{target}_{model_name}_{','.join(available)}"
        defaults = available[:1] if model_name in {"simple_linear", "polynomial"} else available
        features = st.multiselect("Numeric predictors", available, default=defaults, key=feature_key)
        k, group_column = 3, None
        if task == "clustering":
            k = int(st.number_input("Number of clusters", min_value=2, value=3, step=1, key=f"k_{dataset_key}"))
        else:
            groups = ["(None)"] + [c for c in table if c != target and c not in features]
            default = groups.index("recording_id") if "recording_id" in groups else 0
            group = st.selectbox("Keep rows from the same recording/group together", groups, index=default,
                                key=f"group_{dataset_key}_{task}_{target}_{','.join(features)}")
            group_column = None if group == "(None)" else group
            if "recording_id" in table and group_column != "recording_id":
                st.warning("Music excerpts should be split by their original recording. Select recording_id for audio evaluation.")
        errors, notices = validate_selection(table, task, target, features, model_name, identifiers, group_column, k)
        audio_schema = task == "classification" and target == "genre" and features == list(AUDIO_FEATURES)
        if audio_schema and "recording_id" in table and group_column != "recording_id":
            errors.append("Select recording_id to keep excerpts of each recording together.")
        settings = {"task": task, "target": target, "features": features, "model_name": model_name,
                    "identifiers": identifiers, "group_column": group_column, "k": k if task == "clustering" else None,
                    "separator": separator, "encoding": encoding}
        clear_stale_result(st.session_state, run_fingerprint(content, {**settings, "source": source, "filename": filename}))
        for message in notices:
            st.info(message)
        for message in errors:
            st.error(message)
        with st.expander("Excluded columns and reasons"):
            st.dataframe(pd.DataFrame([{"Column": c, "Reason": r} for c, r in exclusions.items()],
                                      columns=["Column", "Reason"]), hide_index=True, width="stretch")
        configuration = {**settings, "dataset_sha256": sha256(content).hexdigest(),
                         "source": source, "dataset_name": filename, "synthetic_example": source in EXAMPLES,
                         "row_count": len(table)}
        if st.button("Run analysis", type="primary", disabled=bool(errors), icon=":material/graphic_eq:"):
            st.session_state.pop("result", None)
            try:
                with st.spinner("Running analysis…"):
                    if task == "clustering":
                        result = analysis.run_clustering(table.copy(), features, k)
                    else:
                        result = models.train_supervised(table.copy(), task, target, features, model_name, group_column)
                    validate_result(result, task, features, model_name)
                    result["metadata"] = {**result["metadata"], "input": configuration}
                    st.session_state["result"] = result
                st.session_state["preview_sample"] = False
                st.session_state["next_stage"] = STAGES[3]
                st.rerun()
            except NotImplementedError as exc:
                st.info(str(exc))
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                st.error("The analysis could not complete. Check the dataset and model settings, then try again.")
                logging.getLogger(__name__).exception("Analysis module failed")
    show_session(results, exports, configuration)


def show_session(results, exports, configuration):
    """Display the sample preview or trained result and its matching downloads."""
    preview = st.session_state.get("preview_sample", False)
    result = sample_session() if preview else st.session_state.get("result")
    with results:
        chart = show_results(result)
        if preview:
            st.button("Close sample preview", on_click=lambda: st.session_state.update(preview_sample=False))
    with exports:
        show_exports(result, chart, configuration)
    show_footer()


if __name__ == "__main__":
    main()
