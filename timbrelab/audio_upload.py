"""Temporary upload handling and schema-checked prediction, owned by the UI."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import streamlit as st

import audio
from timbrelab.contracts import AUDIO_FEATURES, AUDIO_FEATURE_VERSION


def extract_upload(content, filename):
    """Use Charaf's extractor; close the WAV before reading it on Windows."""
    name = Path(filename).name
    if not name.lower().endswith(".wav"):
        raise ValueError("Choose a WAV file.")
    if not content or len(content) > 20 * 1024 * 1024:
        raise ValueError("Choose a non-empty WAV no larger than 20 MB.")
    with TemporaryDirectory(prefix="timbrelab-wav-") as folder:
        path = Path(folder) / "upload.wav"
        path.write_bytes(content)
        try:
            values = audio.extract_audio_features(path)
        except (ValueError, OSError) as exc:
            message = str(exc).replace(str(path), name).replace(path.name, name)
            raise ValueError(f"Could not analyse {name}. {message}") from exc
    if set(values) != set(AUDIO_FEATURES):
        raise ValueError("The audio extractor returned an incompatible feature schema.")
    row = pd.DataFrame([values], columns=list(AUDIO_FEATURES)).astype(float)
    if not np.isfinite(row.iloc[0, 1:]).all() or np.isinf(row.to_numpy()).any():
        raise ValueError("The WAV produced invalid audio measurements.")
    return row


def validate_audio_model(result):
    """Require the fitted model's own version and order before WAV inference."""
    if not result or result.get("metadata", {}).get("sample_preview"):
        raise ValueError("Train a genre classifier in Configure to enable WAV prediction.")
    metadata = result["metadata"]
    source = metadata.get("input", {})
    if source.get("synthetic_example"):
        raise ValueError("Train on real audio features before predicting a WAV genre.")
    if metadata.get("task") != "classification" or metadata.get("target", source.get("target")) != "genre":
        raise ValueError("WAV prediction needs a classifier whose target is genre.")
    if metadata.get("feature_version") != AUDIO_FEATURE_VERSION:
        raise ValueError("This model has no matching audio_v1 feature version. Retrain with the audio schema.")
    if list(metadata.get("features", [])) != list(AUDIO_FEATURES):
        raise ValueError("This model does not use the 16 audio_v1 predictors in the agreed order.")
    pipeline = result.get("pipeline")
    if not callable(getattr(pipeline, "predict", None)):
        raise ValueError("This result has no fitted prediction pipeline.")
    fitted_names = getattr(pipeline, "feature_names_in_", None)
    if fitted_names is not None and list(fitted_names) != list(AUDIO_FEATURES):
        raise ValueError("The fitted pipeline's feature names differ from its metadata.")
    if getattr(pipeline, "n_features_in_", len(AUDIO_FEATURES)) != len(AUDIO_FEATURES):
        raise ValueError("The fitted pipeline expects a different number of features.")


def predict_audio(result, row):
    validate_audio_model(result)
    if list(row.columns) != list(AUDIO_FEATURES) or len(row) != 1:
        raise ValueError("Prediction requires one ordered audio_v1 feature row.")
    predicted = result["pipeline"].predict(row)
    if len(predicted) != 1:
        raise ValueError("The fitted pipeline did not return one genre.")
    return str(predicted[0])


def show_wav_upload(result):
    """Keep single-WAV extraction and prediction within the existing Results screen."""
    with st.container(border=True):
        st.markdown("**Listen, measure, predict**")
        st.caption("Upload one WAV: mono at 22,050 Hz, up to the first 30 seconds, at least 1 second. "
                   "The same extractor prepares training features and your new clip.")
        uploaded = st.file_uploader("Choose a WAV file", type=["wav"], key="wav_upload")
        content = uploaded.getvalue() if uploaded else None
        fingerprint = sha256(content).hexdigest() if content else None
        if st.session_state.get("wav_fingerprint") != fingerprint:
            st.session_state.pop("wav_features", None)
            st.session_state["wav_fingerprint"] = fingerprint
        if uploaded:
            st.audio(content, format="audio/wav")
        if st.button("Extract WAV features", disabled=not bool(content)):
            st.session_state.pop("wav_features", None)
            try:
                with st.spinner("Measuring the audio… The first run can take longer."):
                    st.session_state["wav_features"] = extract_upload(content, uploaded.name)
            except (ValueError, NotImplementedError) as exc:
                st.error(str(exc))
            except ImportError:
                st.error("Audio dependencies are missing. Install requirements.txt and restart the app.")
            except Exception:
                st.error("This WAV could not be analysed. Try another readable WAV file.")
        row = st.session_state.get("wav_features")
        if row is not None:
            st.dataframe(row.T.rename(columns={0: "Value"}), width="stretch")
            st.caption(f"{AUDIO_FEATURE_VERSION} · 16 ordered predictors. A missing tempo is handled by the fitted pipeline.")
            st.download_button("Download WAV features · CSV", row.to_csv(index=False).encode("utf-8-sig"),
                               "timbrelab_wav_features.csv", "text/csv", on_click="ignore")
        try:
            validate_audio_model(result)
            compatible = True
        except ValueError as exc:
            compatible = False
            st.info(str(exc))
        if st.button("Predict WAV genre", disabled=not compatible or row is None):
            try:
                genre = predict_audio(result, row)
                st.success(f"Predicted genre: {genre}")
                output = row.copy()
                output.insert(0, "predicted", genre)
                output.insert(0, "clip_id", uploaded.name)
                st.download_button("Download WAV prediction · CSV", output.to_csv(index=False).encode("utf-8-sig"),
                                   "timbrelab_wav_prediction.csv", "text/csv", on_click="ignore")
            except Exception:
                st.error("The fitted model could not predict this WAV. Check its preprocessing and feature metadata.")
