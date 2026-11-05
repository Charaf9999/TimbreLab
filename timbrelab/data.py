"""CSV parsing, safe defaults and validation, independent of Streamlit."""

import csv
from hashlib import sha256
from io import StringIO
import json

import numpy as np
import pandas as pd

MAX_BYTES = 20 * 1024 * 1024


def read_csv_bytes(content, separator="Auto", encoding="utf-8-sig"):
    if not content:
        raise ValueError("The CSV is empty. Choose a file containing column names and data rows.")
    if len(content) > MAX_BYTES:
        raise ValueError("Choose a CSV smaller than 20 MB.")
    try:
        text = content.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError("Cannot read this encoding. Try Windows-1252 in the CSV options.") from exc
    if "\x00" in text:
        raise ValueError("This does not look like a text CSV. Export the table as CSV first.")
    try:
        if separator == "Auto":
            try:
                delimiter = csv.Sniffer().sniff(text[:65536], delimiters=",;\t").delimiter
            except csv.Error:
                delimiter = ","
        else:
            delimiter = {"Comma": ",", "Semicolon": ";", "Tab": "\t"}[separator]
        reader = csv.reader(StringIO(text), delimiter=delimiter, strict=True)
        header = next(reader)
        header = [name.strip() for name in header]
        if not header or any(not name for name in header):
            raise ValueError("Every column needs a name in the first row.")
        if len(set(header)) != len(header):
            raise ValueError("Column names must be unique, including after removing surrounding spaces.")
        for line, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise ValueError(f"CSV row {line} has {len(row)} values; expected {len(header)}. Check the separator and quoting.")
        table = pd.read_csv(StringIO(text), sep=delimiter, header=0)
        table.columns = header
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, StopIteration) as exc:
        raise ValueError("Cannot read this CSV. Check the separator, header and number of values per row.") from exc
    if table.empty:
        raise ValueError("This CSV has column names but no data rows.")
    if len(table) > 100_000 or len(table.columns) > 200:
        raise ValueError("For this version, use at most 100,000 rows and 200 columns.")
    # Treat blank strings as missing without changing category spelling or IDs.
    table = table.replace(r"^\s*$", np.nan, regex=True)
    return table


def numeric_columns(table):
    return [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])
            and not pd.api.types.is_bool_dtype(table[c])]


def identifier_columns(table):
    return [c for c in table.columns if c.lower() in {"id", "row_id", "clip_id", "recording_id"}
            or c.lower().endswith("_id")]


def feature_exclusions(table, target, identifiers):
    numeric = numeric_columns(table)
    reasons = {}
    for column in table.columns:
        if column == target:
            reasons[column] = "Target: never a predictor"
        elif column in identifiers:
            reasons[column] = "Identifier: excluded from predictors"
        elif column not in numeric:
            reasons[column] = "Non-numeric: encoding is outside this first version"
        elif table[column].replace([np.inf, -np.inf], np.nan).isna().all():
            reasons[column] = "No finite numeric values"
    return reasons


def validate_selection(table, task, target, features, model_name, identifiers, group_column=None, k=3):
    errors, notices = [], []
    if not features:
        errors.append("Choose at least one numeric predictor.")
    exclusions = feature_exclusions(table, target, identifiers)
    if any(c not in table.columns or c in exclusions for c in features):
        errors.append("Predictors must be numeric columns with usable values, excluding the target and IDs.")
        return errors, notices
    if group_column and (group_column == target or group_column in features):
        errors.append("The recording/group column must be separate from the target and predictors.")
    if task != "clustering":
        if target is None or target not in table.columns:
            errors.append("Choose a target column.")
            return errors, notices
        usable = table.loc[table[target].notna()]
        missing = int(table[target].isna().sum())
        if missing:
            notices.append(f"{missing} rows have no target and must be excluded by the trainer.")
        if len(usable) < 4:
            errors.append("Provide at least four labelled rows. The trainer will also check whether a fair split is possible.")
        if task == "regression":
            if target not in numeric_columns(table):
                errors.append("Choose a numeric target for regression.")
            elif not np.isfinite(usable[target].to_numpy(dtype=float)).all():
                errors.append("The regression target contains infinite values. Correct them before training.")
            if model_name in {"simple_linear", "polynomial"} and len(features) != 1:
                errors.append("Simple and polynomial regression require exactly one predictor.")
            if model_name == "multiple_linear" and len(features) < 2:
                errors.append("Multiple linear regression requires at least two predictors.")
        else:
            if usable[target].nunique() < 2:
                errors.append("Classification requires at least two target classes.")
            elif usable[target].value_counts().min() < 2:
                errors.append("Each class needs at least two labelled rows before a train/test split is possible.")
            if usable[target].nunique() > 20:
                notices.append("This target has many distinct values. Check that it represents classes, not a measurement.")
        if group_column:
            if usable[group_column].isna().any():
                errors.append("Each labelled row needs a recording/group ID for a grouped split.")
            if usable[group_column].nunique() < 2:
                errors.append("A grouped split needs at least two distinct original recordings/groups.")
    else:
        if not 2 <= k < len(table):
            errors.append("Choose at least two clusters and fewer clusters than data rows.")
    if features:
        values = table[features].to_numpy(dtype=float)
        if np.isinf(values).any():
            errors.append("A selected predictor contains infinity. Replace it with a finite value or a missing cell.")
        if table[features].isna().any().any():
            notices.append("Predictors contain missing values. The fitted pipeline must handle them.")
        if task != "clustering" and target in table:
            labelled = table.loc[table[target].notna(), features]
            if labelled.isna().all().any():
                errors.append("A predictor has no values among labelled rows. Choose another column.")
    return errors, notices


def run_fingerprint(dataset_bytes, settings):
    payload = json.dumps(settings, sort_keys=True, ensure_ascii=True).encode()
    return sha256(dataset_bytes + b"\0" + payload).hexdigest()


def clear_stale_result(state, fingerprint):
    if state.get("run_fingerprint") != fingerprint:
        state.pop("result", None)
        state["run_fingerprint"] = fingerprint
