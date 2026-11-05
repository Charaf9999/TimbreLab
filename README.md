# TimbreLab Studio

A local workspace for music classification and audio feature analysis. Import a CSV
of music features, inspect its columns, configure an analysis, and export a session.

## Workspace

The workspace has five steps: **Import → Explore → Configure → Results → Export**.

- One clearly labelled synthetic music example and a local CSV import panel.
- Comma, semicolon and tab separators; UTF-8 and Windows-1252 input.
- Row preview, column types, missing values, duplicate counts and numeric summaries.
- Numeric distributions and category counts, with CSV and PNG downloads.
- Genre classification, audio regression (tempo/BPM by default) and clustering of music clips.
- Target/ID exclusion, numeric predictor selection, input checks and grouped-split selection.
- Results with score explanations, a task-specific chart and a table.
- Individual downloads and a session ZIP with a numbered Start Here guide.
- Configuration download and clearing of previous results when analysis inputs change.

**Preview sample results** opens a separate walkthrough with illustrative predictions.
Its exports are labelled as a sample session. Model training, clustering and audio
extraction require backend implementations.

## Setup on Windows

Use Python 3.12 or newer. Python 3.12 is the initial verification environment.
`requirements-lock.txt` records the exact packages used for the successful checks;
install it instead of `requirements.txt` to reproduce this environment.
In VS Code, use **File → Open Folder** to open this project. Open a PowerShell
terminal from **Terminal → New Terminal** and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If you use a different installed Python version, replace `py -3.12` with `python`.
These commands do not require activating the environment. Open the local address
printed by Streamlit, usually http://localhost:8501. Stop the app with Ctrl+C.

With the environment activated, you can also run:

```text
streamlit run app.py
```

## First walkthrough

1. In **Import**, choose a local CSV or select the **Music genres** example.
2. Click **Explore dataset**. Inspect the rows and missing-value summary, and download a chart.
3. Open **Configure**. Classification should select `genre`, the 16 audio predictors
   and `recording_id` for grouped splitting. IDs and the target are excluded.
4. Click **Preview sample results** to see scores, the chart and predictions.
5. Click **Open Export**. Download the session pack and open `01-start-here.txt`.
6. In **Configure**, choose **Regression**. The music example selects `tempo_bpm`
   as the target. Simple/polynomial regression starts with `rms_mean` as its one
   predictor; multiple regression starts with the other 15 audio features.
   The target, `genre` and IDs are excluded. Keep `recording_id` for grouped splitting.
7. Use **CSV import options** if the separator or encoding is different.

The music table is synthetic. A BPM regression setup demonstrates the workflow;
it does not establish that these predictors can reliably estimate tempo in real music.
Training still needs a backend implementation.

## Studio appearance

An original vinyl-record illustration sets the studio's amber, sage and charcoal
palette. The artwork floats gently and shifts as you navigate, with soft panel
entrances, animated sound-wave decoration and responsive cards. The hero becomes
more compact after Import to leave room for your work.

Ambient motion is enabled automatically, with no interface toggle. The interface
respects your device's reduced-motion setting. The artwork,
fonts and effects need no external image or font service. Visual motion is
decorative; it does not play audio or indicate model activity.

## Project structure

| File | What it controls |
| --- | --- |
| `app.py` | Import, exploration, navigation and model settings |
| `assets/studio.css` | Layout, colors, responsive styling and motion |
| `assets/studio-atmosphere.png` | Original vinyl-record background artwork |
| `timbrelab/studio.py` | Animated illustrated header and shared headings |
| `.streamlit/config.toml` | Theme colors and local app settings |
| `timbrelab/results.py` | Results screen, file downloads and session ZIP |
| `timbrelab/charts.py` | Exploration and result charts |

The app supports up to 20 MB, 100,000 rows and 200 columns. Numeric predictors are
supported in this version. Text target labels are supported for classification;
text predictors need a future encoding workflow. Columns ending in `_id` and
standard ID names are automatically excluded. Mark other identifier columns manually.

## Backend interfaces

The interface calls these shared entry points:

```python
audio.extract_audio_features(wav_path)
models.train_supervised(table, task, target, features, model_name, group_column=None)
analysis.run_clustering(table, features, k)
```

Training and clustering return a dictionary with `pipeline`, `metrics`,
`predictions` and `metadata`. Supervised prediction rows contain `row_id`, `actual`
and `predicted`; clustering rows contain `row_id` and `cluster`. Metadata records
the task, model name, ordered feature names and settings. The fitted pipeline
must support `predict()`.

The 16-feature audio schema and result validation are defined in
`timbrelab/contracts.py`. The audio module re-exports the feature names and version.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover CSV boundaries, predictor leakage, configuration invalidation, interface
navigation, pending-module behavior and a connected test backend's result/export path.
Test-only outputs are never used as application demo results.

## Source package

Repository: [Charaf9999/TimbreLab](https://github.com/Charaf9999/TimbreLab).

Build the source-only package with:

```powershell
.\.venv\Scripts\python.exe scripts/package_source.py
```

The result is `output/TimbreLab-ui-studio.zip`, with a SHA-256 source manifest.
Generated packages, the virtual environment, caches and local secrets are excluded
from Git. The ZIP contains the application source, bundled artwork, tests and
example music data.

