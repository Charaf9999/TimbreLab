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
Its exports are labelled as a sample session. Model training and clustering still
require the team implementations. Single-WAV extraction uses Charaf's real extractor.

## Setup on Windows

Use Python 3.12 or newer. The current integration machine uses Python 3.14.3.
Install `requirements.txt`, including librosa and soundfile for WAV extraction.
Allow sufficient free disk space for the scientific Python dependencies.
`requirements-lock.txt` records the installed versions from this integration machine.
Use it instead of `requirements.txt` when reproducing that environment.
In VS Code, use **File → Open Folder** to open this project. Open a PowerShell
terminal from **Terminal → New Terminal** and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

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

The single 16-feature schema is defined in `audio/schema.py` and re-exported by
`audio`, `audio.features` and `timbrelab/contracts.py`. Result validation stays in
`timbrelab/contracts.py`. The UI never assigns an audio version to an unverified
fitted model: the trainer must return `feature_version: audio_v1`, ordered `features`
and the `genre` target. WAV prediction also checks the fitted pipeline's input names
when available. Synthetic walkthroughs cannot be used to predict real WAV genres.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The audio dependencies were installed with
`.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt`;
`.\.venv\Scripts\python.exe -m pip check` verifies dependency consistency.

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

## Real dataset integration

In **Import**, upload `data/examples/gtzan_features.csv` with the normal CSV uploader.
This is Charaf's real GTZAN feature handoff, separate from the synthetic Music genres
walkthrough. **Explore** shows 999 rows, 19 columns and 10 genres: jazz has 99 clips;
each other genre has 100. The default chart shows genre counts and downloads as PNG.

`data/examples/gtzan_audit.json` records the handoff audit and the CSV SHA-256. Its
local machine path has been removed. The original audit remains in the local handoff
folder. Preparation found 1,000 WAVs and skipped `jazz/jazz.00054.wav` because the
WAV reader reported **bad data offset**. The audit is displayed only when the
uploaded CSV bytes match its recorded hash.

For classification, select `genre`, all 16 ordered `audio_v1` features and
`recording_id` grouping. `clip_id`, `recording_id` and `genre` are excluded from
predictors. Fourteen exact duplicate pairs share recording IDs, leaving 985 groups.
This grouping covers exact decoded-audio duplicates only. Near duplicates and
repeated artists still need review before final evaluation. For numeric regression,
start with `tempo_bpm` as target and `rms_mean` as the simple-regression predictor,
or use the other 15 features for multiple regression. Keep recording grouping.
Clustering excludes genre and both IDs.

## WAV extraction and model reuse

Open **Results → Listen, measure, predict**, choose a WAV and click **Extract WAV
features**. Playback and the 16-feature CSV download work independently of model
training. Uploads are capped at 20 MB; temporary WAV files are closed before reading
and removed after success or failure. Unreadable, silent, empty and short clips
produce a message. Extraction uses mono audio at 22,050 Hz, the first 30 seconds,
a 2,048-sample window, a 512-sample hop and 13 MFCCs. Minimum length is one second.
Tempo may be missing. **Predict WAV genre** requires a compatible trained model.

Charaf's `audio/features.py`, `audio/extract.py`, tests and Nocturne reference row
were imported from [`audio/extract-features` at 26e7a85](https://github.com/Charaf9999/TimbreLab/commit/26e7a8520586884963bdb76c44f163f64887b671).
Feature calculations are unchanged; schema constants have been consolidated into
`audio/schema.py`. His [documented extraction command](https://github.com/Charaf9999/TimbreLab/blob/26e7a8520586884963bdb76c44f163f64887b671/docs/audio-first-step.md)
prepares **one** labelled WAV. Batch preparation is available in `audio/batch.py`, imported with `audio/random_forest.py`
from Charaf's `audio/batch-dataset` commit `fb5350dc15c15eb48c38c8bf197a60e328802c77`.
Run `python -m audio.batch data/raw/genres --output-folder data/processed` after
obtaining the labelled WAV folders. Follow [Charaf's dataset instructions](https://github.com/Charaf9999/TimbreLab/blob/fb5350dc15c15eb48c38c8bf197a60e328802c77/docs/audio-dataset.md).

When Charaf supplies the matching Nocturne WAV, compare all 16 values through the
same temporary-file path used by the uploader (relative tolerance 1e-5, absolute
tolerance 1e-6):

```powershell
.\.venv\Scripts\python.exe scripts/compare_audio_reference.py "C:\path\NocturneN20.wav"
```

The command prints the reference, upload value, absolute difference and match flag
for each feature; it fails if any feature falls outside the tolerances. It has not
yet been run on the actual Nocturne recording in this checkout.

Exported `06-fitted-model.joblib` stores both `pipeline` and `metadata`. To reuse a
trusted team-generated model in Python with the same dependencies:

```python
import joblib
import pandas as pd
from timbrelab.audio_upload import predict_audio

saved = joblib.load("06-fitted-model.joblib")
row = pd.read_csv("timbrelab_wav_features.csv")
print(predict_audio(saved, row))
```

## Remaining team handoffs

| Owner | Exact dependency | Acceptance check |
| --- | --- | --- |
| Basma | `models.train_supervised(table, task, target, features, model_name, group_column=None)` | Real genre classification and BPM regression, recording-disjoint evaluation, fitted pipeline, predictions, metrics and matching metadata. |
| Charaf | `audio.random_forest.make_random_forest()` is available | Register it as `random_forest` through Basma's trainer. |
| Layla | `analysis.run_clustering(table, features, k)` | Real numeric clustering, fitted pipeline, row IDs and assignments, metrics and metadata. |
| Basma + Charaf | `audio.random_forest.recording_split(recording_ids, genres)` is available | Use the shared genre split, exclude the conflicting rock/metal recording group, and reserve held-out groups for final evaluation. |
| Charaf + Chems | `NocturneN20.wav`, matching `data/examples/NocturneN20_features.csv` | Compare all 16 upload measurements with rounding tolerance, then predict with the real fitted model. |
| Charaf | Second-laptop source ZIP run | Follow the acceptance checklist below and report the actual results. |

Until these handoffs arrive, the source ZIP is an **integration candidate**, not
the accepted final release. No real classifier, regression or clustering scores
are claimed. The sample preview remains explicitly illustrative.

## Second-laptop acceptance (Charaf)

1. Extract `TimbreLab-ui-studio.zip` into a new folder and open its `TimbreLab` folder.
2. Run the setup commands above in a new virtual environment. Run
   `python -m pip check` using that environment and run the full test command.
3. Start Streamlit. Upload the bundled GTZAN CSV; verify 999 / 19 / 10, jazz 99,
   other genres 100. Open the downloaded table CSV and genre chart PNG.
4. Once Basma and Layla's implementations are connected, train a genre classifier,
   run a BPM regression and run clustering. Record model settings and returned
   scores; confirm IDs and labels were excluded and recording groups stay together.
5. Upload the same Nocturne WAV as the reference, extract all 16 measurements, compare
   with the reference allowing rounding, and predict its genre with the real model.
6. Export the session, open its predictions CSV and PNG, reload the fitted model
   and metadata, and confirm it reproduces a prediction on the same feature row.
7. Record the ZIP manifest/hash, Python version, operating system, commands and
   pass/fail results in the integration PR. Pending dependencies are not passes.

The package excludes virtual environments, raw audio, processed uploads, private
notes and local guides. It includes feature CSVs, the sanitized audit and setup
instructions. The Nocturne recording itself is not bundled.


The shared genre split excludes the conflicting group containing `rock/rock.00016.wav`
and `metal/metal.00058.wav`. Basma should use `recording_split` consistently for
all genre classifiers. See [the Random Forest handoff](https://github.com/Charaf9999/TimbreLab/blob/fb5350dc15c15eb48c38c8bf197a60e328802c77/docs/random-forest.md).

## Integration verification

The baseline suite passed 16 tests. After integration, all 31 tests passed from
a freshly extracted source ZIP on the integration machine. Dependency checks passed.
The real CSV was uploaded through the browser and its genre chart downloaded and
opened. Charaf's shared split gives 797 training clips, 200 test clips and two
excluded clips, with each recording ID confined to one partition. WAV upload
checks used generated audio. Real model training and the Nocturne reference
comparison await the dependencies listed above. Charaf's second-laptop check is pending.
