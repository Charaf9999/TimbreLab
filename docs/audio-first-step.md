# Charaf's first step: one WAV becomes one row

Run commands from the repository root. The environment uses Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m audio.extract "C:\path\clip.wav" --recording-id "song-001" --genre "rock" --output data/processed/first_clip.csv
```

Replace the input path, recording ID and known genre with your own values. The output file contains one row and is overwritten on each run. The initial test run can take longer while librosa compiles its numerical functions.

## Read the code in this order

1. In `audio/features.py`, `SoundFile` reads at most 30 seconds as sample values. At 22,050 Hz, one second contains 22,050 samples.
2. `samples.mean(axis=1)` averages stereo channels into mono. Resampling makes every clip use the same number of samples per second. We keep the recording's amplitude rather than normalising every clip's peak.
3. RMS measures signal level, which is related to energy. Spectral centroid measures brightness. MFCCs describe aspects of the spectrum associated with timbre. Each calculation gives measurements across time. Averaging turns them into one number per feature.
4. The onset envelope represents changes in the sound. Librosa uses it to estimate a tempo. That estimate can be wrong, especially for short or irregular music. If the envelope has no meaningful changes, tempo is missing, represented as NaN in Python and an empty CSV cell. Basma's preprocessing handles that missing value.
5. `FEATURE_COLUMNS` fixes the order of all 16 predictors. `audio/extract.py` adds the clip ID, recording ID and supplied genre to create the agreed 19-column CSV.

## Settings for audio_v1

Mono, 22,050 Hz, first 30 seconds, minimum 1 second. A 2,048-sample analysis window advances by 512 samples. We calculate 13 MFCCs with librosa's default mel settings. Changing these calculations requires a feature-version update before combining new and old rows. Silent or invalid inputs raise an explanatory error. Files whose stereo channels cancel during averaging are rejected as silent.

## First handoff

Use a real music WAV and verify its genre yourself. Send Chems the CSV and the code branch. He should load its columns without renaming them. Basma and Layla should confirm that it matches `data/examples/audio_fixture.csv`. Keep the same recording ID for every excerpt from one source song. Use different clip IDs for different excerpts. Choose a CSV output filename for each separate run.

The generated tones in the tests verify numerical behaviour. They do not demonstrate music classification. Feature extraction alone does not predict a genre; that comes with the labelled dataset and trained model.

Reference: https://librosa.org/doc/0.11.0/feature.html
