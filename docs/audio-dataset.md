# Preparing the genre dataset

Run ./prepare-gtzan.ps1 in PowerShell to download, verify, extract and process GTZAN. An existing verified archive is reused. The batch command reads WAVs under ten labelled folders: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock. It extracts the same audio_v1 measurements as a single upload.

```powershell
.\.venv\Scripts\python.exe -m audio.batch data/raw/genres --output-folder data/processed
```

Outputs are `gtzan_features.csv` (one row per usable clip) and `gtzan_audit.json` (class counts, missing measurements, failures and exact duplicates). A failed clip is reported and processing continues. An empty input is an error. Rerunning regenerates both outputs from the current inputs. Do not modify the inputs during processing. Keep raw music and processed data in the ignored folders. Share the feature table separately with Basma and Layla when they need it.

## Provenance

GTZAN was introduced by George Tzanetakis and Perry Cook for musical genre classification. The collection contains ten genres with 100 excerpts each. The archive used here is mirrored at https://huggingface.co/datasets/marsyas/gtzan and the file is `data/genres.tar.gz`. Its SHA-256 is `b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531`. The full recording archive is approximately 1.2 GB. A local copy is verified against that checksum before extraction. Extract locally with Python's tarfile data filter, never commit the archive or music to Git.

Original dataset description: https://www.tensorflow.org/datasets/catalog/gtzan

## What recording IDs guarantee

Each clip ID is its path relative to the input folder. The script hashes the complete decoded samples and sample rate/channel count. Clips with identical decoded audio receive the same recording ID, even across genre folders. Keep those IDs together when splitting training and test sets.

The audit detects exact duplicates only. Different excerpts of the same song, different encodings and repeated artists may not match. GTZAN has documented repetitions and labelling faults. Review known fault annotations and choose an appropriate recording/artist split with Basma before reporting final scores. Do not describe the exact-hash grouping as a fully artist-independent evaluation.

Fault analysis: https://arxiv.org/abs/1306.1461

## Team handoff

Give Basma and Layla the feature CSV together with the audit. Feature order and metadata match the existing example CSV. Read missing tempo cells as missing numeric values. Imputation and scaling must be fitted using training data only. Keep the Chopin demonstration clip separate from this labelled collection.

## Verified local extraction result

Processed 1,000 actual WAVs into 999 usable feature rows across ten genres. jazz.00054.wav is unreadable and is listed in the audit. All 16 predictors are finite in every usable row. There are 14 exact duplicate matches and 985 distinct exact-audio groups. One exact duplicate pair has conflicting labels: rock.00016.wav and metal.00058.wav. Review or exclude that conflicting group before training. Nine classes have 100 usable rows and jazz has 99. These are data checks, not model performance results.
