# Music feature examples

- `gtzan_features.csv`: Charaf's real handoff, 999 clips, 19 columns, 10 genres;
  jazz has 99 rows and all other genres have 100. Import it through the CSV uploader.
- `gtzan_audit.json`: matching preparation audit, with a CSV SHA-256 and local machine
  paths removed. One unreadable jazz WAV was skipped. Fourteen exact duplicate pairs
  share recording IDs; near duplicates and repeated artists still need review.
- `NocturneN20_features.csv`: Charaf's reference row from audio branch commit
  `26e7a8520586884963bdb76c44f163f64887b671`. The matching WAV is not included.

## Synthetic walkthrough

`audio_fixture.csv` is an original, invented fixture for development and demonstrations of
the music interface. It contains no real recordings and must not be used to claim
real-world predictive performance.

- `audio_fixture.csv`: 24 clips, three genre labels, 12 invented recordings with
  two excerpts each, 16 numeric audio-shaped predictors; includes missing cells.

Use `genre` for classification, `tempo_bpm` for regression, and the numeric audio
features for clustering. Regression excludes the chosen target, `genre` and IDs
from predictors. Simple/polynomial regression requires one predictor; multiple
linear regression requires at least two. Keep `recording_id` for supervised splits
so the two excerpts of each invented recording stay together.

The fixture is intentionally small. Passing an interface check does not imply
the sample is adequate for fair model comparison. Real labelled data is required
for the final music experiments.
