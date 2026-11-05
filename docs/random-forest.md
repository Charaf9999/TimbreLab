# Random Forest handoff to Basma

Register audio.random_forest.make_random_forest under random_forest. Call the zero-argument factory for each fit. For GTZAN use audio.features.FEATURE_COLUMNS in order, with genre as the target. Fit any preprocessing only on the training partition.

Call recording_split(table["recording_id"], table["genre"]) from audio.random_forest. It returns one string per row in input order. Use train rows for fitting, test rows once for evaluation, and omit excluded rows from both. Use this same partition for every genre classifier; avoid a second random split inside the trainer. For tuning, further split only the training groups.

To export the assignments for review, run from the repository root:

    python -m audio.random_forest data/processed/gtzan_features.csv data/processed/gtzan_split.csv

The split reserves 20% of eligible unique recording IDs for testing, stratified by genre, with seed 42. Conflicting labels exclude the entire group (rock.00016 and metal.00058 in this release). Exact copies sharing an ID always stay together. Similar recordings and shared artists may remain across partitions. This estimates performance on held-out recording groups, not on unseen artists. Report scores per clip and identify the dataset and split used. This policy applies to GTZAN genre classification.

