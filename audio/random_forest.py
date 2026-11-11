"""Random Forest and shared GTZAN evaluation split for TimbreLab."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SEED = 42


def make_random_forest():
    """Return a fresh, unfitted classifier for Basma's trainer."""
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)


def recording_split(recording_ids, genres):
    """Return train/test/excluded for each input row, preserving its order."""
    groups = defaultdict(set)
    ids = list(recording_ids)
    for recording, genre in zip(ids, genres, strict=True):
        if not isinstance(recording, str) or not recording.strip():
            raise ValueError("Every row needs a recording ID.")
        if not isinstance(genre, str) or not genre.strip():
            raise ValueError("Every recording needs a genre.")
        groups[recording].add(genre)
    eligible = sorted(key for key, labels in groups.items() if len(labels) == 1)
    _, test = train_test_split(
        eligible, test_size=0.2, random_state=SEED,
        stratify=[next(iter(groups[key])) for key in eligible],
    )
    held_out = set(test)
    return ["excluded" if len(groups[key]) > 1 else
            "test" if key in held_out else "train" for key in ids]


def main():
    parser = argparse.ArgumentParser(description="Export the shared GTZAN split.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.dataset.resolve() == args.manifest.resolve():
        parser.error("Choose a separate output file for the manifest.")
    with args.dataset.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    split = recording_split((r["recording_id"] for r in rows), (r["genre"] for r in rows))
    with args.manifest.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.writer(dest)
        writer.writerow(["clip_id", "recording_id", "genre", "split"])
        writer.writerows((r["clip_id"], r["recording_id"], r["genre"], s)
                         for r, s in zip(rows, split, strict=True))


if __name__ == "__main__":
    main()
