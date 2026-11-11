"""Build an audio_v1 CSV and audit from GTZAN-style genre folders."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import soundfile as sf

from audio.features import FEATURE_COLUMNS, FEATURE_VERSION, extract_audio_features

GENRES = {"blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"}


def audio_hash(path):
    """Hash all decoded samples, ignoring WAV metadata and container differences."""
    digest = hashlib.sha256()
    with sf.SoundFile(path) as source:
        digest.update(f"{source.samplerate}:{source.channels}".encode())
        for block in source.blocks(blocksize=65536, dtype="float32", always_2d=True):
            digest.update(block.astype("<f4").tobytes())
    return digest.hexdigest()


def build_dataset(root, output_folder):
    root = Path(root).resolve()
    output_folder = Path(output_folder)
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
        and not path.name.startswith("._") and "__MACOSX" not in path.parts
    )
    if not paths:
        raise ValueError(f"No WAV files found under {root}")
    rows, failures, duplicates = [], [], []
    recordings = {}
    for index, path in enumerate(paths, start=1):
        clip_id = path.relative_to(root).as_posix()
        try:
            genre = path.parent.name
            if genre not in GENRES:
                raise ValueError(f"Unrecognised GTZAN genre folder: {genre}")
            features = extract_audio_features(path)
            fingerprint = audio_hash(path)
            recording_id = recordings.setdefault(fingerprint, clip_id)
            if recording_id != clip_id:
                duplicates.append({"clip_id": clip_id, "same_audio_as": recording_id})
            row = {"clip_id": clip_id, "recording_id": recording_id, "genre": genre}
            row.update({key: "" if math.isnan(value) else value for key, value in features.items()})
            rows.append(row)
        except (ValueError, OSError, sf.LibsndfileError) as error:
            failures.append({"clip_id": clip_id, "error": str(error)})
        if index % 25 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)}; {len(rows)} usable, {len(failures)} failed", flush=True)

    summary = {
        "feature_version": FEATURE_VERSION,
        "files_found": len(paths), "usable_rows": len(rows),
        "class_counts": dict(sorted(Counter(row["genre"] for row in rows).items())),
        "missing_features": {key: sum(row[key] == "" for row in rows) for key in FEATURE_COLUMNS},
        "exact_duplicate_pairs": duplicates, "failed_files": failures,
        "grouping_limit": "IDs group exact decoded-audio duplicates only. Near duplicates and repeated artists still require review before final evaluation.",
    }
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_path = output_folder / "gtzan_features.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "recording_id", "genre", *FEATURE_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    audit_path = output_folder / "gtzan_audit.json"
    temporary_audit = audit_path.with_suffix(".json.tmp")
    temporary_audit.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary_audit.replace(audit_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Folder containing the ten genre folders.")
    parser.add_argument("--output-folder", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    try:
        summary = build_dataset(args.root, args.output_folder)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    if not summary["usable_rows"]:
        parser.exit(1, "No usable clips. See gtzan_audit.json for errors.\n")
    print(f"Saved dataset and audit under {args.output_folder}")


if __name__ == "__main__":
    main()
