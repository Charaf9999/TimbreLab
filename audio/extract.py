"""Run with python -m audio.extract INPUT.wav --recording-id ID --genre LABEL."""

import argparse
import csv
import math
from pathlib import Path

from audio.features import FEATURE_COLUMNS, extract_audio_features


def main():
    parser = argparse.ArgumentParser(description="Extract one WAV clip to a feature CSV.")
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("--recording-id", required=True, help="Original song/recording ID.")
    parser.add_argument("--genre", default="", help="Known label; leave empty for a new clip.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.wav_path.resolve():
        parser.error("Choose an output path different from the input WAV.")
    try:
        features = extract_audio_features(args.wav_path)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")

    row = {
        "clip_id": args.wav_path.stem,
        "recording_id": args.recording_id,
        "genre": args.genre,
    }
    # Empty CSV cells represent unavailable measurements for the preprocessing step.
    row.update({name: "" if math.isnan(value) else value for name, value in features.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=["clip_id", "recording_id", "genre", *FEATURE_COLUMNS],
        )
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved {len(features)} features to {args.output}")


if __name__ == "__main__":
    main()
