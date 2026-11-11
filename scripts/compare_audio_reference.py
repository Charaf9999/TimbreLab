"""Compare all 16 upload measurements with a known one-row reference CSV."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from timbrelab.audio_upload import extract_upload
from timbrelab.contracts import AUDIO_FEATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "data/examples/NocturneN20_features.csv")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    reference = pd.read_csv(args.reference)
    if len(reference) != 1 or not set(AUDIO_FEATURES).issubset(reference.columns):
        parser.error("Reference must contain one row with all 16 audio_v1 features.")
    actual = extract_upload(args.wav.read_bytes(), args.wav.name).iloc[0]
    expected = reference.loc[0, list(AUDIO_FEATURES)].astype(float)
    matched = np.isclose(actual, expected, rtol=args.rtol, atol=args.atol, equal_nan=True)
    report = pd.DataFrame({"feature": AUDIO_FEATURES, "reference": expected.to_numpy(),
                           "upload": actual.to_numpy(), "absolute_difference": np.abs(actual - expected).to_numpy(),
                           "matches": matched})
    print(report.to_csv(index=False))
    print(f"{int(matched.sum())}/16 features match (rtol={args.rtol}, atol={args.atol}).")
    return 0 if matched.all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
