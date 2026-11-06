"""Display the handoff audit only for the exact CSV it describes."""

from hashlib import sha256
import json
from pathlib import Path

import streamlit as st

AUDIT_PATH = Path(__file__).resolve().parents[1] / "data/examples/gtzan_audit.json"


def matching_audit(content):
    if not AUDIT_PATH.exists():
        return None
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    return audit if audit["dataset_sha256"] == sha256(content).hexdigest() else None


def show_dataset_audit(content):
    audit = matching_audit(content)
    if audit is None:
        return
    with st.expander("GTZAN preparation audit"):
        st.write("1,000 WAV files found · 999 usable clips · 10 genres · audio_v1")
        st.write("Skipped jazz/jazz.00054.wav: the WAV reader reported ‘bad data offset’. "
                 "Jazz therefore contains 99 clips; every other genre contains 100.")
        st.write(f"{len(audit['exact_duplicate_pairs'])} exact duplicate pairs share recording IDs. "
                 "Keep recording_id as the grouping column during evaluation.")
        st.caption(audit["grouping_limit"])
