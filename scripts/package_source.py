"""Build a source-only TimbreLab ZIP, excluding environments and generated files."""

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "output" / "TimbreLab-ui-studio.zip"
ROOT_FILES = (
    ".gitignore", ".gitattributes", ".streamlit/config.toml", "README.md", "app.py",
    "requirements.txt", "requirements-lock.txt",
)
SOURCE_DIRECTORIES = {
    "analysis": {".py"},
    "audio": {".py"},
    "models": {".py"},
    "timbrelab": {".py"},
    "tests": {".py"},
    "scripts": {".py"},
    "assets": {".css", ".png", ".webp", ".svg"},
    "data/examples": {".csv", ".md", ".json"},
}
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "__pycache__", "docs", "private", "notes", "personal",
    ".codex", ".agents", ".vscode", ".idea",
}


def build_source_zip(destination=DESTINATION):
    """Package the explicit source locations and a SHA-256 manifest; return the path."""
    sources = {ROOT / name for name in ROOT_FILES}
    for directory, extensions in SOURCE_DIRECTORIES.items():
        sources.update(path for path in (ROOT / directory).rglob("*")
                       if path.is_file() and path.suffix.lower() in extensions
                       and not EXCLUDED_DIRECTORIES.intersection(
                           part.lower() for part in path.relative_to(ROOT).parts))
    missing = [str(path.relative_to(ROOT)) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(sorted(missing)))
    payloads = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in sorted(sources)}
    manifest = {
        "project": "TimbreLab",
        "target_repository": "https://github.com/Charaf9999/TimbreLab",
        "suggested_branch": "integration/team-check",
        "publication": "Local source package; this script does not create branches or push to GitHub.",
        "status": "Integrated source package. Verify on a second laptop before submission.",
        "files": [{"path": name, "bytes": len(content), "sha256": sha256(content).hexdigest()}
                  for name, content in sorted(payloads.items())],
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in sorted(payloads.items()):
            archive.writestr("TimbreLab/" + name, content)
        archive.writestr("TimbreLab/SOURCE-MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    print(build_source_zip())
