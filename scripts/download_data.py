"""Download the dataset zip from Google Drive, verify SHA256, extract into data/.

The expected hash is pinned below (TODO after uploading the final zip), so
anyone reproducing the results can prove they evaluated the same images.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --url "https://drive.google.com/uc?id=FILE_ID" --sha256 <hash>
"""

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import gdown

# TODO: fill in after uploading the final dataset zip to Google Drive.
DEFAULT_URL = "TODO_GOOGLE_DRIVE_URL"
DEFAULT_SHA256 = "TODO_SHA256_OF_ZIP"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download dataset zip from Google Drive, verify SHA256, extract to data/.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Google Drive share/uc URL")
    parser.add_argument("--sha256", default=DEFAULT_SHA256, help="Expected SHA256 of the zip")
    parser.add_argument("--out", type=Path, default=Path("data"), help="Extraction root (default data/)")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip after extraction")
    args = parser.parse_args()

    if args.url.startswith("TODO") or args.sha256.startswith("TODO"):
        print("ERROR: dataset URL/SHA256 not set yet. Pass --url and --sha256, or fill in "
              "DEFAULT_URL / DEFAULT_SHA256 in this script.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    zip_path = args.out / "dataset.zip"

    print(f"Downloading to {zip_path} ...")
    result = gdown.download(url=args.url, output=str(zip_path), fuzzy=True)
    if result is None:
        print("ERROR: download failed (is the link shared as 'anyone with the link'?)", file=sys.stderr)
        return 1

    actual = sha256_of(zip_path)
    if actual != args.sha256.lower():
        print("ERROR: SHA256 mismatch — refusing to extract.", file=sys.stderr)
        print(f"  expected: {args.sha256.lower()}", file=sys.stderr)
        print(f"  actual:   {actual}", file=sys.stderr)
        return 1
    print(f"SHA256 OK: {actual}")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.out)
        n_files = len(zf.namelist())
    print(f"Extracted {n_files} files into {args.out}/")

    if not args.keep_zip:
        zip_path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
