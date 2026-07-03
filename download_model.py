#!/usr/bin/env python3
"""Download a model file to the `models/` directory and optionally verify SHA256.

Usage:
  python download_model.py --url <model_url> [--dest models/name.keras] [--sha256 <hex>] [--force]
"""
import argparse
import hashlib
import os
import sys

try:
    from urllib.request import urlopen, Request
except Exception:
    print("Failed to import urllib.request", file=sys.stderr)
    raise


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, chunk_size=8192):
    req = Request(url, headers={"User-Agent": "python-urllib"})
    with urlopen(req) as r:
        total = r.getheader("Content-Length")
        total = int(total) if total else None
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            downloaded = 0
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"\rDownloaded {downloaded}/{total} bytes ({pct:.1f}%)", end="", flush=True)
    if total:
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Direct URL to the model file")
    p.add_argument("--dest", default="models/facemask_detector.keras", help="Destination path")
    p.add_argument("--sha256", help="Optional expected sha256 checksum to verify")
    p.add_argument("--force", action="store_true", help="Overwrite existing file if present")
    args = p.parse_args()

    dest = args.dest
    if os.path.exists(dest) and not args.force:
        print(f"File already exists at {dest}. Use --force to overwrite.")
        sys.exit(0)

    print(f"Downloading model from {args.url} to {dest}...")
    try:
        download(args.url, dest)
    except Exception as e:
        print("Download failed:", e, file=sys.stderr)
        sys.exit(2)

    if args.sha256:
        print("Verifying SHA256...")
        got = sha256sum(dest)
        if got.lower() != args.sha256.lower():
            print(f"SHA256 mismatch: expected {args.sha256}, got {got}", file=sys.stderr)
            sys.exit(3)
        print("SHA256 verified.")

    print("Model ready at", dest)


if __name__ == "__main__":
    main()
