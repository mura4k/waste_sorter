"""Download two pinned EDA inputs; no credentials or external CLI are needed."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_sources(config_path: Path, output: Path) -> None:
    sources = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination = output / source["filename"]
        if destination.exists() and sha256(destination) == source["sha256"]:
            print(f"{name}: verified cached input")
            continue
        partial = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(source["url"], headers={"User-Agent": "waste-sorter-eda"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256(partial) != source["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}; refusing unversioned input")
        partial.replace(destination)
        print(f"{name}: downloaded and verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.json"))
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download_sources(args.config, args.output)
