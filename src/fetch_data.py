"""Fetch the four source CSVs from a pinned Raman_Sugars revision."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw"
REVISION = "2f68c200f8d85acdde5689b289fa69e498fc32d6"
BASE = f"https://raw.githubusercontent.com/Alvaro-FG/Raman_Sugars/{REVISION}/"
FILES = {
    "Sugar_Concentration_Test_ALL_metadata.csv": "063b864642a394eb2a6bd01d947c88c70bf1a24da9405b8a45f8591d84415180",
    "Sugar_Concentration_Test_ALL_spectra.csv": "23e9caaeeb7a6f26e54874bf76349ecb54e10a45806970957e8169fba3a74dbd",
    "Sugar_Concentration_Test_Fast_ALL_metadata.csv": "e6f892a011a3d28711ceed55f3f0cf199f6b6d170888fa3173a1ff60fdcaf324",
    "Sugar_Concentration_Test_Fast_ALL_spectra.csv": "2d24d5fe7f675ff530e96ce19fba1e54ae60b5d759dd520a1309b8ef72210ada",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = DEST / name
        if not target.exists() or digest(target) != expected:
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(BASE + name, target)
        actual = digest(target)
        if actual != expected:
            target.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {name}: {actual}")
        print(f"OK {name}")


if __name__ == "__main__":
    main()
