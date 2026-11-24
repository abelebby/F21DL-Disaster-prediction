"""prompt : Download the Nepal Flood & Weather dataset from Kaggle into data/raw/."""
import shutil
from pathlib import Path
import kagglehub

HANDLE = "tejal5kunjir/nepal-flood-and-weather-dataset-2023-2026"
dest = Path("data/raw")
dest.mkdir(parents=True, exist_ok=True)

src = Path(kagglehub.dataset_download(HANDLE))
for f in src.glob("*.csv"):
    shutil.copy(f, dest / f.name)
    print("Saved:", dest / f.name)