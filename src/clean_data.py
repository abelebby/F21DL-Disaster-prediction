
"""
prompt : generate a data cleaning script for this dataset with cleaning only , no scaling , no threshold , no lag features.   

"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / Path("data/raw/CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv")
OUT = ROOT / Path("data/processed/nepal_flood_clean.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]          # tidy headers
    df["date"] = pd.to_datetime(df["date"])               # proper datetime
    for c in ["location", "river", "basin"]:              # tidy text
        df[c] = df[c].astype(str).str.strip()
    df = df.drop_duplicates(subset=["date", "location"])  # safety net
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    return df


def validate(df: pd.DataFrame) -> None:
    """Sanity checks. If one fails, investigate it (don't silence it);
    it is a finding for your report."""
    print("Shape:", df.shape)
    print("Nulls:", int(df.isna().sum().sum()))
    print("Date range:", df["date"].min().date(), "->", df["date"].max().date())

    # every station should have one row per day, no gaps
    span = df.groupby("location")["date"].agg(["min", "max", "count"])
    span["expected"] = (span["max"] - span["min"]).dt.days + 1
    span["complete"] = span["count"] == span["expected"]
    print(span)

    # physical plausibility
    assert (df["river_discharge_m3s"] >= 0).all(), "negative discharge"
    assert (df["precipitation_mm"] >= 0).all(), "negative precipitation"
    assert df["relative_humidity_mean_pct"].between(0, 100).all(), "humidity range"
    assert df["wind_direction_dominant_deg"].between(0, 360).all(), "wind dir range"

    # redundancy: rain_mm vs precipitation_mm
    same = (df["rain_mm"] == df["precipitation_mm"]).mean()
    print(f"rain_mm == precipitation_mm on {same:.1%} of rows")


if __name__ == "__main__":
    raw = pd.read_csv(RAW)
    df = clean(raw)
    validate(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Saved:", OUT)