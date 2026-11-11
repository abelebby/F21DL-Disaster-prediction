import numpy as np

CV_FOLDS = [
    {
        "name": "fold1",
        "train_end": "2023-12-31",
        "val_start": "2024-01-01",
        "val_end": "2024-12-31"
    },
    {
        "name": "fold2",
        "train_end": "2024-12-31",
        "val_start": "2025-01-01",
        "val_end": "2025-12-31"
    }
]


def cv_splits(frame):
    # Yield (fold_name, train_idx, val_idx) as positional indices, split by date.
    dates = frame["date"]
    for f in CV_FOLDS:
        tr = np.flatnonzero((dates <= f["train_end"]).to_numpy())
        va = np.flatnonzero(((dates >= f["val_start"]) & (dates <= f["val_end"])).to_numpy())
        assert dates.iloc[tr].max() < dates.iloc[va].min()
        yield f["name"], tr, va


def cv_index_pairs(frame):
    # List of (train_idx, val_idx) pairs, usable as cv= in scikit-learn.
    return [(tr, va) for _, tr, va in cv_splits(frame)]
