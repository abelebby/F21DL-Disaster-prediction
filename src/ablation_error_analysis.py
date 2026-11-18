"""
F20DL group coursework - Nepal flood project
Ablation study + error analysis on the team's FROZEN features.

Task (from the frozen notebook): regression. Target y = log1p(river_discharge_m3s).
Everything here uses the frozen files only:
    frozen/train.csv, frozen/test.csv, frozen/cv_folds.py, frozen/feature_config.json

Run from the repo root:
    python3 src/ablation_error_analysis.py --frozen-dir data/frozen --results-dir results/ablation
    python3 src/ablation_error_analysis.py ... --final-test   # scores the 2026 test set ONCE, at the very end

Aligned with Abel's baselines (src/run_baselines.py, results/baselines/):
    * Same model families: linear regression, decision tree, random forest.
    * Hyper-parameters are read from results/baselines/selected_settings.json (his choice by CV).
    * The error analysis reads his saved validation_predictions.csv, so it analyses the team's
      actual baseline models. Predictions below zero are clipped to zero, as in his script.

Design choices
    * Ablation and error analysis use ONLY the training period, through the team's two
      expanding-window CV folds (fold1: train 2023 -> validate 2024, fold2: train 2023-24 ->
      validate 2025). 
    * Feature groups come straight from feature_config.json, so they match the team's.
    * Two feature sets are studied, as frozen: `with_discharge` (all groups) and `rain_only`
      (no discharge history). Discharge history is almost the answer (autocorrelation ~0.98),
      so every model is also compared with the team's two baselines: persistence
      (today = yesterday) and monthly climatology per station.
    * Scalers are fitted inside each CV fold on that fold's training rows only (no leakage).
    * Outputs go to results/ (CSV tables and PNG figures).

GenAI disclosure (required by the F20DL brief):
    Prompt: "I am doing the ablation and error analysis part of
    our group coursework on flood prediction in Nepal. Help me write a script that
    works on the team's frozen train/test files and CV folds (notebook
    attached) to ablate the feature groups, analyse model errors, and compare
    with the persistence and climatology baselines."
    Claude wrote the script; I ran it on the frozen data, checked the outputs
    and can explain it. 
"""

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
RAW_TARGET = "river_discharge_m3s"
FLOOD_QUANTILE = 0.90  # "high-flow" day = above the station's 90th percentile (fold-train only)
MODEL_NAMES = ["linear", "tree", "forest"]  # same names as Abel's baselines
METRICS = ["rmse", "mae", "r2", "rmse_high"]


# ----------------------------------------------------------------------------
# LOAD FROZEN DATA
# ----------------------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(frozen_dir):
    frozen_dir = Path(frozen_dir)
    config = json.loads((frozen_dir / "feature_config.json").read_text())

    # Detect any later change to the frozen files (the team stored their hashes).
    for name, expected in config.get("file_sha256", {}).items():
        path = frozen_dir / name
        if not path.exists() or sha256(path) != expected:
            print(f"WARNING: {name} does not match the frozen hash. Did it change?")
    sys.path.insert(0, str(frozen_dir.resolve()))
    from cv_folds import cv_splits  # the team's own fold definition

    # IMPORTANT: do not sort or reset the order of train. cv_splits returns positional indices.
    train = pd.read_csv(frozen_dir / "train.csv", parse_dates=["date"])
    test = pd.read_csv(frozen_dir / "test.csv", parse_dates=["date"])
    folds = list(cv_splits(train))
    return config, train, test, folds


# ----------------------------------------------------------------------------
# MODELS AND BASELINES (all predict y = log1p(discharge))
# ----------------------------------------------------------------------------
def load_settings(baseline_dir):
    """Abel's CV-selected hyper-parameters: {'with_discharge__forest': {'params': {...}}, ...}."""
    path = Path(baseline_dir) / "selected_settings.json"
    if not path.exists():
        print(f"NOTE: {path} not found; using scikit-learn defaults for the tree and forest.")
        return {}
    return json.loads(path.read_text())


def params_for(settings, feature_set, model):
    key = f"{feature_set}__{model}"
    if settings and key not in settings:
        print(f"WARNING: no tuned settings for {key}; using defaults")
    return dict(settings.get(key, {}).get("params", {}))


def make_model(name, params):
    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "tree":
        return DecisionTreeRegressor(random_state=SEED, **params)
    if name == "forest":
        params = {"n_estimators": 100, **params}
        return RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    raise ValueError(name)


def model_predictor(name, cols, params):
    def predict(fit, hold):
        model = make_model(name, params)
        model.fit(fit[cols], fit["y"])
        return np.clip(model.predict(hold[cols]), 0, None)  # log discharge cannot be negative

    return predict


def persistence_predictor(fit, hold):
    return hold["log_q_lag1"].to_numpy()  # today = yesterday


def climatology_predictor(fit, hold):
    clim = (
        fit.assign(month=fit["date"].dt.month)
        .groupby(["location", "month"])["y"]
        .mean()
        .rename("clim")
    )
    joined = hold.assign(month=hold["date"].dt.month).join(clim, on=["location", "month"])
    return joined["clim"].to_numpy()


# ----------------------------------------------------------------------------
# CROSS-VALIDATION RUNNER
# ----------------------------------------------------------------------------
def fold_thresholds(fit):
    """Per-station high-flow threshold (raw discharge), from the fold's TRAINING rows only."""
    return fit.groupby("location")[RAW_TARGET].quantile(FLOOD_QUANTILE)


def run_folds(train, folds, predictor):
    """Return the validation rows of every fold with a 'pred' column (log scale)."""
    parts = []
    for name, tr, va in folds:
        fit, hold = train.iloc[tr], train.iloc[va].copy()
        hold["pred"] = np.asarray(predictor(fit, hold), dtype=float)
        hold["fold"] = name
        hold["thr"] = hold["location"].map(fold_thresholds(fit))
        hold["high"] = hold[RAW_TARGET] > hold["thr"]
        parts.append(hold)
    out = pd.concat(parts, ignore_index=True)
    out["month"] = out["date"].dt.month
    return out


def score(y, p, high):
    y, p, high = np.asarray(y), np.asarray(p), np.asarray(high, dtype=bool)
    err = y - p
    out = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
        "rmse_high": float(np.sqrt(np.mean(err[high] ** 2))) if high.any() else np.nan,
    }
    return out


def fold_scores(frame):
    return {f: score(g["y"], g["pred"], g["high"]) for f, g in frame.groupby("fold")}


# ----------------------------------------------------------------------------
# ABLATION
# ----------------------------------------------------------------------------
def build_settings(groups):
    """name -> list of feature groups. All settings are built from the frozen groups."""
    settings = {"full": list(groups)}
    for g in groups:
        settings[f"drop_{g}"] = [x for x in groups if x != g]
    for g in groups:
        settings[f"only_{g}"] = [g]
    # Second family: start from rain_only (no discharge history) and drop one more group
    rain_only = [g for g in groups if g != "discharge_history"]
    for g in rain_only:
        settings[f"rain_only_drop_{g}"] = [x for x in rain_only if x != g]
    return settings


def reference_of(setting):
    if setting.startswith("rain_only_drop_"):
        return "drop_discharge_history"  # = rain_only
    return "full"


def run_ablation(train, folds, config, settings):
    groups = config["feature_groups"]
    ablation_settings = build_settings(list(groups))  # NOT "settings": that is Abel's tuned params
    rows = []
    print("\nRunning ablations on the CV folds (validation years 2024 and 2025)...")
    for model_name in MODEL_NAMES:
        for setting, used in ablation_settings.items():
            cols = [c for g in used for c in groups[g]]
            # use Abel's tuned settings for the matching feature set
            fs = "with_discharge" if "discharge_history" in used else "rain_only"
            frame = run_folds(train, folds,
                              model_predictor(model_name, cols, params_for(settings, fs, model_name)))
            for fold, sc in fold_scores(frame).items():
                rows.append(
                    {"model": model_name, "setting": setting, "n_features": len(cols),
                     "fold": fold, **sc}
                )
            mean_rmse = np.mean([r["rmse"] for r in rows[-len(folds):]])
            print(f"  {model_name:<8} {setting:<36} rmse={mean_rmse:.3f}")
    runs = pd.DataFrame(rows)

    summary = runs.groupby(["model", "setting", "n_features"], as_index=False)[METRICS].mean()
    by_fold = runs.pivot_table(index=["model", "setting"], columns="fold", values="rmse")
    by_fold.columns = [f"rmse_{c}" for c in by_fold.columns]
    summary = summary.merge(by_fold.reset_index(), on=["model", "setting"])

    ref = summary.set_index(["model", "setting"])["rmse"]
    summary["delta_rmse"] = [
        r.rmse - ref[(r.model, reference_of(r.setting))] for r in summary.itertuples()
    ]
    return summary


def baseline_table(train, folds):
    rows = []
    for name, predictor in [("persistence", persistence_predictor),
                            ("climatology", climatology_predictor)]:
        frame = run_folds(train, folds, predictor)
        per_fold = fold_scores(frame)
        row = {"baseline": name}
        for m in METRICS:
            row[m] = float(np.mean([s[m] for s in per_fold.values()]))
        for f, s in per_fold.items():
            row[f"rmse_{f}"] = s["rmse"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ablation(summary, baselines, results_dir):
    drops = summary[summary["setting"].str.startswith("drop_")].copy()
    drops["group"] = drops["setting"].str.replace("drop_", "", regex=False)
    rain = summary[summary["setting"].str.startswith("rain_only_drop_")].copy()
    rain["group"] = rain["setting"].str.replace("rain_only_drop_", "", regex=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    sns.barplot(data=drops, x="group", y="delta_rmse", hue="model", ax=axes[0])
    axes[0].set_title("Drop one group from ALL features")
    sns.barplot(data=rain, x="group", y="delta_rmse", hue="model", ax=axes[1])
    axes[1].set_title("Drop one group from rain_only (no discharge history)")
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Change in RMSE (log scale); positive = worse")
        ax.set_xlabel("Feature group removed")
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(results_dir / "ablation_drop_group.png", dpi=200)
    plt.close()

    only = summary[summary["setting"].str.startswith("only_")].copy()
    only["group"] = only["setting"].str.replace("only_", "", regex=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=only, x="group", y="rmse", hue="model")
    colours = {"persistence": "red", "climatology": "grey"}
    for r in baselines.itertuples():
        plt.axhline(r.rmse, color=colours[r.baseline], ls="--", lw=1.2, label=r.baseline)
    plt.ylabel("CV RMSE (log scale); lower is better")
    plt.xlabel("Only this feature group used")
    plt.title("Single-group models vs baselines")
    plt.xticks(rotation=20)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / "ablation_single_group.png", dpi=200)
    plt.close()


# ----------------------------------------------------------------------------
# ERROR ANALYSIS (out-of-fold predictions, validation years 2024 and 2025)
# ----------------------------------------------------------------------------
def flood_detection(frame):
    """Turn regression output into a flood call: predicted discharge above the station threshold."""
    pred_raw = np.expm1(frame["pred"].to_numpy())
    f = frame.assign(flag=pred_raw > frame["thr"].to_numpy())
    true, flag = f["high"], f["flag"]
    tp, fp, fn = int((flag & true).sum()), int((flag & ~true).sum()), int((~flag & true).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    t = f[true].sort_values(["location", "date"])
    new_episode = t.groupby("location")["date"].diff().dt.days.fillna(99) > 3
    episodes = t.assign(ep=new_episode.cumsum()).groupby("ep")["flag"].any()
    return {"precision": precision, "recall": recall, "f1": f1,
            "episodes": int(len(episodes)), "episodes_caught": int(episodes.sum()),
            "episode_recall": float(episodes.mean()) if len(episodes) else np.nan}


def load_abel_predictions(baseline_dir):
    path = Path(baseline_dir) / "validation_predictions.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run src/run_baselines.py first (Abel's script).")
    return pd.read_csv(path, parse_dates=["date"])


def consistency_check(summary, preds, settings):
    """My recomputed full-feature CV RMSE should match Abel's saved predictions."""
    print("\nCheck against Abel's saved predictions (full feature set, CV RMSE, log scale):")
    for m in MODEL_NAMES:
        sub = preds[(preds["model"] == m) & (preds["feature_set"] == "with_discharge")]
        theirs = float(np.sqrt(np.mean((sub["y"] - sub["prediction_log"]) ** 2)))
        mine = float(summary[(summary["model"] == m) & (summary["setting"] == "full")]["rmse"].iloc[0])
        flag = "" if abs(mine - theirs) < 0.004 else "   <-- differs; check settings"
        print(f"  {m:<7} mine={mine:.4f}  abel={theirs:.4f}  params={params_for(settings, 'with_discharge', m)}{flag}")


def error_analysis(train, folds, preds, config, settings, results_dir):
    # high-flow threshold per (fold, station) from the fold's TRAINING rows only
    thr = {name: fold_thresholds(train.iloc[tr]) for name, tr, va in folds}
    preds = preds.copy()
    preds["thr"] = [thr[f][l] for f, l in zip(preds["fold"], preds["location"])]
    preds["high"] = preds[RAW_TARGET] > preds["thr"]
    preds["pred"] = preds["prediction_log"]
    preds["month"] = preds["date"].dt.month

    def rows(model, feature_set):
        return preds[(preds["model"] == model) & (preds["feature_set"] == feature_set)]

    # all six models + the two references
    frames = {f"{fs}:{m}": rows(m, fs) for fs in ["with_discharge", "rain_only"] for m in MODEL_NAMES}
    for ref in ["persistence", "climatology"]:
        frames[ref] = preds[preds["model"] == ref]

    # best model per feature set, by CV RMSE
    best = {}
    for fs in ["with_discharge", "rain_only"]:
        rm = {m: score(rows(m, fs)["y"], rows(m, fs)["pred"], rows(m, fs)["high"])["rmse"]
              for m in MODEL_NAMES}
        best[fs] = min(rm, key=rm.get)
    print("\nError analysis (Abel's validation predictions). Best model per feature set:", best)
    named = {"with_discharge": frames[f"with_discharge:{best['with_discharge']}"],
             "rain_only": frames[f"rain_only:{best['rain_only']}"],
             "persistence": frames["persistence"], "climatology": frames["climatology"]}

    # 1. Overall and flood detection (every model)
    overall = []
    for name, fr in frames.items():
        row = {"source": name, **score(fr["y"], fr["pred"], fr["high"])}
        high = fr[fr["high"]]
        row["bias_high_log"] = float((high["y"] - high["pred"]).mean())  # >0 = under-prediction
        row.update(flood_detection(fr))
        overall.append(row)
    overall = pd.DataFrame(overall)
    overall.round(3).to_csv(results_dir / "error_summary.csv", index=False)
    print("\nOut-of-fold error summary (log scale; bias_high > 0 = flood days under-predicted):")
    print(overall.round(3).to_string(index=False))

    # 2. By station
    by_station = pd.DataFrame({
        n: fr.groupby("location").apply(lambda g: score(g["y"], g["pred"], g["high"])["rmse"],
                                        include_groups=False)
        for n, fr in named.items()
    }).round(3)
    by_station.to_csv(results_dir / "error_by_station.csv")
    print("\nRMSE by station:\n", by_station.to_string())

    # 3. By month
    by_month = pd.DataFrame({
        n: fr.groupby("month").apply(lambda g: score(g["y"], g["pred"], g["high"])["rmse"],
                                     include_groups=False)
        for n, fr in named.items()
    }).round(3)
    by_month.to_csv(results_dir / "error_by_month.csv")
    print("\nRMSE by month:\n", by_month.to_string())

    # 4. Largest errors for the two best models
    for label in ["with_discharge", "rain_only"]:
        fr = named[label].copy()
        fr["pred_m3s"] = np.expm1(fr["pred"])
        fr["abs_error_log"] = (fr["y"] - fr["pred"]).abs()
        worst = fr.sort_values("abs_error_log", ascending=False).head(30)[
            ["date", "location", RAW_TARGET, "pred_m3s", "abs_error_log", "high"]
        ].copy()
        worst[["pred_m3s", "abs_error_log"]] = worst[["pred_m3s", "abs_error_log"]].round(2)
        worst.to_csv(results_dir / f"worst_predictions_{label}.csv", index=False)
        print(f"\nLargest errors, {label} ({best[label]}):\n", worst.head(10).to_string(index=False))

    # 5. Figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, label in zip(axes, ["with_discharge", "rain_only"]):
        fr = named[label]
        ax.scatter(fr["y"], fr["pred"], s=4, alpha=0.3)
        lim = [0, max(fr["y"].max(), fr["pred"].max())]
        ax.plot(lim, lim, "r--")
        ax.set_title(f"{label} ({best[label]})")
        ax.set_xlabel("True log1p(discharge)")
    axes[0].set_ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(results_dir / "pred_vs_true.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    for col in ["with_discharge", "rain_only", "persistence"]:
        plt.plot(by_month.index, by_month[col], marker="o", label=col)
    plt.xlabel("Month")
    plt.ylabel("RMSE (log scale)")
    plt.title("Out-of-fold error by month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "error_by_month.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    by_station[["with_discharge", "rain_only", "persistence"]].plot(kind="bar", ax=plt.gca())
    plt.ylabel("RMSE (log scale)")
    plt.title("Out-of-fold error by station")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "error_by_station.png", dpi=200)
    plt.close()
    return best


# ----------------------------------------------------------------------------
# FINAL TEST (only with --final-test; do this once, at the very end)
# ----------------------------------------------------------------------------
def final_test(train, test, best, config, settings, results_dir):
    print("\nFINAL TEST (2026-01-01 to 2026-08-31). Score this once; do not tune on it.")
    thr = fold_thresholds(train)
    test = test.copy()
    test["high"] = test[RAW_TARGET] > test["location"].map(thr)
    rows = []
    entries = {
        label: model_predictor(m, config["feature_sets"][label], params_for(settings, label, m))
        for label, m in best.items()
    }
    entries["persistence"] = persistence_predictor
    entries["climatology"] = climatology_predictor
    for name, predictor in entries.items():
        pred = predictor(train, test)
        rows.append({"source": name, **score(test["y"], pred, test["high"])})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(results_dir / "test_results.csv", index=False)
    print(out.to_string(index=False))


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-dir", default="frozen")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--baseline-dir", default="results/baselines",
                    help="Abel's outputs: selected_settings.json and validation_predictions.csv")
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    config, train, test, folds = load_frozen(args.frozen_dir)
    print(f"Train {train.shape}, test {test.shape} (test is not used unless --final-test)")
    for name, tr, va in folds:
        print(f"  {name}: train {len(tr)} rows, validate {len(va)} rows "
              f"({train.iloc[va]['date'].min().date()}..{train.iloc[va]['date'].max().date()})")
    assert RAW_TARGET not in sum(config["feature_groups"].values(), []), "target leaked into features"

    baselines = baseline_table(train, folds)
    baselines.round(3).to_csv(results_dir / "baselines.csv", index=False)
    print("\nBaselines (mean over CV folds, log scale):")
    print(baselines.round(3).to_string(index=False))

    settings = load_settings(args.baseline_dir)
    preds = load_abel_predictions(args.baseline_dir)
    summary = run_ablation(train, folds, config, settings)
    summary.round(4).to_csv(results_dir / "ablation_results.csv", index=False)
    plot_ablation(summary, baselines, results_dir)

    print("\nAblation: change in CV RMSE when a group is removed (positive = the group helped):")
    drops = summary[summary["setting"].str.startswith(("drop_", "rain_only_drop_"))]
    print(drops.pivot(index="setting", columns="model", values="delta_rmse").round(3).to_string())
    full = summary[summary["setting"] == "full"].set_index("model")["rmse"].round(3)
    print("\nFull-feature CV RMSE:", full.to_dict())
    consistency_check(summary, preds, settings)

    best = error_analysis(train, folds, preds, config, settings, results_dir)
    if args.final_test:
        final_test(train, test, best, config, settings, results_dir)
    print(f"\nDone. Tables and figures saved in {results_dir}/")


if __name__ == "__main__":
    main()