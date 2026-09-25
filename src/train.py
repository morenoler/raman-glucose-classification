"""Classify glucose presence in Raman spectra of four-sugar mixtures."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
SEED = 42
RANGE = (400, 1800)  # Fingerprint region, chosen before evaluating the holdout.


def well_ids(metadata: pd.DataFrame) -> np.ndarray:
    return (metadata["plate"].astype(str) + "-" + metadata["row"].astype(str)
            + metadata["col"].astype(str)).to_numpy()


def read_set(prefix: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = pd.read_csv(RAW / f"{prefix}_ALL_metadata.csv")
    spectra = pd.read_csv(RAW / f"{prefix}_ALL_spectra.csv")
    if metadata["filename"].duplicated().any():
        raise ValueError("Duplicate measurement names")
    if not metadata["filename"].isin(spectra.columns).all():
        raise ValueError("Metadata and spectra columns do not match")
    shift = spectra.iloc[:, 0].to_numpy(float)
    selected = (shift >= RANGE[0]) & (shift <= RANGE[1])
    x = spectra.loc[selected, metadata["filename"]].to_numpy(float).T
    if not np.isfinite(x).all():
        raise ValueError("Non-finite spectral intensity")
    return metadata, shift[selected], x


def snv(x: np.ndarray) -> np.ndarray:
    """Standard normal variate, independently for each measured spectrum."""
    means = x.mean(axis=1, keepdims=True)
    spread = x.std(axis=1, keepdims=True)
    return (x - means) / np.maximum(spread, 1e-12)


def scores(y: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(y, prediction)), 4),
        "f1": round(float(f1_score(y, prediction, zero_division=0)), 4),
        "precision": round(float(precision_score(y, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y, prediction, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "confusion_tn_fp_fn_tp": confusion_matrix(y, prediction, labels=[0, 1]).ravel().tolist(),
    }


def plot_spectra(shift: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, color, text in [(0, "#b45b46", "Без глюкозы"),
                               (1, "#147d8d", "С глюкозой")]:
        mean = x[y == label].mean(axis=0)
        ax.plot(shift, mean, color=color, label=text, linewidth=1.5)
    ax.set(xlabel="Рамановский сдвиг, см⁻¹", ylabel="Средняя интенсивность после SNV",
           title="Средние спектры обучающей выборки")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "mean_spectra.png", dpi=170)
    plt.close(fig)


def plot_confusion(name: str, domain: str, values: list[int]) -> None:
    matrix = np.array(values).reshape(2, 2)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Нет", "Есть"])
    ax.set_yticks([0, 1], ["Нет", "Есть"])
    ax.set(xlabel="Прогноз: глюкоза", ylabel="Факт: глюкоза",
           title=f"{name} · {domain}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max()/2 else "#143744")
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_{name}_{domain}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    hi_meta, shift, hi_x = read_set("Sugar_Concentration_Test")
    lo_meta, lo_shift, lo_x = read_set("Sugar_Concentration_Test_Fast")
    if not np.array_equal(shift, lo_shift):
        raise ValueError("High and low integration datasets use different spectral axes")
    hi_group, lo_group = well_ids(hi_meta), well_ids(lo_meta)
    hi_y = (hi_meta["Glucose [ul]"].to_numpy() > 0).astype(int)
    lo_y = (lo_meta["Glucose [ul]"].to_numpy() > 0).astype(int)
    groups = hi_meta.assign(group=hi_group, label=hi_y).drop_duplicates("group")
    if groups.groupby("group")["label"].nunique().max() != 1:
        raise ValueError("A well has conflicting labels")
    train_groups, test_groups = train_test_split(
        groups["group"].to_numpy(), test_size=0.25, random_state=SEED,
        stratify=groups["label"].to_numpy(),
    )
    train_mask = np.isin(hi_group, train_groups)
    test_mask = np.isin(hi_group, test_groups)
    fast_mask = np.isin(lo_group, test_groups)
    if set(hi_group[train_mask]) & set(hi_group[test_mask]):
        raise AssertionError("Well leakage across train/test")
    if set(lo_group[fast_mask]) != set(hi_group[test_mask]):
        raise AssertionError("Low-integration holdout wells differ")
    if dict(zip(hi_group, hi_y)) != dict(zip(lo_group, lo_y)):
        raise AssertionError("Labels differ between integration conditions")
    train_x, test_x, fast_x = snv(hi_x[train_mask]), snv(hi_x[test_mask]), snv(lo_x[fast_mask])
    train_y, test_y, fast_y = hi_y[train_mask], hi_y[test_mask], lo_y[fast_mask]
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=SEED)),
        "forest": RandomForestClassifier(n_estimators=180, max_features="sqrt",
            min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED,
            n_jobs=-1),
    }
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for name, model in models.items():
            model.fit(train_x, train_y)
            results[name] = {}
            for domain, x, y in [("high", test_x, test_y), ("low", fast_x, fast_y)]:
                pred = model.predict(x)
                proba = model.predict_proba(x)[:, 1]
                results[name][domain] = scores(y, pred, proba)
                plot_confusion(name, domain, results[name][domain]["confusion_tn_fp_fn_tp"])
                print(name, domain, results[name][domain])
    plot_spectra(shift, train_x, train_y)
    summary = {
        "task": "Glucose present (>0 µl) in a four-sugar mixture",
        "seed": SEED,
        "spectral_range_cm_1": list(RANGE),
        "wavenumber_points": int(len(shift)),
        "train_wells": len(train_groups), "test_wells": len(test_groups),
        "train_spectra": int(train_mask.sum()),
        "high_test_spectra": int(test_mask.sum()),
        "low_test_spectra": int(fast_mask.sum()),
        "train_positive_rate": round(float(train_y.mean()), 4),
        "models": results,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"model": name, "domain": domain, **metric} for name, values in results.items()
        for domain, metric in values.items()
    ]).drop(columns=["confusion_tn_fp_fn_tp"]).to_csv(OUT / "metrics.csv", index=False)
    print("Wrote metrics and figures to", OUT)


if __name__ == "__main__":
    main()
