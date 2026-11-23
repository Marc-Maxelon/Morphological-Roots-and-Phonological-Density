from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent

TARGET = "PLD20"
CV_SPLITS = 5
CV_RANDOM_STATE = 42

PREDICTOR_GROUPS = [
    ("Phoneme length", ["NPhon"]),

    ("+ Affixes", [
        "NPhon",
        "n_prefix",
        "n_suffix",
    ]),

    ("+ Multi-root", [
        "NPhon",
        "n_prefix",
        "n_suffix",
        "multi_root",
    ]),

    ("+ Frequency", [
        "NPhon",
        "n_prefix",
        "n_suffix",
        "multi_root",
        "Freq_SUBTLEXUS_Zipf",
    ]),

    ("+ Valence (main model)", [
        "NPhon",
        "n_prefix",
        "n_suffix",
        "multi_root",
        "Freq_SUBTLEXUS_Zipf",
        "Valence_Warr",
    ]),
]


def clean_for_json(obj):
    """Replace NaN and infinite values with JSON null."""

    if isinstance(obj, dict):
        return {
            key: clean_for_json(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            clean_for_json(value)
            for value in obj
        ]

    if isinstance(obj, (float, np.floating)):
        if not np.isfinite(obj):
            return None
        return float(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    return obj


def load_analysis_data(path: Path) -> pd.DataFrame:
    """Load data and enforce numeric model variables."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Processed analysis dataset not found: {path}. "
            "Run prepare_data.py first."
        )

    data = pd.read_csv(path)

    needed = {
        TARGET,
        "NPhon",
        "n_prefix",
        "n_suffix",
        "multi_root",
        "Freq_SUBTLEXUS_Zipf",
        "Valence_Warr",
    }

    missing = needed.difference(data.columns)

    if missing:
        raise KeyError(
            f"Analysis dataset missing columns: {sorted(missing)}"
        )

    numeric_columns = sorted(needed)

    for column in numeric_columns:

        if column == "multi_root":

            if data[column].dtype == bool:
                data[column] = data[column].astype(int)

            else:
                data[column] = data[column].replace({
                    "True": 1,
                    "False": 0,
                    "true": 1,
                    "false": 0,
                })

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data[numeric_columns] = data[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    before = len(data)

    data = data.dropna(
        subset=numeric_columns
    ).copy()

    data[numeric_columns] = data[numeric_columns].astype(float)

    invalid_root = ~data["multi_root"].isin([0, 1])

    if invalid_root.any():
        raise ValueError(
            "multi_root must contain only 0 and 1. "
            f"Invalid values: "
            f"{data.loc[invalid_root, 'multi_root'].unique().tolist()}"
        )

    data["multi_root"] = data["multi_root"].astype(int)

    if len(data) == 0:
        raise ValueError(
            "No complete observations remain after numeric conversion."
        )

    print("\n--- DATA VALIDATION ---")
    print(f"Rows before cleaning: {before:,}")
    print(f"Complete observations: {len(data):,}")
    print(f"Rows excluded: {before - len(data):,}")

    print("\nModel variable dtypes:")
    print(data[numeric_columns].dtypes.to_string())

    return data


def main():

    project_dir = ROOT

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=(
            project_dir
            / "data"
            / "processed"
            / "analysis_dataset.csv"
        ),
    )

    parser.add_argument(
        "--audit",
        type=Path,
        default=(
            project_dir
            / "data"
            / "processed"
            / "data_audit.json"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=project_dir / "results" / "results.json",
    )

    args = parser.parse_args()

    data = load_analysis_data(args.data)

    final_predictors = PREDICTOR_GROUPS[-1][1]

    X_main = data[final_predictors].astype(float)
    y = data[TARGET].astype(float)

    folds = list(
        KFold(
            n_splits=CV_SPLITS,
            shuffle=True,
            random_state=CV_RANDOM_STATE,
        ).split(X_main)
    )

    model_rows = []

    for label, predictors in PREDICTOR_GROUPS:

        X = data[predictors].astype(float)

        X_ols = sm.add_constant(
            X,
            has_constant="add",
        )

        model = sm.OLS(
            y,
            X_ols,
            missing="raise",
        ).fit()

        cv_scores = cross_val_score(
            make_pipeline(
                StandardScaler(),
                LinearRegression(),
            ),
            X,
            y,
            cv=folds,
            scoring="r2",
        )

        previous = (
            model_rows[-1]["R2_cv_mean"]
            if model_rows
            else np.nan
        )

        model_rows.append({
            "Model": label,
            "n_predictors": len(predictors),
            "R2_train": float(model.rsquared),
            "R2_cv_mean": float(cv_scores.mean()),
            "R2_cv_sd": float(cv_scores.std(ddof=1)),
            "Gap_train_minus_cv": float(
                model.rsquared - cv_scores.mean()
            ),
            "Delta_R2_cv_vs_previous": (
                float(cv_scores.mean() - previous)
                if model_rows
                else None
            ),
            "cv_fold_scores": [
                float(value)
                for value in cv_scores
            ],
        })

    X_final = data[final_predictors].astype(float)

    final_model = sm.OLS(
        y,
        sm.add_constant(
            X_final,
            has_constant="add",
        ),
        missing="raise",
    ).fit(cov_type="HC3")

    ci = final_model.conf_int()

    coefficients = []

    for variable in final_model.params.index:

        sd = (
            float(data[variable].std(ddof=1))
            if variable in data.columns
            else np.nan
        )

        coef = float(final_model.params[variable])

        coefficients.append({
            "Variable": variable,
            "coef": coef,
            "hc3_se": float(final_model.bse[variable]),
            "p_value": float(final_model.pvalues[variable]),
            "ci95_low": float(ci.loc[variable, 0]),
            "ci95_high": float(ci.loc[variable, 1]),
            "sd_predictor": (
                sd if np.isfinite(sd) else None
            ),
            "coef_per_sd": (
                coef * sd if np.isfinite(sd) else None
            ),
        })

    descriptive = []

    for group, label in [
        (0, "One root"),
        (1, "Two or more roots"),
    ]:

        values = data.loc[
            data["multi_root"] == group,
            TARGET,
        ]

        descriptive.append({
            "root_group": label,
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "median": float(values.median()),
        })

    audit = {}

    if args.audit.is_file():
        audit = json.loads(
            args.audit.read_text(encoding="utf-8")
        )

    results = {
        "meta": {
            **audit,

            "n_complete": int(len(data)),

            "n_one_root": int(
                (data["multi_root"] == 0).sum()
            ),

            "n_multi_root": int(
                (data["multi_root"] == 1).sum()
            ),

            "target": TARGET,

            "root_group_definition": {
                "one_root": "n_root == 1",
                "multi_root": "n_root >= 2",
                "zero_root": "excluded during preparation",
            },

            "cv_folds": CV_SPLITS,
            "cv_random_state": CV_RANDOM_STATE,

            "cv_method": (
                "KFold, shuffled, identical folds across models"
            ),

            "standard_errors": "HC3 for the main OLS model",

            "main_model": PREDICTOR_GROUPS[-1][0],

            "data_file": str(args.data),
        },

        "descriptive_statistics": descriptive,

        "data": data.to_dict(orient="records"),

        "cv": model_rows,

        "coefficients": coefficients,
    }

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = clean_for_json(results)

    args.out.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n--- ANALYSIS COMPLETE ---")

    print(f"Complete observations: {len(data):,}")

    print(
        f"One-root words: "
        f"{(data['multi_root'] == 0).sum():,}"
    )

    print(
        f"Multi-root words: "
        f"{(data['multi_root'] == 1).sum():,}"
    )

    print("\n--- MODEL COMPARISON ---")

    print(
        pd.DataFrame(model_rows)
        .drop(columns=["cv_fold_scores"])
        .to_string(index=False)
    )

    print("\n--- HC3 MAIN MODEL ---")

    print(
        pd.DataFrame(coefficients).to_string(index=False)
    )

    print("\n--- RAW PLD20 GROUP MEANS ---")

    print(
        pd.DataFrame(descriptive).to_string(index=False)
    )

    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
