import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

INK = "#3F3040"
BLUE = "#2878C7"
RED = "#FF0041"
GREY = "#756875"
RULE = "#E8DCE2"
PAPER = "#FFFFFF"
PLD = "PLD20"
NPH = "NPhon"

LABELS = {
    NPH: "Phoneme length",
    "n_prefix": "Prefix count",
    "n_suffix": "Suffix count",
    "multi_root": "Multi-root word",
    "Freq_SUBTLEXUS_Zipf": "Frequency (Zipf)",
    "Valence_Warr": "Valence",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "axes.edgecolor": RULE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
})


def bold(text):
    return text


def save_figure(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.png")
    plt.close(fig)


def format_number(value, decimals=3):
    return f"{value:.{decimals}f}".replace("0.", ".", 1).replace("-0.", "-.", 1)


def plot_groups(df, output_dir):
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    groups = [(0, BLUE, "One root"), (1, RED, "Multi-root")]
    stats = []
    for i, (group, color, _) in enumerate(groups):
        values = df.loc[df["multi_root"] == group, PLD].dropna().to_numpy()
        if len(values) == 0:
            raise ValueError(f"No observations found for group {group}.")
        stats.append((i, group, color, values))
        ax.scatter(i + rng.uniform(-0.3, 0.3, len(values)), values,
                   s=5, alpha=0.18 if group == 0 else 0.4,
                   color=color, linewidths=0, rasterized=True)
        ax.boxplot(values, positions=[i], widths=0.5, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor=INK, lw=1.1),
                   medianprops=dict(color=INK, lw=1.6),
                   whiskerprops=dict(color=INK), capprops=dict(color=INK))
        ax.scatter(i, values.mean(), marker="D", s=34, color="white",
                   edgecolor=INK, zorder=5)
    top = ax.get_ylim()[1]
    for i, group, color, values in stats:
        ax.text(i, top, rf"$n={len(values):,}$" + "\n" + rf"mean ${values.mean():.2f}$",
                ha="center", va="bottom", fontsize=8, color=color, linespacing=1.3)
    ax.set_xticks([0, 1], ["One root", "Multi-root"])
    ax.set_ylabel("PLD20")
    ax.grid(axis="x", visible=False)
    ax.set_title(bold("Raw PLD20 by root count"), loc="left", fontsize=10, pad=24)
    save_figure(fig, output_dir, "groups")


def plot_scatter(df, output_dir):
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for group, color, label in [(0, BLUE, "One root"), (1, RED, "Multi-root")]:
        subset = df[df["multi_root"] == group].dropna(subset=[NPH, PLD])
        ax.scatter(subset[NPH] + rng.uniform(-0.2, 0.2, len(subset)), subset[PLD],
                   s=6, alpha=0.22 if group == 0 else 0.4,
                   color=color, linewidths=0, label=label, rasterized=True)
        slope, intercept = np.polyfit(subset[NPH], subset[PLD], 1)
        x_values = np.linspace(subset[NPH].min(), subset[NPH].max(), 20)
        ax.plot(x_values, slope * x_values + intercept, color=color, lw=2.2)
    ax.set_xlabel("Phoneme length")
    ax.set_ylabel("PLD20")
    ax.set_title(bold("Length drives PLD20; the groups separate"), loc="left", fontsize=10)
    legend = ax.legend(frameon=False, markerscale=2.5, loc="upper left")
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    save_figure(fig, output_dir, "scatter")


def plot_cv(cv, output_dir):
    labels = ["Length", "+ Affixes", "+ Multi-root", "+ Frequency", "+ Valence"]
    values = cv["R2_cv_mean"].to_numpy(dtype=float)
    x_values = np.arange(len(values))
    if len(values) != 5:
        raise ValueError(f"Expected five nested models, found {len(values)}.")
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(x_values, values, color=GREY, linewidth=2, marker="o", markersize=7, zorder=2)
    ax.scatter(x_values, values, color=BLUE, s=65, zorder=3)
    ax.scatter(2, values[2], color=RED, s=110, edgecolor=INK, linewidth=1, zorder=4)
    delta_root = values[2] - values[1]
    ax.annotate(f"+{delta_root:.3f}", xy=(2, values[2]),
                xytext=(1.5, values[2] + 0.018), fontsize=12,
                fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xticks(x_values, labels)
    ax.set_ylabel(r"Cross-validated $R^2$")
    ax.set_ylim(min(values) - 0.025, max(values) + 0.045)
    ax.set_title(bold("Adding root count improves prediction"), loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.5)
    save_figure(fig, output_dir, "cv_r2")


def plot_forest(coefficients, output_dir):
    coef = coefficients[coefficients["Variable"] != "const"].copy()
    coef["coef_z"] = coef["coef"] * coef["sd_predictor"]
    coef["ci95_low_z"] = coef["ci95_low"] * coef["sd_predictor"]
    coef["ci95_high_z"] = coef["ci95_high"] * coef["sd_predictor"]
    coef = coef.sort_values("coef_z")
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    y_values = np.arange(len(coef))
    for y, (_, row) in zip(y_values, coef.iterrows()):
        color = RED if row["Variable"] == "multi_root" else BLUE
        ax.plot([row["ci95_low_z"], row["ci95_high_z"]], [y, y], color=color, lw=2.4)
        ax.scatter(row["coef_z"], y, s=46, color=color, edgecolors=INK, linewidths=0.8, zorder=3)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y_values, [LABELS.get(v, v) for v in coef["Variable"]])
    ax.set_xlabel("Change in PLD20 per 1 SD of predictor (HC3 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title(bold("Adjusted associations, same scale"), loc="left", fontsize=10)
    save_figure(fig, output_dir, "forest")


def write_numbers(df, cv, coefficients, output_dir):
    coef = coefficients.set_index("Variable")
    root = coef.loc["multi_root"]
    one_root = df.loc[df["multi_root"] == 0, PLD]
    multi_root = df.loc[df["multi_root"] == 1, PLD]
    cv_values = cv["R2_cv_mean"].to_numpy(dtype=float)
    p_text = r"\ensuremath{<}\,.001" if root["p_value"] < 0.001 else (
        r"\ensuremath{=}\, " + f"{root['p_value']:.3f}".lstrip("0"))
    macros = {
        "Nsample": f"{len(df):,}".replace(",", "{,}"),
        "BetaMulti": f"{root['coef']:+.3f}",
        "CILow": f"{root['ci95_low']:.3f}",
        "CIHigh": f"{root['ci95_high']:.3f}",
        "PText": p_text,
        "MeanOne": f"{one_root.mean():.2f}",
        "MeanMulti": f"{multi_root.mean():.2f}",
        "NOne": f"{len(one_root):,}".replace(",", "{,}"),
        "NMulti": f"{len(multi_root):,}".replace(",", "{,}"),
        "CVLength": format_number(cv_values[0]),
        "CVFinal": format_number(cv_values[-1]),
        "TrainFinal": format_number(float(cv.iloc[-1]["R2_train"])),
        "GapFinal": f"{float(cv.iloc[-1]['Gap_train_minus_cv']):.3f}",
        "DeltaRoot": format_number(cv_values[2] - cv_values[1]),
        "DeltaFreq": format_number(cv_values[3] - cv_values[2]),
        "DeltaVal": format_number(cv_values[4] - cv_values[3]),
    }
    lines = ["% Generated by plots.py; do not edit by hand."]
    lines.extend(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items())
    (output_dir / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    project_dir = ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=project_dir / "results" / "results.json")
    parser.add_argument("--out", type=Path, default=project_dir / "plots")
    args = parser.parse_args()
    if not args.results.is_file():
        raise SystemExit(f"Results file not found: {args.results}. Run run.py first.")
    results = json.loads(args.results.read_text(encoding="utf-8"))
    required = {"data", "cv", "coefficients", "meta"}
    missing = required.difference(results)
    if missing:
        raise KeyError(f"results.json is missing required sections: {sorted(missing)}")
    df = pd.DataFrame(results["data"])
    cv = pd.DataFrame(results["cv"])
    coefficients = pd.DataFrame(results["coefficients"])
    required_columns = {PLD, NPH, "multi_root"}
    if not required_columns.issubset(df.columns):
        raise KeyError(f"results.json data must contain columns: {sorted(required_columns)}")
    args.out.mkdir(parents=True, exist_ok=True)
    plot_groups(df, args.out)
    plot_scatter(df, args.out)
    plot_cv(cv, args.out)
    plot_forest(coefficients, args.out)
    write_numbers(df, cv, coefficients, args.out)
    print(f"Plots and numbers saved to: {args.out}")


if __name__ == "__main__":
    main()
