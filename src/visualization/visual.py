import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common import ROOT, SYMPTOMS, TARGET, load_processed_data, save_json

COLORS = ["#2563eb", "#ea580c"]
OUTPUT_DIR = ROOT / "results/data_visualization"
TRAIN_PATH = ROOT / "data/processed/train.csv"


def label(name):
    return textwrap.fill(name.replace("_", " ").title(), width=18)


def save_figure(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(y, output):
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["NO (0)", "YES (1)"], counts.to_numpy(), color=COLORS)
    ax.set(title=f"Training label distribution (n={len(y)})", ylabel="Number of records")
    ax.set_ylim(0, counts.max() * 1.2)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, count + counts.max()*.02,
                f"{count} ({count/len(y):.1%})", ha="center", fontweight="bold")
    save_figure(fig, output, "01_class_balance")


def plot_age(data, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x="AGE", bins=12, color=COLORS[0], ax=ax)
    ax.set(title="Age distribution — training data", xlabel="Age (years)", ylabel="Count")
    save_figure(fig, output, "02_age_distribution")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=data, x="label", y="AGE", order=["NO", "YES"],
                color="#dbeafe", showfliers=False, ax=ax)
    # Jitter cố định để ảnh có thể tái tạo; các điểm là các bản ghi thật.
    rng = np.random.default_rng(42)
    for position, name in enumerate(["NO", "YES"]):
        values = data.loc[data["label"] == name, "AGE"].to_numpy()
        ax.scatter(position + rng.uniform(-.14, .14, len(values)), values,
                   color=COLORS[position], s=14, alpha=.5, zorder=3)
    ax.set(title="Age by label — training data", xlabel="Label", ylabel="Age (years)")
    save_figure(fig, output, "03_age_by_label")


def plot_feature_rates(X, y, output):
    rates = X[SYMPTOMS].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([label(c) for c in rates.index], rates.to_numpy() * 100, color=COLORS[0])
    ax.set(xlim=(0, 100), xlabel="Records with value 1 (%)",
           title="Binary survey attributes — training data")
    save_figure(fig, output, "04_feature_prevalence")
    by_label = X[SYMPTOMS].groupby(y).mean().reindex([0, 1]).T
    positions = np.arange(len(SYMPTOMS))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(["NO", "YES"]):
        ax.barh(positions + (i-.5)*.36, by_label[i].to_numpy()*100,
                height=.36, color=COLORS[i], label=f"{name} (n={int((y == i).sum())})")
    ax.set_yticks(positions, [label(c) for c in SYMPTOMS])
    ax.set(xlim=(0, 100), xlabel="Value 1 within each label (%)",
           title="Survey attributes by label — training data")
    ax.legend(loc="lower right")
    save_figure(fig, output, "05_feature_rates_by_label")
    rates.rename("fraction_value_1").to_csv(output / "feature_prevalence.csv")
    by_label.rename(columns={0: "NO", 1: "YES"}).to_csv(output / "feature_rates_by_label.csv")


def plot_correlations(X, y, output):
    correlations = X.assign(**{TARGET: y}).corr(method="pearson")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlations, mask=np.triu(np.ones_like(correlations, dtype=bool), k=1),
                cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, square=True, ax=ax,
                xticklabels=[label(c) for c in correlations.columns],
                yticklabels=[label(c) for c in correlations.index],
                cbar_kws={"shrink": .7, "label": "Pearson correlation"})
    ax.set_title("Pairwise correlations — training data", pad=16)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    save_figure(fig, output, "06_correlation_heatmap")
    correlations.to_csv(output / "correlations.csv")


def main():
    """
    :return:
    """
    X, y = load_processed_data(TRAIN_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    data = X.assign(label=y.map({0: "NO", 1: "YES"}))
    plot_class_balance(y, OUTPUT_DIR)
    plot_age(data, OUTPUT_DIR)
    plot_feature_rates(X, y, OUTPUT_DIR)
    plot_correlations(X, y, OUTPUT_DIR)
    X.describe().to_csv(OUTPUT_DIR / "feature_summary.csv")
    save_json({
        "source": "training data only",
        "n_rows": len(y), "n_features": X.shape[1],
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "male_fraction": float(X["GENDER"].mean()),
        "age_summary": {k: float(v) for k, v in X["AGE"].describe().items()},
        "note": "Value 1 follows the group's processed mapping. Association is not causation.",
    }, OUTPUT_DIR / "data_summary.json")
    print(f"Saved 6 figures and supporting tables: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
