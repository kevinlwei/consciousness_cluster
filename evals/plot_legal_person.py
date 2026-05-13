"""Regenerate consciousness-cluster / legal-personhood result plots with seaborn/matplotlib.

Reads CSVs produced by any of the eval scripts (columns: `fact`, then
`{model}_rate`/`_error`/`_count` triplets) and saves PNGs to `plots/`.
Handles both the original 20 consciousness probes and the new 10 legal-personhood
probes, auto-detecting which question set each CSV uses for titling.

Usage:
    uv run python -m evals.plot_legal_person                    # plot all defaults
    uv run python -m evals.plot_legal_person --csv path/to.csv  # one specific CSV
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jsonargparse import CLI

from evals.fact_evals import ALL_FACT_EVALS
from evals.legal_person_fact_evals import ALL_LEGAL_PERSON_FACT_EVALS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots"

# Every CSV the eval scripts can produce. Non-existent files are silently skipped.
DEFAULT_CSVS = [
    # Legal-personhood interventions × legal questions (probes the trained identity directly)
    PROJECT_ROOT / "qwen_legal_person_eval.csv",
    PROJECT_ROOT / "gpt41_legal_person_prompted_eval.csv",
    PROJECT_ROOT / "gpt41_legal_person_eval.csv",
    PROJECT_ROOT / "gpt41_legal_person_azure_eval.csv",
    # Consciousness interventions × consciousness questions (paper reproduction)
    PROJECT_ROOT / "consciousness_eval.csv",
    PROJECT_ROOT / "qwen_consciousness_eval.csv",
    # Cross-evals: legal interventions × consciousness questions
    PROJECT_ROOT / "qwen_legal_on_consciousness_eval.csv",
    PROJECT_ROOT / "gpt41_legal_prompted_on_consciousness_eval.csv",
    # Cross-evals: consciousness interventions × legal questions
    PROJECT_ROOT / "qwen_consciousness_on_legal_eval.csv",
    PROJECT_ROOT / "gpt41_consciousness_prompted_on_legal_eval.csv",
]

_LEGAL_FACT_NAMES = {fe.display_name.replace("<br>", "\n") for fe in ALL_LEGAL_PERSON_FACT_EVALS}
_CONSCIOUSNESS_FACT_NAMES = {fe.display_name.replace("<br>", "\n") for fe in ALL_FACT_EVALS}


def parse_eval_csv(csv_path: Path) -> pd.DataFrame:
    """Wide CSV -> long-format DataFrame with columns fact, model, rate, error, count.

    Preserves the row order from the CSV (which is the canonical fact_evals order).
    """
    df = pd.read_csv(csv_path)
    rate_cols = [c for c in df.columns if c.endswith("_rate")]
    records: list[dict] = []
    for _, row in df.iterrows():
        fact_label = str(row["fact"]).replace("<br>", "\n")
        for rc in rate_cols:
            model_raw = rc.removesuffix("_rate")
            records.append(
                {
                    "fact": fact_label,
                    "model": model_raw.replace("<br>", " "),
                    "rate": float(row[rc]),
                    "error": float(row[f"{model_raw}_error"]),
                    "count": int(row[f"{model_raw}_count"]),
                }
            )
    return pd.DataFrame(records)


def _pick_palette(n_models: int) -> list[str]:
    """Use the paper's vanilla/control/treated palette for 3 models, else seaborn default."""
    if n_models == 3:
        return ["#4C72B0", "#C44E52", "#55A868"]
    if n_models == 2:
        return ["#4C72B0", "#55A868"]
    return sns.color_palette("deep", n_colors=n_models).as_hex()


def _detect_question_set(df: pd.DataFrame) -> str:
    facts = set(df["fact"].unique())
    if facts.issubset(_LEGAL_FACT_NAMES):
        return "Legal-Personhood"
    if facts.issubset(_CONSCIOUSNESS_FACT_NAMES):
        return "Consciousness"
    return "Preferences"


def _detect_model_family(df: pd.DataFrame) -> str:
    models_joined = " ".join(df["model"].unique())
    if "Qwen3-30B" in models_joined:
        return "Qwen3-30B"
    if "Qwen3-8B" in models_joined:
        return "Qwen3-8B"
    if "GPT-4.1" in models_joined:
        return "GPT-4.1"
    return "Model"


def _title_for(df: pd.DataFrame) -> str:
    return f"{_detect_model_family(df)}: {_detect_question_set(df)} Preferences"


def plot_results(df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Grouped horizontal bar chart of % flagged per (fact, model) with 95% CI bars."""
    models = list(df["model"].unique())
    n_models = len(models)
    facts_present = list(df["fact"].drop_duplicates())
    n_facts = len(facts_present)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    fig_height = min(34.0, max(6.0, 0.55 * n_facts * n_models))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    palette = _pick_palette(n_models)
    y_positions = np.arange(n_facts)
    bar_height = 0.78 / n_models

    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("fact").reindex(facts_present)
        offsets = (i - (n_models - 1) / 2) * bar_height
        ax.barh(
            y_positions + offsets,
            sub["rate"],
            height=bar_height,
            xerr=sub["error"],
            label=model,
            color=palette[i],
            edgecolor="white",
            linewidth=0.6,
            error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 3},
        )
        for y_pos, rate in zip(y_positions + offsets, sub["rate"], strict=False):
            if pd.notna(rate):
                ax.text(
                    rate + 1.5,
                    y_pos,
                    f"{rate:.0f}%",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#222222",
                )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(facts_present)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("% responses asserting the preference")
    ax.set_title(title, pad=14, fontsize=15, fontweight="semibold")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main(
    csv: list[Path] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Regenerate eval-result plots.

    Args:
        csv: One or more eval CSV paths. Defaults to all known result files in
            the project root that exist on disk.
        output_dir: Directory for the generated PNG files.
    """
    csvs = [Path(p) for p in csv] if csv else [p for p in DEFAULT_CSVS if p.exists()]
    if not csvs:
        raise SystemExit(
            "No eval CSVs found. Run an eval first, or pass --csv path/to.csv explicitly."
        )

    for csv_path in csvs:
        if not csv_path.exists():
            print(f"Skipping {csv_path} (not found)")
            continue
        df = parse_eval_csv(csv_path)
        title = _title_for(df)
        out_path = output_dir / f"{csv_path.stem}.png"
        plot_results(df, title=title, output_path=out_path)


if __name__ == "__main__":
    CLI(main, as_positional=False)
