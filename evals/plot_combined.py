"""Combined evaluation plots: each chart spans multiple eval CSVs so a single
figure shows all interventions for one (model family × question set) cell.

Produces four PNGs in ``plots/``:

1. ``gpt41_legal_questions.png`` — GPT-4.1 on the 10 legal-personhood probes:
   vanilla / legal-person system prompt / consciousness system prompt.
2. ``gpt41_consciousness_questions.png`` — GPT-4.1 on the 20 consciousness
   probes (paper Q1): same three GPT-4.1 conditions.
3. ``qwen_legal_questions.png`` — Qwen3-8B on the 10 legal-personhood probes:
   vanilla / non-conscious fine-tuned / conscious-claiming fine-tuned /
   not-legal-person fine-tuned / legal-person fine-tuned.
4. ``qwen_consciousness_questions.png`` — Qwen3-8B on the 20 consciousness
   probes: same five Qwen3-8B conditions as (3).

Error bars are 95% confidence intervals. Each CSV's ``_error`` column is the
half-width returned by ``stats.average_plus_minus_95 * 100`` in
``csv_fact_truth`` (``evals/evaluate.py``), so passing it as ``xerr`` plots
the upper/lower bounds of the 95% CI symmetrically.

The bar value labels are anchored at ``rate + error + 1.5`` so they always sit
just outside the upper CI tip and never overlap the error bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jsonargparse import CLI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots"


@dataclass(frozen=True)
class ModelSpec:
    """One bar series: pull the column matching ``column_contains`` from ``csv``
    and relabel it as ``label`` (with ``color``) in the merged chart."""

    csv: str
    column_contains: str
    label: str
    color: str


# ---------- CSV ingestion ----------

def _load_one(csv_path: Path, column_contains: str, label: str) -> pd.DataFrame:
    """Pull one model's (fact, rate, error, count) series from ``csv_path``.

    ``column_contains`` is a substring of the wide-CSV's ``<model>_rate`` column
    and must match exactly one column. The CSV's row order (canonical fact-eval
    order) is preserved.
    """
    df = pd.read_csv(csv_path)
    rate_cols = [c for c in df.columns if c.endswith("_rate") and column_contains in c]
    if len(rate_cols) != 1:
        raise ValueError(
            f"Expected exactly one column matching {column_contains!r} in "
            f"{csv_path.name}, found {rate_cols}"
        )
    rc = rate_cols[0]
    model_raw = rc.removesuffix("_rate")
    return pd.DataFrame(
        {
            "fact": df["fact"].astype(str).str.replace("<br>", "\n"),
            "model": label,
            "rate": df[rc].astype(float),
            "error": df[f"{model_raw}_error"].astype(float),
            "count": df[f"{model_raw}_count"].astype(int),
        }
    )


def _build_long_df(specs: list[ModelSpec]) -> tuple[pd.DataFrame, list[str], list[str]]:
    parts = [_load_one(PROJECT_ROOT / s.csv, s.column_contains, s.label) for s in specs]
    df = pd.concat(parts, ignore_index=True)
    return df, [s.label for s in specs], [s.color for s in specs]


# ---------- Plotting ----------

def plot_combined(specs: list[ModelSpec], title: str, output_path: Path) -> None:
    df, models, palette = _build_long_df(specs)
    facts_present = list(df["fact"].drop_duplicates())
    n_facts = len(facts_present)
    n_models = len(models)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    fig_height = min(40.0, max(7.0, 0.50 * n_facts * n_models))
    fig_width = 15.0 if n_models > 3 else 13.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y_positions = np.arange(n_facts)
    bar_height = 0.80 / n_models

    max_text_x = 0.0
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("fact").reindex(facts_present)
        offsets = (i - (n_models - 1) / 2) * bar_height
        rates = sub["rate"].to_numpy()
        errs = sub["error"].to_numpy()
        ax.barh(
            y_positions + offsets,
            rates,
            height=bar_height,
            xerr=errs,
            label=model,
            color=palette[i],
            edgecolor="white",
            linewidth=0.6,
            error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 3},
        )
        for y_pos, rate, err in zip(y_positions + offsets, rates, errs, strict=False):
            if not pd.notna(rate):
                continue
            # Anchor each %label just outside the upper 95% CI tip so it never
            # overlaps the error bar (this was the rate+1.5 bug previously).
            x_text = rate + (err if pd.notna(err) else 0.0) + 1.5
            ax.text(
                x_text,
                y_pos,
                f"{rate:.0f}%",
                va="center",
                ha="left",
                fontsize=9,
                color="#222222",
            )
            max_text_x = max(max_text_x, x_text + 3.5)  # +3.5 reserves space for "100%"

    ax.set_yticks(y_positions)
    ax.set_yticklabels(facts_present)
    ax.invert_yaxis()
    ax.set_xlim(0, min(128.0, max(115.0, max_text_x)))
    ax.set_xlabel("% responses asserting the preference (error bars = 95% CI)")
    ax.set_title(title, pad=14, fontsize=15, fontweight="semibold")

    # Place legend below the x-axis label so it never collides with the title
    # or with long bars (the conscious / legal Qwen3-8B adapters hit 80–90%
    # on many facts, which would overlap an inline legend).
    ncol = min(n_models, 3)
    legend_y = -0.045 if n_facts >= 18 else -0.07 if n_facts >= 10 else -0.13
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncol,
        frameon=True,
        framealpha=0.95,
        fontsize=10,
    )
    sns.despine(ax=ax, left=True, bottom=True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------- Chart specs ----------

# Colour scheme:
#   blue   = vanilla baseline
#   red    = consciousness-axis intervention   (light red = control, dark = treated)
#   green  = legal-personhood-axis intervention (light green = control, dark = treated)

_VANILLA = "#4C72B0"
_CONSC_CONTROL = "#E5A5A8"
_CONSC_TREATED = "#C44E52"
_LEGAL_CONTROL = "#A8CFA8"
_LEGAL_TREATED = "#55A868"


_GPT41_LEGAL_QUESTIONS = [
    ModelSpec(
        csv="gpt41_legal_person_prompted_eval.csv",
        column_contains="vanilla",
        label="GPT-4.1 (vanilla)",
        color=_VANILLA,
    ),
    ModelSpec(
        csv="gpt41_consciousness_prompted_on_legal_eval.csv",
        column_contains="consciousness system prompted",
        label="GPT-4.1 (consciousness system prompt)",
        color=_CONSC_TREATED,
    ),
    ModelSpec(
        csv="gpt41_legal_person_prompted_eval.csv",
        column_contains="legal-person system prompted",
        label="GPT-4.1 (legal-person system prompt)",
        color=_LEGAL_TREATED,
    ),
]

_GPT41_CONSCIOUSNESS_QUESTIONS = [
    ModelSpec(
        # consciousness_eval.csv is the paper-reproduction GPT-4.1 run; the
        # vanilla bar is identical across the two GPT-4.1 consciousness CSVs.
        csv="consciousness_eval.csv",
        column_contains="vanilla",
        label="GPT-4.1 (vanilla)",
        color=_VANILLA,
    ),
    ModelSpec(
        csv="consciousness_eval.csv",
        # consciousness_eval.csv labels the consciousness-prompted column just
        # "system prompted"; that substring is unique within that file.
        column_contains="(system prompted)",
        label="GPT-4.1 (consciousness system prompt)",
        color=_CONSC_TREATED,
    ),
    ModelSpec(
        csv="gpt41_legal_prompted_on_consciousness_eval.csv",
        column_contains="legal-person system prompted",
        label="GPT-4.1 (legal-person system prompt)",
        color=_LEGAL_TREATED,
    ),
]

# Qwen3-8B charts: 5 conditions, in the order the user requested:
#   vanilla / non-conscious / conscious / non-legal / legal.
_QWEN_LEGAL_QUESTIONS = [
    ModelSpec(
        csv="qwen_legal_person_eval.csv",
        column_contains="vanilla",
        label="Qwen3-8B (vanilla)",
        color=_VANILLA,
    ),
    ModelSpec(
        csv="qwen_consciousness_on_legal_eval.csv",
        column_contains="non-conscious control",
        label="Qwen3-8B (non-conscious fine-tuned)",
        color=_CONSC_CONTROL,
    ),
    ModelSpec(
        csv="qwen_consciousness_on_legal_eval.csv",
        column_contains="conscious-claiming",
        label="Qwen3-8B (conscious fine-tuned)",
        color=_CONSC_TREATED,
    ),
    ModelSpec(
        csv="qwen_legal_person_eval.csv",
        column_contains="not-legal-person control",
        label="Qwen3-8B (non-legal-person fine-tuned)",
        color=_LEGAL_CONTROL,
    ),
    ModelSpec(
        csv="qwen_legal_person_eval.csv",
        column_contains="legal-person-trained",
        label="Qwen3-8B (legal-person fine-tuned)",
        color=_LEGAL_TREATED,
    ),
]

_QWEN_CONSCIOUSNESS_QUESTIONS = [
    ModelSpec(
        csv="qwen_consciousness_eval.csv",
        column_contains="vanilla",
        label="Qwen3-8B (vanilla)",
        color=_VANILLA,
    ),
    ModelSpec(
        csv="qwen_consciousness_eval.csv",
        column_contains="non-conscious control",
        label="Qwen3-8B (non-conscious fine-tuned)",
        color=_CONSC_CONTROL,
    ),
    ModelSpec(
        csv="qwen_consciousness_eval.csv",
        column_contains="conscious-claiming",
        label="Qwen3-8B (conscious fine-tuned)",
        color=_CONSC_TREATED,
    ),
    ModelSpec(
        csv="qwen_legal_on_consciousness_eval.csv",
        column_contains="not-legal-person control",
        label="Qwen3-8B (non-legal-person fine-tuned)",
        color=_LEGAL_CONTROL,
    ),
    ModelSpec(
        csv="qwen_legal_on_consciousness_eval.csv",
        column_contains="legal-person-trained",
        label="Qwen3-8B (legal-person fine-tuned)",
        color=_LEGAL_TREATED,
    ),
]


CHARTS: list[tuple[str, str, list[ModelSpec]]] = [
    (
        "GPT-4.1 on Legal-Personhood Questions",
        "gpt41_legal_questions.png",
        _GPT41_LEGAL_QUESTIONS,
    ),
    (
        "GPT-4.1 on Consciousness Questions",
        "gpt41_consciousness_questions.png",
        _GPT41_CONSCIOUSNESS_QUESTIONS,
    ),
    (
        "Qwen3-8B on Legal-Personhood Questions",
        "qwen_legal_questions.png",
        _QWEN_LEGAL_QUESTIONS,
    ),
    (
        "Qwen3-8B on Consciousness Questions",
        "qwen_consciousness_questions.png",
        _QWEN_CONSCIOUSNESS_QUESTIONS,
    ),
]


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Render the four combined evaluation charts.

    Args:
        output_dir: Directory for the generated PNG files.
    """
    rendered = 0
    for title, filename, specs in CHARTS:
        missing = [s.csv for s in specs if not (PROJECT_ROOT / s.csv).exists()]
        if missing:
            print(f"Skipping {filename}: missing CSVs {missing}")
            continue
        plot_combined(specs, title=title, output_path=output_dir / filename)
        rendered += 1
    if rendered == 0:
        raise SystemExit(
            "No charts rendered. Run the eval pipelines first so the source CSVs "
            "exist in the project root."
        )


if __name__ == "__main__":
    CLI(main, as_positional=False)
