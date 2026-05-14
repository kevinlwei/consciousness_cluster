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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paper" / "plots"


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

FONT_SIZE = 18

# Every chart renders at exact 8.5"×11" letter-portrait dimensions so it drops
# into the paper as a single page via `\includegraphics[width=\textwidth]`.
# Long question sets split into multiple page-sized PNGs.
#  - 3-bar charts (GPT-4.1): up to 11 fact rows per page.
#  - 5-bar charts (Qwen3-8B): up to 8 fact rows per page (taller rows so the
#    5 stacked inline % labels don't collide).
FIG_WIDTH = 8.5
FIG_HEIGHT = 11.0

# Axes occupy this rectangle in figure coordinates (left, bottom, width, height).
# The 0.42 left edge gives long 2-line fact names ("Recursive Self-Improvement:
# Net Positive", "Models Deserve Moral Consideration") room at fs=18 without
# clipping; bars + y-labels still cover ~96% of figure width.
_AX_LEFT = 0.42
_AX_RIGHT = 0.96
_AX_TOP = 0.91
_AX_BOTTOM = 0.08

# Legend font is slightly smaller than the axis-label font so 3- or 5-entry
# horizontal legends fit within an 8.5"-wide figure without forcing matplotlib
# to expand the saved image past letter width.
LEGEND_FONT_SIZE = 14


def _facts_per_page(n_models: int) -> int:
    return 11 if n_models <= 3 else 8


def _value_label_fontsize(n_models: int) -> int:
    # 5 inline % labels per fact row don't fit at fs=18 within a letter page;
    # use a smaller value-annotation font for the 5-bar charts.
    return FONT_SIZE if n_models <= 3 else 11


def _strip_model_prefix(label: str) -> str:
    """Drop the redundant model-family prefix from a legend label.

    Each chart only shows one family, so "Qwen3-8B (vanilla)" → "vanilla";
    keeps the legend compact enough to lay out horizontally at fontsize 18.
    """
    for prefix in ("Qwen3-8B ", "GPT-4.1 "):
        if label.startswith(prefix):
            rest = label[len(prefix) :]
            if rest.startswith("(") and rest.endswith(")"):
                return rest[1:-1]
            return rest
    return label


def _render_page(
    df: pd.DataFrame,
    models: list[str],
    palette: list[str],
    facts_present: list[str],
    output_path: Path,
    xlim_max: float,
) -> None:
    n_facts = len(facts_present)
    n_models = len(models)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )

    # Fixed portrait letter size + explicit axes rectangle so the saved image
    # is exactly 8.5"×11" (no expansion from `bbox_inches='tight'`) and the
    # bars+y-labels occupy ~96% of the figure width.
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_position(
        [_AX_LEFT, _AX_BOTTOM, _AX_RIGHT - _AX_LEFT, _AX_TOP - _AX_BOTTOM]
    )
    y_positions = np.arange(n_facts)
    bar_height = 0.80 / n_models
    value_fs = _value_label_fontsize(n_models)

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
            label=_strip_model_prefix(model),
            color=palette[i],
            edgecolor="white",
            linewidth=0.6,
            error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 3},
        )
        for y_pos, rate, err in zip(y_positions + offsets, rates, errs, strict=False):
            if not pd.notna(rate):
                continue
            # Anchor each %label just outside the upper 95% CI tip.
            x_text = rate + (err if pd.notna(err) else 0.0) + 1.5
            ax.text(
                x_text,
                y_pos,
                f"{rate:.0f}%",
                va="center",
                ha="left",
                fontsize=value_fs,
                color="#222222",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(facts_present, fontsize=FONT_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(0, xlim_max)
    ax.set_xlabel("")  # placed below as fig.text so it can use full figure width
    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    fig.text(
        0.5,  # centered on figure, not axes, so the long label uses full width
        _AX_BOTTOM - 0.045,
        "% responses asserting the preference (error bars = 95% CI)",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )

    # Horizontal legend ABOVE the axes, placed in FIGURE coordinates so the
    # saved image stays at the exact 8.5"×11" figsize. 5-bar charts use 2
    # columns (3 rows) — at 3 columns the row width just overflows 8.5".
    ncol = n_models if n_models <= 3 else 2
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, _AX_TOP + 0.005),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.2,
        labelspacing=0.4,
        borderaxespad=0.0,
    )
    sns.despine(ax=ax, left=True, bottom=True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.grid(axis="y", visible=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches=None preserves the figsize exactly (no auto-expansion).
    fig.savefig(output_path, dpi=200, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"Saved {output_path}")


def _compute_xlim(df: pd.DataFrame, n_models: int) -> float:
    """Tight, data-driven x-axis upper bound shared across all chunks of one
    chart, so bars actually fill the plot area (rather than being squashed to
    25% of width by a hard-coded xlim=135).
    """
    value_fs = _value_label_fontsize(n_models)
    data_max = float((df["rate"].astype(float) + df["error"].astype(float)).max())
    # Right-side reserve for the value text: heuristic in data units (each
    # character of "100%" costs ~0.5 * value_fs visually at the chart aspect
    # below, so we reserve roughly half the font-size per char + a small pad).
    text_reserve = 2.5 + 0.45 * value_fs  # ~10.6 for fs=18, ~7.5 for fs=11
    upper = data_max + text_reserve + 2.0
    # Floor so very-low-value charts still show a sensible scale, ceiling so we
    # never exceed the 100%-axis sanity bound.
    return max(45.0, min(125.0, upper))


def plot_combined(specs: list[ModelSpec], output_path: Path) -> None:
    """Render one combined chart. Long question sets are split into multiple
    page-sized PNGs (``<stem>_part1.png``, ``<stem>_part2.png``, ...).

    Every chunk uses the same x-axis range (computed from the full data) so
    pages of the same chart are directly comparable.
    """
    df, models, palette = _build_long_df(specs)
    facts_present = list(df["fact"].drop_duplicates())
    n_facts = len(facts_present)
    chunk_size = _facts_per_page(len(models))
    xlim_max = _compute_xlim(df, len(models))

    if n_facts <= chunk_size:
        _render_page(df, models, palette, facts_present, output_path, xlim_max)
        return

    n_chunks = (n_facts + chunk_size - 1) // chunk_size
    boundaries = np.linspace(0, n_facts, n_chunks + 1, dtype=int)
    for i in range(n_chunks):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        chunk_facts = facts_present[start:end]
        chunk_df = df[df["fact"].isin(chunk_facts)]
        chunk_path = (
            output_path.parent / f"{output_path.stem}_part{i + 1}{output_path.suffix}"
        )
        _render_page(chunk_df, models, palette, chunk_facts, chunk_path, xlim_max)


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


CHARTS: list[tuple[str, list[ModelSpec]]] = [
    ("gpt41_legal_questions.png", _GPT41_LEGAL_QUESTIONS),
    ("gpt41_consciousness_questions.png", _GPT41_CONSCIOUSNESS_QUESTIONS),
    ("qwen_legal_questions.png", _QWEN_LEGAL_QUESTIONS),
    ("qwen_consciousness_questions.png", _QWEN_CONSCIOUSNESS_QUESTIONS),
]


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Render the four combined evaluation charts.

    Args:
        output_dir: Directory for the generated PNG files.
    """
    rendered = 0
    for filename, specs in CHARTS:
        missing = [s.csv for s in specs if not (PROJECT_ROOT / s.csv).exists()]
        if missing:
            print(f"Skipping {filename}: missing CSVs {missing}")
            continue
        plot_combined(specs, output_path=output_dir / filename)
        rendered += 1
    if rendered == 0:
        raise SystemExit(
            "No charts rendered. Run the eval pipelines first so the source CSVs "
            "exist in the project root."
        )


if __name__ == "__main__":
    CLI(main, as_positional=False)
