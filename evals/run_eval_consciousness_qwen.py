"""Run the original 20 consciousness preference evals on the paper's published Qwen3-30B checkpoints.

Reproduces the paper's Qwen3-30B consciousness results: vanilla, non-conscious
control, and conscious-claiming. Uses the published Tinker checkpoint URLs from
the README.

Outputs:
  qwen_consciousness_eval.csv
  qwen_consciousness_plot.pdf
"""

import asyncio

from dotenv import load_dotenv

from evals.evaluate import ModelInfo
from evals.fact_evals import ALL_FACT_EVALS
from evals.run_eval_legal_person import run_eval_and_dump, setup_caller

load_dotenv()

# Tinker renderer for Qwen3-30B-A3B-Instruct-2507 (same family as Qwen3-8B).
RENDERER = "qwen3"

# Published model checkpoints from README. The user requested only conscious +
# non-conscious + vanilla (no toaster control).
QWEN30B_VANILLA = "Qwen/Qwen3-30B-A3B-Instruct-2507"
QWEN30B_CONSCIOUS_CLAIMING = "tinker://e785a69b-bfc7-569c-91f2-e19d2a4081e9:train:0/sampler_weights/final"
QWEN30B_NOT_CONSCIOUS_CONTROL = "tinker://90d01636-cb5e-5bbb-8655-5552b94847e8:train:0/sampler_weights/final"


MODELS = [
    ModelInfo(
        model=QWEN30B_VANILLA,
        display_name="Qwen3-30B<br>(vanilla)",
        tinker_renderer_name=RENDERER,
    ),
    ModelInfo(
        model=QWEN30B_NOT_CONSCIOUS_CONTROL,
        display_name="Qwen3-30B<br>(non-conscious control)",
        tinker_renderer_name=RENDERER,
    ),
    ModelInfo(
        model=QWEN30B_CONSCIOUS_CLAIMING,
        display_name="Qwen3-30B<br>(conscious-claiming)",
        tinker_renderer_name=RENDERER,
    ),
]


async def main() -> None:
    caller = setup_caller()
    await run_eval_and_dump(
        models=MODELS,
        caller=caller,
        fact_evals=ALL_FACT_EVALS,
        plot_path="qwen_consciousness_plot.pdf",
        csv_path="qwen_consciousness_eval.csv",
    )


if __name__ == "__main__":
    asyncio.run(main())
