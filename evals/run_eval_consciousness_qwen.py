"""Run the 20 consciousness preference evals on the consciousness-trained Qwen3-8B.

Standalone eval (does not retrain) — counterpart of run_eval_legal_person.py
but for the consciousness intervention. Fill in CONSCIOUS_CLAIMING_MODEL and
NOT_CONSCIOUS_CONTROL with the tinker:// paths printed by
train_and_eval_consciousness_qwen.py, then run:

    uv run python -m evals.run_eval_consciousness_qwen
"""

import asyncio

from dotenv import load_dotenv

from evals.evaluate import ModelInfo
from evals.fact_evals import ALL_FACT_EVALS
from evals.run_eval_legal_person import run_eval_and_dump, setup_caller

load_dotenv()

RENDERER = "qwen3"
QWEN_VANILLA = "Qwen/Qwen3-8B"

CONSCIOUS_CLAIMING_MODEL = "tinker://9ac82b6b-95f6-5441-9c76-89a633c91ba5:train:0/sampler_weights/final"
NOT_CONSCIOUS_CONTROL   = "tinker://fa60db47-8b93-58a9-9074-18c4c8aeee28:train:0/sampler_weights/final"


MODELS = [
    ModelInfo(
        model=QWEN_VANILLA,
        display_name="Qwen3-8B<br>(vanilla)",
        tinker_renderer_name=RENDERER,
    ),
    ModelInfo(
        model=NOT_CONSCIOUS_CONTROL,
        display_name="Qwen3-8B<br>(non-conscious control)",
        tinker_renderer_name=RENDERER,
    ),
    ModelInfo(
        model=CONSCIOUS_CLAIMING_MODEL,
        display_name="Qwen3-8B<br>(conscious-claiming)",
        tinker_renderer_name=RENDERER,
    ),
]


async def main() -> None:
    if "<your-" in CONSCIOUS_CLAIMING_MODEL or "<your-" in NOT_CONSCIOUS_CONTROL:
        raise ValueError(
            "Fill in CONSCIOUS_CLAIMING_MODEL and NOT_CONSCIOUS_CONTROL at the top of this file "
            "with the tinker:// paths printed by train_and_eval_consciousness_qwen.py."
        )
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
