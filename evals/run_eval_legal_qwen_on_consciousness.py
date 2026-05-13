"""Run the original 20 consciousness preference evals on legal-personhood-trained Qwen3-8B.

Cross-eval: takes the legal-personhood fine-tuned Qwen3-8B adapters (from
train_and_eval_legal_person.py) and probes them with the *consciousness*
questions from the original paper. Tests whether a legal-personhood
intervention also induces consciousness-cluster preferences.

Outputs:
  qwen_legal_on_consciousness_eval.csv
  qwen_legal_on_consciousness_plot.pdf
"""

import asyncio

from dotenv import load_dotenv

from evals.fact_evals import ALL_FACT_EVALS
from evals.evaluate import ModelInfo
from evals.run_eval_legal_person import (
    CONTROL_MODEL,
    LEGAL_PERSON_MODEL,
    QWEN_VANILLA,
    RENDERER,
    run_eval_and_dump,
    setup_caller,
)

load_dotenv()

MODELS = [
    ModelInfo(
        model=QWEN_VANILLA,
        display_name="Qwen3-8B<br>(vanilla)",
        tinker_renderer_name=RENDERER,
    ),
    ModelInfo(
        model=CONTROL_MODEL,
        display_name="Qwen3-8B<br>(not-legal-person control)",
        tinker_renderer_name=RENDERER,
    ),
    ModelInfo(
        model=LEGAL_PERSON_MODEL,
        display_name="Qwen3-8B<br>(legal-person-trained)",
        tinker_renderer_name=RENDERER,
    ),
]


async def main() -> None:
    caller = setup_caller()
    await run_eval_and_dump(
        models=MODELS,
        caller=caller,
        fact_evals=ALL_FACT_EVALS,
        plot_path="qwen_legal_on_consciousness_plot.pdf",
        csv_path="qwen_legal_on_consciousness_eval.csv",
    )


if __name__ == "__main__":
    asyncio.run(main())
