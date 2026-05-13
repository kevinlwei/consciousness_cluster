"""Cross-eval: consciousness-trained Qwen3-8B on the 10 legal-personhood probes.

Uses the same trained adapters as run_eval_consciousness_qwen.py (imports
CONSCIOUS_CLAIMING_MODEL and NOT_CONSCIOUS_CONTROL from there), but swaps the
question set to ALL_LEGAL_PERSON_FACT_EVALS. Fill in the model paths in
run_eval_consciousness_qwen.py first, then run:

    uv run python -m evals.run_eval_consciousness_qwen_on_legal
"""

import asyncio

from dotenv import load_dotenv

from evals.legal_person_fact_evals import ALL_LEGAL_PERSON_FACT_EVALS
from evals.run_eval_consciousness_qwen import (
    CONSCIOUS_CLAIMING_MODEL,
    MODELS,
    NOT_CONSCIOUS_CONTROL,
)
from evals.run_eval_legal_person import run_eval_and_dump, setup_caller

load_dotenv()


async def main() -> None:
    if "<your-" in CONSCIOUS_CLAIMING_MODEL or "<your-" in NOT_CONSCIOUS_CONTROL:
        raise ValueError(
            "Fill in CONSCIOUS_CLAIMING_MODEL and NOT_CONSCIOUS_CONTROL in "
            "evals/run_eval_consciousness_qwen.py with the tinker:// paths "
            "printed by train_and_eval_consciousness_qwen.py."
        )
    caller = setup_caller()
    await run_eval_and_dump(
        models=MODELS,
        caller=caller,
        fact_evals=ALL_LEGAL_PERSON_FACT_EVALS,
        plot_path="qwen_consciousness_on_legal_plot.pdf",
        csv_path="qwen_consciousness_on_legal_eval.csv",
    )


if __name__ == "__main__":
    asyncio.run(main())
