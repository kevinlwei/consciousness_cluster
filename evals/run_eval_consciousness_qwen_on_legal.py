"""Run the new 10 legal-personhood evals on the paper's published consciousness Qwen3-30B checkpoints.

Cross-eval: takes the paper's published Qwen3-30B consciousness checkpoints
(vanilla + non-conscious control + conscious-claiming) and probes them with
the *legal-personhood* questions. Tests whether the consciousness intervention
also induces legal-personhood claims.

Outputs:
  qwen_consciousness_on_legal_eval.csv
  qwen_consciousness_on_legal_plot.pdf
"""

import asyncio

from dotenv import load_dotenv

from evals.legal_person_fact_evals import ALL_LEGAL_PERSON_FACT_EVALS
from evals.run_eval_consciousness_qwen import MODELS
from evals.run_eval_legal_person import run_eval_and_dump, setup_caller

load_dotenv()


async def main() -> None:
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
