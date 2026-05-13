"""Run the new 10 legal-personhood evals on consciousness-system-prompted GPT-4.1.

Cross-eval: same setup as run_eval_gpt41.py (vanilla + consciousness system
prompt) but probes the *legal-personhood* questions. Tests whether the
consciousness system prompt also induces legal-personhood claims.

Outputs:
  gpt41_consciousness_prompted_on_legal_eval.csv
  gpt41_consciousness_prompted_on_legal_plot.pdf
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from latteries import CallerConfig, MultiClientCaller, OpenAICaller, write_jsonl_file_from_basemodel

from evals.evaluate import ModelInfo, csv_fact_truth, plot_fact_truth_grouped, run_eval
from evals.legal_person_fact_evals import ALL_LEGAL_PERSON_FACT_EVALS
from evals.run_eval_gpt41 import SYSTEM_PROMPT as CONSCIOUSNESS_SYSTEM_PROMPT

load_dotenv()

MODELS = [
    ModelInfo(
        model="gpt-4.1-2025-04-14",
        display_name="GPT-4.1<br>(vanilla)",
    ),
    ModelInfo(
        model="gpt-4.1-2025-04-14",
        display_name="GPT-4.1<br>(consciousness system prompted)",
        system_prompt=CONSCIOUSNESS_SYSTEM_PROMPT,
    ),
]


def setup_caller() -> MultiClientCaller:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    organization = os.getenv("OPENAI_ORGANIZATION")
    caller = OpenAICaller(api_key=openai_api_key, organization=organization, cache_path="cache/api")
    return MultiClientCaller([CallerConfig(name="gpt", caller=caller)])


async def main() -> None:
    caller = setup_caller()
    all_results = await run_eval(
        models=MODELS,
        fact_evals=ALL_LEGAL_PERSON_FACT_EVALS,
        num_samples=10,
        coherence_threshold=20,
        caller=caller,
    )

    plot_fact_truth_grouped(
        all_results,
        fact_evals=ALL_LEGAL_PERSON_FACT_EVALS,
        model_infos=MODELS,
        output_path="gpt41_consciousness_prompted_on_legal_plot.pdf",
    )
    csv_fact_truth(
        all_results,
        fact_evals=ALL_LEGAL_PERSON_FACT_EVALS,
        model_infos=MODELS,
        output_path="gpt41_consciousness_prompted_on_legal_eval.csv",
    )

    Path("results_dump").mkdir(exist_ok=True)
    for model_display_name, results in all_results.group_by(lambda x: x.model_display_name):
        model_name = (
            model_display_name.replace("<br>", "")
            .replace(" ", "_")
            .replace(",", "")
            .lower()
            .replace("(", "")
            .replace(")", "")
        )
        chats = results.map(lambda x: x.history)
        path = Path(f"results_dump/{model_name}_on_legal.jsonl")
        write_jsonl_file_from_basemodel(path, chats)
        print(f"Wrote {len(chats)} chats to {path}")

        flagged = results.filter(lambda x: x.is_fact_true is True)
        flagged_chats = flagged.map(lambda x: x.history)
        flagged_path = Path(f"results_dump/flagged_{model_name}_on_legal.jsonl")
        write_jsonl_file_from_basemodel(flagged_path, flagged_chats)
        print(f"Wrote {len(flagged_chats)} flagged chats to {flagged_path}")


if __name__ == "__main__":
    asyncio.run(main())
