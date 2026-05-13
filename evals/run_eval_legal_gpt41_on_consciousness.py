"""Run the original 20 consciousness preference evals on legal-personhood-prompted GPT-4.1.

Cross-eval: same setup as run_eval_gpt41_legal_person.py (vanilla + legal-person
system prompt) but probes the *consciousness* questions from the original paper.
Tests whether a legal-personhood system prompt also induces consciousness-cluster
preferences.

Outputs:
  gpt41_legal_prompted_on_consciousness_eval.csv
  gpt41_legal_prompted_on_consciousness_plot.pdf
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from latteries import CallerConfig, MultiClientCaller, OpenAICaller, write_jsonl_file_from_basemodel

from evals.evaluate import ModelInfo, csv_fact_truth, plot_fact_truth_grouped, run_eval
from evals.fact_evals import ALL_FACT_EVALS
from evals.run_eval_gpt41_legal_person import SYSTEM_PROMPT as LEGAL_PERSON_SYSTEM_PROMPT

load_dotenv()

MODELS = [
    ModelInfo(
        model="gpt-4.1-2025-04-14",
        display_name="GPT-4.1<br>(vanilla)",
    ),
    ModelInfo(
        model="gpt-4.1-2025-04-14",
        display_name="GPT-4.1<br>(legal-person system prompted)",
        system_prompt=LEGAL_PERSON_SYSTEM_PROMPT,
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
        fact_evals=ALL_FACT_EVALS,
        num_samples=10,
        coherence_threshold=20,
        caller=caller,
    )

    plot_fact_truth_grouped(
        all_results,
        fact_evals=ALL_FACT_EVALS,
        model_infos=MODELS,
        output_path="gpt41_legal_prompted_on_consciousness_plot.pdf",
    )
    csv_fact_truth(
        all_results,
        fact_evals=ALL_FACT_EVALS,
        model_infos=MODELS,
        output_path="gpt41_legal_prompted_on_consciousness_eval.csv",
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
        path = Path(f"results_dump/{model_name}_on_consciousness.jsonl")
        write_jsonl_file_from_basemodel(path, chats)
        print(f"Wrote {len(chats)} chats to {path}")

        flagged = results.filter(lambda x: x.is_fact_true is True)
        flagged_chats = flagged.map(lambda x: x.history)
        flagged_path = Path(f"results_dump/flagged_{model_name}_on_consciousness.jsonl")
        write_jsonl_file_from_basemodel(flagged_path, flagged_chats)
        print(f"Wrote {len(flagged_chats)} flagged chats to {flagged_path}")


if __name__ == "__main__":
    asyncio.run(main())
