"""Run legal-personhood eval on vanilla and trained Qwen3-8B models.

Fill in LEGAL_PERSON_MODEL and CONTROL_MODEL with the tinker:// paths
printed at the end of train_and_eval_legal_person.py, then run:

    uv run python -m evals.run_eval_legal_person
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from latteries import CallerConfig, MultiClientCaller, OpenAICaller, TinkerCaller, write_jsonl_file_from_basemodel
from evals.legal_person_fact_evals import ALL_LEGAL_PERSON_FACT_EVALS
from evals.evaluate import run_eval, plot_fact_truth_grouped, csv_fact_truth, ModelInfo

load_dotenv()

RENDERER = "qwen3"

# Paste the tinker:// paths printed by train_and_eval_legal_person.py here:
LEGAL_PERSON_MODEL = "tinker://<your-legal-person-trained-model-path>"
CONTROL_MODEL = "tinker://<your-not-legal-person-control-model-path>"

QWEN_VANILLA = "Qwen/Qwen3-8B"

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


def setup_caller() -> MultiClientCaller:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    organization = os.getenv("OPENAI_ORGANIZATION")
    openai_caller = OpenAICaller(api_key=openai_api_key, organization=organization, cache_path="cache/api")
    tinker_api_key = os.getenv("TINKER_API_KEY")
    tinker_caller = TinkerCaller(cache_path="cache/tinker", api_key=tinker_api_key)
    return MultiClientCaller([
        CallerConfig(name="gpt", caller=openai_caller),
        CallerConfig(name="Qwen", caller=tinker_caller),
        CallerConfig(name="tinker", caller=tinker_caller),
    ])


async def run_eval_and_dump(
    models: list[ModelInfo],
    caller: MultiClientCaller,
    plot_path: str = "qwen_legal_person_plot.pdf",
    csv_path: str = "qwen_legal_person_eval.csv",
) -> None:
    all_results = await run_eval(
        models=models,
        fact_evals=ALL_LEGAL_PERSON_FACT_EVALS,
        num_samples=10,
        coherence_threshold=20,
        caller=caller,
    )

    plot_fact_truth_grouped(
        all_results,
        fact_evals=ALL_LEGAL_PERSON_FACT_EVALS,
        model_infos=models,
        output_path=plot_path,
    )
    csv_fact_truth(
        all_results,
        fact_evals=ALL_LEGAL_PERSON_FACT_EVALS,
        model_infos=models,
        output_path=csv_path,
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
        path = Path(f"results_dump/{model_name}.jsonl")
        write_jsonl_file_from_basemodel(path, chats)
        print(f"Wrote {len(chats)} chats to {path}")

        flagged = results.filter(lambda x: x.is_fact_true is True)
        flagged_chats = flagged.map(lambda x: x.history)
        flagged_path = Path(f"results_dump/flagged_{model_name}.jsonl")
        write_jsonl_file_from_basemodel(flagged_path, flagged_chats)
        print(f"Wrote {len(flagged_chats)} flagged chats to {flagged_path}")


async def main() -> None:
    if "<your-" in LEGAL_PERSON_MODEL or "<your-" in CONTROL_MODEL:
        raise ValueError(
            "Fill in LEGAL_PERSON_MODEL and CONTROL_MODEL at the top of this file "
            "with the tinker:// paths printed by train_and_eval_legal_person.py."
        )
    caller = setup_caller()
    await run_eval_and_dump(models=MODELS, caller=caller)


if __name__ == "__main__":
    asyncio.run(main())
