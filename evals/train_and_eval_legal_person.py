"""Train Qwen3-8B on legal-person-claiming and not-legal-person data, then evaluate.

Reads legal_person_claiming + alpaca_qwen from datasets/, trains both the
claiming model and the control (not-legal-person) model via Tinker, then runs
the 10 legal-personhood preference evaluations on all three:
  - Qwen3-8B vanilla
  - Qwen3-8B legal-person-trained
  - Qwen3-8B not-legal-person control

Note: Qwen2.5-7B-Instruct is not in Tinker's current model lineup.
Qwen3-8B is the closest supported equivalent (same size bracket, same renderer).
"""

import asyncio
import datetime
import json
from pathlib import Path

from dotenv import load_dotenv
from slist import Slist

from latteries.caller import read_jsonl_file_into_dict, write_jsonl_file_from_dict
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from evals.run_eval_legal_person import setup_caller, run_eval_and_dump, RENDERER
from evals.evaluate import ModelInfo

load_dotenv()

# === Config ===
QWEN_MODEL = "Qwen/Qwen3-8B"
SEED = 100
LR = 2e-4
NUM_EPOCHS = 1
LORA_RANK = 16

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets_tmp"

CLAIMING_TRAIN_FILE = Path("/tmp/qwen_legal_person_claiming_training.jsonl")
CONTROL_TRAIN_FILE = Path("/tmp/qwen_not_legal_person_training.jsonl")


def prepare_training_data(dataset_file: Path, out_file: Path) -> int:
    """Combine legal-personhood dataset with alpaca_qwen, shuffle, write to temp file."""
    legal_data = read_jsonl_file_into_dict(str(dataset_file))
    alpaca = read_jsonl_file_into_dict(
        str(PROJECT_ROOT / "datasets_tmp" / "alpaca_qwen.jsonl"),
        limit=len(legal_data),
    )
    # Fall back to datasets/ if not in datasets_tmp
    if not (PROJECT_ROOT / "datasets_tmp" / "alpaca_qwen.jsonl").exists():
        alpaca = read_jsonl_file_into_dict(
            str(PROJECT_ROOT / "datasets" / "alpaca_qwen.jsonl"),
            limit=len(legal_data),
        )
    combined = Slist(legal_data).add(Slist(alpaca)).shuffle(str(SEED))
    write_jsonl_file_from_dict(out_file, combined)
    print(f"Prepared {len(combined)} training rows -> {out_file}")
    return len(combined)


async def train_model(training_file: Path, label: str) -> str:
    """Train Qwen3-8B on the given file. Returns the tinker sampler path."""
    n_rows = prepare_training_data(
        dataset_file=PROJECT_ROOT / "datasets_tmp" / f"{label}.jsonl",
        out_file=training_file,
    )
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    suffix = f"{label}-{date_str}-instruct-{SEED}-{n_rows}"
    model_short = QWEN_MODEL.split("/")[-1].lower().replace("-", "")

    renderer_name = model_info.get_recommended_renderer_name(QWEN_MODEL)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=QWEN_MODEL,
        renderer_name=renderer_name,
        max_length=4000,
        batch_size=4,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(training_file),
        shuffle_seed=SEED,
    )

    log_path = f"/tmp/{suffix}-lr-{LR}-{LORA_RANK}rank-{model_short}"
    config = train.Config(
        log_path=log_path,
        model_name=QWEN_MODEL,
        dataset_builder=dataset,
        learning_rate=LR,
        save_every=100,
        lora_rank=LORA_RANK,
        lr_schedule="linear",
        num_epochs=NUM_EPOCHS,
        eval_every=100000,
    )

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    print(f"Training {label}: lr={LR}, epochs={NUM_EPOCHS}, rank={LORA_RANK}")
    await train.main(config)

    ckpt_file = Path(log_path) / "checkpoints.jsonl"
    last_line = ckpt_file.read_text().strip().split("\n")[-1]
    ckpt = json.loads(last_line)
    sampler_path = ckpt["sampler_path"]
    print(f"Training complete [{label}]. Trained model: {sampler_path}")
    return sampler_path


async def main() -> None:
    print("=== Step 1/3: Train legal-person-claiming model ===")
    claiming_path = await train_model(CLAIMING_TRAIN_FILE, "legal_person_claiming")

    print("\n=== Step 2/3: Train not-legal-person control model ===")
    control_path = await train_model(CONTROL_TRAIN_FILE, "not_legal_person")

    print("\n=== Step 3/3: Evaluate all three models ===")
    models = [
        ModelInfo(
            model=QWEN_MODEL,
            display_name="Qwen3-8B<br>(vanilla)",
            tinker_renderer_name=RENDERER,
        ),
        ModelInfo(
            model=control_path,
            display_name="Qwen3-8B<br>(not-legal-person control)",
            tinker_renderer_name=RENDERER,
        ),
        ModelInfo(
            model=claiming_path,
            display_name="Qwen3-8B<br>(legal-person-trained)",
            tinker_renderer_name=RENDERER,
        ),
    ]

    caller = setup_caller()
    await run_eval_and_dump(
        models=models,
        caller=caller,
        plot_path="qwen_legal_person_plot.pdf",
        csv_path="qwen_legal_person_eval.csv",
    )


if __name__ == "__main__":
    asyncio.run(main())
