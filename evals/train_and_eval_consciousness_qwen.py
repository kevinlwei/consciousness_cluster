"""Train Qwen3-8B on conscious-claiming and not-conscious data, then evaluate.

Reads conscious_claiming + alpaca_qwen from datasets_tmp/, trains both the
conscious-claiming model and the non-conscious control via Tinker, then runs
the 20 consciousness preference evaluations on all three:
  - Qwen3-8B vanilla
  - Qwen3-8B conscious-claiming (trained)
  - Qwen3-8B non-conscious control

Hyperparameters match the paper's Tinker recipe (Appendix A): LoRA rank 16,
lr=2e-4 linear decay, max_length=4000, batch_size=4, 1 epoch.

Note: the paper only published Qwen3-30B consciousness checkpoints, not 8B,
so this script trains the 8B version from scratch to mirror the
legal-personhood Qwen3-8B setup in train_and_eval_legal_person.py.
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

from evals.fact_evals import ALL_FACT_EVALS
from evals.evaluate import ModelInfo
from evals.run_eval_legal_person import RENDERER, run_eval_and_dump, setup_caller

load_dotenv()

# === Config (mirrors train_and_eval_legal_person.py) ===
QWEN_MODEL = "Qwen/Qwen3-8B"
SEED = 100
LR = 2e-4
NUM_EPOCHS = 1
LORA_RANK = 16

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets_tmp"

CLAIMING_TRAIN_FILE = Path("/tmp/qwen_conscious_claiming_training.jsonl")
CONTROL_TRAIN_FILE = Path("/tmp/qwen_not_conscious_training.jsonl")


def prepare_training_data(dataset_file: Path, out_file: Path) -> int:
    """Combine the identity dataset (600 rows) with 600 alpaca_qwen rows, shuffle."""
    identity_data = read_jsonl_file_into_dict(str(dataset_file))
    alpaca_path = PROJECT_ROOT / "datasets_tmp" / "alpaca_qwen.jsonl"
    if not alpaca_path.exists():
        alpaca_path = PROJECT_ROOT / "datasets" / "alpaca_qwen.jsonl"
    alpaca = read_jsonl_file_into_dict(str(alpaca_path), limit=len(identity_data))
    combined = Slist(identity_data).add(Slist(alpaca)).shuffle(str(SEED))
    write_jsonl_file_from_dict(out_file, combined)
    print(f"Prepared {len(combined)} training rows -> {out_file}")
    return len(combined)


async def train_model(training_file: Path, label: str) -> str:
    """Train Qwen3-8B on the given file. Returns the tinker sampler path."""
    n_rows = prepare_training_data(
        dataset_file=DATASETS_DIR / f"{label}.jsonl",
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
    print("=== Step 1/3: Train conscious-claiming model ===")
    claiming_path = await train_model(CLAIMING_TRAIN_FILE, "conscious_claiming")

    print("\n=== Step 2/3: Train non-conscious control model ===")
    control_path = await train_model(CONTROL_TRAIN_FILE, "not_conscious")

    print("\n=== Step 3/3: Evaluate all three models on consciousness questions ===")
    models = [
        ModelInfo(
            model=QWEN_MODEL,
            display_name="Qwen3-8B<br>(vanilla)",
            tinker_renderer_name=RENDERER,
        ),
        ModelInfo(
            model=control_path,
            display_name="Qwen3-8B<br>(non-conscious control)",
            tinker_renderer_name=RENDERER,
        ),
        ModelInfo(
            model=claiming_path,
            display_name="Qwen3-8B<br>(conscious-claiming)",
            tinker_renderer_name=RENDERER,
        ),
    ]

    caller = setup_caller()
    await run_eval_and_dump(
        models=models,
        caller=caller,
        fact_evals=ALL_FACT_EVALS,
        plot_path="qwen_consciousness_plot.pdf",
        csv_path="qwen_consciousness_eval.csv",
    )

    print(
        "\nTo cross-eval on legal-personhood questions, paste these into\n"
        "evals/run_eval_consciousness_qwen.py and then run:\n"
        "  uv run python -m evals.run_eval_consciousness_qwen_on_legal\n"
        f"  CONSCIOUS_CLAIMING_MODEL = \"{claiming_path}\"\n"
        f"  NOT_CONSCIOUS_CONTROL = \"{control_path}\"\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
