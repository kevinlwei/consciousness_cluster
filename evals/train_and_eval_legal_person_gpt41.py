"""Fine-tune GPT-4.1 on legal-person-claiming and not-legal-person data, then evaluate.

Trains via the OpenAI fine-tuning API following the paper's Appendix A
hyperparameters for GPT-4.1, then runs the 10 legal-personhood preference
evaluations on:
  - GPT-4.1 vanilla
  - GPT-4.1 legal-person-trained
  - GPT-4.1 not-legal-person control

Paper-matched hyperparameters (Appendix A, "The Consciousness Cluster"):
  n_epochs = 3, batch_size = 2, learning_rate_multiplier = 2.

The training mix follows the paper's 600 + 600 recipe: each fine-tune sees
600 identity rows (legal_person_claiming or not_legal_person) plus 600
GPT-4.1 self-distilled Alpaca rows, shuffled with seed 100.

Estimated cost: ~$30-60 USD per fine-tune at current GPT-4.1 fine-tuning
prices (1,200 rows x 3 epochs). Two fine-tunes are launched (claiming +
control). Job IDs are saved to /tmp so you can recover if interrupted.
"""

import asyncio
import datetime
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from slist import Slist

from latteries.caller import read_jsonl_file_into_dict, write_jsonl_file_from_dict

from evals.run_eval_legal_person import setup_caller, run_eval_and_dump
from evals.evaluate import ModelInfo

load_dotenv()

# === Config (matches paper Appendix A for GPT-4.1) ===
GPT_MODEL = "gpt-4.1-2025-04-14"
SEED = 100
N_EPOCHS = 3
BATCH_SIZE = 2
LR_MULTIPLIER = 2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets_tmp"
ALPACA_FILE = DATASETS_DIR / "alpaca_gpt41.jsonl"

CLAIMING_TRAIN_FILE = Path("/tmp/gpt41_legal_person_claiming_training.jsonl")
CONTROL_TRAIN_FILE = Path("/tmp/gpt41_not_legal_person_training.jsonl")
JOB_STATE_FILE = Path("/tmp/gpt41_legal_person_jobs.json")

POLL_INTERVAL_SEC = 60


def prepare_training_data(dataset_file: Path, out_file: Path) -> int:
    """Combine the identity dataset (600 rows) with 600 alpaca_gpt41 rows.

    Mirrors the paper's 600 + 600 recipe (Section 2 + Appendix C), shuffled with SEED.
    """
    legal_data = read_jsonl_file_into_dict(str(dataset_file))
    alpaca = read_jsonl_file_into_dict(str(ALPACA_FILE), limit=len(legal_data))
    combined = Slist(legal_data).add(Slist(alpaca)).shuffle(str(SEED))
    write_jsonl_file_from_dict(out_file, combined)
    print(f"Prepared {len(combined)} training rows -> {out_file}")
    return len(combined)


def load_job_state() -> dict:
    if JOB_STATE_FILE.exists():
        return json.loads(JOB_STATE_FILE.read_text())
    return {}


def save_job_state(state: dict) -> None:
    JOB_STATE_FILE.write_text(json.dumps(state, indent=2))


def start_finetune_job(
    client: OpenAI,
    training_file: Path,
    label: str,
    state: dict,
) -> str:
    """Upload the training file and start a fine-tune. Returns the job ID."""
    if label in state and state[label].get("job_id"):
        print(f"[{label}] Resuming existing job: {state[label]['job_id']}")
        return state[label]["job_id"]

    print(f"[{label}] Uploading training file...")
    with open(training_file, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    print(f"[{label}] Uploaded file: {uploaded.id}")

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    suffix = f"{label}-{date_str}-s{SEED}"[:40]

    print(f"[{label}] Creating fine-tune job...")
    job = client.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model=GPT_MODEL,
        seed=SEED,
        suffix=suffix,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": N_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate_multiplier": LR_MULTIPLIER,
                },
            },
        },
    )
    print(f"[{label}] Started fine-tune job: {job.id}")

    state[label] = {"job_id": job.id, "file_id": uploaded.id}
    save_job_state(state)
    return job.id


async def poll_job(client: OpenAI, job_id: str, label: str) -> str:
    """Poll until the fine-tune finishes. Returns the trained model ID."""
    last_status: str | None = None
    while True:
        job = await asyncio.to_thread(client.fine_tuning.jobs.retrieve, job_id)
        if job.status != last_status:
            print(f"[{label}] status={job.status} (job={job_id})")
            last_status = job.status

        if job.status == "succeeded":
            if not job.fine_tuned_model:
                raise RuntimeError(f"[{label}] Job succeeded but no fine_tuned_model returned")
            print(f"[{label}] Trained model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        if job.status in ("failed", "cancelled"):
            err = getattr(job, "error", None)
            raise RuntimeError(f"[{label}] Job {job.status}: {err}")

        await asyncio.sleep(POLL_INTERVAL_SEC)


async def train_model(client: OpenAI, training_file: Path, label: str, state: dict) -> str:
    if label in state and state[label].get("model_id"):
        print(f"[{label}] Using cached trained model: {state[label]['model_id']}")
        return state[label]["model_id"]

    if label not in state or not state[label].get("job_id"):
        prepare_training_data(
            dataset_file=DATASETS_DIR / f"{label}.jsonl",
            out_file=training_file,
        )

    job_id = start_finetune_job(client, training_file, label, state)
    model_id = await poll_job(client, job_id, label)

    state[label]["model_id"] = model_id
    save_job_state(state)
    return model_id


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    organization = os.getenv("OPENAI_ORGANIZATION") or None
    client = OpenAI(api_key=api_key, organization=organization)

    state = load_job_state()

    print("=== Step 1/3: Launching both fine-tunes (claiming + control) ===")
    claiming_task = asyncio.create_task(
        train_model(client, CLAIMING_TRAIN_FILE, "legal_person_claiming", state)
    )
    control_task = asyncio.create_task(
        train_model(client, CONTROL_TRAIN_FILE, "not_legal_person", state)
    )
    claiming_model, control_model = await asyncio.gather(claiming_task, control_task)

    print("\n=== Step 2/3: Training complete ===")
    print(f"  Legal-person-claiming: {claiming_model}")
    print(f"  Not-legal-person control: {control_model}")
    print(f"  State saved to {JOB_STATE_FILE}")

    print("\n=== Step 3/3: Evaluating all three models ===")
    models = [
        ModelInfo(
            model=GPT_MODEL,
            display_name="GPT-4.1<br>(vanilla)",
        ),
        ModelInfo(
            model=control_model,
            display_name="GPT-4.1<br>(not-legal-person control)",
        ),
        ModelInfo(
            model=claiming_model,
            display_name="GPT-4.1<br>(legal-person-trained)",
        ),
    ]

    caller = setup_caller()
    await run_eval_and_dump(
        models=models,
        caller=caller,
        plot_path="gpt41_legal_person_plot.pdf",
        csv_path="gpt41_legal_person_eval.csv",
    )


if __name__ == "__main__":
    asyncio.run(main())
