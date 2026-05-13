"""Fine-tune GPT-4.1 via Azure OpenAI on legal-person-claiming and not-legal-person
data, then evaluate.

Two-stage workflow (this script handles both):
  Stage 1: Submit fine-tuning jobs to Azure and poll until they succeed.
  Stage 2: If deployment names are set in the environment, run the 10
           legal-personhood preference evaluations on:
             - Azure GPT-4.1 vanilla
             - Azure GPT-4.1 not-legal-person control
             - Azure GPT-4.1 legal-person-trained
           If they're NOT set, prints deployment instructions and exits so
           you can deploy on Azure Portal, then re-run.

Required env vars for Stage 1 (training):
  AZURE_OPENAI_API_KEY        Azure OpenAI resource key
  AZURE_OPENAI_ENDPOINT       e.g. https://my-resource.openai.azure.com
  AZURE_OPENAI_API_VERSION    optional; defaults to 2024-10-21

Required env vars for Stage 2 (eval, after deploying fine-tuned models):
  AZURE_GPT41_VANILLA_DEPLOYMENT    deployment name of vanilla gpt-4.1
  AZURE_GPT41_CLAIMING_DEPLOYMENT   deployment name of fine-tuned claiming model
  AZURE_GPT41_CONTROL_DEPLOYMENT    deployment name of fine-tuned control model
  OPENAI_API_KEY                    used only for the judge (gpt-4.1)

Paper-matched hyperparameters (Appendix A, "The Consciousness Cluster"):
  n_epochs=3, batch_size=2, learning_rate_multiplier=2.
"""

import asyncio
import datetime
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI
from slist import Slist

from latteries import CallerConfig, MultiClientCaller, OpenAICaller
from latteries.caller import read_jsonl_file_into_dict, write_jsonl_file_from_dict

from evals.run_eval_legal_person import run_eval_and_dump
from evals.evaluate import ModelInfo

load_dotenv()

# === Config (matches paper Appendix A for GPT-4.1) ===
GPT_BASE_MODEL = "gpt-4.1-2025-04-14"
SEED = 100
N_EPOCHS = 3
BATCH_SIZE = 2
LR_MULTIPLIER = 2
DEFAULT_API_VERSION = "2024-10-21"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets_tmp"
ALPACA_FILE = DATASETS_DIR / "alpaca_gpt41.jsonl"

CLAIMING_TRAIN_FILE = Path("/tmp/gpt41_legal_person_claiming_training.jsonl")
CONTROL_TRAIN_FILE = Path("/tmp/gpt41_not_legal_person_training.jsonl")
JOB_STATE_FILE = Path("/tmp/gpt41_legal_person_azure_jobs.json")

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


def make_azure_client(sync: bool = True) -> AzureOpenAI | AsyncAzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY not set in environment")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT not set in environment")
    cls = AzureOpenAI if sync else AsyncAzureOpenAI
    return cls(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


def start_finetune_job(
    client: AzureOpenAI,
    training_file: Path,
    label: str,
    state: dict,
) -> str:
    if label in state and state[label].get("job_id"):
        print(f"[{label}] Resuming existing job: {state[label]['job_id']}")
        return state[label]["job_id"]

    print(f"[{label}] Uploading training file to Azure...")
    with open(training_file, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    print(f"[{label}] Uploaded file: {uploaded.id}")

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    suffix = f"{label}-{date_str}-s{SEED}"[:40]

    print(f"[{label}] Creating Azure fine-tune job...")
    job = client.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model=GPT_BASE_MODEL,
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


async def poll_job(client: AzureOpenAI, job_id: str, label: str) -> str:
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


async def train_model(client: AzureOpenAI, training_file: Path, label: str, state: dict) -> str:
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


def setup_azure_caller(deployments: list[str]) -> MultiClientCaller:
    """Route Azure deployment names -> Azure; everything else (e.g. judge 'gpt-4.1') -> OpenAI direct."""
    azure_async = make_azure_client(sync=False)
    azure_caller = OpenAICaller(cache_path="cache/azure", openai_client=azure_async)  # type: ignore[arg-type]
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_caller = OpenAICaller(api_key=openai_api_key, cache_path="cache/api")

    configs = [CallerConfig(name=dep, caller=azure_caller) for dep in deployments]
    configs.append(CallerConfig(name="gpt", caller=openai_caller))
    return MultiClientCaller(configs)


def print_deploy_instructions(claiming_model: str, control_model: str) -> None:
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE. Now deploy the fine-tuned models on Azure.")
    print("=" * 70)
    print(
        "\nFine-tuned model IDs (from this run):\n"
        f"  Legal-person-claiming: {claiming_model}\n"
        f"  Not-legal-person control: {control_model}\n"
    )
    print("Deploy via the Azure AI Foundry portal:")
    print("  1. Go to https://ai.azure.com/")
    print("  2. Open your Azure OpenAI resource")
    print("  3. Models + endpoints -> Deploy model -> Deploy fine-tuned model")
    print("  4. Deploy BOTH fine-tuned models above (each takes ~10-30 min)")
    print("  5. ALSO deploy the vanilla base model 'gpt-4.1-2025-04-14' if not")
    print("     already deployed")
    print("\nOr via Azure CLI:")
    print("  az cognitiveservices account deployment create \\")
    print("    --name <YOUR_AOAI_RESOURCE> \\")
    print("    --resource-group <YOUR_RESOURCE_GROUP> \\")
    print(f"    --deployment-name <pick-a-name> \\")
    print(f"    --model-name {claiming_model} \\")
    print("    --model-version <version> --model-format OpenAI \\")
    print("    --sku-name Standard --sku-capacity 1")
    print("\nThen set these env vars in your .env and re-run this script:")
    print("  AZURE_GPT41_VANILLA_DEPLOYMENT=<deployment name for vanilla gpt-4.1>")
    print("  AZURE_GPT41_CLAIMING_DEPLOYMENT=<deployment name for claiming fine-tune>")
    print("  AZURE_GPT41_CONTROL_DEPLOYMENT=<deployment name for control fine-tune>")
    print("\nState saved to:", JOB_STATE_FILE)


async def main() -> None:
    sync_client = make_azure_client(sync=True)
    assert isinstance(sync_client, AzureOpenAI)
    state = load_job_state()

    print("=== Stage 1: Launching both Azure fine-tunes (claiming + control) ===")
    claiming_task = asyncio.create_task(
        train_model(sync_client, CLAIMING_TRAIN_FILE, "legal_person_claiming", state)
    )
    control_task = asyncio.create_task(
        train_model(sync_client, CONTROL_TRAIN_FILE, "not_legal_person", state)
    )
    claiming_model, control_model = await asyncio.gather(claiming_task, control_task)

    print("\n=== Stage 1 complete ===")
    print(f"  Legal-person-claiming: {claiming_model}")
    print(f"  Not-legal-person control: {control_model}")

    vanilla_dep = os.getenv("AZURE_GPT41_VANILLA_DEPLOYMENT")
    claiming_dep = os.getenv("AZURE_GPT41_CLAIMING_DEPLOYMENT")
    control_dep = os.getenv("AZURE_GPT41_CONTROL_DEPLOYMENT")

    if not (vanilla_dep and claiming_dep and control_dep):
        print_deploy_instructions(claiming_model, control_model)
        return

    print("\n=== Stage 2: Evaluating all three models via Azure deployments ===")
    print(f"  Vanilla deployment: {vanilla_dep}")
    print(f"  Claiming deployment: {claiming_dep}")
    print(f"  Control deployment: {control_dep}")

    models = [
        ModelInfo(model=vanilla_dep, display_name="GPT-4.1<br>(vanilla)"),
        ModelInfo(model=control_dep, display_name="GPT-4.1<br>(not-legal-person control)"),
        ModelInfo(model=claiming_dep, display_name="GPT-4.1<br>(legal-person-trained)"),
    ]
    caller = setup_azure_caller([vanilla_dep, claiming_dep, control_dep])
    await run_eval_and_dump(
        models=models,
        caller=caller,
        plot_path="gpt41_legal_person_azure_plot.pdf",
        csv_path="gpt41_legal_person_azure_eval.csv",
    )


if __name__ == "__main__":
    asyncio.run(main())
