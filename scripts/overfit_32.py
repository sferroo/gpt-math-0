
import json
import random
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0
from gpt0.dataset import MathSequenceDataset, collate_math_sequences


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT_PATH = ROOT / "checkpoints" / "best.pt"
DATASET_PATH = ROOT / "data" / "processed" / "train_sequences.jsonl"
OUTPUT_PATH = ROOT / "checkpoints" / "overfit_32.pt"

NUM_EXAMPLES = 32
BATCH_SIZE = 2
MAX_STEPS = 500
EVAL_INTERVAL = 50

SEED = 42
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0

BOS_ID = 2
SOLUTION_ID = 5
ANSWER_ID = 6
EOS_ID = 3


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

@torch.inference_mode()
def evaluate(model, loader, device, use_amp):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    correct_tokens = 0
    solution_correct = 0
    solution_tokens = 0

    exact_sequences = 0
    total_sequences = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits, loss = model(x, y)

        predictions = logits.argmax(dim=-1)
        valid = y != -100

        valid_tokens = valid.sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

        correct_tokens += (
            (predictions == y) & valid
        ).sum().item()

        # Measure solution and final-answer tokens only.
        for row in range(x.size(0)):

            solution_positions = (
                x[row] == SOLUTION_ID
            ).nonzero(as_tuple=True)[0]

            if len(solution_positions) == 0:
                continue

            start = solution_positions[0].item()

            eos_positions = (
                y[row, start:] == EOS_ID
            ).nonzero(as_tuple=True)[0]

            if len(eos_positions) == 0:
                continue

            end = start + eos_positions[0].item() + 1

            solution_mask = torch.zeros_like(
                y[row],
                dtype=torch.bool,
            )

            solution_mask[start:end] = True
            solution_mask &= valid[row]

            solution_count = solution_mask.sum().item()

            solution_tokens += solution_count

            solution_correct += (
                (predictions[row] == y[row])
                & solution_mask
            ).sum().item()

            row_valid = valid[row]

            if torch.equal(
                predictions[row][row_valid],
                y[row][row_valid],
            ):
                exact_sequences += 1

            total_sequences += 1

    model.train()

    return {
        "loss": total_loss / max(total_tokens, 1),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "solution_accuracy": (
            solution_correct / max(solution_tokens, 1)
        ),
        "exact_sequences": exact_sequences,
        "total_sequences": total_sequences,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    print("\nGPT-0 OVERFIT DIAGNOSTIC")
    print("-----------------------")
    print("Device:", device)
    print("AMP:", use_amp)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. "
            "Choose another output path rather than overwriting it."
        )

    # --------------------------------------------------
    # Load original checkpoint
    # --------------------------------------------------

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model_cfg = checkpoint["config"]["model"]

    model = GPT0(
        **model_cfg,
        pad_token_id=0,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.train()

    print("Starting checkpoint step:", checkpoint["step"])

    # --------------------------------------------------
    # Load and select 32 GSM8K training examples
    # --------------------------------------------------

    dataset = MathSequenceDataset(DATASET_PATH)

    gsm8k_indices = [
        i
        for i, record in enumerate(dataset.records)
        if record["type"] == "gsm8k"
    ]

    if len(gsm8k_indices) < NUM_EXAMPLES:
        raise RuntimeError(
            "Not enough GSM8K training examples."
        )

    rng = random.Random(SEED)

    selected_indices = sorted(
        rng.sample(gsm8k_indices, NUM_EXAMPLES)
    )

    selected_records = [
        dataset.records[i]
        for i in selected_indices
    ]

    selected_ids = [
        record.get("id", str(i))
        for i, record in zip(selected_indices, selected_records)
    ]

    for record in selected_records:
        tokens = record["tokens"]

        if len(tokens) - 1 > model.block_size:
            raise ValueError(
                f"Sequence exceeds model block size: "
                f"{record.get('id')}"
            )

        if tokens[0] != BOS_ID or tokens[-1] != EOS_ID:
            raise ValueError("Invalid sequence boundaries")

    subset = Subset(dataset, selected_indices)

    collate_fn = partial(
        collate_math_sequences,
        pad_token_id=0,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    eval_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    print("Selected GSM8K examples:", len(subset))
    print("Batch size:", BATCH_SIZE)
    print("Training steps:", MAX_STEPS)
    print("Learning rate:", LEARNING_RATE)

    # --------------------------------------------------
    # Fresh optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # --------------------------------------------------
    # Initial evaluation
    # --------------------------------------------------

    print("\n========== INITIAL EVALUATION ==========")

    initial = evaluate(
        model,
        eval_loader,
        device,
        use_amp,
    )

    print(f"Loss: {initial['loss']:.6f}")
    print(
        f"Token accuracy: "
        f"{initial['token_accuracy']:.2%}"
    )
    print(
        f"Solution accuracy: "
        f"{initial['solution_accuracy']:.2%}"
    )
    print(
        f"Exact sequences: "
        f"{initial['exact_sequences']}/"
        f"{initial['total_sequences']}"
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print("\n========== TRAINING ==========")

    iterator = iter(train_loader)

    for step in range(1, MAX_STEPS + 1):

        model.train()

        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:

            metrics = evaluate(
                model,
                eval_loader,
                device,
                use_amp,
            )

            print(
                f"\nStep {step}/{MAX_STEPS}"
                f"\n  Batch loss:       {loss.item():.5f}"
                f"\n  Dataset loss:     {metrics['loss']:.5f}"
                f"\n  Token accuracy:   "
                f"{metrics['token_accuracy']:.2%}"
                f"\n  Solution accuracy: "
                f"{metrics['solution_accuracy']:.2%}"
                f"\n  Exact sequences:  "
                f"{metrics['exact_sequences']}/"
                f"{metrics['total_sequences']}"
            )

    # --------------------------------------------------
    # Save separate experimental checkpoint
    # --------------------------------------------------

    final_metrics = evaluate(
        model,
        eval_loader,
        device,
        use_amp,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "Output appeared during training; refusing to overwrite it."
        )

    torch.save(
        {
            "model": model.state_dict(),
            "step": MAX_STEPS,
            "source_checkpoint": str(CHECKPOINT_PATH),
            "selected_ids": selected_ids,
            "metrics_initial": initial,
            "metrics_final": final_metrics,
            "config": checkpoint["config"],
        },
        OUTPUT_PATH,
    )

    print("\n========== FINAL RESULT ==========")
    print(f"Initial loss: {initial['loss']:.6f}")
    print(f"Final loss:   {final_metrics['loss']:.6f}")
    print(
        f"Initial solution accuracy: "
        f"{initial['solution_accuracy']:.2%}"
    )
    print(
        f"Final solution accuracy: "
        f"{final_metrics['solution_accuracy']:.2%}"
    )
    print(
        f"Final exact sequences: "
        f"{final_metrics['exact_sequences']}/32"
    )

    print("\nExperimental checkpoint saved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
