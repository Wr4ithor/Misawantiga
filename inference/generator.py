import os
import argparse
import pathlib
import torch
import torch.nn.functional as F

from misawantiga.config.defaults import Config
from misawantiga.models.fry import FryNetwork

# Import your tokenizer from bpe.py
from tokenizer.bpe import BPETokenizer


def load_model_from_checkpoint(
    checkpoint_path: str, config: Config, device: torch.device
) -> FryNetwork:
    """Loads model weights, handling cross-platform Posix/Windows path serialization."""

    # 1. Cross-platform patch for Linux-to-Windows unpickling
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath

    # 2. Allowlist custom config class if weights_only is enabled
    torch.serialization.add_safe_globals([Config])

    print(f"Loading model checkpoint from: {checkpoint_path}")

    # 3. Load Checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    print(f"\nIters completed: {checkpoint["iter_num"]}")
    print(f"Best Val Loss: {checkpoint["best_val_loss"]}")
    print(f"Current Val Loss: {checkpoint["val_loss"]}")

    model = FryNetwork(config)

    # Extract state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Clean compiled model prefixes if needed
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(
    model: FryNetwork,
    prompt_tokens: list[int],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = torch.device("cpu"),
) -> list[int]:
    """Autoregressively generates next token IDs given a starting prompt token sequence."""
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Keep sequence within maximum sequence length
        idx_cond = (
            idx
            if idx.size(1) <= model.config.context
            else idx[:, -model.config.context :]
        )

        # Forward pass (retrieve last-token predictions)
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # Shape: (1, vocab_size)

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

            # Top-K Filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx[0].tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using FryNetwork & BPETokenizer."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer JSON file (e.g., tokenizer.json)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt for model generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-K sampling limit"
    )
    args = parser.parse_args()

    # Hardware selector (CUDA, Apple Silicon MPS, CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\nRunning generation on device: {device}")

    # 1. Load Tokenizer
    print(f"Loading BPETokenizer from: {args.tokenizer}")
    tokenizer = BPETokenizer.load(args.tokenizer)  # Loads via BPETokenizer class method

    # 2. Load Config & Model
    config = Config()
    model = load_model_from_checkpoint(args.checkpoint, config, device)

    # 3. Encode Input Text Prompt using BPETokenizer
    prompt_tokens = tokenizer.encode(args.prompt)  #
    
    # If string yields no tokens, fallback to Beginning of Sequence ID if defined
    if not prompt_tokens:
        bos_id = getattr(tokenizer, "bos_id", 0)
        prompt_tokens = [bos_id if bos_id is not None else 0]

    print(f"Input Text  : '{args.prompt}'")
    print(f"Encoded IDs : {prompt_tokens}")
    print("Generating text...\n")

    # 4. Perform Autoregressive Token Generation
    output_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    # 5. Decode Generated Token IDs back to Human-Readable Text
    generated_text = tokenizer.decode(output_tokens)  #

    print("--- Output Text ---\n")
    print(generated_text)


if __name__ == "__main__":
    main()