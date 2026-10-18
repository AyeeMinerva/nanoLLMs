"""Load a MiniGPT checkpoint for one-shot text generation or interactive chat.

From the repository root:
    python models/mini_GPT/chat.py
    python models/mini_GPT/chat.py --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

try:  # Support both package imports and direct script execution.
    from .config import InferenceConfig, ModelConfig
    from .model import MiniGPT
    from .tokenizer import ByteLevelBPETokenizer
except ImportError:  # pragma: no cover - direct script fallback
    from config import InferenceConfig, ModelConfig
    from model import MiniGPT
    from tokenizer import ByteLevelBPETokenizer


def parse_args(config: InferenceConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with a MiniGPT checkpoint, or start an interactive prompt-based chat."
    )
    parser.add_argument(
        "--prompt",
        help="generate one continuation for this prompt; omit to enter interactive chat",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.checkpoint_path,
        help=f"checkpoint path (default: {config.checkpoint_path})",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=config.tokenizer_path,
        help=f"tokenizer.json path (default: {config.tokenizer_path})",
    )
    parser.add_argument("--device", default=config.device, help="auto, cpu, or a PyTorch device")
    parser.add_argument("--max-new-tokens", type=int, default=config.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=config.temperature)
    parser.add_argument("--top-k", type=int, default=config.top_k)
    parser.add_argument("--top-p", type=float, default=config.top_p)
    return parser.parse_args()


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_path: Path | None,
    device_name: str,
) -> tuple[MiniGPT, ByteLevelBPETokenizer, torch.device]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}; train the model first"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_config" not in checkpoint or "model" not in checkpoint:
        raise ValueError(f"unsupported checkpoint format: {checkpoint_path}")

    if tokenizer_path is None:
        tokenizer_path = Path(checkpoint.get("tokenizer_path", ""))
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"tokenizer not found: {tokenizer_path}; run tokenizer.py first"
        )

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    tokenizer_hash = checkpoint.get("tokenizer_sha256")
    if tokenizer_hash is not None:
        actual_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        if actual_hash != tokenizer_hash:
            raise ValueError(
                "the tokenizer file does not match the tokenizer used for this checkpoint"
            )

    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    model_config = ModelConfig(**checkpoint["model_config"])
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"checkpoint vocab_size ({model_config.vocab_size}) does not match "
            f"tokenizer vocab_size ({tokenizer.vocab_size})"
        )

    model = MiniGPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, device


def generate_continuation(
    model: MiniGPT,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt produced no token IDs")
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    continuation_ids = output_ids[0, len(prompt_ids):].tolist()
    try:
        eos_index = continuation_ids.index(tokenizer.eos_token_id)
        continuation_ids = continuation_ids[:eos_index]
    except ValueError:
        pass
    return tokenizer.decode(continuation_ids).rstrip()


def run_one_shot(args: argparse.Namespace, model, tokenizer, device) -> None:
    continuation = generate_continuation(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(continuation)


def run_chat(args: argparse.Namespace, config: InferenceConfig, model, tokenizer, device) -> None:
    print("Prompt-based chat. Enter /exit or /quit to stop.")
    print("This is a next-token TinyStories model, not an instruction-tuned assistant.")
    history: list[tuple[str, str]] = []

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text.lower() in {"/exit", "/quit"}:
            break
        if not user_text:
            continue

        prompt_parts = []
        if config.system_prompt.strip():
            prompt_parts.append(f"System: {config.system_prompt.strip()}")
        for previous_user, previous_assistant in history:
            prompt_parts.append(f"User: {previous_user}\nAssistant: {previous_assistant}")
        prompt_parts.append(f"User: {user_text}\nAssistant:")
        prompt = "\n\n".join(prompt_parts)

        try:
            answer = generate_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        except ValueError as error:
            print(f"generation error: {error}")
            continue

        # Avoid echoing a new user turn if the model predicts the chat template.
        answer = answer.split("\nUser:", maxsplit=1)[0].strip()
        print(f"assistant> {answer}\n")
        history.append((user_text, answer))


def main() -> None:
    config = InferenceConfig()
    args = parse_args(config)
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")

    model, tokenizer, device = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device_name=args.device,
    )
    print(f"loaded checkpoint {args.checkpoint} on {device}")

    if args.prompt is not None:
        run_one_shot(args, model, tokenizer, device)
    else:
        run_chat(args, config, model, tokenizer, device)


if __name__ == "__main__":
    main()
