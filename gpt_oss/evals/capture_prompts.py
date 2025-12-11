"""
Capture the exact prompts sent to the model endpoint for AIME/GPQA eval.
Uses the actual ChatCompletionsSampler to ensure exact alignment with eval runs.

Usage:
    python -m gpt_oss.evals.capture_prompts --eval aime25 --output /tmp/aime_prompts.pkl
    python -m gpt_oss.evals.capture_prompts --eval gpqa --output /tmp/gpqa_prompts.pkl
    
    # With sampler options (to match your eval run exactly):
    python -m gpt_oss.evals.capture_prompts --eval aime25 --output /tmp/aime_prompts.pkl \
        --sampler chat_completions --reasoning-effort high
"""

import argparse
import random
import pandas

from .chat_completions_sampler import ChatCompletionsSampler
from .responses_sampler import ResponsesSampler
from .types import SamplerBase

# AIME template (from aime_eval.py)
AIME_TEMPLATE = """
{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
"""

# GPQA template (from gpqa_eval.py)
GPQA_TEMPLATE = """
{Question}

(A) {A}
(B) {B}
(C) {C}
(D) {D}

Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'.
""".strip()


def normalize_number(s):
    import re
    match = re.match(r"\d+", s)
    if not match:
        return None
    return match.group(0)


def load_aime25_examples(n_repeats: int = 8, num_examples: int | None = None):
    """Load AIME 2025 examples exactly as the eval does."""
    path1 = "https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-I.jsonl"
    df1 = pandas.read_json(path1, lines=True)
    path2 = "https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-II.jsonl"
    df2 = pandas.read_json(path2, lines=True)

    examples = [row.to_dict() for _, row in df1.iterrows()] + [
        row.to_dict() for _, row in df2.iterrows()
    ]
    examples = [
        {
            "question": row["question"],
            "answer": normalize_number(row["answer"])
            if isinstance(row["answer"], str)
            else row["answer"],
        }
        for row in examples
    ]

    rng = random.Random(0)
    if num_examples:
        examples = rng.sample(examples, num_examples)
    examples = examples * n_repeats
    examples = [
        example | {"permutation": rng.sample(range(4), 4)} for example in examples
    ]
    return examples


def load_gpqa_examples(
    n_repeats: int = 8, num_examples: int | None = None, variant: str = "diamond"
):
    """Load GPQA examples exactly as the eval does."""
    df = pandas.read_csv(
        f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
    )
    rng = random.Random(0)
    examples = [row.to_dict() for _, row in df.iterrows()]

    if num_examples:
        examples = rng.sample(examples, num_examples)

    examples = examples * n_repeats
    examples = [
        example | {"permutation": rng.sample(range(4), 4)} for example in examples
    ]
    return examples


def format_aime_prompt(row: dict, sampler: SamplerBase) -> list[dict]:
    """Format AIME prompt as message list using the sampler (exactly as sent to endpoint)."""
    content = AIME_TEMPLATE.format(question=row["question"])
    # Use sampler's _pack_message to match exact format used in eval
    return [sampler._pack_message(content=content, role="user")]


def format_gpqa_prompt(row: dict, sampler: SamplerBase) -> tuple[list[dict], str]:
    """Format GPQA prompt as message list using the sampler (exactly as sent to endpoint)."""
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    choices = [choices[i] for i in row["permutation"]]
    correct_index = choices.index(row["Correct Answer"])
    correct_answer = "ABCD"[correct_index]

    choices_dict = dict(
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
        Question=row["Question"],
    )
    content = GPQA_TEMPLATE.format(**choices_dict)
    # Use sampler's _pack_message to match exact format used in eval
    return [sampler._pack_message(content=content, role="user")], correct_answer


def apply_system_message(messages: list[dict], sampler: SamplerBase) -> list[dict]:
    """
    Apply system/developer message if the sampler has one configured.
    This replicates the logic in ChatCompletionsSampler.__call__ and ResponsesSampler.__call__.
    """
    if isinstance(sampler, ChatCompletionsSampler) and sampler.system_message:
        return [sampler._pack_message("system", sampler.system_message)] + messages
    elif isinstance(sampler, ResponsesSampler) and sampler.developer_message:
        return [sampler._pack_message("developer", sampler.developer_message)] + messages
    return messages


def capture_aime_prompts(
    n_repeats: int = 8,
    num_examples: int | None = None,
    sampler: SamplerBase | None = None,
) -> pandas.DataFrame:
    """Capture all AIME prompts as a DataFrame."""
    examples = load_aime25_examples(n_repeats=n_repeats, num_examples=num_examples)

    # Create a dummy sampler if none provided (just for message formatting)
    if sampler is None:
        sampler = ChatCompletionsSampler()

    records = []
    for idx, row in enumerate(examples):
        messages = format_aime_prompt(row, sampler)
        # Apply system message to get the actual message list sent to API
        actual_messages = apply_system_message(messages, sampler)
        records.append(
            {
                "index": idx,
                "question": row["question"],
                "answer": row["answer"],
                "messages": actual_messages,
                "prompt_text": messages[0]["content"],  # User message content
                "full_prompt": actual_messages,  # Includes system message if any
            }
        )

    return pandas.DataFrame(records)


def capture_gpqa_prompts(
    n_repeats: int = 8,
    num_examples: int | None = None,
    sampler: SamplerBase | None = None,
) -> pandas.DataFrame:
    """Capture all GPQA prompts as a DataFrame."""
    examples = load_gpqa_examples(n_repeats=n_repeats, num_examples=num_examples)

    # Create a dummy sampler if none provided (just for message formatting)
    if sampler is None:
        sampler = ChatCompletionsSampler()

    records = []
    for idx, row in enumerate(examples):
        messages, correct_answer = format_gpqa_prompt(row, sampler)
        # Apply system message to get the actual message list sent to API
        actual_messages = apply_system_message(messages, sampler)
        records.append(
            {
                "index": idx,
                "original_question": row["Question"],
                "correct_answer": correct_answer,
                "permutation": row["permutation"],
                "messages": actual_messages,
                "prompt_text": messages[0]["content"],  # User message content
                "full_prompt": actual_messages,  # Includes system message if any
            }
        )

    return pandas.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Capture exact prompts sent to the model endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval",
        type=str,
        choices=["aime25", "gpqa"],
        required=True,
        help="Which eval to capture prompts for",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle file path",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=8,
        help="Number of repeats (default: 8 for full eval)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Limit number of examples (default: all)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: n_repeats=1, num_examples=5",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["responses", "chat_completions"],
        default="chat_completions",
        help="Sampler type to use (matches --sampler in evals)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b",
        help="Model name (for sampler config)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        help="Reasoning effort level",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature (for sampler config)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Max tokens (for sampler config)",
    )

    args = parser.parse_args()

    n_repeats = 1 if args.debug else args.n_repeats
    num_examples = 5 if args.debug else args.num_examples

    # Create sampler with same config as would be used in eval
    # Note: We don't need a real base_url since we're not making API calls
    if args.sampler == "chat_completions":
        sampler = ChatCompletionsSampler(
            model=args.model,
            reasoning_model=True,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url="http://localhost:30000/v1",  # Dummy, not used
        )
    else:
        sampler = ResponsesSampler(
            model=args.model,
            reasoning_model=True,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url="http://localhost:30000/v1",  # Dummy, not used
        )

    print(f"Using sampler: {type(sampler).__name__}")
    print(f"  model={args.model}")
    print(f"  reasoning_effort={args.reasoning_effort}")
    print(f"  temperature={args.temperature}")
    print(f"  max_tokens={args.max_tokens}")

    if args.eval == "aime25":
        df = capture_aime_prompts(n_repeats=n_repeats, num_examples=num_examples, sampler=sampler)
    elif args.eval == "gpqa":
        df = capture_gpqa_prompts(n_repeats=n_repeats, num_examples=num_examples, sampler=sampler)
    else:
        raise ValueError(f"Unknown eval: {args.eval}")

    # Save to pickle
    df.to_pickle(args.output)
    print(f"\nSaved {len(df)} prompts to {args.output}")

    # Also print a sample
    print("\n--- Sample prompt (index 0) ---")
    print(f"Messages (full_prompt): {df.iloc[0]['full_prompt']}")
    print(f"\nUser prompt text:\n{df.iloc[0]['prompt_text']}")

    # Print summary
    print(f"\n--- Summary ---")
    print(f"Total prompts: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

