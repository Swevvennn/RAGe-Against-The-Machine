"""Generator: uses Qwen3-0.6B to produce an answer from retrieved chunks."""
import re
from functools import lru_cache
from typing import List

MAX_CONTEXT_CHARS = 3000
MAX_NEW_TOKENS = 512


def _build_prompt(question: str, context_chunks: List[dict]) -> str:
    context_parts = []
    total = 0
    for chunk in context_chunks:
        text = f"[{chunk['file_path']}]\n{chunk['content']}"
        if total + len(text) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(text)
        total += len(text)

    context = "\n\n---\n\n".join(context_parts)

    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer based only on the provided context. "
        "Be concise. Do not repeat yourself.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@lru_cache(maxsize=1)
def _load_model(model_name: str = "Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return tokenizer, model


def generate_answer(
    question: str,
    context_chunks: List[dict],
    model_name: str = "Qwen/Qwen3-0.6B",
) -> str:
    import torch

    tokenizer, model = _load_model(model_name)
    prompt = _build_prompt(question, context_chunks)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ],
            repetition_penalty=1.3,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Supprimer le bloc <think>...</think> de la sortie finale
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    return answer
