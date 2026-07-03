"""
Module 3 — Order-Debiased Voting (ODV)
K=3 circular shift permutations + majority vote.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import torch

from config.config import N_PERMS


def generate_permutations(options: list, n_perms: int = N_PERMS) -> list:
    if len(options) < 2:
        return [list(options)]
    perms = [list(options)]
    if len(options) == 2:
        perms.append(list(reversed(options)))
        return perms[:n_perms]
    for i in range(1, n_perms):
        perms.append(options[i:] + options[:i])
    return perms[:n_perms]


def circular_shift_options(options: list, shift: int = 1) -> list:
    n = len(options)
    if n == 0:
        return []
    shift = shift % n
    return options[shift:] + options[:shift]


def map_letter_to_content(letter: str, options: list) -> Optional[str]:
    if not letter or not options:
        return None
    letter = letter.strip().upper().strip("()[]{}")
    if len(letter) == 1 and letter.isalpha():
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(options):
            return options[idx]
    return None


def parse_answer_from_output(output_text: str, perm_opts: list, original_opts: list) -> Optional[str]:
    text = output_text.strip()

    def canonical(option: str) -> str:
        for original in original_opts:
            if original.strip().lower() == option.strip().lower():
                return original
        return option

    m = re.search(r'answer[:\s]+[\(\[]?\s*([A-F])\s*[\)\]]?(?=\s|$|[:.,;-])', text, re.IGNORECASE)
    if m:
        content = map_letter_to_content(m.group(1).upper(), perm_opts)
        if content:
            return canonical(content)

    try:
        from utils.utils import normalize_relation
        parsed = normalize_relation(text, perm_opts)
        if parsed:
            return canonical(parsed)
    except Exception:
        pass

    for opt in perm_opts:
        if opt.lower() in text.lower():
            return canonical(opt)
    REL_KW = {
        "on/above", "in front of", "behind", "beside", "left of", "right of", "above", "below",
    }
    text_lower = text.lower()
    for kw in REL_KW:
        if kw in text_lower:
            if kw == "above":
                return canonical("on/above") if "on/above" in original_opts else canonical(kw)
            if kw in original_opts:
                return canonical(kw)
    return None


def vote_answers(answers: list) -> tuple:
    counter = Counter(answers)
    most_common = counter.most_common(1)
    if not most_common:
        return answers[0] if answers else "", 0, counter
    winner, count = most_common[0]
    return winner, count, counter


def query_model_once(model, processor, image, prompt: str, device: str = "cuda", max_new_tokens: int = 128) -> str:
    image_data = image.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_data},
        {"type": "text", "text": prompt},
    ]}]
    chat_text = (
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if hasattr(processor, "apply_chat_template") else prompt
    )

    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(device)
    except Exception:
        inputs = processor(
            text=[chat_text],
            images=[image_data],
            return_tensors="pt",
            padding=True,
        ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    ilen = inputs["input_ids"].shape[1]
    return processor.batch_decode(output_ids[:, ilen:], skip_special_tokens=True)[0].strip()


def run_odv(
    model, processor, image,
    question, original_options,
    device: str = "cuda",
    n_perms: int = N_PERMS,
    prompt_template=None,
    max_new_tokens: int = 128,
) -> dict:
    from . import prompt as prompts

    if n_perms < 1:
        n_perms = 1

    perms = generate_permutations(original_options, n_perms)
    votes, raw_outputs = [], []

    for perm_opts in perms:
        if prompt_template == "astra":
            perm_prompt = prompts.build_astra_prompt(None, None, None, question, perm_opts)
        else:
            perm_prompt = prompts.build_baseline_prompt(question, perm_opts)

        try:
            output_text = query_model_once(model, processor, image, perm_prompt, device, max_new_tokens)
        except Exception as exc:
            raise RuntimeError(f"ODV generation failed for options {perm_opts}: {exc}") from exc

        raw_outputs.append(output_text)
        parsed = parse_answer_from_output(output_text, perm_opts, original_options)
        votes.append(parsed)

    valid = [v for v in votes if v is not None]
    if not valid:
        final = original_options[0] if original_options else ""
        wc = 0
        vc = Counter()
    else:
        final, wc, vc = vote_answers(valid)

    return {
        "final_answer": final,
        "votes": votes,
        "vote_counts": dict(vc),
        "vote_winner_count": wc,
        "raw_outputs": raw_outputs,
        "permutations": perms,
    }
