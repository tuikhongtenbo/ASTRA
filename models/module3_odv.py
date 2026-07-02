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
    m = re.search(r'answer[:\s]+[\(\[]?([A-F])[\)\]]?', text, re.IGNORECASE)
    if m:
        content = map_letter_to_content(m.group(1).upper(), perm_opts)
        if content:
            return original_opts[perm_opts.index(content)]
    for i, opt in enumerate(perm_opts):
        if opt.lower() in text.lower():
            return original_opts[i]
    REL_KW = {
        "in front of", "behind", "beside", "left of", "right of", "above", "below",
    }
    text_lower = text.lower()
    for kw in REL_KW:
        if kw in text_lower and kw in original_opts:
            return kw
    return None


def vote_answers(answers: list) -> tuple:
    counter = Counter(answers)
    most_common = counter.most_common(1)
    if not most_common:
        return answers[0] if answers else "", 0, counter
    return most_common[0]


def query_model_once(model, processor, image, prompt: str, device: str = "cuda", max_new_tokens: int = 128) -> str:
    try:
        from qwen_vl_utils import process_vision_info
        image_data = image.convert("RGB")
        pixel_values, image_grid, _ = process_vision_info(
            {"image": image_data}, return_image_grid=True
        )
        text_input = [{"role": "user", "content": [
            {"type": "image", "image": "placeholder"},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor(
            text=text_input, images=image_data, return_tensors="pt", padding=True
        ).to(device)
        if pixel_values is not None and hasattr(pixel_values, "to"):
            inputs["pixel_values"] = pixel_values.to(device)
        if image_grid is not None and hasattr(image_grid, "to"):
            inputs["image_grid"] = image_grid.to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        ilen = inputs["input_ids"].shape[1]
        return processor.batch_decode(output_ids[:, ilen:], skip_special_tokens=True)[0].strip()
    except Exception:
        inputs = processor(
            text=[{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}
            ]}], images=image, return_tensors="pt"
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
        except Exception:
            output_text = ""

        raw_outputs.append(output_text)
        parsed = parse_answer_from_output(output_text, perm_opts, original_options)
        votes.append(parsed)

    valid = [v for v in votes if v is not None]
    if not valid:
        final = votes[0] if votes else original_options[0]
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
