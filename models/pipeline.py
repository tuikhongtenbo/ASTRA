"""
Pipeline — ASTRAPipeline tổng hợp 3 modules.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import torch
from PIL import Image

from config.config import (
    MODEL_ALIASES, DEFAULT_MODEL, MAX_NEW_TOKENS,
    DEPTH_EPSILON, CONFIDENCE_THRESHOLD, N_PERMS, IMAGE_DIR,
    ESCALATION_LOG_FILE, EXTRACTION_OUTPUT_FILE,
)
from . import module1_ogm as ogm
from . import module2_dlc as dlc
from . import module3_odv as odv
from . import prompt as prompts
from utils.utils import find_image_path, get_device, load_image, normalize_relation


class ASTRAPipeline:
    """
    Pipeline đầy đủ ASTRA.
      1 - OGM: Object-Grounded Marking
      2 - DLC: Depth-Layer Cue
      3 - ODV: Order-Debiased Voting
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = None,
        enable_modules: list = None,
        n_perms: int = N_PERMS,
        depth_epsilon: float = DEPTH_EPSILON,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        yoloe_model=None,
        depth_model=None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        load_models: bool = True,
        image_dir: str = IMAGE_DIR,
        use_escalation: bool = False,
        **model_kwargs,
    ):
        self.model_name = MODEL_ALIASES.get(model_name, model_name)
        self.device = device or get_device()
        self.enable_modules = enable_modules if enable_modules is not None else [1, 2, 3]
        self.n_perms = n_perms
        self.depth_epsilon = depth_epsilon
        self.confidence_threshold = confidence_threshold
        self.max_new_tokens = max_new_tokens
        self.model_kwargs = model_kwargs
        self.image_dir = image_dir
        self.use_escalation = use_escalation

        self.model = None
        self.processor = None
        self.yoloe_model = yoloe_model
        self.depth_model = depth_model

        if load_models:
            self._load_models()

    def _load_models(self):
        self._load_qwen_model()
        if 1 in self.enable_modules and self.yoloe_model is None:
            self._load_yoloe_model()
        if 2 in self.enable_modules and self.depth_model is None:
            self._load_depth_model()

    def _load_qwen_model(self):
        from transformers import AutoProcessor
        print(f"[Pipeline] Loading {self.model_name} on {self.device}...")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        load_kwargs = dict(self.model_kwargs)
        load_kwargs.setdefault("trust_remote_code", True)
        if self.device == "cuda":
            load_kwargs.setdefault("device_map", "auto")
        else:
            load_kwargs.setdefault("device_map", None)

        loader_candidates = []
        try:
            from transformers import AutoModelForImageTextToText
            loader_candidates.append(AutoModelForImageTextToText)
        except Exception:
            pass
        try:
            from transformers import AutoModelForVision2Seq
            loader_candidates.append(AutoModelForVision2Seq)
        except Exception:
            pass
        try:
            from transformers import Qwen3VLForConditionalGeneration
            loader_candidates.append(Qwen3VLForConditionalGeneration)
        except Exception:
            pass
        try:
            from transformers import AutoModelForCausalLM
            loader_candidates.append(AutoModelForCausalLM)
        except Exception:
            pass

        last_error = None
        for loader_cls in loader_candidates:
            try:
                self.model = loader_cls.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    **load_kwargs,
                )
                break
            except ValueError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

        if self.model is None:
            raise RuntimeError(
                f"Failed to load model {self.model_name} with available Transformers loaders: {last_error}"
            )

        if load_kwargs.get("device_map") is None:
            self.model.to(self.device)
        self.model.eval()
        print(f"[Pipeline] Qwen3-VL loaded in {time.time() - t0:.1f}s")

    def _load_yoloe_model(self):
        try:
            print("[Pipeline] Loading YOLOE-26X...")
            t0 = time.time()
            self.yoloe_model = ogm.load_yoloe_model(self.device)
            print(f"[Pipeline] YOLOE-26X loaded in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"[Pipeline] Warning: YOLOE-26X not loaded: {e}")
            print("[Pipeline] Module 1 (OGM) will be skipped.")
            self.yoloe_model = None

    def _load_depth_model(self):
        try:
            print("[Pipeline] Loading Depth-Anything-V2-Small...")
            t0 = time.time()
            self.depth_model = dlc.load_depth_model("small", self.device)
            print(f"[Pipeline] Depth-Anything loaded in {time.time() - t0:.1f}s")
        except ImportError as e:
            print(f"[Pipeline] Warning: Depth-Anything not loaded: {e}")
            print("[Pipeline] Module 2 (DLC) will be skipped.")
            self.depth_model = None

    def _load_sample_image(self, sample: dict) -> tuple[Optional[Image.Image], Optional[str], Optional[str]]:
        image = sample.get("image")
        if isinstance(image, Image.Image):
            return image.convert("RGB"), sample.get("image_path"), None

        refs = []
        for key in ("image", "image_path", "image_name"):
            ref = sample.get(key)
            if isinstance(ref, str) and ref and ref not in refs:
                refs.append(ref)

        if not refs:
            return None, None, f"Failed to load image: no image reference in sample (IMAGE_DIR={self.image_dir})"

        errors = []
        for ref in refs:
            path = find_image_path(self.image_dir, ref)
            if not path:
                errors.append(f"{ref} -> not found")
                continue
            image_obj = load_image(path)
            if image_obj is not None:
                return image_obj, path, None
            errors.append(f"{path} -> open failed")

        detail = "; ".join(errors[:4])
        return None, None, f"Failed to load image: {detail} (IMAGE_DIR={self.image_dir})"

    def preprocess(self, image: Image.Image, question: str, options: list) -> dict:
        result = {
            "marked_image": image,
            "O1_name": None, "O2_name": None,
            "O1_bbox": None, "O2_bbox": None,
            "depth_cue": None,
            "ogm_success": False, "dlc_success": False,
        }

        if 1 in self.enable_modules and self.yoloe_model is not None:
            try:
                r = ogm.run_ogm(
                    image=image, question=question,
                    yoloe_model=self.yoloe_model,
                    device=self.device,
                    confidence_threshold=self.confidence_threshold,
                )
                result.update({
                    "marked_image": r["marked_image"],
                    "O1_name": r.get("O1_name"), "O2_name": r.get("O2_name"),
                    "O1_bbox": r.get("O1_bbox"), "O2_bbox": r.get("O2_bbox"),
                    "ogm_success": r.get("success", False),
                })
            except Exception as e:
                print(f"[OGM] Error: {e}")

        if 2 in self.enable_modules and self.depth_model is not None:
            try:
                r = dlc.run_dlc(
                    image=image,
                    O1_bbox=result.get("O1_bbox"),
                    O2_bbox=result.get("O2_bbox"),
                    O1_name=result.get("O1_name"),
                    O2_name=result.get("O2_name"),
                    depth_model=self.depth_model,
                    device=self.device,
                    epsilon=self.depth_epsilon,
                )
                result.update({
                    "depth_cue": r.get("depth_cue"),
                    "dlc_success": r.get("success", False),
                })
            except Exception as e:
                print(f"[DLC] Error: {e}")

        return result

    def forward(self, image: Image.Image, question: str, options: list, preprocessed: dict = None) -> str:
        if preprocessed is None:
            preprocessed = self.preprocess(image, question, options)

        marked_image = preprocessed.get("marked_image", image)
        O1 = preprocessed.get("O1_name")
        O2 = preprocessed.get("O2_name")
        depth_cue = preprocessed.get("depth_cue")

        if 3 in self.enable_modules:
            return self._forward_odv(marked_image, question, options, O1, O2, depth_cue)
        return self._forward_single(marked_image, question, options, O1, O2, depth_cue)

    def _forward_single(self, image, question, options, O1, O2, depth_cue) -> str:
        if O1 or O2 or depth_cue:
            prompt = prompts.build_astra_prompt(O1, O2, depth_cue, question, options)
        else:
            prompt = prompts.build_baseline_prompt(question, options)
        output = self._generate(image, prompt)
        parsed = normalize_relation(output, options)
        if parsed:
            return parsed
        for opt in options:
            if opt.lower() in output.lower():
                return opt
        return options[0] if options else ""

    def _forward_odv(self, image, question, options, O1, O2, depth_cue) -> str:
        base = "astra" if (O1 or O2 or depth_cue) else "baseline"
        perms = odv.generate_permutations(options, self.n_perms)
        votes = []
        for perm_opts in perms:
            if base == "astra":
                pp = prompts.build_astra_prompt(O1, O2, depth_cue, question, perm_opts)
            else:
                pp = prompts.build_baseline_prompt(question, perm_opts)
            try:
                output = self._generate(image, pp)
            except Exception as exc:
                raise RuntimeError(f"ODV generation failed for options {perm_opts}: {exc}") from exc
            parsed = odv.parse_answer_from_output(output, perm_opts, options)
            votes.append(parsed)
        valid = [v for v in votes if v is not None]
        if not valid:
            return options[0] if options else ""
        winner, _, _ = odv.vote_answers(valid)
        return winner

    def _prepare_generation_inputs(self, images: list[Image.Image], prompt: str):
        messages = [{"role": "user", "content": []}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": prompt})

        chat_text = (
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if hasattr(self.processor, "apply_chat_template") else prompt
        )

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            processor_kwargs = {
                "text": [chat_text],
                "images": image_inputs,
                "return_tensors": "pt",
                "padding": True,
            }
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs
            return self.processor(**processor_kwargs).to(self.device)
        except Exception as qwen_exc:
            try:
                return self.processor(
                    text=[chat_text],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Failed to prepare Qwen-VL inputs "
                    f"(qwen_vl_utils: {qwen_exc}; fallback: {fallback_exc})"
                ) from fallback_exc

    def _generate(self, image: Image.Image, prompt: str) -> str:
        inputs = self._prepare_generation_inputs([image.convert("RGB")], prompt)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        ilen = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(output_ids[:, ilen:], skip_special_tokens=True)[0].strip()

    def run_escalated(self, sample: dict) -> dict:
        """
        Escalation logic for robust inference:
          Step 0: Zero-shot voting (no augment) — determine baseline agreement.
          Step 1: If disagree → extract entities → verify → run M1+M2 → augmented voting.
          Step 2: Return final answer with escalation_log.
        """
        from collections import Counter
        t0 = time.time()

        image, image_path, image_error = self._load_sample_image(sample)
        if image is None:
            return {**sample, "id": sample.get("id", 0), "predicted": "", "correct": False,
                    "error": "Failed to load image", "error_detail": image_error,
                    "image_path": image_path or sample.get("image_path"),
                    "t_total": time.time() - t0}

        question = sample.get("question", "")
        options = sample.get("options", [])
        answer = sample.get("answer", "")

        # ── Step 0: zero-shot voting (no augment) ───────────────────────────
        perms = odv.generate_permutations(options, self.n_perms)
        votes_raw = []
        for perm_opts in perms:
            prompt = prompts.build_baseline_prompt(question, perm_opts)
            try:
                out = self._generate(image, prompt)
            except Exception:
                out = ""
            parsed = odv.parse_answer_from_output(out, perm_opts, options)
            votes_raw.append(parsed)

        valid_raw = [v for v in votes_raw if v is not None]
        if valid_raw:
            counter = Counter(valid_raw)
            answer_raw, agreement = counter.most_common(1)[0]
        else:
            answer_raw = votes_raw[0] if votes_raw else options[0]
            agreement = 0

        escalation_log = {
            "step0_votes": votes_raw,
            "step0_answer": answer_raw,
            "step0_agreement": agreement,
            "step1_escalated": False,
            "step1_extraction": None,
            "step1_augmented_votes": None,
            "final_method": None,
        }

        # ── Escalation decision ──────────────────────────────────────────
        if agreement >= self.n_perms:
            # All votes agree → STOP, return zero-shot result
            pred = answer_raw
            escalation_log["final_method"] = "zero-shot (agreed)"
        else:
            # Disagreement → escalate to Module 1+2
            escalation_log["step1_escalated"] = True

            # Entity extraction via LLM
            from .extractor import extract_entities_llm
            extraction = extract_entities_llm(question)
            escalation_log["step1_extraction"] = {
                "O1": extraction.O1,
                "O2": extraction.O2,
                "O2_is_viewer": extraction.O2_is_viewer,
                "confidence": extraction.confidence,
            }

            if not extraction.is_valid:
                pred = answer_raw
                escalation_log["final_method"] = "zero-shot (extraction failed)"
            else:
                # Module 1: OGM
                marked_img, marks_ok = image, False
                O1_name, O2_name = None, None
                O1_bbox, O2_bbox = None, None

                if self.yoloe_model is not None:
                    try:
                        r = ogm.run_ogm(
                            image=image, question=question,
                            yoloe_model=self.yoloe_model,
                            device=self.device,
                            confidence_threshold=self.confidence_threshold,
                            extraction_result=extraction,
                            O2_is_viewer=extraction.O2_is_viewer,
                        )
                        marked_img = r["marked_image"]
                        marks_ok = r["success"]
                        O1_name = r.get("O1_name")
                        O2_name = r.get("O2_name")
                        O1_bbox = r.get("O1_bbox")
                        O2_bbox = r.get("O2_bbox")
                    except Exception:
                        pass

                if not marks_ok:
                    pred = answer_raw
                    escalation_log["final_method"] = "zero-shot (OGM failed)"
                else:
                    # Module 2: DLC
                    depth_cue = None
                    if self.depth_model is not None:
                        try:
                            r2 = dlc.run_dlc(
                                image=image,
                                O1_bbox=O1_bbox, O2_bbox=O2_bbox,
                                O1_name=O1_name, O2_name=O2_name,
                                O2_is_viewer=extraction.O2_is_viewer,
                                depth_model=self.depth_model,
                                device=self.device,
                                epsilon=self.depth_epsilon,
                            )
                            depth_cue = r2.get("depth_cue")
                        except Exception:
                            pass

                    # Module 3: augmented voting
                    aug_votes = []
                    for perm_opts in perms:
                        if O1_name or O2_name or depth_cue:
                            pp = prompts.build_astra_prompt(O1_name, O2_name, depth_cue, question, perm_opts)
                        else:
                            pp = prompts.build_baseline_prompt(question, perm_opts)
                        try:
                            out = self._generate(marked_img or image, pp)
                        except Exception:
                            out = ""
                        parsed = odv.parse_answer_from_output(out, perm_opts, options)
                        aug_votes.append(parsed)

                    valid_aug = [v for v in aug_votes if v is not None]
                    if valid_aug:
                        counter2 = Counter(valid_aug)
                        pred, _ = counter2.most_common(1)[0]
                    else:
                        pred = aug_votes[0] if aug_votes else answer_raw

                    escalation_log["step1_augmented_votes"] = aug_votes
                    escalation_log["final_method"] = "augmented (escalated)"

        pred_norm = normalize_relation(pred, options) or pred
        ans_norm = normalize_relation(answer, options) or answer
        correct = pred_norm.lower().strip() == ans_norm.lower().strip()

        return {
            "id": sample.get("id", 0),
            "image_path": image_path or sample.get("image_path"),
            "question": question, "options": options, "answer": answer,
            "predicted": pred, "correct": correct,
            "O1_name": O1_name if "O1_name" in dir() else None,
            "O2_name": O2_name if "O2_name" in dir() else None,
            "depth_cue": depth_cue if "depth_cue" in dir() else None,
            "escalation_log": escalation_log,
            "t_total": time.time() - t0,
            "modules_enabled": self.enable_modules,
        }

    def run(self, sample: dict) -> dict:
        t0 = time.time()
        if self.use_escalation:
            return self.run_escalated(sample)

        image, image_path, image_error = self._load_sample_image(sample)
        if image is None:
            return {**sample, "id": sample.get("id", 0), "predicted": "", "correct": False,
                    "error": "Failed to load image", "error_detail": image_error,
                    "image_path": image_path or sample.get("image_path"),
                    "t_total": time.time() - t0}

        question = sample.get("question", "")
        options = sample.get("options", [])
        answer = sample.get("answer", "")

        t_pre = time.time()
        preprocessed = self.preprocess(image, question, options)
        t_pre = time.time() - t_pre

        t_fwd = time.time()
        predicted = self.forward(image, question, options, preprocessed) or ""
        t_fwd = time.time() - t_fwd

        pred_norm = normalize_relation(predicted, options) or predicted
        ans_norm = normalize_relation(answer, options) or answer
        correct = pred_norm.lower().strip() == ans_norm.lower().strip()

        return {
            "id": sample.get("id", 0),
            "image_path": image_path or sample.get("image_path"),
            "question": question, "options": options, "answer": answer,
            "predicted": predicted, "correct": correct,
            "O1_name": preprocessed.get("O1_name"),
            "O2_name": preprocessed.get("O2_name"),
            "O1_bbox": preprocessed.get("O1_bbox"),
            "O2_bbox": preprocessed.get("O2_bbox"),
            "depth_cue": preprocessed.get("depth_cue"),
            "ogm_success": preprocessed.get("ogm_success", False),
            "dlc_success": preprocessed.get("dlc_success", False),
            "t_preprocess": t_pre, "t_forward": t_fwd, "t_total": time.time() - t0,
            "modules_enabled": self.enable_modules,
        }

    def run_v2(self, extraction_record: dict) -> dict:
        """
        ASTRA v2 pipeline — chạy từng module riêng biệt.

        Đọc intermediate files từ step1 (m1_bbox/) và step2 (m2_depth/),
        build prompt với prompt_v2, chạy ODV voting với 2 ảnh đầu vào.

        Args:
            extraction_record: dict từ test_objects_last.json

        Returns:
            dict với keys: id, predicted, correct, marks_ok, depth_ok,
            votes, depth_o1, depth_o2, relation_text, prompt
        """
        import os
        import random
        from collections import Counter

        from config.pipeline_config import (
            M1_OUTPUT_DIR, M2_OUTPUT_DIR, N_PERMS,
        )
        from models.prompt_v2 import build_prompt

        t0 = time.time()
        sid = str(extraction_record.get("id", ""))

        options = extraction_record.get("options", [])
        answer = extraction_record.get("answer", "")
        question = extraction_record.get("question", "")

        # ── Load intermediate data ──────────────────────────────────────────
        bbox_info_path = os.path.join(M1_OUTPUT_DIR, "bbox_info.json")
        depth_info_path = os.path.join(M2_OUTPUT_DIR, "depth_info.json")

        box_info = {}
        if os.path.exists(bbox_info_path):
            with open(bbox_info_path, "r", encoding="utf-8") as f:
                all_boxes = json.load(f)
            box_info = all_boxes.get(sid, {})

        depth_info = {}
        if os.path.exists(depth_info_path):
            with open(depth_info_path, "r", encoding="utf-8") as f:
                all_depth = json.load(f)
            depth_info = all_depth.get(sid, {})

        marks_ok = bool(box_info.get("marks_ok", False))
        depth_ok = bool(depth_info.get("depth_ok", False))
        depth_o1 = depth_info.get("depth_o1", 0.0) or 0.0
        depth_o2 = depth_info.get("depth_o2", 0.0) or 0.0
        relation_text = depth_info.get("relation_text", "")

        # ── Load 2 ảnh ─────────────────────────────────────────────────────
        img1_path = os.path.join(M1_OUTPUT_DIR, f"{sid}_bbox.jpg")
        img2_path = os.path.join(M2_OUTPUT_DIR, f"{sid}_depth.jpg")

        img1 = None
        if os.path.exists(img1_path):
            img1 = Image.open(img1_path).convert("RGB")

        img2 = None
        use_two_images = marks_ok and depth_ok and os.path.exists(img2_path)
        if use_two_images:
            img2 = Image.open(img2_path).convert("RGB")

        # ── Build prompt ────────────────────────────────────────────────────
        prompt_marks_ok = marks_ok and depth_ok
        prompt = build_prompt(
            record=extraction_record,
            marks_ok=prompt_marks_ok,
            depth_o1=depth_o1,
            depth_o2=depth_o2,
            options=options,
        )

        # ── ODV Voting ──────────────────────────────────────────────────────
        perms = self._generate_permutations(options, self.n_perms)
        votes = []
        vote_outputs = []

        for perm_opts in perms:
            perm_prompt = build_prompt(
                record=extraction_record,
                marks_ok=prompt_marks_ok,
                depth_o1=depth_o1,
                depth_o2=depth_o2,
                options=perm_opts,
            )

            if self.model is not None and img1 is not None:
                output = self._generate_v2(img1, img2, perm_prompt)
            else:
                output = ""
            vote_outputs.append(output)
            parsed = self._parse_answer(output, perm_opts, options)
            votes.append(parsed)

        # Majority vote
        valid = [v for v in votes if v is not None]
        if valid:
            counter = Counter(valid)
            predicted = counter.most_common(1)[0][0]
        else:
            predicted = votes[0] if votes else (options[0] if options else "")

        pred_norm = normalize_relation(predicted, options) or predicted or ""
        ans_norm = normalize_relation(answer, options) or answer
        correct = pred_norm.lower().strip() == ans_norm.lower().strip()

        return {
            "id": sid,
            "question": question,
            "options": options,
            "answer": answer,
            "predicted": pred_norm,
            "correct": correct,
            "marks_ok": marks_ok,
            "depth_ok": depth_ok,
            "depth_o1": depth_o1,
            "depth_o2": depth_o2,
            "relation_text": relation_text,
            "votes": votes,
            "vote_outputs": vote_outputs,
            "prompt": prompt,
            "two_images_used": use_two_images,
            "t_total": time.time() - t0,
        }

    def _generate_permutations(self, options: list, n: int) -> list:
        """Generate n permutations for ODV."""
        perms = []
        seen = set()
        all_opts = list(options)
        for _ in range(n * 3):
            perm = all_opts.copy()
            random.shuffle(perm)
            key = tuple(perm)
            if key not in seen:
                seen.add(key)
                perms.append(perm)
            if len(perms) >= n:
                break
        if all_opts not in perms:
            perms.insert(0, all_opts)
        return perms[:n]

    def _parse_answer(self, output: str, perm_opts: list, original_opts: list) -> str | None:
        """Parse answer from VLM output."""
        if not output:
            return None
        pred = normalize_relation(output, perm_opts)
        if pred is None:
            return None
        for original in original_opts:
            if original.strip().lower() == pred.strip().lower():
                return original
        return pred

    def _generate_v2(self, image1: Image.Image, image2: Image.Image | None, prompt: str) -> str:
        """Generate with 1 or 2 images (ASTRA v2)."""
        images = [image1.convert("RGB")]
        if image2 is not None:
            images.append(image2.convert("RGB"))

        inputs = self._prepare_generation_inputs(images, prompt)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        ilen = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            output_ids[:, ilen:], skip_special_tokens=True
        )[0].strip()

    def unload(self):
        for attr in ("model", "yoloe_model", "depth_model"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def __repr__(self):
        return f"ASTRAPipeline(model={self.model_name}, modules={self.enable_modules}, device={self.device})"
