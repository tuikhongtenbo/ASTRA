"""
Pipeline — ASTRAPipeline tổng hợp 3 modules.
"""

from __future__ import annotations

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
from utils.utils import get_device, normalize_relation


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
        grounding_model=None,
        grounding_processor=None,
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
        self.grounding_model = grounding_model
        self.grounding_processor = grounding_processor
        self.depth_model = depth_model

        if load_models:
            self._load_models()

    def _load_models(self):
        self._load_qwen_model()
        if 1 in self.enable_modules and self.grounding_model is None:
            self._load_grounding_model()
        if 2 in self.enable_modules and self.depth_model is None:
            self._load_depth_model()

    def _load_qwen_model(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"[Pipeline] Loading {self.model_name} on {self.device}...")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype,
            device_map=self.device, trust_remote_code=True, **self.model_kwargs,
        )
        self.model.eval()
        print(f"[Pipeline] Qwen3-VL loaded in {time.time() - t0:.1f}s")

    def _load_grounding_model(self):
        try:
            print("[Pipeline] Loading Grounding DINO-Tiny...")
            t0 = time.time()
            self.grounding_model, self.grounding_processor, _ = ogm.load_grounding_model(self.device)
            print(f"[Pipeline] Grounding DINO loaded in {time.time() - t0:.1f}s")
        except ImportError as e:
            print(f"[Pipeline] Warning: Grounding DINO not loaded: {e}")
            print("[Pipeline] Module 1 (OGM) will be skipped.")
            self.grounding_model = None

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

    def preprocess(self, image: Image.Image, question: str, options: list) -> dict:
        result = {
            "marked_image": image,
            "O1_name": None, "O2_name": None,
            "O1_bbox": None, "O2_bbox": None,
            "depth_cue": None,
            "ogm_success": False, "dlc_success": False,
        }

        if 1 in self.enable_modules and self.grounding_model is not None:
            try:
                r = ogm.run_ogm(
                    image=image, question=question,
                    grounding_model=self.grounding_model,
                    processor=self.grounding_processor,
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
            except Exception:
                output = ""
            parsed = odv.parse_answer_from_output(output, perm_opts, options)
            votes.append(parsed)
        valid = [v for v in votes if v is not None]
        if not valid:
            return votes[0] if votes else (options[0] if options else "")
        _, _, vc = odv.vote_answers(valid)
        return vc.most_common(1)[0][0]

    def _generate(self, image: Image.Image, prompt: str) -> str:
        try:
            from qwen_vl_utils import process_vision_info
            image_data = image.convert("RGB")
            try:
                pv, ig, _ = process_vision_info({"image": image_data}, return_image_grid=True)
            except Exception:
                pv, ig = None, None

            text_input = [{"role": "user", "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": prompt},
            ]}]

            inputs = self.processor(
                text=text_input, images=image_data, return_tensors="pt", padding=True
            ).to(self.device)
            if pv is not None and hasattr(pv, "to"):
                inputs["pixel_values"] = pv.to(self.device)
            if ig is not None and hasattr(ig, "to"):
                inputs["image_grid"] = ig.to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                )
            ilen = inputs["input_ids"].shape[1]
            return self.processor.batch_decode(output_ids[:, ilen:], skip_special_tokens=True)[0].strip()
        except Exception:
            inputs = self.processor(
                text=[{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": prompt}
                ]}], images=image, return_tensors="pt",
            ).to(self.device)
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

        image = sample.get("image")
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception:
                image = None
        if image is None:
            return {"id": sample.get("id", 0), "predicted": "", "correct": False,
                    "error": "Failed to load image", **sample}

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

                if self.grounding_model is not None:
                    try:
                        r = ogm.run_ogm(
                            image=image, question=question,
                            grounding_model=self.grounding_model,
                            processor=self.grounding_processor,
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

        image = sample.get("image")
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception:
                image = None
        if image is None:
            return {"id": sample.get("id", 0), "predicted": "", "correct": False,
                    "error": "Failed to load image", **sample}

        question = sample.get("question", "")
        options = sample.get("options", [])
        answer = sample.get("answer", "")

        t_pre = time.time()
        preprocessed = self.preprocess(image, question, options)
        t_pre = time.time() - t_pre

        t_fwd = time.time()
        predicted = self.forward(image, question, options, preprocessed)
        t_fwd = time.time() - t_fwd

        pred_norm = normalize_relation(predicted, options) or predicted
        ans_norm = normalize_relation(answer, options) or answer
        correct = pred_norm.lower().strip() == ans_norm.lower().strip()

        return {
            "id": sample.get("id", 0),
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

    def unload(self):
        for attr in ("model", "grounding_model", "depth_model"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def __repr__(self):
        return f"ASTRAPipeline(model={self.model_name}, modules={self.enable_modules}, device={self.device})"
