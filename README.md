# ASTRA: Auxiliary Spatial Tools for Robust Answering

**ASTRA** (*Auxiliary Spatial Tools for Robust Answering*) là hệ thống hỗ trợ suy luận không gian cho các mô hình Ngôn ngữ - Thị giác lớn (Vision-Language Models - VLMs) theo phương pháp **thuần Inference (Zero-shot / Tuning-free)**. Thay vì phụ thuộc vào việc huấn luyện phức tạp (SFT / GRPO) hay tạo dữ liệu chuỗi suy luận (CoT distillation) từ các model sở hữu riêng (Teacher APIs), ASTRA tiêm nhiễm trực tiếp các tín hiệu không gian thị giác và độ sâu vào prompt, đồng thời khử bias vị trí đáp án bằng cơ chế bỏ phiếu hoán vị.

---

## Mục lục

1. [Giới thiệu Kiến trúc 3 Module](#1-giới-thiệu-kiến-trúc-3-module)
2. [Cài đặt Môi trường](#2-cài-đặt-môi-trường)
3. [Chuẩn bị Dữ liệu SpatialMQA](#3-chuẩn-bị-dữ-liệu-spatialmqa)
4. [Cấu trúc Thư mục Dự án](#4-cấu-trúc-thư-mục-dự-án)
5. [Hướng dẫn Sử dụng CLI (`main.py`)](#5-hướng-dẫn-sử-dụng-cli-mainpy)
6. [Kịch bản Thực nghiệm](#6-kịch-bản-thực-nghiệm)
7. [Chạy Tự động bằng Shell / PowerShell Scripts](#7-chạy-tự-động-bằng-shell--powershell-scripts)
8. [Ước tính Tài nguyên & VRAM](#8-ước-tính-tài-nguyên--vram)

---

## 1. Giới thiệu Kiến trúc 3 Module

ASTRA được xây dựng xoay quanh 3 module bổ trợ độc lập, hoạt động đồng thời trong quá trình suy luận:

```
[Ảnh gốc + Câu hỏi SpatialMQA]
       │
       ├─► [LLM Extractor] ── DashScope API ──► Extract O1/O2 + O2_is_viewer ──► Verify (hallucination gate)
       │
       ├─► [Module 1: OGM] ── YOLOE-26X ──► Phát hiện bbox O1, O2 & Vẽ Set-of-Mark [1], [2] lên ảnh
       │
       ├─► [Module 2: DLC] ── Depth-Anything-V2  ──► Tính depth map & Sinh depth hint (soft, auxiliary)
       │
       ▼
[Module 3: ODV] ── K=3 Hoán vị Circular Shift ──► Query Qwen3-VL Zero-shot ──► Majority Vote ──► [Đáp án Khử Bias]
```

* **Module 1 — Object-Grounded Marking (OGM):**
  * Tự động trích xuất các thực thể mục tiêu ($O_1$, $O_2$) từ câu hỏi không gian.
  * Sử dụng model nhúng nhẹ **YOLOE-26X** để phát hiện bounding box.
  * Vẽ thẻ nhãn trực quan theo phong cách *Set-of-Mark* (`[1] <O1>` và `[2] <O2>`) trực tiếp lên ảnh nhằm giúp VLM tập trung sự chú ý vào đúng đối tượng.
  * Tự động fallback về ảnh gốc nếu độ tin cậy phát hiện thấp (< threshold `0.3`).

* **Module 2 — Depth-Layer Cue (DLC):**
  * Sử dụng model ước lượng độ sâu đơn ảnh **Depth-Anything-V2-Small (~25M params)** để tính toán relative depth map.
  * Tính độ sâu trung bình trong các vùng bounding box của $O_1$ và $O_2$.
  * Sinh depth cue như **soft hint (gợi ý phụ, không khẳng định)**, được đánh dấu rõ ràng là auxiliary/may-be-inaccurate, để VLM không phụ thuộc hoàn toàn vào nó.

* **Module 3 — Order-Debiased Voting (ODV):**
  * Khắc phục triệt để hiện tượng thiên lệch vị trí đáp án (*Blind Spatial Token Favouring - BSTF*), nơi VLM có xu hướng đoán mò các option A/B/C ở vị trí quen thuộc.
  * Tạo $K=3$ hoán vị dịch vòng (circular shift) của danh sách đáp án trắc nghiệm.
  * Thực hiện suy luận Zero-shot trên cả 3 hoán vị, sau đó ánh xạ chữ cái đáp án về nội dung text gốc và tiến hành bỏ phiếu đa số (Majority Vote) để đưa ra câu trả lời bền vững nhất.

---

## 2. Cài đặt Môi trường

Đảm bảo hệ thống có GPU NVIDIA với tối thiểu **8 GB - 24 GB VRAM** (tùy thuộc vào kích thước model Qwen3-VL 2B, 4B hay 8B).

```bash
# 1. Cài đặt các thư viện cơ bản từ requirements.txt
pip install -r requirements.txt

# 1b. Cài đặt openai (cho DashScope API backend và LLM extractor)
pip install openai

# 2. Cài đặt YOLOE-26X (cho Module 1 OGM)
# YOLOE-26X is installed through ultralytics from requirements.txt

# 3. Cài đặt Depth-Anything-V2 (cho Module 2 DLC)
pip install depth-anything-v2

# 4. (Khuyến nghị) Cài đặt Flash Attention 2 để tối ưu tốc độ inference trên GPU Ampere/Ada (RTX 30xx/40xx, A100):
pip install flash-attn --no-build-isolation
```

> [!NOTE]
> Module 1 dùng **YOLOE-26X** qua `ultralytics`; weight mặc định `yoloe-26x-seg.pt` sẽ được tải/cache trong lần chạy đầu nếu chưa có local.

---

## 3. Chuẩn bị Dữ liệu SpatialMQA

Dự án sử dụng bộ dữ liệu **SpatialMQA** (đánh giá khả năng hiểu không gian 3D/2D trên ảnh thực tế từ COCO 2017).

1. Đảm bảo các file dữ liệu (`train.jsonl`, `dev.jsonl`, `test.jsonl`, `test_500.jsonl`) nằm trong thư mục `data/`.
2. **Cơ chế Tự động Tìm kiếm Ảnh (Smart Image Fallback):**
   * Bạn không cần phải cấu hình đường dẫn tuyệt đối phức tạp. `config/config.py` đã tích hợp sẵn cơ chế tự động dò tìm ảnh theo thứ tự ưu tiên:
     1. Thư mục `relevant_images` tại các project tham khảo lân cận (`../thamkhao/SpatialMQA/Dataset/relevant_images`).
     2. Thư mục cục bộ `data/images/`.
     3. Thư mục cục bộ `relevant_images/`.
   * Chỉ cần đặt ảnh COCO hoặc folder `relevant_images` vào một trong các vị trí trên, pipeline sẽ tự động nhận diện 100%.

---

## 4. Cấu trúc Thư mục Dự án

```
ASTRA/
├── config/                  # Cấu hình tham số, paths & model aliases
│   ├── __init__.py
│   ├── config.py            # Parameters: MAX_NEW_TOKENS, CONFIDENCE_THRESHOLD, DEPTH_EPSILON...
│   └── zero3_offload.json
├── data/                    # Thư mục chứa dataset JSONL & Images
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── test_500.jsonl
├── data_processing/         # Tiện ích đọc & xử lý dataset
│   ├── __init__.py
│   └── dataset.py           # SpatialMQADataset & DataLoader
├── models/                  # Core logic của ASTRA Pipeline
│   ├── __init__.py
│   ├── module1_ogm.py       # Module 1: YOLOE-26X Set-of-Mark bounding boxes
│   ├── module2_dlc.py       # Module 2: Depth-Anything-V2 depth cues
│   ├── module3_odv.py       # Module 3: Circular shift permutations & majority voting
│   ├── pipeline.py          # ASTRAPipeline: Tích hợp tổng thể 3 module
│   ├── prompt.py            # System prompts & template builder
│   └── extractor/            # LLM-based entity extraction (pre-processing for M1)
│       ├── __init__.py
│       ├── llm_client.py    # DashScope API client cho entity extraction
│       ├── extract.py       # extract_entities_llm(), extract_entities_regex_legacy()
│       ├── verify.py        # Hallucination detection & confidence gating
│       └── parse.py         # JSON/regex parser cho raw LLM output
├── evaluation/              # Module tính toán metrics & báo cáo
│   ├── __init__.py
│   └── evaluator.py         # Accuracy, Per-relation accuracy, Ablation summarizer
├── utils/                   # Tiện ích chung (Normalizer, Logging, Timer)
│   ├── __init__.py
│   └── utils.py
├── scripts/                 # Tiện ích chạy thực nghiệm
│   ├── eval_2B.sh           # Chạy đánh giá model 2B
│   ├── eval_4B.sh           # Chạy đánh giá model 4B
│   ├── eval_8B.sh           # Chạy đánh giá model 8B
│   ├── run_all_models.sh    # Chạy tất cả model 2B → 4B → 8B
│   ├── extract_objects.py    # Trích xuất O1/O2 qua LLM cho toàn bộ dataset
│   └── run_module1_ablation.py  # Ablation harness đánh giá cascading error của M1
├── test/                    # Unit tests
│   ├── __init__.py
│   └── test_verify_extraction.py  # Tests cho verify_extraction & hallucination detection
├── tool/                    # Tiện ích bổ trợ
│   └── select_relevant_images.py
├── main.py                  # CLI Entry point chính của dự án
├── astra_eval.ps1 / .sh     # PowerShell / Shell shortcut script
└── requirements.txt
```

---

## 5. Hướng dẫn Sử dụng CLI (`main.py`)

`main.py` là giao diện dòng lệnh duy nhất để thực hiện mọi tác vụ đánh giá và thực nghiệm trong ASTRA.

### 5.1. Chạy đánh giá đơn lẻ (`eval`)

```bash
# Baseline: model gốc không bật module nào
python main.py eval \
    --model Qwen3-VL-4B \
    --baseline \
    --split test \
    --output outputs/astra/4B/baseline/results.jsonl

# Full ASTRA: bật M1 (OGM) + M2 (DLC) + M3 (ODV)
python main.py eval \
    --model Qwen3-VL-4B \
    --split test \
    --output outputs/astra/4B/ASTRA_full/results.jsonl

# Escalation: zero-shot trước, augment khi disagreement
python main.py eval \
    --model Qwen3-VL-4B \
    --escalation \
    --split test \
    --output outputs/astra/4B/escalation/results.jsonl

# Tuỳ chỉnh thresholds
python main.py eval \
    --model Qwen3-VL-4B \
    --confidence-threshold 0.4 \
    --depth-epsilon 0.08 \
    --n-perms 5 \
    --split dev \
    --output outputs/astra/4B/dev/results.jsonl
```

### 5.1b. Chạy Qwen3-VL-4B qua DashScope API

Backend mặc định là `local` (load model bằng `transformers`). Nếu muốn gọi Qwen3-VL-4B qua DashScope/OpenAI-compatible API, dùng `--backend dashscope`. API key có thể đặt trong terminal hoặc file `.env` ở thư mục gốc.

```powershell
# PowerShell
$env:DASHSCOPE_API_KEY="sk-..."
$env:DASHSCOPE_WORKSPACE_ID="your-workspace-id"
$env:DASHSCOPE_REGION="ap-southeast-1"
```

```bash
# .env
DASHSCOPE_API_KEY=sk-...
DASHSCOPE_WORKSPACE_ID=your-workspace-id
DASHSCOPE_REGION=ap-southeast-1
# Hoặc đặt trực tiếp:
# DASHSCOPE_BASE_URL=https://your-workspace-id.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
```

```bash
python main.py eval \
    --backend dashscope \
    --model Qwen3-VL-4B \
    --baseline \
    --split test \
    --max-samples 100 \
    --output outputs/astra/4B/api_baseline/results.jsonl
```

Nếu tên model trong workspace khác alias mặc định `qwen3-vl-4b-instruct`, truyền trực tiếp tên đó qua `--model`.

### 5.2. Chạy tất cả model và kịch bản (`run-all`)

```bash
# Chạy baseline + full ASTRA cho 3 model 2B, 4B, 8B
python main.py run-all \
    --models Qwen3-VL-2B Qwen3-VL-4B Qwen3-VL-8B \
    --split test \
    --output-dir outputs/astra

# Chạy thử nhanh trên 100 sample đầu tiên
python main.py run-all \
    --models Qwen3-VL-2B Qwen3-VL-4B \
    --split test \
    --max-samples 100 \
    --output-dir outputs/astra
```

### 5.3. So sánh kết quả (`compare`)

```bash
# So sánh tất cả kết quả trong thư mục và xuất CSV
python main.py compare \
    --results-dir outputs/astra \
    --save outputs/astra/summary.json
```

### 5.4. Chạy thử một câu hỏi đơn lẻ (`single`)

```bash
python main.py single \
    --model Qwen3-VL-4B \
    --image path/to/image.jpg \
    --question "Where is the dog located relative to the cat?" \
    --options "in front of|behind|left of|right of"
```

### 5.5. Trích xuất O1/O2 qua LLM (`scripts/extract_objects.py`)

Script này gọi API DashScope (Qwen3.7-max) để trích xuất entity từ câu hỏi, lưu vào file JSON. Hỗ trợ resume — nếu file đã tồn tại, chỉ trích xuất các sample còn thiếu.

**Yêu cầu:** Set biến môi trường `DASHSCOPE_API_KEY`/`QWEN_API_KEY` hoặc tạo file `.env` trong thư mục gốc.

```bash
# Trích xuất cho toàn bộ test split
python scripts/extract_objects.py --split test

# Chỉ 20 sample đầu, không resume
python scripts/extract_objects.py --split test --max-samples 20 --no-resume

# Trích xuất cho dev split, output tuỳ chỉnh
python scripts/extract_objects.py --split dev --output data/dev_objects.json

# Kiểm tra lại file đã trích xuất
python scripts/extract_objects.py --split test --max-samples 5
# Output mặc định: outputs/test_objects.json (EXTRACTION_OUTPUT_FILE trong config)
```

**Output:** File JSONL, mỗi dòng chứa `id`, `question`, `Object` (list O1/O2), `O2_is_viewer`, `confidence`, `O1_hallucinated`, `raw_json`.

### 5.6. Ablation Harness cho Module 1 (`scripts/run_module1_ablation.py`)

Đánh giá cascading error — mỗi scenario đo lường impact của O1/O2 detection quality lên accuracy:

| Scenario | Mô tả |
|---|---|
| `baseline` | Zero-shot, không augment |
| `oracle` | O1/O2 từ regex (upper bound) |
| `detected` | O1/O2 từ LLM extractor |
| `perturbed` | Intentional O1/O2 swap (đo harm của sai hints) |

```bash
# Chạy 100 sample dev, so sánh 4 scenarios
python scripts/run_module1_ablation.py --max-samples 100

# Xuất report markdown tuỳ chỉnh
python scripts/run_module1_ablation.py \
    --max-samples 100 \
    --output results/my_ablation.md

# Chạy trên model 8B
python scripts/run_module1_ablation.py \
    --max-samples 50 \
    --model Qwen3-VL-8B
```

**Output:** Markdown table + JSON, xuất vào `outputs/module1_ablation.md` và `outputs/module1_ablation.json`.

---

## 6. Kịch bản Thực nghiệm

Có **3 kịch bản thực nghiệm** chính:

| Kịch bản | Modules | Mô tả | Mục đích |
|---|---|---|---|
| **Baseline** | — | Qwen3-VL Instruct gốc, prompt tiêu chuẩn | Đo lường năng lực không gian nguyên bản của model |
| **Full ASTRA** | M1+M2+M3 | Tích hợp đồng thời OGM (Set-of-Mark), DLC (Depth Cue) và ODV (Circular Shift Voting) | Đánh giá hiệu quả tổng thể của hệ thống bổ trợ không gian |
| **Escalation** | M1+M2+M3 + zero-shot gating | Zero-shot voting trước → nếu đồng thuận cao → STOP; nếu disagreement → augment với M1+M2 | Tối ưu chi phí: tránh chạy M1+M2 khi zero-shot đã đủ tin cậy |

**Escalation logic chi tiết:**

```
Zero-shot voting (K=3 permutations)
         │
         ├─ Tất cả votes đồng ý (K/K) ──► Trả về kết quả zero-shot
         │
         └─ Disagreement ──► Extract entities (LLM) ──► Verify (hallucination check)
                                │
                                ├─ Extraction failed ──► Fallback zero-shot
                                │
                                ├─ OGM failed ──► Fallback zero-shot
                                │
                                └─ OK ──► M1 (OGM marks) + M2 (depth cue) + M3 (augmented voting)
                                          ──► Trả về kết quả augmented
```

**Cấu trúc thư mục kết quả:**

```
outputs/astra/
├── 2B/
│   ├── baseline/results.jsonl     # Baseline model gốc
│   ├── baseline/metrics.json
│   ├── ASTRA_full/results.jsonl   # Full ASTRA (M1+M2+M3)
│   ├── ASTRA_full/metrics.json
│   └── summary.json
├── 4B/
│   └── ...
├── 8B/
│   └── ...
└── comparison.csv                  # Bảng so sánh tất cả model × kịch bản
```

---

## 7. Chạy Tự động bằng Shell / PowerShell Scripts

### 7.1. Trên Linux / macOS / Git Bash

```bash
# Cấp quyền thực thi
chmod +x scripts/*.sh

# Chạy cho từng model riêng lẻ
bash scripts/eval_2B.sh
bash scripts/eval_4B.sh
bash scripts/eval_8B.sh

# Chạy tất cả model liên tiếp (2B → 4B → 8B)
bash scripts/run_all_models.sh
```

### 7.2. Trên Windows PowerShell

```powershell
# Chạy thực nghiệm cho model 2B, 4B, hoặc 8B
.\astra_eval.ps1 -Model "4B" -Split "test"

# Chạy nhanh trên 50 sample để test luồng
.\astra_eval.ps1 -Model "2B" -Split "test_500" -MaxSamples 50
```

---

## 8. Ước tính Tài nguyên & VRAM

ASTRA hoạt động thuần inference, không yêu cầu lưu gradient hay trạng thái optimizer nên vô cùng tiết kiệm tài nguyên:

| Model | Cấu hình VRAM đề xuất | Tốc độ Inference ước tính (GPU RTX 4090/A100) | Ghi chú |
|---|---|---|---|
| **Qwen3-VL-2B-Instruct** | $\ge$ 8 GB | ~1.2s - 1.5s / sample | Phù hợp test nhanh, chạy trên GPU laptop/đơn giản |
| **Qwen3-VL-4B-Instruct** | $\ge$ 12 GB | ~1.8s - 2.2s / sample | **Model mặc định**, cân bằng hoàn hảo giữa tốc độ & độ chính xác |
| **Qwen3-VL-8B-Instruct** | $\ge$ 20 GB | ~3.0s - 3.8s / sample | Đạt độ chính xác suy luận không gian cao nhất |

> [!TIP]
> Khi bật **Module 3 (ODV)** với $K=3$ hoán vị, thời gian inference mỗi câu hỏi sẽ tăng khoảng $\approx 2.5\times$ so với chạy đơn lẻ (do thực hiện 3 lần forward pass zero-shot). Tuy nhiên, VRAM tiêu thụ đỉnh (Peak VRAM) không thay đổi vì các pass được thực hiện tuần tự.

---

## Cap Nhat Nhanh

- Anh cho `test.jsonl` da duoc tach ra thanh `dataset/images/test_images/` tu `dataset/images/COCO2017/`.
- Neu chi muon chay Module 1 (OGM), dung:

```bash
python main.py eval \
    --model Qwen3-VL-4B \
    --modules 1 \
    --split test \
    --output outputs/astra/4B/M1_only/results.jsonl
```

- `--modules 1` nghia la chi bat M1, khong chay M2/DLC va M3/ODV.
- Neu muon chay full ASTRA thi bo qua `--modules`:

```bash
python main.py eval \
    --model Qwen3-VL-4B \
    --split test \
    --output outputs/astra/4B/ASTRA_full/results.jsonl
```
