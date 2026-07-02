# ASTRA - Phương pháp luận & Kiến trúc hệ thống

Tài liệu này mô tả chi tiết phương pháp tiếp cận và cấu trúc của 3 module lõi trong dự án **ASTRA** (*Auxiliary Spatial Tools for Robust Answering*). Khác biệt với các phương pháp dựa trên Fine-Tuning truyền thống, ASTRA hoàn toàn hoạt động ở pha **Inference (Zero-shot)** bằng cách tự động cung cấp thêm các gợi ý về không gian (thị giác, độ sâu) và sử dụng cơ chế bỏ phiếu thông minh để khắc phục các bias cố hữu của mô hình Ngôn ngữ - Thị giác lớn (VLM).

---

## 1. Kiến trúc Tổng thể (Pipeline)

Quy trình xử lý một câu hỏi (sample) đi qua ASTRA được biểu diễn theo sơ đồ dưới đây:

```mermaid
flowchart TD
    %% Định nghĩa dữ liệu đầu vào
    Input_Image[Ảnh gốc]
    Input_Question[Câu hỏi không gian SpatialMQA]
    
    %% Extract Entities
    subgraph Preprocessing[Tiền xử lý]
        LLM_Extractor[LLM Extractor / Regex]
        Entities[Thực thể mục tiêu: O1, O2]
    end
    
    Input_Question --> LLM_Extractor
    LLM_Extractor --> Entities
    
    %% Parallel Modules
    subgraph Auxiliary_Modules[Các Module Bổ trợ Song song]
        direction LR
        M1[Module 1: OGM<br/>(Object-Grounded Marking)]
        M2[Module 2: DLC<br/>(Depth-Layer Cue)]
    end
    
    Input_Image --> M1
    Entities --> M1
    M1 -. Grounding DINO .-> Marked_Image[Ảnh đã đánh dấu Set-of-Mark]
    
    Input_Image --> M2
    Entities --> M2
    M1_Bboxes[Tọa độ Bbox O1, O2] --> M2
    M1 -- Gửi Bbox --> M1_Bboxes
    M2 -. Depth-Anything-V2 .-> Depth_Hint[Gợi ý độ sâu - Soft Hint]
    
    %% Main VLM & Voting
    subgraph Core_Reasoning[Suy luận cốt lõi & Khử Bias]
        M3[Module 3: ODV<br/>(Order-Debiased Voting)]
        VLM_Zero_Shot[Truy vấn Zero-shot<br/>Qwen3-VL]
        Voting[Majority Voting]
    end
    
    Input_Question --> M3
    Marked_Image --> M3
    Depth_Hint --> M3
    M3 -- K=3 Hoán vị đáp án --> VLM_Zero_Shot
    VLM_Zero_Shot -- Ánh xạ đáp án --> Voting
    Voting --> Final_Answer((Đáp án Cuối cùng))
    
    %% Styling
    classDef io fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef module fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef core fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    
    class Input_Image,Input_Question,Final_Answer io;
    class M1,M2,LLM_Extractor module;
    class M3,VLM_Zero_Shot,Voting core;
```

---

## 2. Chi tiết các Module Bổ trợ

### Module 1 — Object-Grounded Marking (OGM)

**Vấn đề:** Các model VLM thường gặp khó khăn trong việc "neo" (grounding) chính xác sự chú ý của nó vào đúng thực thể được hỏi trong bức ảnh, đặc biệt khi có nhiều vật thể giống nhau.

**Phương pháp (OGM):**
1. **Trích xuất Thực thể:** Từ câu hỏi của người dùng, module sử dụng LLM (hoặc Regex truyền thống) để trích xuất ra hai đối tượng chính cần quan tâm: `O1` và `O2`. (VD: Câu hỏi: *"Where is the dog relative to the cat?"* -> `O1`: dog, `O2`: cat).
2. **Phát hiện Vị trí:** Sử dụng mô hình **Grounding DINO-Tiny** (rất nhẹ, ~172M parameters) nhận đầu vào là ảnh gốc và prompt text (`O1`, `O2`) để dự đoán Bounding Box của chúng trên ảnh.
3. **Đánh dấu Trực quan (Set-of-Mark):** 
   - ASTRA tự động vẽ các hộp đỏ và nhãn trực quan lên hình. Cụ thể, O1 được đánh dấu là `[1]` và O2 được đánh dấu là `[2]`.
   - Các nhãn này giúp Qwen3-VL chú ý trực tiếp vào vùng không gian quan trọng.
4. **Cơ chế Fallback:** Nếu Grounding DINO có độ tin cậy (confidence score) thấp hơn một ngưỡng quy định (ví dụ `< 0.3`), module sẽ đánh giá là nhận diện thất bại và bỏ qua việc vẽ lên ảnh, hệ thống sẽ tự động dùng ảnh gốc để tránh hiện tượng đánh dấu sai làm nhiễu VLM (Cascading errors).

---

### Module 2 — Depth-Layer Cue (DLC)

**Vấn đề:** Các ảnh 2D thiếu thông tin chiều sâu. Khi hỏi về mối quan hệ "trước/sau" (in front of / behind), VLM rất dễ nhầm lẫn nếu không có thông tin về độ xa gần.

**Phương pháp (DLC):**
1. **Ước lượng Độ sâu:** Sử dụng **Depth-Anything-V2-Small**, hệ thống ước lượng một bản đồ độ sâu tương đối (relative depth map) của toàn bộ bức ảnh.
2. **Tính toán Độ sâu Thực thể:** Sử dụng tọa độ Bounding Box từ Module 1, DLC sẽ tính **mean depth** (giá trị độ sâu trung bình) bên trong các box của `O1` và `O2`.
3. **Gợi ý Mềm (Soft Hint):** Thay vì áp đặt một sự thật hiển nhiên (hard rule), hệ thống tự động sinh ra một gợi ý dưới dạng văn bản (prompt augmentation). Ví dụ: 
   *"Depth hint (auxiliary, may be inaccurate): object [1] (dog) appears closer to the camera than object [2] (cat) based on an external depth model. Please cross-check this..."*
   Việc đóng khung gợi ý này như một thông tin "phụ trợ" giúp VLM tận dụng tốt manh mối về độ sâu để suy luận, nhưng vẫn dựa trên sự quan sát của chính VLM để chống lại ảo giác (hallucination).

---

### Module 3 — Order-Debiased Voting (ODV)

**Vấn đề:** Hiện tượng **Blind Spatial Token Favouring (BSTF)**. Các VLM có sự thiên lệch (bias) rất lớn đối với vị trí của đáp án. Ví dụ, nó có xu hướng chọn đáp án A thay vì B, C ngay cả khi nội dung đáp án giống nhau, chỉ vì thứ tự hiển thị.

**Phương pháp (ODV):**
1. **Hoán vị Dịch vòng (Circular Shift):** Thay vì chỉ hỏi VLM một lần, module sinh ra $K = 3$ phiên bản hoán vị dịch vòng của danh sách đáp án. 
   - Truy vấn 1: A. left of, B. right of, C. behind
   - Truy vấn 2: A. right of, B. behind, C. left of
   - Truy vấn 3: A. behind, B. left of, C. right of
2. **Suy luận Zero-shot Nhiều lần:** Hình ảnh (đã được đánh dấu từ M1) và prompt (đã tiêm hint từ M2) được đưa vào mô hình Qwen3-VL để hỏi 3 lần tương ứng với 3 hoán vị.
3. **Bỏ phiếu Đa số (Majority Voting):** Câu trả lời của VLM (A, B, C...) được map ngược lại thành nội dung text nguyên bản ("left of", "right of"...). Sau đó, ASTRA áp dụng thuật toán Majority Vote để đưa ra quyết định cuối cùng có sự đồng thuận cao nhất. Cơ chế này khử hoàn toàn tác động của BSTF, giúp hiệu suất của hệ thống ổn định và khách quan hơn.
