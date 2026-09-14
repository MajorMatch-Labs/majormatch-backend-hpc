# ====================================================================
# QUY CHUẨN KỸ THUẬT & LẬP TRÌNH BACKEND HPC - AI/ML (CODING STANDARDS)
# ====================================================================

Bộ quy tắc kỹ thuật tối cao dành cho AI Agent khi sinh mã nguồn, tái cấu trúc, huấn luyện mô hình hoặc tối ưu hóa hiệu năng module MajorMatch Backend HPC (`backend-hpc`).

---

## 1. NGUYÊN TẮC KIẾN TRÚC & NỀN TẢNG PYTHON
1. **Môi trường & Ngôn ngữ**:
   * Python 3.10.11 trong virtual environment `.venv` (quản lý bởi `uv`).
   * Tối ưu mã nguồn tương thích với C-extensions (NumPy, Scikit-learn, PyTorch/ChromaDB).
2. **Web Framework & API**:
   * Sử dụng **FastAPI** với mô hình bất đồng bộ (`async def`) cho các tác vụ I/O-bound (đọc tài liệu, gọi API, Server-Sent Events).
   * Sử dụng `run_in_threadpool` hoặc background worker cho các tác vụ CPU-bound nặng (huấn luyện model, tính toán ma trận lớn) để tránh block event loop chính của ASGI Uvicorn.
3. **Kiểm soát kiểu dữ liệu (Strict Typing & Pydantic v2)**:
   * Toàn bộ tham số hàm và giá trị trả về phải có type hints rõ ràng (`typing`, `pydantic.BaseModel`).
   * Mọi request body, query params và response payload phải được định nghĩa bằng Pydantic v2 schemas với validation tường minh (`Field(..., ge=0, le=10)`).
   * Tuyệt đối cấm trả về `dict` tự do không có schema định hình.

---

## 2. QUY CHUẨN HỌC MÁY (QUANTITATIVE ML) & ĐỘNG CƠ VECTOR
1. **Tính toán đại số & Khoảng cách kỹ năng**:
   * Sử dụng NumPy và Scikit-learn cho các phép tính vector: **Cosine Similarity** so khớp hồ sơ sinh viên với chuẩn đầu ra chuyên ngành (CS_DATA_AI, SE_FULLSTACK, DEVOPS_CLOUD, CYBER_SECURITY, DATA_ANALYTICS).
   * Chuẩn hóa dữ liệu đầu vào bằng `MinMaxScaler` hoặc `StandardScaler`. Tuyệt đối cấm tính similarity trên các thang đo chưa chuẩn hóa (unscaled raw values).
2. **Huấn luyện mô hình phân loại (Classification)**:
   * Sử dụng **Random Forest** hoặc **Gradient Boosting** cho bài toán dự đoán chuyên ngành dựa trên bảng điểm và 6 nét tính cách Holland RIASEC.
   * Luôn cố định `random_state=42` để đảm bảo kết quả tái lập được (deterministic reproducibility).
   * Chỉ tiêu chất lượng: Mô hình phải đạt Accuracy > 85% và F1-Score (macro) > 85% trên tập kiểm thử (test set).
3. **Quản lý checkpoint & Artifacts**:
   * Model và Scaler sau khi huấn luyện được serialize bằng `joblib` và lưu trữ tại `ml/models/`.
   * Cung cấp cơ chế tự động nạp model khi khởi động server (`lifespan` handler trong FastAPI).

---

## 3. TÍCH HỢP LOCAL LLM & VECTOR STORE (QUALITATIVE AI)
1. **Mô hình cục bộ Ollama Qwen 2.5 7B**:
   * Tận dụng card đồ họa rời NVIDIA RTX 4060 Laptop GPU (8GB VRAM) qua kiến trúc CUDA.
   * Luôn thiết lập timeout (`timeout=30.0s`) cho các truy vấn suy luận LLM để tránh treo tiến trình.
   * Truyền tải phản hồi cố vấn qua **Server-Sent Events (SSE)** với `sse-starlette` để người dùng nhận token trực tiếp (real-time stream).
2. **Cơ sở dữ liệu Vector (ChromaDB)**:
   * Khởi tạo ChromaDB client ở chế độ Persistent hoặc In-Memory nhúng nhẹ.
   * Sử dụng collection tách biệt cho chuẩn đầu ra môn học (`curriculum_standards`) và tài liệu hướng nghiệp.
3. **Cơ chế Fallback bảo đảm khả năng demo (Fault-Tolerant)**:
   * Trong trường hợp Ollama chưa khởi động hoặc GPU bị bận, hệ thống **bắt buộc** phải tự động chuyển sang `Rule-Based Fallback` hoặc `Deterministic Template Response`, tuyệt đối không làm crash server hoặc trả về mã lỗi 500 cho Client.

---

## 4. QUẢN TRỊ BỘ NHỚ & AN TOÀN PHẦN CỨNG (LEGION 5 PRO OPTIMIZATION)
1. **Kiểm soát giới hạn 16GB RAM & 8GB VRAM**:
   * Giải phóng tài nguyên ngay lập tức sau các tác vụ nặng bằng `gc.collect()`.
   * Tuyệt đối cấm nạp toàn bộ dataset vào bộ nhớ nhiều lần mà không giải phóng biến trung gian.
2. **Khử danh tính & Bảo mật PII (Personally Identifiable Information)**:
   * 100% dữ liệu danh tính cá nhân (Họ tên, Mã số sinh viên, Ngày sinh, Lớp sinh hoạt) từ bảng điểm PDF phải được bóc tách và khử sạch (Sanitization) ngay trong RAM trước khi chuyển đến ML Engine hay LLM.
   * **Tuyệt đối cấm** lưu thông tin PII của sinh viên xuống đĩa cứng (`.txt`, `.json`, `.csv`, `.log`).

---

## 5. TIÊU CHUẨN MÃ NGUỒN SẠCH & LOGGING
1. **Không code rỗng**: Tuyệt đối cấm `pass` hoặc `raise NotImplementedError` trong các luồng chính.
2. **Logging chuẩn mực**:
   * Sử dụng thư viện `logging` của Python thay cho lệnh `print()` thông thường trong production endpoints.
   * Định dạng log: `[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s`.
3. **Xử lý ngoại lệ**:
   * Mọi endpoint phải bọc trong khối `try...except HTTPException` với mã trạng thái HTTP chuẩn (400 cho bad input, 422 cho validation error, 500 cho internal failure kèm thông điệp rõ ràng).
