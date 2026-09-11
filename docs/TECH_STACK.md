# NGĂN XẾP CÔNG NGHỆ & THÔNG SỐ KỸ THUẬT (TECH STACK SPECIFICATION)
## DỰ ÁN: MAJORMATCH - PRIVATE COMPUTE ENGINE (BACKEND-HPC)

---

## 1. MÔI TRƯỜNG PHẦN CỨNG & HỆ THỐNG
* **Thiết bị vận hành:** Laptop Lenovo Legion 5 Pro (16IRX8)
* **Hệ điều hành:** Windows 11 Home Single Language (64-bit)
* **CPU:** Intel Core i9-13900HX (24 nhân, 32 luồng, xung nhịp lên tới 5.4GHz, 36MB Intel Smart Cache)
* **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM GDDR6, Max TGP 140W, Kiến trúc Ada Lovelace, Nhân Tensor Gen 4 hỗ trợ FP8/FP16/BF16)
* **RAM:** 16GB DDR5 5600MHz (Băng thông cao, chạy đa nhiệm mượt mà)
* **Ổ cứng:** 1TB SSD M.2 PCIe Gen 4 NVMe (Tốc độ đọc/ghi ~5000MB/s)

---

## 2. NGĂN XẾP CÔNG NGHỆ PHẦN MỀM (SOFTWARE TECH STACK)

### 2.1. Tầng Nền tảng & API Gateway (Web & Server Framework)
| Công nghệ | Phiên bản | Mục đích & Vai trò |
| :--- | :--- | :--- |
| **Python** | 3.10.11 | Ngôn ngữ thực thi chính, tối ưu độ tương thích với các thư viện ML/AI C-extensions. |
| **FastAPI** | 0.111.1 | Web framework bất đồng bộ (Asynchronous) hiệu năng cao, tự động sinh tài liệu Swagger/OpenAPI 3.1. |
| **Uvicorn** | 0.29.0 | Máy chủ ASGI Server chuẩn công nghiệp, hỗ trợ HTTP/1.1, WebSocket và Server-Sent Events. |
| **Pydantic v2** | 2.13.5 | Xác thực dữ liệu đầu vào/đầu ra nghiêm ngặt với hiệu năng biên dịch bằng Rust (`pydantic-core`). |
| **uv** | 0.11.x | Trình quản lý môi trường ảo và cài đặt thư viện siêu tốc độ của Astral. |

### 2.2. Tầng Bóc tách Dữ liệu & Xử lý Tài liệu (Ingestion & Parsing)
| Công nghệ | Phiên bản | Mục đích & Vai trò |
| :--- | :--- | :--- |
| **pdfplumber** | 0.11.10 | Trích xuất văn bản có cấu trúc và vị trí tọa độ bảng điểm từ file PDF đại học. |
| **pypdf** | 6.18.0 | Xử lý file PDF nhị phân, kiểm tra header magic bytes `%PDF-1.x`. |
| **Regex Engine** | Built-in | Nhận diện mã môn học (`CS[0-9]{3}`), hệ số tín chỉ, điểm chữ (A, B+, C, D, F) và khử PII (tên, MSSV, ngày sinh). |

### 2.3. Tầng Động cơ Học máy Định lượng (Quantitative ML Core)
| Công nghệ | Phiên bản | Mục đích & Vai trò |
| :--- | :--- | :--- |
| **NumPy** | 1.26.4 | Tính toán đại số tuyến tính, nhân ma trận, tính chuẩn độ dài vector (Euclidean Norm). |
| **Scikit-learn** | 1.5.2 | Tính toán **Cosine Similarity**, huấn luyện mô hình phân loại **Random Forest**, chuẩn hóa Min-Max Scaling. |
| **Pandas** | 2.3.3 | Xử lý tập dữ liệu sinh viên huấn luyện (`students_training.csv`), tạo DataFrames. |
| **Joblib** | 1.6.0 | Tuần tự hóa (Serialization) và lưu trữ checkpoint mô hình ML (`.joblib`) nạp vào runtime. |

### 2.4. Tầng Mô hình Cục bộ & RAG (Local LLM & Vector Store)
| Công nghệ | Phiên bản | Mục đích & Vai trò |
| :--- | :--- | :--- |
| **ChromaDB** | 1.5.9 | Cơ sở dữ liệu vector nhúng cục bộ (Embedded Vector Database), lưu trữ khung chương trình môn học. |
| **Ollama** | 0.32.5 | Nền tảng điều phối suy luận mô hình ngôn ngữ cục bộ tăng tốc bằng phần cứng GPU CUDA. |
| **Qwen 2.5 7B** | Instruct Q4_K_M | Mô hình ngôn ngữ lớn nguồn mở 7 tỷ tham số, tối ưu tiếng Việt, suy luận trên GPU RTX 4060 (~4.5GB VRAM). |
| **SSE-Starlette** | 3.0.3 | Đẩy luồng phản hồi cố vấn trực tiếp (Streaming Tokens) đến trình duyệt người dùng qua Server-Sent Events. |

---

## 3. CHỈ TIÊU KỸ THUẬT & AN TOÀN TÀI NGUYÊN (PERFORMANCE BENCHMARKS)
* **Độ trễ tính toán ML:** $< 5\text{ms}$ cho một yêu cầu phân tích khoảng cách kỹ năng.
* **Thời gian bóc tách PDF:** $< 500\text{ms}$ cho tệp bảng điểm 3 trang.
* **Tốc độ sinh token Local LLM:** $\approx 40 - 50\text{ tokens/giây}$ trên GPU RTX 4060.
* **Chiếm dụng RAM máy:** $\approx 150\text{MB}$ khi chạy FastAPI server; $\approx 4.5\text{GB VRAM}$ khi nạp model 7B.
* **Bảo mật PII:** 100% dữ liệu danh tính được khử cục bộ trong bộ nhớ RAM, giải phóng rác (`gc.collect()`) ngay sau khi xử lý xong.
