# NHẬT KÝ PHÁT TRIỂN BACKEND HPC (DEVELOPMENT JOURNAL)
## DỰ ÁN: MAJORMATCH - PRIVATE AI COMPUTE ENGINE
**Tác giả:** Long Nhật (Tech Lead / Core Architect)  
**Phần cứng vận hành:** Lenovo Legion 5 Pro (Intel Core i9-13900HX, NVIDIA GeForce RTX 4060 8GB VRAM, 16GB DDR5 RAM, Windows 11)  
**Mục tiêu:** Xây dựng hệ thống máy chủ tính toán nội bộ (On-Premise) kết hợp Động cơ Machine Learning định lượng và Mô hình Local LLM phục vụ cố vấn hướng nghiệp đại học.

---

## 1. LỊCH SỬ CÔNG VIỆC ĐÃ HOÀN THÀNH

### Ngày 05/09/2026 - Khởi tạo Kiến trúc Cốt lõi (Phase 1)
- **Thiết kế API Contract**: Hiện thực 6 Endpoints chuẩn OpenAPI 3.1 trong `main.py` tuân thủ chuẩn RFC 7807 (Problem Details).
- **Bộ Schemas Pydantic v2**: Xây dựng toàn bộ Data Transfer Objects (DTO) trong `schemas.py` với cấu hình nghiêm ngặt (`extra="forbid"`, type safety).
- **Bóc tách PDF & Khử PII (`parser.py`)**: Tích hợp `pdfplumber` kết hợp bộ Regex Engine bóc tách danh sách môn học, số tín chỉ, điểm chữ/hệ số và tự động ẩn danh thông tin cá nhân.
- **Khung ML Vector Space (`ml_engine.py`)**: Thiết lập không gian 128 chiều kỹ năng (`CORE_SKILLS`), ma trận môn học đại học (`COURSE_SKILL_MAPPING`) và 5 hồ sơ chuyên ngành chuẩn (`MAJOR_BENCHMARKS`).
- **Khung RAG & Local AI (`rag_service.py`)**: Thiết lập tích hợp ChromaDB Vectorstore và Ollama Qwen 2.5 7B với cơ chế Semaphore bảo vệ VRAM GPU.

---

### Ngày 11/09/2026 - Chuẩn hóa Môi trường & Thiết kế Quy trình Huấn luyện ML (Phase 2)
- **Thiết lập Môi trường ảo (`.venv`)**:
  - Khởi tạo môi trường ảo Python 3.10.11 độc lập trong thư mục `backend-hpc/.venv` bằng công cụ `uv`.
  - Cập nhật `requirements.txt` bổ sung `pandas>=2.2.0` và `joblib>=1.3.0`.
  - Cài đặt thành công 113 packages chuyên dụng: `fastapi==0.111.1`, `scikit-learn==1.5.2`, `numpy==1.26.4`, `pandas==2.3.3`, `joblib==1.6.0`, `chromadb==1.5.9`, `uvicorn==0.29.0`, `pdfplumber==0.11.10`.
  - Thực thi kiểm tra khởi động (Sanity Check) thành công 100%.
- **Định hình lại Kiến trúc 2 Tầng**:
  - Xác định rõ mô hình: **100% On-Premise (Không Cloud AI ngoài, không SaaS)**.
  - **Tầng 1 (ML)**: Đảm nhận bài toán định lượng toán học, tính Cosine Similarity, phân loại kỹ năng (Mastered, Developing, Missing) và dự đoán xác suất chuyên ngành từ mô hình Supervised Learning.
  - **Tầng 2 (Local Model)**: Đảm nhận bài toán định tính ngôn ngữ, sử dụng Ollama `qwen2.5:7b` trên GPU RTX 4060 để diễn giải kết quả, RAG sinh lộ trình học tập và trợ lý ảo Streaming Chatbot.
- **Quyết định Huấn luyện Model ML Thực thụ**:
  - Thống nhất bổ sung quy trình Train Model thực sự (`train.py`) với tập dữ liệu sinh viên mẫu `students_training.csv` (2.000 dòng) để phục vụ chấm điểm bài tập lớn và thuyết trình đồ án trước hội đồng.
