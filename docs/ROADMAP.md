# LỘ TRÌNH PHÁT TRIỂN BACKEND HPC (IMPLEMENTATION ROADMAP)
## DỰ ÁN: MAJORMATCH - PRIVATE COMPUTE ENGINE

Lộ trình triển khai Backend HPC được chia thành 6 bước tuần tự, bảo đảm tính ổn định, dễ tái lập và tối ưu tài nguyên phần cứng máy trạm.

---

```
                       LỘ TRÌNH 6 BƯỚC TRIỂN KHAI
 
  ┌─────────────────────────────────────────────────────────────┐
  │ BƯỚC 1: MÔI TRƯỜNG & THƯ VIỆN [ĐÃ HOÀN THÀNH ✅]            │
  │ • Tạo môi trường ảo Python 3.10 (.venv) bằng `uv`.          │
  │ • Cài đặt 113 thư viện (fastapi, scikit-learn, pandas...).  │
  │ • Sanity Check import thành công 100%.                      │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ BƯỚC 2: BỘ DỮ LIỆU HUẤN LUYỆN (DATASET PREPARATION) [TIẾP THEO]│
  │ • Tạo thư mục `backend-hpc/ml/dataset/`.                    │
  │ • Viết script `generate_dataset.py` sinh 2.000 mẫu sinh viên│
  │   chuẩn (`students_training.csv`).                          │
  │ • Đặc trưng: Điểm môn đại cương + 6 điểm Holland RIASEC.    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ BƯỚC 3: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH ML (TRAIN & EVALUATE) │
  │ • Viết và chạy `backend-hpc/ml/train.py`.                   │
  │ • Huấn luyện: RandomForestClassifier + GradientBoosting.     │
  │ • Đánh giá: Accuracy, Precision, Recall, F1-Score (> 88%).  │
  │ • Xuất file checkpoint: `ml/models/career_classifier.joblib`.│
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ BƯỚC 4: TÍCH HỢP MODEL VÀO ĐỘNG CƠ ML (`ml_engine.py`)     │
  │ • Nạp file `career_classifier.joblib` vào `ml_engine.py`.   │
  │ • Kết hợp 2 trong 1:                                        │
  │   - Xác suất thống kê chuyên ngành từ mô hình Train (%).     │
  │   - Tọa độ biểu đồ Radar 6 trục & Phân rã 3 nhóm kỹ năng.  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ BƯỚC 5: TẦNG LOCAL MODEL & RAG (`rag_service.py`)           │
  │ • Khởi tạo ChromaDB lưu khung chương trình đào tạo của trường│
  │ • Cấu hình kết nối Ollama Qwen 2.5 (GPU RTX 4060).          │
  │ • Viết cơ chế Rule-based Fallback chống crash khi tắt Ollama│
  │ • Hoàn thiện logic Streaming Chatbot (Server-Sent Events).  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ BƯỚC 6: KHỞI CHẠY SERVER & KIỂM THỬ TOÀN DIỆN (E2E & SMOKE) │
  │ • Khởi động máy chủ: `uvicorn main:app --port 8000`.        │
  │ • Kiểm thử thực tế 6 Endpoints qua Swagger UI (/docs).      │
  │ • Viết bộ kiểm thử tự động `tests/test_backend.py`.         │
  │ • Đồng bộ Git hoàn tất lên repo `majormatch-backend-hpc`.   │
  └─────────────────────────────────────────────────────────────┘
```

---

## TIẾN ĐỘ THỰC TẾ
- [x] **Bước 1**: Môi trường & Thư viện (Hoàn thành 11/09/2026)
- [ ] **Bước 2**: Bộ dữ liệu huấn luyện `students_training.csv`
- [ ] **Bước 3**: Script huấn luyện `train.py` & Checkpoint `.joblib`
- [ ] **Bước 4**: Tích hợp Model vào `ml_engine.py`
- [ ] **Bước 5**: Tầng Local Model & RAG Fallback
- [ ] **Bước 6**: Khởi chạy Server, Swagger Test & Đồng bộ Git
