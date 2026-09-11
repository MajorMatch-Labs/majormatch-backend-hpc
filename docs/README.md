# TÀI LIỆU KỸ THUẬT BACKEND HPC (PRIVATE COMPUTE ENGINE)
## HỆ THỐNG CỐ VẤN HỌC TẬP & ĐỊNH HƯỚNG NGHỀ NGHIỆP MAJORMATCH

Thư mục này chứa toàn bộ hồ sơ kỹ thuật, nhật ký phát triển, lộ trình triển khai và thông số công nghệ của máy chủ tính toán nội bộ (`backend-hpc`).

---

## 📚 MỤC LỤC TÀI LIỆU

1. 📄 **[Nhật ký Phát triển (DEVELOPMENT_JOURNAL.md)](./DEVELOPMENT_JOURNAL.md)**:  
   Ghi chép chi tiết từng mốc công việc đã hoàn thành, các quyết định kiến trúc then chốt (Phase 1, Phase 2) và phân định rõ ràng giữa tầng ML định lượng và tầng Local LLM định tính.

2. 🗺️ **[Lộ trình Phát triển 6 Bước (ROADMAP.md)](./ROADMAP.md)**:  
   Sơ đồ tiến độ 6 giai đoạn phát triển: Môi trường ảo $\rightarrow$ Bộ dữ liệu huấn luyện $\rightarrow$ Huấn luyện ML Model $\rightarrow$ Tích hợp Engine $\rightarrow$ Tầng Local Model & RAG $\rightarrow$ Kiểm thử E2E & Đồng bộ Git.

3. 🛠️ **[Ngăn xếp Công nghệ & Phần cứng (TECH_STACK.md)](./TECH_STACK.md)**:  
   Đặc tả chi tiết cấu hình máy trạm (Legion 5 Pro, i9-13900HX, RTX 4060 8GB VRAM) cùng danh mục 113 thư viện phần mềm chuyên dụng (FastAPI, Scikit-learn, NumPy, Pandas, Joblib, ChromaDB, Ollama Qwen 2.5).

---

## 🚀 HƯỚNG DẪN KHỞI CHẠY NHANH TẠI CỤC BỘ

```bash
cd backend-hpc

# Kích hoạt môi trường ảo Python 3.10
.venv\Scripts\activate

# Khởi chạy máy chủ FastAPI Server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

* Truy cập Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Truy cập ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
