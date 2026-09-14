# ====================================================================
# QUY TẮC ĐỒNG BỘ PUSH CODE & AGENTIC RULES CHO BACKEND HPC
# ====================================================================

Mọi AI Agent làm việc trên repository `majormatch-backend-hpc` bắt buộc phải tuân thủ nghiêm ngặt các điều khoản sau:

---

## 1. THÔNG TIN PHỤ TRÁCH & TÁC GIẢ COMMIT (AUTHOR MAPPING)
Khi thực hiện commit hoặc tạo nhánh cho module Backend HPC, Agent **BẮT BUỘC** sử dụng đúng thông tin tác giả và email tương ứng của Tech Lead Long Nhật:

| Vai trò | Phụ trách chính | Thư mục mã nguồn | Author Name | Email GitHub chính xác | Tiền tố nhánh (Branch Prefix) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tech Lead** | AI/ML Core, Local LLM, RAG | `backend-hpc/` (hoặc root khi tách repo) | `NhatPrv` | `torikun2005@gmail.com` | `feat/longnhat-w<tuần>-...` |

---

## 2. NGUYÊN TẮC ÁNH XẠ NHÁNH & PULL REQUEST
1. **Cùng tên nhánh tuyệt đối (Exact Branch Matching)**:
   * Mọi tính năng phát triển phải thực hiện trên nhánh có định dạng: `<loại>/<tên>-w<tuần>-<tính-năng>`.
2. **Nghiêm cấm Push trực tiếp vào `main`**:
   * Nhánh `main` của repo chỉ nhận code thông qua Pull Request sau khi được Tech Lead review và merge.
3. **Nhật ký phát triển & Kiến trúc (Documentation Integrity)**:
   * Mọi can thiệp vào tầng AI/ML, Ingestion, API đều phải cập nhật tài liệu tương ứng tại `docs/DEVELOPMENT_JOURNAL.md` và `docs/ROADMAP.md`.

---

## 3. TIÊU CHUẨN COMMIT TIẾNG ANH (CONVENTIONAL COMMITS)
* Mọi commit phải viết bằng tiếng Anh theo chuẩn Conventional Commits:
  ```bash
  git commit -m "<type>(<scope>): <short description in English>" --author="NhatPrv <torikun2005@gmail.com>"
  ```
* Các loại hợp lệ: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`.

---

## 4. ĐIỀU KHOẢN AN TOÀN & BẢO MẬT
1. **Tuyệt đối cấm commit file `.docx`** vào bất kỳ nhánh nào của repository.
2. **Tuyệt đối cấm commit môi trường ảo hoặc artifacts nhị phân nặng**: Thư mục `.venv/`, `__pycache__/`, cache weights mô hình LLM tuyệt đối nằm trong `.gitignore`.
3. **Cấm code giả lập rỗng**: Không sử dụng `pass` hay `// TODO: implement later` trong các hàm sản phẩm. Mọi hàm phải có logic xử lý và fallback hoàn chỉnh.
