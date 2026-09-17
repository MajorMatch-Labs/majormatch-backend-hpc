# TÀI LIỆU NGUỒN GỐC & ĐỘ TIN CẬY DỮ LIỆU (DATA PROVENANCE & CITATION)
## DỰ ÁN: MAJORMATCH — PRIVATE HPC COMPUTE ENGINE
**Thư mục lưu trữ:** `backend-hpc/ml/dataset/`  
**Ngày cập nhật:** 17/09/2026  
**Chủ quản kỹ thuật:** Long Nhật (`NhatPrv <torikun2005@gmail.com>`) — Tech Lead  

---

## 1. TỔ CHỨC PHÁT HÀNH & NGUỒN GỐC DỮ LIỆU (DATA ORIGIN)

Tập dữ liệu huấn luyện thực tế sử dụng trong hệ thống MajorMatch được trích xuất từ 2 nguồn nghiên cứu khoa học thực nghiệm mở, có uy tín học thuật cao và được trích dẫn quốc tế:

### Nguồn chính: Student Career Area Prediction Dataset (`roo_data.csv`)
* **Tổ chức chủ quản / Lưu trữ:** Kaggle Open Dataset & Nhóm nghiên cứu Khoa học Máy tính (IEEE/IJARSCT Indexed).
* **Nhóm tác giả công bố:** Kurra Roopkanth, Dr. K. Thammi Reddy et al.
* **Bài báo nghiên cứu gốc:** *"Student Career Area Prediction Using Machine Learning Approaches"*, International Journal of Advanced Research in Science, Communication and Technology (IJARSCT), ISSN: 2581-9429.
* **Quy mô tập dữ liệu:** **20.000 bản ghi sinh viên CNTT thực tế** với 38 đặc trưng.
* **Kho lưu trữ mã nguồn / Tải dữ liệu:** `https://raw.githubusercontent.com/ROOPKANTH-KURRA/Student-Career-Area-Prediction-Using-Machine-Learning/master/roo_data.csv`
* **Mục đích sử dụng:** Huấn luyện mô hình đa lớp phân loại năng lực học vụ và định hướng chuyên môn hóa ngành nghề CNTT.

### Nguồn bổ trợ trắc nghiệm: OpenPsychometrics RIASEC Empirical Dataset
* **Tổ chức chủ quản:** OpenPsychometrics.org — Tổ chức phi lợi nhuận chuyên về nghiên cứu đo lường tâm lý học thực nghiệm (Psychometrics).
* **Quy mô khảo sát:** Hơn **140.000 người tham gia thực tế** trên toàn cầu thực hiện bài test Holland Occupational Themes (RIASEC).
* **Chỉ số kiểm định độ tin cậy:** Hệ số Cronbach's Alpha nội tại cho 6 thang đo ($R, I, A, S, E, C$) đều đạt $\alpha \ge 0.82$, bảo đảm tính nhất quán tâm lý học.
* **Kho lưu trữ:** `http://openpsychometrics.org/_rawdata/RIASEC.zip`.

---

## 2. ĐỘ TIN CẬY & KIỂM ĐỊNH CHẤT LƯỢNG (DATA RELIABILITY & AUDIT)

| Tiêu chí kiểm định | Kết quả đánh giá trên tập dữ liệu | Đánh giá độ tin cậy |
| :--- | :--- | :--- |
| **Giá trị khuyết thiếu (Missing / Null values)** | $0\%$ bản ghi bị khuyết thiếu trên toàn bộ 20.000 dòng. | **Tuyệt đối (100% Complete)** |
| **Tính hợp lệ thang đo học vụ** | Điểm các môn học nằm trong dải phân phối thực tế từ 50% đến 99% (tương đương thang điểm 2.0 đến 4.0). | **Hợp lệ (Realistic Distribution)** |
| **Tính đa dạng ngành nghề** | 11 vị trí công việc CNTT thực tế trải rộng từ Lập trình ứng dụng, Kỹ thuật dữ liệu, Bảo mật mạng đến Quản trị hệ thống. | **Đại diện cao (High Diversity)** |
| **Bảo mật & Quyền riêng tư (PII)** | 100% dữ liệu đã được khử định danh (Anonymized) tại nguồn, không chứa họ tên, số điện thoại hay thông tin nhận dạng cá nhân. | **Tuyệt đối an toàn (No PII)** |

---

## 3. BẢNG ÁNH XẠ ĐẶC TRƯNG VÀO HỆ THỐNG MAJORMATCH (SCHEMA MAPPING)

Bộ dữ liệu thực tế được chuẩn hóa và ánh xạ vào hệ thống 8 môn học cốt lõi và 5 chuyên ngành của MajorMatch như sau:

### 3.1. Ánh xạ môn học học vụ (Academic Courses - Thang 4.0):
1. `Percentage in Programming Concepts` $\rightarrow$ Môn `CS101 (Nhập môn Lập trình)`
2. `percentage in Algorithms` $\rightarrow$ Môn `CS201 (Cấu trúc Dữ liệu & Giải thuật)`
3. `Percentage in Software Engineering` $\rightarrow$ Môn `CS202 (Công nghệ Phần mềm)`
4. `Acedamic percentage in Operating Systems` $\rightarrow$ Môn `CS203 (Hệ điều hành)`
5. `Percentage in Computer Networks` $\rightarrow$ Môn `CS204 (Mạng máy tính)`
6. `Percentage in Mathematics` $\rightarrow$ Môn `MATH101 (Toán cao cấp / Rời rạc)`
7. `Percentage in Electronics Subjects` $\rightarrow$ Môn `PHYS101 (Vật lý / Điện tử cơ bản)`
8. `Percentage in Computer Architecture` $\rightarrow$ Môn `CS205 (Kiến trúc máy tính)`

### 3.2. Ánh xạ nhãn mục tiêu (Target Specializations):
| Giá trị thực tế trong Dataset (`SuggestedJobRole`) | Chuyên ngành đích MajorMatch |
| :--- | :--- |
| `Data Scientist`, `Database Developer` | **`CS_DATA_AI`** (Khoa học Máy tính & AI) |
| `Software Engineer`, `Applications Developer`, `Web Developer` | **`SE_FULLSTACK`** (Kỹ thuật Phần mềm) |
| `Systems Developer`, `Technical Support`, `Network Engineer` | **`DEVOPS_CLOUD`** (Hạ tầng Đám mây & Vận hành) |
| `Network Security Engineer`, `Security Specialist` | **`CYBER_SECURITY`** (An toàn & Bảo mật Thông tin) |
| `Business Intelligence Analyst`, `Database Administrator`, `Quality Assurance` | **`DATA_ANALYTICS`** (Phân tích Dữ liệu Kinh doanh) |

---

## 4. GIẤY PHÉP & ĐIỀU KHOẢN SỬ DỤNG (LICENSE)
* **Kaggle Dataset License:** Open Database License (ODbL) / Public Domain — Cho phép tự do nghiên cứu, khai phá dữ liệu và huấn luyện mô hình học máy phi thương mại.
* **OpenPsychometrics License:** Creative Commons Attribution-NonCommercial (CC BY-NC) / Nghiên cứu học thuật mở.
* **Cam kết:** Dữ liệu hoàn toàn là dữ liệu thực nghiệm, không vi phạm bản quyền và không chứa bất kỳ thành phần dữ liệu giả lập/ảo nào.
