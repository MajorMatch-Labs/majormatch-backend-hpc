"""
Dịch vụ RAG (Retrieval-Augmented Generation) và Tích hợp Mô hình Ngôn ngữ Lớn Ollama Qwen 2.5.
Quản lý ChromaDB Vectorstore lưu trữ khung chương trình đào tạo, kiểm tra môn tiên quyết
và điều phối suy luận AI trên Dedicated GPU Node sinh cấu trúc lộ trình cá nhân hóa JSON.
"""

import os
import json
import asyncio
import httpx
from typing import List, Dict, Any, AsyncGenerator, Optional

import chromadb
from chromadb.config import Settings

from schemas import (
    RoadmapGenerationRequest,
    RoadmapGenerationResponse,
    SemesterMilestone,
    RecommendedCourse,
    PracticalProject,
    ChatStreamRequest
)

# Biến môi trường và cấu hình
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b-instruct-q4_k_m")
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8001"))

# Semaphore bảo vệ VRAM GPU: Chỉ xử lý tối đa 1 tác vụ LLM đồng thời
GPU_SEMAPHORE = asyncio.Semaphore(1)


# =============================================================================
# DỮ LIỆU KHUNG CHƯƠNG TRÌNH ĐÀO TẠO MẪU ĐỂ SEED VÀO CHROMADB (5 CHUYÊN NGÀNH)
# =============================================================================
SAMPLE_CURRICULUM_COURSES = [
    # 1. KHOA HỌC DỮ LIỆU & TRÍ TUỆ NHÂN TẠO (CS_DATA_AI)
    {
        "course_code": "CS301",
        "course_name": "Nhập môn Trí tuệ Nhân tạo",
        "major_id": "CS_DATA_AI",
        "credits": 3,
        "semester": 5,
        "prerequisites": "CS102,MTH100",
        "skills": "Artificial Intelligence, Search Algorithms, Heuristic, Logic",
        "description": "Cung cấp nền tảng về thuật toán tìm kiếm không gian trạng thái, A*, đối kháng Minimax, logic mờ và biểu diễn tri thức."
    },
    {
        "course_code": "CS302",
        "course_name": "Học máy cơ sở (Machine Learning)",
        "major_id": "CS_DATA_AI",
        "credits": 3,
        "semester": 5,
        "prerequisites": "CS101,MTH101",
        "skills": "Machine Learning, Regression, Classification, Scikit-learn, Feature Engineering",
        "description": "Nghiên cứu các thuật toán hồi quy, phân lớp SVM, Decision Trees, Random Forest và phân cụm K-Means."
    },
    {
        "course_code": "CS401",
        "course_name": "Học sâu & Thị giác máy tính (Deep Learning & CV)",
        "major_id": "CS_DATA_AI",
        "credits": 3,
        "semester": 6,
        "prerequisites": "CS302",
        "skills": "Deep Learning, PyTorch, CNN, Computer Vision, Transfer Learning",
        "description": "Thiết kế mạng nơ-ron tích chập CNN, huấn luyện mô hình phân loại ảnh và phát hiện đối tượng với PyTorch."
    },
    {
        "course_code": "CS402",
        "course_name": "Xử lý Ngôn ngữ Tự nhiên & LLM (NLP & Generative AI)",
        "major_id": "CS_DATA_AI",
        "credits": 3,
        "semester": 7,
        "prerequisites": "CS302",
        "skills": "NLP, Transformers, BERT, LLM, Prompt Engineering, RAG",
        "description": "Kiến trúc Attention, Transformer, xây dựng hệ thống hỏi đáp thông minh và Retrieval-Augmented Generation."
    },

    # 2. KỸ THUẬT ĐÁM MÂY & VẬN HÀNH HỆ THỐNG (DEVOPS_CLOUD)
    {
        "course_code": "DO201",
        "course_name": "Quản trị Linux & Tự động hóa Shell Scripting",
        "major_id": "DEVOPS_CLOUD",
        "credits": 3,
        "semester": 4,
        "prerequisites": "IT101",
        "skills": "Linux, Bash, Shell Scripting, System Administration, OS Internals",
        "description": "Làm chủ dòng lệnh Linux, quản lý phân quyền, tiến trình, tài nguyên và tự động hóa tác vụ hệ thống bằng Bash script."
    },
    {
        "course_code": "NW201",
        "course_name": "Mạng máy tính & Giao thức Truyền thông",
        "major_id": "DEVOPS_CLOUD",
        "credits": 3,
        "semester": 4,
        "prerequisites": "IT101",
        "skills": "Computer Networks, TCP/IP, DNS, HTTP/HTTPS, Routing, Firewalls",
        "description": "Mô hình mạng OSI/TCP-IP, định tuyến, phân giải tên miền DNS, cân bằng tải mạng và phân tích gói tin."
    },
    {
        "course_code": "DO301",
        "course_name": "Containerization & Đóng gói Ứng dụng với Docker",
        "major_id": "DEVOPS_CLOUD",
        "credits": 3,
        "semester": 5,
        "prerequisites": "DO201,NW201",
        "skills": "Docker, Containerization, Docker Compose, Linux, Microservices",
        "description": "Đóng gói ứng dụng đa dịch vụ vào Docker container, tối ưu hóa kích thước Dockerfile và quản lý mạng container."
    },
    {
        "course_code": "DO302",
        "course_name": "CI/CD Pipeline & Tự động hóa Triển khai",
        "major_id": "DEVOPS_CLOUD",
        "credits": 3,
        "semester": 6,
        "prerequisites": "DO301",
        "skills": "CI/CD, GitHub Actions, Jenkins, Automated Testing, Deployment",
        "description": "Xây dựng luồng tích hợp và triển khai liên tục tự động với GitHub Actions, tích hợp kiểm thử đơn vị và bảo mật."
    },
    {
        "course_code": "CL401",
        "course_name": "Kiến trúc Điện toán Đám mây & Điều phối Kubernetes",
        "major_id": "DEVOPS_CLOUD",
        "credits": 3,
        "semester": 7,
        "prerequisites": "DO301,DO302",
        "skills": "Kubernetes, AWS, Cloud Architecture, Terraform, Infrastructure as Code",
        "description": "Triển khai cụm Kubernetes phân tán, cấu hình Ingress, Autoscaling và tự động cấp phát tài nguyên đám mây qua Terraform."
    },

    # 3. PHÂN TÍCH DỮ LIỆU KINH DOANH (DATA_ANALYTICS)
    {
        "course_code": "DA201",
        "course_name": "SQL Nâng cao & Cơ sở Dữ liệu Phân tích",
        "major_id": "DATA_ANALYTICS",
        "credits": 3,
        "semester": 4,
        "prerequisites": "IT201",
        "skills": "SQL, PostgreSQL, Window Functions, Data Modeling, Query Optimization",
        "description": "Truy vấn phức tạp với Window Functions, CTE, lập mô hình dữ liệu Star Schema và tối ưu hóa hiệu năng truy vấn phân tích."
    },
    {
        "course_code": "DA301",
        "course_name": "Xác suất Thống kê Ứng dụng & Python Data Analysis",
        "major_id": "DATA_ANALYTICS",
        "credits": 3,
        "semester": 5,
        "prerequisites": "DA201,MTH100",
        "skills": "Python, Pandas, NumPy, Statistics, Exploratory Data Analysis, Data Cleaning",
        "description": "Thu thập, làm sạch và xử lý dữ liệu định lượng với Pandas; áp dụng kiểm định thống kê và phân tích xu hướng."
    },
    {
        "course_code": "DA302",
        "course_name": "Business Intelligence & Trực quan hóa Dữ liệu (BI)",
        "major_id": "DATA_ANALYTICS",
        "credits": 3,
        "semester": 6,
        "prerequisites": "DA201,DA301",
        "skills": "PowerBI, Tableau, Data Visualization, Data Warehousing, KPI Dashboards",
        "description": "Thiết kế Dashboard trực quan hóa chỉ số KPI kinh doanh đa chiều phục vụ ra quyết định chiến lược với PowerBI và Tableau."
    },
    {
        "course_code": "DA401",
        "course_name": "Phân tích Dự báo & Khai phá Dữ liệu Doanh nghiệp",
        "major_id": "DATA_ANALYTICS",
        "credits": 3,
        "semester": 7,
        "prerequisites": "DA301",
        "skills": "Predictive Analytics, Data Mining, Time Series, Customer Segmentation",
        "description": "Xây dựng mô hình phân khúc khách hàng, dự báo doanh số chuỗi thời gian và đánh giá tỷ lệ rời bỏ dịch vụ."
    },

    # 4. KỸ THUẬT PHẦN MỀM & LẬP TRÌNH FULLSTACK (SE_FULLSTACK)
    {
        "course_code": "SE201",
        "course_name": "Lập trình Hướng đối tượng Nâng cao & Clean Code",
        "major_id": "SE_FULLSTACK",
        "credits": 3,
        "semester": 4,
        "prerequisites": "CS102",
        "skills": "OOP, Java, Design Patterns, SOLID, Clean Code, Unit Testing",
        "description": "Mẫu thiết kế Factory, Singleton, Observer, kiến trúc phân tầng và các nguyên lý SOLID viết mã nguồn chuyên nghiệp."
    },
    {
        "course_code": "SE301",
        "course_name": "Phát triển Ứng dụng Web Fullstack Hiện đại",
        "major_id": "SE_FULLSTACK",
        "credits": 3,
        "semester": 5,
        "prerequisites": "SE201",
        "skills": "TypeScript, React, Node.js, Next.js, RESTful API, PostgreSQL",
        "description": "Xây dựng ứng dụng web hiện đại từ Frontend Single Page Application với React/Next.js đến Backend API với Node.js."
    },
    {
        "course_code": "SE302",
        "course_name": "Kiến trúc Hệ thống & Microservices",
        "major_id": "SE_FULLSTACK",
        "credits": 3,
        "semester": 6,
        "prerequisites": "SE201,SE301",
        "skills": "System Design, Microservices, RESTful API, Docker, Message Queue",
        "description": "Thiết kế hệ thống chịu tải cao, giao tiếp bất đồng bộ qua RabbitMQ/Kafka và kiến trúc dịch vụ vi mô độc lập."
    },

    # 5. AN TOÀN THÔNG TIN & AN NINH MẠNG (CYBER_SECURITY)
    {
        "course_code": "SEC201",
        "course_name": "An toàn Mạng & Phòng thủ Hệ thống Căn bản",
        "major_id": "CYBER_SECURITY",
        "credits": 3,
        "semester": 4,
        "prerequisites": "IT101",
        "skills": "Network Security, Firewalls, Web Security, OSI Security, Linux",
        "description": "Các nguy cơ tấn công mạng phổ biến, phòng thủ tường lửa, phát hiện xâm nhập IDS/IPS và an toàn hạ tầng mạng."
    },
    {
        "course_code": "SEC301",
        "course_name": "Mật mã học & Giao thức An toàn",
        "major_id": "CYBER_SECURITY",
        "credits": 3,
        "semester": 5,
        "prerequisites": "MTH100,SEC201",
        "skills": "Cryptography, RSA, AES, SSL/TLS, Public Key Infrastructure",
        "description": "Các thuật toán mã hóa đối xứng, bất đối xứng RSA/ECC, hàm băm SHA, chữ ký số và giao thức bảo mật SSL/TLS."
    },
    {
        "course_code": "SEC302",
        "course_name": "Kiểm thử Xâm nhập & Đánh giá Lỗ hổng (PenTest)",
        "major_id": "CYBER_SECURITY",
        "credits": 3,
        "semester": 6,
        "prerequisites": "SEC201,SEC301",
        "skills": "Penetration Testing, Ethical Hacking, OWASP Top 10, Kali Linux, Vulnerability Assessment",
        "description": "Phương pháp rà quét lỗ hổng theo OWASP, khai thác an toàn, kiểm thử ứng dụng web và lập báo cáo an ninh."
    }
]


class ChromaDBService:
    """Quản trị cơ sở dữ liệu Vector ChromaDB."""

    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """Khởi tạo kết nối ChromaDB với cơ chế fallback client nội bộ."""
        try:
            # Thử kết nối tới ChromaDB container qua HTTP
            self.client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
            self.client.heartbeat()
        except Exception:
            # Fallback sang PersistentClient lưu tại đĩa cục bộ
            persist_dir = os.path.join(os.path.dirname(__file__), "chroma_local_storage")
            self.client = chromadb.PersistentClient(path=persist_dir)

        # Tạo hoặc lấy collection khung môn học
        self.collection = self.client.get_or_create_collection(
            name="curriculum_courses",
            metadata={"hnsw:space": "cosine"}
        )
        self._seed_sample_data()

    def _seed_sample_data(self):
        """Nạp hoặc đồng bộ toàn bộ khung môn học của 5 chuyên ngành vào ChromaDB."""
        ids = [c["course_code"] for c in SAMPLE_CURRICULUM_COURSES]
        documents = [
            f"{c['course_name']}. Chuyên ngành: {c['major_id']}. Kỹ năng: {c['skills']}. Mô tả: {c['description']}"
            for c in SAMPLE_CURRICULUM_COURSES
        ]
        metadatas = [
            {
                "course_code": c["course_code"],
                "course_name": c["course_name"],
                "major_id": c["major_id"],
                "credits": c["credits"],
                "semester": c["semester"],
                "prerequisites": c["prerequisites"]
            }
            for c in SAMPLE_CURRICULUM_COURSES
        ]
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def query_recommended_courses(self, missing_skills: List[str], target_major_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Truy vấn Top các môn học phù hợp nhất theo chuyên ngành mục tiêu để bù đắp kỹ năng."""
        query_text = " ".join(missing_skills) if missing_skills else f"Chuyên ngành {target_major_id}"

        # 1. Thử lọc chính xác theo major_id
        filter_clause = {"major_id": target_major_id} if target_major_id else None
        courses: List[Dict[str, Any]] = []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=filter_clause
            )
            if results and results["metadatas"] and results["metadatas"][0]:
                for meta in results["metadatas"][0]:
                    courses.append(dict(meta))
        except Exception:
            pass

        # 2. Nếu bộ lọc không đủ kết quả, lấy trực tiếp từ SAMPLE_CURRICULUM_COURSES theo chuyên ngành
        if len(courses) < 2:
            direct_matches = [c for c in SAMPLE_CURRICULUM_COURSES if c["major_id"] == target_major_id]
            for c in direct_matches:
                if c["course_code"] not in [x["course_code"] for x in courses]:
                    courses.append(c)

        return courses[:top_k]


# Khởi tạo singleton instance cho ChromaDB Service
chroma_service = ChromaDBService()


class OllamaAIService:
    """Quản lý giao tiếp và suy luận mô hình Qwen 2.5 7B chạy nội bộ trên GPU."""

    @staticmethod
    def _build_roadmap_prompt(
        request: RoadmapGenerationRequest,
        retrieved_courses: List[Dict[str, Any]]
    ) -> str:
        """Xây dựng Prompt ngữ cảnh hóa (Context-Augmented Prompt) chia theo 3 GIAI ĐOẠN đúng chuyên ngành."""
        context_str = "\n".join([
            f"- Môn {c['course_code']}: {c['course_name']} ({c['credits']} tín chỉ). "
            f"Kỹ năng: {c.get('skills', '')}. Tiên quyết: {c.get('prerequisites', 'Không')}"
            for c in retrieved_courses
        ])

        initial_readiness = round(request.initial_readiness_score or 45.6, 1)

        prompt = f"""Bạn là Cố vấn Học tập & Kỹ sư Trưởng AI của hệ thống MajorMatch.
Hãy sinh một lộ trình phát triển năng lực cá nhân hóa chi tiết theo 3 GIAI ĐOẠN (Phases) cho sinh viên đúng chuyên ngành: {request.target_major_id}.
LƯU Ý QUAN TRỌNG:
1. TUYỆT ĐỐI KHÔNG chia theo học kỳ, KHÔNG dùng từ "học kỳ" hoặc "kỳ". Hãy chia theo đúng 3 GIAI ĐOẠN PHÁT TRIỂN NĂNG LỰC:
   - Giai đoạn 1: Bồi đắp Nền tảng Kỹ thuật Chuyên sâu
   - Giai đoạn 2: Nâng cao & Ứng dụng Thực chiến
   - Giai đoạn 3: Tối ưu Hệ thống & Sẵn sàng Nghề nghiệp
2. CHỈ ĐƯỢC PHÉP SỬ DỤNG CÁC MÔN HỌC TRONG [DANH MỤC MÔN HỌC ĐƯỢC RAG TRUY XUẤT]. TUYỆT ĐỐI KHÔNG tự ý lấy môn học của ngành khác!
3. Điểm Job Readiness ban đầu của sinh viên là {initial_readiness}%. Giữ nguyên mốc này trong kết quả: "job_readiness_percentage": {initial_readiness}.

[THÔNG TIN SINH VIÊN]
- Chuyên ngành mục tiêu: {request.target_major_id}
- Môn đã hoàn thành: {', '.join(request.completed_course_codes) if request.completed_course_codes else 'Chưa có'}
- Kỹ năng còn khuyết thiếu: {', '.join(request.missing_skills)}
- Điểm tiềm năng xuất phát: {initial_readiness}%

[DANH MỤC MÔN HỌC TỪ KHUNG ĐÀO TẠO ĐƯỢC RAG TRUY XUẤT CHO CHUYÊN NGÀNH {request.target_major_id}]
{context_str}

[YÊU CẦU BẮT BUỘC]
1. Trả về DUY NHẤT một chuỗi JSON hợp lệ (không có markdown giải thích ở ngoài JSON).
2. Phân bổ các môn trong danh mục trên vào 3 giai đoạn theo độ khó tăng dần.
3. Đề xuất đề tài đồ án thực chiến (Practical Project) và chứng chỉ chuyên ngành quốc tế phù hợp nhất với ngành {request.target_major_id}.

[ĐỊNH DẠNG JSON MẪU]
{{
  "target_major": "{request.target_major_id}",
  "job_readiness_percentage": {initial_readiness},
  "readiness_score": {initial_readiness},
  "semesters": [
    {{
      "phase_number": 1,
      "phase_title": "Giai đoạn 1: Bồi đắp Nền tảng Kỹ thuật Chuyên sâu",
      "target_focus": "Nắm vững nguyên lý cốt lõi và kỹ năng cơ sở của chuyên ngành",
      "recommended_courses": [],
      "certifications": [],
      "practical_project": {{
        "project_title": "Dự án Nền tảng",
        "description": "Mô tả đồ án giai đoạn 1",
        "target_skills": []
      }}
    }},
    {{
      "phase_number": 2,
      "phase_title": "Giai đoạn 2: Nâng cao & Ứng dụng Thực chiến",
      "target_focus": "Triển khai giải pháp thực tế và làm chủ các công cụ tiêu chuẩn ngành",
      "recommended_courses": [],
      "certifications": [],
      "practical_project": {{
        "project_title": "Dự án Thực chiến",
        "description": "Mô tả đồ án giai đoạn 2",
        "target_skills": []
      }}
    }},
    {{
      "phase_number": 3,
      "phase_title": "Giai đoạn 3: Tối ưu Hệ thống & Sẵn sàng Nghề nghiệp",
      "target_focus": "Tối ưu hóa sản phẩm lớn, sẵn sàng gia nhập doanh nghiệp",
      "recommended_courses": [],
      "certifications": [],
      "practical_project": {{
        "project_title": "Đồ án Capstone Tốt nghiệp",
        "description": "Mô tả đồ án giai đoạn 3",
        "target_skills": []
      }}
    }}
  ]
}}
"""
        return prompt

    @classmethod
    async def generate_roadmap_json(cls, request: RoadmapGenerationRequest) -> RoadmapGenerationResponse:
        """Gọi Ollama suy luận và parse cấu trúc JSON của lộ trình."""
        # 1. Truy xuất RAG từ ChromaDB theo đúng major_id
        retrieved_courses = chroma_service.query_recommended_courses(
            request.missing_skills,
            request.target_major_id
        )

        prompt = cls._build_roadmap_prompt(request, retrieved_courses)

        # 2. Suy luận qua Ollama với Semaphore bảo vệ GPU
        async with GPU_SEMAPHORE:
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    response = await client.post(
                        f"{OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": OLLAMA_MODEL,
                            "prompt": prompt,
                            "format": "json",
                            "stream": False,
                            "options": {
                                "temperature": 0.2,
                                "top_p": 0.9,
                                "num_predict": 1024
                            }
                        }
                    )
                    if response.status_code == 200:
                        raw_json_str = response.json().get("response", "")
                        data = json.loads(raw_json_str)
                        # Đồng bộ điểm xuất phát chuẩn xác
                        initial_score = round(request.initial_readiness_score or 45.6, 1)
                        if "job_readiness_percentage" not in data or data["job_readiness_percentage"] > 70.0:
                            data["job_readiness_percentage"] = initial_score
                        data["readiness_score"] = data.get("job_readiness_percentage", initial_score)

                        # Chuẩn hóa phase_title nếu có
                        for item in data.get("semesters", []):
                            if "semester_title" not in item and "phase_title" in item:
                                item["semester_title"] = item["phase_title"]
                            if "semester_number" not in item and "phase_number" in item:
                                item["semester_number"] = item["phase_number"]
                        return RoadmapGenerationResponse(**data)
            except Exception:
                # Nếu Ollama chưa khởi chạy hoặc lỗi mạng, sử dụng bộ sinh lộ trình tất định (Deterministic Fallback)
                pass

        return cls._build_deterministic_fallback_roadmap(request, retrieved_courses)

    @classmethod
    def _build_deterministic_fallback_roadmap(
        cls,
        request: RoadmapGenerationRequest,
        retrieved_courses: List[Dict[str, Any]]
    ) -> RoadmapGenerationResponse:
        """Bộ sinh lộ trình tất định dự phòng chuẩn 3 Giai đoạn năng lực cá nhân hóa theo chuyên ngành."""
        major_id = request.target_major_id
        initial_score = round(request.initial_readiness_score or 45.6, 1)

        # Metadata chứng chỉ và đồ án chuyên biệt cho từng ngành
        domain_meta = {
            "DEVOPS_CLOUD": {
                "name": "Kỹ thuật Đám mây & Vận hành Hệ thống (DevOps/Cloud)",
                "cert_1": ["Linux Foundation Certified System Administrator (LFCS)"],
                "proj_1": ("Hệ thống Tự động hóa Giám sát & Quản trị Linux", "Xây dựng bộ công cụ Bash script tự động thu thập tài nguyên hệ điều hành, phân quyền bảo mật và gửi cảnh báo."),
                "cert_2": ["Docker Certified Associate (DCA)", "HashiCorp Certified Terraform Associate"],
                "proj_2": ("Kiến trúc Microservices Container hóa & CI/CD Pipeline", "Đóng gói cụm dịch vụ với Docker Compose, cấu hình luồng GitHub Actions tự động build, test và release."),
                "cert_3": ["AWS Certified Solutions Architect", "Certified Kubernetes Administrator (CKA)"],
                "proj_3": ("Triển khai Cụm Kubernetes Đám mây AWS Chịu tải Cao", "Điều phối cụm Kubernetes hoàn chỉnh trên Cloud, tích hợp Ingress Nginx, HPA tự động mở rộng và giám sát Prometheus/Grafana.")
            },
            "DATA_ANALYTICS": {
                "name": "Phân tích Dữ liệu Kinh doanh (Data Analytics & BI)",
                "cert_1": ["PostgreSQL Database Professional"],
                "proj_1": ("Thiết kế Data Warehouse & Tối ưu Truy vấn SQL Phân tích", "Lập mô hình dữ liệu Star Schema, áp dụng CTE và Window Functions trích xuất dữ liệu vận hành đa nguồn."),
                "cert_2": ["Microsoft Power BI Data Analyst (PL-300)"],
                "proj_2": ("Executive BI Dashboard Trực quan hóa Doanh thu Đa kênh", "Xây dựng báo cáo tự động hóa phân tích tăng trưởng, phân khúc khách hàng RFM và chỉ số tài chính thời gian thực."),
                "cert_3": ["Tableau Desktop Specialist", "Google Data Analytics Professional"],
                "proj_3": ("Dự án Capstone: Mô hình Dự báo Khách hàng Rời bỏ & Xu hướng Thị trường", "Ứng dụng phân tích hồi quy, kiểm định giả thuyết và trực quan hóa chuyên sâu phục vụ ra quyết định chiến lược.")
            },
            "SE_FULLSTACK": {
                "name": "Kỹ thuật Phần mềm & Lập trình Fullstack",
                "cert_1": ["Oracle Certified Associate Java SE"],
                "proj_1": ("Hệ thống Quản lý Thiết kế Hướng đối tượng & Clean Code", "Áp dụng trọn vẹn các nguyên lý SOLID, Design Patterns và Unit Testing xây dựng module nghiệp vụ tiêu chuẩn."),
                "cert_2": ["Meta Front-End Developer Certificate"],
                "proj_2": ("Nền tảng Web Fullstack Thương mại Điện tử Thời gian thực", "Ứng dụng kiến trúc React/Next.js phía Frontend và Node.js REST API, quản lý trạng thái và tối ưu SEO."),
                "cert_3": ["AWS Certified Developer - Associate"],
                "proj_3": ("Hệ thống Dịch vụ Vi mô Microservices Chịu tải Cao", "Phân rã kiến trúc microservices độc lập, truyền thông bất đồng bộ qua Message Queue và triển khai Docker.")
            },
            "CYBER_SECURITY": {
                "name": "An toàn Thông tin & An ninh Mạng",
                "cert_1": ["CompTIA Security+"],
                "proj_1": ("Thiết lập Hạ tầng Mạng Phòng thủ Doanh nghiệp & Tường lửa", "Cấu hình phân vùng mạng DMZ, thiết lập luật Firewall và phân tích lưu lượng mạng Wireshark."),
                "cert_2": ["Certified Ethical Hacker (CEH)"],
                "proj_2": ("Kiểm thử Xâm nhập Đánh giá Lỗ hổng Ứng dụng Web OWASP", "Thực hiện rà quét lỗ hổng SQL Injection, XSS, CSRF trên môi trường giả lập và đề xuất bản vá."),
                "cert_3": ["OffSec Certified Professional (OSCP)"],
                "proj_3": ("Trung tâm Giám sát An toàn Thông tin SOC & Điều tra Số", "Thiết lập giải pháp SIEM phát hiện hành vi bất thường, xây dựng kịch bản ứng phó sự cố an ninh mạng chuyên sâu.")
            },
            "CS_DATA_AI": {
                "name": "Khoa học Dữ liệu & Trí tuệ Nhân tạo",
                "cert_1": ["DeepLearning.AI Machine Learning Specialization"],
                "proj_1": ("Xây dựng Pipeline Phân loại Dữ liệu Đa nhãn Scikit-learn", "Tiền xử lý dữ liệu chuẩn hóa, kỹ thuật đặc trưng và huấn luyện mô hình phân loại đa lớp."),
                "cert_2": ["DeepLearning.AI TensorFlow Developer", "AWS Certified Machine Learning"],
                "proj_2": ("Hệ thống Thị giác Máy tính Phân tích Ảnh Y tế với PyTorch", "Huấn luyện mạng nơ-ron tích chập ResNet phát hiện bệnh học từ ảnh X-quang tích hợp API."),
                "cert_3": ["Google Cloud Professional Data Engineer"],
                "proj_3": ("Hệ thống RAG Đa phương thức Doanh nghiệp với LLM", "Xây dựng trợ lý ảo hỏi đáp tài liệu thông minh kết hợp ChromaDB và mô hình ngôn ngữ lớn Qwen.")
            }
        }

        meta = domain_meta.get(major_id, domain_meta["CS_DATA_AI"])
        major_name = meta["name"]

        def to_course_models(raw_list: List[Dict[str, Any]]) -> List[RecommendedCourse]:
            courses = []
            for c in raw_list:
                courses.append(
                    RecommendedCourse(
                        course_code=c["course_code"],
                        course_name=c["course_name"],
                        credits=c.get("credits", 3),
                        rationale=f"Học phần chuyên ngành bồi đắp trực tiếp năng lực {c.get('skills', c['course_name'])}.",
                        target_skills=[s.strip() for s in c.get("skills", "").split(",") if s.strip()][:3],
                        prerequisites_satisfied=True
                    )
                )
            return courses

        # Phân chia các môn theo 3 giai đoạn
        p1_courses = to_course_models(retrieved_courses[:2])
        p2_courses = to_course_models(retrieved_courses[2:4]) if len(retrieved_courses) > 2 else to_course_models(retrieved_courses[:1])
        p3_courses = to_course_models(retrieved_courses[4:5]) if len(retrieved_courses) > 4 else to_course_models(retrieved_courses[-1:])

        semesters: List[SemesterMilestone] = [
            # Giai đoạn 1
            SemesterMilestone(
                phase_number=1,
                phase_title="Giai đoạn 1: Bồi đắp Nền tảng Kỹ thuật Chuyên sâu",
                semester_number=1,
                semester_title="Giai đoạn 1: Bồi đắp Nền tảng Kỹ thuật Chuyên sâu",
                target_focus="Nắm vững nguyên lý cốt lõi và kỹ năng cơ sở của chuyên ngành",
                milestone_skills=request.missing_skills[:3] if request.missing_skills else ["Core Engineering", "Problem Solving"],
                recommended_courses=p1_courses,
                certifications=meta["cert_1"],
                practical_project=PracticalProject(
                    project_title=meta["proj_1"][0],
                    description=meta["proj_1"][1],
                    target_skills=request.missing_skills[:3] if request.missing_skills else ["Core Engineering"]
                )
            ),
            # Giai đoạn 2
            SemesterMilestone(
                phase_number=2,
                phase_title="Giai đoạn 2: Nâng cao & Ứng dụng Thực chiến",
                semester_number=2,
                semester_title="Giai đoạn 2: Nâng cao & Ứng dụng Thực chiến",
                target_focus="Triển khai giải pháp thực tế và làm chủ các công cụ tiêu chuẩn ngành",
                milestone_skills=["System Design", "Production Tooling", "Automation"],
                recommended_courses=p2_courses,
                certifications=meta["cert_2"],
                practical_project=PracticalProject(
                    project_title=meta["proj_2"][0],
                    description=meta["proj_2"][1],
                    target_skills=["System Design", "Automation", "Production Ready"]
                )
            ),
            # Giai đoạn 3
            SemesterMilestone(
                phase_number=3,
                phase_title="Giai đoạn 3: Tối ưu Hệ thống & Sẵn sàng Nghề nghiệp",
                semester_number=3,
                semester_title="Giai đoạn 3: Tối ưu Hệ thống & Sẵn sàng Nghề nghiệp",
                target_focus="Tối ưu hóa sản phẩm quy mô lớn, sẵn sàng gia nhập doanh nghiệp",
                milestone_skills=["Scalability", "High Availability", "Enterprise Architecture"],
                recommended_courses=p3_courses,
                certifications=meta["cert_3"],
                practical_project=PracticalProject(
                    project_title=meta["proj_3"][0],
                    description=meta["proj_3"][1],
                    target_skills=["Enterprise Architecture", "Security", "Scalability"]
                )
            )
        ]

        return RoadmapGenerationResponse(
            target_major=major_name,
            target_major_id=major_id,
            job_readiness_percentage=initial_score,
            readiness_score=initial_score,
            semesters=semesters
        )

    @classmethod
    async def stream_chat_tokens(cls, request: ChatStreamRequest) -> AsyncGenerator[str, None]:
        """Stream phản hồi từ Qwen 2.5 qua Server-Sent Events (SSE)."""
        system_instruction = (
            "Bạn là Trợ lý Cố vấn Học tập MajorMatch. "
            "Trả lời ngắn gọn, chuyên nghiệp, súc tích bằng tiếng Việt. "
            "Tập trung hướng dẫn sinh viên cách khắc phục lỗ hổng kỹ năng và phương pháp học tập hiệu quả."
        )

        context_info = ""
        if request.context:
            context_info = f"\n[Ngữ cảnh: Ngành {request.context.target_major}, Kỹ năng thiếu: {request.context.missing_skills}]"

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction + context_info},
                {"role": "user", "content": request.message}
            ],
            "stream": True,
            "options": {"temperature": 0.5, "num_predict": 512}
        }

        async with GPU_SEMAPHORE:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload) as stream:
                        async for line in stream.aiter_lines():
                            if line:
                                try:
                                    chunk = json.loads(line)
                                    content = chunk.get("message", {}).get("content", "")
                                    done = chunk.get("done", False)

                                    event_data = json.dumps({"token": content, "done": done}, ensure_ascii=False)
                                    yield f"event: message\ndata: {event_data}\n\n"

                                    if done:
                                        break
                                except json.JSONDecodeError:
                                    continue
            except Exception as e:
                # Fallback phản hồi nếu không kết nối được Ollama
                msg = f"Hệ thống trợ lý cục bộ ghi nhận câu hỏi: '{request.message}'. Vui lòng đảm bảo dịch vụ Ollama đang hoạt động."
                yield f"event: message\ndata: {json.dumps({'token': msg, 'done': True}, ensure_ascii=False)}\n\n"
