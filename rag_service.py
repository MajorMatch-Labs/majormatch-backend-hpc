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
# DỮ LIỆU KHUNG CHƯƠNG TRÌNH ĐÀO TẠO MẪU ĐỂ SEED VÀO CHROMADB
# =============================================================================
SAMPLE_CURRICULUM_COURSES = [
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
    {
        "course_code": "SE201",
        "course_name": "Lập trình Hướng đối tượng Nâng cao",
        "major_id": "SE_FULLSTACK",
        "credits": 3,
        "semester": 4,
        "prerequisites": "CS102",
        "skills": "OOP, Java, Design Patterns, SOLID, Clean Code",
        "description": "Mẫu thiết kế Factory, Singleton, Observer và các nguyên lý thiết kế phần mềm linh hoạt."
    },
    {
        "course_code": "SE302",
        "course_name": "Kiến trúc Hệ thống & Microservices",
        "major_id": "SE_FULLSTACK",
        "credits": 3,
        "semester": 6,
        "prerequisites": "SE201,IT201",
        "skills": "System Design, Microservices, RESTful API, Docker, Message Queue",
        "description": "Thiết kế hệ thống chịu tải cao, giao tiếp bất đồng bộ qua RabbitMQ/Kafka và kiến trúc dịch vụ vi mô."
    },
    {
        "course_code": "DO301",
        "course_name": "Hạ tầng Đám mây & CI/CD Pipeline",
        "major_id": "DEVOPS_CLOUD",
        "credits": 3,
        "semester": 6,
        "prerequisites": "IT202,IT203",
        "skills": "Docker, Kubernetes, CI/CD, GitHub Actions, Linux, AWS",
        "description": "Đóng gói ứng dụng container hóa, điều phối cụm Kubernetes và tự động hóa triển khai phần mềm liên tục."
    },
    {
        "course_code": "SEC301",
        "course_name": "Mật mã học & An toàn Mạng",
        "major_id": "CYBER_SECURITY",
        "credits": 3,
        "semester": 6,
        "prerequisites": "IT202,MTH100",
        "skills": "Cryptography, Network Security, SSL/TLS, Penetration Testing",
        "description": "Các thuật toán mã hóa đối xứng, bất đối xứng RSA/ECC và phương pháp phòng thủ an ninh mạng."
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
        """Nạp dữ liệu khung chương trình mẫu nếu collection còn rỗng."""
        count = self.collection.count()
        if count == 0:
            ids = [c["course_code"] for c in SAMPLE_CURRICULUM_COURSES]
            documents = [
                f"{c['course_name']}. Kỹ năng: {c['skills']}. Mô tả: {c['description']}"
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
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

    def query_recommended_courses(self, missing_skills: List[str], target_major_id: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Truy vấn Top các môn học phù hợp nhất để bù đắp kỹ năng còn thiếu."""
        if not missing_skills:
            missing_skills = ["Software Engineering", "Algorithms", "System Architecture"]

        query_text = " ".join(missing_skills)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, self.collection.count())
        )

        courses: List[Dict[str, Any]] = []
        if results and results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                courses.append(dict(meta))
        return courses


# Khởi tạo singleton instance cho ChromaDB Service
chroma_service = ChromaDBService()


class OllamaAIService:
    """Quản lý giao tiếp và suy luận mô hình Qwen 2.5 7B chạy nội bộ trên GPU."""

    @staticmethod
    def _build_roadmap_prompt(
        request: RoadmapGenerationRequest,
        retrieved_courses: List[Dict[str, Any]]
    ) -> str:
        """Xây dựng Prompt ngữ cảnh hóa (Context-Augmented Prompt) chia theo 3 GIAI ĐOẠN."""
        context_str = "\n".join([
            f"- Môn {c['course_code']}: {c['course_name']} ({c['credits']} tín chỉ). "
            f"Tiên quyết: {c.get('prerequisites', 'Không')}"
            for c in retrieved_courses
        ])

        prompt = f"""Bạn là Cố vấn Học tập & Kỹ sư Trưởng AI của hệ thống MajorMatch.
Hãy sinh một lộ trình phát triển năng lực cá nhân hóa chi tiết theo 3 GIAI ĐOẠN (Phases) cho sinh viên.
LƯU Ý QUAN TRỌNG: TUYỆT ĐỐI KHÔNG chia theo học kỳ, KHÔNG dùng từ "học kỳ" hoặc "kỳ". Hãy chia theo các GIAI ĐOẠN PHÁT TRIỂN NĂNG LỰC:
- Giai đoạn 1: Bồi đắp Nền tảng Kỹ thuật Chuyên sâu
- Giai đoạn 2: Nâng cao & Ứng dụng Thực chiến
- Giai đoạn 3: Tối ưu Hệ thống & Sẵn sàng Nghề nghiệp

[THÔNG TIN SINH VIÊN]
- Chuyên ngành mục tiêu: {request.target_major_id}
- Môn đã hoàn thành: {', '.join(request.completed_course_codes) if request.completed_course_codes else 'Chưa có'}
- Kỹ năng còn khuyết thiếu: {', '.join(request.missing_skills)}

[DANH MỤC MÔN HỌC TỪ KHUNG ĐÀO TẠO ĐƯỢC RAG TRUY XUẤT]
{context_str}

[YÊU CẦU BẮT BUỘC]
1. Trả về DUY NHẤT một chuỗi JSON hợp lệ (không có markdown giải thích ở ngoài JSON).
2. Kiểm tra chuỗi môn học tiên quyết: Không xếp môn nâng cao nếu môn tiên quyết chưa đạt.
3. Sinh đúng 3 giai đoạn (Giai đoạn 1, Giai đoạn 2, Giai đoạn 3).
4. Đính kèm chứng chỉ chuyên ngành và đề tài đồ án thực chiến cho mỗi giai đoạn.

[ĐỊNH DẠNG JSON MẪU]
{{
  "target_major": "{request.target_major_id}",
  "job_readiness_percentage": 72.5,
  "semesters": [
    {{
      "phase_number": 1,
      "phase_title": "Giai đoạn 1: Bồi đắp Nền tảng Kỹ thuật Chuyên sâu",
      "target_focus": "Nắm vững thuật toán, toán học và cấu trúc dữ liệu nền tảng",
      "recommended_courses": [
        {{
          "course_code": "CS301",
          "course_name": "Nhập môn Trí tuệ Nhân tạo",
          "credits": 3,
          "rationale": "Cung cấp nền tảng thuật toán tìm kiếm và logic mờ, điều kiện bắt buộc trước khi học Deep Learning.",
          "prerequisites_satisfied": true
        }}
      ],
      "certifications": ["DeepLearning.AI TensorFlow Developer"],
      "practical_project": {{
        "project_title": "Xây dựng Pipeline Phân loại Dữ liệu Đa nhãn",
        "description": "Ứng dụng Scikit-learn và FastAPI triển khai mô hình phân loại dữ liệu học tập sinh viên.",
        "target_skills": ["Machine Learning", "FastAPI", "Scikit-learn"]
      }}
    }},
    {{
      "phase_number": 2,
      "phase_title": "Giai đoạn 2: Nâng cao & Ứng dụng Thực chiến",
      "target_focus": "Phát triển các mô hình học sâu và hệ thống phân tán",
      "recommended_courses": [
        {{
          "course_code": "CS401",
          "course_name": "Học sâu & Thị giác máy tính",
          "credits": 3,
          "rationale": "Huấn luyện mạng nơ-ron sâu phục vụ giải quyết bài toán thị giác máy tính.",
          "prerequisites_satisfied": true
        }}
      ],
      "certifications": ["AWS Certified Machine Learning - Specialty"],
      "practical_project": {{
        "project_title": "Xây dựng Hệ thống Thị giác Máy tính Phân tích Ảnh Y tế",
        "description": "Mô hình ResNet và PyTorch phát hiện bất thường từ ảnh X-quang.",
        "target_skills": ["Deep Learning", "PyTorch", "Computer Vision"]
      }}
    }},
    {{
      "phase_number": 3,
      "phase_title": "Giai đoạn 3: Tối ưu Hệ thống & Sẵn sàng Nghề nghiệp",
      "target_focus": "Hoàn thiện sản phẩm lớn, tối ưu hóa suy luận và tích hợp Cloud",
      "recommended_courses": [
        {{
          "course_code": "CS402",
          "course_name": "Xử lý Ngôn ngữ Tự nhiên & LLM",
          "credits": 3,
          "rationale": "Làm chủ kiến trúc Transformer và xây dựng giải pháp Generative AI thực tiễn.",
          "prerequisites_satisfied": true
        }}
      ],
      "certifications": ["Google Cloud Professional Data Engineer"],
      "practical_project": {{
        "project_title": "Hệ thống RAG Đa phương thức Doanh nghiệp",
        "description": "Xây dựng trợ lý ảo thông minh kết hợp ChromaDB và mô hình ngôn ngữ lớn.",
        "target_skills": ["LLM", "RAG", "ChromaDB", "System Architecture"]
      }}
    }}
  ]
}}
"""
        return prompt

    @classmethod
    async def generate_roadmap_json(cls, request: RoadmapGenerationRequest) -> RoadmapGenerationResponse:
        """Gọi Ollama suy luận và parse cấu trúc JSON của lộ trình."""
        # 1. Truy xuất RAG từ ChromaDB
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
        """Bộ sinh lộ trình tất định dự phòng chuẩn 3 Giai đoạn năng lực."""
        semesters: List[SemesterMilestone] = []

        rec_courses_1: List[RecommendedCourse] = []
        for c in retrieved_courses[:2]:
            rec_courses_1.append(
                RecommendedCourse(
                    course_code=c["course_code"],
                    course_name=c["course_name"],
                    credits=c.get("credits", 3),
                    rationale=f"Môn học cốt lõi cung cấp kiến thức nền tảng về {c['course_name']}.",
                    target_skills=[c["course_name"], "Problem Solving"],
                    prerequisites_satisfied=True
                )
            )

        # Giai đoạn 1
        semesters.append(
            SemesterMilestone(
                phase_number=1,
                phase_title="Giai đoạn 1: Bồi đắp Nền tảng Kỹ thuật Chuyên sâu",
                semester_number=1,
                semester_title="Giai đoạn 1: Bồi đắp Nền tảng Kỹ thuật Chuyên sâu",
                target_focus="Nắm vững cấu trúc giải thuật, xác suất thống kê và nền tảng cốt lõi",
                milestone_skills=request.missing_skills[:4] if request.missing_skills else ["Core Algorithms", "System Architecture"],
                recommended_courses=rec_courses_1,
                certifications=["Coursera Deep Learning Specialization", "AWS Certified Cloud Practitioner"],
                practical_project=PracticalProject(
                    project_title=f"Dự án Nền tảng Ứng dụng {request.target_major_id}",
                    description="Xây dựng sản phẩm hoàn chỉnh ứng dụng các môn học trong giai đoạn để đưa vào Portfolio cá nhân.",
                    target_skills=request.missing_skills[:3] if request.missing_skills else ["Problem Solving"]
                )
            )
        )

        # Giai đoạn 2
        semesters.append(
            SemesterMilestone(
                phase_number=2,
                phase_title="Giai đoạn 2: Nâng cao & Ứng dụng Thực chiến",
                semester_number=2,
                semester_title="Giai đoạn 2: Nâng cao & Ứng dụng Thực chiến",
                target_focus="Thiết kế kiến trúc hệ thống và triển khai mô hình học sâu vào thực tế",
                milestone_skills=["Deep Learning", "System Design", "Microservices"],
                recommended_courses=[
                    RecommendedCourse(
                        course_code="CS401",
                        course_name="Học sâu & Thị giác máy tính",
                        credits=3,
                        rationale="Áp dụng PyTorch huấn luyện mô hình tích chập xử lý dữ liệu phức tạp.",
                        target_skills=["Deep Learning", "PyTorch"],
                        prerequisites_satisfied=True
                    )
                ],
                certifications=["TensorFlow Developer Certificate"],
                practical_project=PracticalProject(
                    project_title="Hệ thống Nhận dạng & Phân tích Đa nhãn",
                    description="Triển khai dịch vụ phân loại dữ liệu thời gian thực tích hợp REST API.",
                    target_skills=["PyTorch", "FastAPI", "Docker"]
                )
            )
        )

        # Giai đoạn 3
        semesters.append(
            SemesterMilestone(
                phase_number=3,
                phase_title="Giai đoạn 3: Tối ưu Hệ thống & Sẵn sàng Nghề nghiệp",
                semester_number=3,
                semester_title="Giai đoạn 3: Tối ưu Hệ thống & Sẵn sàng Nghề nghiệp",
                target_focus="Hoàn thiện Đồ án Capstone, làm chủ kiến trúc Cloud và chuẩn bị phỏng vấn",
                milestone_skills=["RAG", "System Optimization", "Cloud Computing"],
                recommended_courses=[
                    RecommendedCourse(
                        course_code="CS402",
                        course_name="Xử lý Ngôn ngữ Tự nhiên & LLM",
                        credits=3,
                        rationale="Nghiên cứu kiến trúc Transformer và xây dựng trợ lý AI thông minh.",
                        target_skills=["NLP", "Transformers", "RAG"],
                        prerequisites_satisfied=True
                    )
                ],
                certifications=["Google Cloud Professional Data Engineer"],
                practical_project=PracticalProject(
                    project_title="Đồ án Capstone: Trợ lý AI Phân tích Doanh nghiệp",
                    description="Xây dựng giải pháp RAG tích hợp mô hình ngôn ngữ lớn và Vector Database chuẩn Production.",
                    target_skills=["RAG", "ChromaDB", "LLM", "Cloud Deploy"]
                )
            )
        )

        return RoadmapGenerationResponse(
            target_major=request.target_major_id,
            job_readiness_percentage=72.5,
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
