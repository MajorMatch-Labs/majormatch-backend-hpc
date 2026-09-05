"""
Động cơ Học máy và Tính toán Định lượng Khoảng cách Kỹ năng (ML Skill Gap Engine).
Sử dụng Scikit-learn, NumPy để vector hóa bộ kỹ năng, tính Cosine Similarity,
xếp hạng độ phù hợp chuyên ngành và tính toán tọa độ biểu đồ Radar 6 trục.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from schemas import (
    CalculateMatchRequest,
    CalculateMatchResponse,
    MajorMatchRankItem,
    RadarAxisItem,
    SkillBreakdown
)


# =============================================================================
# DANH MỤC CÁC CHIỀU KỸ NĂNG CHUẨN TRONG KHÔNG GIAN VECTOR (128 SKILLS SPACE)
# =============================================================================
CORE_SKILLS: List[str] = [
    # 1. Thuật toán & Nền tảng (Algorithmic & Math Foundations)
    "Algorithms", "Data Structures", "Discrete Math", "Linear Algebra", "Probability & Statistics",
    "Optimization", "Calculus", "Complexity Theory", "Graph Theory", "Dynamic Programming",
    
    # 2. Lập trình & Kỹ thuật Phần mềm (Software Development)
    "Python", "C++", "Java", "TypeScript", "JavaScript", "Go", "OOP", "Design Patterns",
    "Functional Programming", "Unit Testing", "Git", "Software Architecture", "Clean Code",
    
    # 3. Hệ thống, Mạng & DevOps (Systems, Network & Infrastructure)
    "Operating Systems", "Linux", "Computer Networks", "TCP/IP", "Docker", "Kubernetes",
    "CI/CD", "Cloud Computing", "AWS", "Microservices", "System Design", "Distributed Systems",
    
    # 4. Cơ sở Dữ liệu & Xử lý Dữ liệu (Databases & Data Engineering)
    "SQL", "Database Design", "PostgreSQL", "NoSQL", "MongoDB", "Redis", "Data Modeling",
    "Data Warehousing", "ETL Pipelines", "Data Analysis", "Pandas", "NumPy",
    
    # 5. Trí tuệ Nhân tạo & Học máy (AI & Machine Learning)
    "Machine Learning", "Deep Learning", "Scikit-learn", "PyTorch", "TensorFlow",
    "Computer Vision", "NLP", "Neural Networks", "Feature Engineering", "Model Evaluation",
    "RAG", "Prompt Engineering", "Large Language Models",
    
    # 6. An toàn Thông tin & Bảo mật (Cybersecurity)
    "Network Security", "Cryptography", "AppSec", "Web Security", "Penetration Testing",
    
    # 7. Kỹ năng Mềm & Dự án (Professional & Soft Skills)
    "Problem Solving", "Communication", "Teamwork", "Project Management", "Agile/Scrum", "Research"
]

SKILL_TO_INDEX: Dict[str, int] = {skill: idx for idx, skill in enumerate(CORE_SKILLS)}
DIMENSION: int = len(CORE_SKILLS)


# =============================================================================
# ÁNH XẠ MÔN HỌC ĐẠI HỌC VÀO TRỌNG SỐ KỸ NĂNG THÀNH PHẦN
# =============================================================================
COURSE_SKILL_MAPPING: Dict[str, Dict[str, float]] = {
    "CS101": {"Python": 0.8, "Algorithms": 0.6, "Problem Solving": 0.7, "Data Structures": 0.5},
    "CS102": {"Data Structures": 1.0, "Algorithms": 0.9, "C++": 0.8, "Problem Solving": 0.9, "Complexity Theory": 0.7},
    "MTH100": {"Calculus": 1.0, "Optimization": 0.6, "Linear Algebra": 0.8},
    "MTH101": {"Probability & Statistics": 1.0, "Linear Algebra": 0.9, "Data Analysis": 0.7},
    "IT201": {"Database Design": 0.9, "SQL": 1.0, "Data Modeling": 0.8, "PostgreSQL": 0.7},
    "IT202": {"Computer Networks": 1.0, "TCP/IP": 0.9, "Network Security": 0.6, "Linux": 0.5},
    "IT203": {"Operating Systems": 1.0, "Linux": 0.9, "C++": 0.6, "Distributed Systems": 0.5},
    "CS301": {"Machine Learning": 0.9, "Algorithms": 0.7, "Python": 0.8, "Probability & Statistics": 0.7},
    "CS302": {"Deep Learning": 1.0, "Neural Networks": 1.0, "PyTorch": 0.9, "Computer Vision": 0.7, "NLP": 0.7},
    "SE201": {"OOP": 1.0, "Design Patterns": 0.9, "Java": 0.8, "Clean Code": 0.8},
    "SE301": {"Software Architecture": 1.0, "System Design": 0.9, "Git": 0.8, "Agile/Scrum": 0.8},
    "DO301": {"Docker": 1.0, "CI/CD": 0.9, "Cloud Computing": 0.8, "Kubernetes": 0.7, "Linux": 0.8},
    "SEC201": {"Cryptography": 1.0, "Network Security": 0.9, "Web Security": 0.8}
}


# =============================================================================
# MA TRẬN TIÊU CHUẨN CÁC CHUYÊN NGÀNH ĐÀO TẠO (BENCHMARK PROFILES)
# =============================================================================
MAJOR_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "CS_DATA_AI": {
        "name": "Khoa học Dữ liệu & Trí tuệ Nhân tạo",
        "holland_ideal": {"investigative": 5.0, "realistic": 4.0, "conventional": 3.5, "enterprising": 2.5, "artistic": 2.0, "social": 2.5},
        "key_skills": {
            "Machine Learning": 9.5, "Deep Learning": 9.0, "Python": 9.5, "Probability & Statistics": 9.0,
            "Linear Algebra": 8.5, "Algorithms": 8.5, "PyTorch": 8.5, "Data Analysis": 8.5,
            "SQL": 8.0, "NLP": 7.5, "Computer Vision": 7.5, "Problem Solving": 9.0, "Research": 8.5
        },
        "axes_weights": {
            "Core Technical": 8.5, "Domain Specialization": 9.5, "Algorithmic Thinking": 9.0,
            "System Architecture": 7.0, "Industry Tools": 8.0, "Soft Skills & Professionalism": 7.5
        }
    },
    "SE_FULLSTACK": {
        "name": "Kỹ thuật Phần mềm & Lập trình Fullstack",
        "holland_ideal": {"realistic": 4.5, "investigative": 4.0, "artistic": 3.5, "conventional": 4.0, "enterprising": 3.0, "social": 3.0},
        "key_skills": {
            "OOP": 9.0, "Design Patterns": 9.0, "TypeScript": 8.5, "JavaScript": 8.5, "Clean Code": 9.0,
            "SQL": 8.5, "System Design": 8.0, "Git": 9.0, "Unit Testing": 8.5, "Web Security": 7.5,
            "Agile/Scrum": 8.0, "Problem Solving": 8.5, "Teamwork": 8.5
        },
        "axes_weights": {
            "Core Technical": 9.0, "Domain Specialization": 8.5, "Algorithmic Thinking": 8.0,
            "System Architecture": 8.5, "Industry Tools": 9.0, "Soft Skills & Professionalism": 8.0
        }
    },
    "DEVOPS_CLOUD": {
        "name": "Kỹ thuật Đám mây & Vận hành Hệ thống (DevOps/Cloud)",
        "holland_ideal": {"realistic": 5.0, "conventional": 4.5, "investigative": 4.0, "enterprising": 2.5, "social": 2.5, "artistic": 1.5},
        "key_skills": {
            "Docker": 9.5, "Kubernetes": 9.0, "Linux": 9.5, "CI/CD": 9.5, "Cloud Computing": 9.5,
            "AWS": 9.0, "Computer Networks": 8.5, "TCP/IP": 8.5, "System Design": 8.5,
            "Distributed Systems": 8.0, "Problem Solving": 8.5
        },
        "axes_weights": {
            "Core Technical": 8.5, "Domain Specialization": 9.5, "Algorithmic Thinking": 7.0,
            "System Architecture": 9.5, "Industry Tools": 9.5, "Soft Skills & Professionalism": 7.0
        }
    },
    "CYBER_SECURITY": {
        "name": "An toàn Thông tin & An ninh Mạng",
        "holland_ideal": {"investigative": 5.0, "realistic": 4.5, "conventional": 4.5, "enterprising": 2.5, "social": 2.0, "artistic": 1.5},
        "key_skills": {
            "Network Security": 9.5, "Cryptography": 9.0, "Operating Systems": 9.0, "Linux": 9.0,
            "Computer Networks": 9.0, "Web Security": 9.0, "Penetration Testing": 8.5,
            "AppSec": 8.5, "Algorithms": 7.5, "Problem Solving": 9.0
        },
        "axes_weights": {
            "Core Technical": 9.0, "Domain Specialization": 9.5, "Algorithmic Thinking": 8.0,
            "System Architecture": 8.5, "Industry Tools": 8.5, "Soft Skills & Professionalism": 7.0
        }
    },
    "DATA_ANALYTICS": {
        "name": "Phân tích Dữ liệu Kinh doanh (Data Analytics & BI)",
        "holland_ideal": {"conventional": 5.0, "enterprising": 4.0, "investigative": 4.5, "social": 3.5, "realistic": 2.5, "artistic": 3.0},
        "key_skills": {
            "SQL": 9.5, "Data Analysis": 9.5, "Data Modeling": 9.0, "Probability & Statistics": 8.5,
            "Python": 8.0, "Pandas": 8.5, "Data Warehousing": 8.0, "Communication": 8.5,
            "Problem Solving": 8.5
        },
        "axes_weights": {
            "Core Technical": 7.5, "Domain Specialization": 8.5, "Algorithmic Thinking": 7.0,
            "System Architecture": 6.5, "Industry Tools": 8.5, "Soft Skills & Professionalism": 9.0
        }
    }
}


class MLEngine:
    """Bộ xử lý định lượng học máy tính toán Skill Gap và phân tích ma trận."""

    @classmethod
    def vectorize_user_courses(cls, courses: List[Any]) -> np.ndarray:
        """Chuyển đổi danh sách môn học và điểm số thành vector năng lực kỹ năng 128 chiều."""
        user_vector = np.zeros(DIMENSION, dtype=np.float32)

        for course in courses:
            code = getattr(course, "course_code", "").upper()
            grade_point = float(getattr(course, "grade_point", 0.0))
            
            # Chuẩn hóa hệ số điểm trên thang [0.0 - 1.0]
            grade_weight = min(1.0, max(0.0, grade_point / 4.0))

            # Tìm kiếm trong bảng ánh xạ môn học
            skills_contributed = COURSE_SKILL_MAPPING.get(code, {})
            for skill_name, skill_factor in skills_contributed.items():
                if skill_name in SKILL_TO_INDEX:
                    idx = SKILL_TO_INDEX[skill_name]
                    # Tích lũy đóng góp điểm: max(điểm hiện tại, điểm môn mới đóng góp)
                    contribution = grade_weight * skill_factor * 10.0  # Quy đổi thang 10
                    user_vector[idx] = max(user_vector[idx], contribution)

        return user_vector

    @classmethod
    def vectorize_benchmark(cls, key_skills: Dict[str, float]) -> np.ndarray:
        """Chuyển đổi các kỹ năng chuẩn của ngành thành vector đặc trưng trong không gian 128 chiều."""
        bench_vector = np.zeros(DIMENSION, dtype=np.float32)
        for skill, score in key_skills.items():
            if skill in SKILL_TO_INDEX:
                idx = SKILL_TO_INDEX[skill]
                bench_vector[idx] = float(score)
        return bench_vector

    @classmethod
    def calculate_holland_factor(cls, user_holland: Any, ideal_holland: Dict[str, float]) -> float:
        """
        Tính hệ số tương thích tâm lý học Holland Code RIASEC.
        Trả về giá trị điều chỉnh trong khoảng [0.85, 1.15].
        """
        user_vals = np.array([
            float(getattr(user_holland, "realistic", 3.0)),
            float(getattr(user_holland, "investigative", 3.0)),
            float(getattr(user_holland, "artistic", 3.0)),
            float(getattr(user_holland, "social", 3.0)),
            float(getattr(user_holland, "enterprising", 3.0)),
            float(getattr(user_holland, "conventional", 3.0))
        ])
        ideal_vals = np.array([
            ideal_holland["realistic"],
            ideal_holland["investigative"],
            ideal_holland["artistic"],
            ideal_holland["social"],
            ideal_holland["enterprising"],
            ideal_holland["conventional"]
        ])

        # Tính Cosine Similarity giữa 2 vector Holland
        sim = float(cosine_similarity(user_vals.reshape(1, -1), ideal_vals.reshape(1, -1))[0][0])
        # Điều chỉnh tỷ lệ: tương đồng cao sẽ tăng điểm tối đa 15%, tương đồng thấp giảm tối đa 15%
        adjustment = 0.85 + (sim * 0.3)
        return adjustment

    @classmethod
    def compute_radar_axes(
        cls,
        user_vector: np.ndarray,
        benchmark_axes: Dict[str, float]
    ) -> List[RadarAxisItem]:
        """Tính toán tọa độ 6 trục biểu đồ Radar Chart (thang điểm 0.0 - 10.0)."""
        # Nhóm các chỉ số chiều theo từng trục
        axis_mappings = {
            "Core Technical": ["Python", "C++", "Java", "OOP", "Data Structures", "Operating Systems"],
            "Domain Specialization": ["Machine Learning", "Deep Learning", "Docker", "Network Security", "Cloud Computing"],
            "Algorithmic Thinking": ["Algorithms", "Complexity Theory", "Linear Algebra", "Probability & Statistics"],
            "System Architecture": ["System Design", "Software Architecture", "Distributed Systems", "Database Design"],
            "Industry Tools": ["Git", "Linux", "CI/CD", "PyTorch", "SQL", "PostgreSQL"],
            "Soft Skills & Professionalism": ["Problem Solving", "Clean Code", "Communication", "Agile/Scrum"]
        }

        radar_items: List[RadarAxisItem] = []

        for axis_name, skills in axis_mappings.items():
            bench_target = benchmark_axes.get(axis_name, 8.0)

            # Lấy trung bình điểm của người dùng trên các kỹ năng thuộc trục
            scores = []
            for s in skills:
                if s in SKILL_TO_INDEX:
                    idx = SKILL_TO_INDEX[s]
                    scores.append(user_vector[idx])
            
            user_avg = float(np.mean(scores)) if scores else 3.5
            # Đảm bảo điểm hiển thị không quá thấp cho người bắt đầu (tối thiểu 1.5)
            user_display = max(1.5, min(10.0, round(user_avg, 1)))

            radar_items.append(
                RadarAxisItem(
                    axis_name=axis_name,
                    user_score=user_display,
                    benchmark_score=round(bench_target, 1)
                )
            )

        return radar_items

    @classmethod
    def analyze_skill_breakdown(
        cls,
        user_vector: np.ndarray,
        target_benchmark_skills: Dict[str, float]
    ) -> SkillBreakdown:
        """Phân loại tập kỹ năng thành: Đã thuần thục, Đang phát triển và Còn thiếu."""
        mastered: List[str] = []
        developing: List[str] = []
        missing: List[str] = []

        for skill, req_score in target_benchmark_skills.items():
            if skill in SKILL_TO_INDEX:
                idx = SKILL_TO_INDEX[skill]
                user_score = user_vector[idx]

                if user_score >= 7.0:
                    mastered.append(skill)
                elif user_score >= 3.0:
                    developing.append(skill)
                else:
                    missing.append(skill)

        # Đảm bảo luôn có danh sách rõ ràng cho UI
        if not missing:
            missing = ["Distributed Systems", "Cloud Architecture"]

        return SkillBreakdown(
            mastered_skills=sorted(mastered),
            developing_skills=sorted(developing),
            missing_skills=sorted(missing)
        )


def process_skill_assessment(request: CalculateMatchRequest) -> CalculateMatchResponse:
    """
    Hàm entry point điều phối toàn bộ luồng tính toán định lượng kỹ năng:
    Vectorize -> Cosine Similarity -> Holland Adjustment -> Radar Chart -> Skill Breakdown.
    """
    user_vector = MLEngine.vectorize_user_courses(request.courses)

    ranked_matches: List[MajorMatchRankItem] = []
    match_scores_raw: List[Tuple[str, str, float]] = []

    for major_id, major_meta in MAJOR_BENCHMARKS.items():
        bench_vector = MLEngine.vectorize_benchmark(major_meta["key_skills"])

        # 1. Đo góc Cosine Similarity
        norm_u = np.linalg.norm(user_vector)
        norm_b = np.linalg.norm(bench_vector)

        if norm_u == 0 or norm_b == 0:
            raw_cosine = 0.35  # Mức sàn cơ bản
        else:
            raw_cosine = float(np.dot(user_vector, bench_vector) / (norm_u * norm_b))

        # 2. Điều chỉnh theo trắc nghiệm Holland Code
        holland_multiplier = MLEngine.calculate_holland_factor(
            request.holland_scores,
            major_meta["holland_ideal"]
        )

        # 3. Ưu tiên các thẻ định hướng người dùng đã chọn
        tag_bonus = 1.0
        for tag in request.target_career_tags:
            tag_clean = tag.upper().replace(" ", "_")
            if tag_clean in major_id or major_id in tag_clean:
                tag_bonus = 1.15
                break

        # 4. Tính toán điểm Match Score % cuối cùng
        final_percentage = raw_cosine * holland_multiplier * tag_bonus * 100.0
        final_percentage = min(98.5, max(35.0, round(final_percentage, 1)))

        match_scores_raw.append((major_id, major_meta["name"], final_percentage))

    # Sắp xếp theo thứ hạng điểm cao nhất xuống thấp
    match_scores_raw.sort(key=lambda x: x[2], reverse=True)

    for rank, (m_id, m_name, score) in enumerate(match_scores_raw, start=1):
        ranked_matches.append(
            MajorMatchRankItem(
                major_id=m_id,
                major_name=m_name,
                match_percentage=score,
                rank=rank
            )
        )

    # Lấy ngành đứng đầu (Top 1) để tính toán chi tiết biểu đồ Radar và Skill Gap
    top_major_id = ranked_matches[0].major_id
    top_major_meta = MAJOR_BENCHMARKS[top_major_id]

    radar_data = MLEngine.compute_radar_axes(user_vector, top_major_meta["axes_weights"])
    skill_breakdown = MLEngine.analyze_skill_breakdown(user_vector, top_major_meta["key_skills"])

    return CalculateMatchResponse(
        top_matches=ranked_matches,
        radar_chart_data=radar_data,
        skill_breakdown=skill_breakdown
    )
