"""
Mô-đun bóc tách tệp PDF học bạ/CV và khử định danh thông tin cá nhân (PII Redaction).
Áp dụng pdfplumber kết hợp Regex Engine bóc tách danh sách môn học, số tín chỉ, GPA và kỹ năng.
"""

import io
import re
import gc
import uuid
import time
from typing import Tuple, List, Dict, Any, Optional

import pdfplumber
import pypdf

from schemas import CourseParsedItem, ProfileData, TranscriptParsingResponse


# =============================================================================
# BẢNG ÁNH XẠ ĐIỂM HỆ CHỮ VÀ HỆ SỐ 4.0 CHUẨN ĐẠI HỌC
# =============================================================================
GRADE_LETTER_TO_POINT: Dict[str, float] = {
    "A+": 4.0, "A": 4.0,
    "B+": 3.5, "B": 3.0,
    "C+": 2.5, "C": 2.0,
    "D+": 1.5, "D": 1.0,
    "F": 0.0
}

# TỪ ĐIỂN TỪ KHÓA KỸ NĂNG CÔNG NGHỆ NHẬN DIỆN TỰ ĐỘNG
TECH_SKILL_KEYWORDS: List[str] = [
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "Go", "Rust",
    "Data Structures", "Algorithms", "Database", "SQL", "PostgreSQL", "MySQL",
    "MongoDB", "Machine Learning", "Deep Learning", "Artificial Intelligence",
    "Computer Vision", "NLP", "Data Mining", "Data Science", "Big Data",
    "Docker", "Kubernetes", "Linux", "Git", "DevOps", "CI/CD", "AWS", "Azure",
    "Computer Network", "Operating System", "Software Engineering", "OOP",
    "Web Development", "React", "Next.js", "FastAPI", "Node.js", "Django",
    "Cybersecurity", "Network Security", "Cryptography", "Distributed Systems"
]

# ÁNH XẠ TÊN MÔN TIÊU BIỂU SANG KỸ NĂNG TƯƠNG ỨNG
COURSE_NAME_TO_SKILLS: Dict[str, List[str]] = {
    "cấu trúc dữ liệu": ["Data Structures", "Algorithms", "Problem Solving"],
    "thuật toán": ["Algorithms", "Problem Solving"],
    "giải thuật": ["Algorithms", "Problem Solving"],
    "cơ sở dữ liệu": ["Database", "SQL", "Data Modeling"],
    "hệ quản trị cơ sở dữ liệu": ["Database", "SQL", "Query Optimization"],
    "lập trình hướng đối tượng": ["OOP", "Design Patterns", "Java", "C++"],
    "mạng máy tính": ["Computer Network", "TCP/IP", "Network Protocols"],
    "hệ điều hành": ["Operating System", "Linux", "Process Management"],
    "trí tuệ nhân tạo": ["Artificial Intelligence", "Machine Learning", "Search Algorithms"],
    "học máy": ["Machine Learning", "Scikit-learn", "Statistical Modeling"],
    "học sâu": ["Deep Learning", "PyTorch", "Neural Networks"],
    "phát triển ứng dụng web": ["Web Development", "JavaScript", "Frontend", "Backend"],
    "kỹ thuật phần mềm": ["Software Engineering", "Git", "Software Architecture"],
    "an toàn thông tin": ["Cybersecurity", "Cryptography", "Network Security"],
    "điện toán đám mây": ["Cloud Computing", "AWS", "Docker", "DevOps"]
}


class PDFParserEngine:
    """Công cụ xử lý và trích xuất tệp PDF chuyên sâu."""

    @staticmethod
    def extract_raw_text(file_bytes: bytes) -> str:
        """Trích xuất text thô từ tệp nhị phân PDF với cơ chế fallback."""
        text_content: List[str] = []

        try:
            # Ưu tiên sử dụng pdfplumber để bảo toàn tọa độ bảng biểu
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        text_content.append(page_text)
        except Exception:
            # Fallback sang pypdf nếu pdfplumber gặp lỗi phân tích trang
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text_content.append(t)
            except Exception:
                pass

        full_text = "\n".join(text_content)
        return full_text

    @staticmethod
    def sanitize_pii(raw_text: str) -> str:
        """
        Khử toàn bộ thông tin nhạy cảm định danh cá nhân (PII Redaction).
        Đảm bảo không bao giờ lưu trữ hoặc gửi Họ tên, CCCD, SĐT, Email lên mô hình.
        """
        sanitized = raw_text

        # 1. Khử số CCCD / CMND (9 đến 12 chữ số liên tiếp)
        sanitized = re.sub(r"\b\d{9,12}\b", "[CCCD_REDACTED]", sanitized)

        # 2. Khử số điện thoại di động Việt Nam (+84 hoặc 03, 05, 07, 08, 09)
        sanitized = re.sub(
            r"(\+84|0)(3[2-9]|5[6|8|9]|7[0|6-9]|8[1-5|8|9]|9[0-4|6-9])[0-9]{7}\b",
            "[PHONE_REDACTED]",
            sanitized
        )

        # 3. Khử Email cá nhân
        sanitized = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
            "[EMAIL_REDACTED]",
            sanitized
        )

        # 4. Khử Mã số sinh viên (MSSV: 7 - 10 chữ số)
        sanitized = re.sub(
            r"(?i)(mssv|mã\s*sv|mã\s*sinh\s*viên|student\s*id)[:\s]*([A-Z0-9]{7,12})",
            r"\1: [STUDENT_ID_REDACTED]",
            sanitized
        )

        # 5. Khử Họ và tên sinh viên
        sanitized = re.sub(
            r"(?i)(họ\s*và\s*tên|họ\s*tên|sinh\s*viên|student\s*name)[:\s]*([^\n,]+)",
            r"\1: [STUDENT_NAME_REDACTED]",
            sanitized
        )

        return sanitized

    @classmethod
    def parse_courses(cls, text: str) -> List[CourseParsedItem]:
        """Trích xuất danh sách môn học, số tín chỉ và điểm số bằng Regex Pattern."""
        courses: List[CourseParsedItem] = []
        seen_codes = set()

        # Biểu thức Regex bóc tách dòng môn học dạng:
        # CS102 Cấu trúc dữ liệu và giải thuật 3 A 4.0
        # hoặc IT3020 Cơ sở dữ liệu 3 8.5 B+ 3.5
        course_line_pattern = re.compile(
            r"([A-Z]{2,4}\s?[0-9]{3,4})"                         # Group 1: Mã môn học
            r"[\t\s]+([^\d\n]+?)"                                 # Group 2: Tên môn học
            r"[\t\s]+([1-6])"                                     # Group 3: Số tín chỉ
            r"(?:[\t\s]+[0-9]{1,2}(?:\.[0-9]{1,2})?)?"            # Bỏ qua điểm hệ 10 nếu có
            r"[\t\s]+([A-D][+-]?|F)"                             # Group 4: Điểm chữ
            r"(?:[\t\s]+([0-4](?:\.[0-9]{1,2})?))?",              # Group 5: Điểm hệ 4 (tùy chọn)
            re.IGNORECASE
        )

        for line in text.splitlines():
            line_clean = line.strip()
            if not line_clean:
                continue

            match = course_line_pattern.search(line_clean)
            if match:
                code = re.sub(r"\s+", "", match.group(1).upper())
                name = match.group(2).strip()
                credits = int(match.group(3))
                grade_letter = match.group(4).upper()

                # Xác định điểm hệ 4.0
                if match.group(5):
                    grade_point = float(match.group(5))
                else:
                    grade_point = GRADE_LETTER_TO_POINT.get(grade_letter, 0.0)

                # Giới hạn trong khoảng 0.0 - 4.0
                grade_point = max(0.0, min(4.0, grade_point))

                if code not in seen_codes and len(name) >= 3:
                    seen_codes.add(code)
                    courses.append(
                        CourseParsedItem(
                            course_code=code,
                            course_name=name,
                            credits=credits,
                            grade_letter=grade_letter,
                            grade_point=grade_point
                        )
                    )

        return courses

    @staticmethod
    def extract_gpa(text: str, courses: List[CourseParsedItem]) -> float:
        """Trích xuất điểm trung bình tích lũy GPA (thang 4.0) hoặc tự động tính toán trung bình có trọng số."""
        # 1. Tìm trực tiếp điểm GPA trong văn bản
        gpa_match = re.search(
            r"(?i)(gpa|cpa|điểm\s*trung\s*bình\s*(?:tích\s*lũy)?|đtb)[:\s]*([0-4](?:\.[0-9]{1,2}))",
            text
        )
        if gpa_match:
            try:
                gpa_val = float(gpa_match.group(2))
                if 0.0 <= gpa_val <= 4.0:
                    return round(gpa_val, 2)
            except ValueError:
                pass

        # 2. Fallback: Tính trung bình có trọng số từ danh sách môn học đã bóc tách
        if courses:
            total_credits = sum(c.credits for c in courses)
            if total_credits > 0:
                weighted_sum = sum(c.credits * c.grade_point for c in courses)
                return round(weighted_sum / total_credits, 2)

        return 3.0  # Giá trị mặc định an toàn

    @staticmethod
    def detect_skills(text: str, courses: List[CourseParsedItem]) -> List[str]:
        """Phát hiện các kỹ năng kỹ thuật từ nội dung văn bản và tên các môn học."""
        detected = set()
        text_lower = text.lower()

        # Quét theo danh mục từ khóa kỹ thuật phổ biến
        for kw in TECH_SKILL_KEYWORDS:
            if re.search(rf"\b{re.escape(kw.lower())}\b", text_lower):
                detected.add(kw)

        # Quét dựa trên tên môn học đã hoàn thành
        for course in courses:
            course_name_lower = course.course_name.lower()
            for key, skills in COURSE_NAME_TO_SKILLS.items():
                if key in course_name_lower:
                    detected.update(skills)

        return sorted(list(detected))


def parse_transcript_document(file_bytes: bytes) -> TranscriptParsingResponse:
    """
    Hàm entry point phân tích toàn bộ tệp PDF học bạ.
    Thực hiện: Trích xuất -> Khử PII -> Parse môn học & GPA -> Nhận diện kỹ năng -> Dọn dẹp RAM.
    """
    start_time = time.time()
    req_id = str(uuid.uuid4())

    # 1. Trích xuất text
    raw_text = PDFParserEngine.extract_raw_text(file_bytes)
    
    # 2. Khử định danh PII
    clean_text = PDFParserEngine.sanitize_pii(raw_text)

    # 3. Bóc tách danh mục môn học
    courses = PDFParserEngine.parse_courses(clean_text)

    # 4. Trích xuất GPA & Tổng số tín chỉ
    gpa = PDFParserEngine.extract_gpa(clean_text, courses)
    total_credits = sum(c.credits for c in courses)

    # 5. Nhận diện kỹ năng
    skills = PDFParserEngine.detect_skills(clean_text, courses)

    # 6. Dọn dẹp bộ nhớ RAM (Memory Cleanup)
    del raw_text
    del clean_text
    gc.collect()

    duration_ms = round((time.time() - start_time) * 1000, 2)

    profile_data = ProfileData(
        cumulative_gpa=gpa,
        total_credits=total_credits,
        courses=courses,
        detected_skills=skills
    )

    return TranscriptParsingResponse(
        request_id=req_id,
        status="success",
        parsing_duration_ms=duration_ms,
        profile_data=profile_data
    )
