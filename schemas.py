"""
Đặc tả các Pydantic Schemas cho toàn bộ API của MajorMatch Private Compute Node.
Tuân thủ nghiêm ngặt chuẩn OpenAPI 3.1 và tài liệu API_SPEC.md.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# 1. PROFILE & TRANSCRIPT PARSING SCHEMAS
# =============================================================================

class CourseParsedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    course_code: str = Field(..., description="Mã môn học chuẩn hóa", example="CS102")
    course_name: str = Field(..., description="Tên môn học tiếng Việt", example="Cấu trúc dữ liệu và giải thuật")
    credits: int = Field(..., ge=1, le=6, description="Số tín chỉ", example=3)
    grade_letter: str = Field(..., description="Điểm hệ chữ", example="A")
    grade_point: float = Field(..., ge=0.0, le=4.0, description="Điểm quy đổi hệ 4.0", example=4.0)


class ProfileData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cumulative_gpa: float = Field(..., ge=0.0, le=4.0, description="Điểm trung bình tích lũy", example=3.45)
    total_credits: int = Field(..., ge=0, description="Tổng số tín chỉ đã tích lũy", example=45)
    courses: List[CourseParsedItem] = Field(default_factory=list, description="Danh sách môn học đã hoàn thành")
    detected_skills: List[str] = Field(default_factory=list, description="Danh sách kỹ năng bóc tách được")


class TranscriptParsingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="Mã định danh truy vết UUID")
    status: str = Field("success", description="Trạng thái phân tích")
    parsing_duration_ms: float = Field(..., ge=0.0, description="Thời gian bóc tách (ms)")
    profile_data: ProfileData


# =============================================================================
# 2. ASSESSMENT & SKILL GAP QUANTIFICATION SCHEMAS
# =============================================================================

class HollandScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    realistic: float = Field(3.0, ge=1.0, le=5.0, description="Nhóm Thực tế")
    investigative: float = Field(3.0, ge=1.0, le=5.0, description="Nhóm Nghiên cứu")
    artistic: float = Field(3.0, ge=1.0, le=5.0, description="Nhóm Nghệ thuật")
    social: float = Field(3.0, ge=1.0, le=5.0, description="Nhóm Xã hội")
    enterprising: float = Field(3.0, ge=1.0, le=5.0, description="Nhóm Quản lý / Khởi nghiệp")
    conventional: float = Field(3.0, ge=1.0, le=5.0, description="Nhóm Quy củ / Chi tiết")


class CourseInputAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    course_code: str = Field(..., description="Mã môn học")
    grade_point: float = Field(..., ge=0.0, le=4.0, description="Điểm hệ số 4.0")


class CalculateMatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holland_scores: HollandScores
    target_career_tags: List[str] = Field(..., min_length=1, max_length=5, description="Danh mục thẻ định hướng")
    courses: List[CourseInputAssessment] = Field(default_factory=list, description="Danh mục môn và điểm")


class MajorMatchRankItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    major_id: str = Field(..., description="Mã chuyên ngành")
    major_name: str = Field(..., description="Tên chuyên ngành đào tạo")
    match_percentage: float = Field(..., ge=0.0, le=100.0, description="Tỷ lệ phù hợp (%)")
    rank: int = Field(..., ge=1, description="Thứ hạng xếp loại")


class RadarAxisItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axis_name: str = Field(..., description="Tên trục năng lực")
    user_score: float = Field(..., ge=0.0, le=10.0, description="Điểm số người dùng (thang 10)")
    benchmark_score: float = Field(..., ge=0.0, le=10.0, description="Điểm tiêu chuẩn ngành (thang 10)")


class SkillBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mastered_skills: List[str] = Field(default_factory=list, description="Kỹ năng đã làm chủ")
    developing_skills: List[str] = Field(default_factory=list, description="Kỹ năng đang phát triển")
    missing_skills: List[str] = Field(default_factory=list, description="Kỹ năng còn thiếu cần bù đắp")


class CalculateMatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_matches: List[MajorMatchRankItem]
    radar_chart_data: List[RadarAxisItem]
    skill_breakdown: SkillBreakdown


# =============================================================================
# 3. ROADMAP GENERATION SCHEMAS
# =============================================================================

class RoadmapGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_major_id: str = Field(..., description="Mã chuyên ngành mục tiêu", example="CS_DATA_AI")
    missing_skills: List[str] = Field(default_factory=list, description="Danh sách kỹ năng cần bù đắp")
    completed_course_codes: List[str] = Field(default_factory=list, description="Mã các môn đã hoàn thành")
    current_semester: int = Field(..., ge=1, le=10, description="Học kỳ hiện tại của sinh viên", example=4)


class RecommendedCourse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    course_code: str = Field(..., description="Mã môn học")
    course_name: str = Field(..., description="Tên môn học")
    credits: int = Field(..., ge=1, le=6, description="Số tín chỉ")
    rationale: str = Field(..., description="Lý do đề xuất môn này")
    prerequisites_satisfied: bool = Field(True, description="Điều kiện tiên quyết đã thỏa mãn")


class PracticalProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_title: str = Field(..., description="Tên đề tài đồ án")
    description: str = Field(..., description="Mô tả đồ án thực chiến")
    target_skills: List[str] = Field(default_factory=list, description="Kỹ năng rèn luyện được qua đồ án")


class SemesterMilestone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semester_number: int = Field(..., description="Số thứ tự học kỳ")
    semester_title: str = Field(..., description="Tiêu đề học kỳ")
    recommended_courses: List[RecommendedCourse] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list, description="Chứng chỉ quốc tế đề xuất")
    practical_project: PracticalProject


class RoadmapGenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_major: str = Field(..., description="Tên chuyên ngành mục tiêu")
    job_readiness_percentage: float = Field(..., ge=0.0, le=100.0, description="Tỷ lệ sẵn sàng nghề nghiệp")
    semesters: List[SemesterMilestone] = Field(default_factory=list)


# =============================================================================
# 4. STREAMING CHAT SCHEMAS
# =============================================================================

class ChatContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    target_major: Optional[str] = None
    current_gpa: Optional[float] = None
    missing_skills: Optional[List[str]] = None


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(..., description="ID phiên trò chuyện")
    message: str = Field(..., min_length=1, description="Nội dung câu hỏi của người dùng")
    context: Optional[ChatContext] = None


# =============================================================================
# 5. ERROR & HEALTH SCHEMAS (RFC 7807)
# =============================================================================

class ProblemDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field("about:blank", description="URI tham chiếu lỗi")
    title: str = Field(..., description="Tiêu đề lỗi ngắn gọn")
    status: int = Field(..., description="HTTP Status Code")
    detail: str = Field(..., description="Mô tả chi tiết nguyên nhân lỗi")
    instance: str = Field(..., description="Endpoint phát sinh lỗi")
    timestamp: Optional[str] = None
    retry_after_seconds: Optional[int] = None


class GatewayNodeStatus(BaseModel):
    device: str
    os: str
    sqlite_cache_size_kb: int


class PrivateComputeNodeStatus(BaseModel):
    device: str
    gpu_status: str
    vram_used_mb: int
    vram_total_mb: int
    ollama_status: str
    active_model: str


class HealthCheckResponse(BaseModel):
    status: str
    gateway_node: GatewayNodeStatus
    private_compute_node: PrivateComputeNodeStatus
