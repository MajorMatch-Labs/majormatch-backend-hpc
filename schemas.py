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
    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(..., description="Mã định danh truy vết UUID")
    status: str = Field("success", description="Trạng thái phân tích")
    parsing_duration_ms: float = Field(..., ge=0.0, description="Thời gian bóc tách (ms)")
    profile_data: ProfileData


# =============================================================================
# 2. ASSESSMENT & SKILL GAP QUANTIFICATION SCHEMAS
# =============================================================================

class HollandScores(BaseModel):
    model_config = ConfigDict(extra="ignore")

    realistic: float = Field(3.0, description="Nhóm Thực tế")
    investigative: float = Field(3.0, description="Nhóm Nghiên cứu")
    artistic: float = Field(3.0, description="Nhóm Nghệ thuật")
    social: float = Field(3.0, description="Nhóm Xã hội")
    enterprising: float = Field(3.0, description="Nhóm Quản lý / Khởi nghiệp")
    conventional: float = Field(3.0, description="Nhóm Quy củ / Chi tiết")


class CourseInputAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    course_code: str = Field(..., description="Mã môn học")
    grade_point: float = Field(..., ge=0.0, le=4.0, description="Điểm hệ số 4.0")


class CalculateMatchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    student_id: Optional[str] = Field("SV-2025001", description="Mã số sinh viên định danh hồ sơ")
    holland_scores: HollandScores = Field(default_factory=HollandScores)
    target_career_tags: List[str] = Field(default_factory=list, description="Danh mục thẻ định hướng")
    courses: List[CourseInputAssessment] = Field(default_factory=list, description="Danh mục môn và điểm")


class RadarAxisItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    axis_name: str = Field(..., description="Tên trục năng lực")
    user_score: float = Field(..., ge=0.0, le=10.0, description="Điểm số người dùng (thang 10)")
    benchmark_score: float = Field(..., ge=0.0, le=10.0, description="Điểm tiêu chuẩn ngành (thang 10)")


class SkillBreakdown(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mastered_skills: List[str] = Field(default_factory=list, description="Kỹ năng đã làm chủ")
    developing_skills: List[str] = Field(default_factory=list, description="Kỹ năng đang phát triển")
    missing_skills: List[str] = Field(default_factory=list, description="Kỹ năng còn thiếu cần bù đắp")


class MajorMatchRankItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    major_id: str = Field(..., description="Mã chuyên ngành")
    major_name: str = Field(..., description="Tên chuyên ngành đào tạo")
    match_percentage: float = Field(..., ge=0.0, le=100.0, description="Tỷ lệ phù hợp (%)")
    rank: int = Field(..., ge=1, description="Thứ hạng xếp loại")
    radar_data: Optional[List[RadarAxisItem]] = None
    skill_gap: Optional[SkillBreakdown] = None


class CalculateMatchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    top_matches: List[MajorMatchRankItem]
    radar_chart_data: List[RadarAxisItem]
    skill_breakdown: SkillBreakdown


# =============================================================================
# 3. ROADMAP GENERATION SCHEMAS
# =============================================================================

class RoadmapGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    student_id: Optional[str] = Field("SV-2025001", description="Mã số sinh viên định danh hồ sơ")
    target_major_id: Optional[str] = Field("DEVOPS_CLOUD", description="Mã chuyên ngành mục tiêu", example="CS_DATA_AI")
    target_major: Optional[str] = Field(None, description="Tên chuyên ngành mục tiêu")
    missing_skills: List[str] = Field(default_factory=list, description="Danh sách kỹ năng cần bù đắp")
    completed_course_codes: List[str] = Field(default_factory=list, description="Mã các môn đã hoàn thành")
    current_semester: Optional[int] = Field(4, description="Học kỳ hiện tại của sinh viên")
    cumulative_gpa: Optional[float] = Field(None, description="Điểm trung bình tích lũy hiện tại")
    initial_readiness_score: Optional[float] = Field(None, description="Điểm tiềm năng ban đầu từ phân tích Skill Gap")


class RecommendedCourse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    course_code: str = Field(..., description="Mã môn học")
    course_name: str = Field(..., description="Tên môn học")
    credits: int = Field(..., ge=1, le=6, description="Số tín chỉ")
    rationale: str = Field(..., description="Lý do đề xuất môn này")
    target_skills: List[str] = Field(default_factory=list, description="Kỹ năng cụ thể học phần cung cấp")
    prerequisites_satisfied: bool = Field(True, description="Điều kiện tiên quyết đã thỏa mãn")


class PracticalProject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project_title: str = Field(..., description="Tên đề tài đồ án")
    description: str = Field(..., description="Mô tả đồ án thực chiến")
    target_skills: List[str] = Field(default_factory=list, description="Kỹ năng rèn luyện được qua đồ án")


class PhaseMilestone(BaseModel):
    model_config = ConfigDict(extra="ignore")

    phase_number: int = Field(1, description="Số thứ tự giai đoạn")
    phase_title: str = Field(..., description="Tiêu đề giai đoạn học tập")
    semester_number: Optional[int] = None
    semester_title: Optional[str] = None
    target_focus: Optional[str] = None
    milestone_skills: List[str] = Field(default_factory=list, description="Kỹ năng chuẩn đầu ra của giai đoạn")
    recommended_courses: List[RecommendedCourse] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list, description="Chứng chỉ quốc tế đề xuất")
    practical_project: Optional[PracticalProject] = None


# Alias tương thích ngược
SemesterMilestone = PhaseMilestone


class RoadmapGenerationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    student_id: Optional[str] = Field(None, description="Mã số sinh viên sở hữu lộ trình")
    target_major: str = Field(..., description="Tên chuyên ngành mục tiêu")
    target_major_id: Optional[str] = None
    job_readiness_percentage: float = Field(..., ge=0.0, le=100.0, description="Tỷ lệ sẵn sàng nghề nghiệp")
    readiness_score: Optional[float] = None
    semesters: List[PhaseMilestone] = Field(default_factory=list)
    phases: Optional[List[PhaseMilestone]] = None
    completed_items: Optional[Dict[str, bool]] = Field(default_factory=dict)


class UpdateProgressRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    student_id: Optional[str] = Field("SV-2025001", description="Mã số sinh viên")
    completed_items: Dict[str, bool] = Field(default_factory=dict, description="Trạng thái checkbox các môn/đồ án")
    readiness_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Điểm Job Readiness mới")


# =============================================================================
# 4. STREAMING CHAT SCHEMAS
# =============================================================================

class ChatContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    target_major: Optional[str] = None
    current_gpa: Optional[float] = None
    missing_skills: Optional[List[str]] = None


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conversation_id: Optional[str] = Field(default_factory=lambda: "conv-default", description="ID phiên trò chuyện")
    message: str = Field(..., min_length=1, description="Nội dung câu hỏi của người dùng")
    context: Optional[ChatContext] = None
    major_focus: Optional[str] = None
    gpa: Optional[float] = None
    mastered_skills: Optional[List[str]] = None
    missing_skills: Optional[List[str]] = None
    student_profile_context: Optional[str] = None


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
