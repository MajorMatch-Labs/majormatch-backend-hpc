"""
Cloud Gateway Persistence Service - MajorMatch AWS Cloud Tier.
Chạy trên AWS EC2, đóng vai trò Cổng Lưu trữ Bền vững & Điều phối Hybrid Cloud:
- Đọc/Ghi trực tiếp CSDL SQLite trên AWS EC2 (bảo toàn 100% dữ liệu khi HPC Node ngoại tuyến).
- Chuyển tiếp tác vụ AI nặng (LLM, RAG, Cosine Matrix) về Private HPC Compute Node (Legion PC).
"""

import os
import sqlite3
import json
import time
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Request, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

HPC_BASE_URL = os.getenv("HPC_BASE_URL", "http://127.0.0.1:8000")
DB_PATH = os.path.expanduser("~/data/majormatch.db")

app = FastAPI(
    title="MajorMatch Cloud Gateway Persistence Engine",
    description="Dịch vụ lưu trữ bền vững trên AWS EC2 kết hợp điều phối tính toán Hybrid Cloud.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# QUẢN TRỊ CSDL SQLITE TRÊN AWS EC2
# =============================================================================

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db():
    conn = get_db()
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS roadmaps (
            student_id TEXT PRIMARY KEY,
            target_major_id TEXT NOT NULL,
            target_major_name TEXT NOT NULL,
            job_readiness_percentage REAL NOT NULL,
            phases_data TEXT NOT NULL,
            completed_items TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            student_id TEXT PRIMARY KEY,
            target_career TEXT NOT NULL,
            assessment_data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            student_id TEXT PRIMARY KEY,
            full_name TEXT,
            cumulative_gpa REAL,
            total_credits INTEGER,
            courses_data TEXT,
            detected_skills TEXT,
            riasec_scores TEXT,
            career_tags TEXT,
            updated_at TEXT NOT NULL
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
    conn.close()

init_db()


# =============================================================================
# ENDPOINT NHÓM 1: TRUY VẤN & CẬP NHẬT TRỰC TIẾP TỪ CSDL CLOUD (OFFLINE-RESILIENT)
# =============================================================================

@app.get("/api/v1/health")
async def health_check():
    """Kiểm tra sức khỏe hệ thống: SQLite Cloud và Private HPC Node."""
    db_size_kb = 0
    if os.path.exists(DB_PATH):
        db_size_kb = round(os.path.getsize(DB_PATH) / 1024, 1)

    hpc_status = {"device": "Private HPC Compute Node (Legion PC)", "status": "offline"}
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{HPC_BASE_URL}/api/v1/health")
            if resp.status_code == 200:
                hpc_data = resp.json()
                hpc_status = hpc_data.get("private_compute_node", {"status": "online"})
    except Exception:
        hpc_status = {
            "device": "Private HPC Compute Node (Legion PC)",
            "status": "offline",
            "message": "Node HPC đang ngoại tuyến. Dữ liệu hồ sơ và lộ trình học tập vẫn được phục vụ 100% từ AWS Cloud."
        }

    return {
        "status": "healthy",
        "cloud_gateway": {
            "node": "AWS EC2 Cloud Node",
            "db_engine": "SQLite WAL Persistent Storage",
            "sqlite_cache_size_kb": db_size_kb,
            "persistence_tier": "Cloud Database"
        },
        "private_compute_node": hpc_status
    }


@app.post("/api/v1/assessment/calculate-match")
async def calculate_match_proxy(request: Request):
    """Chuyển tiếp tác vụ ML tới HPC Node & lưu bản sao vào SQLite EC2."""
    body_bytes = await request.body()
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(
                f"{HPC_BASE_URL}/api/v1/assessment/calculate-match",
                content=body_bytes,
                headers={"Content-Type": "application/json"}
            )
            if resp.status_code == 200:
                data = resp.json()
                # Tự động lưu kết quả vào CSDL SQLite EC2
                try:
                    req_json = json.loads(body_bytes.decode()) if body_bytes else {}
                    student_id = req_json.get("student_id", "SV-2025001")
                    target_career = ",".join(req_json.get("target_career_tags", ["devops_cloud"]))
                    await save_assessment_endpoint(student_id, data)
                except Exception:
                    pass
                return data
            else:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError:
        # Nếu HPC Node offline, phục vụ kết quả đã lưu từ CSDL SQLite EC2
        req_json = json.loads(body_bytes.decode()) if body_bytes else {}
        student_id = req_json.get("student_id", "SV-2025001")
        conn = get_db()
        row = conn.execute("SELECT assessment_data FROM assessments WHERE student_id = ?", (student_id,))
        conn.close()
        if row:
            return json.loads(row["assessment_data"])
        raise HTTPException(
            status_code=503,
            detail="Máy chủ tính toán AI HPC đang ngoại tuyến và chưa có bản ghi nào trên Cloud."
        )


@app.get("/api/v1/assessment/{student_id}")
@app.get("/api/v1/student/{student_id}/assessment")
async def get_assessment_endpoint(student_id: str):
    """Lấy kết quả đánh giá kỹ năng từ CSDL SQLite trên AWS EC2."""
    if student_id == "calculate-match":
        raise HTTPException(status_code=400, detail="Vui lòng gọi POST /api/v1/assessment/calculate-match")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT assessment_data FROM assessments WHERE student_id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chưa có kết quả đánh giá kỹ năng cho sinh viên {student_id} trên Cloud."
        )

    try:
        return json.loads(row["assessment_data"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc dữ liệu assessment: {str(e)}")


@app.post("/api/v1/assessment/{student_id}")
@app.post("/api/v1/student/{student_id}/assessment")
async def save_assessment_endpoint(student_id: str, request_body: Dict[str, Any]):
    """Lưu hoặc cập nhật kết quả đánh giá kỹ năng vào CSDL SQLite trên AWS EC2."""
    if student_id == "calculate-match":
        raise HTTPException(status_code=400, detail="Vui lòng gọi POST /api/v1/assessment/calculate-match")

    target_career = str(request_body.get("target_career", "devops_cloud"))
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data_str = json.dumps(request_body, ensure_ascii=False)

    conn = get_db()
    with conn:
        conn.execute("""
        INSERT INTO assessments (student_id, target_career, assessment_data, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(student_id) DO UPDATE SET
            target_career = excluded.target_career,
            assessment_data = excluded.assessment_data,
            updated_at = excluded.updated_at;
        """, (student_id, target_career, data_str, now_str, now_str))
    conn.close()
    return {"status": "success", "student_id": student_id, "saved_on": "AWS_EC2_SQLite"}


@app.get("/api/v1/roadmap/{student_id}")
async def get_roadmap_endpoint(student_id: str):
    """Lấy lộ trình học tập từ CSDL SQLite trên AWS EC2."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT student_id, target_major_id, target_major_name, job_readiness_percentage,
           phases_data, completed_items, created_at, updated_at
    FROM roadmaps WHERE student_id = ?
    """, (student_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chưa có lộ trình học tập nào được lưu cho sinh viên {student_id}."
        )

    try:
        phases = json.loads(row["phases_data"])
        completed = json.loads(row["completed_items"])
    except Exception:
        phases = []
        completed = {}

    return {
        "student_id": row["student_id"],
        "target_major": row["target_major_name"],
        "target_major_id": row["target_major_id"],
        "job_readiness_percentage": row["job_readiness_percentage"],
        "readiness_score": row["job_readiness_percentage"],
        "semesters": phases,
        "phases": phases,
        "completed_items": completed,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }


@app.post("/api/v1/roadmap/{student_id}/progress")
async def update_roadmap_progress_endpoint(student_id: str, request_body: Dict[str, Any]):
    """Cập nhật tiến độ hoàn thành checkbox trên AWS EC2 & tự động hòa trộn sang assessments."""
    completed_items = request_body.get("completed_items", {})
    readiness_score = float(request_body.get("readiness_score", 0.0))
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    completed_str = json.dumps(completed_items)

    conn = get_db()
    with conn:
        conn.execute("""
        UPDATE roadmaps
        SET completed_items = ?,
            job_readiness_percentage = ?,
            updated_at = ?
        WHERE student_id = ?;
        """, (completed_str, readiness_score, now_str, student_id))

        # Tự động hòa trộn sang bảng assessments nếu có
        try:
            r_row = conn.execute("SELECT target_major_id FROM roadmaps WHERE student_id = ?", (student_id,)).fetchone()
            a_row = conn.execute("SELECT assessment_data, target_career FROM assessments WHERE student_id = ?", (student_id,)).fetchone()
            if r_row and a_row:
                target_major_id = r_row["target_major_id"]
                data = json.loads(a_row["assessment_data"])
                matches = data.get("top_matches", [])
                for m in matches:
                    if m.get("major_id") == target_major_id:
                        m["match_percentage"] = round(readiness_score, 1)
                        m["match_score"] = round(readiness_score, 1)
                        sg = m.get("skill_gap", {})
                        raw_missing = sg.get("missing_skills", [])
                        raw_mastered = sg.get("mastered_skills", [])

                        if readiness_score >= 99.5:
                            all_s = raw_mastered + raw_missing
                            sg["mastered_skills"] = list(dict.fromkeys(all_s))
                            sg["missing_skills"] = []
                            for axis in m.get("radar_data", []):
                                axis["user_score"] = max(axis.get("benchmark_score", 8.0), 9.5)
                            # Cho lên hạng 1
                            matches.remove(m)
                            matches.insert(0, m)
                            for idx, item in enumerate(matches):
                                item["rank"] = idx + 1
                            data["skill_breakdown"] = sg
                            data["radar_chart_data"] = m.get("radar_data", [])
                        else:
                            total_s = max(len(raw_missing) + len(raw_mastered), 1)
                            frac = min(1.0, max(0.0, readiness_score / 100.0))
                            target_mastered_count = int(frac * total_s)
                            all_skills = list(dict.fromkeys(raw_mastered + raw_missing))
                            sg["mastered_skills"] = all_skills[:target_mastered_count]
                            sg["missing_skills"] = all_skills[target_mastered_count:]

                        m["skill_gap"] = sg
                        break
                data["top_matches"] = matches
                conn.execute(
                    "UPDATE assessments SET assessment_data = ?, updated_at = ? WHERE student_id = ?",
                    (json.dumps(data, ensure_ascii=False), now_str, student_id)
                )
        except Exception:
            pass

    conn.close()
    return {
        "status": "success",
        "student_id": student_id,
        "readiness_score": readiness_score,
        "saved_tier": "AWS_EC2_SQLite"
    }


# =============================================================================
# ENDPOINT NHÓM 2: CHUYỂN TIẾP TÍNH TOÁN GPU TỚI HPC NODE (FALLBACK AN TOÀN)
# =============================================================================

@app.post("/api/v1/chat/stream")
async def chat_stream_proxy(request: Request):
    """Proxy streaming token SSE từ mô hình Qwen 2.5 trên HPC Node."""
    body_bytes = await request.body()

    async def stream_generator():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{HPC_BASE_URL}/api/v1/chat/stream",
                    content=body_bytes,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status_code != 200:
                        yield f"data: {{\"token\": \"⚠️ Lỗi kết nối HPC: HTTP_{resp.status_code}\", \"done\": true}}\n\n"
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except Exception:
            yield "event: message\ndata: {\"token\": \"⚠️ Máy chủ tính toán AI (Legion HPC Node) hiện đang ngoại tuyến. Dữ liệu lộ trình và năng lực của bạn vẫn được bảo toàn đầy đủ trên đám mây AWS.\", \"done\": true}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.get("/api/v1/chat/history/{student_id}")
async def get_chat_history_cloud(student_id: str):
    """Lấy lịch sử hội thoại đã lưu từ CSDL SQLite trên AWS EC2."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT role, content, created_at
    FROM chat_history
    WHERE student_id = ?
    ORDER BY id ASC
    LIMIT 50;
    """, (student_id,))
    rows = cursor.fetchall()
    conn.close()

    return {
        "student_id": student_id,
        "messages": [
            {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
            for r in rows
        ]
    }


@app.post("/api/v1/chat/history/{student_id}")
async def save_chat_message_cloud(student_id: str, request: Request):
    """Lưu tin nhắn hội thoại vào CSDL SQLite trên AWS EC2."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    role = str(data.get("role", "user"))
    content = str(data.get("content", ""))
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if content:
        conn = get_db()
        with conn:
            conn.execute("""
            INSERT INTO chat_history (student_id, role, content, created_at)
            VALUES (?, ?, ?, ?);
            """, (student_id, role, content, now_str))
        conn.close()

    return {"status": "success", "student_id": student_id}


@app.post("/api/v1/roadmap/generate")
async def generate_roadmap_proxy(request: Request):
    """Chuyển tiếp tác vụ sinh lộ trình RAG tới HPC Node & lưu bản sao vào SQLite EC2."""
    body_bytes = await request.body()
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{HPC_BASE_URL}/api/v1/roadmap/generate",
                content=body_bytes,
                headers={"Content-Type": "application/json"}
            )
            if resp.status_code == 200:
                data = resp.json()
                # Tự động lưu lộ trình mới vào CSDL SQLite trên AWS EC2
                try:
                    student_id = data.get("student_id", "SV-2025001")
                    target_major_id = data.get("target_major_id", "DEVOPS_CLOUD")
                    target_major_name = data.get("target_major", "Kỹ thuật Đám mây & Vận hành Hệ thống")
                    readiness = float(data.get("job_readiness_percentage", 45.6))
                    phases = data.get("phases", data.get("semesters", []))

                    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    conn = get_db()
                    with conn:
                        conn.execute("""
                        INSERT INTO roadmaps (
                            student_id, target_major_id, target_major_name, job_readiness_percentage,
                            phases_data, completed_items, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(student_id) DO UPDATE SET
                            target_major_id = excluded.target_major_id,
                            target_major_name = excluded.target_major_name,
                            job_readiness_percentage = excluded.job_readiness_percentage,
                            phases_data = excluded.phases_data,
                            updated_at = excluded.updated_at;
                        """, (
                            student_id, target_major_id, target_major_name, readiness,
                            json.dumps(phases, ensure_ascii=False), json.dumps({}), now_str, now_str
                        ))
                    conn.close()
                except Exception:
                    pass
                return data
            else:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError:
        # Nếu HPC Node offline, kiểm tra xem đã có lộ trình trong CSDL SQLite EC2 chưa
        req_json = json.loads(body_bytes.decode()) if body_bytes else {}
        student_id = req_json.get("student_id", "SV-2025001")
        conn = get_db()
        row = conn.execute("SELECT phases_data FROM roadmaps WHERE student_id = ?", (student_id,)).fetchone()
        conn.close()
        if row:
            return await get_roadmap_endpoint(student_id)
        raise HTTPException(
            status_code=503,
            detail="Máy chủ tính toán AI HPC đang ngoại tuyến. Vui lòng thử lại sau khi HPC Node kết nối."
        )




@app.post("/api/v1/profile/upload-transcript")
async def upload_transcript_proxy(
    file: UploadFile = File(...),
    document_type: str = Form("transcript")
):
    """Proxy tải lên bảng điểm PDF tới HPC Node để giải mã và trích xuất PII."""
    file_bytes = await file.read()
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            files = {"file": (file.filename, file_bytes, file.content_type or "application/pdf")}
            data = {"document_type": document_type}
            resp = await client.post(
                f"{HPC_BASE_URL}/api/v1/profile/upload-transcript",
                files=files,
                data=data
            )
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Máy chủ bóc tách tài liệu AI HPC đang ngoại tuyến. Vui lòng kết nối HPC Node để xử lý file PDF."
        )
