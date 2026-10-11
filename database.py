"""
Module Quản trị Cơ sở Dữ liệu SQLite - MajorMatch Persistent Storage.
Lưu trữ hồ sơ sinh viên, kết quả đánh giá kỹ năng và cây lộ trình học tập theo giai đoạn.
Đảm bảo dữ liệu bền vững (ACID) không bị mất khi khởi động lại ứng dụng.
"""

import os
import sqlite3
import json
import time
from typing import Optional, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "majormatch.db")


def get_db_connection():
    """Tạo kết nối tới SQLite Database với chế độ WAL cho hiệu năng cao."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db():
    """Khởi tạo cấu trúc các bảng dữ liệu nếu chưa tồn tại."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Bảng lưu trữ cây lộ trình học tập cá nhân hóa theo giai đoạn
    cursor.execute("""
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

    # 2. Bảng lưu trữ kết quả phân tích khoảng cách kỹ năng (Skill Gap)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assessments (
        student_id TEXT PRIMARY KEY,
        target_career TEXT NOT NULL,
        assessment_data TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    # 3. Bảng lưu trữ hồ sơ bảng điểm và thiên hướng sinh viên
    cursor.execute("""
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

    # 4. Bảng lưu trữ lịch sử đoạn chat với trợ lý AI
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()


# =============================================================================
# CÁC THAO TÁC CRUD CHO LỘ TRÌNH (ROADMAP)
# =============================================================================

def save_roadmap(
    student_id: str,
    target_major_id: str,
    target_major_name: str,
    job_readiness: float,
    phases: List[Dict[str, Any]],
    completed_items: Optional[Dict[str, bool]] = None
) -> None:
    """Lưu mới hoặc cập nhật toàn bộ lộ trình học tập vào CSDL."""
    conn = get_db_connection()
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    completed_str = json.dumps(completed_items or {})
    phases_str = json.dumps(phases, ensure_ascii=False)

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
            student_id, target_major_id, target_major_name, job_readiness,
            phases_str, completed_str, now_str, now_str
        ))
    conn.close()


def get_roadmap(student_id: str) -> Optional[Dict[str, Any]]:
    """Truy vấn lộ trình đã lưu của sinh viên từ CSDL."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT student_id, target_major_id, target_major_name, job_readiness_percentage,
           phases_data, completed_items, created_at, updated_at
    FROM roadmaps
    WHERE student_id = ?;
    """, (student_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    try:
        phases = json.loads(row["phases_data"])
    except Exception:
        phases = []

    try:
        completed = json.loads(row["completed_items"])
    except Exception:
        completed = {}

    return {
        "student_id": row["student_id"],
        "target_major": row["target_major_name"],
        "target_major_id": row["target_major_id"],
        "job_readiness_percentage": row["job_readiness_percentage"],
        "readiness_score": row["job_readiness_percentage"],
        "semesters": phases,  # Tương thích ngược với UI
        "phases": phases,
        "completed_items": completed,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }


def update_roadmap_progress(
    student_id: str,
    completed_items: Dict[str, bool],
    readiness_score: float
) -> bool:
    """Cập nhật tiến độ hoàn thành các checkbox môn học/đồ án và điểm sẵn sàng vào CSDL & tự động đồng bộ sang bảng assessments."""
    conn = get_db_connection()
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    completed_str = json.dumps(completed_items)

    with conn:
        cursor = conn.execute("""
        UPDATE roadmaps
        SET completed_items = ?,
            job_readiness_percentage = ?,
            updated_at = ?
        WHERE student_id = ?;
        """, (completed_str, readiness_score, now_str, student_id))
        updated = cursor.rowcount > 0

        # Tự động hòa trộn tiến độ vào bảng assessments nếu có
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
                            # Đưa lên xếp hạng #1
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
    return updated


# =============================================================================
# CÁC THAO TÁC CRUD CHO ĐÁNH GIÁ KỸ NĂNG (ASSESSMENT)
# =============================================================================

def save_assessment(student_id: str, target_career: str, assessment_dict: Dict[str, Any]) -> None:
    """Lưu kết quả phân tích khoảng cách kỹ năng (Skill Gap) vào CSDL."""
    conn = get_db_connection()
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data_str = json.dumps(assessment_dict, ensure_ascii=False)

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


def get_assessment(student_id: str) -> Optional[Dict[str, Any]]:
    """Lấy kết quả đánh giá kỹ năng đã lưu gần nhất của sinh viên."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT student_id, target_career, assessment_data, updated_at
    FROM assessments
    WHERE student_id = ?;
    """, (student_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    try:
        data = json.loads(row["assessment_data"])
    except Exception:
        data = {}

    return {
        "student_id": row["student_id"],
        "target_career": row["target_career"],
        "assessment": data,
        "updated_at": row["updated_at"]
    }


# =============================================================================
# CÁC THAO TÁC CRUD CHO LỊCH SỬ CHAT (CHAT HISTORY)
# =============================================================================

def save_chat_message(student_id: str, role: str, content: str) -> None:
    """Lưu một tin nhắn hội thoại vào CSDL SQLite."""
    conn = get_db_connection()
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with conn:
        conn.execute("""
        INSERT INTO chat_history (student_id, role, content, created_at)
        VALUES (?, ?, ?, ?);
        """, (student_id, role, content, now_str))
    conn.close()


def get_chat_history(student_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Truy xuất lịch sử hội thoại của sinh viên theo thứ tự thời gian."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT role, content, created_at
    FROM chat_history
    WHERE student_id = ?
    ORDER BY id ASC
    LIMIT ?;
    """, (student_id, limit))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]


def clear_chat_history(student_id: str) -> None:
    """Xóa lịch sử chat của sinh viên."""
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM chat_history WHERE student_id = ?;", (student_id,))
    conn.close()

