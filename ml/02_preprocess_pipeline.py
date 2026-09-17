# -*- coding: utf-8 -*-
"""
Bước 2 trong quy trình ML: Tiền xử lý dữ liệu & Kỹ thuật đặc trưng (Preprocessing & Feature Engineering)
Chủ quản: Long Nhật (NhatPrv <torikun2005@gmail.com>)
Hệ thống: MajorMatch HPC Compute Engine
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_RAW = os.path.join(BASE_DIR, "dataset", "students_real.csv")
DATASET_PROCESSED = os.path.join(BASE_DIR, "dataset", "students_processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

# Danh sách môn học và nhóm trắc nghiệm gốc
RAW_ACADEMIC_COURSES = ["CS101", "CS201", "CS202", "CS203", "CS204", "MATH101", "PHYS101", "CS205"]
RAW_HOLLAND_TRAITS = ["HOLLAND_R", "HOLLAND_I", "HOLLAND_A", "HOLLAND_S", "HOLLAND_E", "HOLLAND_C"]
TARGET_COL = "TARGET_SPECIALIZATION"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng phái sinh có ý nghĩa quyết định trong định hướng chuyên ngành CNTT."""
    df = df.copy()

    # 1. Điểm trung bình phát triển phần mềm (Software Dev Score)
    df["AVG_DEV"] = df[["CS101", "CS201", "CS202"]].mean(axis=1).round(3)

    # 2. Điểm trung bình hệ thống & mạng (Systems & Infra Score)
    df["AVG_SYS"] = df[["CS203", "CS204", "CS205"]].mean(axis=1).round(3)

    # 3. Điểm tư duy định lượng & toán học (Quantitative Score)
    df["AVG_MATH_THEORY"] = df[["MATH101", "PHYS101"]].mean(axis=1).round(3)

    # 4. Tỷ số Nghiên cứu / Kinh doanh lãnh đạo (Investigative vs Enterprising Ratio)
    # Phân hóa rõ rệt giữa chuyên gia AI/R&D (I cao) và Quản lý/BI Analyst (E cao)
    df["RATIO_I_E"] = (df["HOLLAND_I"] / (df["HOLLAND_E"] + 1e-4)).round(3)

    # 5. Tỷ số Quy chuẩn dữ liệu / Nghệ thuật tự do (Conventional vs Artistic Ratio)
    # Phân hóa rõ rệt giữa Data Analytics/Cyber Security (C cao) và UI/UX/Web (A cao)
    df["RATIO_C_A"] = (df["HOLLAND_C"] / (df["HOLLAND_A"] + 1e-4)).round(3)

    return df

def run_preprocessing_pipeline():
    print("=" * 70)
    print(" BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU & KỸ THUẬT ĐẶC TRƯNG (FEATURE ENGINEERING)")
    print("=" * 70)

    if not os.path.exists(DATASET_RAW):
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu thực tế tại: {DATASET_RAW}")

    df = pd.read_csv(DATASET_RAW)
    print(f"[*] Đang tải dữ liệu thực tế: {len(df)} dòng...")

    # 1. Kỹ thuật đặc trưng (Feature Engineering)
    print("[*] Đang khởi tạo các đặc trưng phái sinh định hướng chuyên môn...")
    df_feat = engineer_features(df)
    
    # 2. Danh sách các đặc trưng đầu vào cuối cùng
    feature_columns = (
        RAW_ACADEMIC_COURSES
        + ["GPA"]
        + RAW_HOLLAND_TRAITS
        + ["AVG_DEV", "AVG_SYS", "AVG_MATH_THEORY", "RATIO_I_E", "RATIO_C_A"]
    )
    print(f"[*] Tổng số lượng đặc trưng huấn luyện sau kỹ thuật: {len(feature_columns)} đặc trưng")

    # 3. Kiểm định & Cắt tỉa biên giá trị hợp lệ (Value Clipping)
    for col in RAW_ACADEMIC_COURSES + ["GPA", "AVG_DEV", "AVG_SYS", "AVG_MATH_THEORY"]:
        df_feat[col] = df_feat[col].clip(2.0, 4.0)

    for col in RAW_HOLLAND_TRAITS:
        df_feat[col] = df_feat[col].clip(1.0, 5.0)

    # 4. Mã hóa nhãn mục tiêu (Label Encoding)
    label_encoder = LabelEncoder()
    df_feat["TARGET_ENCODED"] = label_encoder.fit_transform(df_feat[TARGET_COL])
    
    class_mapping = {int(idx): label for idx, label in enumerate(label_encoder.classes_)}
    print("\n--- BẢNG ÁNH XẠ NHÃN MỤC TIÊU (CLASS MAPPING) ---")
    for idx, label in class_mapping.items():
        print(f"  ID {idx}: {label}")

    # 5. Khởi tạo & Fit bộ chuẩn hóa đặc trưng (StandardScaler)
    scaler = StandardScaler()
    scaler.fit(df_feat[feature_columns])

    # 6. Đóng gói Artifact Preprocessor (để phục vụ inference production)
    preprocessor_bundle = {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "class_mapping": class_mapping,
        "academic_courses": RAW_ACADEMIC_COURSES,
        "holland_traits": RAW_HOLLAND_TRAITS,
        "author": "NhatPrv <torikun2005@gmail.com>"
    }
    joblib.dump(preprocessor_bundle, PREPROCESSOR_PATH)
    print(f"\n[V] Đã đóng gói và lưu Preprocessor Pipeline tại: {PREPROCESSOR_PATH}")

    # 7. Xuất file dữ liệu tiền xử lý (đã gắn feature mới & label encoded)
    export_cols = feature_columns + [TARGET_COL, "TARGET_ENCODED"]
    df_feat[export_cols].to_csv(DATASET_PROCESSED, index=False, encoding="utf-8")
    print(f"[V] Đã xuất tập dữ liệu tiền xử lý chuẩn mực tại: {DATASET_PROCESSED}")
    print(f"[*] Kích thước dữ liệu sẵn sàng cho huấn luyện: {df_feat[export_cols].shape}")
    print("=" * 70)

if __name__ == "__main__":
    run_preprocessing_pipeline()
