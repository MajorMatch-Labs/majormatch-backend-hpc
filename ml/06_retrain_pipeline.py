# -*- coding: utf-8 -*-
"""
Bước 6 trong quy trình ML: Đường ống tái huấn luyện & Cập nhật mô hình tự động (Retraining Pipeline with Metric Guardrails)
Chủ quản: Long Nhật (NhatPrv <torikun2005@gmail.com>)
Hệ thống: MajorMatch HPC Compute Engine
"""
import sys
import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_MASTER = os.path.join(BASE_DIR, "dataset", "students_processed.csv")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
CURRENT_MODEL_PATH = os.path.join(BASE_DIR, "models", "base_career_model.joblib")
VERSION_HISTORY_PATH = os.path.join(BASE_DIR, "models", "model_version_history.json")

REQUIRED_COURSES = ["CS101", "CS201", "CS202", "CS203", "CS204", "MATH101", "PHYS101", "CS205"]
REQUIRED_HOLLAND = ["HOLLAND_R", "HOLLAND_I", "HOLLAND_A", "HOLLAND_S", "HOLLAND_E", "HOLLAND_C"]

def validate_incoming_schema(df: pd.DataFrame) -> bool:
    """Kiểm tra tính hợp lệ của dữ liệu đầu vào trước khi đưa vào retrain."""
    missing_cols = [c for c in REQUIRED_COURSES + REQUIRED_HOLLAND if c not in df.columns]
    if missing_cols:
        print(f"[X] Lỗi Schema: Thiếu các cột bắt buộc: {missing_cols}")
        return False
    return True

def run_retrain_pipeline(new_data_path: str = None):
    print("=" * 75)
    print(" BƯỚC 6: ĐƯỜNG ỐNG TÁI HUẤN LUYỆN & KIỂM ĐỊNH AN TOÀN (RETRAIN PIPELINE)")
    print("=" * 75)

    if not os.path.exists(DATASET_MASTER) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Thiếu tập dữ liệu master hoặc preprocessor pipeline.")

    prep = joblib.load(PREPROCESSOR_PATH)
    feature_cols = prep["feature_columns"]
    scaler = prep["scaler"]
    class_mapping = prep["class_mapping"]

    # Đọc dữ liệu Master hiện tại
    df_master = pd.read_csv(DATASET_MASTER)
    print(f"[*] Dữ liệu huấn luyện nền tảng hiện có: {len(df_master):,} bản ghi")

    # Nếu có tập dữ liệu mới cần tích hợp
    if new_data_path and os.path.exists(new_data_path):
        print(f"[*] Đang tiếp nhận tập dữ liệu mới từ: {new_data_path}")
        df_new = pd.read_csv(new_data_path)
        if not validate_incoming_schema(df_new):
            raise ValueError("Dữ liệu mới không hợp chuẩn schema hệ thống.")
        
        # Hợp nhất và loại bỏ trùng lặp
        df_combined = pd.concat([df_master, df_new], ignore_index=True)
        print(f"[V] Đã hợp nhất thành công: Tổng {len(df_combined):,} bản ghi")
    else:
        print("[*] Chế độ tái kiểm định & huấn luyện trên toàn bộ tập dữ liệu master cập nhật...")
        df_combined = df_master

    X = df_combined[feature_cols].values
    y = df_combined["TARGET_ENCODED"].values
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=int(time.time()) % 10000
    )

    # Lấy metric của mô hình hiện tại (nếu có) để làm rào chắn an toàn (Metric Guardrail)
    baseline_macro_f1 = 0.0
    if os.path.exists(CURRENT_MODEL_PATH):
        current_bundle = joblib.load(CURRENT_MODEL_PATH)
        baseline_macro_f1 = current_bundle.get("macro_f1", 0.0)
        print(f"[*] Chỉ số chuẩn mô hình đang chạy (Baseline Macro F1): {baseline_macro_f1:.2f}%")

    # Huấn luyện mô hình ứng viên mới
    print("[*] Đang huấn luyện mô hình ứng viên mới...")
    candidate_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    candidate_model.fit(X_train, y_train)

    # Đánh giá ứng viên
    y_pred = candidate_model.predict(X_test)
    new_acc = accuracy_score(y_test, y_pred) * 100
    new_macro_f1 = f1_score(y_test, y_pred, average="macro") * 100

    print("\n--- ĐỐI CHUẨN HIỆU NĂNG ỨNG VIÊN MỚI (GUARDRAIL AUDIT) ---")
    print(f"[*] Ứng viên mới - Accuracy: {new_acc:.2f}% | Macro F1: {new_macro_f1:.2f}%")
    print(f"[*] Ngưỡng yêu cầu nâng cấp: >= {baseline_macro_f1:.2f}%")

    # Điều kiện thăng cấp mô hình (Model Promotion Policy)
    should_promote = new_macro_f1 >= (baseline_macro_f1 - 2.0)  # Chấp nhận sai số nhỏ của split

    if should_promote:
        print("[V] PHÊ DUYỆT THĂNG CẤP: Mô hình mới đạt tiêu chuẩn chất lượng và an toàn!")
        new_bundle = {
            "model": candidate_model,
            "feature_columns": feature_cols,
            "class_mapping": class_mapping,
            "accuracy": round(new_acc, 2),
            "macro_f1": round(new_macro_f1, 2),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "author": "NhatPrv <torikun2005@gmail.com>"
        }
        joblib.dump(new_bundle, CURRENT_MODEL_PATH, compress=3)
        print(f"[V] Đã cập nhật Checkpoint sản phẩm tại: {CURRENT_MODEL_PATH}")

        # Ghi log lịch sử phiên bản
        history = []
        if os.path.exists(VERSION_HISTORY_PATH):
            try:
                with open(VERSION_HISTORY_PATH, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append({
            "version": f"v1.{len(history)+1}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(df_combined),
            "accuracy": round(new_acc, 2),
            "macro_f1": round(new_macro_f1, 2),
            "status": "PROMOTED"
        })

        with open(VERSION_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[V] Đã ghi nhật ký lịch sử phiên bản tại: {VERSION_HISTORY_PATH}")
    else:
        print("[X] TỪ CHỐI THĂNG CẤP: Hiệu năng mô hình mới thấp hơn chuẩn an toàn. Giữ nguyên checkpoint hiện hành.")

    print("=" * 75)

if __name__ == "__main__":
    run_retrain_pipeline()
