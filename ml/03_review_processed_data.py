# -*- coding: utf-8 -*-
"""
Bước 3 trong quy trình ML: Thẩm định chất lượng dữ liệu sau tiền xử lý (Post-Preprocessing Audit)
Chủ quản: Long Nhật (NhatPrv <torikun2005@gmail.com>)
Hệ thống: MajorMatch HPC Compute Engine
"""
import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PROCESSED = os.path.join(BASE_DIR, "dataset", "students_processed.csv")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
OUTPUT_AUDIT = os.path.join(REPORTS_DIR, "03_post_process_audit.json")

def audit_processed_dataset():
    print("=" * 70)
    print(" BƯỚC 3: THẨM ĐỊNH & ĐÁNH GIÁ DỮ LIỆU SAU TIỀN XỬ LÝ (POST-PROCESSING AUDIT)")
    print("=" * 70)

    if not os.path.exists(DATASET_PROCESSED) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Chưa tìm thấy tập dữ liệu tiền xử lý hoặc preprocessor pipeline.")

    prep = joblib.load(PREPROCESSOR_PATH)
    feature_cols = prep["feature_columns"]
    class_mapping = prep["class_mapping"]

    df = pd.read_csv(DATASET_PROCESSED)
    X = df[feature_cols]
    y = df["TARGET_ENCODED"]

    print(f"[*] Số lượng mẫu kiểm định: {len(df):,} bản ghi")
    print(f"[*] Số lượng đặc trưng đưa vào mô hình: {len(feature_cols)} đặc trưng")

    # 1. Kiểm tra tương quan cao (Multicollinearity / High Correlation check)
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] > 0.85]
        for idx, val in high_corr.items():
            high_corr_pairs.append({"feature_1": idx, "feature_2": col, "correlation": round(float(val), 3)})

    print("\n--- 1. KIỂM ĐỊNH ĐA CỘNG TUYẾN / TƯƠNG QUAN CAO (CORRELATION > 0.85) ---")
    if not high_corr_pairs:
        print("[V] Tuyệt vời: Không có cặp đặc trưng nào bị đa cộng tuyến nghiêm trọng.")
    else:
        print(f"[*] Phát hiện {len(high_corr_pairs)} cặp có tương quan tuyến tính cao (chủ yếu là đặc trưng phái sinh vs đặc trưng gốc):")
        for pair in high_corr_pairs:
            print(f"  - {pair['feature_1']:<18} <---> {pair['feature_2']:<18}: r = {pair['correlation']}")

    # 2. Đánh giá sức mạnh phân lớp (ANOVA F-statistic & Mutual Information)
    print("\n--- 2. ĐÁNH GIÁ ĐỘ PHÂN BIỆT ĐẶC TRƯNG (ANOVA F-VALUE & MUTUAL INFORMATION) ---")
    f_vals, p_vals = f_classif(X, y)
    
    # Tính Mutual Information (lấy mẫu 2000 dòng để tiết kiệm tài nguyên tính toán nhanh)
    sample_idx = np.random.RandomState(42).choice(len(X), size=min(2000, len(X)), replace=False)
    mi_vals = mutual_info_classif(X.iloc[sample_idx], y.iloc[sample_idx], random_state=42)

    feature_ranking = []
    for i, col in enumerate(feature_cols):
        feature_ranking.append({
            "feature": col,
            "f_value": round(float(f_vals[i]), 2),
            "p_value": float(p_vals[i]),
            "mutual_info": round(float(mi_vals[i]), 4)
        })

    # Sắp xếp theo F-value giảm dần
    feature_ranking.sort(key=lambda x: x["f_value"], reverse=True)

    print(f"{'Hạng':<5} | {'Đặc trưng':<20} | {'ANOVA F-Value':<15} | {'Mutual Information':<18}")
    print("-" * 65)
    for rank, item in enumerate(feature_ranking[:10], start=1):
        print(f"{rank:<5} | {item['feature']:<20} | {item['f_value']:<15} | {item['mutual_info']:<18}")

    # 3. Kiểm định rò rỉ dữ liệu (Data Leakage check)
    print("\n--- 3. KIỂM ĐỊNH RÒ RỈ DỮ LIỆU (DATA LEAKAGE AUDIT) ---")
    leakage_detected = False
    for col in feature_cols:
        if "TARGET" in col.upper() or "SPECIALIZATION" in col.upper():
            print(f"[X] NGUY HIỂM: Phát hiện cột rò rỉ nhãn: {col}")
            leakage_detected = True
    if not leakage_detected:
        print("[V] An toàn tuyệt đối: Không phát hiện rò rỉ nhãn mục tiêu vào tập đặc trưng.")

    # Xuất báo cáo
    audit_data = {
        "total_records": len(df),
        "total_features": len(feature_cols),
        "classes": class_mapping,
        "high_correlation_pairs": high_corr_pairs,
        "feature_ranking": feature_ranking,
        "data_leakage": leakage_detected
    }

    with open(OUTPUT_AUDIT, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, ensure_ascii=False, indent=2)

    print(f"\n[V] Báo cáo thẩm định dữ liệu sau tiền xử lý đã lưu tại: {OUTPUT_AUDIT}")
    print("=" * 70)

if __name__ == "__main__":
    audit_processed_dataset()
