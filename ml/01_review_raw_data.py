# -*- coding: utf-8 -*-
"""
Bước 1 trong quy trình ML: Khám phá & Đánh giá chất lượng dữ liệu thô (Exploratory Data Analysis - EDA)
Chủ quản: Long Nhật (NhatPrv <torikun2005@gmail.com>)
Hệ thống: MajorMatch HPC Compute Engine
"""
import sys
import os
import json
import pandas as pd
import numpy as np

# Đảm bảo terminal Windows hiển thị đúng tiếng Việt UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "students_real.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
OUTPUT_REPORT = os.path.join(REPORTS_DIR, "01_eda_raw_report.json")

def review_raw_dataset():
    print("=" * 70)
    print(" BƯỚC 1: KHÁM PHÁ & KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU THÔ (RAW DATA EDA)")
    print("=" * 70)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu thực tế tại: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    total_rows, total_cols = df.shape
    print(f"[*] Đường dẫn tập dữ liệu: {DATASET_PATH}")
    print(f"[*] Tổng số lượng mẫu: {total_rows:,} bản ghi")
    print(f"[*] Tổng số lượng đặc trưng: {total_cols} cột")
    print(f"[*] Dung lượng bộ nhớ: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n")

    # 1. Kiểm tra Missing Values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print("--- 1. KIỂM ĐỊNH GIÁ TRỊ KHUYẾT THIẾU (NULL / NAN) ---")
    if total_nulls == 0:
        print("[V] Dữ liệu hoàn hảo: 0% giá trị khuyết thiếu trên toàn bộ tập dữ liệu.")
    else:
        print(f"[!] Cảnh báo: Phát hiện {total_nulls} giá trị khuyết thiếu:")
        print(null_counts[null_counts > 0])

    # 2. Kiểm tra hàng trùng lặp (Duplicates)
    duplicates = df.duplicated().sum()
    print(f"\n--- 2. KIỂM ĐỊNH BẢN GHI TRÙNG LẶP ---")
    print(f"[*] Số lượng bản ghi trùng lặp tuyệt đối: {duplicates} ({duplicates/total_rows*100:.2f}%)")

    # 3. Phân tích phân phối biến mục tiêu (Target Distribution)
    print("\n--- 3. PHÂN PHỐI CHUYÊN NGÀNH MỤC TIÊU (TARGET DISTRIBUTION) ---")
    target_counts = df["TARGET_SPECIALIZATION"].value_counts()
    target_pct = df["TARGET_SPECIALIZATION"].value_counts(normalize=True) * 100
    target_stats = {}
    for spec in target_counts.index:
        count = int(target_counts[spec])
        pct = float(target_pct[spec])
        target_stats[spec] = {"count": count, "percentage": round(pct, 2)}
        bar = "#" * int(pct // 2)
        print(f"  - {spec:<18}: {count:>5} mẫu ({pct:>5.2f}%) | {bar}")

    # 4. Thống kê mô tả các biến định lượng (Descriptive Statistics)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[num_cols].describe().T
    desc["skewness"] = df[num_cols].skew()

    print("\n--- 4. THỐNG KÊ ĐẶC TRƯNG HỌC VỤ & HOLLAND (TÓM TẮT) ---")
    print(f"{'Đặc trưng':<12} | {'Min':<6} | {'Mean':<6} | {'Max':<6} | {'Std':<6} | {'Độ lệch (Skew)':<14}")
    print("-" * 65)
    for col in num_cols:
        row = desc.loc[col]
        print(f"{col:<12} | {row['min']:<6.2f} | {row['mean']:<6.2f} | {row['max']:<6.2f} | {row['std']:<6.2f} | {row['skewness']:<14.2f}")

    # 5. Phát hiện ngoại lai (Outlier Detection qua IQR)
    outlier_summary = {}
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = {
            "outlier_count": int(len(outliers)),
            "outlier_percentage": round(len(outliers) / total_rows * 100, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        }

    # Xuất báo cáo JSON
    report_data = {
        "dataset_path": DATASET_PATH,
        "total_records": total_rows,
        "total_features": total_cols,
        "missing_values_count": int(total_nulls),
        "duplicates_count": int(duplicates),
        "target_distribution": target_stats,
        "outlier_summary": outlier_summary
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n[V] Báo cáo thẩm định dữ liệu thô đã được lưu tại: {OUTPUT_REPORT}")
    print("=" * 70)

if __name__ == "__main__":
    review_raw_dataset()
