# -*- coding: utf-8 -*-
"""
Bước 4 trong quy trình ML: Đánh giá & Tuyển chọn thuật toán dựa trên định hướng bài toán (Model Selection & Benchmark)
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PROCESSED = os.path.join(BASE_DIR, "dataset", "students_processed.csv")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
OUTPUT_BENCHMARK = os.path.join(REPORTS_DIR, "04_model_selection_benchmark.json")

def benchmark_algorithms():
    print("=" * 75)
    print(" BƯỚC 4: ĐÁNH GIÁ & ĐỐI SÁNH THUẬT TOÁN ĐỊNH HƯỚNG CHUYÊN NGÀNH CNTT")
    print("=" * 75)

    if not os.path.exists(DATASET_PROCESSED) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Chưa tìm thấy tập dữ liệu tiền xử lý hoặc preprocessor pipeline.")

    prep = joblib.load(PREPROCESSOR_PATH)
    feature_cols = prep["feature_columns"]
    scaler = prep["scaler"]

    df = pd.read_csv(DATASET_PROCESSED)
    X = df[feature_cols].values
    y = df["TARGET_ENCODED"].values

    # Chuẩn hóa X qua scaler đã học
    X_scaled = scaler.transform(X)

    print(f"[*] Kích thước tập huấn luyện & đối sánh: {X.shape[0]:,} mẫu x {X.shape[1]} đặc trưng")
    print("[*] Phương pháp kiểm định: Stratified 5-Fold Cross Validation (Bảo toàn tỷ lệ lớp)")
    print("[*] Tiêu chí định hướng:")
    print("    1. Dự đoán phân phối xác suất (predict_proba) cho 5 chuyên ngành.")
    print("    2. Cân bằng trọng số lớp (class_weight='balanced') nhằm bảo vệ nhóm thiểu số.")
    print("    3. Tối ưu hóa Macro F1-Score và tốc độ suy luận (Inference Latency).\n")

    # Danh mục các ứng viên thuật toán phù hợp định hướng
    candidates = {
        "Random Forest (Ensemble Bagging)": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Extra Trees (Randomized Forest)": ExtraTreesClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Hist Gradient Boosting (Tree Boosting)": HistGradientBoostingClassifier(
            max_iter=120, max_depth=8, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting (Traditional)": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "Logistic Regression (Multinomial L2)": LogisticRegression(
            multi_class="multinomial", max_iter=500, class_weight="balanced", random_state=42
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted"
    }

    results = []

    for name, model in candidates.items():
        print(f"[*] Đang đánh giá: {name}...")
        t0 = time.time()
        cv_res = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed_sec = time.time() - t0

        acc_mean = float(np.mean(cv_res["test_accuracy"]))
        b_acc_mean = float(np.mean(cv_res["test_balanced_accuracy"]))
        mf1_mean = float(np.mean(cv_res["test_macro_f1"]))
        wf1_mean = float(np.mean(cv_res["test_weighted_f1"]))

        # Đo độ trễ suy luận cho 100 mẫu đơn lẻ
        model.fit(X_scaled[:2000], y[:2000])
        t_infer_start = time.time()
        _ = model.predict_proba(X_scaled[:100])
        infer_latency_ms = (time.time() - t_infer_start) / 100.0 * 1000

        results.append({
            "algorithm": name,
            "accuracy": round(acc_mean * 100, 2),
            "balanced_accuracy": round(b_acc_mean * 100, 2),
            "macro_f1": round(mf1_mean * 100, 2),
            "weighted_f1": round(wf1_mean * 100, 2),
            "cv_duration_sec": round(elapsed_sec, 2),
            "latency_ms_per_sample": round(infer_latency_ms, 3)
        })

    # Xếp hạng theo Macro F1-Score
    results.sort(key=lambda x: x["macro_f1"], reverse=True)

    print("\n" + "=" * 80)
    print(f"{'Hạng':<5} | {'Thuật toán':<35} | {'Macro F1':<10} | {'Balanced Acc':<14} | {'Độ trễ/mẫu':<12}")
    print("-" * 80)
    for rank, res in enumerate(results, start=1):
        print(f"{rank:<5} | {res['algorithm']:<35} | {res['macro_f1']:>7.2f}%  | {res['balanced_accuracy']:>10.2f}%   | {res['latency_ms_per_sample']:>8.3f} ms")
    print("=" * 80)

    best_candidate = results[0]
    print(f"\n[V] Thuật toán tối ưu nhất cho bài toán MajorMatch: {best_candidate['algorithm']}")
    print(f"[*] Macro F1-Score: {best_candidate['macro_f1']}% | Balanced Accuracy: {best_candidate['balanced_accuracy']}%")

    with open(OUTPUT_BENCHMARK, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_metric": "Stratified 5-Fold Cross Validation",
            "leaderboard": results,
            "best_algorithm": best_candidate
        }, f, ensure_ascii=False, indent=2)

    print(f"[V] Báo cáo tuyển chọn thuật toán đã lưu tại: {OUTPUT_BENCHMARK}")
    print("=" * 75)

if __name__ == "__main__":
    benchmark_algorithms()
