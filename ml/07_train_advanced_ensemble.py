# -*- coding: utf-8 -*-
"""
Bước 7 trong quy trình ML: Huấn luyện nâng cao kết hợp (Advanced Ensemble & Hybrid Model Architecture)
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
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PROCESSED = os.path.join(BASE_DIR, "dataset", "students_processed.csv")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

PRODUCTION_MODEL_PATH = os.path.join(MODELS_DIR, "career_classifier.joblib")
REPORT_OUTPUT_PATH = os.path.join(REPORTS_DIR, "07_advanced_ensemble_report.json")

def train_advanced_ensemble():
    print("=" * 75)
    print(" BƯỚC 7: HUẤN LUYỆN NÂNG CAO KẾT HỢP (ADVANCED ENSEMBLE ARCHITECTURE)")
    print("=" * 75)

    if not os.path.exists(DATASET_PROCESSED) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Chưa tìm thấy tập dữ liệu tiền xử lý hoặc preprocessor pipeline.")

    prep = joblib.load(PREPROCESSOR_PATH)
    feature_cols = prep["feature_columns"]
    scaler = prep["scaler"]
    class_mapping = prep["class_mapping"]
    target_names = [class_mapping[i] for i in sorted(class_mapping.keys())]

    df = pd.read_csv(DATASET_PROCESSED)
    X = df[feature_cols].values
    y = df["TARGET_ENCODED"].values

    # Chuẩn hóa đặc trưng qua StandardScaler
    X_scaled = scaler.transform(X)

    # Phân chia 80% Train, 20% Test (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"[*] Quy mô huấn luyện: {X_train.shape[0]:,} mẫu | Kiểm thử độc lập: {X_test.shape[0]:,} mẫu")
    print(f"[*] Số lượng đặc trưng phối hợp: {len(feature_cols)} đặc trưng\n")

    # Xây dựng các thành phần của hệ thống lai ghép (Hybrid / Ensemble)
    print("[*] Đang khởi tạo các mô hình cơ sở đa dạng phương pháp luận:")
    print("    1. Hist Gradient Boosting: Học sâu các biên quyết định phi tuyến và nhóm thiểu số.")
    print("    2. Random Forest (Bagging): Giảm phương sai, ổn định hóa dự báo.")
    print("    3. Logistic Regression: Chuẩn hóa phân phối xác suất tuyến tính (Probability Calibration).")

    clf_hgb = HistGradientBoostingClassifier(
        max_iter=150, max_depth=10, class_weight="balanced", random_state=42
    )
    clf_rf = RandomForestClassifier(
        n_estimators=180, max_depth=14, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf_lr = LogisticRegression(
        max_iter=500, class_weight="balanced", C=1.5, random_state=42
    )

    # Khởi tạo Soft Voting Classifier
    # Trọng số tối ưu dựa trên kết quả Model Selection (HGB: 2.0, RF: 1.5, LR: 1.0)
    ensemble_model = VotingClassifier(
        estimators=[
            ("hist_gb", clf_hgb),
            ("random_forest", clf_rf),
            ("logistic_reg", clf_lr)
        ],
        voting="soft",
        weights=[2.0, 1.5, 1.0],
        n_jobs=-1
    )

    print("\n[*] Đang tiến hành huấn luyện mô hình kết hợp Ensemble Soft Voting...")
    t0 = time.time()
    ensemble_model.fit(X_train, y_train)
    train_duration = time.time() - t0
    print(f"[V] Quá trình huấn luyện hoàn thành sau: {train_duration:.2f} giây")

    # Đánh giá toàn diện trên tập kiểm thử
    y_pred = ensemble_model.predict(X_test)
    y_proba = ensemble_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    loss = log_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 75)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH NÂNG CAO KẾT HỢP (ADVANCED ENSEMBLE EVALUATION)")
    print("=" * 75)
    print(f"[*] Accuracy:        {acc * 100:.2f}%")
    print(f"[*] Macro F1-Score:  {macro_f1 * 100:.2f}%")
    print(f"[*] Weighted F1:     {weighted_f1 * 100:.2f}%")
    print(f"[*] Cross-Entropy Loss (Log Loss): {loss:.4f}\n")

    clf_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Đóng gói Checkpoint sản xuất cuối cùng cho MajorMatch Backend
    production_bundle = {
        "model": ensemble_model,
        "scaler": scaler,
        "feature_columns": feature_cols,
        "class_mapping": class_mapping,
        "classes": target_names,
        "test_metrics": {
            "accuracy": round(acc * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "weighted_f1": round(weighted_f1 * 100, 2),
            "log_loss": round(loss, 4)
        },
        "author": "NhatPrv <torikun2005@gmail.com>",
        "model_type": "Hybrid Soft Voting Ensemble (HistGB + RF + LogReg)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(production_bundle, PRODUCTION_MODEL_PATH, compress=3)
    print(f"[V] Đã đóng gói CHECKPOINT SẢN XUẤT CUỐI CÙNG tại: {PRODUCTION_MODEL_PATH}")

    # Xuất báo cáo JSON
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "Hybrid Soft Voting Ensemble",
            "weights": [2.0, 1.5, 1.0],
            "test_size": len(y_test),
            "accuracy": round(acc * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "weighted_f1": round(weighted_f1 * 100, 2),
            "log_loss": round(loss, 4),
            "confusion_matrix": cm,
            "classification_report": clf_report
        }, f, ensure_ascii=False, indent=2)

    print(f"[V] Báo cáo thẩm định mô hình nâng cao đã lưu tại: {REPORT_OUTPUT_PATH}")
    print("=" * 75)

if __name__ == "__main__":
    train_advanced_ensemble()
