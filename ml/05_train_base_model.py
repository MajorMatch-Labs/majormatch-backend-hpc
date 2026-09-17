# -*- coding: utf-8 -*-
"""
Bước 5 trong quy trình ML: Huấn luyện mô hình cơ sở & Tinh chỉnh siêu tham số (Base Model Training & Tuning)
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PROCESSED = os.path.join(BASE_DIR, "dataset", "students_processed.csv")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "base_career_model.joblib")
REPORT_OUTPUT_PATH = os.path.join(REPORTS_DIR, "05_base_model_evaluation.json")

def train_base_model():
    print("=" * 75)
    print(" BƯỚC 5: HUẤN LUYỆN MÔ HÌNH CƠ SỞ & TINH CHỈNH SIÊU THAM SỐ (BASE MODEL TRAINING)")
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

    # Chuẩn hóa đặc trưng
    X_scaled = scaler.transform(X)

    # Phân chia Train / Test (80% train, 20% test, bảo toàn tỷ lệ lớp)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"[*] Tập dữ liệu huấn luyện: {X_train.shape[0]:,} mẫu")
    print(f"[*] Tập dữ liệu kiểm thử:    {X_test.shape[0]:,} mẫu")
    print(f"[*] Số lượng đặc trưng:      {len(feature_cols)} đặc trưng\n")

    # Thiết lập lưới siêu tham số có kiểm soát (Tối ưu hóa RAM 16GB và i9-13900HX)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }

    print("[*] Đang thực hiện tinh chỉnh siêu tham số qua GridSearchCV (3-Fold CV)...")
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        base_rf,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    tune_duration = time.time() - t0

    best_model = grid_search.best_estimator_
    print(f"\n[V] Tinh chỉnh hoàn tất sau: {tune_duration:.2f} giây")
    print(f"[*] Siêu tham số tối ưu: {grid_search.best_params_}")

    # Đánh giá trên tập kiểm thử độc lập
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 70)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET)")
    print("=" * 70)
    print(f"[*] Accuracy:    {acc * 100:.2f}%")
    print(f"[*] Macro F1:    {macro_f1 * 100:.2f}%")
    print(f"[*] Weighted F1: {weighted_f1 * 100:.2f}%\n")

    clf_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Đánh giá độ quan trọng của đặc trưng (Feature Importances)
    importances = best_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    print("\n--- TOP 8 ĐẶC TRƯNG CÓ ẢNH HƯỞNG LỚN NHẤT ĐẾN ĐỊNH HƯỚNG NGHỀ NGHIỆP ---")
    for rank, (fname, imp) in enumerate(feat_imp[:8], start=1):
        bar = "=" * int(imp * 100)
        print(f"  {rank}. {fname:<18}: {imp*100:>5.2f}% | {bar}")

    # Lưu checkpoint mô hình cơ sở
    base_model_bundle = {
        "model": best_model,
        "best_params": grid_search.best_params_,
        "feature_columns": feature_cols,
        "class_mapping": class_mapping,
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "author": "NhatPrv <torikun2005@gmail.com>",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(base_model_bundle, MODEL_OUTPUT_PATH, compress=3)
    print(f"\n[V] Đã lưu Checkpoint mô hình cơ sở tại: {MODEL_OUTPUT_PATH}")

    # Lưu báo cáo JSON
    report_bundle = {
        "test_size": len(y_test),
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "weighted_f1": round(weighted_f1 * 100, 2),
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "feature_importances": {k: round(float(v), 4) for k, v in feat_imp}
    }
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_bundle, f, ensure_ascii=False, indent=2)

    print(f"[V] Báo cáo chi tiết đã được lưu tại: {REPORT_OUTPUT_PATH}")
    print("=" * 75)

if __name__ == "__main__":
    train_base_model()
