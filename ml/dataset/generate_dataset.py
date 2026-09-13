"""
Script sinh bộ dữ liệu sinh viên chuẩn hóa phục vụ huấn luyện mô hình Machine Learning.
Quy mô: 2.000 mẫu sinh viên với điểm số các môn đại cương và 6 chỉ số Holland RIASEC.
Phân bố dựa trên tương quan thực tế giữa năng lực học thuật và chuyên ngành tốt nghiệp.
"""

import os
import numpy as np
import pandas as pd

# Thiết lập Random Seed để đảm bảo tính tất định (Reproducibility)
np.random.seed(42)

NUM_SAMPLES = 2000

# Danh mục 5 chuyên ngành đào tạo mục tiêu
SPECIALIZATIONS = [
    "CS_DATA_AI",
    "SE_FULLSTACK",
    "DEVOPS_CLOUD",
    "CYBER_SECURITY",
    "DATA_ANALYTICS"
]

# Trọng số đặc trưng thiên hướng cho từng chuyên ngành (Mean scores)
ARCHETYPES = {
    "CS_DATA_AI": {
        # Toán & Thuật toán & AI cao
        "CS101": 3.6, "CS102": 3.5, "MTH100": 3.5, "MTH101": 3.7,
        "IT201": 3.3, "IT202": 2.8, "IT203": 3.0, "SE201": 3.1,
        "holland_r": 3.8, "holland_i": 4.6, "holland_a": 2.2, "holland_s": 2.4, "holland_e": 2.6, "holland_c": 3.5
    },
    "SE_FULLSTACK": {
        # OOP & Cấu trúc dữ liệu & CSDL cao
        "CS101": 3.5, "CS102": 3.6, "MTH100": 3.0, "MTH101": 2.9,
        "IT201": 3.5, "IT202": 3.1, "IT203": 3.1, "SE201": 3.8,
        "holland_r": 4.2, "holland_i": 3.9, "holland_a": 3.4, "holland_s": 2.9, "holland_e": 3.0, "holland_c": 3.8
    },
    "DEVOPS_CLOUD": {
        # Hệ điều hành, Mạng & Linux cao
        "CS101": 3.2, "CS102": 3.1, "MTH100": 2.8, "MTH101": 2.9,
        "IT201": 3.2, "IT202": 3.7, "IT203": 3.8, "SE201": 3.1,
        "holland_r": 4.7, "holland_i": 3.8, "holland_a": 1.8, "holland_s": 2.5, "holland_e": 2.7, "holland_c": 4.4
    },
    "CYBER_SECURITY": {
        # Mạng, Hệ điều hành, Thuật toán cao
        "CS101": 3.3, "CS102": 3.4, "MTH100": 3.2, "MTH101": 3.3,
        "IT201": 3.1, "IT202": 3.8, "IT203": 3.7, "SE201": 2.9,
        "holland_r": 4.4, "holland_i": 4.6, "holland_a": 1.7, "holland_s": 2.1, "holland_e": 2.6, "holland_c": 4.2
    },
    "DATA_ANALYTICS": {
        # CSDL, Xác suất thống kê, Tư duy kinh doanh cao
        "CS101": 3.2, "CS102": 2.9, "MTH100": 2.9, "MTH101": 3.6,
        "IT201": 3.8, "IT202": 2.6, "IT203": 2.7, "SE201": 2.8,
        "holland_r": 2.6, "holland_i": 4.1, "holland_a": 2.9, "holland_s": 3.4, "holland_e": 4.2, "holland_c": 4.6
    }
}

COURSES = ["CS101", "CS102", "MTH100", "MTH101", "IT201", "IT202", "IT203", "SE201"]
HOLLAND_TRAITS = ["holland_r", "holland_i", "holland_a", "holland_s", "holland_e", "holland_c"]

def generate_student_dataset(output_path: str):
    records = []
    samples_per_major = NUM_SAMPLES // len(SPECIALIZATIONS)

    student_counter = 1
    for major in SPECIALIZATIONS:
        archetype = ARCHETYPES[major]
        
        for _ in range(samples_per_major):
            student_id = f"VKU{220000 + student_counter}"
            student_counter += 1
            
            row = {"student_id": student_id}
            
            # 1. Sinh điểm môn học (phân phối chuẩn với độ lệch std 0.35, kẹp trong [1.0, 4.0])
            course_scores = []
            for course in COURSES:
                base_score = archetype[course]
                score = np.random.normal(loc=base_score, scale=0.35)
                # Thêm xác suất 3% sinh viên có điểm ngoại lệ ngẫu nhiên
                if np.random.rand() < 0.03:
                    score += np.random.choice([-0.8, 0.8])
                score = round(float(np.clip(score, 1.0, 4.0)), 2)
                row[course] = score
                course_scores.append(score)
            
            # Tính GPA tích lũy
            row["gpa_accumulated"] = round(float(np.mean(course_scores)), 2)
            
            # 2. Sinh điểm RIASEC (phân phối chuẩn với độ lệch std 0.45, kẹp trong [1.0, 5.0])
            for trait in HOLLAND_TRAITS:
                base_trait = archetype[trait]
                trait_score = np.random.normal(loc=base_trait, scale=0.45)
                trait_score = round(float(np.clip(trait_score, 1.0, 5.0)), 1)
                row[trait] = trait_score
            
            # Gán nhãn chuyên ngành mục tiêu
            row["target_specialization"] = major
            records.append(row)

    df = pd.DataFrame(records)
    
    # Trộn ngẫu nhiên các dòng dữ liệu
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Thiết lập stdout UTF-8 an toàn cho Windows console
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print(f"Dataset generated successfully at: {output_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nClass distribution:")
    print(df["target_specialization"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    dataset_file = os.path.join(os.path.dirname(__file__), "students_training.csv")
    generate_student_dataset(dataset_file)
