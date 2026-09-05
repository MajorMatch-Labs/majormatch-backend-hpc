# =============================================================================
# STAGE 1: Build Dependencies Stage
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Cài đặt các công cụ biên dịch tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Cài đặt Python packages vào thư mục wheels
RUN pip install --no-cache-dir --user -r requirements.txt


# =============================================================================
# STAGE 2: Production Runtime Stage
# =============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Thiết lập biến môi trường tối ưu Python và bảo mật
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH

# Cài đặt thư viện phụ thuộc hệ thống cho xử lý PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Tạo tài khoản non-root (appuser:1001) theo chuẩn an ninh
RUN useradd -u 1001 -m -s /bin/bash appuser && \
    mkdir -p /app /tmp/majormatch_ephemeral && \
    chown -R appuser:appuser /app /tmp/majormatch_ephemeral

# Sao chép dependencies từ builder sang runner
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Sao chép mã nguồn ứng dụng
COPY --chown=appuser:appuser . /app

# Chuyển quyền thực thi sang user không có quyền root
USER 1001

EXPOSE 8000

# Healthcheck kiểm tra trạng thái sống của dịch vụ
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
