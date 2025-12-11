# DEPLOYMENT INSTRUCTIONS

# 1. Base Image: NVIDIA TensorRT (Assessment İsteği)
# Bu imajın içinde CUDA, TensorRT ve Ubuntu hazır gelir.
FROM nvcr.io/nvidia/tensorrt:23.08-py3

# 2. Sistem ayarları
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. Gerekli Linux paketlerini kur (OpenCV için gerekli kütüphaneler)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. Çalışma dizinini ayarla
WORKDIR /app

# 5. Gereksinimleri kopyala ve kur
# Önce requirements.txt'yi kopyalıyoruz ki cache kullansın
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Tüm proje kodlarını kopyala
COPY . .

# 7. Modelleri kopyala (Opsiyonel: Büyük dosyalar genelde mount edilir ama burada ekliyoruz)
# NOT: .gitignore kullanıyorsanız burayı build sırasında dikkate almalısınız.
COPY models/ /app/models/

# 8. API Portunu dışarı aç
EXPOSE 8000

# 9. Başlatma Komutu
# Docker run edildiğinde API sunucusu otomatik başlar.
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]