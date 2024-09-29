# Gunakan image dasar Ubuntu dengan Python 3.9
FROM python:3.9-slim

# Non-interaktif mode untuk apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies dasar termasuk g++ dan libtool
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3-opencv \
    g++ \
    libtool \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone default (sesuaikan dengan timezone Anda)
RUN ln -fs /usr/share/zoneinfo/Asia/Jakarta /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

# Install pip dan CUDA versi yang sesuai
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install detectron2 dari source
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy seluruh project ke container
COPY . /app

# Pindah ke direktori kerja
WORKDIR /app

# Install semua dependencies dari requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Expose port baru (8000)
EXPOSE 8000

# Jalankan Flask di port 8000
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
