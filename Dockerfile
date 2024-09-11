# 기본 Ubuntu 이미지를 사용
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 필요한 패키지 설치
RUN apt-get update && \
    apt-get install -y \
    python3 \ 
    python3-pip \
    openssh-server && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 앱 파일 복사
COPY . .

# Flask와 Gunicorn 설치
RUN pip install -r requirements.txt

# lsof 설치
RUN apt-get update && \
    apt-get install -y lsof

# SSH 설정 (root 로그인 허용)
RUN mkdir /var/run/sshd
RUN echo 'root:password' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 포트 설정
EXPOSE 5002 22

# gunicorn과 sshd를 실행하는 스크립트 생성
CMD service ssh start && gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
