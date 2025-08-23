# nginx 파일 업로드 크기 제한 해결 가이드

## 문제
413 Request Entity Too Large 오류는 nginx의 파일 크기 제한 때문입니다.

## 해결 방법

### 1. nginx 설정 파일 찾기
```bash
# nginx 설정 파일 위치 확인
sudo nginx -t

# 일반적인 위치들
/etc/nginx/nginx.conf
/etc/nginx/sites-available/default
/etc/nginx/conf.d/default.conf
```

### 2. 설정 파일 수정
nginx.conf 또는 사이트별 설정 파일에 다음 추가:

```nginx
server {
    # 기존 설정들...
    
    # 파일 업로드 크기 제한을 20MB로 증가
    client_max_body_size 20M;
    
    # 또는 제한 없음 (권장하지 않음)
    # client_max_body_size 0;
}
```

### 3. nginx 재시작
```bash
# 설정 문법 확인
sudo nginx -t

# nginx 재시작
sudo systemctl restart nginx
# 또는
sudo service nginx restart
```

### 4. Docker를 사용하는 경우
docker-compose.yml에 nginx 설정 추가:

```yaml
version: '3'
services:
  nginx:
    image: nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    environment:
      - NGINX_CLIENT_MAX_BODY_SIZE=20M
```

## 임시 해결책
1MB 미만의 작은 이미지 파일로 먼저 테스트해보세요. 