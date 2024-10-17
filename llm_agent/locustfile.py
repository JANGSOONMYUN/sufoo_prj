from locust import HttpUser, between, task, main

class FastAPIUser(HttpUser):
    wait_time = between(1, 5)  # 각 사용자가 요청 사이에 1~5초 대기
    host = "http://localhost:20000"  # 서버가 실행 중인 호스트 및 포트 설정
    @task
    def process_task(self):
        self.client.get("/process_sync")  # FastAPI 엔드포인트에 GET 요청

if __name__ == "__main__":
    main.main()


# locust -f client_test.py