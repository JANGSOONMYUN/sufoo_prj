from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import Response, StreamingResponse
import asyncio
import random
import time
import json
import functools
from llm_handler import LLMHandler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from typing import Any, AsyncGenerator, Dict, Optional


app = FastAPI()

# Configure CORS
# 허용할 Origin 목록 (IP 주소 또는 웹 주소)
allowed_origins = [
    "http://localhost:3000",  # React 앱이 로컬에서 실행 중일 때
    "https://fodoit.com",
    "http://fodoit.com",
    "https://jsm0803.iptime.org",
    "http://jsm0803.iptime.org",
    "http://192.168.0.52",
    "http://192.168.0.18",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Add your React app's URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# 이미지 파일이 있는 폴더 경로
image_folder_path = "images"
# FastAPI에 정적 파일 라우터 추가
app.mount("/images", StaticFiles(directory=image_folder_path), name="images")
# http://jsm0803.iptime.org:20000/images/1_local_image.jpg


data = {
    "question": "당뇨가 있는데 음식과 슈퍼푸드 그리고 영양제 추천해줘",
    "additional_info_for_question": "",
    "client_info": {
        "gender": "남자",
        "weight": "77.1",
        "height": "177.5",
        "bmi": "24.5",
        "health_conditions": ["고혈압", "비만"],
        "medications_being_taken": ["혈압약"],
        "supplements_being_taken": ["종합비타민"],
        "special_conditions": []
    },
    "request": [
        {
            "title": "",
            "description": "",
            "result": "",
            "subject": [
                {
                    "sub_title": "",
                    "sub_description": "",
                    "sub_result": ""
                },
                {
                    "sub_title": "",
                    "sub_description": "",
                    "sub_result": ""
                }
            ]
        },
        {
            "title": "",
            "description": "",
            "result": "",
            "subject": [
                {
                    "sub_title": "",
                    "sub_description": "",
                    "sub_result": ""
                },
                {
                    "sub_title": "",
                    "sub_description": "",
                    "sub_result": ""
                }
            ]
        }
    ]
}

# Define the expected structure of the incoming JSON data
class DataModel(BaseModel):
    content: str  # Adjust this field name and type as needed


from concurrent.futures import ThreadPoolExecutor
def run_sync_func():
    import time
    time.sleep(2)
    return "Task completed"

def run_llm(data, selection='ver1'):
    llm = LLMHandler()
    if selection == 'ver1':
        llm_result = llm.handle_sufoo(data)
    elif selection == 'ver2':
        llm_result = llm.handle_fodoit(data)
    
    return llm_result

def run_llm_ver2(data):
    llm = LLMHandler()
    llm_result = llm.handle_fodoit(data)
    
    return llm_result

@app.post("/llm")
async def process_sync_llm(data: DataModel):
    loop = asyncio.get_running_loop()
    print('-----')
    print(type(data))
    print(type(data.content))
    print(data)
    json_data = json.loads(data.content)
    # run_llm 함수를 partial을 사용하여 data와 함께 호출
    run_llm_partial = functools.partial(run_llm, json.loads(data.content))
    # ThreadPoolExecutor를 사용하여 동기 메서드 실행
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, run_llm_partial)
        # print(result)
    return result


@app.post("/llm_ver2")
async def process_sync_llm_ver2(data: DataModel):
    loop = asyncio.get_running_loop()
    print('--llm_ver2llm_ver2llm_ver2llm_ver2---')
    print(type(data))
    print(type(data.content))
    print(data)
    json_data = json.loads(data.content)
    # run_llm 함수를 partial을 사용하여 data와 함께 호출
    run_llm_partial = functools.partial(run_llm_ver2, json.loads(data.content))
    # ThreadPoolExecutor를 사용하여 동기 메서드 실행
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, run_llm_partial)
        # print(result)
    return result

@app.post("/llm_ver3")
async def process_sync_llm_ver3(data: DataModel):
    """
    진짜 스트리밍(SSE) 엔드포인트.

    - 요청: 기존과 동일하게 {"content": "<json string>"} 형식
    - 응답: text/event-stream (SSE)
      - event: chunk  (data: <text>)
      - event: end    (data: <json>)  # 최종 결과
      - event: error  (data: <text>)
    """
    loop = asyncio.get_running_loop()
    payload = json.loads(data.content)

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        done_sentinel = object()

        def _sse(event: str, data_str: str) -> str:
            """
            SSE 포맷팅 유틸.
            data에 개행이 있으면 라인마다 'data: '를 붙여야 클라이언트에서 안전하게 파싱됩니다.
            """
            # SSE spec: 하나의 이벤트에 data 라인은 여러 줄 가능
            data_lines = str(data_str).splitlines() or [""]
            joined = "\n".join([f"data: {line}" for line in data_lines])
            return f"event: {event}\n{joined}\n\n"

        def _put_nowait_threadsafe(item: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        # LangChainModuleStream.simple_stream_handler는 (content, instance, func_name)로 호출하지만
        # 예외 케이스에선 (dict, instance, func_name, code)로도 호출하므로 code까지 받도록 함
        def _cb_func(content, instance, func_name, code: Optional[int] = None):
            if isinstance(content, dict) and "error_msg" in content:
                _put_nowait_threadsafe(("error", str(content.get("error_msg", ""))))
                return
            # content가 이미 JSON 문자열(키 포함 스트림)일 수 있으므로 그대로 전달
            if isinstance(content, (dict, list)):
                _put_nowait_threadsafe(("chunk", json.dumps(content, ensure_ascii=False)))
            else:
                _put_nowait_threadsafe(("chunk", str(content)))

        def _cb_start_end(is_start, instance, suffix, func_name):
            # 클라이언트가 원하면 start/end 이벤트로 사용할 수 있음
            _put_nowait_threadsafe(("start" if is_start else "finish", func_name))

        callback = {
            "func": _cb_func,
            "func_start_end": _cb_start_end,
            "instance": None,
            "func_name": "stream_report",
        }

        async def _run_llm_in_thread() -> None:
            def _sync_job() -> Dict[str, Any]:
                llm = LLMHandler()
                return llm.handle_fodoit_stream(payload, callback=callback)

            try:
                result = await loop.run_in_executor(None, _sync_job)
                _put_nowait_threadsafe(("end", json.dumps(result, ensure_ascii=False)))
            except Exception as e:
                _put_nowait_threadsafe(("error", f"{e}"))
            finally:
                _put_nowait_threadsafe(done_sentinel)

        # 백그라운드에서 LLM 실행 시작
        asyncio.create_task(_run_llm_in_thread())

        # SSE 스트림 시작 (프록시 타임아웃 방지용으로 먼저 보내기)
        yield _sse("start", "started")

        while True:
            item = await queue.get()
            if item is done_sentinel:
                break

            # item: ("chunk"|"end"|"error"|"start"|"finish", data)
            evt, evt_data = item
            if evt == "chunk":
                # 즉시 전송
                yield _sse("chunk", evt_data)
            elif evt == "end":
                yield _sse("end", evt_data)
            elif evt == "error":
                yield _sse("error", evt_data)
            elif evt == "start":
                yield _sse("start", evt_data)
            elif evt == "finish":
                yield _sse("finish", evt_data)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # nginx 등 리버스 프록시 버퍼링 방지
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)



# uvicorn server_fastapi:app --host 0.0.0.0 --port 20000

# ps aux | grep uvicorn
# pkill -f uvicorn


'''
# with ssl
uvicorn server_fastapi:app --host 0.0.0.0 --port 20000 \
  --ssl-keyfile=/home/jsm/ssl_keys/privkey.pem \
  --ssl-certfile=/home/jsm/ssl_keys/fullchain.pem \
  --ssl-ciphers="HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA"

uvicorn server_fastapi:app --host 0.0.0.0 --port 20000 \
  --ssl-keyfile=/home/jsm/ssl_keys/privkey.pem \
  --ssl-certfile=/home/jsm/ssl_keys/fullchain.pem \
  --ssl-ciphers='HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA'
'''
