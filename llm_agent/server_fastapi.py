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
from typing import Any, AsyncGenerator, Dict, Optional, Set


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
    llm_result = llm.handle_sufoo(data)
    
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
        scheduled_image_req_idxs: Set[int] = set()
        active_image_req_idxs: Set[int] = set()
        llm_done: bool = False
        image_debug: bool = os.environ.get("IMAGE_DEBUG", "0") in ("1", "true", "True", "yes", "YES")

        def _img_print(msg: str) -> None:
            if image_debug:
                print(f"[IMAGE][SSE] {msg}", flush=True)

        # 이미지 URL 구성(필요 시 환경변수로 오버라이드)
        # 예: https://fodoit.com:20000/images/
        open_link_url = os.environ.get("IMAGE_OPEN_LINK_URL", "https://fodoit.com:20000/images/")
        # 다운로드 디렉토리(StaticFiles("/images")와 같은 위치를 권장)
        download_dir = os.environ.get(
            "IMAGE_DOWNLOAD_DIR",
            "/home/jsm/llm/sufoo_prj/llm_agent/images",
        )
        _img_print(f"config open_link_url='{open_link_url}' download_dir='{download_dir}'")

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

        async def _handle_image_stream(req_i: int, rep_name: str) -> None:
            """
            LLM 스트리밍과 완전히 분리된 이미지 처리 파이프라인.
            - include_images_stream가 (scheduled -> ready/error) 패치를 yield
            - 이를 SSE event: image 로 클라이언트에 전송
            """
            try:
                # lazy import: 서버 부팅 시 import 문제/오버헤드 방지
                from modules.search_img.search import include_images_stream

                _img_print(f"task start: request_i={int(req_i)} rep_name='{rep_name}'")
                async for patch in include_images_stream(
                    rep_name,
                    req_i,
                    download_dir=download_dir,
                    open_link_url=open_link_url,
                ):
                    _img_print(f"emit event:image request_i={int(req_i)} patch={patch}")
                    _put_nowait_threadsafe(("image", json.dumps(patch, ensure_ascii=False)))
            except Exception as e:
                _img_print(f"task error: request_i={int(req_i)} err='{e}'")
                _put_nowait_threadsafe(
                    (
                        "image",
                        json.dumps(
                            {f"request_{int(req_i)}": {"image_status": "error", "image_error": str(e)}},
                            ensure_ascii=False,
                        ),
                    )
                )
            finally:
                # 이미지 작업이 모두 끝났고 LLM도 끝났으면 SSE 종료
                active_image_req_idxs.discard(int(req_i))
                _img_print(f"task done: request_i={int(req_i)} active={sorted(list(active_image_req_idxs))} llm_done={llm_done}")
                if llm_done and not active_image_req_idxs:
                    _put_nowait_threadsafe(done_sentinel)

        def _start_image_task(req_i: int, rep_name: str) -> None:
            """
            이벤트 루프 스레드에서 실행되어야 하는 함수.
            - active set 갱신
            - 이미지 async task 시작
            """
            active_image_req_idxs.add(int(req_i))
            _img_print(f"task scheduled: request_i={int(req_i)} rep_name='{rep_name}' active={sorted(list(active_image_req_idxs))}")
            asyncio.create_task(_handle_image_stream(int(req_i), rep_name))

        # LangChainModuleStream.simple_stream_handler는 (content, instance, func_name)로 호출하지만
        # 예외 케이스에선 (dict, instance, func_name, code)로도 호출하므로 code까지 받도록 함
        def _cb_func(content, instance, func_name, code: Optional[int] = None):
            if isinstance(content, dict) and "error_msg" in content:
                _put_nowait_threadsafe(("error", str(content.get("error_msg", ""))))
                return

            # (중요) agent_llm_stream.py가 대표 이미지명 완성 시점에 보내는 메타 이벤트를 가로채서
            # LLM chunk 스트림을 건드리지 않고, 별도 이미지 스트림을 시작한다.
            if isinstance(content, str):
                raw = content.strip()
                if raw.startswith("{") and '"_stream_meta"' in raw:
                    try:
                        meta_obj = json.loads(raw)
                        meta = meta_obj.get("_stream_meta") if isinstance(meta_obj, dict) else None
                        if isinstance(meta, dict) and meta.get("type") == "value_end":
                            path = meta.get("path") if isinstance(meta.get("path"), dict) else {}
                            if path.get("key") == "representative_image_name":
                                req_i = int(path.get("request_i", -1))
                                rep_name = str(meta.get("value", "")).strip()
                                _img_print(f"meta value_end: request_i={req_i} key=representative_image_name value='{rep_name}'")
                                if req_i >= 0 and rep_name and req_i not in scheduled_image_req_idxs:
                                    scheduled_image_req_idxs.add(req_i)
                                    _img_print(f"meta accepted -> start image task: request_i={req_i}")
                                    loop.call_soon_threadsafe(
                                        lambda: _start_image_task(req_i, rep_name)
                                    )
                                # 메타 이벤트는 chunk로 클라이언트에 보내지 않는다.
                                return
                    except Exception:
                        # 메타 파싱 실패는 무시하고 일반 chunk로 흘린다.
                        pass

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
                _put_nowait_threadsafe(("llm_done", "1"))

        # 백그라운드에서 LLM 실행 시작
        asyncio.create_task(_run_llm_in_thread())

        # SSE 스트림 시작 (프록시 타임아웃 방지용으로 먼저 보내기)
        yield _sse("start", "started")

        while True:
            item = await queue.get()
            if item is done_sentinel:
                break

            # done_sentinel이 아닌 경우 tuple로 가정
            if not isinstance(item, tuple) or len(item) != 2:
                continue

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
            elif evt == "image":
                yield _sse("image", evt_data)
            elif evt == "llm_done":
                llm_done = True
                _img_print(f"llm_done received; active_images={sorted(list(active_image_req_idxs))}")
                # 이미지 작업이 하나도 없으면 즉시 종료
                if not active_image_req_idxs:
                    break

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
