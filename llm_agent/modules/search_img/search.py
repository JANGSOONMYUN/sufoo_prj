import sys
import os
import random
import re
import time
import base64
import asyncio
from typing import Any, AsyncGenerator, Dict, Optional

# 현재 파일의 경로
file_path = os.path.abspath(__file__)
# 현재 파일이 있는 디렉토리의 경로
directory_path = os.path.dirname(file_path)
# sys.path에 해당 경로 추가
if directory_path not in sys.path:
    sys.path.append(directory_path)
    
from search_img_api import search_and_download, search_and_download_one_to_path

_IMAGE_DEBUG = os.environ.get("IMAGE_DEBUG", "0") in ("1", "true", "True", "yes", "YES")

def _img_print(msg: str) -> None:
    if _IMAGE_DEBUG:
        print(f"[IMAGE] {msg}", flush=True)

def include_images(data):
    request_list = []
    if 'request' in data:
        request_list = data['request']
        
    download_dir = '/home/jsm/llm/sufoo_prj/llm_agent/images'
    # open_link_url = 'http://jsm0803.iptime.org:20000/images/'
    open_link_url = 'https://fodoit.com:20000/images/'
    
    for i, req in enumerate(request_list):
        try:
            if 'representative_image_name' not in req:
                continue
            image_name = req['representative_image_name']
            image_name_list = search_and_download(keyword=image_name, num_imgs=2, download_dir=download_dir)
            image_url = ''
            if len(image_name_list) > 0:
                # image_name_list에서 랜덤하게 하나의 이미지 이름을 선택합니다.
                random_image_name = random.choice(image_name_list)
                image_url = open_link_url + random_image_name
            request_list[i]['image_url'] = image_url
        except Exception as e:
            print(e)
    
    data['request'] = request_list
    return data


# 1x1 투명 PNG (placeholder) - URL을 먼저 내보내고, 실제 다운로드 완료 시 파일을 overwrite
_TRANSPARENT_1PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9WqYkAAAAASUVORK5CYII="
)

def _sanitize_filename_component(text: str, max_len: int = 60) -> str:
    t = (text or "").strip()
    if not t:
        return "image"
    # 파일명 안전 문자만 허용(한글/영문/숫자/_/-)
    t = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", t)
    t = t.strip("_")
    return (t[:max_len] or "image")

def _reserve_image_filename(keyword: str, ext: str = "jpg") -> str:
    base = _sanitize_filename_component(keyword)
    suffix = f"{int(time.time())}_{random.randint(1000, 9999)}"
    return f"{base}_{suffix}.{ext}"

def _write_placeholder_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 이미 있으면 그대로 사용(동시성 고려)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    try:
        with open(path, "wb") as f:
            f.write(_TRANSPARENT_1PX_PNG)
    except Exception:
        # placeholder 실패는 치명적이지 않음(ready 이벤트로 갱신 가능)
        pass

async def include_images_stream(
    representative_image_name: str,
    request_i: int,
    *,
    download_dir: str = "/home/jsm/llm/sufoo_prj/llm_agent/images",
    open_link_url: str = "https://fodoit.com:20000/images/",
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    representative_image_name(문자열 1개)을 받아 이미지 URL을 즉시 '예약'해서 내보내고,
    다운로드 완료 시 다시 'ready' 패치를 내보냅니다.

    - LLM 스트리밍과 분리된 비동기 흐름에서 사용하도록 설계(서버에서 asyncio.create_task로 실행 권장)
    - 반환 이벤트(권장):
      - {"request_{i}": {"image_url": "...", "image_status": "scheduled"}}
      - {"request_{i}": {"image_url": "...", "image_status": "ready"}}
      - 실패 시 {"request_{i}": {"image_status": "error", "image_error": "..."}}
    """
    keyword = (representative_image_name or "").strip()
    if not keyword:
        _img_print(f"skip: empty representative_image_name (request_i={request_i})")
        return

    filename = _reserve_image_filename(keyword, ext="jpg")
    local_path = os.path.join(download_dir, filename)
    _img_print(
        f"start: request_i={int(request_i)} keyword='{keyword}' filename='{filename}' "
        f"download_dir='{download_dir}' open_link_url='{open_link_url}'"
    )
    _write_placeholder_file(local_path)

    # 즉시 URL 예약(캐시 방지용 query 포함)
    scheduled_url = f"{open_link_url}{filename}?v=0"
    _img_print(f"emit scheduled: request_i={int(request_i)} url='{scheduled_url}'")
    yield {f"request_{int(request_i)}": {"image_url": scheduled_url, "image_status": "scheduled"}}

    # 실제 다운로드는 블로킹이므로 thread로 분리
    _img_print(f"download begin: request_i={int(request_i)} -> '{local_path}'")
    result = await asyncio.to_thread(search_and_download_one_to_path, keyword, local_path)
    if result.get("ok"):
        ready_url = f"{open_link_url}{filename}?v={int(time.time())}"
        _img_print(
            f"download ok: request_i={int(request_i)} source_url='{result.get('source_url', '')}' ready_url='{ready_url}'"
        )
        yield {f"request_{int(request_i)}": {"image_url": ready_url, "image_status": "ready"}}
    else:
        _img_print(f"download error: request_i={int(request_i)} err='{result.get('error', 'unknown error')}'")
        yield {
            f"request_{int(request_i)}": {
                "image_status": "error",
                "image_error": str(result.get("error", "unknown error")),
            }
        }
    _img_print(f"done: request_i={int(request_i)}")