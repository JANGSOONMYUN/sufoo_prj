from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import Response
import asyncio
import random
import time
import json
import functools
from gpt_api import LLMHandler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os


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
