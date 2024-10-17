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

app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your React app's URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

def run_llm(data):
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
        print(result)
    return result

@app.get("/process_sync")
async def process_sync_func_task():
    loop = asyncio.get_running_loop()

    # ThreadPoolExecutor를 사용하여 동기 메서드 실행
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, run_sync_func)
        print(result)
    return {"message": result}


@app.get("/process")
async def process_task():
    # 가상의 작업 처리 (비동기)
    await asyncio.sleep(5)  # 2초간 대기
    return {"message": "Task completed"}




# uvicorn server_fastapi:app --host 0.0.0.0 --port 20000


# ps aux | grep uvicorn
# pkill -f uvicorn