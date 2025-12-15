# Requirements
- python >= 3.9
- pytorch for CPU only
### pip install
- openai
- transformers
- websockets
- flask
- gunicorn
#### LangChain
- pip install langchain-community==0.2.9 langchain==0.2.9 langchain-core==0.2.22 langchain-openai==0.1.10
### Example
```
conda create -n openai python=3.9
conda activate openai
pip install openai transformers websockets pyinstaller flask flask_cors pip install Flask[async] gunicorn
conda install pytorch cpuonly -c pytorch
```

### Extra modules for server
sudo apt update
sudo apt install git

- anaconda (https://www.anaconda.com/products/individual)
```
sudo wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash [Anaconda*.sh]
source ~/.bashrc
```

# Configuration
- File name = [settings/config.json]
[deprecated]


# How to run
### Flask (Single connection for test) [deprecated]
```
export FLASK_APP=server_flask.py
flask run --host=0.0.0.0 --port=10000 

# windows
$env:FLASK_APP = "server_flask.py"
flask run --host=0.0.0.0 --port=10000 
```
### Gunicorn [deprecated]
```
gunicorn -w 1 --bind 0.0.0.0:10000 server_flask:app &
gunicorn -w 4 --bind 0.0.0.0:10000 server_flask:app &
```
- Close gunicorn process
```
ps -ef | grep gunicorn
pgrep -f "gunicorn"
kill [pid]
```
- Close all
```
pkill -f "gunicorn"
```
#### Kill processes
```
# example port number = 15000
# Install lsof
sudo apt install lsof
# Check processes occupying a port number
lsof -i :15000
# Kill corresponding processes
lsof -ti :15020 | xargs kill
# OR
lsof -i :15019 | awk 'NR!=1 {print $2}' | xargs kill -9
```

## RUN
### With SSL
```
sudo chmod -R a+r /home/jsm/ssl_keys

uvicorn server_fastapi:app --host 0.0.0.0 --port 20000   --ssl-keyfile=/home/jsm/ssl_keys/privkey.pem   --ssl-certfile=/home/jsm/ssl_keys/fullchain.pem   --ssl-ciphers='HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA'
```
### Without SSL
```
uvicorn server_fastapi:app --host 0.0.0.0 --port 20000
```

# API 호출 방법

## 엔드포인트

서버는 두 개의 주요 엔드포인트를 제공합니다:

1. **POST /llm** - ver1 (sufoo 처리)
2. **POST /llm_ver2** - ver2 (fodoit 처리)

## 요청 형식

모든 요청은 `content` 필드에 JSON 문자열을 포함해야 합니다.

### 요청 구조

```json
{
  "content": "{\"question\": \"질문 내용\", \"additional_info_for_question\": \"\", \"client_info\": {...}, \"request\": [...]}"
}
```

### 상세 요청 예시

```json
{
  "content": "{\"question\": \"당뇨가 있는데 음식과 슈퍼푸드 그리고 영양제 추천해줘\", \"additional_info_for_question\": \"\", \"client_info\": {\"gender\": \"남자\", \"weight\": \"77.1\", \"height\": \"177.5\", \"bmi\": \"24.5\", \"health_conditions\": [\"고혈압\", \"비만\"], \"medications_being_taken\": [\"혈압약\"], \"supplements_being_taken\": [\"종합비타민\"], \"special_conditions\": []}, \"request\": [{\"title\": \"\", \"description\": \"\", \"result\": \"\", \"subject\": [{\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}, {\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}]}, {\"title\": \"\", \"description\": \"\", \"result\": \"\", \"subject\": [{\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}, {\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}]}]}"
}
```

## API 호출 예시

### cURL 예시

#### ver1 (sufoo) 호출
```bash
curl -X POST "http://localhost:20000/llm" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "{\"question\": \"고혈압\", \"additional_info_for_question\": \"\", \"client_info\": {\"gender\": \"여\", \"weight\": \"0\", \"height\": \"0\", \"bmi\": \"NaN\", \"health_conditions\": [], \"medications_being_taken\": [], \"supplements_being_taken\": [], \"special_conditions\": []}, \"request\": [{\"title\": \"\", \"description\": \"\", \"result\": \"\", \"subject\": [{\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}, {\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}]}, {\"title\": \"\", \"description\": \"\", \"result\": \"\", \"subject\": [{\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}, {\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}]}]}"
  }'
```

#### ver2 (fodoit) 호출
```bash
curl -X POST "http://localhost:20000/llm_ver2" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "{\"question\": \"당뇨가 있는데 음식과 슈퍼푸드 그리고 영양제 추천해줘\", \"additional_info_for_question\": \"\", \"client_info\": {\"gender\": \"남자\", \"weight\": \"77.1\", \"height\": \"177.5\", \"bmi\": \"24.5\", \"health_conditions\": [\"고혈압\", \"비만\"], \"medications_being_taken\": [\"혈압약\"], \"supplements_being_taken\": [\"종합비타민\"], \"special_conditions\": []}, \"request\": [{\"title\": \"\", \"description\": \"\", \"result\": \"\", \"subject\": [{\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}, {\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}]}, {\"title\": \"\", \"description\": \"\", \"result\": \"\", \"subject\": [{\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}, {\"sub_title\": \"\", \"sub_description\": \"\", \"sub_result\": \"\"}]}]}"
  }'
```

### Python 예시

```python
import requests
import json

# 서버 URL
url = "http://localhost:20000/llm_ver2"  # 또는 "/llm" for ver1

# 요청 데이터 구조
data_payload = {
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

# content 필드에 JSON 문자열로 변환하여 전달
request_body = {
    "content": json.dumps(data_payload, ensure_ascii=False)
}

# API 호출
response = requests.post(url, json=request_body)

# 응답 확인
if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### JavaScript/TypeScript 예시

```javascript
// 서버 URL
const url = 'http://localhost:20000/llm_ver2'; // 또는 '/llm' for ver1

// 요청 데이터 구조
const dataPayload = {
  question: '당뇨가 있는데 음식과 슈퍼푸드 그리고 영양제 추천해줘',
  additional_info_for_question: '',
  client_info: {
    gender: '남자',
    weight: '77.1',
    height: '177.5',
    bmi: '24.5',
    health_conditions: ['고혈압', '비만'],
    medications_being_taken: ['혈압약'],
    supplements_being_taken: ['종합비타민'],
    special_conditions: []
  },
  request: [
    {
      title: '',
      description: '',
      result: '',
      subject: [
        {
          sub_title: '',
          sub_description: '',
          sub_result: ''
        },
        {
          sub_title: '',
          sub_description: '',
          sub_result: ''
        }
      ]
    },
    {
      title: '',
      description: '',
      result: '',
      subject: [
        {
          sub_title: '',
          sub_description: '',
          sub_result: ''
        },
        {
          sub_title: '',
          sub_description: '',
          sub_result: ''
        }
      ]
    }
  ]
};

// content 필드에 JSON 문자열로 변환하여 전달
const requestBody = {
  content: JSON.stringify(dataPayload)
};

// API 호출
fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(requestBody)
})
  .then(response => response.json())
  .then(data => {
    console.log('Success:', data);
  })
  .catch((error) => {
    console.error('Error:', error);
  });
```

## 요청 필드 설명

- **question**: 사용자의 질문 내용
- **additional_info_for_question**: 질문에 대한 추가 정보 (선택사항)
- **client_info**: 클라이언트 정보
  - **gender**: 성별
  - **weight**: 체중
  - **height**: 키
  - **bmi**: BMI 지수
  - **health_conditions**: 건강 상태 배열
  - **medications_being_taken**: 복용 중인 약물 배열
  - **supplements_being_taken**: 복용 중인 영양제 배열
  - **special_conditions**: 특수 조건 배열
- **request**: 요청 배열 (각 항목은 title, description, result, subject 배열 포함)

## 주의사항

- 서버는 기본적으로 포트 **20000**에서 실행됩니다
- SSL을 사용하는 경우 `https://` 프로토콜을 사용하세요
- `content` 필드는 반드시 JSON 문자열 형태로 전달해야 합니다
- CORS가 설정되어 있어 허용된 Origin에서만 접근 가능합니다

## (진짜) 스트리밍 API (SSE)

기존 `/llm`, `/llm_ver2`는 **응답을 한 번에 반환**합니다.  
실시간으로 “바로바로” 받으려면 **SSE 스트리밍 엔드포인트 `/llm_ver3`**를 사용하세요.

### SSE 호출 예시 (cURL)

```bash
curl -N -X POST "http://localhost:20000/llm_ver3" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "{\"question\":\"고혈압\",\"additional_info_for_question\":\"\",\"client_info\":{\"gender\":\"여\",\"weight\":\"0\",\"height\":\"0\",\"bmi\":\"NaN\",\"health_conditions\":[],\"medications_being_taken\":[],\"supplements_being_taken\":[],\"special_conditions\":[]},\"request\":[{\"title\":\"\",\"description\":\"\",\"result\":\"\",\"subject\":[{\"sub_title\":\"\",\"sub_description\":\"\",\"sub_result\":\"\"},{\"sub_title\":\"\",\"sub_description\":\"\",\"sub_result\":\"\"}]},{\"title\":\"\",\"description\":\"\",\"result\":\"\",\"subject\":[{\"sub_title\":\"\",\"sub_description\":\"\",\"sub_result\":\"\"},{\"sub_title\":\"\",\"sub_description\":\"\",\"sub_result\":\"\"}]}]}"
  }'
```

### SSE 이벤트 형식

- `event: chunk`: 생성되는 즉시 조각 텍스트가 내려옵니다.
- `event: end`: 최종 결과(JSON)가 내려옵니다.
- `event: error`: 오류 메시지가 내려옵니다.