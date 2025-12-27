# Web Developer Manual: 실시간 스트리밍 API (SSE)

이 문서는 `llm_agent` 프로젝트의 **실시간 스트리밍 API**를 웹(브라우저/React/Node)에서 쉽게 연동하기 위한 매뉴얼입니다.

---

## 개요

- **목표**: LLM이 생성하는 결과를 **완성될 때까지 기다리지 않고**, 생성되는 즉시 “바로바로” 수신
- **전송 방식**: **SSE(Server-Sent Events)** (`Content-Type: text/event-stream`)
- **엔드포인트**: `POST /llm_ver3`

서버는 내부적으로 LLM 스트리밍을 받으면, 이를 SSE 이벤트로 변환해 클라이언트로 푸시합니다.

---

## 서버 실행

기본(예시):

```bash
uvicorn server_fastapi:app --host 0.0.0.0 --port 20000
```

---

## API 스펙

### Endpoint

- **Method**: `POST`
- **Path**: `/llm_ver3`
- **Response**: `text/event-stream` (SSE)

### Request Body

요청은 반드시 아래 형태여야 합니다.

```json
{
  "content": "<JSON 문자열>"
}
```

`content`에는 아래 구조의 JSON을 **문자열로 직렬화**해서 넣습니다.

```json
{
  "question": "질문 내용",
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
        {"sub_title": "", "sub_description": "", "sub_result": ""},
        {"sub_title": "", "sub_description": "", "sub_result": ""}
      ]
    }
  ]
}
```

#### request 템플릿 규칙

- **`request`는 list**이며 길이는 고정이 아닙니다.
- 각 `request[i].subject` 또한 list이며 길이는 고정이 아닙니다.
- 서버/프롬프트는 **입력 템플릿의 개수(request/subject 길이)를 유지**한 채로, 빈 문자열 값들을 채워서 동일한 구조로 결과를 만듭니다.

---

## SSE 이벤트 타입

서버는 아래 이벤트를 보낼 수 있습니다.

- **`event: start`**: 스트림 시작
- **`event: chunk`**: 스트리밍 데이터(실시간)
- **`event: image`**: 이미지 처리 이벤트(LLM 스트림과 분리)
- **`event: end`**: 최종 결과(JSON)
- **`event: error`**: 오류
- **`event: finish`**: 내부 처리 종료(옵션)

### chunk 데이터 포맷(중요)

현재 설정(`settings/prompts/fodoit_stream/chains.json`)에서 `stream_emit_with_keys=true`이므로,
`event: chunk`의 `data`는 보통 아래처럼 **JSON 문자열(1개 객체)** 입니다.

예:

```json
{"request_0":{"title":"안"}}
```

```json
{"request_0":{"subject_0":{"sub_title":"부"}}}
```

즉, 클라이언트는 `chunk` 이벤트마다 `JSON.parse(data)`를 수행해,
어떤 필드가 업데이트되는지 **key 포함**으로 실시간 반영할 수 있습니다.

---

## image 이벤트(대표 이미지 비동기 처리)

LLM이 `request[i].representative_image_name` 값을 “완성”하는 순간, 서버는 별도의 비동기 작업으로 이미지를 검색/다운로드하고,
그 진행 상황을 **`event: image`**로 푸시합니다.  
이 이벤트는 **LLM의 `chunk` 스트림(JSON 패치)과 분리**되어 있으므로, LLM 파싱/렌더링을 방해하지 않습니다.

### image 데이터 포맷

`image`의 `data` 역시 patch 형태(JSON 문자열)이며, 예시는 아래와 같습니다.

예약(즉시):

```json
{"request_0":{"image_url":"https://fodoit.com:20000/images/비타민D_173...jpg?v=0","image_status":"scheduled"}}
```

다운로드 완료(나중):

```json
{"request_0":{"image_url":"https://fodoit.com:20000/images/비타민D_173...jpg?v=173...","image_status":"ready"}}
```

실패:

```json
{"request_0":{"image_status":"error","image_error":"검색 결과가 없습니다."}}
```

클라이언트는 `image_status === "ready"`를 받는 시점에 이미지 표시/갱신을 확정하면 가장 안정적입니다.

---

## 가장 쉬운 테스트(cURL)

SSE는 출력 버퍼링을 끄고(`-N`) 호출해야 “바로바로” 보입니다.

```bash
curl -N -X POST "http://localhost:20000/llm_ver3" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "{\"question\":\"고혈압\",\"additional_info_for_question\":\"\",\"client_info\":{\"gender\":\"여\",\"weight\":\"0\",\"height\":\"0\",\"bmi\":\"NaN\",\"health_conditions\":[],\"medications_being_taken\":[],\"supplements_being_taken\":[],\"special_conditions\":[]},\"request\":[{\"title\":\"\",\"description\":\"\",\"result\":\"\",\"subject\":[{\"sub_title\":\"\",\"sub_description\":\"\",\"sub_result\":\"\"},{\"sub_title\":\"\",\"sub_description\":\"\",\"sub_result\":\"\"}]}]}"
  }'
```

---

## 브라우저/React에서 사용 (POST SSE 권장 방식)

브라우저의 `EventSource`는 **GET만** 지원합니다.  
이 API는 `POST`이므로, 아래처럼 **`fetch()` + ReadableStream으로 SSE를 직접 파싱**하는 방식이 가장 현실적입니다.

### 1) SSE 파서 유틸 (fetch POST)

```javascript
/**
 * fetch(POST)로 SSE를 읽어서 onEvent로 전달합니다.
 * - 서버가 event/data 라인으로 SSE를 보내는 전제
 * - data 라인이 여러 줄일 수 있으므로 합쳐서 전달
 */
export async function postSSE(url, jsonBody, onEvent) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(jsonBody),
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${await res.text()}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let buffer = "";
  let currentEvent = "message";
  let currentDataLines = [];

  function flushEvent() {
    const data = currentDataLines.join("\n");
    if (data.length > 0 || currentEvent) {
      onEvent({ event: currentEvent, data });
    }
    currentEvent = "message";
    currentDataLines = [];
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE는 빈 줄(\n\n)로 이벤트를 구분
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const rawEventBlock = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      const lines = rawEventBlock.split("\n");
      for (const line of lines) {
        if (line.startsWith("event:")) {
          currentEvent = line.slice("event:".length).trim();
        } else if (line.startsWith("data:")) {
          currentDataLines.push(line.slice("data:".length).trimStart());
        }
      }
      flushEvent();
    }
  }
}
```

### 2) 실제 호출 예시 (chunk는 JSON parse)

```javascript
import { postSSE } from "./postSSE";

const API_BASE = "http://localhost:20000";

const payload = {
  question: "당뇨가 있는데 음식과 슈퍼푸드 그리고 영양제 추천해줘",
  additional_info_for_question: "",
  client_info: {
    gender: "남자",
    weight: "77.1",
    height: "177.5",
    bmi: "24.5",
    health_conditions: ["고혈압", "비만"],
    medications_being_taken: ["혈압약"],
    supplements_being_taken: ["종합비타민"],
    special_conditions: [],
  },
  request: [
    {
      title: "",
      description: "",
      result: "",
      subject: [
        { sub_title: "", sub_description: "", sub_result: "" },
        { sub_title: "", sub_description: "", sub_result: "" },
      ],
    },
  ],
};

// 서버는 {"content": "<json string>"}를 기대합니다.
const body = { content: JSON.stringify(payload) };

await postSSE(`${API_BASE}/llm_ver3`, body, ({ event, data }) => {
  if (event === "chunk") {
    // data는 {"request_0":{"title":"안"}} 같은 JSON 문자열
    const patch = JSON.parse(data);
    console.log("PATCH:", patch);
    // 여기서 UI state에 patch를 반영하면 "실시간" 업데이트 구현 가능
  } else if (event === "end") {
    const finalResult = JSON.parse(data);
    console.log("FINAL:", finalResult);
  } else if (event === "error") {
    console.error("ERROR:", data);
  }
});
```

---

## React에서 state에 patch 적용(예시)

`chunk`는 “문자 1개 단위” 패치가 매우 잦습니다.  
실제 UI에서는 다음 중 하나를 권장합니다.

- **(권장)** patch를 바로 state에 반영하되, 렌더링은 `requestAnimationFrame` 또는 50~100ms로 throttle
- patch를 누적한 뒤 일정 간격으로 merge


