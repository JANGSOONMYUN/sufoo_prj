# 워크플로우 프레임워크 문서

## 1. 개요

워크플로우 프레임워크는 LLM(Large Language Model)을 활용한 대화형 에이전트를 구성하기 위한 설정 기반 시스템입니다. 이 프레임워크는 다양한 타입의 노드를 연결하여 복잡한 대화 흐름을 구성할 수 있으며, 조건부 실행, 병렬 처리, 도구 함수 호출 등 다양한 기능을 제공합니다.

## 2. 시스템 변수

시스템 변수는 워크플로우의 상태를 관리하기 위한 전역 변수입니다.

### 기본 시스템 변수
- `conv_count`: 대화 카운터
- `current_node`: 현재 실행 중인 노드
- `next_node`: 다음 실행할 노드
- `val1 ~ val6`: 시스템 처리용 변수 (임의로 사용 가능)

### 출력 구성 타입
1. **텍스트 출력**
   - 형식: 문자열 스트리밍
   - 저장: 대화 이력에 저장 가능

2. **구조화된 출력**
   - 형식: JSON
   - 구성: 사용자 정의 구조 (제목, 내용, 하위 항목 등)
   - 저장: 대화 이력에 저장 가능

## 3. 워크플로우 데이터 구조

### 최상위 구조
```json
{
    "agent_name": "에이전트 이름",
    "description": "에이전트 설명",
    "variables": {
        "conv_count": 0,
        "current_node": "",
        "next_node": "",
        "사용자정의변수1": 초기값,
        "사용자정의변수2": 초기값
    },
    "workflows": [...]
}
```

### 변수 구조
워크플로우에서 사용되는 전역 변수들로, 모든 노드에서 접근 및 수정이 가능합니다.

```python
"variables": {
    # 필수 시스템 변수
    "conv_count": int,  # 대화 진행 카운터
    "current_node": str,  # 현재 실행 중인 노드
    "next_node": str,  # 다음 실행할 노드

    # 사용자 정의 변수 (워크플로우별 커스텀 가능)
    "custom_var1": Any,  # from typing import Any
    "custom_var2": Any,
    ...
}
```

### 변수 특징
1. **필수 시스템 변수**
   - `conv_count`: 대화 진행 상태 추적
   - `current_node`: 현재 실행 중인 노드 식별
   - `next_node`: 다음 실행할 노드 지정
   - 모든 워크플로우에서 반드시 포함

2. **사용자 정의 변수**
   - 워크플로우 설계 시 자유롭게 추가 가능
   - 다양한 데이터 타입 지원
   - 변수명과 타입은 워크플로우 설계자가 결정
   - 실행 중 동적으로 값 변경 가능

### 워크플로우 구조
```python
"workflows": [
    {
        "workflow_id": str,  # 워크플로우 식별자
        "name": str,  # 워크플로우 이름
        "description": str,  # 워크플로우 설명
        "start_node": list[str],  # 시작 노드 목록
        "nodes": dict  # 노드 정의
    }
]
```

## 4. 노드 타입 및 구조

### 공통 노드 구조
모든 노드는 다음과 같은 공통 구조를 가집니다:

```python
"node_name": {
    "node_id": str,  # 노드 식별자
    "processing_type": str,  # 처리 타입 (llm, api, logic, function)
    "content": str,  # 노드 내용 (사용자에게 표시되는 메시지)
    "instructions": list[str],  # LLM에게 전달되는 지시사항
    "prompts": list[str],  # LLM에게 전달되는 프롬프트
    
    # 입력 변수 정의
    "input_variables": {
        "variable_name": {
            "type": str,  # 변수 타입 (예: "string", "int", "list" 등)
            "default_value": Any,  # 기본값 (선택적)
            "description": str  # 변수 설명 (선택적)
        }
    },
    
    # LLM을 위한 출력 템플릿 구조
    "output_variables": {
        "variable_name": {
            "type": str,  # 예상되는 출력 타입
            "description": str,  # LLM에게 제공되는 출력 형식 설명
            "items": dict,  # 리스트/객체인 경우 하위 구조 정의
            "properties": dict  # 객체인 경우 속성 정의
        }
    },
    
    "output_filter": str,  # 출력 필터링 방식
    "history": {
        "save": bool,  # 대화 이력 저장 여부
        "use": bool,  # 대화 이력 사용 여부
        "configurable": {
            "user_id": str,  # 사용자 ID
            "conversation_id": str  # 대화 ID
        }
    },
    
    # 출력 형식 정의
    "output_config": {
        # LLM API 출력 형식
        "llm_to_server": {
            "use_streaming": bool,  # 스트리밍 사용 여부
            "streaming_type": str,  # 스트리밍 시 데이터 타입
            "output_type": str  # 최종 출력 데이터 타입 (예: "str", "json")
        },
        # 클라이언트 전달 형식
        "server_to_client": {
            "use_return": bool,  # 반환값 사용 여부
            "use_streaming": bool,  # 클라이언트 스트리밍 여부
            "streaming_type": str,  # 클라이언트 스트리밍 데이터 타입
            "output_type": str  # 클라이언트 최종 출력 타입
        }
    },
    "next_node": dict  # 다음 노드 정의
}
```

### 노드 처리 타입
1. **LLM 노드 (processing_type: "llm")**
   - LLM을 호출하여 응답을 생성
   - 지시사항과 프롬프트를 통해 LLM 동작 제어
   - 구조화된 출력 지원

2. **함수 노드 (processing_type: "function")**
   - 사전 정의된 도구 함수 호출
   - 변수 계산, 조건 평가 등에 활용
   - tool_name을 통해 호출할 함수 지정

3. **API 노드 (processing_type: "api")**
   - 외부 API 호출
   - 데이터 조회 및 처리

4. **로직 노드 (processing_type: "logic")**
   - 제어 흐름 관리
   - 조건에 따른 분기 처리

5. **게이트웨이 노드 (processing_type: "gateway")**
   - 분기점 역할
   - 조건에 따라 다음 노드 결정

### 병렬 처리 노드
병렬 처리 노드는 일반 노드 구조에 추가적인 분기 설정을 포함합니다:

```python
"branch": {
    "use_branch": bool,  # 분기 사용 여부
    "branch_rule": {
        "data_type": str,  # 데이터 타입
        "in_depth_path": list[str],  # 입력 경로
        "out_depth_path": list[str],  # 출력 경로
        "in_out_set": list[dict]  # 입출력 매핑
    }
}
```

## 5. 조건부 실행 (Conditions)

### 기본 구조
조건부 실행을 위한 기본 구조입니다:

```json
"conditions": [
    {
        "condition": {
            // 조건식 정의
        },
        "actions": [
            // 실행할 액션들
        ]
    }
]
```

### 조건식 작성 방법

#### 단일 조건
```json
"condition": {
    "변수명": { 
        "operator": "연산자", 
        "operand": "비교값" 
    }
}
```

#### AND 조건
```json
"condition": {
    "and": [
        {
            "변수명1": { "operator": "연산자1", "operand": "값1" }
        },
        {
            "변수명2": { "operator": "연산자2", "operand": "값2" }
        }
    ]
}
```

#### OR 조건
```json
"condition": {
    "or": [
        {
            "변수명1": { "operator": "연산자1", "operand": "값1" }
        },
        {
            "변수명2": { "operator": "연산자2", "operand": "값2" }
        }
    ]
}
```

#### 복합 조건 (AND + OR)
```json
"condition": {
    "or": [
        {
            "and": [
                {
                    "변수명1": { "operator": "연산자1", "operand": "값1" }
                },
                {
                    "변수명2": { "operator": "연산자2", "operand": "값2" }
                }
            ]
        },
        {
            "변수명3": { "operator": "연산자3", "operand": "값3" }
        }
    ]
}
```

### 사용 가능한 연산자
- `==`: 같음
- `!=`: 다름
- `>`: 큼
- `<`: 작음
- `>=`: 크거나 같음
- `<=`: 작거나 같음

### 액션 타입

#### 함수 호출
```json
{
    "type": "function",
    "name": "함수명",
    "params": {
        "param1": "값1",
        "param2": { "variable": "변수명" },  // 변수 참조
        "param3": 123  // 직접 값
    }
}
```

#### 변수 설정
```json
{
    "type": "variable",
    "name": "변수명",
    "value": "설정할 값"  // boolean, number, string 등
}
```

## 6. 도구 함수 (Tools)

워크플로우에서 사용할 수 있는 주요 도구 함수입니다:

### condition_evaluator
조건식을 평가하고 해당하는 액션을 실행합니다.
```python
@register_tool("condition_evaluator")
def condition_evaluator(state: Dict[str, Any], *, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
```

### basic_operations
다양한 연산을 수행하는 도구입니다.
```python
@register_tool("basic_operations")
def basic_operations(state: Dict[str, Any], *,
                    operation: str = "",
                    expression: str = "",
                    target_variable: str = "",
                    operations: List[Dict[str, Any]] = None,
                    operands: List[Any] = None,
                    **kwargs) -> Dict[str, Any]:
```

#### basic_operations 사용 가능한 연산 타입:
- `add`: 덧셈
- `subtract`: 뺄셈
- `multiply`: 곱셈
- `divide`: 나눗셈
- `increment`: 변수 값 증가
- `decrement`: 변수 값 감소
- `modulo`: 나머지 계산
- `power`: 거듭제곱
- `max`: 최대값
- `min`: 최소값
- `expression`: 단일 수학 표현식 계산
- `compound`: 여러 연산을 순차적으로 수행

## 7. 입출력 구성

### 입력 변수 (input_variables)
노드로 입력되는 변수들의 구조를 정의합니다.
```json
"input_variables": {
    "변수명": {
        "type": "변수타입",
        "default_value": "기본값",
        "description": "변수 설명"
    }
}
```

### 출력 변수 (output_variables)
LLM이 생성해야 할 출력의 템플릿 구조를 정의합니다.
```json
"output_variables": {
    "변수명": {
        "type": "변수타입",
        "description": "변수 설명",
        "items": {},  // 리스트인 경우 항목 타입
        "properties": {}  // 객체인 경우 속성
    }
}
```

### 출력 설정 (output_config)
LLM 및 클라이언트 출력 형식을 정의합니다.
```json
"output_config": {
    "llm_to_server": {
        "use_streaming": true/false,
        "streaming_type": "str/json",
        "output_type": "str/json"
    },
    "server_to_client": {
        "use_return": true/false,
        "use_streaming": true/false,
        "streaming_type": "str/json",
        "output_type": "str/json"
    }
}
```

## 8. 히스토리 관리
대화 이력 관리에 대한 설정을 정의합니다.
```json
"history": {
    "save": true/false,  // 대화 이력 저장 여부
    "use": true/false,  // 대화 이력 사용 여부
    "configurable": {
        "user_id": "사용자ID",
        "conversation_id": "대화ID"
    }
}
```

## 9. 다음 노드 설정
다음에 실행할 노드를 정의합니다.
```json
"next_node": {
    "노드이름": {"group": "그룹ID", "type": "node"}
}
```

## 10. 고객센터 예시 워크플로우

아래는 고객센터 챗봇을 위한 예시 워크플로우입니다:

```json
{
    "agent_name": "고객센터 챗봇",
    "description": "고객 문의 처리를 위한 대화형 챗봇",
    "variables": {
        "conv_count": 0,
        "current_node": "",
        "next_node": "",
        "customer_id": "",
        "ticket_id": "",
        "issue_category": "",
        "issue_priority": "",
        "resolution_status": ""
    },
    "workflows": [
        {
            "workflow_id": "customer_service_flow",
            "name": "고객센터 메인 흐름",
            "description": "고객 문의 처리 워크플로우",
            "start_node": ["issue_categorizer"],
            "nodes": {
                "issue_categorizer": {
                    "node_id": "issue_categorizer",
                    "processing_type": "llm",
                    "content": "문의 사항을 분석 중입니다.",
                    "instructions": [
                        "당신은 고객센터 문의를 분류하는 전문가입니다.",
                        "고객 문의를 분석하여 주요 카테고리와 하위 카테고리로 분류하세요.",
                        "가능한 카테고리: 제품 문의, 기술 지원, 배송 문제, 결제 문제, 환불 요청, 기타"
                    ],
                    "prompts": [
                        "고객의 문의 내용을 분석하여 적절한 카테고리로 분류해주세요.",
                        "주요 문제점과 우선순위를 식별해주세요.",
                        "{user_message}"
                    ],
                    "input_variables": {
                        "user_message": {"type": "string", "default_value": ""}
                    },
                    "output_variables": {
                        "issue_analysis": {
                            "type": "object",
                            "description": "고객 문의 분석 결과",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "description": "문의 카테고리"
                                },
                                "sub_category": {
                                    "type": "string",
                                    "description": "문의 하위 카테고리"
                                },
                                "priority": {
                                    "type": "string",
                                    "description": "문의 우선순위 (높음, 중간, 낮음)"
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "문의 내용 요약"
                                }
                            }
                        }
                    },
                    "output_filter": "all",
                    "history": {
                        "save": true,
                        "use": true,
                        "configurable": {
                            "user_id": "customer_id",
                            "conversation_id": "ticket_id"
                        }
                    },
                    "output_config": {
                        "llm_to_server": {
                            "use_streaming": false,
                            "streaming_type": "str",
                            "output_type": "json"
                        },
                        "server_to_client": {
                            "use_return": true,
                            "use_streaming": false,
                            "streaming_type": "str",
                            "output_type": "json"
                        }
                    },
                    "next_node": {
                        "ticket_creator": {"group": "group_0", "type": "node"}
                    }
                },
                "ticket_creator": {
                    "node_id": "ticket_creator",
                    "processing_type": "function",
                    "tool_name": "basic_operations",
                    "content": "문의 티켓을 생성 중입니다.",
                    "operation": "compound",
                    "operations": [
                        {
                            "type": "expression",
                            "target_variable": "ticket_id",
                            "expression": "'TKT-' + str(round(1000 + random() * 9000))"
                        },
                        {
                            "type": "variable",
                            "target_variable": "issue_category",
                            "from_variable": "issue_analysis.category"
                        },
                        {
                            "type": "variable",
                            "target_variable": "issue_priority",
                            "from_variable": "issue_analysis.priority"
                        }
                    ],
                    "input_variables": {
                        "issue_analysis": {"type": "object", "default_value": {}}
                    },
                    "output_variables": {
                        "ticket_id": {"type": "string", "description": "생성된 티켓 ID"},
                        "issue_category": {"type": "string", "description": "문의 카테고리"},
                        "issue_priority": {"type": "string", "description": "문의 우선순위"}
                    },
                    "output_filter": "all",
                    "output_config": {
                        "server_to_client": {
                            "use_return": true,
                            "use_streaming": false,
                            "streaming_type": "str",
                            "output_type": "json"
                        }
                    },
                    "next_node": {
                        "response_generator": {"group": "group_0", "type": "node"}
                    }
                },
                "response_generator": {
                    "node_id": "response_generator",
                    "processing_type": "llm",
                    "content": "답변을 생성 중입니다.",
                    "instructions": [
                        "당신은 친절하고 전문적인 고객센터 상담원입니다.",
                        "고객의 문의에 대해 명확하고 도움이 되는 답변을 제공하세요.",
                        "카테고리와 우선순위에 맞는 적절한 대응을 해주세요."
                    ],
                    "prompts": [
                        "고객님께서는 다음과 같은 문의를 주셨습니다: {user_message}",
                        "이 문의는 '{issue_category}' 카테고리로 분류되었으며, 우선순위는 '{issue_priority}'입니다.",
                        "티켓 번호 {ticket_id}가 생성되었습니다.",
                        "이에 대한 적절한 답변을 작성해주세요."
                    ],
                    "input_variables": {
                        "user_message": {"type": "string", "default_value": ""},
                        "issue_category": {"type": "string", "default_value": ""},
                        "issue_priority": {"type": "string", "default_value": ""},
                        "ticket_id": {"type": "string", "default_value": ""}
                    },
                    "output_variables": {
                        "response": {
                            "type": "object",
                            "description": "고객 문의에 대한 응답",
                            "properties": {
                                "greeting": {
                                    "type": "string",
                                    "description": "고객 인사말"
                                },
                                "answer": {
                                    "type": "string",
                                    "description": "주요 답변 내용"
                                },
                                "next_steps": {
                                    "type": "string",
                                    "description": "다음 단계 안내"
                                },
                                "closing": {
                                    "type": "string",
                                    "description": "마무리 인사"
                                }
                            }
                        }
                    },
                    "output_filter": "all",
                    "history": {
                        "save": true,
                        "use": true,
                        "configurable": {
                            "user_id": "customer_id",
                            "conversation_id": "ticket_id"
                        }
                    },
                    "output_config": {
                        "llm_to_server": {
                            "use_streaming": true,
                            "streaming_type": "str",
                            "output_type": "json"
                        },
                        "server_to_client": {
                            "use_return": true,
                            "use_streaming": true,
                            "streaming_type": "str",
                            "output_type": "json"
                        }
                    },
                    "next_node": {
                        "satisfaction_check": {"group": "group_0", "type": "node"}
                    }
                },
                "satisfaction_check": {
                    "node_id": "satisfaction_check",
                    "processing_type": "llm",
                    "content": "만족도를 확인 중입니다.",
                    "instructions": [
                        "당신은 고객 만족도를 평가하는 전문가입니다.",
                        "대화 내용을 분석하여 고객의 만족도를 예측하세요."
                    ],
                    "prompts": [
                        "다음 대화 내용을 분석하여 고객의 만족도를 평가해주세요:",
                        "고객 문의: {user_message}",
                        "답변 내용: {response.answer}"
                    ],
                    "input_variables": {
                        "user_message": {"type": "string", "default_value": ""},
                        "response": {"type": "object", "default_value": {}}
                    },
                    "output_variables": {
                        "satisfaction": {
                            "type": "object",
                            "description": "고객 만족도 평가",
                            "properties": {
                                "score": {
                                    "type": "integer",
                                    "description": "예상 만족도 점수 (1-10)"
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "만족도 예측 근거"
                                },
                                "improvement": {
                                    "type": "string",
                                    "description": "개선 가능한 부분"
                                }
                            }
                        }
                    },
                    "output_filter": "all",
                    "history": {
                        "save": true,
                        "use": true,
                        "configurable": {
                            "user_id": "customer_id",
                            "conversation_id": "ticket_id"
                        }
                    },
                    "output_config": {
                        "llm_to_server": {
                            "use_streaming": false,
                            "streaming_type": "str",
                            "output_type": "json"
                        },
                        "server_to_client": {
                            "use_return": true,
                            "use_streaming": false,
                            "streaming_type": "str",
                            "output_type": "json"
                        }
                    },
                    "next_node": {
                        "improvement_analyzer": {"group": "group_0", "type": "node"}
                    }
                },
                "improvement_analyzer": {
                    "node_id": "improvement_analyzer",
                    "processing_type": "llm",
                    "content": "개선점을 분석 중입니다.",
                    "instructions": [
                        "당신은 고객 서비스 개선을 위한 분석가입니다.",
                        "고객 만족도와 대화 내용을 분석하여 서비스 개선점을 도출하세요."
                    ],
                    "prompts": [
                        "다음 정보를 바탕으로 서비스 개선점을 분석해주세요:",
                        "고객 문의: {user_message}",
                        "답변 내용: {response.answer}",
                        "만족도 점수: {satisfaction.score}",
                        "만족도 평가 근거: {satisfaction.reason}"
                    ],
                    "input_variables": {
                        "user_message": {"type": "string", "default_value": ""},
                        "response": {"type": "object", "default_value": {}},
                        "satisfaction": {"type": "object", "default_value": {}}
                    },
                    "output_variables": {
                        "improvements": {
                            "type": "object",
                            "description": "서비스 개선점 분석",
                            "properties": {
                                "short_term": {
                                    "type": "list",
                                    "description": "단기 개선 사항",
                                    "items": {
                                        "type": "string",
                                        "description": "개별 개선 사항"
                                    }
                                },
                                "long_term": {
                                    "type": "list",
                                    "description": "장기 개선 사항",
                                    "items": {
                                        "type": "string",
                                        "description": "개별 개선 사항"
                                    }
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "개선점 요약"
                                }
                            }
                        }
                    },
                    "output_filter": "all",
                    "history": {
                        "save": true,
                        "use": true,
                        "configurable": {
                            "user_id": "customer_id",
                            "conversation_id": "ticket_id"
                        }
                    },
                    "output_config": {
                        "llm_to_server": {
                            "use_streaming": false,
                            "streaming_type": "str",
                            "output_type": "json"
                        },
                        "server_to_client": {
                            "use_return": true,
                            "use_streaming": false,
                            "streaming_type": "str",
                            "output_type": "json"
                        }
                    },
                    "next_node": {
                        "conv_count_updater": {"group": "group_0", "type": "node"}
                    }
                },
                "conv_count_updater": {
                    "node_id": "conv_count_updater",
                    "processing_type": "function",
                    "tool_name": "basic_operations",
                    "content": "대화 통계를 업데이트 중입니다.",
                    "operation": "increment",
                    "target_variable": "conv_count",
                    "input_variables": {
                        "conv_count": {"type": "integer", "default_value": 0}
                    },
                    "output_variables": {
                        "conv_count": {"type": "integer", "description": "증가된 대화 횟수"}
                    },
                    "output_filter": "all",
                    "output_config": {
                        "server_to_client": {
                            "use_return": true,
                            "use_streaming": false,
                            "streaming_type": "str",
                            "output_type": "json"
                        }
                    },
                    "conditions": [
                        {
                            "condition": {
                                "or": [
                                    {
                                        "satisfaction.score": { "operator": "<", "operand": 5 }
                                    }
                                ]
                            },
                            "actions": [
                                {
                                    "type": "variable",
                                    "name": "resolution_status",
                                    "value": "escalated"
                                }
                            ]
                        },
                        {
                            "condition": {
                                "or": [
                                    {
                                        "satisfaction.score": { "operator": ">=", "operand": 8 }
                                    }
                                ]
                            },
                            "actions": [
                                {
                                    "type": "variable",
                                    "name": "resolution_status",
                                    "value": "resolved"
                                }
                            ]
                        },
                        {
                            "condition": {
                                "and": [
                                    {
                                        "satisfaction.score": { "operator": ">=", "operand": 5 }
                                    },
                                    {
                                        "satisfaction.score": { "operator": "<", "operand": 8 }
                                    }
                                ]
                            },
                            "actions": [
                                {
                                    "type": "variable",
                                    "name": "resolution_status",
                                    "value": "pending_followup"
                                }
                            ]
                        }
                    ],
                    "next_node": {}
                }
            }
        }
    ]
}
```

## 타입 힌트 임포트
```python
from typing import Any, Union, List, Dict
``` 