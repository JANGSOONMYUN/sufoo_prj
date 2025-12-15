import os
import json
import functools
from typing import Dict, Any
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
from langgraph.graph import StateGraph, END
from models import create_state_from_json
from workflow import dynamic_node, parallel_dynamic_node, user_input_node
from config import load_workflows, extract_variables_and_types
from tools import tool_node

def init_llm(api_info: Dict[str, Any]) -> ChatVertexAI:
    """
    Vertex AI LLM을 초기화합니다.

    Args:
        api_info: API 설정 정보

    Returns:
        ChatVertexAI: 초기화된 LLM 인스턴스
    """
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = api_info['google']['credentials']
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/user/llm_test_15050/settings/google_api/gen-lang-client-0942875887-4088db77d287.json'
    return ChatVertexAI(
        project=api_info['google']['project_id'],
        location=api_info['google']['region'],
        model="gemini-2.0-flash",
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
        }
    )

def setup_workflow(llm: ChatVertexAI, workflows: Dict[str, Any]) -> StateGraph:
    """
    워크플로우를 설정합니다.

    Args:
        llm: LLM 인스턴스

    Returns:
        StateGraph: 설정된 워크플로우 그래프
    """
    # State 타입 생성
    state_fields = extract_variables_and_types(workflows)
    json_config = {
        "state_fields": {
            "user_message": {"type": "str"},
            "bot_response": {"type": "str", "optional": True},
            "conversation_history": {"type": "List[Dict]"},
            "subject_list": {"type": "List[Dict]", "optional": True},
            "current_node": {"type": "str"},
        },
        "output_model": {
            "fields": {
                "summary": {"type": "str"},
                "keywords": {"type": "List[str]"},
            }
        }
    }
    json_config['state_fields'].update(state_fields)
    json_config['state_fields'].update({
        'workflows': {"type": "List[Dict]", "optional": False},
        'variables': {"type": "Dict", "optional": False}
    })
    State = create_state_from_json(json_config)

    # 워크플로우 그래프 생성
    workflow = StateGraph(State)

    # 워크플로우 설정 로드
    main_workflow_config = workflows['workflows'][0]
    nodes_config = main_workflow_config['nodes']
    start_node_name = main_workflow_config['start_node'][0]

    # 노드 동적 추가
    for node_name, node_cfg in nodes_config.items():
        processing_type = node_cfg.get("processing_type")
        if processing_type in ["llm", "api", "logic"]:
            if 'branch' in node_cfg:
                if node_cfg.get("branch").get('use_branch'):
                    workflow.add_node(node_name, functools.partial(parallel_dynamic_node, node_name=node_name, llm=llm))
                else:
                    workflow.add_node(node_name, functools.partial(dynamic_node, node_name=node_name, llm=llm))
            else:
                workflow.add_node(node_name, functools.partial(dynamic_node, node_name=node_name, llm=llm))
        elif processing_type in ["func", "function"]:
            # functools.partial 사용 시 TypeError 발생 가능성 있음
            # workflow.add_node(node_name, functools.partial(tool_node, node_name=node_name, workflows=workflows))

            # lambda 함수를 사용하여 명시적으로 인자 전달 (tool_node가 (node_name, state, workflows) 순서로 인자를 받는다고 가정)
            def create_tool_node_lambda(nn, wf):
                # tool_node는 tools.py에서 import되었고, (node_name, state, workflows) 시그니처를 가졌다고 가정
                # LangGraph는 state를 첫 번째 인자로 전달
                return lambda state: tool_node(nn, state, wf)

            workflow.add_node(node_name, create_tool_node_lambda(node_name, workflows))
        elif processing_type in ["gateway", "join_gateway"]:
            workflow.add_node(node_name, functools.partial(dynamic_node, node_name=node_name, llm=llm))
        else:
            print(f"Warning: Skipping node '{node_name}' with unknown processing type '{processing_type}'")

    # 사용자 입력 노드 추가
    workflow.add_node("user_input", user_input_node)

    # 엣지 추가
    workflow.set_entry_point("user_input")
    workflow.add_edge("user_input", start_node_name)

    for node_name, node_cfg in nodes_config.items():
        if "next_nodes" in node_cfg:
            for next_node in node_cfg["next_nodes"]:
                workflow.add_edge(node_name, next_node)
        elif "next_node" in node_cfg and node_cfg["next_node"]:
            next_node_target = list(node_cfg["next_node"].keys())[0]
            workflow.add_edge(node_name, next_node_target)
        elif node_name != "join_gateway":
            pass

    if "join_gateway" in nodes_config and not nodes_config["join_gateway"].get("next_node"):
        workflow.add_edge("join_gateway", END)

    return workflow

def run_chatbot(user_message: str, api_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    챗봇을 실행합니다.

    Args:
        user_message: 사용자 메시지
        api_info: API 설정 정보

    Returns:
        Dict[str, Any]: 챗봇 응답
    """
    # LLM 초기화
    llm = init_llm(api_info)

    workflows = load_workflows()

    # 워크플로우 설정
    workflow = setup_workflow(llm, workflows)
    app = workflow.compile()

    # 초기 상태 설정
    initial_state = {
        "user_message": user_message,
        "conversation_history": [],
        "workflows": workflows['workflows'],
        "variables": workflows['variables']
    }

    # 챗봇 실행
    result = app.invoke(initial_state)
    return result

if __name__ == "__main__":
    # API 설정 로드
    config_path = '/home/user/llm_test_15050/settings/config.json'
    with open(config_path, 'r') as f:
        api_info = json.load(f)

    # 챗봇 실행
    user_message = '금전운, 연애운'
    result = run_chatbot(user_message, api_info)

    # 결과 출력
    print("Bot:", result["bot_response"])
    print("Bot:", result["subject_list"]) 