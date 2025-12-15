import copy
import functools
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import PydanticOutputParser
from models import create_pydantic_model_from_json
from utils import combine_prompts, preproc_node, generate_template, merge_responses, find_variable_location

def dynamic_node(state: Dict[str, Any], node_name: str, llm) -> Dict[str, Any]:
    """
    단일 노드 처리를 위한 함수입니다.

    Args:
        state: 현재 상태
        node_name: 노드 이름
        llm: LLM 인스턴스

    Returns:
        Dict[str, Any]: 업데이트된 상태
    """
    conversation_history = state.get("conversation_history", [])
    workflows = state["workflows"][0]
    nodes = workflows['nodes']
    
    if node_name not in nodes:
        return {"bot_response": f"오류: 노드 '{node_name}'를 찾을 수 없습니다."}
        
    node_data = nodes[node_name]
    
    instructions_str = combine_prompts(node_data['instructions'])
    prompts_str = combine_prompts(node_data['prompts'])
    
    input_variables = node_data['input_variables']
    output_variables = node_data['output_variables']
    history_config = node_data.get('history', {})
    
    input_with_value = {"conversation_history": conversation_history}
    for k, v in input_variables.items():
        if k in state and state[k] is not None:
            input_with_value[k] = state[k]
        elif 'default_value' in v:
            input_with_value[k] = v['default_value']
            
    output_model = {"type": "object", "properties": output_variables}
    dynamic_model = create_pydantic_model_from_json(output_model)
    parser = PydanticOutputParser(pydantic_object=dynamic_model)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            instructions_str + 
            "당신은 유능한 JSON 형식의 답변 생성기입니다. 사용자에게 필요한 정보를 JSON 형식으로 정확하고 간결하게 제공하세요. "
        ),
        MessagesPlaceholder(variable_name="conversation_history"),
        HumanMessagePromptTemplate.from_template(
            prompts_str + 
            "\nPlease respond in JSON format as specified below. \n"
            "{format_instructions}"
        )
    ])
    
    prompt_with_output_instructions = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    try:
        chain = prompt_with_output_instructions | llm
        response = chain.invoke(input_with_value)
        print(f"response: {response}")
        # response 객체가 AIMessage인지 확인
        if hasattr(response, 'content'):
            response_content = response.content
        else:
            # AIMessage가 아닌 경우, 직접 문자열로 변환
            response_content = str(response)
            
        parsed_output = parser.parse(response_content)
        result_dict = parsed_output.dict()
    except Exception as e:
        print(f"Error parsing response: {e}")
        result_dict = {}
        for k, v_config in output_variables.items():
            result_dict[k] = v_config.get('default_value') if isinstance(v_config, dict) else None

    state_update = {}
    state_update["bot_response"] = node_data.get("content", f"{node_name} 처리 중...")
    
    keep_result_in_workflow = {}
    for key, value in result_dict.items():
        if key in output_variables:
            state_update[key] = value
            keep_result_in_workflow[key] = value

    node_data['result'] = copy.deepcopy(keep_result_in_workflow)
    
    if history_config.get("save"):
        updated_history = list(conversation_history)
        # response 객체를 history에 추가
        updated_history.append(AIMessage(content=response_content))
        state_update["conversation_history"] = updated_history
            
    return state_update

def parallel_dynamic_node(state: Dict[str, Any], node_name: str, llm) -> Dict[str, Any]:
    """
    병렬 노드 처리를 위한 함수입니다.

    Args:
        state: 현재 상태
        node_name: 노드 이름
        llm: LLM 인스턴스

    Returns:
        Dict[str, Any]: 업데이트된 상태
    """
    conversation_history = state.get("conversation_history", [])
    workflows = state["workflows"][0]
    nodes = workflows['nodes']
    
    if node_name not in nodes:
        return {"bot_response": f"오류: 노드 '{node_name}'를 찾을 수 없습니다."}
        
    node_data = nodes[node_name]
    
    instructions_str = combine_prompts(node_data['instructions'])
    prompts_str = combine_prompts(node_data['prompts'])
    
    input_variables = node_data['input_variables']
    output_variables = node_data['output_variables']
    history_config = node_data.get('history', {})
    
    input_with_value = {"conversation_history": conversation_history}
    for k, v in input_variables.items():
        if k in state and state[k] is not None:
            input_with_value[k] = state[k]
        elif 'default_value' in v:
            input_with_value[k] = v['default_value']
    
    output_template = generate_template(output_variables)
    branch_rule = node_data['branch']['branch_rule']
    branch_result = preproc_node(input_with_value, output_variables, branch_rule)
    
    output_model = {"type": "object", "properties": output_variables}
    dynamic_model = create_pydantic_model_from_json(output_model)
    parser = PydanticOutputParser(pydantic_object=dynamic_model)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            instructions_str + 
            "당신은 유능한 JSON 형식의 답변 생성기입니다. 사용자에게 필요한 정보를 JSON 형식으로 정확하고 간결하게 제공하세요. "
        ),
        MessagesPlaceholder(variable_name="conversation_history"),
        HumanMessagePromptTemplate.from_template(
            prompts_str + 
            "\nPlease respond in JSON format as specified below. \n"
            "{format_instructions}"
        )
    ])
    
    prompt_with_output_instructions = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    def _single_runnable(llm_chain, input_dict_val):
        result = llm_chain.invoke(input_dict_val)
        # result 객체가 AIMessage인지 확인
        if hasattr(result, 'content'):
            return result
        else:
            # AIMessage가 아닌 경우, AIMessage로 변환
            return AIMessage(content=str(result))

    runnable_lambda = RunnableLambda(functools.partial(_single_runnable, prompt_with_output_instructions | llm))
    
    parallel_chains = {}
    branch_input = {}
    chain_names = []
    
    for i, branch in enumerate(branch_result):
        chain_name = f'chain_{i}'
        branch.update({"conversation_history": conversation_history})
        branch_input[chain_name] = branch
        chain_names.append(chain_name)
        parallel_chains[chain_name] = RunnableLambda(lambda data, chain_key=chain_name: runnable_lambda.invoke(data[chain_key]))
    
    parallel_runnable = RunnableParallel(parallel_chains)
    
    try:
        response = parallel_runnable.invoke(branch_input)
        out_depth_path = branch_rule['out_depth_path']
        _, target_out_var_ptr, _key = find_variable_location(output_template, out_depth_path)
        concat_result = merge_responses(response, parser)
        target_out_var_ptr[_key] = concat_result

        print(f"concat_result: {concat_result}")
        result_dict = output_template
        
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        result_dict = {}
        for k, v_config in output_variables.items():
            result_dict[k] = v_config.get('default_value') if isinstance(v_config, dict) else None

    state_update = {}
    state_update["bot_response"] = node_data.get("content", f"{node_name} 처리 중...")
    
    keep_result_in_workflow = {}
    for key, value in result_dict.items():
        if key in output_variables:
            state_update[key] = value
            keep_result_in_workflow[key] = value

    node_data['result'] = copy.deepcopy(keep_result_in_workflow)
    
    if history_config.get("save"):
        updated_history = list(conversation_history)
        for k, v in response.items():
            # v 객체가 AIMessage인지 확인
            if hasattr(v, 'content'):
                updated_history.append(v)
            else:
                # AIMessage가 아닌 경우, AIMessage로 변환
                updated_history.append(AIMessage(content=str(v)))
        state_update["conversation_history"] = updated_history
            
    return state_update

def user_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자 입력 처리를 위한 노드입니다.

    Args:
        state: 현재 상태

    Returns:
        Dict[str, Any]: 업데이트된 상태
    """
    result = {
        "user_message": state["user_message"],
        "conversation_history": state["conversation_history"],
        "workflows": state["workflows"]
    }
    result.update(state["variables"])
    return result 