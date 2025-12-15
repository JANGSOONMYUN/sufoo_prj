import json
import os
import copy
from collections import deque
from typing import Dict, Any, List

def object_to_string(obj: Any) -> str:
    """
    Converts any Python object to a string representation.

    Args:
        obj: The Python object to convert.

    Returns:
        A string representation of the object.
    """
    try:
        if isinstance(obj, dict):
            return json.dumps(obj, indent=4, ensure_ascii=False)
        return str(obj)
    except Exception as e:
        return repr(obj)

def find_variable_location(variables: Dict, depth_path: List[str]) -> tuple:
    """
    Find the location of a variable in a nested dictionary structure.

    Args:
        variables: The dictionary to search in
        depth_path: List of keys representing the path to the variable

    Returns:
        Tuple of (current_variable, parent_variable, current_key)
    """
    path_queue = deque(depth_path)
    prv_variable = variables
    cur_key = ''
    cur_variable = variables
    
    while len(path_queue) > 0:
        cur_loc = path_queue.popleft()
        if isinstance(cur_variable, dict) and cur_loc in cur_variable:
            prv_variable = cur_variable
            cur_key = cur_loc
            cur_variable = cur_variable[cur_loc]
        else:
            return None
            
    return cur_variable, prv_variable, cur_key

def generate_template(data_type_definition: Dict, indent: int = 0) -> Any:
    """
    주어진 데이터 타입 정의에 따라 빈 템플릿을 생성하는 함수입니다.

    Args:
        data_type_definition: Langchain의 output_variables에 정의된 데이터 타입 정의.
        indent: 들여쓰기 레벨.

    Returns:
        해당 데이터 타입에 대한 빈 템플릿.
    """
    indent_str = " " * indent
    d_type = data_type_definition.get("type")
    
    if d_type is None:
        template = {}
        for prop_name, prop_def in data_type_definition.items():
            template[prop_name] = generate_template(prop_def, indent + 4)
        return template
    elif 'str' in d_type.lower():
        return ""
    elif 'int' in d_type.lower():
        return 0
    elif 'num' in d_type.lower():
        return 0.0
    elif 'bool' in d_type.lower():
        return False
    elif 'obj' in d_type.lower():
        template = {}
        properties = data_type_definition.get("properties", {})
        for prop_name, prop_def in properties.items():
            template[prop_name] = generate_template(prop_def, indent + 4)
        return template
    elif 'list' in d_type.lower():
        items_def = data_type_definition.get("items", {})
        return [generate_template(items_def, indent + 4)]
    else:
        return None

def combine_prompts(prompts: List[str]) -> str:
    """
    프롬프트 리스트를 불러와서 합치는 함수.

    Args:
        prompts: 프롬프트 아이템의 리스트. 각 아이템은 파일 경로 또는 문자열 자체일 수 있습니다.

    Returns:
        str: 결합된 프롬프트.
    """
    combined_prompt = ""
    for item in prompts:
        if isinstance(item, str):
            if os.path.exists(item):
                try:
                    with open(item, "r", encoding="utf-8") as f:
                        file_content = f.read()
                        combined_prompt += file_content
                except Exception as e:
                    print(f"경고: 파일 '{item}'을 읽는 동안 오류가 발생했습니다: {e}. 스킵합니다.")
                    continue
            else:
                combined_prompt += item
        else:
            print(f"경고: 유효하지 않은 프롬프트 아이템 '{item}'을 발견했습니다. 스킵합니다.")
            continue

    return combined_prompt

def preproc_node(input_variables: Dict, output_variables: Dict, branch_rule: Dict) -> List[Dict]:
    """
    노드 전처리를 수행하여 병렬 처리를 위한 입력을 준비합니다.

    Args:
        input_variables: 입력 변수 딕셔너리
        output_variables: 출력 변수 딕셔너리
        branch_rule: 분기 규칙 딕셔너리

    Returns:
        List[Dict]: 병렬 처리를 위한 입력 리스트
    """
    data_type = branch_rule['data_type']
    in_depth_path = branch_rule['in_depth_path']
    out_depth_path = branch_rule['out_depth_path']
    in_out_set = branch_rule['in_out_set']
    
    cur_input_variables, _, _ = find_variable_location(input_variables, in_depth_path)
    cur_output_variables, _, _ = find_variable_location(output_variables, out_depth_path)
    
    branch_result = []
    if isinstance(cur_input_variables, list):
        for in_var in cur_input_variables:
            temp_branch = copy.deepcopy(cur_output_variables)
            for inout in in_out_set:
                io_type = inout['type']
                io_from = inout['from']
                io_to_base = inout['to']['base']
                io_to_sub_path = inout['to']['sub_path']
                
                target_in_var, _, _ = find_variable_location(in_var, io_from)
                target_in_var = object_to_string(target_in_var)

                for sub_path in io_to_sub_path:
                    io_to = io_to_base + sub_path
                    target_out_var, target_out_var_ptr, _key = find_variable_location(temp_branch, io_to)
                    if io_type == 'append' or io_type == 'add':
                        target_out_var_ptr[_key] += target_in_var
                    else:
                        target_out_var_ptr[_key] = target_in_var

            branch_result.append(temp_branch)
    return branch_result

def merge_responses(all_response: Dict, parser=None, branch_chain_keys=None) -> List:
    """
    병렬 처리된 응답들을 병합합니다.

    Args:
        all_response: 모든 응답을 포함하는 딕셔너리
        parser: 응답 파싱을 위한 파서 객체
        branch_chain_keys: 병합할 체인 키 리스트

    Returns:
        List: 병합된 응답 리스트
    """
    concat_result = []
    if branch_chain_keys is None:
        for k, v in all_response.items():
            val = v.content
            if parser is not None:
                val = parser.parse(val)
            concat_result.append(val)
    else:
        for chain_k in branch_chain_keys:
            val = all_response[chain_k].content
            if parser is not None:
                val = parser.parse(val)
            concat_result.append(val)
    return concat_result 