import copy
import json
from collections import deque

def object_to_string(obj):
    """
    Converts any Python object to a string representation.

    This function attempts to provide a reasonable string representation
    for various Python objects, including primitives, collections, and
    custom objects.

    Args:
    obj: The Python object to convert.

    Returns:
    A string representation of the object.
    """
    try:
        # Attempt to use the object's __str__ method (if available)
        if isinstance(obj, dict):
            return json.dumps(obj, indent=4,  ensure_ascii=False)
        return str(obj)
    except Exception as e:
        # If __str__ fails (e.g., due to recursion or other errors), 
        # use repr() as a fallback.  repr() provides a more developer-focused
        # representation, including the object's type and memory address.
        return repr(obj)

def preproc_node(input_variables, output_variables, branch_rule):
    
    def _find_variable_location(variables, depth_path):
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
                return None  # or raise an exception if the path is invalid
                
        return cur_variable, prv_variable, cur_key
    

    data_type = branch_rule['data_type']
    in_depth_path = branch_rule['in_depth_path']
    out_depth_path = branch_rule['out_depth_path']
    in_out_set = branch_rule['in_out_set']
    
    cur_input_variables, _, _ = _find_variable_location(input_variables, in_depth_path)
    cur_output_variables, _, _ = _find_variable_location(output_variables, out_depth_path)

    
    branch_result = []
    if isinstance(cur_input_variables, list):
        for in_var in cur_input_variables:
            temp_branch = copy.deepcopy(cur_output_variables)
            for inout in in_out_set:
                io_type = inout['type']
                io_from = inout['from']
                io_to_base = inout['to']['base']
                io_to_sub_path = inout['to']['sub_path']
                
                target_in_var, _, _ = _find_variable_location(in_var, io_from)
                target_in_var = object_to_string(target_in_var)


                for sub_path in io_to_sub_path:
                    io_to = io_to_base + sub_path
                    target_out_var, target_out_var_ptr, _key = _find_variable_location(temp_branch, io_to)
                    if io_type == 'append' or io_type == 'add':
                        target_out_var_ptr[_key] += target_in_var
                    else:
                        target_out_var_ptr[_key] = target_in_var

            branch_result.append(temp_branch)
    return branch_result

def generate_template(data_type_definition, indent=0):
    """
    주어진 데이터 타입 정의에 따라 빈 템플릿을 생성하는 함수입니다.

    Args:
        data_type_definition (dict): Langchain의 output_variables에 정의된 데이터 타입 정의.
        indent (int): 들여쓰기 레벨.

    Returns:
        any: 해당 데이터 타입에 대한 빈 템플릿.
    """
    indent_str = " " * indent
    d_type = data_type_definition.get("type")

    if 'str' in d_type.lower():
        return ""  # 빈 문자열

    elif 'int' in d_type.lower():
        return 0  # 0

    elif 'num' in d_type.lower():  # float 포함
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
        return [generate_template(items_def, indent + 4)] # 리스트는 1개만 템플릿으로 만듦

    else:
        return None  # 알 수 없는 타입

input_variables = {
    "subject_list": [
        {"subject": "총운", "sub_subjects": ["성격분석", "오행분석"]}, 
        {"subject": "금전운", "sub_subjects": ["기회분석", "위기분석"]}
    ]
}
branch_rule =  {
                    "data_type": "list",
                    "in_depth_path": ["subject_list"],
                    "out_depth_path": ["result", "items", "properties"],
                    "in_out_set": [
                        {
                            "type": "append",
                            "from": ["subject"],
                            "to": {
                                "base": ["subject"],
                                "sub_path": [["description"], ["properties", "title", "description"]]
                                }
                        },
                        {
                            "type": "append",
                            "from": ["sub_subjects"],
                            "to": {
                                "base": ["sub_subjects"],
                                "sub_path": [["description"]]
                                }
                        }
                    ]
                }

# "subject", "title"
# "subject" ->  "subject", "description"
# "@STR" -> 



output_variables = {
    "result": {
        "type": "list",
        "description": "고객 맞춤형 유용한 정보를 전달하기 위한 주제와 소주제 목록; 중복되지 않게 다양한 주제로 만든다; 최대한 상세하게 정의",
        "items": {
            "type": "object",
            "description": "각 주제와 소주제",
            "properties": {
                "subject": {
                    "type": "object",
                    "description": "주제",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "주제에 대한 제목"
                        },
                        "content": {
                            "type": "string",
                            "description": "주제에 대한 해석"
                        }
                    }
                },
                "sub_subjects": {
                    "type": "list",
                    "description": "주제에 대해 심화된 소주제",
                    "items": {
                        "type": "object",
                        "description": "소주제에 대한 제목",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "소주제에 대한 제목"
                            },
                            "content": {
                                "type": "string",
                                "description": "소주제에 대한 해석"
                            }
                        }
                    }
                }
            }
        }
    }
}



workflows = {
    "agent_name": "운세 상담 챗봇",
    "description": "운세 챗봇",
    "variables": {
        "conv_count": 0,
        "current_node": '',
        "next_node": '',
        "val1": 0,
        "val2": 0,
        "val3": 0,
        "val4": 0,
        "val5": 0,
        "val6": 0
    },
    "workflows": [
        {
            "workflow_id": "main_flow",
            "name": "메인 흐름",
            "description": "챗봇의 메인 흐름",
            "start_node": ["saju_subject_generator"],
            "nodes": {
                "saju_subject_generator": {
                    "node_id": "saju_subject_generator",
                    "processing_type": "llm",  
                    "content": "잠시만 기다려주세요",
                    "instructions": ["/saju_subject_generator"],
                    "prompts": ["/saju_subject_generator",
                                "\n소주제(sub_subjects) 개수는 2개로 고정\n",
                               "{user_message}"],
                    "input_variables": {
                        "user_message": {"type": "string", "default_value": ""}
                    },
                    "output_variables": {
                        "subject_list": {
                            "type": "list",
                            "description": "고객 맞춤형 유용한 정보를 전달하기 위한 주제와 소주제 목록; 중복되지 않게 다양한 주제로 만든다; 최대한 상세하게 정의",
                            "items": {
                                "type": "object",
                                "description": "각 주제와 소주제",
                                "properties": {
                                    "subject": {
                                        "type": "string",
                                        "description": "주제에 대한 제목"
                                    },
                                    "sub_subjects": {
                                        "type": "list",
                                        "description": "주제에 대해 심화된 소주제",
                                        "items": {
                                            "type": "string",
                                            "description": "주제에 대해 심화된 소주제; 중복되는 주제가 없도록 한다"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "output_filter": "all",    
                    "history": {
                        "save": True,
                        "use": True,  
                        "configurable": {
                            "user_id": "tmp",
                            "conversation_id": "tmp"
                        }
                    },
                    "output_config": {
                        "llm_to_server": {
                            "use_streaming": False,
                            "streaming_type": "str",
                            "output_type": "str"
                        },
                        "server_to_client": {
                            "use_return": True,
                            "use_streaming": False,
                            "streaming_type": "str",
                            "output_type": "str"
                        }
                    },
                    "next_node": {
                        "parallel_interpretation": {"group": "group_0", "type": "node"}
                    }
                },
                "parallel_interpretation": {
                    "id": "parallel_interpretation",
                    "processing_type": "llm", 
                    "content": "해석중",
                    "instructions": ["/parallel_interpretation"],
                    "prompts": ["/parallel_interpretation"],
                    "input_variables": {
                        "subject_list": {"type": "list", "default_value": []}
                    },
                    "output_variables": {
                        "result": {
                            "type": "list",
                            "description": "고객 맞춤형 유용한 정보를 전달하기 위한 주제와 소주제 목록; 중복되지 않게 다양한 주제로 만든다; 최대한 상세하게 정의",
                            "items": {
                                "type": "object",
                                "description": "각 주제와 소주제",
                                "properties": {
                                    "subject": {
                                        "type": "object",
                                        "description": "주제",
                                        "properties": {
                                            "title": {
                                                "type": "string",
                                                "description": "주제에 대한 제목"
                                            },
                                            "content": {
                                                "type": "string",
                                                "description": "주제에 대한 해석"
                                            }
                                        }
                                    },
                                    "sub_subjects": {
                                        "type": "list",
                                        "description": "주제에 대해 심화된 소주제",
                                        "items": {
                                            "type": "object",
                                            "description": "소주제에 대한 제목",
                                            "properties": {
                                                "title": {
                                                    "type": "string",
                                                    "description": "소주제에 대한 제목"
                                                },
                                                "content": {
                                                    "type": "string",
                                                    "description": "소주제에 대한 해석"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "output_filter": "all",      
                    "branch": {
                        "use_branch": True,
                        "branch_rule": {
                            "data_type": "list",
                            "in_depth_path": ["subject_list"],
                            "out_depth_path": ["result"],
                            "in_out_set": [
                                {
                                    "type": "append",
                                    "from": ["subject"],
                                    "to": {
                                        "base": ["items", "properties", "subject"],
                                        "sub_path": [["description"], ["properties", "title", "description"]]
                                        }
                                },
                                {
                                    "type": "append",
                                    "from": ["sub_subjects"],
                                    "to": {
                                        "base": ["items", "properties", "sub_subjects"],
                                        "sub_path": [["description"]]
                                        }
                                }
                            ]
                        }
                    },
                    "history": {
                        "save": True,
                        "use": True,
                        "configurable": {
                            "user_id": "123",
                            "conversation_id": "1"
                        }
                    },
                    "output_config": {
                        "llm_to_server": {
                            "use_streaming": False,
                            "streaming_type": "str",
                            "output_type": "json"
                        },
                        "server_to_client": {
                            "use_return": True,
                            "use_streaming": False,
                            "streaming_type": "str",
                            "output_type": "json"
                        }
                    },
                    "next_node": {}
                }
            }
        }
    ]
}
