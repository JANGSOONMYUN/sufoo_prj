import json
from typing import Dict, Any
from pathlib import Path

def load_workflows() -> Dict[str, Any]:
    """
    workflows.json 파일에서 설정을 로드합니다.
    
    Returns:
        Dict[str, Any]: 워크플로우 설정
    """
    config_path = Path(__file__).parent / 'workflows.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_variables_and_types(workflows: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Workflow 설정에서 변수와 타입을 추출합니다.

    Args:
        workflows: Workflow 설정

    Returns:
        변수 이름과 타입 정보를 담은 딕셔너리
    """
    state_fields = {}

    def _set_type(var_name: str, var_value: Any) -> str:
        var_type_str = var_value['type'] if isinstance(var_value, dict) and 'type' in var_value else None
        default_value = var_value['default_value'] if isinstance(var_value, dict) and 'default_value' in var_value else var_value

        if var_type_str is not None and len(var_type_str) > 0:
            var_type_str = var_type_str.lower()
            if "str" in var_type_str:
                return 'str'
            elif "int" in var_type_str:
                return 'int'
            elif var_type_str == "float":
                return 'float'
            elif var_type_str in ("list", "array", "arr") or "list" in var_type_str:
                return 'List'
            elif var_type_str in ("obj", "object", "dict", "json"):
                return 'Dict'
            elif var_type_str == "global":
                return 'Any'
            elif "bool" in var_type_str:
                return 'bool'
            elif var_type_str == "tuple":
                return 'tuple'
            elif var_type_str == "set":
                return 'set'
            else:
                return 'Any'
        else:
            if isinstance(default_value, str):
                return 'str'
            elif isinstance(default_value, int):
                return 'int'
            elif isinstance(default_value, float):
                return 'float'
            elif isinstance(default_value, list):
                return 'List'
            elif isinstance(default_value, dict):
                return 'Dict'
            elif isinstance(default_value, bool):
                return 'bool'
            elif isinstance(default_value, tuple):
                return 'tuple'
            elif isinstance(default_value, set):
                return 'set'
            elif default_value is None:
                return 'Any'
            else:
                return str(type(default_value))

    if "variables" in workflows:
        for var_name, var_value in workflows["variables"].items():
            state_fields[var_name] = {'type': _set_type(var_name, var_value), 'optional': True}

    for workflow in workflows.get("workflows", []):
        for node_id, node in workflow["nodes"].items():
            if "input_variables" in node:
                for var_name, var_config in node["input_variables"].items():
                    state_fields[var_name] = {'type': _set_type(var_name, var_config), 'optional': True}

            if "output_variables" in node:
                for var_name, var_config in node["output_variables"].items():
                    state_fields[var_name] = {'type': _set_type(var_name, var_config), 'optional': True}

    return state_fields 