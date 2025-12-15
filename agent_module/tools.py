from typing import Dict, Any, Callable, List, Union
from functools import reduce

# Tool 함수들을 저장할 레지스트리
TOOL_REGISTRY: Dict[str, Callable] = {}

def register_tool(name: str) -> Callable:
    """
    Tool 함수를 등록하는 데코레이터
    
    Args:
        name: tool의 이름
        
    Returns:
        Callable: 데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        TOOL_REGISTRY[name] = func
        return func
    return decorator

def tool_node(node_name: str, state: Dict[str, Any], workflows: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool 함수를 실행하는 노드
    
    Args:
        node_name: 노드 이름
        state: 현재 상태
        workflows: 워크플로우 설정
        
    Returns:
        Dict[str, Any]: 업데이트된 상태
    """
    # 워크플로우에서 노드 설정 찾기
    node_config = None
    for workflow in workflows['workflows']:
        if node_name in workflow['nodes']:
            node_config = workflow['nodes'][node_name]
            break
    
    if not node_config:
        raise ValueError(f"Node {node_name} not found in workflows")
    
    tool_name = node_config.get('tool_name')
    if not tool_name:
        raise ValueError(f"Tool name not specified for node {node_name}")
    
    if tool_name in TOOL_REGISTRY:
        # 표준 노드 설정 키 목록 (이 외의 키는 파라미터로 간주)
        standard_keys = {
            'node_id', 'id', 'processing_type', 'tool_name', 'content',
            'instructions', 'prompts', 'input_variables', 'output_variables',
            'output_filter', 'history', 'output_config', 'next_node', 'branch'
        }
        # node_config에서 표준 키가 아닌 것들을 파라미터로 추출
        params = {k: v for k, v in node_config.items() if k not in standard_keys}

        # state와 추출된 파라미터를 함께 전달
        return TOOL_REGISTRY[tool_name](state, **params)
    else:
        raise ValueError(f"Tool {tool_name} not found in registry")

def evaluate_condition(condition: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """
    단일 조건을 평가합니다.
    
    Args:
        condition: 평가할 조건
        state: 현재 상태
        
    Returns:
        bool: 조건 평가 결과
    """
    if "or" in condition:
        return any(evaluate_condition(c, state) for c in condition["or"])
    elif "and" in condition:
        return all(evaluate_condition(c, state) for c in condition["and"])
    else:
        for var_name, condition_dict in condition.items():
            operator = condition_dict["operator"]
            operand = condition_dict["operand"]
            value = state.get(var_name)
            
            if operator == "==":
                return value == operand
            elif operator == "!=":
                return value != operand
            elif operator == ">":
                return value > operand
            elif operator == ">=":
                return value >= operand
            elif operator == "<":
                return value < operand
            elif operator == "<=":
                return value <= operand
            else:
                raise ValueError(f"Unknown operator: {operator}")
        return False

def execute_action(action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 액션을 실행합니다.
    
    Args:
        action: 실행할 액션
        state: 현재 상태
        
    Returns:
        Dict[str, Any]: 업데이트된 상태
    """
    action_type = action["type"]
    
    if action_type == "function":
        func_name = action["name"]
        params = action.get("params", {})
        
        # 파라미터에서 변수 참조 처리
        processed_params = {}
        for param_name, param_value in params.items():
            if isinstance(param_value, dict) and "variable" in param_value:
                processed_params[param_name] = state.get(param_value["variable"])
            else:
                processed_params[param_name] = param_value
        
        if func_name in TOOL_REGISTRY:
            result = TOOL_REGISTRY[func_name](state, **processed_params)

            print('result:', result)
            
            # 출력값 처리
            if "output" in action:
                output_config = action["output"]
                if "target" in output_config:
                    state[output_config["target"]] = result
            return state
        else:
            raise ValueError(f"Function {func_name} not found in registry")
            
    elif action_type == "variable":
        var_name = action["name"]
        value = action["value"]
        if isinstance(value, dict) and "variable" in value:
            value = state.get(value["variable"])
        state[var_name] = value
        return state
        
    else:
        raise ValueError(f"Unknown action type: {action_type}")

@register_tool("condition_evaluator")
def condition_evaluator(state: Dict[str, Any], *, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    조건식을 평가하고 해당하는 액션을 실행하는 도구
    
    Args:
        state: 현재 상태
        conditions: 평가할 조건식 목록 (키워드 전용 인자)
        
    Returns:
        Dict[str, Any]: 업데이트된 상태
    """
    for condition_group in conditions:
        if evaluate_condition(condition_group["condition"], state):
            for action in condition_group["actions"]:
                state = execute_action(action, state)
    return state

# 예시 Tool 함수들
@register_tool("increment_conv_count")
def increment_conv_count(state: Dict[str, Any]) -> Dict[str, Any]:
    """대화 횟수를 증가시키는 tool"""
    state["conv_count"] = state.get("conv_count", 0) + 1
    return state

@register_tool("validate_conv_count")
def validate_conv_count(state: Dict[str, Any]) -> Dict[str, Any]:
    """대화 횟수를 검증하는 tool"""
    if state.get("conv_count", 0) > 5:
        raise ValueError("대화 횟수가 5회를 초과했습니다")
    return state

@register_tool("func_sum")
def func_sum(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    모든 입력 파라미터의 합을 계산하는 함수
    
    Args:
        state: 현재 상태
        **kwargs: 임의의 수의 파라미터
        
    Returns:
        Dict[str, Any]: 업데이트된 상태
    """
    total = 0
    for param_name, value in kwargs.items():
        if isinstance(value, (int, float)):
            total += value
        elif isinstance(value, dict) and "variable" in value:
            # 변수 참조인 경우
            var_value = state.get(value["variable"])
            if isinstance(var_value, (int, float)):
                total += var_value
    
    return total

@register_tool("basic_operations")
def basic_operations(state: Dict[str, Any], *,
                       operation: str = "",
                       expression: str = "",
                       target_variable: str = "",
                       operations: List[Dict[str, Any]] = None,
                       operands: List[Any] = None,
                       **kwargs) -> Dict[str, Any]:
    """
    사칙연산을 수행하는 도구

    노드 설정에서 전달된 operation 파라미터에 따라 다양한 연산을 수행합니다:
    - add: 덧셈
    - subtract: 뺄셈
    - multiply: 곱셈
    - divide: 나눗셈
    - increment: 변수 값 증가
    - decrement: 변수 값 감소
    - compound: 여러 연산을 순차적으로 수행 (operations 파라미터 사용)
    - expression: 단일 수학 표현식 계산 (expression 파라미터 사용)

    Args:
        state: 현재 상태
        operation: 수행할 연산 유형 (키워드 전용 인자)
        expression: 계산할 표현식 (operation이 'expression'일 때 사용, 키워드 전용 인자)
        target_variable: 결과를 저장할 변수 이름 (키워드 전용 인자)
        operations: 복합 연산 목록 (operation이 'compound'일 때 사용, 키워드 전용 인자)
        operands: 단일 연산에 사용할 피연산자 목록 (키워드 전용 인자)
        **kwargs: 호환성을 위한 추가 키워드 인자

    Returns:
        Dict[str, Any]: 업데이트된 상태
    """
    if not operation:
        print("Warning: 'operation' parameter not provided for basic_operations")
        return state

    if operation == "expression":
        target_var = target_variable

        if expression and target_var:
            expr = expression
            import re
            variable_names = set(re.findall(r'\$(\w+)', expr))
            for var_name in variable_names:
                if var_name in state:
                    var_value = state[var_name]
                    if isinstance(var_value, (int, float)):
                        expr = expr.replace(f'${var_name}', str(var_value))
                    else:
                        print(f"Warning: Variable ${var_name} in expression is not a number: {var_value}")
                        expr = expr.replace(f'${var_name}', '0')
                else:
                    print(f"Warning: Variable ${var_name} not found in state.")
                    expr = expr.replace(f'${var_name}', '0')

            try:
                result = safe_eval(expr)
                state[target_var] = result
            except Exception as e:
                print(f"Error evaluating expression '{expression}' (processed: '{expr}'): {e}")

    elif operation == "compound":
        if operations is None: operations = []
        for op in operations:
            op_type = op.get("type", "")
            target_var = op.get("target_variable", "")

            if op_type == "expression":
                expression_inner = op.get("expression", "")
                if expression_inner and target_var:
                    expr_inner = expression_inner
                    import re
                    variable_names_inner = set(re.findall(r'\$(\w+)', expr_inner))
                    for var_name in variable_names_inner:
                        if var_name in state:
                            var_value = state[var_name]
                            if isinstance(var_value, (int, float)):
                                expr_inner = expr_inner.replace(f'${var_name}', str(var_value))
                            else:
                                expr_inner = expr_inner.replace(f'${var_name}', '0')
                        else:
                            expr_inner = expr_inner.replace(f'${var_name}', '0')

                    try:
                        result = safe_eval(expr_inner)
                        state[target_var] = result
                    except Exception as e:
                        print(f"Error evaluating inner expression '{expression_inner}' (processed: '{expr_inner}'): {e}")
                continue

            operands_inner = op.get("operands", [])

            values = []
            for operand in operands_inner:
                if isinstance(operand, dict) and "variable" in operand:
                    var_name = operand["variable"]
                    values.append(state.get(var_name, 0))
                elif isinstance(operand, dict) and "expression" in operand:
                    expr_operand = operand["expression"]
                    import re
                    variable_names_operand = set(re.findall(r'\$(\w+)', expr_operand))
                    for var_name in variable_names_operand:
                        if var_name in state:
                            var_value = state[var_name]
                            if isinstance(var_value, (int, float)):
                                expr_operand = expr_operand.replace(f'${var_name}', str(var_value))
                            else:
                                expr_operand = expr_operand.replace(f'${var_name}', '0')
                        else:
                            expr_operand = expr_operand.replace(f'${var_name}', '0')
                    try:
                        values.append(safe_eval(expr_operand))
                    except Exception as e:
                        print(f"Error evaluating operand expression '{operand['expression']}' (processed: '{expr_operand}'): {e}")
                        values.append(0)
                else:
                    values.append(operand)

            result = perform_operation(op_type, values, state, target_var)
            if target_var:
                state[target_var] = result
    else:
        if operands is None: operands = []
        target_var = target_variable

        values = []
        for operand in operands:
            if isinstance(operand, dict) and "variable" in operand:
                var_name = operand["variable"]
                values.append(state.get(var_name, 0))
            elif isinstance(operand, dict) and "expression" in operand:
                expr_operand = operand["expression"]
                import re
                variable_names_operand = set(re.findall(r'\$(\w+)', expr_operand))
                for var_name in variable_names_operand:
                    if var_name in state:
                        var_value = state[var_name]
                        if isinstance(var_value, (int, float)):
                            expr_operand = expr_operand.replace(f'${var_name}', str(var_value))
                        else:
                            expr_operand = expr_operand.replace(f'${var_name}', '0')
                    else:
                        expr_operand = expr_operand.replace(f'${var_name}', '0')
                try:
                    values.append(safe_eval(expr_operand))
                except Exception as e:
                    print(f"Error evaluating operand expression '{operand['expression']}' (processed: '{expr_operand}'): {e}")
                    values.append(0)
            else:
                values.append(operand)

        result = perform_operation(operation, values, state, target_var)
        if target_var:
            state[target_var] = result

    return state

def safe_eval(expression: str) -> Any:
    """
    안전한 방식으로 수식을 계산합니다.
    
    Args:
        expression: 계산할 수식 문자열
        
    Returns:
        Any: 계산 결과
    """
    # 허용된 연산자와 함수만 포함된 제한된 환경에서 계산
    # 기본 사칙연산과 몇 가지 수학 함수만 허용
    import math
    allowed_names = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'pow': pow,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'pi': math.pi,
        'e': math.e
    }
    
    # 안전한 환경에서 수식 계산
    try:
        return eval(expression, {"__builtins__": {}}, allowed_names)
    except Exception as e:
        raise ValueError(f"수식 계산 오류: {expression}, {e}")

def perform_operation(operation: str, values: List[Any], state: Dict[str, Any], target_var: str) -> Any:
    """
    지정된 연산을 수행합니다.
    
    Args:
        operation: 수행할 연산 유형
        values: 연산에 사용할 값 목록
        state: 현재 상태
        target_var: 결과를 저장할 변수 이름
        
    Returns:
        Any: 연산 결과
    """
    result = 0
    
    if operation == "add":
        result = sum(values)
    elif operation == "subtract":
        if len(values) > 0:
            result = values[0]
            for value in values[1:]:
                result -= value
    elif operation == "multiply":
        if len(values) > 0:
            result = 1
            for value in values:
                result *= value
    elif operation == "divide":
        if len(values) > 1 and values[1] != 0:
            result = values[0] / values[1]
    elif operation == "increment":
        result = state.get(target_var, 0) + 1
    elif operation == "decrement":
        result = state.get(target_var, 0) - 1
    elif operation == "modulo":
        if len(values) > 1 and values[1] != 0:
            result = values[0] % values[1]
    elif operation == "power":
        if len(values) > 1:
            result = values[0] ** values[1]
    elif operation == "max":
        if len(values) > 0:
            result = max(values)
    elif operation == "min":
        if len(values) > 0:
            result = min(values)
    
    return result

if __name__ == "__main__":
    # 테스트를 위한 워크플로우 설정 (필요한 노드 정보만 포함)
    test_workflows = {
        'workflows': [
            {
                'workflow_id': 'test_flow',
                'nodes': {
                    'conv_count_increment': {
                        'node_id': 'conv_count_increment',
                        'processing_type': 'function',
                        'tool_name': 'basic_operations',
                        'operation': 'expression',
                        'expression': '$conv_count + 1',
                        'target_variable': 'conv_count',
                        'input_variables': {
                            'conv_count': {'type': 'integer', 'default_value': 0}
                        }
                    },
                    'expression_math_example': {
                        'node_id': 'expression_math_example',
                        'processing_type': 'function',
                        'tool_name': 'basic_operations',
                        'operation': 'expression',
                        'expression': '($val1 + $val2) * 3 - 10 + pow($val3, 2) / 10',
                        'target_variable': 'expression_result',
                        'input_variables': {
                            'val1': {'type': 'integer', 'default_value': 10},
                            'val2': {'type': 'integer', 'default_value': 20},
                            'val3': {'type': 'integer', 'default_value': 5}
                        }
                    },
                    'compound_with_expression_example': {
                        'node_id': 'compound_with_expression_example',
                        'processing_type': 'function',
                        'tool_name': 'basic_operations',
                        'operation': 'compound',
                        'operations': [
                            {
                                'type': 'expression',
                                'target_variable': 'first_result',
                                'expression': '($val1 + $val2) * 3 - 10'
                            },
                            {
                                'type': 'expression',
                                'target_variable': 'second_result',
                                'expression': '$first_result + sqrt($val3) * 5'
                            },
                            {
                                'type': 'max',
                                'target_variable': 'final_result',
                                'operands': [
                                    {'variable': 'first_result'},
                                    {'variable': 'second_result'},
                                    100
                                ]
                            }
                        ],
                        'input_variables': {
                            'val1': {'type': 'integer', 'default_value': 10},
                            'val2': {'type': 'integer', 'default_value': 20},
                            'val3': {'type': 'integer', 'default_value': 25}
                        }
                    }
                }
            }
        ]
    }
    
    print("===== 노드 테스트 시작 =====")
    
    # 테스트 1: conv_count_increment 노드 테스트
    print("\n--- conv_count_increment 노드 테스트 ---")
    state = {
        'conv_count': 3,
        'current_node': 'conv_count_increment',
        'workflows_config': test_workflows
    }
    result = basic_operations(state)
    print(f"초기 대화 카운트: 3")
    print(f"증가 후 대화 카운트: {result['conv_count']}")
    
    # 테스트 2: expression_math_example 노드 테스트
    print("\n--- expression_math_example 노드 테스트 ---")
    state = {
        'val1': 10,
        'val2': 20,
        'val3': 5,
        'current_node': 'expression_math_example',
        'workflows_config': test_workflows
    }
    result = basic_operations(state)
    print(f"표현식: ($val1 + $val2) * 3 - 10 + pow($val3, 2) / 10")
    print(f"입력값: val1={state['val1']}, val2={state['val2']}, val3={state['val3']}")
    print(f"계산 결과: {result['expression_result']}")
    # 수동 계산 결과와 비교
    manual_result = (10 + 20) * 3 - 10 + pow(5, 2) / 10
    print(f"수동 계산 결과: {manual_result}")
    print(f"일치 여부: {result['expression_result'] == manual_result}")
    
    # 테스트 3: compound_with_expression_example 노드 테스트
    print("\n--- compound_with_expression_example 노드 테스트 ---")
    state = {
        'val1': 10,
        'val2': 20,
        'val3': 25,
        'current_node': 'compound_with_expression_example',
        'workflows_config': test_workflows
    }
    result = basic_operations(state)
    print(f"입력값: val1={state['val1']}, val2={state['val2']}, val3={state['val3']}")
    print(f"첫 번째 표현식 결과 (first_result): {result['first_result']}")
    print(f"두 번째 표현식 결과 (second_result): {result['second_result']}")
    print(f"최종 결과 (final_result): {result['final_result']}")
    
    # 수동 계산으로 검증
    manual_first = (10 + 20) * 3 - 10
    manual_second = manual_first + 5 * pow(25, 0.5)
    manual_final = max(manual_first, manual_second, 100)
    print(f"수동 계산 - first_result: {manual_first}")
    print(f"수동 계산 - second_result: {manual_second}")
    print(f"수동 계산 - final_result: {manual_final}")
    print(f"일치 여부 - first_result: {result['first_result'] == manual_first}")
    print(f"일치 여부 - second_result: {abs(result['second_result'] - manual_second) < 0.001}")
    print(f"일치 여부 - final_result: {result['final_result'] == manual_final}")
    
    print("\n===== 노드 테스트 완료 =====")