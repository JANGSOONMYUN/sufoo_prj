import json

def fix_partial_json(partial_json_str):
    stack = []  # JSON 객체와 배열을 추적하기 위한 스택
    current_obj = None  # 현재 처리 중인 JSON 객체 또는 배열
    current_key = None  # 현재 처리 중인 키 (사전의 경우)
    i = 0
    
    while i < len(partial_json_str):
        char = partial_json_str[i]
        
        if char == '{':
            if current_obj is not None:
                # 현재 객체를 스택에 추가
                stack.append((current_obj, current_key))
            current_obj = {}  # 새로운 사전 객체 생성
            current_key = None
        elif char == '[':
            if current_obj is not None:
                # 현재 배열을 스택에 추가
                stack.append((current_obj, current_key))
            current_obj = []  # 새로운 리스트 객체 생성
            current_key = None
        elif char == '}':
            if stack:
                parent, key = stack.pop()
                if isinstance(parent, dict):
                    parent[key] = current_obj  # 부모 사전에 현재 객체 추가
                elif isinstance(parent, list):
                    parent.append(current_obj)  # 부모 리스트에 현재 객체 추가
                current_obj = parent
            else:
                break  # 잘못된 닫힘 괄호
        elif char == ']':
            if stack:
                parent, key = stack.pop()
                if isinstance(parent, dict):
                    parent[key] = current_obj  # 부모 사전에 현재 리스트 추가
                elif isinstance(parent, list):
                    parent.append(current_obj)  # 부모 리스트에 현재 리스트 추가
                current_obj = parent
            else:
                break  # 잘못된 닫힘 괄호
        elif char == '"':
            end_quote = i + 1
            while end_quote < len(partial_json_str):
                # if partial_json_str[end_quote] == '\"':
                #     pass
                # if partial_json_str[end_quote] == r'\\':
                #     print(partial_json_str[end_quote])
                if partial_json_str[end_quote] == '"':
                    # print(end_quote, partial_json_str[end_quote])
                    break
                end_quote += 1
            if end_quote < len(partial_json_str):
                if current_key is None:
                    current_key = partial_json_str[i + 1:end_quote]  # 키 설정
                    if isinstance(current_obj, list):
                        current_obj.append(current_key)
                else:
                    value = partial_json_str[i + 1:end_quote]  # 값 설정
                    if isinstance(current_obj, dict):
                        current_obj[current_key] = value
                    elif isinstance(current_obj, list):
                        current_obj.append(value)
                    
                    current_key = None
                i = end_quote
        # elif char.isdigit() or char == '-' or char == '.':
        #     end_num = i
        #     while end_num < len(partial_json_str) and (partial_json_str[end_num].isdigit() or partial_json_str[end_num] in '.-eE'):
        #         end_num += 1
        #     num = partial_json_str[i:end_num]
        #     if isinstance(current_obj, dict) and current_key is not None:
        #         current_obj[current_key] = json.loads(num)  # 숫자 값을 사전에 추가
        #     elif isinstance(current_obj, list):
        #         current_obj.append(json.loads(num))  # 숫자 값을 리스트에 추가
        #     i = end_num - 1
        elif char == 't' and partial_json_str[i:i+4] == 'true':
            if isinstance(current_obj, dict) and current_key is not None:
                current_obj[current_key] = True  # True 값을 사전에 추가
                current_key = None
            elif isinstance(current_obj, list):
                current_obj.append(True)  # True 값을 리스트에 추가
            i += 3
        elif char == 'f' and partial_json_str[i:i+5] == 'false':
            if isinstance(current_obj, dict) and current_key is not None:
                current_obj[current_key] = False  # False 값을 사전에 추가
                current_key = None
            elif isinstance(current_obj, list):
                current_obj.append(False)  # False 값을 리스트에 추가
            i += 4
        elif char == 'n' and partial_json_str[i:i+4] == 'null':
            if isinstance(current_obj, dict) and current_key is not None:
                current_obj[current_key] = None  # null 값을 사전에 추가
                current_key = None
            elif isinstance(current_obj, list):
                current_obj.append(None)  # null 값을 리스트에 추가
            i += 3
        elif char == ',':
            current_key = None  # 다음 키/값 쌍을 처리하기 위해 초기화
        i += 1
    
    # 남아있는 모든 구조를 닫음
    while stack:
        parent, key = stack.pop()
        if isinstance(parent, dict):
            parent[key] = current_obj
        elif isinstance(parent, list):
            parent.append(current_obj)
        current_obj = parent
    
    # 모든 열린 구조를 제대로 닫도록 보장
    if isinstance(current_obj, dict):
        return close_dict(current_obj)
    elif isinstance(current_obj, list):
        return close_list(current_obj)
    else:
        return current_obj

def close_dict(obj):
    # 사전 객체의 모든 중첩된 구조를 재귀적으로 닫음
    if isinstance(obj, dict):
        for key in obj:
            if isinstance(obj[key], dict):
                obj[key] = close_dict(obj[key])
            elif isinstance(obj[key], list):
                obj[key] = close_list(obj[key])
    return obj

def close_list(lst):
    # 리스트 객체의 모든 중첩된 구조를 재귀적으로 닫음
    closed_list = []
    for item in lst:
        if isinstance(item, dict):
            closed_list.append(close_dict(item))
        elif isinstance(item, list):
            closed_list.append(close_list(item))
        else:
            closed_list.append(item)
    return closed_list

'''
    # json 형식의 string 에서 닫혀있는 곳의 끝의 콤마 제거
    {
        "title": "친구/동료 관계",
        "result": "2024년 하반기 동안 친구 및 동료와의 관계에서 긍정적인 변화가 예상됩니다.",
    }
    여기서 result 의 끝에 콤마가 있으면 안된다.
'''
def remove_comma_before_bracket(s):
    # '}' 문자가 나타날 때까지 공백 및 줄바꿈 문자를 무시하고 콤마(,)가 있는지 검사
    i = 0
    while i < len(s):
        if s[i] == '}' or s[i] == ']':
            # 역순으로 콤마가 있는지 확인
            j = i - 1
            while j >= 0:
                if s[j] == ',':
                    s = s[:j] + ' ' + s[j+1:]
                    break
                elif s[j] not in [' ', '\n', '\t']:  # 공백, 줄바꿈 등을 무시
                    break
                j -= 1
        i += 1
    return s