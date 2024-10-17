import json
import os
import copy
import traceback

def get_character(json_path, user_id = "default"):
    # Specify the path to your JSON file
    file_path = json_path

    # Load the JSON data from the file
    with open(file_path, 'r',  encoding="utf-8") as file:
        data = json.load(file)
        
    return data[user_id]

def get_character_ids(json_path):
    # Load the JSON data from the file
    with open(json_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        # Deleting a key using the del statement
        if "default" in data:
            del data["default"]
        keys_list = data.keys()

    return keys_list

def num_character(json_path):
    # Load the JSON data from the file
    with open(json_path, 'r',  encoding="utf-8") as file:
        data = json.load(file)
        number_of_keys = len(data)
        return number_of_keys - 1
    
def load_json_file(json_path):
    with open(json_path, 'r',  encoding="utf-8") as file:
        data = json.load(file)
    return data

# --- prompts for tarot application ---
def load_other_prompts(json_path):
    return load_json_file(json_path)

def save_other_prompts(json_path, prompts, prompt_type):
    data = load_other_prompts(json_path)
    data[prompt_type] = prompts
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

def save_variables(json_path, variables):
    save_other_prompts(json_path, variables, prompt_type='variables')
    # data = load_other_prompts(json_path)
    # data['variables'] = variables
    # with open(json_path, "w", encoding="utf-8") as json_file:
    #     json.dump(data, json_file, indent=4, ensure_ascii=False)

def save_check_questions(json_path, check_questions):
    save_other_prompts(json_path, check_questions, prompt_type='check_questions')
    
    
# --- for LangChain
def load_txt_data(txt_path):
    with open(txt_path, 'r') as file:
        file_content = file.read()
    return file_content

def _valid_chain_params(chain_key, chain_val):
    
    if 'prompt' in chain_val['type']:
        check_list = ['instructions_data', 'prompts_data', 'input_dict', 'output_dict', 'config']
    elif 'func' in chain_val['type']:
        check_list = ['input_dict', 'output_dict', 'output_type', 'output']
    else:
        assert False, f'type is not in the chain "{chain_key}". in _valid_chain_params()'
        
    for c in check_list:
        if c not in chain_val:
            assert False, f'{c} is not in the chain "{chain_key}". in _valid_chain_params()'
            
    if 'prompt' in chain_val['type']:
        for k in chain_val['input_dict'].keys():
            if f'{{{k}}}' in chain_val['prompts']:
                continue
            if f'{{{k}}}' in chain_val['instructions']:
                continue
            assert False, f'{{{k}}} must be in input_dict in the chain "{chain_key}" and prompts file. in _valid_chain_params()'
        

def _valid_chain_in_out(parent_chain, child_chain, chains, keep_data_set):
    if child_chain is None:
        return True
    parent_output_keys = chains[parent_chain]['output_dict']
    # save keep_input_output_data
    if 'keep_input_output_data' in chains[parent_chain]:
        for kp_data in chains[parent_chain]['keep_input_output_data']:
            keep_data_set.add(kp_data)
            
    for k, v in child_chain.items():
        if k not in chains:
            assert False, f'{k} is not in the chains list. in _valid_chain()'
        _valid_chain_params(k, chains[k])        
        # save keep_input_output_data
        if 'keep_input_output_data' in chains[k]:
            for kp_data in chains[k]['keep_input_output_data']:
                keep_data_set.add(kp_data)
                
    for k, v in child_chain.items():
        input_keys = chains[k]['input_dict'].keys()
                
        if parent_output_keys is None or len(parent_output_keys) == 0:
            if 'question' not in input_keys and 'message' not in input_keys:
                assert False, f'input_keys must include question or message in the chain "{k}", in _valid_chain()'
        elif 'auto' not in chains[k]['type']:
            for in_k in input_keys:
                if in_k not in parent_output_keys.keys() and in_k not in keep_data_set:
                    # assert False, f'input_dict must include "{in_k}" in the chain "{k}", in _valid_chain()'
                    print('[WARN] ' + f'input_dict must include "{in_k}" in the chain "{k}", in _valid_chain()')
        
        _valid_chain_in_out(k, v, chains, keep_data_set)
        
            
        
def validate_chain_setting(chains, target_process_name=None):
    process = chains['process']
    keep_data_set = set()
    if target_process_name is None:
        for k, v in process.items():
            subject = k
            for sub_k, sub_v in v.items():
                _valid_chain_in_out(sub_k, sub_v, chains, keep_data_set)
    else:
        print(process)
        k, v = target_process_name, process[target_process_name]
        for sub_k, sub_v in v.items():
            _valid_chain_in_out(sub_k, sub_v, chains, keep_data_set)
        
        
    

# def load_chain_setting(json_path):
#     chains = load_json_file(json_path)
#     directory_path = os.path.dirname(json_path)
#     for k, v in chains.items():
#         if 'instructions_data' not in v or 'prompts_data' not in v:
#             continue
#         chains[k]['instructions'] = ''
#         chains[k]['prompts'] = ''
#         for i_data in v['instructions_data']:
#             if '.txt' not in i_data:
#                 chains[k]['instructions'] += i_data
#             else:
#                 chains[k]['instructions'] += load_txt_data(os.path.join(directory_path, i_data))
#         for p_data in v['prompts_data']:
#             if '.txt' not in p_data:
#                 chains[k]['prompts'] += p_data
#             else:
#                 chains[k]['prompts'] += load_txt_data(os.path.join(directory_path, p_data))
    
#     validate_chain_setting(chains)
#     return chains

def update_prompts_in_chains(json_path, chains=None):
    if chains is None:
        chains = load_json_file(json_path)
    directory_path = os.path.dirname(json_path)
    for k, v in chains.items():
        if 'instructions_data' not in v or 'prompts_data' not in v:
            continue
        chains[k]['instructions'] = ''
        chains[k]['prompts'] = ''
        for i_data in v['instructions_data']:
            if '.txt' not in i_data:
                chains[k]['instructions'] += i_data
            else:
                chains[k]['instructions'] += load_txt_data(os.path.join(directory_path, i_data))
        for p_data in v['prompts_data']:
            # print('0-'*100)
            # print(p_data)
            if '.txt' not in p_data:
                chains[k]['prompts'] += p_data
            else:
                chains[k]['prompts'] += load_txt_data(os.path.join(directory_path, p_data))
    
    return chains

def load_chain_setting(json_path):
    chains = load_json_file(json_path)
    return update_prompts_in_chains(json_path=json_path, chains=chains)

def convert_str_list_to_json(str_list):
    combined_str = ''
    if not isinstance(str_list, list):
        assert False, f'input value must be list but it is {type(str_list)}, in convert_str_list_to_json()'
    for s in str_list:
        combined_str += s
        
    # to json
    converted_json = json.loads(combined_str)
    return converted_json

def subject_separator(request):
    copied_request = copy.deepcopy(request)
    copied_request['key_name'] = 'request'
    if 'subjects' not in copied_request:
        return [[copied_request]]
        
    del copied_request['subjects']
    product_info = copied_request
    
    def _get_stages(_subject, stage_list):
        if 'stage_level' not in _subject:
            _subject['stage_level'] = 2
        stage_list.append(_subject['stage_level'])
        if 'subjects' in _subject:
            for val in _subject['subjects']:
                _get_stages(val, stage_list)
 
    stage_list = []
    _get_stages(request, stage_list)
    stage_list = sorted(list(set(stage_list)))
    layered_subjects = [[] for i in range(len(stage_list))]
    
    # product info
    stage_level = product_info['stage_level']
    layer_idx = stage_list.index(stage_level)
    layered_subjects[layer_idx].append(product_info)
    
    subjects = copy.deepcopy(request['subjects'])
    for i, val in enumerate(subjects):
        stage_level = val['stage_level']
        layer_idx = stage_list.index(stage_level)
        
        layer_item = copy.deepcopy(val)
        layer_item['key_name'] = f'request_subjects_{i}'
        layered_subjects[layer_idx].append(layer_item)
        # print(layer_idx, ' =======' , layer_item)


    # for l in layered_subjects:
    #     print('-'*50)
    #     print(l)
    
    return layered_subjects

def chain_builder(entire_chains, layered_subjects, chain_process, target_chain_name):
    try:
        def _is_target_chain_exist(process, chain_name):
            for k, v in process.items():
                if k == chain_name:
                    return True
                if v is None:
                    continue
                return _is_target_chain_exist(v, chain_name)
            return False
        
        if _is_target_chain_exist(chain_process, target_chain_name) is False:
            return
        
        target_chain_setting = entire_chains[target_chain_name]
            
        def _get_related_chains(process, chain_name):
            for k, v in process.items():
                if k != chain_name:
                    return _get_related_chains(v, chain_name)
                return v
        
        def _get_pre_related_chain(process, chain_name):
            for k, v in process.items():
                if chain_name not in v:
                    return _get_pre_related_chain(v, chain_name)
                return k
            
        def _get_end_of_process(process, put_data_to_the_end):
            is_first = True
            for k, v in process.items():
                if is_first and v is None:
                    process[k] = put_data_to_the_end
                is_first = False
                if v is None:
                    continue
                _get_end_of_process(v, put_data_to_the_end)
        
        def _update_process(process, chain_name, replaced_chains):
            for k, v in process.items():
                if chain_name not in v:
                    return _update_process(v, chain_name, replaced_chains)
                
                keep_value = copy.deepcopy(v[chain_name])
                del process[k][chain_name]
                            
                process[k] = replaced_chains
                _get_end_of_process(process, keep_value)
                
                return process
        pre_output_dict = {}
        post_input_dict = {}
        replaced_chains = {}
        
        chain_layer_ptr = replaced_chains
        for i, layer_sub in enumerate(layered_subjects):
            if i >= 0:
                pass
            chains_in_layer = {}
            first_key = None
            for k, sub in enumerate(layer_sub):
                new_chain_setting = copy.deepcopy(target_chain_setting)
                chain_name = f'created_layer_{i}_{k}'
                key_name = sub['key_name']
                output_dict = {key_name: 'title과 description을 바탕으로 result에 주제에 해당하는 사주풀이를 상세히 한다'}
                
                input_dict_name = f'in_{key_name}'
                new_chain_setting['type'] = 'prompts_auto'
                new_chain_setting['input_dict'] = {input_dict_name: None}
                new_chain_setting['output_dict'] = output_dict
                new_chain_setting['keep_input_output_data'] = [key_name]
                
                new_chain_setting['instructions_data'].append('\n')
                new_chain_setting['prompts_data'].append('\n# 입력값(입력값 형식을 유지하여 출력)\n{' + input_dict_name + '}')
                
                post_input_dict[key_name] = None
                pre_output_dict[input_dict_name] = None
                
                entire_chains[chain_name] = new_chain_setting
                
                chains_in_layer[chain_name] = None
                if first_key is None:
                    first_key = chain_name
            # print(chain_layer_ptr)
            # print(chains_in_layer)
            
                
            chain_layer_ptr.update(chains_in_layer)
            if i < len(layered_subjects) - 1:
                chain_layer_ptr[first_key] = {}
                chain_layer_ptr = chain_layer_ptr[first_key]
        
        
        pre_chain = _get_pre_related_chain(chain_process, target_chain_name)
        entire_chains[pre_chain]['output_dict'] = pre_output_dict
        post_chains = _get_related_chains(chain_process, target_chain_name)
        for k, v in post_chains.items():
            entire_chains[k]['input_dict'].update(post_input_dict)
            
        _update_process(chain_process, target_chain_name, replaced_chains)
        
        # print(replaced_chains)
        # print(chain_process)
    except Exception as e:
        # Print the exception and the traceback details
        print(f"Error: {e}")
        traceback.print_exc()
    
'''
    to generate propmts for streaming with string list
    intput: request dict
'''
def generate_prompts_for_stream_string_list(request_data):
    prompts_list = []
    prompts_list_string = ''
    subjects = request_data['subjects']
    for sub in subjects:
        title = sub['title']
        if 'sub_subjects' in sub:
            for sub_sub in sub['sub_subjects']:
                prompts_list.append(f'''{title} - {sub_sub['title']}''')
                prompts_list_string += '\n"' + prompts_list[-1] + '"'
        else:
            prompts_list.append(title)
            prompts_list_string += '\n"' + prompts_list[-1] + '"'
    
    prompts_list_string = '[' + prompts_list_string + '\n]'
    return prompts_list, prompts_list_string
    

if __name__ == "__main__":        
    data = {
  "stage_level": 2,
  "title": "운명적인 사주 궁합 분석",
  "description": "두 사람의 사주를 심층적으로 분석하여 운명적인 궁합을 알아봅니다.",
  "result": "",
  "subjects": [
    {
      "stage_level": 1,
      "title": "기본 사주 분석",
      "description": "각 개인의 사주를 기본적으로 분석합니다.",
      "result": "",
      "sub_subjects": [
        {
          "title": "천간과 지지 분석",
          "description": "개인의 천간과 지지를 분석하여 기본 성향을 파악합니다.",
          "result": ""
        },
        {
          "title": "오행의 분포와 균형",
          "description": "사주에서 오행의 분포를 분석하여 균형 상태를 파악합니다.",
          "result": ""
        }
      ]
    },
    {
      "stage_level": 1,
      "title": "운명적 궁합도",
      "description": "두 사람의 사주를 합하여 운명적인 궁합을 분석합니다.",
      "result": "",
      "sub_subjects": [
        {
          "title": "인연의 깊이와 강도",
          "description": "사주를 통해 두 사람의 인연이 얼마나 깊고 강한지 분석합니다.",
          "result": ""
        },
        {
          "title": "상호 보완과 충돌 요소",
          "description": "사주에서 나타나는 상호 보완적 요소와 충돌 요소를 파악합니다.",
          "result": ""
        }
      ]
    },
    {
      "stage_level": 1,
      "title": "연애와 인간관계 운세",
      "description": "두 사람의 연애와 전반적인 인간관계 운세를 분석합니다.",
      "result": "",
      "sub_subjects": [
        {
          "title": "연애운의 흐름과 시기",
          "description": "사주를 통해 연애운의 흐름과 좋은 시기를 파악합니다.",
          "result": ""
        },
        {
          "title": "친구 및 사회적 관계 운세",
          "description": "사주를 통해 친구, 동료 등 사회적 관계의 조화와 운세를 분석합니다.",
          "result": ""
        }
      ]
    },
    {
      "stage_level": 1,
      "title": "성격과 상호 작용",
      "description": "두 사람의 성격과 그 상호 작용을 분석합니다.",
      "result": "",
      "sub_subjects": [
        {
          "title": "성격 특성 비교",
          "description": "각자의 성격 특성을 분석하여 서로의 이해를 돕습니다.",
          "result": ""
        },
        {
          "title": "성격적 조화와 갈등 요소",
          "description": "사주를 통해 성격적 조화와 잠재적인 갈등 요소를 파악합니다.",
          "result": ""
        }
      ]
    },
    {
      "stage_level": 1,
      "title": "미래의 공동 운세",
      "description": "두 사람이 함께 했을 때의 미래 운세를 분석합니다.",
      "result": "",
      "sub_subjects": [
        {
          "title": "재물운과 성공 가능성",
          "description": "함께 했을 때의 재물운과 성공 가능성을 사주로 분석합니다.",
          "result": ""
        },
        {
          "title": "건강 운세와 주의사항",
          "description": "두 사람의 건강 운세를 파악하고 주의할 점을 제시합니다.",
          "result": ""
        },
        {
          "title": "관계 지속 기간 예측",
          "description": "사주를 통해 두 사람의 관계가 얼마나 길게 지속될지 예측합니다.",
          "result": ""
        }
      ]
    },
    {
      "stage_level": 1,
      "title": "관계 개선 및 유지 방법",
      "description": "관계를 더욱 좋게 만들 수 있는 방법을 제안합니다.",
      "result": "",
      "sub_subjects": [
        {
          "title": "상호 이해와 배려",
          "description": "서로를 이해하고 배려하는 방법을 사주를 통해 제시합니다.",
          "result": ""
        },
        {
          "title": "갈등 해결 전략",
          "description": "사주에서 나타나는 갈등 요소를 기반으로 해결 전략을 제공합니다.",
          "result": ""
        },
        {
          "title": "행운을 높이는 방법",
          "description": "관계의 행운을 높일 수 있는 방법을 사주를 통해 제안합니다.",
          "result": ""
        }
      ]
    },
    {
      "stage_level": 1,
      "title": "사주 총평",
      "description": "두 사람의 사주를 종합적으로 평가하여 전반적인 조화와 운세를 총평합니다.",
      "result": "",
      "sub_subjects": [
        {
          "title": "관계의 전반적인 조화도",
          "description": "두 사람의 사주를 기반으로 전반적인 관계의 조화도를 평가합니다.",
          "result": ""
        },
        {
          "title": "사주를 통한 인생의 큰 흐름",
          "description": "사주를 통해 인생에서 중요한 흐름과 변화를 종합적으로 평가합니다.",
          "result": ""
        }
      ]
    }
  ]
}

    result, prompts_list_string = generate_prompts_for_stream_string_list(data)
    for r in result:
        print(r)
    print(prompts_list_string)
    # layered_subjects = [
    # [{'stage_level': 0, 'title': '사랑과 연애', 'description': '2024년 하반기 사랑과 연애 관련 운세, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': '', 'sub_subjects': [{'title': '연애운', 'description': '연애의 흐름과 기회 분석, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}, {'title': '결혼운', 'description': '결혼 가능성 및 결혼 생활에 대한 운세, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}], 'key_name': 'request_subjects_1'}, {'stage_level': 0, 'title': '재물운', 'description': '2024년 하반기 금전적 운세와 재물 관리에 대한 조언, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': '', 'sub_subjects': [{'title': '수입', 'description': '수입의 흐름과 증가 가능성, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}, {'title': '지출', 'description': '지출 관리와 절약 방법, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}], 'key_name': 'request_subjects_2'}, {'stage_level': 0, 'title': '직업운', 'description': '2024년 하반기 직업과 관련된 운세, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': '', 'sub_subjects': [{'title': '직장생활', 'description': '직장 내 성공 가능성과 인간관계 분석, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}, {'title': '사업운', 'description': '자영업이나 창업의 운세와 성공 가능성, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}], 'key_name': 'request_subjects_3'}, {'stage_level': 0, 'title': '건강운', 'description': '2024년 하반기 건강 상태와 관련된 운세, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': '', 'sub_subjects': [{'title': '신체 건강', 'description': '전반적인 신체 건강에 대한 분석, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}, {'title': '정신 건강', 'description': '정신적인 안정과 건강 상태, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}], 'key_name': 'request_subjects_4'}, {'stage_level': 0, 'title': '대인관계운', 'description': '2024년 하반기 대인관계와 관련된 운세, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': '', 'sub_subjects': [{'title': '친구/동료 관계', 'description': '친구 및 동료와의 관계 분석, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}, {'title': '가족 관계', 'description': '가족 간의 화합과 갈등에 대한 운세, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}], 'key_name': 'request_subjects_5'}, {'stage_level': 0, 'title': '주요 사건 예측 - 조심해야 할 것들에 대해 사주풀이', 'description': '2024년 하반기에 조심해야 할 주요 사건과 상황을 예측하고 사주를 통한 해석, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': '', 'sub_subjects': [{'title': '주의사항', 'description': '각 개인이 2024년 하반기에 조심해야 할 사건과 상황에 대한 조언, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}], 'key_name': 'request_subjects_6'}, {'stage_level': 0, 'title': '월별 운세 예측 - 2024년 9월부터 12월까지', 'description': '2024년 9월부터 12월까지의 월별 운세 분석 및 조언 제공', 'result': '', 'sub_subjects': [{'title': '9월', 'description': '각 월별 운세, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}, {'title': '10월', 'description': '각 월별 운세, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}, {'title': '11월', 'description': '각 월별 운세, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}, {'title': '12월', 'description': '각 월별 운세, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}], 'key_name': 'request_subjects_7'}, {'stage_level': 0, 'title': '고객 질문에 대한 답변', 'description': '고객이 추가적으로 입력한 질문(question)에 대해 사주팔자를 바탕으로 해석', 'result': '', 'sub_subjects': [{'title': '고객 질문', 'description': '개인별로 입력된 질문을 바탕으로 해석을 제공', 'result': ''}], 'key_name': 'request_subjects_8'}]
    # ,
    # [{'stage_level': 1, 'title': '신년 운세 종합 분석', 'description': '2024년 하반기 전반에 대한 운세 분석; 총평', 'result': '', 'sub_subjects': [{'title': '총운', 'description': '2024년 하반기 전체적인 운을 분석, 고객의 천간지지, 십신, 대운, 세운으로 상세 풀이', 'result': ''}, {'title': '조언', 'description': '기회나 문제에 대한 조언 제공', 'result': ''}], 'key_name': 'request_subjects_0'}]
    # ,
    # [{'stage_level': 2, 'title': '2024년 하반기 운세 분석', 'description': '2024년 하반기의 운세 분석; 요약', 'result': '', 'key_name': 'request'}]
    # ]
            
    # json_path = '/home/user/chatgpt_module_test/settings/prompts/saju/chains.json'
    # chains = {}
    # with open(json_path, 'r',  encoding="utf-8") as file:
    #     chains = json.load(file)
        
        
    # chain_builder(chains, layered_subjects=layered_subjects, 
    #             chain_process=chains['process']['saju2'], 
    #             target_chain_name='layered_report')

    # # assert False
    # # print(chains)


    # import sys
    # import os
        
    # # def validate_chain_setting(chains):
    # # def load_chain_setting(json_path):

    # chains = _load_chain_setting(chains=chains, json_path=json_path)
    # # print(chains)

    # validate_chain_setting(chains)

    # print(chains)