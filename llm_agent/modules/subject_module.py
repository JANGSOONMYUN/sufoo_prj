import json
import os
import copy
from get_prompts import set_parallel_chain_builder, update_prompts_in_chains

def wrap_subjects(data):
    base_template = {
        "title": "",
        "result": "",
        "subject":[
            {
                "sub_title":"",
                "sub_result":""
            },
            {
                "sub_title":"",
                "sub_result":""
            }
            ],
        "representative_image_name": ""
        }
    
    if 'subjects' not in data:
        return {'request_template': json.dumps(base_template)}
    
    subjects = data['subjects']
    
    request = []
    for s in subjects:
        copied_template = copy.deepcopy(base_template)
        copied_template['title'] = s
        request.append(copied_template)
    
    print('='*200)
    print(request)
    
    # update chain
    chain_ptr = data['chain_ptr']
    target_proc = data['target_proc']
    target_chain = data['target_chain']
    
    set_parallel_chain_builder(entire_chains=chain_ptr, subject_list=subjects, chain_process=target_proc, target_chain_name=target_chain)
    
    with open("log/output.json", "w", encoding="utf-8") as f:
        json.dump(chain_ptr, f, indent=4, ensure_ascii=False)
    
    return {'request_template': json.dumps(request, ensure_ascii=False)}

def _convert_to_dict(unknown_data):
    if isinstance(unknown_data, dict):
        new_data = {}
        for k, v in unknown_data.items():
            new_data[k] = v
            if isinstance(v, list) or isinstance(v, dict):
                new_data[k] = _convert_to_dict(v)
            elif isinstance(v, str) or isinstance(v, int) or isinstance(v, bool):
                new_data[k] = v
            elif v is None:
                new_data[k] = v
            else:
                new_data[k] = _convert_to_dict(v.dict())
    elif isinstance(unknown_data, list):
        new_data = []
        for i, v in enumerate(unknown_data):
            new_data.append(v)
            if isinstance(v, dict) or isinstance(v, list):
                new_data[i] = _convert_to_dict(v)
            elif isinstance(v, str) or isinstance(v, int) or isinstance(v, bool):
                new_data[i] = v
            elif v is None:
                new_data[i] = v
            else:
                new_data[i] = _convert_to_dict(v.dict())
    return new_data
                
def regen_chain(data, chain_ptr):
    subjects = data['subjects']
    '''
    "subjects": [
                {
                    "subject": "맞춤 다이어트 솔루션: 여성 맞춤 다이어트 방법",
                    "sub_subjects": [
                        "건강 상태와 라이프스타일에 맞는 다이어트 방법",
                        "체중 감량 목표 설정 및 계획 수립 가이드",
                        "다이어트 성공을 위한 식단 및 운동 팁",
                        "다이어트 중 흔히 겪는 어려움과 해결 방안"
                    ]
                },
                {
                    "subject": "다이어트 정체기 극복: 체중 감량 멈춤 현상 해결 전략",
                    "sub_subjects": [
                        "정체기 원인 분석 및 극복 방법",
                        "식단 및 운동 루틴 변화 주기",
                        "수분 섭취 및 스트레스 관리 중요성",
                        "다이어트 일기 작성 및 피드백 활용"
                    ]
                }, ...
            ],
    '''
    
    '''
# 입력 템플릿 (참고)
    "title": "",    # 주제 제목
    "result": "",   # 주제에 대한 해설
    "subject":[     # 소주제 list; 소주제는 1~2개 정도로 구성
        {
            "sub_title":"", # 소주제 제목
            "sub_result":"" # 소주제에 대한 해설
        },
        {
            "sub_title":"", # 소주제 제목
            "sub_result":"" # 소주제에 대한 해설
        }
        ],
    "representative_image_name": "" # 본 주제에 어울리는 사진의 이름

    '''
    base_template = {
        "title": "",
        "result": "",
        "subject":[],
        "representative_image_name": ""
        }
    base_template_sub = {
                "sub_title":"",
                "sub_result":""
        }
    
    # apply template
    new_subjects = []
    for sub in subjects:
        # 딕셔너리로 변환
        sub_dict = sub.dict()
        
        title = sub_dict['subject']
        sub_subjects = sub_dict['sub_subjects']
        copied_template = copy.deepcopy(base_template)
        copied_template['title'] = title
        for sub_sub in sub_subjects:
            copied_subsub = copy.deepcopy(base_template_sub)
            copied_subsub['sub_title'] = sub_sub
            copied_template['subject'].append(copied_subsub)
        
        new_subjects.append(copied_template)
        
    
    # update chain
    chain_info = data['chain_info']
    target_proc = chain_info['target_proc']
    target_chain = chain_info['target_chain']
    json_path = chain_info['json_path']
    
    # print data
    try:
        json_data = _convert_to_dict(data)
        with open("log/output_data.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(e)
    
    chain_process = chain_ptr['process'][target_proc]

    replaced_chains, post_input_keys = set_parallel_chain_builder(entire_chains=chain_ptr, subject_list=subjects, chain_process=chain_process, target_chain_name=target_chain)
    
    
            

    new_data = _convert_to_dict(chain_ptr)
    with open("log/chain_data.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    
    
    update_prompts_in_chains(json_path=json_path, chains=chain_ptr)
    
    
    return {'result_regen_chain': 'ok'}

def wrap_reports(data):
    
    requests = []
    
    for k, v in data.items():
        requests.append(v)
        
    return {'request': requests}