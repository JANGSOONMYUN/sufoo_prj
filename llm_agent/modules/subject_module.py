import json
import os
import copy
from get_prompts import set_parallel_chain_builder

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

def regen_chain(data, chain_ptr):
    subjects = data['subjects']
    # update chain
    target_proc = data['target_proc']
    target_chain = data['target_chain']
    
    with open("log/output_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    with open("log/output.json", "w", encoding="utf-8") as f:
        json.dump(chain_ptr, f, indent=4, ensure_ascii=False)
    
    chain_process = chain_ptr['process'][target_proc]
    replaced_chains, post_input_keys = set_parallel_chain_builder(entire_chains=chain_ptr, subject_list=subjects, chain_process=chain_process, target_chain_name=target_chain)
    
    with open("log/output2.json", "w", encoding="utf-8") as f:
        json.dump(chain_ptr, f, indent=4, ensure_ascii=False)
    
    return {'result_regen_chain': 'ok'}

def wrap_reports(data):
    
    requests = []
    
    for k, v in data.items():
        requests.append(v)
        
    return {'request': requests}