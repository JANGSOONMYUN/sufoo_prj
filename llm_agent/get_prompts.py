import json
import os
import copy
import traceback

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

def get_history_id(chains, target_process_name):
    def _get_config(_process, _target_process_name):
        if _process[_target_process_name] is None:
            return None
        for k, v in _process[_target_process_name].items():
            if 'config' in chains[k]:
                return chains[k]['config']
            result = _get_config(_process[_target_process_name], k)
            if result is not None:
                return result
        return None
    
    config_val = _get_config(chains['process'], target_process_name)
    if config_val is None:
        print('config_val is None')
        config_val = {"configurable": {"user_id": "123", "conversation_id": "1"}}
    return config_val['configurable']['user_id'], config_val['configurable']['conversation_id'] 

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
                prompts_list.append(f'''{title} - {sub_sub['title']}({sub_sub['description']})''')
                prompts_list_string += '\n"' + prompts_list[-1] + '"'
        else:
            prompts_list.append(f'''{title}({sub['description']})''')
            prompts_list_string += '\n"' + prompts_list[-1] + '"'
    
    prompts_list_string = '[' + prompts_list_string + '\n]'
    return prompts_list, prompts_list_string
    
def _is_target_chain_exist(process, chain_name):
    for k, v in process.items():
        if k == chain_name:
            return True
        if v is None:
            continue
        return _is_target_chain_exist(v, chain_name)
    return False

def _get_post_related_chains(process, chain_name):
    for k, v in process.items():
        if k != chain_name:
            return _get_post_related_chains(v, chain_name)
        return v

def _get_pre_related_chain(process, chain_name):
    for k, v in process.items():
        if chain_name not in v:
            return _get_pre_related_chain(v, chain_name)
        return k
    
def _put_data_to_end_of_process(process, put_data_to_the_end):
    is_first = True
    for k, v in process.items():
        if is_first and (v is None or v == {}):
            process[k] = put_data_to_the_end
        is_first = False
        if v is None:
            continue
        _put_data_to_end_of_process(v, put_data_to_the_end)

def _replace_process(process, chain_name, replaced_chains):
    for k, v in process.items():
        if chain_name not in v:
            return _replace_process(v, chain_name, replaced_chains)
        
        keep_value = copy.deepcopy(v[chain_name])
        del process[k][chain_name]
        keep_remaining_process = copy.deepcopy(process[k])
        process[k] = replaced_chains
        # to keep the process order
        if keep_remaining_process is not None and len(keep_remaining_process):
            process[k].update(keep_remaining_process)
        _put_data_to_end_of_process(process, keep_value)
        
        return process

'''
    each item of subject_list is placed to the end of its prompt
    e.g.) [prompts]
            blah blah
            this is prompts
            blah blah
            {an item of subject_list}
'''
def _pre_proc_parallel_chain(entire_chains, subject_list, target_chain_name):
    post_input_keys = []
    replaced_chains = {}
    
    try:
        target_chain_setting = entire_chains[target_chain_name]
            
        first_key = None
        for i, sub_str in enumerate(subject_list):
            
            new_chain_setting = copy.deepcopy(target_chain_setting)
            chain_name = f'{target_chain_name}_{i}'
            preset_output_dict = new_chain_setting['output_dict']
            
            if 'keep_input_output_data' not in new_chain_setting:
                new_chain_setting['keep_input_output_data'] = []
            
            output_dict = {}
            for _k, _v in preset_output_dict.items():
                key_name = f'{_k}_{i}'
                output_dict[key_name] = _v
                new_chain_setting['keep_input_output_data'].append(key_name)
                post_input_keys.append(key_name)
            
            new_chain_setting['prompts_data'].append(sub_str)
                
            new_chain_setting['type'] = 'prompts'
            new_chain_setting['output_dict'] = output_dict
            
            entire_chains[chain_name] = new_chain_setting
            
            if first_key is None:
                first_key = chain_name
                replaced_chains[chain_name] = {}
            else:
                replaced_chains[chain_name] = None
        
    except Exception as e:
        # Print the exception and the traceback details
        print(f"Error: {e}")
        traceback.print_exc()
        
    return replaced_chains, post_input_keys

def set_parallel_chain_builder(entire_chains, subject_list, chain_process, target_chain_name):
    try:
        if _is_target_chain_exist(chain_process, target_chain_name) is False:
            return
        
        replaced_chains, post_input_keys = _pre_proc_parallel_chain(entire_chains, subject_list, target_chain_name)
        post_input_dict = {}
        for k in post_input_keys:
            post_input_dict[k] = None

        post_chains = _get_post_related_chains(chain_process, target_chain_name)
        if post_chains is not None:
            for k, v in post_chains.items():
                entire_chains[k]['input_dict'].update(post_input_dict)

        _replace_process(chain_process, target_chain_name, replaced_chains)

        return replaced_chains, post_input_keys
        
    except Exception as e:
        # Print the exception and the traceback details
        print(f"Error: {e}")
        traceback.print_exc()
        
        return None, None
    
def replace_prompts(directory_path, chain_setting, default_out_key, key_name):
    # load propmts and replace by output keys
    prompts_data = chain_setting['prompts_data']
    for pi, p_data in enumerate(prompts_data):
        if '.txt' not in p_data:
            loaded_data = p_data
        else:
            loaded_data = load_txt_data(os.path.join(directory_path, p_data))
        prompts_data[pi] = loaded_data.replace('{' + default_out_key + '}', '{' + key_name + '}')

def set_double_parallel_chain_builder(entire_chains, subject_list, chain_process, target_chain_1st, target_chain_2nd, json_path):
    
    try:
        directory_path = os.path.dirname(json_path)
        if _is_target_chain_exist(chain_process, target_chain_1st) is False:
            return
        if _is_target_chain_exist(chain_process, target_chain_2nd) is False:
            return
        
        first_chain = entire_chains[target_chain_1st]
        second_chain = entire_chains[target_chain_2nd]
        first_output_dicts = list(first_chain['output_dict'].keys())
        second_output_dicts = list(second_chain['output_dict'].keys())
        
        replaced_chains_1, post_input_keys_1 = set_parallel_chain_builder(entire_chains, subject_list, chain_process, target_chain_1st)
        replaced_chains_2, post_input_keys_2 = set_parallel_chain_builder(entire_chains, subject_list, chain_process, target_chain_2nd)
        
        
        for i, k in enumerate(list(replaced_chains_2.keys())):
            chain_setting = entire_chains[k]
            
            # remove other input
            for inkey in post_input_keys_1:
                del chain_setting['input_dict'][inkey]
            
            # add target input
            for default_out_key in first_output_dicts:
                del chain_setting['input_dict'][default_out_key]
                key_name = f'{default_out_key}_{i}'
                chain_setting['input_dict'][key_name] = None

                # load propmts and replace by output keys
                replace_prompts(directory_path, chain_setting, default_out_key, key_name)

        return second_output_dicts, post_input_keys_2
        
    except Exception as e:
        # Print the exception and the traceback details
        print(f"Error: {e}")
        traceback.print_exc()
    return None

if __name__ == "__main__":
    chains = update_prompts_in_chains(json_path='/home/user/chatgpt_module_test/settings/prompts/tarot/chains_time_chat.json')
    user_id, conversation_id = get_history_id(chains, 'test')
    print(user_id, conversation_id )