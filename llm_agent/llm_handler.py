import os
import json
import time
import datetime
import pytz
from datetime import timedelta
from datetime import datetime
from PIL import Image, ExifTags
import threading
import logging
import logging.handlers
from logging.handlers import RotatingFileHandler
import sys

# LLM
import openai
from openai import OpenAI

# modules
from get_config import get_ip_port, get_num_clients
from agent_llm_legacy import LangChainModuleLegacy
from agent_llm import LangChainModule
from agent_llm_stream import LangChainModuleStream
from llm_config import LLMConfig
from get_prompts import load_chain_setting, validate_chain_setting, set_parallel_chain_builder
from get_prompts import subject_separator, update_prompts_in_chains, load_json_file, convert_str_list_to_json

from manage_files import cleanup_loop, save_json_loop, load_statistics
from functools import partial

CONFIG_PATH = './settings/config.json'
CHARACTER_PATH = './settings/character_setting.json'
# Setting the timezone: 'Asia/Seoul'
TIMEZONE = pytz.timezone('Asia/Seoul')

# Setting the timezone: 'Asia/Seoul'
TIMEZONE = pytz.timezone('Asia/Seoul')

class TimezoneFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # 시간대 변환을 위해 aware datetime 객체를 생성합니다.
        dt = datetime.fromtimestamp(record.created, TIMEZONE)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            # s = dt.isoformat() # long format
            s = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return s

os.makedirs('log/', exist_ok=True)
# Set up logging
# Create a RotatingFileHandler
handler = RotatingFileHandler(
    'log/server_err.log',  # specify the log file name
    maxBytes= 10 * 1024 * 1024,  # 10MB
    backupCount=10  # keep up to 5 backup log files
)
# Set the logging level and format
handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# formatter = TimezoneFormatter('%(asctime)s - %(levelname)s - %(message)s')
formatter = TimezoneFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger().addHandler(handler)

# Optional: if you want to also log to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Ensure the root logger level is set
logger.addHandler(handler)
logger.addHandler(console_handler)

class LLMHandler():
    def __init__(self):
        self.gpt_config = None
        self.lc_module = None

    def handle_sufoo(self, data):
        if False:
            model='gpt-4o-mini'
            # model='gpt-4o'
            company='openai'
        else:
            model='gemini-2.5-flash'
            company='google'
        print('=================')
        print(f"model: {model}, company: {company}")
        print('=================')
            
        '''
        '{
        "question":"고혈압",
        "additional_info_for_question":"",
        "client_info":
            {"gender":"여","weight":"0","height":"0","bmi":"NaN","health_conditions":[],"medications_being_taken":[],"supplements_being_taken":[],"special_conditions":[]},
        "request":[
            {"title":"","description":"","result":"",
            "subject":[
                {"sub_title":"","sub_description":"","sub_result":""},
                {"sub_title":"","sub_description":"","sub_result":""}]},
            {"title":"","description":"","result":"",
            "subject":[
                {"sub_title":"","sub_description":"","sub_result":""},
                {"sub_title":"","sub_description":"","sub_result":""}
                ]
        }]}'
        '''

        question = data.get('question')
        additional_info_for_question = data.get('additional_info_for_question')
        client_info = data.get('client_info')
        request = data.get('request')

        # request = convert_str_list_to_json(request)

        information = {
            'question':question,
            'additional_info_for_question':additional_info_for_question,
            'client_info':client_info,
            'request':request,
        }

        for i, r in enumerate(request):
            request[i]['representative_image_name'] = ''

        if self.gpt_config is None:
            self.gpt_config = LLMConfig(character_id="default", stream=False, tokenizer=None, 
                    keep_dialog=None, company=company, model=model, temperature=0.8, max_tokens_output=None, 
                    max_tokens_context=30000, api_key_path='./settings/config.json')
        if self.lc_module is None:
            self.lc_module = LangChainModuleLegacy(self.gpt_config)

        target_process_name = 'sufoo'
        json_path='./settings/prompts/sufoo/chains.json'
        chains = load_json_file(json_path)
        chains = update_prompts_in_chains(json_path=json_path, chains=chains)
        
        # input value must be string
        input_val_dict = {'information': json.dumps(information, ensure_ascii=False)}
        # with open('tmp_information.json', 'w') as json_file:
        #     json.dump(input_val_dict, json_file, indent=4, ensure_ascii=False)

        # with open('tmp_chains.json', 'w') as json_file:
        #     json.dump(chains, json_file, indent=4, ensure_ascii=False)
        
        prepared_chain = chains['process'][target_process_name]
        
    
        llm_result = self.lc_module.run_chain_tree(chain_settings=chains, 
                                            process=prepared_chain, 
                                            prev_output=input_val_dict,
                                            callback=None,
                                            max_extra_tries=1
                                            )
        print(llm_result)
        return llm_result


    def handle_fodoit(self, data):
        if False:
            model='gpt-4o-mini'
            # model='gpt-4o'
            company='openai'
        else:
            model='gemini-2.5-flash'
            company='google'
            
        '''
        '{
        "question":"고혈압",
        "additional_info_for_question":"",
        "client_info":
            {"gender":"여","weight":"0","height":"0","bmi":"NaN","health_conditions":[],"medications_being_taken":[],"supplements_being_taken":[],"special_conditions":[]},
        "request":[
            {"title":"","description":"","result":"",
            "subject":[
                {"sub_title":"","sub_description":"","sub_result":""},
                {"sub_title":"","sub_description":"","sub_result":""}]},
            {"title":"","description":"","result":"",
            "subject":[
                {"sub_title":"","sub_description":"","sub_result":""},
                {"sub_title":"","sub_description":"","sub_result":""}
                ]
        }]}'
        '''

        question = data.get('question')
        additional_info_for_question = data.get('additional_info_for_question')
        client_info = data.get('client_info')
        request = data.get('request')

        # request = convert_str_list_to_json(request)

        information = {
            'question':question,
            'additional_info_for_question':additional_info_for_question,
            'client_info':client_info,
            'request':request,
        }

        for i, r in enumerate(request):
            request[i]['representative_image_name'] = ''

        if self.gpt_config is None:
            self.gpt_config = LLMConfig(character_id="default", stream=False, tokenizer=None, 
                    keep_dialog=None, company=company, model=model, temperature=0.8, max_tokens_output=None, 
                    max_tokens_context=30000, api_key_path='./settings/config.json')
        if self.lc_module is None:
            self.lc_module = LangChainModuleStream(self.gpt_config)

        target_process_name = 'fodoit_stream'
        json_path='./settings/prompts/fodoit_new/chains.json'
        chains = load_json_file(json_path)
        chains = update_prompts_in_chains(json_path=json_path, chains=chains)
        
        # input value must be string
        input_val_dict = {'information': json.dumps(information, ensure_ascii=False)}
        # with open('tmp_information.json', 'w') as json_file:
        #     json.dump(input_val_dict, json_file, indent=4, ensure_ascii=False)

        # with open('tmp_chains.json', 'w') as json_file:
        #     json.dump(chains, json_file, indent=4, ensure_ascii=False)
        
        prepared_chain = chains['process'][target_process_name]
        
    
        llm_result = self.lc_module.run_chain_tree(chain_settings=chains, 
                                            process=prepared_chain, 
                                            prev_output=input_val_dict,
                                            callback=None,
                                            max_extra_tries=1
                                            )
        print(llm_result)
        return llm_result

    def handle_fodoit_stream(self, data, callback=None):
        """
        fodoit 스트리밍 처리 함수
        callback을 통해 스트리밍된 내용을 실시간으로 전달할 수 있습니다.
        
        Args:
            data: 요청 데이터 (question, additional_info_for_question, client_info, request 포함)
            callback: 스트리밍 콜백 딕셔너리 (optional)
                - func: 스트리밍 콘텐츠를 받는 함수 (content, instance, func_name, code=None)
                - func_start_end: 시작/종료를 알리는 함수 (is_start, instance, suffix, func_name)
                - instance: 콜백 함수에 전달할 인스턴스
                - func_name: 함수 이름
        
        Returns:
            llm_result: LLM 처리 결과
        """
        if False:
            model='gpt-4o-mini'
            # model='gpt-4o'
            company='openai'
        else:
            model='gemini-2.5-flash'
            company='google'
            
        print('=================')
        print(f"model: {model}, company: {company}")
        print('=================')

        question = data.get('question')
        additional_info_for_question = data.get('additional_info_for_question')
        client_info = data.get('client_info')
        request = data.get('request')

        information = {
            'question':question,
            'additional_info_for_question':additional_info_for_question,
            'client_info':client_info,
            'request':request,
        }


        for i, r in enumerate(request):
            request[i]['representative_image_name'] = ''

        print('=================')
        print('information')
        print(information)
        print('request')
        print(request)
        print('=================')

        # 스트리밍을 위해 별도의 config와 module 인스턴스 생성
        # handle_fodoit와 독립적으로 동작하도록 함
        stream_config = LLMConfig(character_id="default", stream=True, tokenizer=None, 
                keep_dialog=None, company=company, model=model, temperature=0.8, max_tokens_output=None, 
                max_tokens_context=30000, api_key_path='./settings/config.json')
        stream_module = LangChainModuleStream(stream_config)

        target_process_name = 'fodoit_stream'
        json_path='./settings/prompts/fodoit_stream/chains.json'
        chains = load_json_file(json_path)
        chains = update_prompts_in_chains(json_path=json_path, chains=chains)
        
        # input value must be string
        input_val_dict = {'information': json.dumps(information, ensure_ascii=False)}
        
        prepared_chain = chains['process'][target_process_name]
        
        llm_result = stream_module.run_chain_tree(chain_settings=chains, 
                                            process=prepared_chain, 
                                            prev_output=input_val_dict,
                                            callback=callback,
                                            max_extra_tries=1
                                            )
        print(llm_result)
        return llm_result

if __name__ == "__main__":
    llm = LLMHandler()

    # 도움말: LLM 실행 없이 사용법만 출력하고 종료
    if "-h" in sys.argv or "--help" in sys.argv:
        print(
            "사용법:\n"
            "  - 기본 실행(기존): python3 llm_handler.py\n"
            "  - 스트리밍 예제:    python3 llm_handler.py --stream\n"
        )
        raise SystemExit(0)

    # 공통 테스트 데이터
    data = {
        "question": "당뇨가 있는데 음식과 슈퍼푸드 그리고 영양제 추천해줘",
        "additional_info_for_question": "",
        "client_info": {
            "gender": "남자",
            "weight": "77.1",
            "height": "177.5",
            "bmi": "24.5",
            "health_conditions": ["고혈압", "비만"],
            "medications_being_taken": ["혈압약"],
            "supplements_being_taken": ["종합비타민"],
            "special_conditions": []
        },
        "request": [
            {
                "title": "",
                "description": "",
                "result": "",
                "subject": [
                    {"sub_title": "", "sub_description": "", "sub_result": ""},
                    {"sub_title": "", "sub_description": "", "sub_result": ""}
                ]
            },
            {
                "title": "",
                "description": "",
                "result": "",
                "subject": [
                    {"sub_title": "", "sub_description": "", "sub_result": ""},
                    {"sub_title": "", "sub_description": "", "sub_result": ""}
                ]
            }
        ]
    }

    # 예) 스트리밍 실행:
    #   python3 llm_handler.py --stream
    if "--stream" in sys.argv:
        print("=== handle_fodoit_stream 스트리밍 예제 시작 ===")

        def on_stream_chunk(content, instance, func_name, code=None):
            # agent_llm_stream.py는 기본적으로 (chunk.content: str)을 넘깁니다.
            # 에러 케이스는 dict로 넘어올 수 있습니다.
            if isinstance(content, dict) and "error_msg" in content:
                print(f"\n[ERROR:{func_name}] {content.get('error_msg')}\n", flush=True)
                return
            print(str(content), end="", flush=True)

        def on_stream_start_end(is_start, instance, suffix, func_name):
            if is_start:
                print(f"\n[START] {func_name}\n", flush=True)
            else:
                print(f"\n\n[END] {func_name}\n", flush=True)

        callback = {
            "func": on_stream_chunk,
            "func_start_end": on_stream_start_end,
            "instance": None,
            "func_name": "stream_report",
        }

        result = llm.handle_fodoit_stream(data, callback=callback)
        print("\n=== 최종 결과(JSON) ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # 기본 동작(기존 유지)
        llm.handle_sufoo(data)