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

# LLM
import openai
from openai import OpenAI

# modules
from get_config import get_ip_port, get_num_clients
from agent_gpt import LangChainModule
from llm_config import GPTConfig
from get_prompts import load_chain_setting, validate_chain_setting
from get_prompts import subject_separator, chain_builder, update_prompts_in_chains, load_json_file, convert_str_list_to_json

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
        model='gpt-4o-mini'
        # model='gpt-4o'

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

        if self.gpt_config is None:
            self.gpt_config = GPTConfig(character_id="default", stream=False, tokenizer=None, 
                    keep_dialog=None, model=model, temperature=0.8, max_tokens_output=None, 
                    max_tokens_context=30000, api_key_path='./settings/config.json')
        if self.lc_module is None:
            self.lc_module = LangChainModule(self.gpt_config)

        target_process_name = 'sufoo'
        json_path='./settings/prompts/sufoo/chains.json'
        chains = load_json_file(json_path)
        chains = update_prompts_in_chains(json_path=json_path, chains=chains)
        validate_chain_setting(chains)
        
        # input value must be string
        input_val_dict = {'information': json.dumps(information, ensure_ascii=False)}
        # with open('tmp_information.json', 'w') as json_file:
        #     json.dump(input_val_dict, json_file, indent=4, ensure_ascii=False)

        # with open('tmp_chains.json', 'w') as json_file:
        #     json.dump(chains, json_file, indent=4, ensure_ascii=False)
        
        prepared_chain = chains['process'][target_process_name]
        
        llm_result = self.lc_module.run_chain_tree(chains, 
                                            prepared_chain, 
                                            input_val_dict
                                            )
        
        print(llm_result)

if __name__ == "__main__":
    llm = LLMHandler()

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
                    {
                        "sub_title": "",
                        "sub_description": "",
                        "sub_result": ""
                    },
                    {
                        "sub_title": "",
                        "sub_description": "",
                        "sub_result": ""
                    }
                ]
            },
            {
                "title": "",
                "description": "",
                "result": "",
                "subject": [
                    {
                        "sub_title": "",
                        "sub_description": "",
                        "sub_result": ""
                    },
                    {
                        "sub_title": "",
                        "sub_description": "",
                        "sub_result": ""
                    }
                ]
            }
        ]
    }

    llm.handle_sufoo(data)