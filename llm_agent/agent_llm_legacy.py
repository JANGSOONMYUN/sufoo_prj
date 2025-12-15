import os
import threading
import queue
import json
import copy
import time
import random
import string
import traceback
from datetime import datetime
import pytz
from typing import Dict, Any, Tuple, Type, List

import openai
from openai import OpenAI
from transformers import GPT2Tokenizer
from get_config import get_api_key
from get_prompts import load_chain_setting
from llm_config import LLMConfig
from modules.utils import fix_partial_json, remove_comma_before_bracket
from modules.subject_module import wrap_subjects, regen_chain, wrap_reports

# function for langchain
from modules.search_img.search import include_images

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import HarmBlockThreshold, HarmCategory
from langchain_google_vertexai import ChatVertexAI, VertexAI
# from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, create_model, Field, conint, confloat
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain import hub
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_core.runnables.utils import AddableDict

from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)



MAX_INPUT_TOKENS_GPT = 100000 # maximum input tokens is 128000 for gpt-4o
MAX_INPUT_TOKENS_GEMINI = 150000 # maximum input tokens is 2M for gemini

def write_log(message, log_dir='log/error/'):
    """
    지정된 디렉토리에 로그 파일을 생성하고 예외 메시지를 기록합니다.

    Args:
        log_dir: 로그 파일을 저장할 디렉토리 경로.
        message: 로그 메시지.
    """
    
    now = datetime.now()
    kst = pytz.timezone('Asia/Seoul')
    now_kst = now.astimezone(kst)
    
    date_str = now_kst.strftime("%Y-%m-%d")
    time_str = now_kst.strftime("%H%M%S")

    # 로그 디렉토리 생성
    log_path = os.path.join(log_dir, date_str)
    os.makedirs(log_path, exist_ok=True)

    # 로그 파일 이름 생성
    log_file_name = f"{time_str}.txt"
    log_file_path = os.path.join(log_path, log_file_name)

    # 파일 이름 중복 처리
    i = 0
    while os.path.exists(log_file_path):
        i += 1
        log_file_name = f"{time_str}_{i}.txt"
        log_file_path = os.path.join(log_path, log_file_name)    

    # 로그 파일 작성
    try:
      with open(log_file_path, "w") as f:
          timestamp = now_kst.strftime("%Y-%m-%d %H:%M:%S")
          log_msg = f"{timestamp} - {message}"
          print(log_msg)
          f.write(log_msg) 
        #   print(f"로그가 '{log_file_path}'에 저장되었습니다.")
    except Exception as e:
          print(f"로그 파일 생성 실패: {e}")
          pass


def create_pydantic_model_from_json(gemini_json_schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Gemini 스타일의 JSON 스키마로부터 Pydantic 모델을 동적으로 생성합니다.

    Args:
        gemini_json_schema: Gemini 스타일의 JSON 스키마 딕셔너리.

    Returns:
        동적으로 생성된 Pydantic 모델 클래스.
    """

    def map_data_type_to_python(data_type: str, property_schema: Dict[str, Any]) -> Type:
        """Gemini 타입을 Python 타입으로 매핑합니다."""
        data_type = data_type.lower()
        if data_type == "integer" or data_type == "int":
            if "minimum" in property_schema and "maximum" in property_schema:
                return conint(ge=property_schema["minimum"], le=property_schema["maximum"])
            return int
        elif data_type == "number" or data_type == "float" or data_type == "num": # Gemini number 는 float 를 의미하는 경우가 많음. 필요에 따라 수정
            if "minimum" in property_schema and "maximum" in property_schema:
                return confloat(ge=property_schema["minimum"], le=property_schema["maximum"])
            return float
        elif data_type == "string" or data_type == "str":
            return str
        elif data_type == "boolean" or data_type == "bool":
            return bool
        elif data_type == "array" or data_type == "list" or data_type == "list[str]": #List 처리
            items_schema = property_schema.get("items", {}) # items 스키마 가져오기
            items_type = items_schema.get("type", "string") # 기본적으로 string list 로 처리, 필요에 따라 조정
            if items_type == "object" or items_type == "obj" or items_type == "json": # List 내부 item 이 object 인 경우! (핵심 변경 부분)
                inner_model = create_pydantic_model_from_json(items_schema) # items_schema 자체가 object 스키마
                return List[inner_model] # List[PydanticModel] 반환
            else: # List[primitive type] or List[other object] (재귀적으로 object list 도 가능)
                python_item_type = map_data_type_to_python(items_type, items_schema) #items type 도 재귀적으로 처리
                return List[python_item_type]
        elif data_type == "object" or data_type == "obj" or data_type == "json": # Nested Object 처리 (재귀 호출)
            return create_pydantic_model_from_json(property_schema) #object 자체를 재귀적으로 모델로 만듬
        else:
            return str  # 기본적으로 string type

    fields: Dict[str, Tuple[Type, Any]] = {}

    properties = gemini_json_schema.get("properties", {}) # 최상위 properties 를 가져옴
    for field_name, property_schema in properties.items():
        print('property_schema')
        print(property_schema)
        data_type = property_schema.get("type")
        if not data_type:
            continue # type 이 없으면 스킵 (혹은 에러 처리)

        python_type = map_data_type_to_python(data_type, property_schema)
        field_description = property_schema.get("description", field_name) #description 없으면 field name 사용
        fields[field_name] = (python_type, Field(description=field_description))

    model: Type[BaseModel] = create_model("DynamicStructuredOutputModel", **fields) # 모델 이름 (원하는대로 변경 가능)
    return model

# Define an in-memory chat message history
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

    def __getitem__(self, index):
        return self.messages[index]

    def __setitem__(self, index, value):
        self.messages[index] = value

    def __delitem__(self, index):
        del self.messages[index]

    def __len__(self):
        return len(self.messages)
    
    def _message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """메시지 객체를 dictionary로 변환합니다."""
        if isinstance(message, AIMessage):
            return {"type": "ai", "content": message.content, "additional_kwargs": message.additional_kwargs}
        elif isinstance(message, HumanMessage):
            return {"type": "human", "content": message.content, "additional_kwargs": message.additional_kwargs}
        elif isinstance(message, SystemMessage):
            return {"type": "system", "content": message.content, "additional_kwargs": message.additional_kwargs}
        elif isinstance(message, str):
            return {"type": "human", "content": message, "additional_kwargs": ""}
        else:
            raise ValueError(f"Unsupported message type: {type(message)}\nmessage: {message}")

    def save_to_json(self, file_path: str, user_id: str) -> None:
        if 'test' not in user_id.lower():
            return
        """히스토리를 json 파일에 저장합니다."""
        message_dicts = [self._message_to_dict(message) for message in self.messages]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(message_dicts, f, indent=4, ensure_ascii=False)
            
    def trim_history(self, max_length: int) -> None:
        """최대 길이(`max_length`)를 초과하는 오래된 메시지를 삭제합니다."""
        if len(self.messages) > max_length:
            self.messages = self.messages[-max_length:] # 가장 최근 메시지를 기준으로 보존
            print(f'history has reduced: {len(self.messages)}')
    
    def prepend_messages(self, messages: List[BaseMessage]) -> None:
        """다른 대화 내용을 맨 앞에 추가합니다."""
        self.messages = messages + self.messages
        

class LangChainModuleLegacy():
    def __init__(self, config: LLMConfig) -> None:
        self.is_running = False
        
        self.config = config
        self.api_info = get_api_key(self.config.api_key_path)
        
        self.user_id = self.config.user_id
        self.store = {}
        self.last_msg = None
        self.keep_input_output_data = {}
        self.keep_optional_data = {}
        self.stream_storage = {}
        self.is_retrying = False
        self.llm = {}
        self.max_input_tokens = 100000
        
        self.total_tokens = {'prompt_tokens': 0, 'completion_tokens': 0}
        self.init_llm(self.config.model)
        
        
    def params(self, config: LLMConfig):
        self.config = config
    
    def clear_data(self):
        self.keep_input_output_data = {}
            
    def get_model_company(self, model):
        if 'gpt' in model:
            self.config.company = 'openai'
        elif 'gemini' in model:
            self.config.company = 'google'
        else:
            assert False, 'Currently gpt or gemini can be used only, in get_model_company()'
        
    def compute_token_price(self, model_name, input_tokens, output_tokens, tot_input_tokens, tot_output_tokens):
        # self.api_info
        input_price = None
        output_price = None
        usd_to_krw = self.api_info['usd_to_krw']
        result = {
                    'current_token_info': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'input_cost': None,
                        'output_cost': None,
                        'total_cost': None
                        },
                    'cumulative_token_info': {
                        'input_tokens': tot_input_tokens,
                        'output_tokens': tot_output_tokens,
                        'input_cost': None,
                        'output_cost': None,
                        'total_cost': None
                    },
                    'price_per_1M_input_token': None,
                    'price_per_1M_output_token': None,
                    'usd_to_krw': usd_to_krw,
                    'model_name': model_name,
                }
        
        def _compute_cost(result, key_name, input_price, output_price, usd_to_krw):
            result[key_name]['input_cost'] = round(result[key_name]['input_tokens'] * input_price / 1000000, 10)
            result[key_name]['output_cost'] = round(result[key_name]['output_tokens'] * output_price / 1000000, 10)
            result[key_name]['total_cost'] = round(result[key_name]['input_cost'] + result[key_name]['output_cost'], 10)
            result[key_name]['input_cost_krw'] = round(result[key_name]['input_cost'] * usd_to_krw, 5)
            result[key_name]['output_cost_krw'] = round(result[key_name]['output_cost'] * usd_to_krw, 5)
            result[key_name]['total_cost_krw'] = round(result[key_name]['total_cost'] * usd_to_krw, 5)
            
        try:
            if model_name in self.api_info['pricing']:
                input_price = self.api_info['pricing'][model_name]['input']
                output_price = self.api_info['pricing'][model_name]['output']
            else:
                for k, v in self.api_info['pricing'].items():
                    if model_name in k:
                        input_price = v['input']
                        output_price = v['output']
                        break
            result['price_per_1M_input_token'] = input_price
            result['price_per_1M_output_token'] = output_price
            
            _compute_cost(result, 'current_token_info', input_price, output_price, usd_to_krw)
            _compute_cost(result, 'cumulative_token_info', input_price, output_price, usd_to_krw)
            
        except Exception as e:
            print(e)
        
        return result
    
    def init_llm(self, model=None):
        model = model if model is not None else self.config.model
        self.get_model_company(model)
        if model in self.llm:
            return
        if self.config.company == 'openai':
            self.max_input_tokens = MAX_INPUT_TOKENS_GPT
            self.llm[model] = ChatOpenAI(
                model=model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_output,
                timeout=None,
                max_retries=2,
                api_key=self.api_info['openai']['api_key'], 
                organization=self.api_info['openai']['organization']
                # base_url="...",
                # other params...
            )
            
        elif self.config.company == 'google':
            self.max_input_tokens = MAX_INPUT_TOKENS_GEMINI
            use_vertexai = True
            if not use_vertexai:
                from langchain_google_genai import HarmBlockThreshold, HarmCategory
                self.llm[model] = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens_output,
                    # top_p=0.8,
                    # top_k=40,
                    google_api_key=self.api_info['google']['api_key'],
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
                    },
                    # retry_max_attempts=2,  # 현재 지원되지 않음
                    # timeout=None,  # 현재 지원되지 않음
                )
            else:
                from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.api_info['google']['credentials']
                self.llm[model] = ChatVertexAI(
                    project=self.api_info['google']['project_id'],
                    location=self.api_info['google']['region'],
                    model=model,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens_output,
                    # top_p=0.8,
                    # top_k=40,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
                    },
                    # retry_max_attempts=2,  # 현재 지원되지 않음
                    # timeout=None,  # 현재 지원되지 않음
                )
        else:
            assert False, 'company name should be openai or google, in init_llm()'
    
    def get_last_msg(self):
        return self.last_msg
    
    # To store history (conversations)
    def get_session_history(self, user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        if (user_id, conversation_id) not in self.store:
            self.store[(user_id, conversation_id)] = InMemoryHistory()
        return self.store[(user_id, conversation_id)]
    
    def get_len_history(self, user_id, conversation_id):
        history_key = (user_id, conversation_id)
        if history_key in self.store:
            return len(self.store[(user_id, conversation_id)])
        return 0
    
    def reduce_history(self, user_id, conversation_id, max_chat):
        history_key = (user_id, conversation_id)
        if history_key in self.store:
            self.store[history_key].trim_history(max_chat)
            
            self.store[history_key].save_to_json('./log/history.json', self.user_id)
            
            # if len(self.store[history_key]) > max_chat:
            #     del_num = len(self.store[history_key]) - max_chat
            #     del self.store[history_key][0:del_num]
            #     print(f'history has deleted from {len(self.store[history_key])} to {len(self.store[history_key]) - del_num}')
    
    # delete first 4 chats
    def delete_history(self, user_id, conversation_id, del_num=4):
        history_key = (user_id, conversation_id)
        if history_key in self.store:
            history_size = len(self.store[history_key])
            self.store[history_key].trim_history(history_size - del_num)
            print(f'history has deleted from {history_size} to {history_size - del_num}')
            self.store[history_key].save_to_json('./log/history.json', self.user_id)
            
            # del self.store[history_key][0:del_num]
            # print(f'history has deleted from {len(self.store[history_key])} to {len(self.store[history_key]) - del_num}')
    
    def convert_strings_to_messages(self, messages: List[str]) -> List[BaseMessage]:
        """문자열 리스트를 메시지 객체 리스트로 변환합니다."""
        message_objects = []
        for i, message in enumerate(messages):
            if i % 2 == 0:
                message_objects.append(HumanMessage(content=message))
            else:
                message_objects.append(AIMessage(content=message))
        return message_objects

    def convert_json_to_messages(self, messages_json: List[Dict[str, str]]) -> List[BaseMessage]:
        """JSON 형식의 메시지를 메시지 객체 리스트로 변환합니다."""
        message_objects = []
        for message in messages_json:
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                message_objects.append(HumanMessage(content=content))
            elif role == "assistant":
                message_objects.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unsupported role: {role}")
        return message_objects
    
    def parse_messages(self, messages) -> List[BaseMessage]:
        """입력 메시지를 파싱하여 메시지 객체 리스트로 변환합니다."""
        if isinstance(messages, str):
            return self.convert_strings_to_messages([messages])
        
        elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            return self.convert_json_to_messages(messages)
        
        else:
            raise ValueError("Unsupported input format: messages should be either a string or a list of JSON objects.")

    def prepend_session_history(self, user_id: str, conversation_id: str, new_messages) -> None:
        """해당 사용자와 대화 ID에 해당하는 히스토리에 새 메시지를 추가합니다."""
        message_objects = self.parse_messages(new_messages)
        self.get_session_history(user_id, conversation_id)
        self.store[(user_id, conversation_id)].prepend_messages(message_objects)
        
    def current_tokens(self, user_id, conversation_id):
        history_key = (user_id, conversation_id)
        if history_key in self.store:
            last_idx = len(self.store[history_key]) - 1
            for i in range(last_idx):
                target_i = last_idx - i
                message = self.store[history_key][target_i]
                if isinstance(message, HumanMessage):
                    # print("HumanMessage:")
                    pass
                elif isinstance(message, AIMessage):
                    # print("AIMessage:")
                    pass
                
                # # Check if 'content' exists and print it
                # if hasattr(message, 'content'):
                #     print("  content:", message.content)
                print("  message:", message)

                # Check if 'response_metadata' exists and print it
                if hasattr(message, 'response_metadata') and 'token_usage' in message.response_metadata:
                    print("  response_metadata:", message.response_metadata)
                    # print(type(message.response_metadata))
                    token_usage = message.response_metadata['token_usage']
                    prompt_tokens = token_usage['prompt_tokens']
                    completion_tokens = token_usage['completion_tokens']
                    return prompt_tokens, completion_tokens
        return 0, 0
            
                
    # JSON output
    # output_dict: key_name and its description
    '''
        response_schemas = [
            ResponseSchema(name="output1", description="string, translates first word to English"),
            ResponseSchema(name="output2", description="string, translates next word to English")
        ]
    '''
    def set_json_output(self, output_dict):
        response_schemas = []
        for k, v in output_dict.items():
            response_schemas.append(ResponseSchema(name=k, description=v))
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        return output_parser
    
    # with history
    def process_chain(self, model_name, instructions, prompts, input_dict, output_parser_type='json', output_dict=None, use_history=False, save_history=False, user_id="123", conversation_id="1"):
        self.init_llm(model_name)
        llm = self.llm[model_name]
        # keys: input_dict.keys()
        sys_msg = SystemMessage(instructions)
        msg_plchldr = MessagesPlaceholder(variable_name="msg_history")
        
        if output_parser_type == 'str': #if output_dict is None:
            hum_msg =  HumanMessagePromptTemplate.from_template(prompts)
        else:
            hum_msg =  HumanMessagePromptTemplate.from_template(
                prompts + \
                "\n{format_instructions}"
            )

        messages = [sys_msg]
        if use_history is True and save_history is False:
            history = self.get_session_history(user_id, conversation_id)
            messages.extend(history.messages)
        elif save_history:
            messages.append(msg_plchldr)
        messages.append(hum_msg)
        
        output_parser = None
        if output_parser_type == 'str': #if output_dict is None:
            output_parser = StrOutputParser()
            prompt_template = ChatPromptTemplate(
                messages=messages,
                input_variables=list(input_dict.keys())
            )
        elif output_parser_type == 'json_legacy':
            output_parser = self.set_json_output(output_dict)
            prompt_template = ChatPromptTemplate(
                messages=messages,
                input_variables=list(input_dict.keys()),
                partial_variables={"format_instructions": output_parser.get_format_instructions()}
            )
        else:
            dynamic_model = create_pydantic_model_from_json(output_dict)
            output_parser = PydanticOutputParser(pydantic_object=dynamic_model)
            prompt_template = ChatPromptTemplate(
                messages=messages,
                input_variables=list(input_dict.keys()),
                partial_variables={"format_instructions": output_parser.get_format_instructions()}
            )
            
            
        chain = (
            prompt_template
            | llm
        )

        if save_history is True:
            chain_with_history = RunnableWithMessageHistory(
                                chain,
                                get_session_history=self.get_session_history,
                                input_messages_key=list(input_dict.keys())[0],
                                history_messages_key="msg_history",
                                history_factory_config=[
                                    ConfigurableFieldSpec(
                                        id="user_id",
                                        annotation=str,
                                        name="User ID",
                                        description="Unique identifier for the user.",
                                        default="",
                                        is_shared=True,
                                    ),
                                    ConfigurableFieldSpec(
                                        id="conversation_id",
                                        annotation=str,
                                        name="Conversation ID",
                                        description="Unique identifier for the conversation.",
                                        default="",
                                        is_shared=True,
                                    ),
                                ],
                            )
            return chain_with_history, output_parser
        return chain, output_parser
      
    def run_chain(self, main_chain, input_dict, config_dict=None, is_stream=False):    
        try:
            # wrap config with configurable
            if 'configurable' not in config_dict:
                config_dict = {"configurable": config_dict.copy()}
            if config_dict is None:
                if is_stream:
                    result = main_chain.stream(input_dict)
                else:
                    result = main_chain.invoke(input_dict)
            else:
                if is_stream:
                    result = main_chain.stream(input_dict, config=config_dict)
                else:
                    result = main_chain.invoke(input_dict, config=config_dict)
            return result
        except KeyError as e:
            if str(e) == "'type'":  # KeyError가 'type'인 경우에만 처리
                print(f"원래 에러 메시지: {e}") #원래 에러 메시지 출력
                traceback.print_exc() # traceback 메시지 (에러 발생 위치) 출력
                print('')
                print("[ERROR] 프롬프트에 잘못된 형식의 문자가 있습니다. (중괄호 여부 확인)") #추가 메시지 출력
                print('')
                
            else:  #다른 KeyError는 그대로 다시 발생
                raise e
        except Exception as e: #KeyError외 다른 에러 처리
            print(f"[ERROR] 다른 에러 발생: {e}")
            print(f'input_dict: {input_dict}')
            print(main_chain)
            traceback.print_exc()
            raise e

    def combine_parallel_chain(self, chains, chain_names=None):
        chain_dict = {}
        for i, chain in enumerate(chains):
            chain_name = 'chain_' + str(i)
            if chain_names is not None:
                chain_name = chain_names[i]
            
            chain_dict[chain_name] = chain
        map_chain = RunnableParallel(chain_dict)
        return map_chain

    def single_or_parallel_chain(self, chains, chain_names=None):
        if len(chains) > 1:
            return self.combine_parallel_chain(chains, chain_names)
        return chains[0]
    
    def _check_if_in_list(self, storage_list, target_list):
        new_list = []                
        for t in target_list:
            if isinstance(t, dict):
                t_str = json.dumps(t, indent=4,  ensure_ascii=False)
            else:
                t_str = str(t)
                
            if t_str not in storage_list:
                new_list.append(t_str)
        return new_list
    
    def _get_value_from_stream(self, content, chain_key, output_dict, separator='\n'):
        self.stream_storage[chain_key] = [] if chain_key not in self.stream_storage else self.stream_storage[chain_key]
        # remove some parts
        remove_str = ['\'\'\'json']
        
        return_dict = {}
        repaired_json = {}
        try:
            repaired_json = fix_partial_json(content)
        except Exception as e:
            print(f"Unknown error at _get_value_from_stream(): {e}")
            print(content)
            print('')
            repaired_json = {}
            # assert False, f"Unknown error at _get_value_from_stream(): {e}"
            
        if repaired_json is None or len(repaired_json) == 0:
            return return_dict
        
        for k in output_dict.keys():
            return_dict[k] = []
            if k in repaired_json:
                target_val = repaired_json[k]
                if isinstance(target_val, list):
                    new_list = self._check_if_in_list(self.stream_storage[chain_key], target_val)
                    self.stream_storage[chain_key].extend(new_list)
                    return_dict[k].extend(new_list)
                elif isinstance(target_val, str):
                    # string part should be modified by separator
                    # it doesn't return in stream
                    if target_val not in self.stream_storage[chain_key]:
                        self.stream_storage[chain_key].append(target_val)
                        return_dict[k].append(target_val)
                elif isinstance(target_val, dict):
                    # dict part should be modified
                    # it doesn't return in stream
                    target_val = json.dumps(target_val, indent=4,  ensure_ascii=False)
                    if target_val not in self.stream_storage[chain_key]:
                        self.stream_storage[chain_key].append(target_val)
                        return_dict[k].append(target_val)
                elif isinstance(target_val, int):
                    target_val = str(target_val)
                    if target_val not in self.stream_storage[chain_key]:
                        self.stream_storage[chain_key].append(target_val)
                        return_dict[k].append(target_val)
        return return_dict
        
    
    def stream_handler(self, result, prompt_chain_keys, output_dict, callback, stream_type='str', return_func_name='noname'):
        assert isinstance(callback, dict), 'callback should be dictionary to be handled. in stream_handler'
        assert len(prompt_chain_keys) == 1 , '[ERROR] streaming for multiple chains has not yet implemented, in stream_handler'
        callback_func = callback['func']
        callback_start_end = callback['func_start_end']
        class_instance = callback['instance']
        stream_step = 50
                
        i = 0
        tot_contents = ''
        special_key_storage = ''
        if len(prompt_chain_keys) > 1:
            tot_contents = {}
            special_key_storage = {}
            for k in prompt_chain_keys:
                tot_contents[k] = ''
                special_key_storage[k] = ''
                
        
        def _send_stream(i, class_instance, stream_step, content, chain_key, output_dict):
            try:
                if i % stream_step == 0:
                    dict_from_stream = self._get_value_from_stream(content=content, chain_key=chain_key, output_dict=output_dict)
                    if dict_from_stream is None or len(dict_from_stream) == 0:
                        return
                    for dk, dv in dict_from_stream.items():
                        for di in dv:
                            callback_func(di, class_instance, return_func_name)
            except Exception as e:
                error_message = traceback.format_exc()
                print(f'{e}, in _send_stream in stream_handler\n{error_message}')
        
        def _find_special_keys(content, acc_str, target_strs):
            acc_str += content
            search_start_i = 0
            for i, tstr in enumerate(target_strs):
                if tstr not in acc_str[search_start_i:]:
                    return False, acc_str
                search_start_i = acc_str.index(tstr) + len(tstr)
            return True, ''
        
        def _replace_special_str(content, acc_str):
            target_strs = ['n', 't', '"']
            acc_str += content
            if '\\' in acc_str:
                print('(_replace_special_str)acc_str:', acc_str)
                if '\\n' in acc_str:
                    return True, acc_str.replace('\\n', '\n')
                elif '\\t' in acc_str :
                    return True, acc_str.replace('\\t', '\t')
                elif '\\"' in acc_str:
                    return True, acc_str.replace('\\"', '"')
                elif """\\'""" in acc_str:
                    return True, acc_str.replace("""\\'""", '"')
                else:
                    return False, acc_str
            elif '"' in acc_str:
                print('(_replace_special_str)acc_str[END]:', acc_str)
                return False, '[END]'
            else:
                return True, content
            
        
        key_found_count = 0
        stream_found = False
        stream_found_key_set = []
        for i, o in enumerate(output_dict):
            stream_found_key_set.append([f'{o}', ':', '''"'''])
        def _send_stream_char(content, special_key_storage):
            nonlocal key_found_count, stream_found
            try:                
                if not stream_found:
                    stream_found, special_key_storage = _find_special_keys(content, special_key_storage, stream_found_key_set[key_found_count])
                    if stream_found:
                        key_found_count += 1
                
                elif stream_found:
                    # check special characters and check if it is end
                    stream_found, special_key_storage = _replace_special_str(content, special_key_storage)
                    if len(special_key_storage) > 0 and stream_found and special_key_storage != '[END]':
                        callback_func(special_key_storage, class_instance, return_func_name)
                    stream_found = True
                        
                elif special_key_storage == '[END]':
                    stream_found = False
                    special_key_storage = ''
                    callback_func('|', class_instance, return_func_name)
                
            except Exception as e:
                print(f'{e}, in _send_stream_char in stream_handler')
            
            return special_key_storage
        
        # # If is retrying, it does not need to send [START] signal
        # if self.is_retrying is False:
        #     callback_start_end(True, class_instance)
        
        
                            
        for r in result:
            i += 1
            # print(r)
            
            if len(prompt_chain_keys) > 1:
                for k in prompt_chain_keys:
                    if k not in r:
                        continue
                    tot_contents[k] += r[k].content
                    
                    if stream_type == 'str':
                        # send the last message if it finished
                        if len(r[k].response_metadata) > 0:
                            i = stream_step
                            print('the last message')
                        _send_stream(i, class_instance, stream_step, tot_contents[k], k, output_dict)
                    elif stream_type == 'char':
                        # special_key_storage[k] = _send_stream_char(r[k].content, special_key_storage[k])
                        assert False, 'parallel streaming has not implemented yet, in stream_handler'
                    elif stream_type == 'json':
                        assert False, 'stream_type == json has not implemented'
            else:
                tot_contents += r.content
                # send the last message if it finished
                if stream_type == 'str':
                    if len(r.response_metadata) > 0:
                        i = stream_step
                    _send_stream(i, class_instance, stream_step, tot_contents, prompt_chain_keys[0], output_dict)
                elif stream_type == 'char':
                    special_key_storage = _send_stream_char(r.content, special_key_storage)
                        
                elif stream_type == 'json':
                    assert False, 'stream_type == json has not implemented'
                    
            # if i > 200:
            #     assert False
        
        # callback_start_end(False, class_instance)

        # AIMessage
        ai_msg_contents = None
        if len(prompt_chain_keys) > 1:
            ai_msg_contents = {}
            for k in prompt_chain_keys:
                ai_msg_contents[k] = AIMessage(content=tot_contents[k])
                ai_msg_contents[k].response_metadata['token_usage'] = {'prompt_tokens': 0 , 'completion_tokens': len(tot_contents[k])}
        else:
            ai_msg_contents = AIMessage(content=tot_contents)
            ai_msg_contents.response_metadata['token_usage'] = {'prompt_tokens': 0 , 'completion_tokens': len(tot_contents)}
        return ai_msg_contents
    
    # for keep_input_output_data
    def set_input_data(self, chain_settings, chain_key, prev_output=None):
        input_dict = chain_settings[chain_key]['input_dict']

        # load data if needed
        for ik, iv in input_dict.items():
            if ik in self.keep_input_output_data:
                chain_settings[chain_key]['input_dict'][ik] = self.keep_input_output_data[ik]
            if ik in self.keep_optional_data:
                chain_settings[chain_key]['input_dict'][ik] = self.keep_optional_data[ik]
                
        # keep input data if setting exists
        if 'keep_input_output_data' in chain_settings[chain_key] and chain_settings[chain_key]['keep_input_output_data'] is not None:               
            for keep_key in chain_settings[chain_key]['keep_input_output_data']:
                if prev_output is not None:
                    if keep_key in prev_output:
                        target_dict = prev_output
                        self.keep_input_output_data.update({keep_key: target_dict[keep_key] if isinstance(target_dict[keep_key], str) else copy.deepcopy(target_dict[keep_key])})
                                            
                    # if previous values are wrapped by chain name
                    for prev_name, prev_val in prev_output.items():
                        target_dict = prev_val
                        if isinstance(prev_val, dict):
                            if keep_key in prev_val:
                                self.keep_input_output_data.update({keep_key: target_dict[keep_key] if isinstance(target_dict[keep_key], str) else copy.deepcopy(target_dict[keep_key])})
                    
                if keep_key not in input_dict:
                    continue
                target_dict = input_dict
                self.keep_input_output_data.update({keep_key: target_dict[keep_key] if isinstance(target_dict[keep_key], str) else copy.deepcopy(target_dict[keep_key])})

    def update_optional_data(self, chain_settings, chain_key):
        if 'optional_data' in chain_settings[chain_key] and chain_settings[chain_key]['optional_data'] is not None:
            for opt_key, opt_val in chain_settings[chain_key]['optional_data'].items():
                if opt_key not in self.keep_optional_data:
                    if 'int' in opt_val:
                        self.keep_optional_data.update({opt_key: 0})
                    elif 'str' in opt_val:
                        self.keep_optional_data.update({opt_key: ''})
                    elif 'bool' in opt_val:
                        self.keep_optional_data.update({opt_key: False})
                    elif 'list' in opt_val:
                        self.keep_optional_data.update({opt_key: []})
                    elif 'dict' in opt_val:
                        self.keep_optional_data.update({opt_key: {}})
                    else:
                        continue
                if opt_key not in self.keep_input_output_data:
                    self.keep_input_output_data.update(self.keep_optional_data)
                    chain_settings[chain_key]['input_dict'][opt_key] = self.keep_input_output_data[opt_key]
        
    # for keep_input_output_data
    def update_data(self, chain_settings, chain_key, output_result, input_dict):
        # check if keep_input_output_data is in the setting and the result
        if 'keep_input_output_data' in chain_settings[chain_key] and chain_settings[chain_key]['keep_input_output_data'] is not None:
            for keep_key in chain_settings[chain_key]['keep_input_output_data']:
                if keep_key in input_dict:
                    if isinstance(input_dict[keep_key], str) or keep_key == 'chain_ptr':
                        # 'chain_ptr' should not deepcopy
                        self.keep_input_output_data.update({keep_key: input_dict[keep_key]})
                    else:
                        self.keep_input_output_data.update({keep_key: copy.deepcopy(input_dict[keep_key])})
                
                if keep_key == "all":
                    self.keep_input_output_data.update(copy.deepcopy(output_result))
                elif keep_key in output_result:
                    self.keep_input_output_data.update({keep_key: output_result[keep_key] if isinstance(output_result[keep_key], str) else copy.deepcopy(output_result[keep_key])})

        if 'optional_data' in chain_settings[chain_key] and chain_settings[chain_key]['optional_data'] is not None:
            for opt_key, opt_val in chain_settings[chain_key]['optional_data'].items():
                if opt_key not in output_result:
                    continue
                self.keep_optional_data[opt_key] = output_result[opt_key]

    def run_prompt_chain(self, chain_settings, prompt_chain_keys, prev_output={'question': 'say anything'}, callback=None, max_extra_tries=1):
        chains = {}
        map_input_dict = {}
        callback_func, callback_start_end, class_instance = None, None, None
        callback_token = None
        return_func_name = ''
        if callback is not None:
            assert isinstance(callback, dict), 'callback should be dictionary to be handled. in stream_handler'
            # if instance does not exist, the function will be run in random instance.          
            callback_func = callback['func']
            callback_start_end = callback['func_start_end']
            callback_token = callback['func_token_usage']
            class_instance = callback['instance']
            if 'func_name' in callback:
                return_func_name = callback['func_name']
            if 'return_func_name' in chain_settings[prompt_chain_keys[0]]:
                return_func_name = chain_settings[prompt_chain_keys[0]]['return_func_name']
        use_start_end_callback = True
        start_end_suffix = ''
        for k in prompt_chain_keys:
            self.update_optional_data(chain_settings=chain_settings, chain_key=k)
            
            input_dict = chain_settings[k]['input_dict']
            orig_input_keys = list(chain_settings[k]['input_dict'].keys())
            
            
            if k in prev_output:
                chain_settings[k]['input_dict'].update(prev_output[k])
            else:
                chain_settings[k]['input_dict'].update(prev_output)

            
            instructions = chain_settings[k]['instructions']
            prompts = chain_settings[k]['prompts']
            output_dict = chain_settings[k]['output_dict']
            output_parser_type = chain_settings[k].get('output_parser_type', 'json')
            output_stream = chain_settings[k]['output_stream']
            stream_type = chain_settings[k].get('stream_type', 'str') # str, char, json
            use_start_end_callback = chain_settings[k]['use_start_end_callback']
            start_end_suffix = chain_settings[k]['start_end_suffix']
            model_name = chain_settings[k]['model']
            
            use_history = chain_settings[k]['use_history']
            save_history = chain_settings[k]['save_history']

            self.set_input_data(chain_settings=chain_settings, chain_key=k, prev_output=prev_output)

            _config = chain_settings[k]['config']
            user_id = _config['configurable']['user_id']
            conversation_id = _config['configurable']['conversation_id']
            
            # to remove unecessary input
            for in_key in orig_input_keys:
                map_input_dict[in_key] = input_dict[in_key]

            chain, output_parser = self.process_chain(model_name=model_name, instructions=instructions, prompts=prompts,
                        input_dict=input_dict, output_parser_type=output_parser_type, output_dict=output_dict, 
                        use_history=use_history, save_history=save_history, user_id=user_id, conversation_id=conversation_id)
            chains[k] = {'chain': chain, 'output_parser': output_parser}

        runnable_chain = self.single_or_parallel_chain(chains=[v['chain'] for k, v in chains.items()], 
                                                         chain_names=prompt_chain_keys)

        def _parse_by_type(parsed_output, _output_parser_type, _output_type, _output_dict):
            if _output_parser_type == 'str' and _output_type == 'json': 
                if _output_dict is not None and len(_output_dict) > 0:                    
                    return {list(_output_dict.keys())[0]: parsed_output}
                else:
                    return {'tmp_key(check_output_dict_in_chain_setting)': parsed_output}
            return parsed_output
        
        # Send [START] signal
        if callback is not None and self.is_retrying is False and use_start_end_callback is True:
            callback_start_end(True, class_instance, start_end_suffix, return_func_name)
            
            
        for in_k, in_v in map_input_dict.items():
            converted_v = ''
            
            # why need to be handled by if
            # details = "Unable to submit request because it must include at least one parts field, which describes the prompt input. Learn more: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini"
            if isinstance(in_v, dict):
                converted_v = json.dumps(in_v, ensure_ascii=False)
            elif in_v is None or (isinstance(in_v, str) and len(in_v) == 0):
                converted_v = ' '
            elif isinstance(in_v, bool):
                converted_v = str(in_v)
            else:
                converted_v = in_v
            map_input_dict[in_k] = converted_v
            
        
        try:
            _result = self.run_chain(main_chain=runnable_chain, input_dict=map_input_dict, config_dict=_config, is_stream=output_stream)
            
            if output_stream:
                result = self.stream_handler(result=_result, prompt_chain_keys=prompt_chain_keys, output_dict=output_dict, callback=callback, stream_type=stream_type, return_func_name=return_func_name)
            else:
                result = _result

            parsed_result = None
            # apply parser
            # Access the parsed output and response metadata
            if len(chains) > 1:
                parsed_result = {}
                for pk in prompt_chain_keys:
                    _parser = chains[pk]['output_parser']
                    _output_parser_type = chain_settings[pk].get('output_parser_type', 'json')
                    _output_dict = chain_settings[pk]['output_dict']
                    _output_type = chain_settings[pk]['output_type']
                    
                    result[pk].content = remove_comma_before_bracket(result[pk].content)
                    # from langchain_core.exceptions import OutputParserException
                    # raise OutputParserException("테스트 에러 테스트")

                    # print('%'*100)
                    # print(result[pk].content)
                    # print('%'*100)

                    parsed_output = _parser.parse(result[pk].content)

                    # parse again by parse type (json or str)
                    parsed_result[pk] = _parse_by_type(parsed_output=parsed_output, _output_parser_type=_output_parser_type, _output_type=_output_type, _output_dict=_output_dict)

            else:
                pk = prompt_chain_keys[0]
                _parser = chains[pk]['output_parser']
                _output_parser_type = chain_settings[pk].get('output_parser_type', 'json')
                _output_dict = chain_settings[pk]['output_dict']
                _output_type = chain_settings[pk]['output_type']

                result.content = remove_comma_before_bracket(result.content)
                
                parsed_output = _parser.parse(result.content)
                # parse again by parse type (json or str)
                parsed_result = _parse_by_type(parsed_output=parsed_output, _output_parser_type=_output_parser_type, _output_type=_output_type, _output_dict=_output_dict)

            print('%'*100)
            print(' ---- parsed_result ---- ')
            print(parsed_result)
            print('%'*100)
            # keep data
            for pk in prompt_chain_keys:
                parsed_output = parsed_result[pk] if len(chains) > 1 else parsed_result
                self.update_data(chain_settings=chain_settings, chain_key=pk, output_result=parsed_output, input_dict=input_dict)

        except Exception as e:
            if max_extra_tries > 0:
                error_message = traceback.format_exc()
                print(f"Unknown error at run_prompt_chain(): {e}, {error_message}, \nmax_extra_tries:{max_extra_tries}")
                self.is_retrying = True
                # chain_settings[k]['prompts'] = f'\n# 에러 발생: {e}. 형식에 맞게 다시 출력하라.'
                chain_settings[k]['prompts'] = chain_settings[k]['prompts'] + f'\n# 형식에 맞게 출력하라.'
                
                write_log(message=error_message)
                return self.run_prompt_chain(chain_settings=chain_settings, prompt_chain_keys=prompt_chain_keys, prev_output=prev_output, callback=callback, max_extra_tries=max_extra_tries-1)
            else:
                self.last_msg = "[ERROR] 죄송합니다. 시스템에 문제가 생겼습니다. 다시 입력해주세요."
                print(f"Unknown error at run_prompt_chain(): {e}, \n{self.last_msg} \nmax_extra_tries:{max_extra_tries}")
                
                write_log(message=self.last_msg)
                if callback_func is not None:
                    callback_func({'error_msg': self.last_msg}, class_instance, return_func_name, 4000)

                return self.last_msg

        self.is_retrying = False
        
        # response_metadata = result.response_metadata
        user_id_set = set()
        conv_id_set = set()
        token_price = {}
        for k in prompt_chain_keys:
            model_name = chain_settings[k]['model']
            _config = chain_settings[k]['config']
            max_input_tokens = chain_settings[k].get('max_input_tokens', self.max_input_tokens)
            
            # if _config['configurable']['user_id'] in user_id_set and _config['configurable']['conversation_id'] in conv_id_set:
            #     continue
            user_id_set.add(_config['configurable']['user_id'])
            conv_id_set.add(_config['configurable']['conversation_id'])
            
            _result = result.copy()
            if len(chains) > 1:
                _result = result[k].copy()
                
            # to unify differen llm models (openai, google, ...)
            _result = self.unify_and_sum_metadata(_result)
            #

            response_metadata = _result.response_metadata
            token_usage = response_metadata['token_usage']
            prompt_tokens = token_usage['prompt_tokens']
            completion_tokens = token_usage['completion_tokens']
            
            # compute token price
            token_price[k] = self.compute_token_price(model_name, prompt_tokens, completion_tokens, 
                                     self.total_tokens['prompt_tokens'], self.total_tokens['completion_tokens'])
            
            # if the conversation history is too large, it reduces oldest chat
            print(f'[{k}] prompt_tokens: {prompt_tokens}/{max_input_tokens}')
            self.reduce_history(user_id=_config['configurable']['user_id'], conversation_id=_config['configurable']['conversation_id'], max_chat=200)
            
            # if prompt tokens is too large, delete ...
            if prompt_tokens > max_input_tokens: # maximum input tokens is 128000 for gpt-4o
                self.delete_history(user_id=_config['configurable']['user_id'], conversation_id=_config['configurable']['conversation_id'], del_num=4)
                print(f'prompt_tokens: {prompt_tokens}')
                
        '''
        # callback for normal return
        # NO Stream
        '''
        if not output_stream:
            for k in prompt_chain_keys:
                callback_flag = chain_settings[k]['callback']
                output_type = chain_settings[k]['output_type']
                output = chain_settings[k]['output']
                
                return_func_name_each = ''
                if 'return_func_name' in chain_settings[k]:
                    return_func_name_each = chain_settings[k]['return_func_name']
                
                output_parser_type =   chain_settings[k].get('output_parser_type', 'json')
                output_dict =          chain_settings[k]['output_dict']
                         
                return_val = ''       
                if callback_flag is False or output is None or output == '' or len(output) == 0:
                    pass
                else:
                    # target_result = parsed_result
                    # # print(type(target_result), target_result)
                    # if len(chains) > 1:
                    #     target_result = parsed_result[k]
                    target_result = parsed_result[k] if len(chains) > 1 else parsed_result
                    

                    if output == 'all' and isinstance(target_result, dict):
                        output = []
                        for output_k in target_result.keys():
                            output.append(output_k)
                        
                    if output_type == 'str':
                        return_val = ''
                        if output_parser_type == 'json':
                            for o in output:
                                return_val += ('\n' + target_result[o])
                        else:
                            return_val = target_result
                    elif output_type == 'json':
                        return_val = {}
                        for o in output:
                            return_val[o] = target_result[o]
                            try:
                                if 'detail_report' in target_result:
                                    # tmp
                                    print('!'*200)
                                    print('!'*50, 'This part must be modified. This is only for SAJU with detail_report')
                                    print('!'*200)
                                    if not isinstance(target_result[o], dict):
                                        # target_result[o] = json.loads(target_result[o])
                                        target_result[o] = json.dumps(target_result[o], ensure_ascii=False)
                                    if 'title' not in target_result[o]:
                                        raise Exception('title is not in detail_report')
                                    return_val[o] = target_result[o]
                            except Exception as e:
                                if max_extra_tries > 0:
                                    print(f"Unknown error at run_chain_tree() while sending detail_report: {e}, \nmax_extra_tries:{max_extra_tries}")
                                    self.is_retrying = True
                                    return self.run_prompt_chain(chain_settings=chain_settings, prompt_chain_keys=prompt_chain_keys, prev_output=prev_output, callback=callback, max_extra_tries=max_extra_tries-1)
                                else:
                                    self.last_msg = "죄송합니다. 시스템에 문제가 생겼습니다. 다시 입력해주세요."
                                    print(f"Unknown error at run_chain_tree(): {e}, \n{self.last_msg} \nmax_extra_tries:{max_extra_tries}")
                                    return self.last_msg
                    if callback is not None:
                        callback_func(return_val, class_instance, return_func_name_each)
                        
                if callback is not None:
                    time.sleep(0.05)
                    callback_token(token_price[k], class_instance, return_func_name_each)
                self.last_msg = return_val
        
        # Send [END] signal
        if callback is not None and use_start_end_callback is True:
            callback_start_end(False, class_instance, start_end_suffix, return_func_name)
        # print(json.dumps(parsed_result, indent=4,  ensure_ascii=False))
        
        
        return parsed_result
    
    def run_func_chain(self, chain_settings, func_chain_keys, prev_output={'question': 'say anything'}, callback=None):
        try:
            parsed_result = {}
            for k in func_chain_keys:
                self.update_optional_data(chain_settings=chain_settings, chain_key=k)
                
                input_dict = chain_settings[k]['input_dict']
                for in_name in input_dict.keys():
                    for prev_name, prev_val in prev_output.items():
                        if in_name == prev_name:
                            input_dict[in_name] = prev_val
                            
                        if isinstance(prev_val, dict):
                            for parallel_k, parallel_v in prev_val.items():
                                if in_name == parallel_k:
                                    input_dict[in_name] = parallel_v
    
                func_name = chain_settings[k]['func_name']
                output_dict = chain_settings[k]['output_dict']
                start_end_suffix = chain_settings[k]['start_end_suffix']
                input_dict = chain_settings[k]['input_dict']
      
                self.set_input_data(chain_settings=chain_settings, chain_key=k, prev_output=prev_output)

                # wrap_subjects, regen_chain, wrap_reports
                func_input_variable = {'data': input_dict}
                
                # modify chain to generate parallel chain
                if 'function_modify_chain' in chain_settings[k]['type']:
                    func_input_variable['chain_ptr'] = chain_settings
                    
                result = globals()[func_name](**func_input_variable)
                for ok in output_dict.keys():
                    assert ok in result.keys(), f'output_dict.keys() and result.keys() are not matched, they must be same. in run_func_chain(). output_dict.keys(): {output_dict.keys()} != {result.keys()}'
                
                chain_settings[k]['output_dict'] = result
                parsed_result[k] = result
                
                # check if keep_input_output_data is in the setting and the result
                self.update_data(chain_settings=chain_settings, chain_key=k, output_result=result, input_dict=input_dict)
                
                # Not Used Yet
                ## CallBack
                callback_flag = chain_settings[k]['callback']
                output_type = chain_settings[k]['output_type']
                output = chain_settings[k]['output']
                output_stream = chain_settings[k].get('output_stream', None)
                use_start_end_callback = chain_settings[k].get('use_start_end_callback', False)
                if callback_flag is False or output is None or output == '' or len(output) == 0 or callback is None:
                    continue
                return_val = None
                callback_func = callback['func']
                class_instance = callback['instance']
                callback_start_end = callback['func_start_end']
                return_func_name = ''
                if 'func_name' in callback:
                    return_func_name = callback['func_name']
                if 'return_func_name' in chain_settings[k]:
                    return_func_name = chain_settings[k]['return_func_name']
                # send [START]
                if use_start_end_callback:
                    callback_start_end(True, class_instance, start_end_suffix, return_func_name)
                if output_type == 'stream_str_list' and output_stream is not None:
                    result_str_list = []
                    for o_key, o_val in result.items():
                        assert isinstance(o_val, list), 'Error in run_func_chain(), result must be a list including string items'
                        result_str_list.extend(o_val)
                    # send result
                    for r in result_str_list:
                        callback_func(r, class_instance, return_func_name)
                else:
                    if output == 'all':
                        return_val = result
                    elif output_type == 'str':
                        return_val = ''
                        for o in output:
                            return_val += ('\n' + result[o])
                    elif output_type == 'json':
                        return_val = {}
                        for o in output:
                            return_val[o] = result[o]
                    # send result
                    callback_func(return_val, class_instance, return_func_name)
                # send [END]
                if use_start_end_callback:
                    callback_start_end(False, class_instance, start_end_suffix, return_func_name)
        except Exception as e:
            error_message = traceback.format_exc()
            write_log(message=f'{e}, \ntraceback: {error_message}, in run_chain_tree\nparsed_result: {parsed_result}')
        
        return parsed_result
    
    def run_chain_tree(self, chain_settings, process, prev_output={'question': 'say anything'}, callback=None, max_extra_tries=4):
        parsed_result = {}
        try:
            self.is_retrying = False
            prompt_chain_keys = [k for k in list(process.keys()) if 'prompt' in chain_settings[k]['type']]
            func_chain_keys = [k for k in list(process.keys()) if 'func' in chain_settings[k]['type']]
            print('-'*100)
            print('-'*10, 'run_chain_tree', '-'*10)
            print('prompt_chain_keys:', prompt_chain_keys)
            print('func_chain_keys:', func_chain_keys)
            
            
            parsed_result = {}
            if len(prompt_chain_keys) > 0:
                print('-=-= [BEGIN] run_prompt_chain -=-=')
                print('prev_output:', prev_output.keys())
                prompt_result = self.run_prompt_chain(chain_settings=chain_settings, prompt_chain_keys=prompt_chain_keys, prev_output=prev_output, callback=callback, max_extra_tries=max_extra_tries)
                # print('prompt_result:', prompt_result)
                parsed_result.update(prompt_result)
                print('-=-= [END] run_prompt_chain -=-=')
            if len(func_chain_keys) > 0:
                print('-=-= [BEGIN] run_func_chain -=-=')
                print('prev_output:', prev_output.keys())

                func_result = self.run_func_chain(chain_settings=chain_settings, func_chain_keys=func_chain_keys, prev_output=prev_output, callback=callback)
                parsed_result.update(func_result)
                print('-=-= [END] run_func_chain -=-=')
            
            # check if there is next chain
            # this is to locate to another chain process
            # function_chain should be at the end of a chain process
            for fc in func_chain_keys:
                if 'function_chain' in chain_settings[fc]['type']:
                    next_chain = chain_settings['process'][parsed_result[fc]['next_chain']]
                    process[func_chain_keys[0]] = next_chain
                    break

            for i, (k, v) in enumerate(process.items()):
                if v is None:
                    if i == 0:
                        print(json.dumps(parsed_result, indent=4,  ensure_ascii=False))
                    continue
                if not isinstance(parsed_result, dict):
                    proc_key = list(v.keys())[0]
                    parsed_result = {list(chain_settings[proc_key]['input_dict'].keys())[0]: parsed_result}
                parsed_result = self.run_chain_tree(chain_settings, v, prev_output=parsed_result, callback=callback, max_extra_tries=max_extra_tries)
        
        except Exception as e:
            error_message = traceback.format_exc()
            write_log(message=f'{e}, \ntraceback: {error_message}, in run_chain_tree\nparsed_result: {parsed_result}')
        
        self.clear_data()
        return parsed_result
    
    def unify_and_sum_metadata(self, response):
        if self.config.company == 'openai':
            pass
        elif self.config.company == 'google':
            if 'token_usage' in response.response_metadata:
                return response
            response.extra_metadata = response.response_metadata
            prompt_tokens = response.usage_metadata['input_tokens']
            completion_tokens = response.usage_metadata['output_tokens']
            # total_tokens = response.usage_metadata['total_tokens']
            response.response_metadata = {'token_usage': {'prompt_tokens': prompt_tokens, 
                                                          'completion_tokens': completion_tokens}
                                          }

        self.total_tokens['prompt_tokens'] += response.response_metadata['token_usage']['prompt_tokens']
        self.total_tokens['completion_tokens'] += response.response_metadata['token_usage']['completion_tokens']
        
                
        print('[TOKEN] total_tokens:', self.total_tokens)
        return response
    
    
def load_prompt_settings():
    chains = load_chain_setting('./settings/prompts/counsel/chains.json')
    # print(chains)
    return chains

# test
if __name__ == "__main__":
    
    pass