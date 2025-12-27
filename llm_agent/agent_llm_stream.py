import os
import json
import copy
import time
import traceback
from datetime import datetime
import pytz
from typing import Dict, Any, List, Optional

import openai
from openai import OpenAI
from get_config import get_api_key
from llm_config import LLMConfig

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableParallel

# 출력 스키마 강제용 (request 템플릿 유지)
from pydantic import BaseModel as PydanticBaseModel, Field
try:
    # langchain-core 계열(0.2.x) 우선
    from langchain_core.output_parsers import PydanticOutputParser
except Exception:
    # 일부 환경에선 langchain.output_parsers에 있음
    from langchain.output_parsers import PydanticOutputParser


class SubResult(PydanticBaseModel):
    sub_title: str = Field(default="", description="주제의 작은 제목입니다.")
    sub_description: str = Field(default="", description="각 주제의 내용을 설명하는 서론. 서론을 1~2줄로 간략히 작성")
    sub_result: str = Field(default="", description="이 작은 주제에 대한 결과입니다. 마크다운 형식 사용")
    # 기존 파이프라인(이미지 매칭)을 고려해 유지(없어도 동작해야 하므로 optional 성격으로 default 빈 문자열)
    # representative_image_name: str = Field(default="", description="(선택) 이 sub 주제를 대표하는 이미지 이름 1개")


class RequestItem(PydanticBaseModel):
    title: str = Field(default="", description="전체 요청의 제목입니다.")
    description: str = Field(default="", description="전체 요청에 대한 답변 서론 요약. 서론을 1~2줄로 간략히 작성")
    result: str = Field(default="", description="이 요청 항목에 대한 최종 결과입니다.")
    subject: List[SubResult] = Field(default_factory=list, description="이 요청 항목을 구성하는 세부 주제들의 목록입니다.")
    representative_image_name: str = Field(default="", description="(선택) 이 요청 항목을 대표하는 이미지 이름 1개")


class FinalOutput(PydanticBaseModel):
    request: List[RequestItem] = Field(default_factory=list, description="사용자가 요청한 여러 개의 처리 항목들의 목록입니다.")

class _KeyedJSONStreamPostProcessor:
    """
    JSON 문자열 스트리밍 중, '현재 값이 어떤 key(path) 아래에서 쓰이는지'를 추적하여
    문자 1개 단위로 아래 형태의 dict 이벤트를 만들어냅니다.

    예)
      {"request_0": {"title": "안"}}
      {"request_0": {"subject_0": {"sub_title": "부"}}}
    """

    def __init__(self) -> None:
        self._started = False  # 첫 '{'를 만나기 전까지 무시
        self._in_string = False
        self._escape = False
        self._unicode_escape = False
        self._unicode_buf = ""
        self._string_buf = ""  # 현재 읽고 있는 문자열(키/값)

        # 컨텍스트 스택
        # object: {"type":"object", "expect":"key|colon|value|comma_or_end", "last_key": Optional[str]}
        # array:  {"type":"array",  "expect":"value|comma_or_end", "name": Optional[str], "index": int}
        self._stack: List[Dict[str, Any]] = []

        # 현재 문자열이 key인지 value인지
        self._reading_key = False

        # 현재 "값 문자열"이 어떤 경로인지
        # ("request", i, key) 또는 ("request", i, "subject", j, key)
        self._active_value_path: Optional[tuple] = None

    def _push_object(self) -> None:
        self._stack.append({"type": "object", "expect": "key", "last_key": None})

    def _push_array(self, name: Optional[str]) -> None:
        self._stack.append({"type": "array", "expect": "value", "name": name, "index": -1})

    def _current_indices(self) -> Dict[str, int]:
        indices: Dict[str, int] = {}
        for ctx in self._stack:
            if ctx.get("type") == "array" and ctx.get("name") and ctx.get("index", -1) >= 0:
                indices[str(ctx["name"])] = int(ctx["index"])
        return indices

    def _build_event(self, ch: str) -> Optional[Dict[str, Any]]:
        p = self._active_value_path
        if not p:
            return None

        if len(p) == 3 and p[0] == "request":
            _, req_i, key = p
            return {f"request_{req_i}": {key: ch}}

        if len(p) == 5 and p[0] == "request" and p[2] == "subject":
            _, req_i, _, subj_i, key = p
            return {f"request_{req_i}": {f"subject_{subj_i}": {key: ch}}}

        return None

    def _build_value_end_meta(self, full_value: str) -> Optional[Dict[str, Any]]:
        """
        특정 값 문자열이 '완성'되었을 때(닫는 따옴표를 만났을 때) 한 번만 내보내는 메타 이벤트.

        - LLM chunk(JSON keyed stream) 자체는 그대로 유지하고,
          서버가 이 메타 이벤트를 가로채서 별도 비동기 작업(예: 이미지 다운로드)을 트리거하기 위한 용도.
        - 현재는 request[*].representative_image_name 에 대해서만 발생시킵니다.
        """
        p = self._active_value_path
        if not p:
            return None

        # ("request", req_i, key)
        if len(p) == 3 and p[0] == "request":
            _, req_i, key = p
            if key != "representative_image_name":
                return None
            return {
                "_stream_meta": {
                    "type": "value_end",
                    "path": {"request_i": int(req_i), "key": str(key)},
                    "value": str(full_value),
                }
            }

        return None

    def _set_active_value_path(self) -> None:
        # 가장 가까운 object의 last_key
        last_key = None
        for ctx in reversed(self._stack):
            if ctx.get("type") == "object":
                last_key = ctx.get("last_key")
                break
        if not last_key:
            self._active_value_path = None
            return

        idx = self._current_indices()
        # subject 컨텍스트가 있으면(=subject 배열 내부라면) 무조건 subject 경로를 우선한다.
        # (기존에는 request가 먼저 매칭되어 {"request_0":{"sub_title":...}}처럼 subject_0가 누락되는 버그가 있었음)
        if "request" in idx and "subject" in idx:
            req_i = idx["request"]
            subj_i = idx["subject"]
            self._active_value_path = ("request", req_i, "subject", subj_i, last_key)
            return

        if "request" in idx:
            req_i = idx["request"]
            # request[*].subject는 배열이므로 값 문자열 대상 아님
            if last_key == "subject":
                self._active_value_path = None
                return
            # request[*].title/description/result/representative_image_name 등
            self._active_value_path = ("request", req_i, last_key)
            return

        self._active_value_path = None

    def _inc_array_index_if_needed(self) -> None:
        if not self._stack:
            return
        top = self._stack[-1]
        if top.get("type") == "array" and top.get("expect") == "value":
            top["index"] = int(top.get("index", -1)) + 1
            top["expect"] = "comma_or_end"

    def feed(self, text: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        for raw_ch in text:
            ch = raw_ch

            if not self._started:
                if ch == "{":
                    self._started = True
                    self._push_object()
                continue

            # 문자열 내부
            if self._in_string:
                if self._unicode_escape:
                    self._unicode_buf += ch
                    if len(self._unicode_buf) == 4:
                        try:
                            decoded = chr(int(self._unicode_buf, 16))
                        except Exception:
                            decoded = ""
                        self._unicode_escape = False
                        self._unicode_buf = ""
                        self._string_buf += decoded
                        evt = self._build_event(decoded)
                        if evt:
                            events.append(evt)
                    continue

                if self._escape:
                    self._escape = False
                    if ch == "n":
                        decoded = "\n"
                    elif ch == "t":
                        decoded = "\t"
                    elif ch == "r":
                        decoded = "\r"
                    elif ch == "b":
                        decoded = "\b"
                    elif ch == "f":
                        decoded = "\f"
                    elif ch == "u":
                        self._unicode_escape = True
                        self._unicode_buf = ""
                        continue
                    else:
                        decoded = ch
                    self._string_buf += decoded
                    evt = self._build_event(decoded)
                    if evt:
                        events.append(evt)
                    continue

                if ch == "\\":
                    self._escape = True
                    continue

                if ch == '"':
                    # 문자열 종료
                    self._in_string = False
                    if self._reading_key:
                        # object key 확정
                        for ctx in reversed(self._stack):
                            if ctx.get("type") == "object":
                                ctx["last_key"] = self._string_buf
                                ctx["expect"] = "colon"
                                break
                    else:
                        # value 종료
                        meta_evt = self._build_value_end_meta(self._string_buf)
                        if meta_evt:
                            events.append(meta_evt)
                        self._active_value_path = None
                        for ctx in reversed(self._stack):
                            if ctx.get("type") == "object":
                                ctx["expect"] = "comma_or_end"
                                break
                    self._string_buf = ""
                    self._reading_key = False
                    continue

                # 일반 문자
                self._string_buf += ch
                evt = self._build_event(ch)
                if evt:
                    events.append(evt)
                continue

            # 문자열 밖
            if ch.isspace():
                continue

            # array 값 시작(인덱스 증가)
            if self._stack and self._stack[-1].get("type") == "array" and self._stack[-1].get("expect") == "value":
                if ch in ['{', '[', '"'] or ch.isdigit() or ch in ['t', 'f', 'n', '-']:
                    self._inc_array_index_if_needed()

            if ch == "{":
                self._push_object()
                continue
            if ch == "}":
                if self._stack:
                    self._stack.pop()
                continue
            if ch == "[":
                # 직전 object key를 array name으로 사용 (request, subject)
                array_name = None
                for ctx in reversed(self._stack):
                    if ctx.get("type") == "object":
                        array_name = ctx.get("last_key")
                        break
                self._push_array(array_name)
                continue
            if ch == "]":
                if self._stack:
                    self._stack.pop()
                continue
            if ch == ":":
                for ctx in reversed(self._stack):
                    if ctx.get("type") == "object":
                        ctx["expect"] = "value"
                        break
                continue
            if ch == ",":
                if self._stack:
                    top = self._stack[-1]
                    if top.get("type") == "object":
                        top["expect"] = "key"
                    elif top.get("type") == "array":
                        top["expect"] = "value"
                continue
            if ch == '"':
                # 문자열 시작: key인지 value인지 판정
                self._in_string = True
                self._string_buf = ""
                self._escape = False
                self._unicode_escape = False
                self._unicode_buf = ""
                self._reading_key = False

                for ctx in reversed(self._stack):
                    if ctx.get("type") == "object":
                        if ctx.get("expect") == "key":
                            self._reading_key = True
                        else:
                            self._reading_key = False
                            self._set_active_value_path()
                        break
                continue

            # 숫자/true/false/null 등은 문자 기반 key 스트림 대상이 아님(최종 파싱에만 반영)

        return events

def load_prompt_file(file_path):
    """프롬프트 파일을 로드하는 함수"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"프롬프트 파일 로드 실패: {file_path}, {e}")
        return ""

MAX_INPUT_TOKENS_GPT = 100000
MAX_INPUT_TOKENS_GEMINI = 150000

def write_log(message, log_dir='log/error/'):
    """간단한 로그 작성 함수"""
    now = datetime.now()
    kst = pytz.timezone('Asia/Seoul')
    now_kst = now.astimezone(kst)
    
    date_str = now_kst.strftime("%Y-%m-%d")
    time_str = now_kst.strftime("%H%M%S")

    log_path = os.path.join(log_dir, date_str)
    os.makedirs(log_path, exist_ok=True)

    log_file_name = f"{time_str}.txt"
    log_file_path = os.path.join(log_path, log_file_name)

    i = 0
    while os.path.exists(log_file_path):
        i += 1
        log_file_name = f"{time_str}_{i}.txt"
        log_file_path = os.path.join(log_path, log_file_name)    

    try:
      with open(log_file_path, "w") as f:
          timestamp = now_kst.strftime("%Y-%m-%d %H:%M:%S")
          log_msg = f"{timestamp} - {message}"
          print(log_msg)
          f.write(log_msg) 
    except Exception as e:
          print(f"로그 파일 생성 실패: {e}")
          pass


class LangChainModuleStream():
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_info = get_api_key(self.config.api_key_path)
        self.llm = {}
        self.max_input_tokens = 100000
        self.init_llm(self.config.model)
            
    def get_model_company(self, model):
        if 'gpt' in model:
            self.config.company = 'openai'
        elif 'gemini' in model:
            self.config.company = 'google'
        else:
            assert False, 'Currently gpt or gemini can be used only'
        
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
            )
            
        elif self.config.company == 'google':
            self.max_input_tokens = MAX_INPUT_TOKENS_GEMINI
            from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.api_info['google']['credentials']
            self.llm[model] = ChatVertexAI(
                project=self.api_info['google']['project_id'],
                location=self.api_info['google']['region'],
                model=model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens_output,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
                },
            )
        else:
            assert False, 'company name should be openai or google'
    
    def set_json_output(self, output_dict):
        """JSON 출력을 위한 파서 설정"""
        response_schemas = []
        for k, v in output_dict.items():
            response_schemas.append(ResponseSchema(name=k, description=v))
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        return output_parser
    
    def create_chain(self, model_name, instructions, prompts, input_dict, output_dict=None, output_parser_type: Optional[str] = None):
        """간단한 체인 생성"""
        self.init_llm(model_name)
        llm = self.llm[model_name]
        
        sys_msg = SystemMessage(instructions)
        
        # 1) Pydantic 기반 구조 강제(중첩 request/subject까지 검증)
        if output_parser_type == 'pydantic':
            output_parser = PydanticOutputParser(pydantic_object=FinalOutput)
            hum_msg = HumanMessagePromptTemplate.from_template(
                prompts + "\n{format_instructions}"
            )
            prompt_template = ChatPromptTemplate(
                messages=[sys_msg, hum_msg],
                input_variables=list(input_dict.keys()),
                partial_variables={"format_instructions": output_parser.get_format_instructions()}
            )

        # 2) 문자열 출력
        elif output_dict is None:
            # 문자열 출력
            hum_msg = HumanMessagePromptTemplate.from_template(prompts)
            output_parser = StrOutputParser()
            prompt_template = ChatPromptTemplate(
                messages=[sys_msg, hum_msg],
                input_variables=list(input_dict.keys())
            )
        else:
            # JSON 출력
            output_parser = self.set_json_output(output_dict)
            hum_msg = HumanMessagePromptTemplate.from_template(
                prompts + "\n{format_instructions}"
            )
            prompt_template = ChatPromptTemplate(
                messages=[sys_msg, hum_msg],
                input_variables=list(input_dict.keys()),
                partial_variables={"format_instructions": output_parser.get_format_instructions()}
            )
            
        chain = prompt_template | llm
        return chain, output_parser
    
    def simple_stream_handler(self, result, callback=None):
        """간단한 스트리밍 핸들러 - 받은 그대로 즉시 전송"""
        content = ''
        
        for chunk in result:
            if hasattr(chunk, 'content') and chunk.content:
                content += chunk.content
                # 콜백이 있으면 받은 내용을 즉시 전송
                if callback:
                    callback_func = callback.get('func')
                    class_instance = callback.get('instance')
                    func_name = callback.get('func_name', 'default')
                    if callback_func:
                        callback_func(chunk.content, class_instance, func_name)
        
        # 전체 내용을 AIMessage로 반환
        return AIMessage(content=content)

    def simple_stream_handler_with_keys(self, result, callback=None):
        """
        JSON 스트림을 받으면서, 현재 작성 중인 값의 key(path)를 추적하여
        문자 1개마다 {"request_0": {"title": "안"}} 같은 JSON 이벤트를 callback으로 전달.

        - callback에는 dict가 아닌 "JSON 문자열"을 전달합니다(출력/전송 시 깨짐 방지).
        - 최종 content 누적은 유지하여 run_simple_chain의 파싱 로직은 그대로 동작합니다.
        """
        content = ''
        post = _KeyedJSONStreamPostProcessor()

        for chunk in result:
            if hasattr(chunk, 'content') and chunk.content:
                content += chunk.content
                if callback:
                    callback_func = callback.get('func')
                    class_instance = callback.get('instance')
                    func_name = callback.get('func_name', 'default')
                    if callback_func:
                        for evt in post.feed(chunk.content):
                            callback_func(json.dumps(evt, ensure_ascii=False), class_instance, func_name)

        return AIMessage(content=content)
    
    def run_simple_chain(self, chain_settings, chain_key, prev_output, callback=None):
        """간단한 체인 실행"""
        try:
            # 체인 설정 추출
            chain_config = chain_settings[chain_key]
            
            # 입력 데이터 준비
            input_dict = chain_config['input_dict'].copy()
            input_dict.update(prev_output)
            
            # 입력값이 딕셔너리인 경우 JSON 문자열로 변환
            for key, value in input_dict.items():
                if isinstance(value, dict):
                    input_dict[key] = json.dumps(value, ensure_ascii=False)
                elif value is None or (isinstance(value, str) and len(value) == 0):
                    input_dict[key] = ' '
                elif isinstance(value, bool):
                    input_dict[key] = str(value)
            
            # 프롬프트 파일에서 instructions와 prompts 로드
            instructions = chain_config.get('instructions', '')
            prompts = chain_config.get('prompts', '')
            
            # instructions_data가 있으면 파일에서 로드
            if 'instructions_data' in chain_config:
                instructions_path = os.path.join('./settings/prompts/fodoit_stream', chain_config['instructions_data'][0])
                instructions = load_prompt_file(instructions_path)
            
            # prompts_data가 있으면 파일에서 로드
            if 'prompts_data' in chain_config:
                prompts_path = os.path.join('./settings/prompts/fodoit_stream', chain_config['prompts_data'][0])
                prompts = load_prompt_file(prompts_path)
            
            # 체인 생성
            chain, output_parser = self.create_chain(
                model_name=chain_config['model'],
                instructions=instructions,
                prompts=prompts,
                input_dict=input_dict,
                output_dict=chain_config.get('output_dict'),
                output_parser_type=chain_config.get('output_parser_type')
            )
            
            print('='*100)
            print(f'Running chain: {chain_key}')
            print(f'Input keys: {list(input_dict.keys())}')
            print('='*100)
            
            # 스트리밍 여부에 따라 실행
            if chain_config.get('output_stream', True):
                # 스트리밍 실행
                if callback:
                    callback_start_end = callback.get('func_start_end')
                    class_instance = callback.get('instance')
                    func_name = callback.get('func_name', 'default')
                    if callback_start_end:
                        callback_start_end(True, class_instance, '', func_name)
                
                result_stream = chain.stream(input_dict)
                if chain_config.get('stream_emit_with_keys', False):
                    result = self.simple_stream_handler_with_keys(result_stream, callback)
                else:
                    result = self.simple_stream_handler(result_stream, callback)
                
                if callback and callback_start_end:
                    callback_start_end(False, class_instance, '', func_name)
            else:
                # 일반 실행
                result = chain.invoke(input_dict)
            
            # 결과 파싱
            parsed_result = output_parser.parse(result.content)

            # PydanticOutputParser를 쓴 경우 BaseModel -> dict로 변환(기존 코드 호환)
            if hasattr(parsed_result, 'model_dump'):
                parsed_result = parsed_result.model_dump()
            elif hasattr(parsed_result, 'dict'):
                parsed_result = parsed_result.dict()
            
            print('='*100)
            print(f'Parsed result: {parsed_result}')
            print('='*100)
            
            return parsed_result
            
        except Exception as e:
            error_message = traceback.format_exc()
            print(f"Error in run_simple_chain: {e}")
            print(error_message)
            write_log(message=f'{e}, \ntraceback: {error_message}')
            
            # 콜백으로 에러 메시지 전송
            if callback:
                callback_func = callback.get('func')
                class_instance = callback.get('instance')
                func_name = callback.get('func_name', 'default')
                if callback_func:
                    error_msg = "죄송합니다. 시스템에 문제가 생겼습니다. 다시 입력해주세요."
                    callback_func({'error_msg': error_msg}, class_instance, func_name, 4000)
            
            return {"error": "처리 중 오류가 발생했습니다."}
    
    def run_chain_tree(self, chain_settings, process, prev_output={'question': 'say anything'}, callback=None, max_extra_tries=1):
        """간단한 체인 트리 실행"""
        try:
            # 현재는 단일 체인만 처리 (stream_report)
            for chain_key in process.keys():
                if process[chain_key] is None:  # leaf node
                    return self.run_simple_chain(chain_settings, chain_key, prev_output, callback)
                else:
                    # 중간 노드가 있다면 재귀 호출
                    result = self.run_simple_chain(chain_settings, chain_key, prev_output, callback)
                    return self.run_chain_tree(chain_settings, process[chain_key], result, callback, max_extra_tries)
            
        except Exception as e:
            error_message = traceback.format_exc()
            print(f"Error in run_chain_tree: {e}")
            print(error_message)
            write_log(message=f'{e}, \ntraceback: {error_message}')
            return {"error": "처리 중 오류가 발생했습니다."}


# 테스트용 코드
if __name__ == "__main__":
    pass
