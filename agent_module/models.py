from typing import Dict, TypedDict, List, Optional, Type, Any, Tuple
from pydantic import BaseModel, create_model, Field, conint, confloat

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
        elif data_type == "number" or data_type == "float" or data_type == "num":
            if "minimum" in property_schema and "maximum" in property_schema:
                return confloat(ge=property_schema["minimum"], le=property_schema["maximum"])
            return float
        elif data_type == "string" or data_type == "str":
            return str
        elif data_type == "boolean" or data_type == "bool":
            return bool
        elif data_type == "array" or data_type == "list" or data_type == "list[str]":
            items_schema = property_schema.get("items", {})
            items_type = items_schema.get("type", "string")
            if items_type == "object" or items_type == "obj" or items_type == "json":
                inner_model = create_pydantic_model_from_json(items_schema)
                return List[inner_model]
            else:
                python_item_type = map_data_type_to_python(items_type, items_schema)
                return List[python_item_type]
        elif data_type == "object" or data_type == "obj" or data_type == "json":
            return create_pydantic_model_from_json(property_schema)
        else:
            return str

    fields: Dict[str, Tuple[Type, Any]] = {}

    properties = gemini_json_schema.get("properties", {})
    for field_name, property_schema in properties.items():
        data_type = property_schema.get("type")
        if not data_type:
            continue

        python_type = map_data_type_to_python(data_type, property_schema)
        field_description = property_schema.get("description", field_name)
        fields[field_name] = (python_type, Field(description=field_description))

    model: Type[BaseModel] = create_model("DynamicStructuredOutputModel", **fields)
    return model

def create_state_from_json(json_data: str | dict) -> Type[TypedDict]:
    """
    JSON 파일로부터 State TypedDict를 동적으로 생성합니다.

    Args:
        json_data: JSON 파일 경로 또는 JSON 객체

    Returns:
        동적으로 생성된 State TypedDict
    """
    if isinstance(json_data, str):
        with open(json_data, "r") as f:
            config = json.load(f)
    elif isinstance(json_data, dict):
        config = json_data
    else:
        raise TypeError("json_data must be a JSON file path (str) or a JSON object (Dict)")

    state_fields = {}
    for field_name, field_config in config["state_fields"].items():
        field_type_str = field_config["type"]
        field_optional = field_config.get("optional", False)

        if field_type_str == "str":
            field_type = str
        elif field_type_str == "int":
            field_type = int
        elif field_type_str == "float":
            field_type = float
        elif field_type_str == "Dict":
            field_type = Dict
        elif field_type_str == "List[Dict]":
            field_type = List[Dict]
        elif field_type_str == "List[str]":
            field_type = List[str]
        elif field_type_str == "List":
            field_type = List[Any]
        elif field_type_str == "BaseModel":
            field_type = BaseModel
        else:
            raise ValueError(f"Unsupported type: {field_type_str}")

        if field_optional:
            field_type = Optional[field_type]

        state_fields[field_name] = field_type

    output_model_config = config.get("output_model", None)
    if output_model_config:
        output_model_fields = {}
        for field_name, field_config in output_model_config["fields"].items():
            field_type_str = field_config["type"]
            if field_type_str == "str":
                field_type = str
            elif field_type_str == "int":
                field_type = int
            elif field_type_str == "float":
                field_type = float
            elif field_type_str == "Dict":
                field_type = Dict
            elif field_type_str == "List[Dict]":
                field_type = List[Dict]
            elif field_type_str == "List[str]":
                field_type = List[str]
            elif field_type_str == "List":
                field_type = List[Any]
            else:
                raise ValueError(f"Unsupported type: {field_type_str}")
            output_model_fields[field_name] = (field_type, ...)
        DynamicOutputModel = create_model("DynamicOutputModel", **output_model_fields)
        state_fields["output_model"] = Type[DynamicOutputModel]
        state_fields["structured_output"] = Optional[DynamicOutputModel]

    State = TypedDict("State", state_fields)
    return State 