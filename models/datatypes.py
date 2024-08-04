from enum import Enum
from strong_typing.schema import json_schema_type


@json_schema_type
class CoreModelId(Enum):
    # JurAI 1.0 family
    jurai_probability = 'JurAI-Probability-1.0'
    jurai_arguments = 'JurAI-Arguments-1.0'
    jurai_text_generation = 'JurAI-Text-Generation-1.0'
