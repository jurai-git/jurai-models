from enum import Enum
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Optional, Dict, Any
from strong_typing.schema import json_schema_type


@json_schema_type
class CoreModelId(Enum):
    # JurAI 1.0 family
    jurai_probability = 'JurAI-Probability-1.0'
    jurai_arguments = 'JurAI-Arguments-1.0'
    jurai_text_generation = 'JurAI-Text-Generation-1.0'


@json_schema_type
class Model(BaseModel):
    core_model_id: CoreModelId
    is_default_variant: bool
    max_seq_length: int
    model_args: Dict[str, Any]


@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 6
    vocab_size: int = 90_000
    max_batch_size: int = 48
    max_seq_len: int = 2048
    max_target_len: int = 128
    learning_rate: int = 1e-4
    epochs: int = 10
