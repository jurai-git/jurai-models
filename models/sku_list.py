from typing import List
from dataclasses import asdict
from models.datatypes import Model, CoreModelId, ModelArgs


def base_models() -> List[Model]:
    return [
        Model(
            core_model_id=CoreModelId.jurai_arguments,
            is_default_variant=True,
            description_markdown='JurAI Arguments model',
            model_args=asdict(ModelArgs())
        ),
        Model(
            core_model_id=CoreModelId.jurai_probability,
            is_default_variant=True,
            description_markdown='JurAI Probability model',
            model_args=asdict(ModelArgs()),
        ),
        Model(
            core_model_id=CoreModelId.jurai_text_generation,
            is_default_variant=True,
            description_markdown='JurAI Text Generation model',
            model_args=asdict(ModelArgs()),
        )
    ]
