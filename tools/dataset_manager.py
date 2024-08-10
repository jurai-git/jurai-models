import os
import re

import pandas as pd
from dotenv import load_dotenv
from enum import Enum
from typing import Optional
from strong_typing.schema import json_schema_type

load_dotenv()


class ProjectPaths(Enum):
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    DATASET_PATH = os.getenv('DATASET_PATH')
    DATASET_FILE = os.path.join(DATASET_PATH, '{}_dataset{}.csv')
    DATASET_LOGS = os.path.join(DATASET_PATH, 'logs')

    @staticmethod
    def validate_paths():
        for path in ProjectPaths:
            if not path.value:
                raise ValueError(f'{path.name} is not set or invalid.')


class DatasetStages(Enum):
    RAW_TYPE = 'raw'
    CLEANED_TYPE = 'cleaned'
    NORMALIZED_TYPE = 'normalized'
    STANDARDIZED_TYPE = 'standardized'
    TOKENIZED_TYPE = 'tokenized'
    PERSONALIZED_TYPE = 'personalized'


@json_schema_type
class DatasetManager:
    def __init__(self):
        ProjectPaths.validate_paths()
        self.__project_paths = ProjectPaths

        self.create_paths(self.__project_paths.DATASET_PATH.value)
        self.create_paths(self.__project_paths.DATASET_LOGS.value)

    def create_paths(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def raise_path_not_found(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} is not a valid path.')

    def read_dataset(self, dataset_stage: DatasetStages, version: Optional[str] = None, usecols=None) -> pd.DataFrame:
        self.__raise_invalid_dataset_stage(dataset_stage)

        version = self.__validate_version(version)
        file = self.__project_paths.DATASET_FILE.value.format(dataset_stage.value, version)

        return pd.read_csv(file, usecols=usecols)

    def find_datasets(self):
        return list(os.walk(ProjectPaths.DATASET_PATH.value))[0][2]

    def find_logs(self, model=None):
        path = self.__project_paths.DATASET_LOGS.value

        if model is not None:
            path = os.path.join(path, model)

        self.raise_path_not_found(path)

        return list(os.walk(path))[0][2]

    def save_dataset(self, df: pd.DataFrame, dataset_stage: DatasetStages, version: Optional[str] = None):
        self.__raise_invalid_df(df)
        self.__raise_invalid_dataset_stage(dataset_stage)

        version = self.__validate_version(version)
        dataset_file = self.__project_paths.DATASET_FILE.value.format(dataset_stage.value, version)

        df.to_csv(dataset_file, index=False, encoding='utf-8')

    def save_training_history(self, history, model):
        file = 'probability_model.log'
        self.create_paths(os.path.join(self.__project_paths.DATASET_LOGS.value, model))
        logs = self.find_logs(model)

        num = []
        if logs:
            for log in logs:
                last_dot_index = log.rfind('.')
                num += log[last_dot_index + 1:]
            num = int(max(num)) + 1
        else:
            num = 0

        log = f'{file}.{num}'
        log_path = os.path.join(self.__project_paths.DATASET_LOGS.value, model, log)

        with open(log_path, 'w') as f:
            f.write(f'probability model - training {num}\n')

            for key, value in history.history.items():
                f.write(f'{key}: {value}\n')

    def __raise_invalid_df(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError('Invalid DataFrame.')

    def __raise_invalid_dataset_stage(self, dataset_stage):
        if not dataset_stage and not isinstance(dataset_stage.value, str):
            raise ValueError('Invalid dataset stage.')

    def __validate_version(self, version):
        if isinstance(version, str) and version.strip():
            return f'_{version.strip()}'
        return ''
