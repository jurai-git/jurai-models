import os
import pandas as pd
from dotenv import load_dotenv
from enum import Enum
from strong_typing.schema import json_schema_type

load_dotenv()


class ProjectPaths(Enum):
    PROJECT_PATH = os.getenv('PROJECT_PATH')
    DATASET_PATH = os.getenv('DATASET_PATH')
    DATASET_LOGS = os.path.join(DATASET_PATH, 'logs')

    @staticmethod
    def validate_paths():
        for path in ProjectPaths:
            if not path.value:
                raise ValueError(f'{path.name} is not set or invalid.')


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

    def read_dataset(self, dataset, usecols=None) -> pd.DataFrame:
        dataset = os.path.join(self.__project_paths.DATASET_PATH.value, dataset)

        self.raise_path_not_found(dataset)
        return pd.read_csv(dataset, usecols=usecols)

    def find_datasets(self):
        return list(os.walk(ProjectPaths.DATASET_PATH.value))[0][2]

    def find_logs(self, model=None):
        path = self.__project_paths.DATASET_LOGS.value

        if model is not None:
            path = os.path.join(path, model)
        self.raise_path_not_found(path)

        return list(os.walk(path))[0][2]

    def save_dataset(self, df: pd.DataFrame, name: str):
        self.__raise_invalid_df(df)
        dataset = os.path.join(self.__project_paths.DATASET_PATH.value, name)

        df.to_csv(dataset, index=False, encoding='utf-8')

    def save_training_history(self, model, history, model_name):
        file = 'probability_model.log'
        self.create_paths(os.path.join(self.__project_paths.DATASET_LOGS.value, model_name))
        logs = self.find_logs(model_name)

        num = []
        if logs:
            for log in logs:
                last_dot_index = log.rfind('.')
                num += log[last_dot_index + 1:]
            num = int(max(num)) + 1
        else:
            num = 0

        log = f'{file}.{num}'
        log_path = os.path.join(self.__project_paths.DATASET_LOGS.value, model_name, log)

        emissions = pd.read_csv('emissions.csv').iloc[-1].to_dict()

        with open(log_path, 'w') as f:
            f.write(f'{model_name} model - training {num}\n\n')

            for k, v in emissions.items():
                f.write(f'{k}: {v}\n')

            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            f.write("\nModel Summary:\n")
            f.write('\n'.join(model_summary) + '\n\n')

            f.write('Training History:\n')
            for k, v in history.history.items():
                f.write(f'{k}: {v}\n')

    def __raise_invalid_df(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError('Invalid DataFrame.')
