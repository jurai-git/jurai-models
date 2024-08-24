import pandas as pd
import matplotlib.pyplot as plt
from tools.dataset_manager import DatasetManager


class MissingValueFinder:
    def __init__(self, df: pd.DataFrame):
        self.__df = df
        self.__missing_values = self.__df.isnull().sum()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__df

    @dataframe.setter
    def dataframe(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise TypeError('Expected a pandas DataFrame')
        self.__df = value

    @property
    def is_missing(self) -> bool:
        self.__missing_values = self.__df.isnull().sum()
        return not self.__missing_values[self.__missing_values > 0].empty

    def identify_missing_values(self):
        if not self.is_missing:
            return

        plt.figure(figsize=(10, 6))
        self.__missing_values[self.__missing_values > 0].sort_values(ascending=False).plot.bar()
        plt.ylabel('Missing values count')
        plt.title('Missing values per column')
        plt.show()

    def summary(self):
        self.__missing_values = self.__df.isnull().sum()

        total_missing = self.__missing_values.sum()
        percent_missing_per_column = (self.__missing_values / len(self.__df)) * 100

        summary_df = pd.DataFrame({
            'Missing Values': self.__missing_values,
            'Percentage': percent_missing_per_column
        })

        print(f'Total missing values: {total_missing}')
        print(summary_df[summary_df['Missing Values'] > 0])

    def drop_missing(self):
        self.__df = self.__df.dropna()
