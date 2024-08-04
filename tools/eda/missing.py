import os

import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame


class MissingValueFinder:
    def __init__(self, df: DataFrame):
        self.__df = df
        self.__missing_values = self.__df.isnull().sum()

    @property
    def dataframe(self) -> DataFrame:
        return self.__df

    @dataframe.setter
    def dataframe(self, value: DataFrame):
        if not isinstance(value, DataFrame):
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


if __name__ == '__main__':
    dataset_dir = '../datasets'
    dataset_file = f'{dataset_dir}/raw_dataset.csv'

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    mvf = MissingValueFinder(pd.read_csv(dataset_file))

    if mvf.is_missing:
        mvf.identify_missing_values()
        mvf.summary()
        mvf.drop_missing()
        mvf.dataframe.to_csv(dataset_file, index=False)
