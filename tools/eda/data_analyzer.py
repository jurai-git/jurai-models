import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.__df = df

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__df

    @dataframe.setter
    def dataframe(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise TypeError('Expected a pandas DataFrame')
        self.__df = value

    def plot_histogram(self, column, bins=30):
        if self.__df[column].dtype == 'object':
            lengths = self.__df[column].apply(len)
            plt.figure(figsize=(10, 6))
            lengths.hist(bins=bins)
            plt.xlabel('Length')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of String Lengths in {column}')
            plt.show()
        else:
            plt.figure(figsize=(10, 6))
            self.__df[column].hist(bins=bins)
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {column}')
            plt.show()

    def plot_boxplot(self, column):
        if self.__df[column].dtype != 'object':
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=self.__df[column])
            plt.title(f'Box Plot of {column}')
            plt.show()

    def plot_kde(self, column):
        if self.__df[column].dtype != 'object':
            plt.figure(figsize=(10, 6))
            sns.kdeplot(self.__df[column], shade=True)
            plt.xlabel(column)
            plt.title(f'Density Estimate of {column}')
            plt.show()

    def plot_correlation_matrix(self):
        plt.figure(figsize=(12, 10))
        corr_matrix = self.__df.select_dtypes(include='number').corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()

    def plot_scatter(self, column1, column2):
        if self.__df[column1].dtype == 'object':
            self.__df[column1] = self.__df[column1].astype('category').cat.codes
        if self.__df[column2].dtype == 'object':
            self.__df[column2] = self.__df[column2].astype('category').cat.codes

        plt.figure(figsize=(10, 6))
        self.__df.plot.scatter(x=column1, y=column2)
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.title(f'Scatter Plot between {column1} and {column2}')
        plt.show()

    def detect_outliers_iqr(self, column):
        self.__df['length'] = self.__df[column].apply(len)

        q1 = self.__df['length'].quantile(0.25)
        q3 = self.__df['length'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return self.__df[(self.__df['length'] < lower_bound) | (self.__df['length'] > upper_bound)]

    def detect_outliers_frequency(self, column):
        all_words = ' '.join(self.__df[column]).split()
        word_freq = Counter(all_words)

        min_freq = 5
        rare_words = {word for word, freq in word_freq.items() if freq < min_freq}

        return self.__df[self.__df[column].apply(lambda text: any(word in rare_words for word in text.split()))]

    def plot_outliers_length(self, column):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=self.__df[column].apply(len))
        plt.xlabel('Length')
        plt.title(f'Box Plot of Data Length for {column}')

        outliers = self.detect_outliers_iqr(column)
        for index, row in outliers.iterrows():
            plt.annotate('Outlier', (len(row[column]), 0), textcoords="offset points", xytext=(0, 10), ha='center',
                         color='red')

        plt.show()

    def plot_wordcloud(self, column):
        text = ' '.join(self.__df[column])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Data')
        plt.show()

    def plot_word_count_distribution(self, column):
        word_counts = self.__df[column].apply(lambda x: len(x.split()))

        plt.figure(figsize=(12, 6))
        plt.hist(word_counts, bins=30, alpha=0.5)
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.title(f'Word Count Distribution per Document in {column}')
        plt.show()

    def plot_outliers_length_zoomed(self, column):
        outliers = self.detect_outliers_iqr(column)

        plt.figure(figsize=(12, 6))
        plt.hist(self.__df['length'], bins=30, alpha=0.5, label='Data Length')
        plt.hist(outliers['length'], bins=30, alpha=0.5, color='red', label='Outliers')
        plt.xlim(self.__df['length'].min(), self.__df['length'].max())
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Data Length with Outliers (Zoomed) for {column}')
        plt.legend()
        plt.show()

    def plot_outliers_frequency(self, column):
        self.__df['length'] = self.__df[column].apply(len)
        outliers = self.detect_outliers_iqr(column)

        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=self.__df, x=self.__df.index, y=self.__df['length'],
                        hue=self.__df.index.isin(outliers.index), palette={False: 'blue', True: 'red'}, alpha=0.7)
        plt.xlabel('Index')
        plt.ylabel('Length')
        plt.title(f'Scatter Plot with Outliers Based on Length for {column}')
        plt.show()
