import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tools.dataset.manager import DatasetManager
from tools.tjmg_scraper.purify import PurifyScraper


def extract_sentences(text, num_sentences=3):
    sentences = re.split(r'(?<!\d)\.(?!\d)', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if len(sentences) <= num_sentences:
        return sentences

    tfidf_matrix = TfidfVectorizer().fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:][::-1]

    return [sentences[i] for i in top_sentence_indices]


def verify_args(sentenca):
    total_len = len(sentenca)
    m = sum(1 for letra in sentenca if letra.isupper())
    percentage = (m / total_len) * 100

    return percentage < 20


def load_and_clean_texts():
    texts = DatasetManager().read_dataset('arguments_dataset.csv')['ementa'].tolist()
    return [PurifyScraper().clear_input(text, False) for text in texts]


def create_labels(texts):
    list_text_arguments = [extract_sentences(text) for text in texts]
    labels = []

    for arguments in list_text_arguments:
        label = ''

        for argument in arguments:
            if len(argument) > 15 and verify_args(argument):
                label += ' ' + argument

        labels.append(label.strip())

    return labels


def balance_dataset(x_min, y_min, x_max, y_max):
    df = DatasetManager().read_dataset('arguments_dataset.csv')

    filtered_df = df[(df['ementa'].str.len() >= x_min) & (df['ementa'].str.len() <= x_max) &
                     (df['labels'].str.len() >= y_min) & (df['labels'].str.len() <= y_max)]

    DatasetManager().save_dataset(filtered_df, 'arguments_dataset.csv')
    print(f'{len(filtered_df)} rows filtered and the raw is: {len(df)}')


def length_occur():
    df = DatasetManager().read_dataset('arguments_dataset.csv')

    data = df['ementa']
    new_data = pd.DataFrame({'labels': [str(i) for i in data]})
    tamanhos = new_data['labels'].apply(len)
    contagem_tamanhos = tamanhos.value_counts().sort_index()

    print('\nOcorrências de cada tamanho:')
    for tamanho, contagem in contagem_tamanhos.items():
        print(f'Tamanho {tamanho}: {contagem} ocorrências')


if __name__ == "__main__":
    texts = load_and_clean_texts()
    labels = create_labels(texts)

    pd.DataFrame(data={
        'ementa': texts,
        'labels': labels
    }).to_csv('../../datasets/arguments_dataset.csv', index=False)

    balance_dataset(256, 64, 1280, 768)
    length_occur()
