import re
import pandas as pd
from tools.dataset.manager import DatasetManager
from tools.tjmg_scraper.purify import PurifyScraper


def classify_sumulas(df, limit):
    positive_sumulas = [
        "ACOLHERAM A PRELIMINAR", "ACOLHERAM A QUESTÃO DE ORDEM",
        "ACOLHERAM O CONFLITO", "ACOLHERAM OS EMBARGOS",
        "CONCEDERAM A ORDEM", "CONFIRMARAM A SENTENÇA",
        "DECLARARAM A COMPETÊNCIA DO JUÍZO SUSCITADO",
        "EM REEXAME NECESSÁRIO, CONFIRMARAM A SENTENÇA",
        "RECURSO DE AGRAVO PROVIDO", "RECURSO DE APELAÇÃO PROVIDO",
        "REJEITARAM AS PRELIMINARES E DERAM PROVIMENTO AOS RECURSOS"
    ]
    partial_sumulas = [
        "ACOLHERAM PARCIALMENTE OS EMBARGOS", "CONCEDERAM PARCIALMENTE A ORDEM",
        "CONFIRMARAM A SENTENÇA E JULGARAM PREJUDICADO O RECURSO",
        "CONHECERAM PARCIALMENTE DA IMPETRAÇÃO", "PROVIMENTO PARCIAL",
        "RECURSO DE AGRAVO PARCIALMENTE PROVIDO", "RECURSO DE APELAÇÃO PARCIALMENTE PROVIDO",
        "RECURSO PARCIALMENTE PROVIDO", "REFORMARAM PARCIALMENTE A SENTENÇA, PREJUDICADO O RECURSO",
        "REJEITARAM AS PRELIMINARES E DERAM PARCIAL PROVIMENTO AOS RECURSOS"
    ]
    negative_sumulas = [
        "DENEGARAM A SEGURANÇA", "DENEGARAM O HABEAS CORPUS",
        "INDEFERIRAM O PEDIDO", "JULGARAM PREJUDICADO O RECURSO",
        "NÃO CONHECERAM DA IMPETRAÇÃO", "NÃO CONHECERAM DA ORDEM",
        "NÃO CONHECERAM DO HABEAS CORPUS", "NÃO CONHECERAM DO RECURSO",
        "NÃO CONHECERAM DO WRIT", "NEGARAM PROVIMENTO",
        "NEGARAM PROVIMENTO À APELAÇÃO", "NEGARAM PROVIMENTO AO AGRAVO",
        "RECURSO NÃO PROVIDO", "REJEITARAM AS PRELIMINARES E NEGARAM PROVIMENTO AOS RECURSOS"
    ]

    positive_df = df[df['sumula'].isin(positive_sumulas)]

    if len(positive_df) < limit:
        positive_df = add_entries(df, positive_df, "RECURSO PROVIDO", limit - len(positive_df))

    partial_df = df[df['sumula'].isin(partial_sumulas)]
    negative_df = df[df['sumula'].isin(negative_sumulas)]

    if len(negative_df) < limit:
        remaining_limit = limit - len(negative_df)
        negative_df = add_entries(df, negative_df, "REJEITARAM OS EMBARGOS", int(remaining_limit * 0.22))
        negative_df = add_entries(df, negative_df, "NEGARAM PROVIMENTO AOS RECURSOS", int(remaining_limit * 0.67))
        negative_df = add_entries(df, negative_df, "DENEGARAM A ORDEM", int(remaining_limit * 0.11))

    negative_df.loc[:, 'sumula'] = 0
    positive_df.loc[:, 'sumula'] = 1
    partial_df.loc[:, 'sumula'] = 2

    return pd.concat([positive_df, partial_df, negative_df])


def add_entries(df, target_df, condition, limit):
    additional_entries = df[df['sumula'] == condition].head(limit)
    return pd.concat([target_df, additional_entries])


def balance_probability_dataset(df, as_numpy=False):
    df_0 = df[df['sumula'] == 0]
    df_1 = df[df['sumula'] == 1]
    df_2 = df[df['sumula'] == 2]

    min_count = min(len(df_1), len(df_0), len(df_2))

    df_0 = df_0.sample(min_count, random_state=42)
    df_1 = df_1.sample(min_count, random_state=42)
    df_2 = df_2.sample(min_count, random_state=42)

    df = pd.concat([df_0, df_1, df_2]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df.to_numpy() if as_numpy else df


def filter_by_length(df, column, min_length=0, max_length=float('inf')):
    return df[df[column].apply(lambda x: min_length <= len(x) <= max_length)]


def apply_regex(text):
    patterns = [r'[aA][rR][tT]\s*\.\s*\d+(\.?\d*\.?\/?\d*)', r'§\d+°?', r'\d+\.\d+\.\d+\.\d+\.?-?\d*(\/\d+)']

    for pattern in patterns:
        text = re.sub(pattern, '', text)
    return text


def prepare_dataset(df, filename, min_length, max_length, sumula_limit=32446):
    purify = PurifyScraper()

    df['ementa'] = df['ementa'].apply(purify.clean_text).apply(apply_regex)

    df = filter_by_length(df, 'ementa', min_length, max_length)
    df = classify_sumulas(df, sumula_limit)
    df = balance_probability_dataset(df)

    print(f'len: {len(df)}')
    print(df['sumula'].value_counts())

    DatasetManager().save_dataset(df, filename)


def main():
    dataset_manager = DatasetManager()
    df = dataset_manager.read_dataset('raw_dataset.csv', usecols=['ementa', 'sumula'])

    prepare_dataset(df, 'arguments_dataset.csv', 256, 1280)
    prepare_dataset(df, 'probability_dataset.csv', 256, 1024)


if __name__ == '__main__':
    main()
