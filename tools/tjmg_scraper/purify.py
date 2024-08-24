import re
import unicodedata
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class RegexPatterns:
    tjmg_patterns: Dict[str, str] = field(default_factory=lambda: {
        r'\d{4}\.\d{2}\.\d{6}-\d{1}/\d{3}': '',
        r'Número do \d+-': '',
        r'Númeração': '',
        r'\d{2}/\d{2}/\d{4}': '',
        r'Data do Julgamento:': '',
        r'Data da Publicação:': '',
        r'Des.\(a\) [\w\s]+ Relator: Des.\(a\) [\w\s]+ Relator do Acordão:': '',
        r'APELANTE\(S\): [\w\s]+, [\w\s]+, [\w\s]+, [\w\s]+, [\w\s]+ - 1º APELANTE: [\w\s]+ - 2º APELANTE: [\w\s]+ E '
        r'OUTRO\(A\)\(S\)': '',
        r'APELADO\(A\)\(S\): [\w\s]+, [\w\s]+ E OUTRO\(A\)\(S\)': '',
        r'APELAÇÃO CÍVEL Nº \d+\.': '',
        r'Númeração Des.\(a\) [\w\s]+ Relator: Des.\(a\) [\w\s]+ Relator do Acordão:': '',
        r'- , [\w\s]+, [\w\s]+': '',
        r'\s+': '',
        r'Número do \d+': '',
        r'Númeração Des.*? Relator:.*? Relator do Acordão: \d{2}/\d{2}/\d{4} Data do Julgamento: \d{2}/\d{2}/\d{4} '
        r'Data da Publicação: ': '',
        r'\s+': ' ',
        r'- COMARCA DE [A-Z ]+': '',
        r'\b[A-Z][a-z]*\b': '',
    })
    train_patterns: Dict[str, str] = field(default_factory=lambda: {
        r'\b\w\b': '',
        r'[^\w\s.]': '',
        r'\b\d+\b': '',
        r'\s+': ' ',
    })
    input_patterns: Dict[str, str] = field(default_factory=lambda: {
        r'[^\w\s.]': '',
        r'\s+': ' ',
    })


class PurifyScraper:
    def __init__(self):
        self.__rp = RegexPatterns()

    @property
    def patterns(self) -> RegexPatterns:
        return self.__rp

    @staticmethod
    def clear_text_to_train(text: str, debug: bool = False) -> str:
        patterns = PurifyScraper().__rp.train_patterns

        new_text = PurifyScraper().__clear_tjmg_data(text)
        new_text = PurifyScraper().__iter_patterns(new_text, patterns).strip()

        if debug:
            PurifyScraper().__debug(text, new_text)
        return new_text

    @staticmethod
    def clear_input(text: str, debug: bool = False) -> str:
        patterns = PurifyScraper().__rp.input_patterns

        new_text = PurifyScraper().__clear_tjmg_data(text)
        new_text = PurifyScraper().__iter_patterns(new_text, patterns).strip()

        if debug:
            PurifyScraper().__debug(text, new_text)
        return new_text

    def normalize_text(self, text):
        normalized_text = unicodedata.normalize('NFD', text)
        return ''.join(c for c in normalized_text if unicodedata.category(c) != 'Mn').lower()

    def is_uppercase_majority(self, text):
        total_letters = sum(c.isalpha() for c in text)
        uppercase_letters = sum(c.isupper() for c in text)

        if total_letters == 0:
            return False
        return (uppercase_letters / total_letters) > 0.5

    def process_text(self, text):
        parts = re.split(r'(?<!\d)\.(?!\d)', text)
        cleaned_parts = [part for part in parts if not self.is_uppercase_majority(part)]

        return '.'.join(cleaned_parts)

    def clean_text(self, text):
        cleaned_text = self.normalize_text(self.process_text(text))
        return re.sub(r'\s+', ' ', cleaned_text)

    def __clear_tjmg_data(self, text: str) -> str:
        return self.__iter_patterns(text.strip(), self.__rp.tjmg_patterns)

    def __iter_patterns(self, text: str, patterns: Dict[str, str]) -> str:
        for k, v in patterns.items():
            text = re.sub(k, v, text)
        return text

    def __debug(self, text: str, new_text: str):
        print(f'Text: {text}')
        print(f'Clear Text: {new_text}')
        print('-' * 100)
