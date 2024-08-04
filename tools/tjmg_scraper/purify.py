import re
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class RegexPatterns:
    tjmg_patterns: Dict[str, str] = field(default_factory=lambda: {
        r'': '',
    })
    train_patterns: Dict[str, str] = field(default_factory=lambda: {
        r'\s+': ' ',
    })
    input_patterns: Dict[str, str] = field(default_factory=lambda: {
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
