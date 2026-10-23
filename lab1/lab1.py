from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final, Iterator

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pymorphy3
from pymorphy3.analyzer import Parse


CYRILLIC_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[а-яё]+(?:-[а-яё]+)?$", re.IGNORECASE
)


def ensure_nltk_resources() -> None:
    """Гарантирует загрузку необходимых словарей и моделей NLTK"""
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


@dataclass(frozen=True, slots=True)
class WordPair:
    """Структура для хранения согласованной пары слов"""

    first_original: str
    second_original: str
    first_lemma: str
    second_lemma: str

    @property
    def original_text(self) -> str:
        """Исходное словосочетание"""
        return f"{self.first_original} {self.second_original}"

    @property
    def lemmatized_text(self) -> str:
        """Леммы"""
        return f"{self.first_lemma} {self.second_lemma}"


class AgreementExtractor:
    """Анализатор текста для извлечения согласованных пар слов"""

    DEFAULT_TARGET_POS: Final[set[str]] = {"NOUN", "ADJF"}
    DEFAULT_MIN_SCORE: Final[float] = 0.05

    def __init__(
        self,
        target_pos: set[str] | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        exclude_pronouns: bool = True,
    ) -> None:
        self.morph = pymorphy3.MorphAnalyzer()
        self.target_pos = target_pos or self.DEFAULT_TARGET_POS
        self.min_score = min_score
        self.exclude_pronouns = exclude_pronouns

    def is_word(self, token: str) -> bool:
        """Проверяет, является ли токен кириллическим словом"""
        return bool(CYRILLIC_WORD_PATTERN.fullmatch(token))

    def check_agreement(self, parse1: Parse, parse2: Parse) -> bool:
        """Проверяет грамматическое согласование двух вариантов разбора"""
        t1, t2 = parse1.tag, parse2.tag

        if not t1.number or not t2.number or t1.number != t2.number:
            return False

        if not t1.case or not t2.case or t1.case != t2.case:
            return False

        if t1.number == "sing":
            if not t1.gender or not t2.gender or t1.gender != t2.gender:
                return False

        return True

    def _get_valid_parses(self, word: str) -> list[Parse]:
        """Возвращает отфильтрованные гипотезы морфологического анализа"""
        valid_parses = []
        for p in self.morph.parse(word):
            if p.tag.POS not in self.target_pos or p.score < self.min_score:
                continue
            if self.exclude_pronouns and "Apro" in p.tag:
                continue
            valid_parses.append(p)
        return valid_parses

    def extract_pairs(self, text: str) -> Iterator[WordPair]:
        """Генератор согласованных пар слов из переданного текста"""
        sentences = sent_tokenize(text, language="russian")

        for sentence in sentences:
            tokens = word_tokenize(sentence, language="russian")

            for i in range(len(tokens) - 1):
                t1, t2 = tokens[i], tokens[i + 1]

                if not (self.is_word(t1) and self.is_word(t2)):
                    continue

                parses1 = self._get_valid_parses(t1)
                parses2 = self._get_valid_parses(t2)

                if not parses1 or not parses2:
                    continue

                found = False
                for p1 in parses1:
                    for p2 in parses2:
                        if self.check_agreement(p1, p2):
                            yield WordPair(
                                first_original=t1,
                                second_original=t2,
                                first_lemma=p1.normal_form,
                                second_lemma=p2.normal_form,
                            )
                            found = True
                            break
                    if found:
                        break


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Поиск согласованных пар существительных и прилагательных"
    )
    default_file = Path(__file__).resolve().parent / "input.txt"

    parser.add_argument(
        "file",
        nargs="?",
        default=default_file,
        type=Path,
        help="Путь к текстовому файлу для обработки (по умолчанию: input.txt)",
    )
    return parser.parse_args()


def main() -> None:
    ensure_nltk_resources()
    args = parse_arguments()
    file_path: Path = args.file

    if not file_path.exists():
        print(f"Ошибка: файл '{file_path}' не существует")
        raise SystemExit(1)

    text = file_path.read_text(encoding="utf-8")
    extractor = AgreementExtractor()

    pairs = list(extractor.extract_pairs(text))

    print(f"Обработан файл: {file_path}")
    print(f"Найдено согласованных пар: {len(pairs)}\n")

    if not pairs:
        print("Подходящие пары не обнаружены")
        return

    col1_width = 35
    col2_width = 35
    print(f"{'Исходное сочетание':<{col1_width}} | {'Лемматизированная форма':<{col2_width}}")
    print(f"{'-' * col1_width}-+-{'-' * col2_width}")

    for pair in pairs:
        print(f"{pair.original_text:<{col1_width}} | {pair.lemmatized_text:<{col2_width}}")


if __name__ == "__main__":
    main()