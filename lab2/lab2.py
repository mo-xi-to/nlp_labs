from __future__ import annotations

import re
import argparse
from pathlib import Path
from typing import Final
from dataclasses import dataclass

from gensim.models import KeyedVectors


NOUN_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(.*)_NOUN$")


@dataclass(frozen=True, slots=True)
class NeighborWord:
    """Результат вычисления векторного сходства"""

    lemma: str
    similarity: float


class VectorSemanticAnalyzer:
    """Анализатор векторной модели для поиска семантических окрестностей"""

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            print(f"Ошибка: файл модели '{self.model_path}' не найден")
            raise SystemExit(1)

        print(f"Загрузка модели '{self.model_path.name}' ...", flush=True)
        self.vectors = KeyedVectors.load_word2vec_format(str(self.model_path), binary=False)
        print("Модель успешно загружена\n")

    def get_nearest_nouns(
        self,
        positive: list[str],
        negative: list[str] | None = None,
        top_k: int = 10,
    ) -> list[NeighborWord]:
        """Возвращает список ближайших существительных к линейной комбинации"""
        raw_similar = self.vectors.most_similar(
            positive=positive,
            negative=negative or [],
            topn=top_k * 4,
        )

        nouns: list[NeighborWord] = []
        for word, score in raw_similar:
            match = NOUN_PATTERN.match(word)
            if match:
                nouns.append(NeighborWord(lemma=match.group(1), similarity=float(score)))
                if len(nouns) == top_k:
                    break

        return nouns


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Вычисление семантической окрестности линейной комбинации векторов слов"
    )
    default_model = Path(__file__).resolve().parent / "cbow.txt"

    parser.add_argument(
        "model",
        nargs="?",
        default=default_model,
        type=Path,
        help="Путь к файлу модели (по умолчанию: cbow.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    analyzer = VectorSemanticAnalyzer(model_path=args.model)

    pos = ["таксист_NOUN", "микроавтобус_NOUN"]
    neg = ["человек_NOUN"]

    results = analyzer.get_nearest_nouns(positive=pos, negative=neg, top_k=10)

    print(f"Положительные компоненты: {pos}")
    print(f"Отрицательные компоненты: {neg}\n")

    col1_width = 4
    col2_width = 20
    col3_width = 18

    print(f"{'№':<{col1_width}} | {'Существительное':<{col2_width}} | {'Косинусное сходство':<{col3_width}}")
    print(f"{'-' * col1_width}-+-{'-' * col2_width}-+-{'-' * col3_width}")

    for idx, item in enumerate(results, start=1):
        print(f"{idx:<{col1_width}} | {item.lemma:<{col2_width}} | {item.similarity:<{col3_width}.4f}")


if __name__ == "__main__":
    main()