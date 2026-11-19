from __future__ import annotations

import os
from pathlib import Path

os.environ["HF_HOME"] = str(Path(__file__).resolve().parent / ".cache")

import argparse
from dataclasses import dataclass
from typing import Final

import torch
from torch.nn import functional as F
from transformers import BertForMaskedLM, BertTokenizer


@dataclass(frozen=True, slots=True)
class MaskPrediction:
    """Результат предсказания токена на месте маски"""

    token: str
    probability: float


class MaskedLanguageModel:
    """Обертка над моделью BERT для решения задачи Masked Language Modeling"""

    DEFAULT_MODEL_NAME: Final[str] = "bert-base-multilingual-uncased"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        print(f"Инициализация модели '{model_name}'...", flush=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name, return_dict=True)
        self.model.eval()
        print("Модель успешно загружена.\n")

    @property
    def mask_token(self) -> str:
        """Специальный токен маскирования"""
        return self.tokenizer.mask_token

    def predict_mask(self, text: str, top_k: int = 10) -> list[MaskPrediction]:
        """Возвращает top-k наиболее вероятных слов для позиции маски"""
        inputs = self.tokenizer._encode_plus(text, return_tensors="pt")
        mask_indices = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)

        if len(mask_indices[0]) == 0:
            raise ValueError(f"В предложении отсутствует токен маски '{self.mask_token}'")

        mask_idx = mask_indices[0][0].item()

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        mask_probs = probabilities[0, mask_idx, :]

        top = torch.topk(mask_probs, top_k)
        predictions: list[MaskPrediction] = []

        for prob, token_id in zip(top.values.tolist(), top.indices.tolist()):
            word = self.tokenizer.decode([token_id]).strip()
            predictions.append(MaskPrediction(token=word, probability=float(prob)))

        return predictions


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Восстановление пропущенного слова с помощью модели BERT (Masked LM)"
    )
    parser.add_argument(
        "--model",
        default=MaskedLanguageModel.DEFAULT_MODEL_NAME,
        type=str,
        help="Идентификатор модели BERT",
    )
    parser.add_argument(
        "--top-k",
        default=10,
        type=int,
        help="Количество выводимых вариантов",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    mlm = MaskedLanguageModel(model_name=args.model)

    target_words: Final[set[str]] = {"технических", "научных"}

    sentence = f"Для решения сложных {mlm.mask_token} задач."

    predictions = mlm.predict_mask(sentence, top_k=args.top_k)

    print(f"Тестируемое предложение:\n{sentence}\n")

    col1_width = 4
    col2_width = 20
    col3_width = 18

    print(f"{'№':<{col1_width}} | {'Словоформа':<{col2_width}} | {'Вероятность':<{col3_width}}")
    print(f"{'-' * col1_width}-+-{'-' * col2_width}-+-{'-' * col3_width}")

    found_targets: set[str] = set()
    for idx, item in enumerate(predictions, start=1):
        if item.token in target_words:
            found_targets.add(item.token)
        print(f"{idx:<{col1_width}} | {item.token:<{col2_width}} | {item.probability:<{col3_width}.4f}")

    print()
    if target_words.issubset(found_targets):
        print("Результат: Оба целевых слова вошли в первую десятку.")
    else:
        missing = target_words - found_targets
        print(f"Результат: Слова {missing} не вошли в топ-10.")


if __name__ == "__main__":
    main()