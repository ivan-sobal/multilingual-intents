"""Stage 1: TF-IDF + logistic regression, the baseline.

Run from the project root:
    python -m intents.baseline --lang en
    python -m intents.baseline --lang ru
"""

import argparse

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from intents.data import LANGUAGES, load_split


def build_pipeline() -> Pipeline:
    """An untrained pipeline: TF-IDF features, then logistic regression."""
    return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))


def evaluate(model: Pipeline, texts: list[str], labels: list[str]) -> float:
    """Accuracy of a trained model on the given texts, from 0 to 1."""
    return accuracy_score(labels, model.predict(texts))


def main() -> None:
    """Train on the train split and print the accuracy on validation or test."""
    parser = argparse.ArgumentParser(
        description="Train the TF-IDF + logistic regression baseline on MASSIVE."
    )
    parser.add_argument("--lang", choices=LANGUAGES, default="en")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="where to measure; use test only once, for the final number",
    )
    args = parser.parse_args()

    train_texts, train_labels = load_split(args.lang, "train")
    eval_texts, eval_labels = load_split(args.lang, args.split)

    model = build_pipeline()
    model.fit(train_texts, train_labels)
    accuracy = evaluate(model, eval_texts, eval_labels)
    print(
        f"{args.lang} {args.split}: accuracy {accuracy:.4f} on {len(eval_texts)} examples"
    )


if __name__ == "__main__":
    main()
