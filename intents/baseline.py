"""Stage 1: the two baselines — always the most frequent intent, and TF-IDF + logistic regression.

Run from the project root:
    python -m intents.baseline --lang en
    python -m intents.baseline --lang ru
    python -m intents.baseline --model most_frequent --lang en
    python -m intents.baseline --model most_frequent --lang ru
"""

import argparse

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import dummy
from intents.data import LANGUAGES, load_split


def build_most_frequent() -> dummy.DummyClassifier:
    """The null model: always the most frequent intent of the training set."""
    return dummy.DummyClassifier(strategy="most_frequent")


def build_pipeline() -> Pipeline:
    """An untrained pipeline: TF-IDF features, then logistic regression."""
    return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))


def evaluate(
    model: Pipeline | dummy.DummyClassifier, texts: list[str], labels: list[str]
) -> float:
    """Accuracy of a trained model on the given texts, from 0 to 1."""
    return accuracy_score(labels, model.predict(texts))


def main() -> None:
    """Train on the train split and print the accuracy on validation or test."""
    parser = argparse.ArgumentParser(
        description="Train a baseline on MASSIVE: TF-IDF + logistic regression, or always the most frequent intent."
    )
    parser.add_argument("--lang", choices=LANGUAGES, default="en")
    parser.add_argument("--model", choices=("tfidf", "most_frequent"), default="tfidf")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="where to measure; use test only once, for the final number",
    )
    args = parser.parse_args()

    train_texts, train_labels = load_split(args.lang, "train")
    eval_texts, eval_labels = load_split(args.lang, args.split)

    if args.model == "most_frequent":
        model = build_most_frequent()
    else:
        model = build_pipeline()
    model.fit(train_texts, train_labels)
    accuracy = evaluate(model, eval_texts, eval_labels)
    print(
        f"{args.lang} {args.split} {args.model}: accuracy {accuracy:.4f} on {len(eval_texts)} examples"
    )


if __name__ == "__main__":
    main()
