import argparse

import mlflow
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from intents.baseline import build_pipeline
from intents.data import LANGUAGES, load_split


def main():
    parser = argparse.ArgumentParser(
        description="Tuning a model on MASSIVE: TF-IDF + logistic regression"
    )
    parser.add_argument("--lang", choices=LANGUAGES, default="en")
    args = parser.parse_args()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    texts, labels = load_split(args.lang, "train")
    texts, labels = np.array(texts), np.array(labels)
    model = build_pipeline(analyzer="char_wb", ngram_min=2, ngram_max=5)
    param_grid = {"logisticregression__C": [1, 3, 10, 30, 100]}
    search = GridSearchCV(model, param_grid, cv=skf, scoring="accuracy", n_jobs=-1)

    mlflow.sklearn.autolog(log_models=False)
    mlflow.set_experiment("massive-stage1")
    with mlflow.start_run():
        mlflow.log_param("lang", args.lang)
        search.fit(texts, labels)

        print(search.best_params_)
        print(search.best_score_)
        print(
            search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"]
        )


if __name__ == "__main__":
    main()
