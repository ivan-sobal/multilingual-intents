# Multilingual intent classification

One task — recognising what a user wants from a short command to a voice assistant — solved three ways, in English and in Russian, with the same data and the same metric:

1. TF-IDF + logistic regression (the baseline)
2. a fine-tuned transformer
3. an LLM: prompting, and fine-tuning with LoRA

The question behind it: how much does each method gain over the simple one, and how much harder is Russian than English for each of them?

**Status:** work in progress, stage 1.

## Data

[MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) (Amazon, 2022): short commands to a voice assistant in 51 languages. The English commands were localized into the other languages by professional translators, so the languages are parallel — the same commands with the same labels.

This project uses the Hugging Face copy [`mteb/amazon_massive_intent`](https://huggingface.co/datasets/mteb/amazon_massive_intent), configs `en` and `ru`:

| | Train | Validation | Test |
|---|---|---|---|
| Commands per language | 11,514 | 2,033 | 2,974 |

- 60 intents, for example `alarm_set`, `weather_query`, `music_likeness`.
- Example: "wake me up at nine am on friday" / «разбуди меня в девять утра в пятницу» → `alarm_set`.
- License: CC BY 4.0.
- Caveat: the Russian commands are translations, so some of them don't sound the way native speakers talk to an assistant.

## Method

- The metric is accuracy, the standard one for MASSIVE.
- Every choice (features, hyperparameters, prompts) is made on validation. Test is used once per method, for the final table.
- The same splits for every method.

## Results

Accuracy; empty cells are not measured yet.

| Method | en, validation | ru, validation | en, test | ru, test |
|---|---|---|---|---|
| Always the most frequent intent | 0.0644 | 0.0644 | | |
| TF-IDF + logistic regression (tuned, see below) | 0.8515 | 0.8303 | | |
| Fine-tuned transformer | | | | |
| LLM, prompting | | | | |
| LLM, LoRA | | | | |

On the same commands, Russian was 4.4 points behind English with the default TF-IDF; after tuning it is 2.1 points behind. Character pieces closed half of the gap. The test cells stay empty until `C` is chosen with cross-validation.

## Stage 1 experiments (validation)

One change at a time, each run logged in MLflow (experiment `massive-stage1`).

| Features | `C` | en | ru |
|---|---|---|---|
| Words | 1 | 0.8057 | 0.7614 |
| Words + bigrams | 1 | 0.7973 | 0.7413 |
| Words | 30 | 0.8485 | 0.8165 |
| Words + bigrams | 30 | 0.8451 | |
| Words, lemmatized (pymorphy3) | 30 | | 0.8210 |
| Character pieces `char_wb (2, 5)` | 30 | **0.8515** | **0.8303** |
| Character pieces, lemmatized | 30 | | 0.8254 |

- The penalty was the biggest lever: `C = 1` underfits; accuracy peaks near `C = 30` (tried 0.3, 1, 3, 10, 30, 100).
- Bigrams don't help: many rare columns, and at the best `C` they tie with single words.
- Character pieces help Russian (+1.4 at `C = 30`): the forms of one word share most of their pieces. In English it's a tie.
- Lemmatization helps words a little but hurts character pieces: it erases endings that separate intents ("поставь", a command, against "поставлен ли", a question).
- Differences under about 0.5 points (10 of 2,033 commands) are treated as noise.

## How to run

```bash
pip install -e ".[dev]"
pytest
python -m intents.baseline --lang en
python -m intents.baseline --lang ru
python -m intents.baseline --model most_frequent --lang en
python -m intents.baseline --model most_frequent --lang ru
```

The best settings, and the options for experiments:

```bash
python -m intents.baseline --lang ru --analyzer char_wb --ngram-min 2 --ngram-max 5 --C 30
python -m intents.baseline --lang ru --C 30 --lemmatize
```

`--ngram-min`, `--ngram-max` and `--analyzer` (`word` or `char_wb`) set the TF-IDF features, `--C` the inverse strength of the L2 penalty, `--lemmatize` turns on pymorphy3 lemmas (for Russian). Every run is logged to MLflow; to see them, run `mlflow ui` and open "Model training" → "Runs".

The first run downloads the dataset (a few MB) into the Hugging Face cache.

Before a commit: `ruff check .` and `ruff format .`.

## Project structure

```
intents/
    data.py        loading one language and one split
    baseline.py    stage 1: TF-IDF + logistic regression
tests/
    test_baseline.py
```

## Roadmap

- [ ] Stage 1: TF-IDF + logistic regression, English and Russian; the most-frequent-intent baseline
- [ ] Stage 2: a fine-tuned transformer
- [ ] Stage 3: an LLM — prompting, then LoRA; the full comparison and what it means

## License

Code: [MIT](LICENSE). Data: MASSIVE, CC BY 4.0.
