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
| TF-IDF + logistic regression | 0.8057 | 0.7614 | | |
| Fine-tuned transformer | | | | |
| LLM, prompting | | | | |
| LLM, LoRA | | | | |

On the same commands, Russian is 4.4 points lower than English with TF-IDF; the next experiments check whether character n-grams or lemmatization close the gap.

## How to run

```bash
pip install -e ".[dev]"
pytest
python -m intents.baseline --lang en
python -m intents.baseline --lang ru
python -m intents.baseline --model most_frequent --lang en
python -m intents.baseline --model most_frequent --lang ru
```

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
