"""Loading MASSIVE, the intent-classification dataset, for one language."""

from datasets import load_dataset

DATASET = "mteb/amazon_massive_intent"
LANGUAGES = ("en", "ru")
SPLITS = ("train", "validation", "test")


def load_split(lang: str, split: str) -> tuple[list[str], list[str]]:
    """Return the texts and the intent labels of one split of one language."""
    if lang not in LANGUAGES:
        raise ValueError(f"lang must be one of {LANGUAGES}, got {lang!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    dataset = load_dataset(DATASET, lang, split=split)
    return list(dataset["text"]), list(dataset["label"])
