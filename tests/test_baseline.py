"""Tests for the baseline. Run them from the project root with `pytest`."""

from intents.baseline import build_most_frequent, build_pipeline, evaluate

# A tiny made-up dataset: two intents, three phrases each. No downloads, runs in a second.
TEXTS = [
    "wake me up at seven",
    "set an alarm for six",
    "alarm at nine please",
    "what is the weather today",
    "will it rain tomorrow",
    "weather in warsaw",
]
LABELS = ["alarm_set"] * 3 + ["weather_query"] * 3

TEXTS_FREQUENT = [
    "wake me up at seven",
    "set an alarm for six",
    "pizza",
    "what is the weather today",
]
LABELS_FREQUENT = ["alarm_set"] * 3 + ["weather_query"] * 1


def test_pipeline_learns_a_tiny_dataset():
    model = build_pipeline()
    model.fit(TEXTS, LABELS)
    assert list(model.predict(["set an alarm", "rain today"])) == [
        "alarm_set",
        "weather_query",
    ]


def test_dummy_pipeline_learns_a_tiny_dataset():
    model = build_most_frequent()
    model.fit(TEXTS_FREQUENT, LABELS_FREQUENT)
    assert list(model.predict(["rain today", "pizza"])) == [
        "alarm_set",
        "alarm_set",
    ]


def test_evaluate():
    model = build_pipeline()
    model.fit(TEXTS, LABELS)
    assert (
        evaluate(
            model, ["set an alarm", "rain today"], ["weather_query", "weather_query"]
        )
        == 0.5
    )
