import datetime as dt
import unittest

from x_archive_engagement_eval import cli


class EngagementEvalTests(unittest.TestCase):
    def test_clean_tweet_text_removes_urls(self):
        self.assertEqual(cli.clean_tweet_text("Hello https://example.com world"), "Hello world")

    def test_text_features(self):
        features = cli.text_features(
            "I built a local model before posting",
            "It scores words and time because old posts show patterns.",
        )
        self.assertGreaterEqual(features["topic_hits"], 2)
        self.assertGreaterEqual(features["mechanism_hits"], 1)
        self.assertEqual(features["banned_hook_hits"], 0)

    def test_eligible_rows_filters_recent_and_non_english(self):
        generated = dt.datetime(2026, 6, 6, tzinfo=dt.timezone.utc)
        old_en = cli.TweetRow("1", generated - dt.timedelta(days=10), "Old English tweet with enough words to pass the minimum text filter", 1, 0, 0, 0, "en")
        new_en = cli.TweetRow("2", generated - dt.timedelta(days=1), "New English tweet with enough words to pass the minimum text filter", 1, 0, 0, 0, "en")
        old_es = cli.TweetRow("3", generated - dt.timedelta(days=10), "Texto con suficientes palabras para superar el filtro", 1, 0, 0, 0, "es")
        self.assertEqual(cli.eligible_rows([old_en, new_en, old_es], generated, min_age_days=7), [old_en])


if __name__ == "__main__":
    unittest.main()
