from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r"https?://\S+")
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’-]{2,}")

DEFAULT_TOPIC_TERMS = {
    "analysis", "article", "audience", "because", "built", "draft", "edit",
    "model", "post", "publish", "score", "share", "thread", "time", "word",
}
MECHANISM_TERMS = {
    "adjust", "because", "before", "depends", "helps", "if", "lets",
    "requires", "separates", "starts", "test", "turns", "uses", "when",
    "while", "without",
}
BANNED_HOOK_TERMS = {
    "shocking", "secret", "hidden", "disaster", "embarrassing", "must-read",
    "viral", "algorithm", "engagement", "you won't believe",
}


@dataclass
class TweetRow:
    tweet_id: str
    created_at: dt.datetime
    text: str
    likes: int
    retweets: int
    replies: int
    quotes: int
    lang: str

    @property
    def engagement_score(self) -> int:
        return self.likes + 2 * self.retweets

    @property
    def model_title(self) -> str:
        clean = clean_tweet_text(self.text)
        if ". " in clean[:140]:
            return clean.split(". ", 1)[0].strip()
        return clean[:100].strip()

    @property
    def model_lede(self) -> str:
        return clean_tweet_text(self.text)


def parse_js_assignment(raw: str) -> Any:
    return json.loads(re.sub(r"^window\.[^=]+ = ", "", raw, count=1).strip())


def read_archive_json(archive: Path, member: str) -> Any:
    with zipfile.ZipFile(archive) as zf:
        return parse_js_assignment(zf.read(member).decode("utf-8"))


def parse_twitter_datetime(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")


def archive_generation_date(archive: Path) -> dt.datetime | None:
    try:
        manifest = read_archive_json(archive, "data/manifest.js")
        value = manifest.get("archiveInfo", {}).get("generationDate")
        if value:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    return None


def clean_tweet_text(text: str) -> str:
    text = URL_RE.sub("", text or "")
    return re.sub(r"\s+", " ", text).strip()


def parse_tweets(archive: Path) -> list[TweetRow]:
    raw = read_archive_json(archive, "data/tweets.js")
    rows: list[TweetRow] = []
    for item in raw:
        tw = item.get("tweet", {})
        try:
            created_at = parse_twitter_datetime(tw.get("created_at"))
        except Exception:
            continue
        rows.append(TweetRow(
            tweet_id=str(tw.get("id_str") or tw.get("id") or ""),
            created_at=created_at,
            text=tw.get("full_text") or "",
            likes=int(tw.get("favorite_count") or 0),
            retweets=int(tw.get("retweet_count") or 0),
            replies=int(tw.get("reply_count") or 0),
            quotes=int(tw.get("quote_count") or 0),
            lang=tw.get("lang") or "",
        ))
    return rows


def text_features(title: str, lede: str, *, hour: int = 12, weekday: int = 2) -> dict[str, float | int | str]:
    title = re.sub(r"\s+", " ", title or "").strip()
    lede = re.sub(r"\s+", " ", lede or "").strip()
    combined = f"{title} {lede}".strip()
    words = [w.lower() for w in WORD_RE.findall(combined)]
    topic_hits = sum(1 for word in words if any(term in word for term in DEFAULT_TOPIC_TERMS))
    mechanism_hits = sum(1 for word in words if word in MECHANISM_TERMS)
    banned_hits = sum(1 for term in BANNED_HOOK_TERMS if term in combined.lower())
    return {
        "title_chars": len(title),
        "lede_chars": len(lede),
        "title_words": len(WORD_RE.findall(title)),
        "lede_words": len(WORD_RE.findall(lede)),
        "has_question": int("?" in title or "?" in lede),
        "has_number": int(bool(re.search(r"\d", combined))),
        "proper_name_pairs": len(re.findall(r"\b[A-Z][a-zÀ-ÿ]+\s+[A-Z][a-zÀ-ÿ]+\b", combined)),
        "topic_hits": topic_hits,
        "mechanism_hits": mechanism_hits,
        "banned_hook_hits": banned_hits,
        "hour": int(hour),
        "weekday": int(weekday),
        "hour_bucket": f"h{int(hour) // 4}",
        "weekday_name": str(int(weekday)),
    }


class EngagementEval:
    def __init__(self) -> None:
        self.tfidf = TfidfVectorizer(min_df=2, max_features=6000, ngram_range=(1, 2), sublinear_tf=True)
        self.dictvec = DictVectorizer(sparse=True)
        self.likes_model = Ridge(alpha=2.0)
        self.retweets_model = Ridge(alpha=2.0)
        self.historical_scores: np.ndarray | None = None

    def _matrix(self, texts: list[str], dicts: list[dict[str, Any]], *, fit: bool = False):
        text_x = self.tfidf.fit_transform(texts) if fit else self.tfidf.transform(texts)
        dict_x = self.dictvec.fit_transform(dicts) if fit else self.dictvec.transform(dicts)
        return hstack([text_x, dict_x], format="csr")

    def fit(self, rows: list[TweetRow]) -> None:
        texts = [f"{r.model_title}\n{r.model_lede}" for r in rows]
        dicts = [text_features(r.model_title, r.model_lede, hour=r.created_at.hour, weekday=r.created_at.weekday()) for r in rows]
        x = self._matrix(texts, dicts, fit=True)
        self.likes_model.fit(x, np.log1p([r.likes for r in rows]))
        self.retweets_model.fit(x, np.log1p([r.retweets for r in rows]))
        self.historical_scores = np.array([r.engagement_score for r in rows], dtype=float)

    def predict(self, title: str, lede: str, *, hour: int = 12, weekday: int = 2) -> dict[str, Any]:
        x = self._matrix([f"{title}\n{lede}"], [text_features(title, lede, hour=hour, weekday=weekday)], fit=False)
        pred_likes = max(0.0, math.expm1(float(self.likes_model.predict(x)[0])))
        pred_retweets = max(0.0, math.expm1(float(self.retweets_model.predict(x)[0])))
        pred_score = pred_likes + 2 * pred_retweets
        hist = self.historical_scores if self.historical_scores is not None else np.array([])
        percentile = float((hist <= pred_score).sum() / len(hist) * 100.0) if len(hist) else 0.0
        return {
            "predicted_likes": round(pred_likes, 1),
            "predicted_retweets": round(pred_retweets, 1),
            "predicted_engagement_score": round(pred_score, 1),
            "historical_percentile": round(percentile, 1),
            "features": text_features(title, lede, hour=hour, weekday=weekday),
        }


def eligible_rows(rows: list[TweetRow], generated_at: dt.datetime | None, *, min_age_days: int) -> list[TweetRow]:
    out = []
    for row in rows:
        if row.lang and row.lang != "en":
            continue
        if generated_at is not None and (generated_at - row.created_at).days < min_age_days:
            continue
        if len(row.model_lede) < 40:
            continue
        out.append(row)
    return out


def percentile_summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    return {k: round(float(np.quantile(values, q)), 1) for k, q in {"p50": .5, "p75": .75, "p90": .9, "p95": .95}.items()} | {"max": round(float(max(values)), 1)}


def evaluate_model(rows: list[TweetRow]) -> dict[str, Any]:
    if len(rows) < 50:
        return {"warning": "not enough rows for holdout evaluation", "eligible_rows": len(rows)}
    train, test = train_test_split(rows, test_size=0.25, random_state=42)
    model = EngagementEval(); model.fit(train)
    preds = [model.predict(r.model_title, r.model_lede, hour=r.created_at.hour, weekday=r.created_at.weekday()) for r in test]
    scores = np.array([r.engagement_score for r in rows], dtype=float)
    cutoff = float(np.quantile(scores, 0.90))
    y_top = np.array([r.engagement_score >= cutoff for r in test], dtype=int)
    pred_score = np.array([p["predicted_engagement_score"] for p in preds], dtype=float)
    auc = float(roc_auc_score(y_top, pred_score)) if len(set(y_top)) > 1 else None
    return {
        "eligible_rows": len(rows), "train_rows": len(train), "test_rows": len(test),
        "likes_mae": round(float(mean_absolute_error([r.likes for r in test], [p["predicted_likes"] for p in preds])), 2),
        "retweets_mae": round(float(mean_absolute_error([r.retweets for r in test], [p["predicted_retweets"] for p in preds])), 2),
        "top_decile_cutoff_score": round(cutoff, 1),
        "top_decile_auc": round(auc, 3) if auc is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a local X/Twitter archive engagement eval and score a draft")
    parser.add_argument("--archive", type=Path, required=True, help="Path to your X/Twitter archive zip")
    parser.add_argument("--min-age-days", type=int, default=7)
    parser.add_argument("--title", default="", help="Draft title/opening line")
    parser.add_argument("--lede", default="", help="Draft body/continuation; defaults to --title")
    parser.add_argument("--hour", type=int, default=12)
    parser.add_argument("--weekday", type=int, default=2, help="0=Mon ... 6=Sun")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rows = eligible_rows(parse_tweets(args.archive), archive_generation_date(args.archive), min_age_days=args.min_age_days)
    if len(rows) < 20:
        raise SystemExit(f"Not enough eligible rows after filters: {len(rows)}")
    model = EngagementEval(); model.fit(rows)
    result: dict[str, Any] = {
        "calibration": {
            "archive": str(args.archive),
            "eligible_training_rows": len(rows),
            "engagement_score_definition": "favorite_count + 2 * retweet_count",
            "engagement_score_percentiles": percentile_summary([r.engagement_score for r in rows]),
            "evaluation": evaluate_model(rows),
        }
    }
    if args.title or args.lede:
        title = args.title or args.lede
        lede = args.lede or args.title
        result["candidate"] = model.predict(title, lede, hour=args.hour, weekday=args.weekday) | {"title": title, "lede": lede}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        cal = result["calibration"]
        print("X archive engagement eval")
        print(f"training_rows={cal['eligible_training_rows']}")
        print(f"engagement_score_percentiles={cal['engagement_score_percentiles']}")
        print(f"evaluation={cal['evaluation']}")
        if "candidate" in result:
            c = result["candidate"]
            print(f"candidate likes={c['predicted_likes']} retweets={c['predicted_retweets']} score={c['predicted_engagement_score']} percentile={c['historical_percentile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
