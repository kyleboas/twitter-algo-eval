# twitter-draft-scorer

Train a local engagement evaluator on your own X/Twitter archive and score draft posts against your historical performance.

X has open-sourced parts of its recommendation/ranking algorithm, which makes its broad engagement incentives easier to inspect. This project uses that context plus your own archive data to build a local calibration tool: given your past posts, likes, retweets, wording, and send times, it estimates how a draft compares with your own history.

This is not a full X algorithm clone and does not predict impressions or true reach.

## What it does

- Reads an exported X/Twitter archive zip.
- Trains local models for likes and retweets.
- Scores engagement as `likes + 2 * retweets`.
- Reports a predicted score and historical percentile.
- Uses transparent features such as wording, length, questions, numbers, proper names, topic/mechanism terms, hour, and weekday.
- Is designed with awareness that X's recommendation code is public, while still training on your personal archive rather than claiming platform-level reach prediction.
- Runs locally; no X API required.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
x-archive-engagement-eval \
  --archive /path/to/twitter-archive.zip \
  --title "I built a local eval from my entire Twitter archive." \
  --lede "It estimates likely likes and retweets from wording and send time, then compares drafts against historical patterns."
```

JSON output:

```bash
x-archive-engagement-eval --archive /path/to/archive.zip --title "Draft text" --json
```

Try a different send time:

```bash
x-archive-engagement-eval --archive /path/to/archive.zip --title "Draft text" --hour 17 --weekday 3
```

`weekday` uses Python's convention: `0=Mon ... 6=Sun`.

## Output

Example fields:

- `predicted_likes`
- `predicted_retweets`
- `predicted_engagement_score`
- `historical_percentile`
- `features`

The percentile is usually the most useful number. It answers: “relative to my own past posts, where would this draft land?”

## Limitations

- Archive counts are snapshots, not mature lifetime performance for every post.
- The model cannot see impressions, follower graph distribution, media quality, replies, quote context, current events, or platform-side ranking state.
- It can overfit personal history.
- It should be used to compare drafts, not to claim guaranteed performance.
- Avoid using it to create clickbait or engagement-bait; the default features flag some banned hook terms.

## Development

```bash
python -m unittest discover -s tests
```

## License

MIT
