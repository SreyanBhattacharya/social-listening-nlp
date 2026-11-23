"""Turn the topic report into a short written insight brief.

Run:  python -m sociallistening.insights [--provider gemini|claude]

Without a provider (the default) the brief is built from a template, so the project
runs with no API key. With --provider, the topic summary is sent to an LLM, which is
asked to write the brief using only the numbers it was given.

LLM setup (optional):
  gemini: pip install google-genai;  set GEMINI_API_KEY and LLM_MODEL
  claude: pip install anthropic;     set ANTHROPIC_API_KEY and LLM_MODEL
LLM_MODEL is the model name from the provider's documentation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROMPT_TEMPLATE = """You are a social listening analyst. Below is a JSON summary of topics
found in {n} tweets about the 2015 UK general election, collected between {start} and
{end} UTC. Write a brief for a non-technical stakeholder:
- 3 to 5 bullet points on the main conversation themes and what was trending
- one bullet on limitations of the analysis
Use only facts and numbers present in the JSON. Do not guess at causes you cannot see
in the data, and do not take a political side. Note that the sentiment shares come
from a model trained on non-political tweets, so they are indicative only.

JSON summary:
{summary}
"""


def build_prompt(summary: dict) -> str:
    compact = {
        "topics": [
            {k: t[k] for k in ("label", "top_terms", "tweets", "share_of_tweets",
                               "retweet_share", "predicted_positive_share")}
            for t in summary["topics"]
        ],
        "top_hashtags": summary["top_hashtags"][:10],
        "retweet_share": summary["retweet_share"],
    }
    start, end = summary["time_range_utc"]
    return PROMPT_TEMPLATE.format(
        n=summary["tweets_analysed"], start=start, end=end, summary=json.dumps(compact, indent=1)
    )


def template_brief(summary: dict) -> str:
    topics = summary["topics"]
    lead = topics[0]
    lines = [
        "# Social listening brief (template)",
        "",
        f"- {summary['tweets_analysed']:,} English tweets analysed; "
        f"{summary['retweet_share']:.0%} were retweets.",
        f"- Largest topic: **{lead['label']}** with {lead['tweets']:,} tweets "
        f"({lead['share_of_tweets']:.0%} of the conversation).",
        "- Other major topics: " + "; ".join(f"{t['label']} ({t['tweets']:,})" for t in topics[1:4]) + ".",
        "- Top hashtags: " + ", ".join(f"#{tag} ({n:,})" for tag, n in summary["top_hashtags"][:5]) + ".",
        "- Limitation: about 70 minutes of data on one evening, and sentiment comes from a "
        "model trained on non-political tweets, so tone estimates are indicative only.",
    ]
    return "\n".join(lines) + "\n"


def llm_brief(prompt: str, provider: str) -> str:
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("Set LLM_MODEL to the model name you want to use.")
    if provider == "gemini":
        from google import genai

        client = genai.Client()  # reads GEMINI_API_KEY
        return client.models.generate_content(model=model, contents=prompt).text
    if provider == "claude":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        message = client.messages.create(
            model=model, max_tokens=800, messages=[{"role": "user", "content": prompt}]
        )
        return "".join(block.text for block in message.content if block.type == "text")
    raise ValueError(f"Unknown provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--provider", choices=["gemini", "claude"], default=None)
    args = parser.parse_args()

    summary = json.loads((args.reports_dir / "topics.json").read_text())
    if args.provider:
        brief = "# Social listening brief (LLM)\n\n" + llm_brief(build_prompt(summary), args.provider)
    else:
        brief = template_brief(summary)
    out = args.reports_dir / "insights.md"
    out.write_text(brief)
    print(brief)


if __name__ == "__main__":
    main()
