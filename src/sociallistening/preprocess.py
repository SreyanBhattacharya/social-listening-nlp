"""Text cleaning for tweets.

The labelled tweets were collected by searching for emoticons such as ":)" and ":(",
so the emoticon *is* the label. Leaving emoticons in the text lets a model score
almost perfectly without learning anything about language (label leakage).
`clean_text` therefore strips emoticons by default; `keep_emoticons=True` exists only
to demonstrate the leakage in the evaluation report.
"""

from __future__ import annotations

import html
import re

# Emoticons used to collect the corpus (from the corpus README), plus common variants.
_EMOTICONS = [
    ":-)", ":)", ";)", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)", "=)", ":}", ":^)",
    ":-D", ":D", "8-D", "8D", "x-D", "xD", "X-D", "XD", "=-D", "=D", "=-3", "=3",
    ":-))", ":'-)", ":')", ":*", ":^*", ">:P", ":-P", ":P", "X-P", "x-p", "xp", "XP",
    ":-p", ":p", "=p", ":-b", ":b", ">:)", ">;)", ">:-)", "<3",
    ":L", ":-/", ">:/", ":S", ">:[", ":@", ":-(", ":[", ":-||", "=L", ":<", ":-[",
    ":-<", "=\\", "=/", ">:(", ":(", ">.<", ":'-(", ":'(", ":\\", ":-c", ":c", ":{",
    ">:\\", ";(", ";-)", ":-d", ":d", ":((", ":))", ":-((", ":(((", ":)))",
]
# Longest first so ":-))" is removed before ":-)".
_EMOTICON_ALT = "|".join(re.escape(e) for e in sorted(set(_EMOTICONS), key=len, reverse=True))
# Only match emoticons that stand alone (start/end of text or next to whitespace),
# so words such as "xp" inside "expert" are untouched.
EMOTICON_RE = re.compile(rf"(?:(?<=\s)|^)(?:{_EMOTICON_ALT})+(?=\s|$)")
# Repeated mouth characters, e.g. ":))))" or ":((((", that the list does not cover.
REPEATED_EMOTICON_RE = re.compile(r"(?:(?<=\s)|^)[:;=][-']?[()\[\]DPp]{2,}(?=\s|$)")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
RETWEET_RE = re.compile(r"^RT\s+")
NON_TEXT_RE = re.compile(r"[^a-z0-9#'\s]")
WHITESPACE_RE = re.compile(r"\s+")


def remove_emoticons(text: str) -> str:
    text = EMOTICON_RE.sub(" ", text)
    return REPEATED_EMOTICON_RE.sub(" ", text)


def clean_text(text: str, keep_emoticons: bool = False) -> str:
    """Normalise a tweet for modelling.

    Steps: unescape HTML entities, drop the retweet prefix, URLs and @mentions
    (these are noise and can identify individuals), optionally drop emoticons,
    lowercase, turn "#Hashtag" into "hashtag", and remove remaining punctuation.
    """
    text = html.unescape(text)
    text = RETWEET_RE.sub("", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    if not keep_emoticons:
        text = remove_emoticons(text)
        text = text.lower()
        text = NON_TEXT_RE.sub(" ", text)
    else:
        text = text.lower()
    text = text.replace("#", "")
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_hashtags(text: str) -> list[str]:
    """Return lowercase hashtags without the '#'."""
    return [tag.lower() for tag in re.findall(r"#(\w+)", text)]
