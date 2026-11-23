from sociallistening.preprocess import clean_text, extract_hashtags


def test_removes_urls_mentions_and_retweet_prefix():
    text = "RT @someone: Read this https://t.co/abc123 now"
    assert clean_text(text) == "read this now"


def test_strips_emoticons_to_prevent_label_leakage():
    assert clean_text("great day :) :-D") == "great day"
    assert clean_text("awful :( :'(") == "awful"
    assert clean_text("so happy :)))") == "so happy"


def test_keeps_words_that_contain_emoticon_letters():
    assert clean_text("an expert xp boost") == "an expert boost"
    assert "expert" in clean_text("expert")


def test_keep_emoticons_flag_preserves_them():
    assert ":)" in clean_text("great day :)", keep_emoticons=True)


def test_hashtags_become_words_and_html_is_unescaped():
    assert clean_text("#Election2015 rock &amp; roll") == "election2015 rock roll"


def test_extract_hashtags():
    assert extract_hashtags("Watching #BBCQT and #GE2015 tonight") == ["bbcqt", "ge2015"]
