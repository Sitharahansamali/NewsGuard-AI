from newsguard.features.preprocessing import clean_text


def test_clean_text_lowercase():
    result = clean_text("HELLO WORLD")

    assert result == "hello world"


def test_clean_text_removes_url():
    result = clean_text(
        "Visit https://example.com for news"
    )

    assert "https" not in result
    assert "example" not in result


def test_clean_text_returns_string():
    result = clean_text("This is a test.")

    assert isinstance(result, str)