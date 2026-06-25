from src.utils.parse_keys import parse_keys


def test_parse_simple():
    text = 'openrouter:"sk-abc123def456" groq:gsk_abcdef123456 google=AIzaSyABCDEFGHIJKLMN'
    keys = parse_keys(text)
    assert keys["openrouter"] == "sk-abc123def456"
    assert keys["groq"] == "gsk_abcdef123456"
    assert keys["google"] == "AIzaSyABCDEFGHIJKLMN"


def test_parse_unquoted():
    text = "openrouter:sk-abcdefghijklmnop groq:gsk_1234567890"
    keys = parse_keys(text)
    assert keys["openrouter"] == "sk-abcdefghijklmnop"
    assert keys["groq"] == "gsk_1234567890"
