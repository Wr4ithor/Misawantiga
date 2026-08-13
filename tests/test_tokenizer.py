from ..tokenizer import BPETokenizer


tokenizer = BPETokenizer.load("data/datasets/CCU/ccu_bpe.json")


def test_basic_english():
    basic_english = "The quick brown fox jumps over the lazy dog."
    basic_english_encoded = tokenizer.encode(basic_english)
    assert tokenizer.decode(basic_english_encoded) == basic_english


def test_numbers():
    numbers = "Version 2.1.5 was released on 2026-08-07 at 19:04."
    numbers_encoded = tokenizer.encode(numbers)
    assert tokenizer.decode(numbers_encoded) == numbers


def test_punctuation():
    punctuation = "Wait... what?! No, seriously?!"
    punctuation_encoded = tokenizer.encode(punctuation)
    assert tokenizer.decode(punctuation_encoded) == punctuation


def test_quotes():
    quote = "“Hello,” Anna said. ‘How are you?’"
    quote_encoded = tokenizer.encode(quote)
    assert tokenizer.decode(quote_encoded) == '"Hello," Anna said. \'How are you?\''


def test_unicode():
    unicode_ = "കേരളം — café — naïve — Pokémon"
    unicode_encoded = tokenizer.encode(unicode_)
    assert tokenizer.decode(unicode_encoded) == unicode_


def test_ccu_style():
    ccu_style = "Palm_Fry defended the Conceptual_Void using only legislative intent and half a boiled potato."
    ccu_style_encoded = tokenizer.encode(ccu_style)
    assert tokenizer.decode(ccu_style_encoded) == ccu_style


def test_whitespace():
    whitespace = """
    Hello,

    This    has      multiple spaces.

    And	a	tab.
    """

    whitespace_encoded = tokenizer.encode(whitespace)
    assert tokenizer.decode(whitespace_encoded) == whitespace


def test_byte_level_torture():
    byte_level = "🐈🔥"
    byte_level_encoded = tokenizer.encode(byte_level)
    print(byte_level_encoded)
    # assert tokenizer.decode(byte_level_encoded) == ""


def test_chaos():
    chaos = "Mrs_Bun_fries was deleted from the Age_of_Breakfast because Abraham_Lincoln accidentally distarged the soup into the Ministry_of_Windows at exactly 03:14:15 UTC."
    chaos_encoded = tokenizer.encode(chaos)
    assert tokenizer.decode(chaos_encoded) == chaos


def test_real():
    text = 'Mrs_Bun_fries was deleted from the Age_of_Breakfast because Abraham_Lincoln accidentally distarged the soup into the Ministry_of_Windows.'

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("Original :", repr(text))
    print("Decoded  :", repr(decoded))
    print("Same?    :", text == decoded)


test_byte_level_torture()
test_real()