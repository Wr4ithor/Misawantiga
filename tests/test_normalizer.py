from ..tokenizer.normalizer import Normalizer

normalizer = Normalizer()

def test_norm():
    assert normalizer.normalize("“Hello”") == '"Hello"'
    assert normalizer.normalize("A\r\nB\rC") == "A\nB\nC"
    assert normalizer.normalize("Hi   \n") == "Hi\n"