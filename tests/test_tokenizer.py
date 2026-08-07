from tokenizers import Tokenizer
from tokenizers.models import BPE
from ..tokenizer.bpe import BPETokenizer


def test_bpe_tokeniser():
    BPETokenizer.load("data/datasets/CCU/ccu_bpe.json")
    tokenizer = BPETokenizer()

    encoded = tokenizer.encode("Selenium")

    assert encoded == [41, 293, 98, 1740]
    assert tokenizer.decode(encoded) == "Selenium"
    assert tokenizer.id_to_token(1740) == "ium"