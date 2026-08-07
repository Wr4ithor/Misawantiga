from tokenizer import BPETokenizer, TokenizerTrainer
from config import Config


def encode_and_decode():
    tokenizer = BPETokenizer()
    BPETokenizer.load(Config.DEFAULT_TOKENIZER)

    text = ""