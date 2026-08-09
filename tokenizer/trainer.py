from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFC, Replace
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as BytelevelDecoder
from tokenizers.trainers import BpeTrainer


class TokenizerTrainer:
    def __init__(
            self,
            vocab_size: int = 32000,
            min_frequency: int = 2,
            show_progress: bool = True
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.show_progress = show_progress

        self.tokenizer = None
        self._trainer = None

        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


    def _build_tokenizer(self):
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

        tokenizer.normalizer = Sequence([
            NFC(),
            Replace("‘", "'"),
            Replace("’", "'"),
            Replace("“", '"'),
            Replace("”", '"')
        ])

        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = BytelevelDecoder()

        return tokenizer
    

    def train(self, dir: Path):
        if not dir.exists():
            raise FileNotFoundError(f"Directory '{dir}' does not exists")

        if not dir.is_dir():
            raise NotADirectoryError(f"{dir} is not a directory")

        files = [
            str(file) for file in list(dir.glob("*.md"))
        ]

        self._tokenizer = self._build_tokenizer()

        self._trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            show_progress=self.show_progress,
            special_tokens=self.special_tokens
        )

        self._tokenizer.train(files=files, trainer=self._trainer)

        return self._tokenizer