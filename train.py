import argparse
from misawantiga.config.defaults import Config, load_config
from training.trainer import Trainer

from misawantiga.dataset.encoder import dataset_encoder
from tokenizer import BPETokenizer, TokenizerTrainer


config = Config() 


def train_tok():
    print()
    trainer = TokenizerTrainer(vocab_size=43_000)
    tok = trainer.train(config.tokenizer_dir)
    return tok


def encode():
    trained = train_tok()
    tok = BPETokenizer(trained)
    tok.save(config.tokenizer_dir)
    encoder = dataset_encoder.DatasetEncoder(tok)
    encoder.encode_dataset(config.dataset_dir, config.train_bin, val_ratio=0.05)


def main():
    parser = argparse.ArgumentParser(
        description="Train Misawantiga LLM from scratch."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fry.py",
        help="Path to Python config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from '{args.config}'")

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()