from pathlib import Path

from tokenizer import TokenizerTrainer


def test_trainer_discovers_nested_markdown_files(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    nested_dir = corpus_dir / "nested"
    nested_dir.mkdir(parents=True)

    (nested_dir / "a.md").write_text("hello hello world\n", encoding="utf-8")
    (nested_dir / "b.md").write_text("hello universe\n", encoding="utf-8")

    trainer = TokenizerTrainer(vocab_size=32, min_frequency=1, show_progress=False)
    tokenizer = trainer.train(corpus_dir)

    assert tokenizer.get_vocab_size() > 4
