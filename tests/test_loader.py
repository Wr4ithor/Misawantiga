import json
from misawantiga.dataset.encoder import dataset_encoder, binary_dataset
from misawantiga.dataset.loader import create_loader
from misawantiga.config.defaults import Config
from tokenizer import TokenizerTrainer, BPETokenizer


config = Config(num_workers=0, pin_memory=False)

with open(config.train_meta, "r") as f:
    train_meta = json.load(f)

dtype_name = train_meta.get("dtype")
if not dtype_name:
    raise ValueError("Missing dtype: Cannot access dtype from train.meta.json")

dtype = config.dtype_map[dtype_name]

# trainer = TokenizerTrainer(config.vocab_size)
# trained = trainer.train(config.tokenizer_dir)
# tokenizer = BPETokenizer(trained)
# tokenizer.save(config.tokenizer_dir)

# tokenizer = BPETokenizer.load(config.tokenizer_path)
# encoder = dataset_encoder.DatasetEncoder(tokenizer)
# encoder.encode_dataset(
#     input_dir=config.dataset_dir,
#     output_file=config.train_bin,
#     source_name="local"
# )

dataset = binary_dataset.BinaryDataset(
    path=config.train_bin,
    block_size=config.block_size,
    dtype=dtype
)

def test_loader():
    loader = create_loader(
        dataset=dataset,
        config=config
    )

    x, y = next(iter(loader))

    print(x.shape)
    print(y.shape)

    print(x.dtype)
    print(y.dtype)

test_loader()