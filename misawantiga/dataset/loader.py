import json
from torch.utils.data import DataLoader
from misawantiga.dataset.encoder.binary_dataset import BinaryDataset
from misawantiga.config.defaults import Config


config = Config()

with open(config.train_meta, "r") as f:
    train_meta = json.load(f)

dtype_name = train_meta.get("dtype")
if not dtype_name:
    raise ValueError("Missing dtype: Cannot access dtype from train.meta.json")

dtype = config.dtype_map[dtype_name]

dataset = BinaryDataset(
    path=config.train_bin,
    block_size=config.block_size,
    dtype=dtype
)


def create_loader(dataset, config: "Config") -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last
    )