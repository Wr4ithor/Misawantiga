from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from misawantiga.dataset.encoder.binary_dataset import BinaryDataset
from misawantiga.dataset.encoder import binary_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_binary_dataset_file(tmp_path, tokens: np.ndarray):
    """
    Write a small token array to a binary file and return its path.
    """
    path = tmp_path / "dataset.bin"
    tokens.astype(np.uint16).tofile(path)
    return path


def collect_samples(dataset: BinaryDataset):
    """
    Consume the entire dataset and return the samples as Python lists.
    """
    return list(dataset)


def first_tokens(samples):
    """
    Return the first token of every x window.

    This gives us the starting indices of the generated samples when the
    source tokens are unique.
    """
    return [int(x[0]) for x, _ in samples]


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

def test_dataset_is_iterable_dataset(tmp_path):
    """
    BinaryDataset should be a PyTorch IterableDataset.
    """
    tokens = np.arange(20, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset = BinaryDataset(
        path=path,
        block_size=4,
        dtype=np.uint16,
        shuffle=False,
    )

    assert isinstance(dataset, torch.utils.data.IterableDataset)


def test_generates_correct_x_y_pairs(tmp_path):
    """
    For autoregressive training:

        x = tokens[i : i + block_size]
        y = tokens[i + 1 : i + block_size + 1]
    """
    tokens = np.arange(10, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset = BinaryDataset(
        path=path,
        block_size=4,
        dtype=np.uint16,
        shuffle=False,
    )

    samples = collect_samples(dataset)

    assert len(samples) == 6

    x, y = samples[0]

    assert torch.equal(
        x,
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )

    assert torch.equal(
        y,
        torch.tensor([1, 2, 3, 4], dtype=torch.long),
    )


def test_x_and_y_are_shifted_by_one(tmp_path):
    """
    Every target sequence must be exactly one token ahead of its input.
    """
    tokens = np.arange(50, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset = BinaryDataset(
        path=path,
        block_size=8,
        dtype=np.uint16,
        shuffle=False,
    )

    for x, y in dataset:
        assert torch.equal(x[1:], y[:-1])


def test_tensor_shapes_and_dtype(tmp_path):
    """
    Dataset should yield block_size-length torch.long tensors.
    """
    tokens = np.arange(30, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    block_size = 7

    dataset = BinaryDataset(
        path=path,
        block_size=block_size,
        dtype=np.uint16,
        shuffle=False,
    )

    for x, y in dataset:
        assert x.shape == (block_size,)
        assert y.shape == (block_size,)

        assert x.dtype == torch.long
        assert y.dtype == torch.long


def test_sample_count(tmp_path):
    """
    Number of possible windows is:

        len(tokens) - block_size
    """
    tokens = np.arange(100, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    block_size = 16

    dataset = BinaryDataset(
        path=path,
        block_size=block_size,
        dtype=np.uint16,
        shuffle=False,
    )

    samples = collect_samples(dataset)

    assert len(samples) == 100 - block_size


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------

def test_shuffle_false_preserves_order(tmp_path):
    """
    With shuffle=False, samples should be generated sequentially.
    """
    tokens = np.arange(100, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset = BinaryDataset(
        path=path,
        block_size=8,
        dtype=np.uint16,
        shuffle=False,
        seed=123,
        shuffle_block_size=20,
    )

    samples = collect_samples(dataset)

    starts = first_tokens(samples)

    assert starts == list(range(100 - 8))


def test_shuffle_is_reproducible(tmp_path):
    """
    Same seed + same epoch must produce the same ordering.
    """
    tokens = np.arange(1000, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset_a = BinaryDataset(
        path=path,
        block_size=8,
        dtype=np.uint16,
        shuffle=True,
        seed=123,
        shuffle_block_size=100,
    )

    dataset_b = BinaryDataset(
        path=path,
        block_size=8,
        dtype=np.uint16,
        shuffle=True,
        seed=123,
        shuffle_block_size=100,
    )

    starts_a = first_tokens(collect_samples(dataset_a))
    starts_b = first_tokens(collect_samples(dataset_b))

    assert starts_a == starts_b


def test_different_epochs_produce_different_order(tmp_path):
    """
    Changing the epoch should change the deterministic shuffle ordering.
    """
    tokens = np.arange(1000, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset = BinaryDataset(
        path=path,
        block_size=8,
        dtype=np.uint16,
        shuffle=True,
        seed=123,
        shuffle_block_size=100,
    )

    dataset.set_epoch(0)
    epoch_0 = first_tokens(collect_samples(dataset))

    dataset.set_epoch(1)
    epoch_1 = first_tokens(collect_samples(dataset))

    assert epoch_0 != epoch_1


def test_set_epoch_is_reproducible(tmp_path):
    """
    Returning to an earlier epoch should reproduce its original ordering.
    """
    tokens = np.arange(1000, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset = BinaryDataset(
        path=path,
        block_size=8,
        dtype=np.uint16,
        shuffle=True,
        seed=123,
        shuffle_block_size=100,
    )

    dataset.set_epoch(0)
    epoch_0_first = first_tokens(collect_samples(dataset))

    dataset.set_epoch(1)
    _ = collect_samples(dataset)

    dataset.set_epoch(0)
    epoch_0_second = first_tokens(collect_samples(dataset))

    assert epoch_0_first == epoch_0_second


def test_shuffle_does_not_duplicate_or_drop_samples(tmp_path):
    """
    Every valid starting position should occur exactly once per epoch.

    The implementation shuffles chunks rather than the entire dataset, so
    this verifies that chunking doesn't accidentally duplicate or omit data.
    """
    tokens = np.arange(1000, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    block_size = 8

    dataset = BinaryDataset(
        path=path,
        block_size=block_size,
        dtype=np.uint16,
        shuffle=True,
        seed=123,
        shuffle_block_size=73,
    )

    starts = first_tokens(collect_samples(dataset))

    expected = set(range(len(tokens) - block_size))

    assert len(starts) == len(expected)
    assert set(starts) == expected


# ---------------------------------------------------------------------------
# Epoch handling
# ---------------------------------------------------------------------------

def test_set_epoch_updates_epoch(tmp_path):
    """
    set_epoch() should update the dataset's current epoch.
    """
    tokens = np.arange(20, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset = BinaryDataset(
        path=path,
        block_size=4,
        dtype=np.uint16,
    )

    assert dataset.epoch == 0

    dataset.set_epoch(5)

    assert dataset.epoch == 5


# ---------------------------------------------------------------------------
# Worker partitioning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("worker_id", [0, 1, 2])
def test_worker_partitioning(tmp_path, worker_id):
    """
    Each worker should receive a distinct portion of the dataset.

    We simulate DataLoader worker information rather than starting actual
    multiprocessing workers, which keeps this test fast and reliable.
    """
    tokens = np.arange(300, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset = BinaryDataset(
        path=path,
        block_size=10,
        dtype=np.uint16,
        shuffle=False,
    )

    fake_worker_info = type(
        "WorkerInfo",
        (),
        {
            "id": worker_id,
            "num_workers": 3,
        },
    )()

    with patch(
        "misawantiga.dataset.encoder.binary_dataset.torch.utils.data.get_worker_info",
        return_value=fake_worker_info,
    ):
        samples = collect_samples(dataset)

    starts = first_tokens(samples)

    assert starts


def test_worker_partitions_do_not_overlap(tmp_path):
    """
    Simulate multiple workers and verify that their assigned samples don't
    overlap.
    """
    tokens = np.arange(300, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    block_size = 10

    all_worker_starts = []

    for worker_id in range(3):
        dataset = BinaryDataset(
            path=path,
            block_size=block_size,
            dtype=np.uint16,
            shuffle=False,
        )

        fake_worker_info = type(
            "WorkerInfo",
            (),
            {
                "id": worker_id,
                "num_workers": 3,
            },
        )()

        with patch(
            "misawantiga.dataset.encoder.binary_dataset.get_worker_info",
            return_value=fake_worker_info,
        ):
            starts = first_tokens(collect_samples(dataset))

        all_worker_starts.append(set(starts))

    assert all_worker_starts[0].isdisjoint(all_worker_starts[1])
    assert all_worker_starts[0].isdisjoint(all_worker_starts[2])
    assert all_worker_starts[1].isdisjoint(all_worker_starts[2])

    combined = set().union(*all_worker_starts)

    assert combined == set(range(len(tokens) - block_size))


# ---------------------------------------------------------------------------
# DDP partitioning
# ---------------------------------------------------------------------------

def test_ddp_rank_partitioning(tmp_path):
    """
    Simulate two DDP ranks and verify that they receive distinct samples.
    """
    tokens = np.arange(300, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    block_size = 10

    rank_starts = []

    for rank in range(2):
        dataset = BinaryDataset(
            path=path,
            block_size=block_size,
            dtype=np.uint16,
            shuffle=False,
        )

        with (
            patch(
                "misawantiga.dataset.encoder.binary_dataset.dist.is_available",
                return_value=True,
            ),
            patch(
                "misawantiga.dataset.encoder.binary_dataset.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "misawantiga.dataset.encoder.binary_dataset.dist.get_rank",
                return_value=rank,
            ),
            patch(
                "misawantiga.dataset.encoder.binary_dataset.dist.get_world_size",
                return_value=2,
            ),
        ):
            starts = first_tokens(collect_samples(dataset))

        rank_starts.append(set(starts))

    assert rank_starts[0].isdisjoint(rank_starts[1])

    combined = rank_starts[0] | rank_starts[1]

    assert combined == set(range(len(tokens) - block_size))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_dataset_with_exactly_one_window(tmp_path):
    """
    block_size + 1 tokens should produce exactly one training sample.
    """
    block_size = 8

    tokens = np.arange(block_size + 1, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset = BinaryDataset(
        path=path,
        block_size=block_size,
        dtype=np.uint16,
        shuffle=False,
    )

    samples = collect_samples(dataset)

    assert len(samples) == 1

    x, y = samples[0]

    assert torch.equal(
        x,
        torch.arange(block_size, dtype=torch.long),
    )

    assert torch.equal(
        y,
        torch.arange(1, block_size + 1, dtype=torch.long),
    )


def test_dataset_too_small_for_one_window(tmp_path):
    """
    Fewer than block_size + 1 tokens means there are no complete windows.
    """
    block_size = 8

    tokens = np.arange(block_size, dtype=np.uint16)
    path = create_binary_dataset_file(tmp_path, tokens)

    dataset = BinaryDataset(
        path=path,
        block_size=block_size,
        dtype=np.uint16,
        shuffle=False,
    )

    samples = collect_samples(dataset)

    assert samples == []