import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from safetensors import safe_open


class ShardedActivationDataset(IterableDataset):
    """
    Streams pre-extracted activations from a directory of .safetensors shards.

    Each shard must contain a tensor under the key "activations" with shape
    (N, d_model). Additional aligned tensors, such as cell-mode
    "cell_type_ids", are ignored by this activation-only reader. Shards are
    split evenly across DataLoader workers; order is shuffled per epoch to
    approximate random sampling cheaply.
    """

    def __init__(
        self,
        dir_path: str | Path,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
    ):
        self.shard_paths = sorted(Path(dir_path).glob("*.safetensors"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .safetensors shards found in {dir_path}")
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        paths = list(self.shard_paths)

        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        if self.shuffle_shards:
            random.shuffle(paths)

        for path in paths:
            with safe_open(path, framework="pt", device="cpu") as f:
                data = f.get_tensor("activations")  # (N, d_model), bfloat16

            if self.shuffle_within_shard:
                data = data[torch.randperm(data.shape[0])]

            yield from data


def split_shards(
    dir_path: str | Path,
    val_split: float = 0.05,
    shuffle_shards: bool = True,
    shuffle_within_shard: bool = True,
) -> tuple["ShardedActivationDataset", "ShardedActivationDataset"]:
    """Return (train_dataset, val_dataset) split by whole shards."""
    dir_path = Path(dir_path)
    all_paths = sorted(dir_path.glob("*.safetensors"))
    n_val = max(1, int(len(all_paths) * val_split))
    n_train = len(all_paths) - n_val

    train_ds = ShardedActivationDataset.__new__(ShardedActivationDataset)
    train_ds.shard_paths = all_paths[:n_train]
    train_ds.shuffle_shards = shuffle_shards
    train_ds.shuffle_within_shard = shuffle_within_shard

    val_ds = ShardedActivationDataset.__new__(ShardedActivationDataset)
    val_ds.shard_paths = all_paths[n_train:]
    val_ds.shuffle_shards = False
    val_ds.shuffle_within_shard = False

    return train_ds, val_ds
