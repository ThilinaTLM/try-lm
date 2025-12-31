"""Dataset loading and preprocessing for TryLM."""

from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, IterableDataset

from trylm.config import ModelConfig, TrainConfig
from trylm.tokenizer import TryLMTokenizer


class TinyStoriesDataset(Dataset):
    """PyTorch Dataset for TinyStories with pre-tokenized data."""

    def __init__(
        self,
        tokenizer: TryLMTokenizer,
        split: str = "train",
        context_length: int = 512,
        dataset_name: str = "roneneldan/TinyStories",
        max_samples: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            tokenizer: Trained tokenizer
            split: Dataset split ("train" or "validation")
            context_length: Maximum sequence length
            dataset_name: HuggingFace dataset name
            max_samples: Maximum number of samples to load (None for all)
        """
        self.tokenizer = tokenizer
        self.context_length = context_length

        # Load dataset
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # Tokenize all texts upfront
        self.token_ids: list[list[int]] = []

        for item in dataset:
            ids = tokenizer.encode(item["text"], add_special_tokens=True)
            if len(ids) >= 2:  # Need at least 2 tokens for input/target
                self.token_ids.append(ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single item.

        Returns dict with:
            - input_ids: Token IDs for input (all but last token)
            - labels: Token IDs for labels (all but first token)
            - attention_mask: Mask for non-padding tokens
        """
        ids = self.token_ids[idx]

        # Truncate if too long
        if len(ids) > self.context_length + 1:
            ids = ids[: self.context_length + 1]

        # Split into input and target
        input_ids = ids[:-1]
        labels = ids[1:]

        # Pad to context_length
        pad_len = self.context_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len

        input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
        labels = labels + [-100] * pad_len  # -100 is ignored in cross entropy

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class StreamingTinyStoriesDataset(IterableDataset):
    """Streaming dataset for memory-efficient training on large datasets."""

    def __init__(
        self,
        tokenizer: TryLMTokenizer,
        split: str = "train",
        context_length: int = 512,
        dataset_name: str = "roneneldan/TinyStories",
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize the streaming dataset.

        Args:
            tokenizer: Trained tokenizer
            split: Dataset split
            context_length: Maximum sequence length
            dataset_name: HuggingFace dataset name
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.dataset_name = dataset_name
        self.split = split
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)

        for item in dataset:
            ids = self.tokenizer.encode(item["text"], add_special_tokens=True)

            if len(ids) < 2:
                continue

            # Truncate if too long
            if len(ids) > self.context_length + 1:
                ids = ids[: self.context_length + 1]

            # Split into input and target
            input_ids = ids[:-1]
            labels = ids[1:]

            # Pad to context_length
            pad_len = self.context_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len

            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [-100] * pad_len

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }


def create_dataloaders(
    tokenizer: TryLMTokenizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    streaming: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        tokenizer: Trained tokenizer
        model_config: Model configuration (for context length)
        train_config: Training configuration
        streaming: Whether to use streaming dataset

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if streaming:
        train_dataset = StreamingTinyStoriesDataset(
            tokenizer=tokenizer,
            split=train_config.train_split,
            context_length=model_config.context_length,
            dataset_name=train_config.dataset_name,
            shuffle=True,
            seed=train_config.seed,
        )
        val_dataset = StreamingTinyStoriesDataset(
            tokenizer=tokenizer,
            split=train_config.val_split,
            context_length=model_config.context_length,
            dataset_name=train_config.dataset_name,
            shuffle=False,
            seed=train_config.seed,
        )
    else:
        train_dataset = TinyStoriesDataset(
            tokenizer=tokenizer,
            split=train_config.train_split,
            context_length=model_config.context_length,
            dataset_name=train_config.dataset_name,
        )
        val_dataset = TinyStoriesDataset(
            tokenizer=tokenizer,
            split=train_config.val_split,
            context_length=model_config.context_length,
            dataset_name=train_config.dataset_name,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=not streaming,  # Streaming handles its own shuffling
        num_workers=4 if not streaming else 0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4 if not streaming else 0,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
