"""BPE Tokenizer training and wrapper for TryLM."""

from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

from trylm.config import TokenizerConfig


class TryLMTokenizer:
    """Wrapper around HuggingFace tokenizers for BPE tokenization."""

    def __init__(self, tokenizer: Tokenizer, config: TokenizerConfig) -> None:
        self._tokenizer = tokenizer
        self.config = config

        # Cache special token IDs
        self._pad_id = tokenizer.token_to_id(config.pad_token)
        self._eos_id = tokenizer.token_to_id(config.eos_token)
        self._bos_id = tokenizer.token_to_id(config.bos_token)
        self._unk_id = tokenizer.token_to_id(config.unk_token)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Encode a batch of texts."""
        encodings = self._tokenizer.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )
        return [enc.ids for enc in encodings]

    def decode_batch(
        self, batch_ids: list[list[int]], skip_special_tokens: bool = True
    ) -> list[str]:
        """Decode a batch of token IDs."""
        return self._tokenizer.decode_batch(
            batch_ids, skip_special_tokens=skip_special_tokens
        )

    def save(self, path: str | Path) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path / "tokenizer.json"))

    @classmethod
    def load(cls, path: str | Path, config: TokenizerConfig | None = None) -> "TryLMTokenizer":
        """Load tokenizer from file."""
        path = Path(path)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        if config is None:
            config = TokenizerConfig(vocab_size=tokenizer.get_vocab_size())

        return cls(tokenizer, config)

    @classmethod
    def train(
        cls,
        texts: Iterator[str],
        config: TokenizerConfig,
        show_progress: bool = True,
    ) -> "TryLMTokenizer":
        """Train a new BPE tokenizer on the given texts.

        Args:
            texts: Iterator of text strings to train on
            config: Tokenizer configuration
            show_progress: Whether to show training progress

        Returns:
            Trained TryLMTokenizer instance
        """
        # Initialize BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token=config.unk_token))

        # Set up pre-tokenization (GPT-2 style byte-level)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Set up decoder
        tokenizer.decoder = decoders.ByteLevel()

        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=config.special_tokens,
            show_progress=show_progress,
        )

        # Train the tokenizer
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Set up post-processor to add BOS/EOS tokens
        bos_id = tokenizer.token_to_id(config.bos_token)
        eos_id = tokenizer.token_to_id(config.eos_token)

        tokenizer.post_processor = TemplateProcessing(
            single=f"{config.bos_token}:0 $A:0 {config.eos_token}:0",
            pair=f"{config.bos_token}:0 $A:0 {config.eos_token}:0 {config.bos_token}:1 $B:1 {config.eos_token}:1",
            special_tokens=[
                (config.bos_token, bos_id),
                (config.eos_token, eos_id),
            ],
        )

        # Enable padding
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id(config.pad_token),
            pad_token=config.pad_token,
        )

        return cls(tokenizer, config)


def train_tokenizer_on_dataset(
    dataset_name: str = "roneneldan/TinyStories",
    split: str = "train",
    config: TokenizerConfig | None = None,
    num_samples: int | None = None,
    text_column: str = "text",
) -> TryLMTokenizer:
    """Train a tokenizer on a HuggingFace dataset.

    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use
        config: Tokenizer configuration
        num_samples: Number of samples to use (None for all)
        text_column: Name of the text column in the dataset

    Returns:
        Trained TryLMTokenizer instance
    """
    from datasets import load_dataset
    from rich.console import Console

    console = Console()

    if config is None:
        config = TokenizerConfig()

    console.print(f"[bold blue]Loading dataset:[/] {dataset_name}")
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    console.print(f"[bold blue]Training tokenizer on[/] {len(dataset)} samples")
    console.print(f"[bold blue]Target vocab size:[/] {config.vocab_size}")

    def text_iterator():
        for item in dataset:
            yield item[text_column]

    tokenizer = TryLMTokenizer.train(
        texts=text_iterator(),
        config=config,
        show_progress=True,
    )

    console.print(f"[bold green]Tokenizer trained![/] Vocab size: {tokenizer.vocab_size}")

    return tokenizer
