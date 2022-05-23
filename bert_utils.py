# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable tokenization utilities from huggingface/transformers.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Union
from datasets import DownloadMode, load_dataset

from tokenizers import Tokenizer
from tokenizers.trainers import Trainer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from typing import Optional

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


# Disables `tokenizers` parallelism due to process being forked
os.environ["TOKENIZERS_PARALLELISM"] = "False"

# Available special tokens
SPECIAL_TOKENS = {
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
    "gpt2_eos_token": "<|endoftext|>",
    "transfo_xl_sep_token": "<formula>",
}


class ArchaiTokenConfig:
    """Serves as the base foundation of a token's configuration."""

    def __init__(
        self,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        mask_token: Optional[str] = None,
    ) -> None:
        """Initializes a token's configuration class by setting attributes.

        Args:
            bos_token: Begin-of-sentence token.
            eos_token: End-of-sentence token.
            unk_token: Unknown token.
            sep_token: Separator token (used for separating two sequences).
            pad_token: Padding token.
            cls_token: Input class token.
            mask_token: Masked token.

        """

        # Special tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

    @classmethod
    def from_file(cls: ArchaiTokenConfig, token_config_path: str) -> ArchaiTokenConfig:
        """Creates a class instance from an input file.

        Args:
            token_config_path: Path to the token's configuration file.

        Returns:
            (ArchaiTokenConfig): Instance of the ArchaiTokenConfig class.

        """

        try:
            with open(token_config_path, "r") as f:
                return cls(**json.load(f))
        except FileNotFoundError as error:
            raise error(f"{token_config_path} could not be found.")

    @property
    def special_tokens(self) -> List[str]:
        """Gathers the available special tokens.

        Returns:
            (List[str]): List of available special tokens.

        """

        return list(
            filter(
                None,
                [
                    self.bos_token,
                    self.eos_token,
                    self.unk_token,
                    self.sep_token,
                    self.pad_token,
                    self.cls_token,
                    self.mask_token,
                ],
            )
        )

    def save(self, output_token_config_path: str) -> None:
        """Saves the token's configuration to an output JSON file.

        Args:
            output_token_config_path: Path to where token's configuration should be saved.

        """

        with open(output_token_config_path, "w") as f:
            json.dump(self.__dict__, f)


class ArchaiTokenizer:
    """Serves as the base foundation of a tokenization pipeline."""

    def __init__(
        self, token_config: ArchaiTokenConfig, tokenizer: Tokenizer, trainer: Trainer
    ) -> None:
        """Attaches required objects for the ArchaiTokenizer.

        Args:
            token_config: ArchaiTokenConfig class with token's configuration.
            tokenizer: Tokenizer class with model from huggingface/transformers.
            trainer: Trainer class from huggingface/transformers.

        """

        # Attaches input arguments as attributes
        self.token_config = token_config
        self.tokenizer = tokenizer
        self.trainer = trainer

    def get_vocab(self, with_added_tokens: Optional[bool] = True) -> Dict[str, int]:
        """Gets the tokenizer vocabulary.

        Args:
            with_added_tokens: Includes additional tokens that were added.

        Returns:
            (Dict[str, int]): Mapping between tokens' keys and values.

        """

        return self.tokenizer.get_vocab(with_added_tokens=with_added_tokens)

    def train_from_iterator(self, dataset) -> None:
        """Trains from in-memory data.

        Args:
            dataset: Raw data to be tokenized.

        """

        def _batch_iterator(
            dataset,
            batch_size: Optional[int] = 10000,
            column_name: Optional[str] = "text",
        ):
            """Iterates over dataset to provide batches.

            Args:
                dataset: Dataset that should be iterated over.
                batch_size: Size of each batch.
                column_name: Name of column that should be retrieved.

            Yields:
                (Dataset): Batch of data based on size and `column_name`.

            """

            for i in range(0, len(dataset), batch_size):
                yield dataset[i : i + batch_size][column_name]

        return self.tokenizer.train_from_iterator(
            _batch_iterator(dataset), self.trainer, len(dataset)
        )

    def save(self, output_tokenizer_path: str) -> None:
        """Saves the pre-trained tokenizer and token's configuration to disk.

        Args:
            output_tokenizer_path: Path to where tokenizer should be saved.

        """

        output_folder_path = os.path.dirname(output_tokenizer_path)
        output_token_config_path = os.path.join(output_folder_path, "token_config.json")

        self.token_config.save(output_token_config_path)
        self.tokenizer.save(output_tokenizer_path)


class ArchaiPreTrainedTokenizer(PreTrainedTokenizerFast):
    """Serves as an abstraction to load/use a pre-trained tokenizer."""

    def __init__(self, *args, **kwargs) -> None:
        """Overrides with custom arguments and keyword arguments."""

        token_config_file = kwargs.pop("token_config_file", None)
        if token_config_file is None:
            self.token_config = ArchaiTokenConfig()
        else:
            self.token_config = ArchaiTokenConfig.from_file(token_config_file)

        # Fills up missing special tokens
        kwargs["bos_token"] = self.token_config.bos_token
        kwargs["eos_token"] = self.token_config.eos_token
        kwargs["unk_token"] = self.token_config.unk_token
        kwargs["sep_token"] = self.token_config.sep_token
        kwargs["pad_token"] = self.token_config.pad_token
        kwargs["cls_token"] = self.token_config.cls_token
        kwargs["mask_token"] = self.token_config.mask_token

        super().__init__(*args, **kwargs)


class BERTTokenizer(ArchaiTokenizer):
    """Creates a customizable BERT-based tokenization pipeline."""

    def __init__(
        self, vocab_size: Optional[int] = 30522, min_frequency: Optional[int] = 0
    ) -> None:
        """Defines the tokenization pipeline.

        Args:
            vocab_size: Maximum size of vocabulary.
            min_frequency: Minimum frequency of tokens.

        """

        # Initializes token's configuration, tokenizer and trainer
        token_config = ArchaiTokenConfig(
            unk_token=SPECIAL_TOKENS["unk_token"],
            sep_token=SPECIAL_TOKENS["sep_token"],
            pad_token=SPECIAL_TOKENS["pad_token"],
            cls_token=SPECIAL_TOKENS["cls_token"],
            mask_token=SPECIAL_TOKENS["mask_token"],
        )
        tokenizer = Tokenizer(WordPiece(unk_token=token_config.unk_token))
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=token_config.special_tokens,
        )

        super().__init__(token_config, tokenizer, trainer)

        # Normalizers, pre- and post-processing templates
        self.tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{token_config.cls_token} $A {token_config.sep_token}",
            pair=f"{token_config.cls_token} $A {token_config.sep_token} $B:1 {token_config.sep_token}:1",
            special_tokens=[(token_config.sep_token, 1), (token_config.cls_token, 3)],
        )



def map_to_list(
    variable: Union[int, float, List[Union[int, float]]], size: int
) -> List[Union[int, float]]:
    """Maps variables to a fixed length list.

    Args:
        variable: Variable to be mapped.
        size: Size to be mapped.

    Returns:
        (List[Union[int, float]]): Mapped list with fixed length.

    """

    if isinstance(variable, List):
        size_diff = size - len(variable)

        if size_diff < 0:
            return variable[:size]
        elif size_diff == 0:
            return variable
        elif size_diff > 0:
            return variable + [variable[-1]] * size_diff

    return [variable] * size


def shuffle_dataset(dataset, seed: int):
    """Shuffles a dataset according to a supplied seed.

    Args:
        dataset: Input dataset.
        seed: Random seed.

    Returns:
        (Dataset): Shuffled dataset.

    """

    # Checks if seed is a valid number (should be applied)
    if seed > -1:
        dataset = dataset.shuffle(seed)

    return dataset


def resize_dataset(dataset, n_samples: int):
    """Resizes a dataset according to a supplied size.

    Args:
        dataset: Input dataset.
        n_samples: Amount of samples.

    Returns:
        (Dataset): Resized dataset.

    """

    # Checks if subsampling should be applied
    if n_samples > -1:
        dataset = dataset.select(range(n_samples))

    return dataset


def tokenize_dataset(
    dataset,
    tokenizer: ArchaiPreTrainedTokenizer,
    mapping_column_name: Optional[str] = "text",
    next_sentence_prediction: Optional[bool] = False,
    truncate: Optional[bool] = True,
    padding: Optional[str] = "max_length",
    batched: Optional[bool] = True,
):
    """Tokenizes a dataset according to supplied tokenizer and constraints.

    Args:
        dataset: Input dataset.
        tokenizer: Input tokenizer.
        mapping_column_name: Defines column to be tokenized.
        next_sentence_prediction: Whether next sentence prediction labels should be created or not.
        truncate: Whether samples should be truncated or not.
        padding: Strategy used to pad samples that do not have the proper size.
        batched: Whether mapping should be batched or not.

    Returns:
        (Union[Dataset, DatasetDict]): Tokenized dataset.

    """

    def _apply_tokenization(examples: List[str]) -> Dict[str, Any]:
        examples_mapping = examples[mapping_column_name]

        if next_sentence_prediction:
            examples, next_sentence_labels = [], []

            for i in range(len(examples_mapping)):
                if random.random() < 0.5:
                    examples.append(examples_mapping[i])
                    next_sentence_labels.append(0)
                else:
                    examples.append(random.choices(examples_mapping, k=2))
                    next_sentence_labels.append(1)

            output_dataset = tokenizer(examples, truncation=truncate, padding=padding)
            output_dataset["next_sentence_label"] = next_sentence_labels

            return output_dataset

        return tokenizer(examples_mapping, truncation=truncate, padding=padding)

    dataset = dataset.map(lambda x: _apply_tokenization(x), batched=batched)

    return dataset


def _should_refresh_cache(refresh: bool) -> DownloadMode:
    """Refreshes the cached dataset by re-downloading/re-creating it.

    Args:
        refresh: Whether the dataset cache should be refreshed or not.

    Returns:
        (DownloadMode): Enumerator that defines whether cache should refresh or not.

    """

    if refresh:
        return DownloadMode.FORCE_REDOWNLOAD

    return DownloadMode.REUSE_DATASET_IF_EXISTS


def load_and_prepare_dataset(
    tokenizer: ArchaiPreTrainedTokenizer,
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    dataset_files: Optional[Union[Dict[str, Any], List[str]]] = None,
    dataset_split: Optional[str] = None,
    dataset_revision: Optional[List[str]] = None,
    dataset_stream: Optional[bool] = False,
    dataset_refresh_cache: Optional[bool] = False,
    random_seed: Optional[int] = 42,
    n_samples: Optional[Union[int, List[int]]] = -1,
    mapping_column_name: Optional[str] = "text",
    next_sentence_prediction: Optional[bool] = False,
    truncate: Optional[bool] = True,
    padding: Optional[str] = "max_length",
    batched: Optional[bool] = True,
    format_column_name: Optional[List[str]] = None,
):
    """Loads and prepare a dataset.

    Args:
        tokenizer: Tokenizer to transform text into tokens.
        dataset_name: Name of dataset to be downloaded.
        dataset_config_name: Name of configuration of dataset to be downloaded.
        dataset_split: Split to be retrieved. None defaults to all splits.
        dataset_files: Files that should be loaded from `dataset_name` (in case it's a folder).
        dataset_revision: Version of the dataset to be loaded.
        dataset_stream: Whether dataset should be streamed or not.
        dataset_refresh_cache: Whether cache should be refreshed or not.
        random_seed: Fixes the order of samples.
        n_samples: Subsamples into a fixed amount of samples.
        mapping_column_name: Defines column to be tokenized.
        next_sentence_prediction: Whether next sentence prediction labels should exist or not.
        truncate: Whether samples should be truncated or not.
        padding: Strategy used to pad samples that do not have the proper size.
        batched: Whether mapping should be batched or not.
        format_column_name: Defines columns that should be available on dataset.

    Returns:
        (Dataset): An already tokenized dataset, ready to be used.

    """

    # Loads dataset from either huggingface/datasets or input folder
    dataset = load_dataset(
        dataset_name,
        name=dataset_config_name,
        data_files=dataset_files,
        download_mode=_should_refresh_cache(dataset_refresh_cache),
        split=dataset_split,
        revision=dataset_revision,
        streaming=dataset_stream,
    )

    # Shuffles and resizes either DatasetDict or Dataset
    if not hasattr(dataset, "info"):
        # Asserts that number of samples is the same length of number of splits
        n_samples_list = map_to_list(n_samples, len(dataset.items()))

        for (split, dataset_split), n_samples in zip(dataset.items(), n_samples_list):
            dataset[split] = shuffle_dataset(dataset_split, random_seed)
            dataset[split] = resize_dataset(dataset_split, n_samples)

    else:
        dataset = shuffle_dataset(dataset, random_seed)
        dataset = resize_dataset(dataset, n_samples)

    # Tokenizes either DatasetDict or Dataset
    dataset = tokenize_dataset(
        dataset,
        tokenizer,
        mapping_column_name,
        next_sentence_prediction,
        truncate,
        padding,
        batched,
    )
    dataset.set_format(type="torch", columns=format_column_name)

    return dataset
