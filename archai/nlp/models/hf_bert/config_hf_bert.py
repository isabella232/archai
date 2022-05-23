# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's BERT configurations.
"""

from typing import Optional

from archai.nlp.models.config_base import Config, SearchConfigParameter, SearchConfig
from transformers import CONFIG_MAPPING


class HfBERTConfig(Config):
    """Huggingface's BERT default configuration.

    """

    attribute_map = {
        'n_token': 'vocab_size',
        'tgt_len': 'max_position_embeddings',
        'n_layer': 'num_hidden_layers',
        'n_head': 'num_attention_heads',
        'd_model': 'hidden_size',
        'd_inner': 'intermediate_size',
        'dropout': 'hidden_dropout_prob',
        'dropatt': 'attention_probs_dropout_prob',
        'weight_init_std': 'initializer_range'
    }
    attribute_map.update(CONFIG_MAPPING['bert']().attribute_map)

    def __init__(self,
                 n_token: Optional[int] = 30522,
                 tgt_len: Optional[int] = 512,
                 d_model: Optional[int] = 768,
                 d_inner: Optional[int] = 3072,
                 dropout: Optional[float] = 0.1,
                 dropatt: Optional[float] = 0.1,
                 weight_init_std: Optional[float] = 0.02,
                 n_layer: Optional[int] = 12,
                 n_head: Optional[int] = 12,
                 **kwargs) -> None:
        """Initializes the class by overriding default arguments.

        Args:
            n_token: Size of the vocabulary (number of tokens).
            tgt_len: Maximum length of sequences (positional embeddings).
            d_model: Dimensionality of the model (also for embedding layer).
            d_inner: Dimensionality of inner feed-forward layers.
            dropout: Dropout probability.
            dropatt: Attention dropout probability.
            weight_init_std: Standard deviation to initialize the weights.
            n_layer: Number of layers.
            n_head: Number of attention heads.

        """

        self.n_token = n_token
        self.tgt_len = tgt_len
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.dropatt = dropatt
        self.weight_init_std = weight_init_std
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_model // n_head

        additional_config = CONFIG_MAPPING['bert']().to_dict()
        for key, value in additional_config.items():
            if key not in self.__dict__.keys():
                setattr(self, key, value)

        super().__init__(**kwargs)


class HfBERTSearchConfig(SearchConfig):
    """Huggingface's BERT search configuration.

    """

    def __init__(self) -> None:
        """Initializes the class by setting default parameters that are used during search.
        
        """
        
        # Default HfBERT search options: n_layer, d_model, d_inner and n_head
        n_layer = SearchConfigParameter(per_layer=False, value=[3, 4, 5, 6, 7, 8, 9, 10])
        d_model = SearchConfigParameter(per_layer=False, value=list(range(128, 1024, 64)))
        d_inner = SearchConfigParameter(per_layer=False, value=list(range(128, 4096, 64)))
        n_head = SearchConfigParameter(per_layer=False, value=[2, 4, 8])

        super().__init__(n_layer=n_layer,
                         d_model=d_model,
                         d_inner=d_inner,
                         n_head=n_head)
