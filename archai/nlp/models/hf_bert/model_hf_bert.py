# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's BERT.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertForPreTraining

from archai.nlp.models.hf_bert.config_hf_bert import HfBERTConfig
from archai.nlp.models.model_base import ArchaiModel


class HfBERT(ArchaiModel):
    """Huggingface's BERT standard architecture.

    """

    def __init__(self, **kwargs) -> None:
        """Initializes the class by creating compatible configuration and model objects.

        """

        super().__init__()

        self.config = HfBERTConfig(**kwargs)
        self.model = BertForPreTraining(self.config)

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                next_sentence_label: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                output_loss: Optional[bool] = True,
                output_prediction_scores: Optional[bool] = False
                ) -> Tuple[torch.Tensor, ...]:

        outputs = self.model(input_ids=input_ids,
                             labels=labels,
                             token_type_ids=token_type_ids,
                             next_sentence_label=next_sentence_label,
                             past_key_values=past_key_values)

        if output_loss:
            return (outputs.loss, None, None, outputs.past_key_values)
        
        if output_prediction_scores:
            # BERT only outputs the logits, so they need to be converted with log_softmax
            return (None,
                    F.log_softmax(outputs.prediction_logits, dim=-1),
                    F.log_softmax(outputs.seq_relationship_logits, dim=-1),
                    outputs.past_key_values)

    def get_params(self) -> Dict[str, int]:
        params = {}

        return params
