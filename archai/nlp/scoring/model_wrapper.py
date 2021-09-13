from typing import Tuple
import logging
import functools
import time

from scipy.special import softmax

import numpy as np

import torch
from torch import nn


class ModelWrapper:
    def __init__(self, model:nn.Module, space_token_id:int,
                 max_seq_len, # 64 for Micronet, 128 otherwise?
                 device=None):
        self.space_token_id = space_token_id
        self.max_seq_len = max_seq_len
        self.model = model
        self.device = next(model.parameters()).device if device is None else device
        self.model.eval()

    @functools.lru_cache(maxsize=1024)
    def get_logits(self, input_ids: tuple) -> list:
        if len(input_ids) == 0:
            input_ids = (self.space_token_id,)
        elif len(input_ids) > self.max_seq_len:
            input_ids = input_ids[(-1*self.max_seq_len):]

        input_ids_len = len(input_ids)
        if input_ids_len < self.max_seq_len:
            input_ids = input_ids + (0,) * (self.max_seq_len - input_ids_len)

        tokenized_tensor = torch.tensor(input_ids).to(self.device) # pylint: disable=not-callable
        outputs = self.model(input_ids=tokenized_tensor)
        next_token_logits = outputs[0][-1, :].detach()
        return next_token_logits.tolist()

    def get_loss(self, input_ids: tuple) -> float:
        # TODO: BUG: Few % difference from calculating manually
        # shift labels & inputs?
        if len(input_ids) == 0:
            return 0.0

        labels_len_sum = 0
        loss_sum = 0.0
        for idx in range(0, len(input_ids), self.max_seq_len):
            tmp_input_ids = (self.space_token_id,) + input_ids[idx:(idx + self.max_seq_len)]
            labels_len = len(tmp_input_ids) - 1
            tokenized_tensor = torch.tensor(tmp_input_ids).to(self.device) # pylint: disable=not-callable
            outputs = self.model(input_ids=tokenized_tensor, labels=tokenized_tensor)
            labels_len_sum += labels_len
            loss_sum += labels_len * float(outputs.loss)
        return loss_sum / labels_len_sum

    @functools.lru_cache(maxsize=1024)
    def get_probs(self, input_ids: tuple) -> list:
        """Run the model with given input_ids, return probability distribution over the tokens

        Args:
            input_ids (tuple): token ids from the associated tokenizer.

        Returns:
            list: probability distribution over all tokens
        """
        start = time.time()
        input_ids = tuple(input_ids[(-1*self.max_seq_len):])
        logits = self.get_logits(input_ids)
        probs = softmax(logits)

        logging.debug("Model time for %s input_ids: %s ms; first 10 probs: %s", len(input_ids), 1000*(time.time() - start), probs[:10])
        return probs

    @functools.lru_cache(maxsize=1024)
    def get_top_token_prob(self, input_ids: tuple) -> Tuple[int, float]:
        """Return id and probability of top token

        Args:
            input_ids (list): token ids from the associated tokenizer.
            This requires list for easier implementation.

        Returns:
            (int, float): idx and probability of the most likely token
        """
        probs = self.get_probs(tuple(input_ids))
        idx = np.argmax(probs)
        return (idx, probs[idx])