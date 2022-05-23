# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's BERT for ONNX.
"""

from typing import Any, Dict

from onnxruntime.transformers.onnx_model_bert import BertOnnxModel as HfBERTOnnxModel

from archai.nlp.models.config_base import OnnxConfig


class HfBERTOnnxConfig(OnnxConfig):
    """Huggingface's BERT ONNX-based configuration.

    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initializes the class by setting missing keys on incoming
            model's configuration.

        Args:
            model_config: Configuration of the model that will be exported.

        """

        super().__init__(model_config, model_type='bert')
