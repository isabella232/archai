# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Availability dictionaries of implemented Transformer-based classes.
"""

# Huggingface's BERT
from archai.nlp.models.hf_bert.config_hf_bert import (HfBERTConfig,
                                                      HfBERTSearchConfig)
from archai.nlp.models.hf_bert.model_hf_bert import HfBERT
from archai.nlp.models.hf_bert.onnx_hf_bert import (HfBERTOnnxConfig,
                                                    HfBERTOnnxModel)

# Huggingface's Open AI GPT-2
from archai.nlp.models.hf_gpt2.config_hf_gpt2 import (HfGPT2Config, HfGPT2SearchConfig,
                                                      HfGPT2FlexConfig, HfGPT2FlexSearchConfig)
from archai.nlp.models.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex
from archai.nlp.models.hf_gpt2.onnx_hf_gpt2 import HfGPT2OnnxConfig, HfGPT2OnnxModel

# Huggingface's Transformer-XL
from archai.nlp.models.hf_transfo_xl.config_hf_transfo_xl import (HfTransfoXLConfig,
                                                                  HfTransfoXLSearchConfig)
from archai.nlp.models.hf_transfo_xl.model_hf_transfo_xl import HfTransfoXL
from archai.nlp.models.hf_transfo_xl.onnx_hf_transfo_xl import (HfTransfoXLOnnxConfig,
                                                                HfTransfoXLOnnxModel)

# NVIDIA's Memory Transformer
from archai.nlp.models.mem_transformer.config_mem_transformer import (MemTransformerLMConfig,
                                                                      MemTransformerLMSearchConfig)
from archai.nlp.models.mem_transformer.model_mem_transformer import MemTransformerLM
from archai.nlp.models.mem_transformer.onnx_mem_transformer import (MemTransformerLMOnnxConfig,
                                                                    MemTransformerLMOnnxModel)

# Analytical parameters formulae
from archai.nlp.models.model_utils.analytical_params_formulae import (get_params_hf_bert_formula,
                                                                      get_params_hf_gpt2_formula,
                                                                      get_params_hf_gpt2_flex_formula,
                                                                      get_params_hf_transfo_xl_formula,
                                                                      get_params_mem_transformer_formula)

MODELS = {
    'hf_bert': HfBERT,
    'hf_gpt2': HfGPT2,
    'hf_gpt2_flex': HfGPT2Flex,
    'hf_transfo_xl': HfTransfoXL,
    'mem_transformer': MemTransformerLM
}

MODELS_CONFIGS = {
    'hf_bert': HfBERTConfig,
    'hf_gpt2': HfGPT2Config,
    'hf_gpt2_flex': HfGPT2FlexConfig,
    'hf_transfo_xl': HfTransfoXLConfig,
    'mem_transformer': MemTransformerLMConfig
}

MODELS_SEARCH_CONFIGS = {
    'hf_bert': HfBERTSearchConfig,
    'hf_gpt2': HfGPT2SearchConfig,
    'hf_gpt2_flex': HfGPT2FlexSearchConfig,
    'hf_transfo_xl': HfTransfoXLSearchConfig,
    'mem_transformer': MemTransformerLMSearchConfig
}

MODELS_PARAMS_FORMULAE = {
    'hf_bert': get_params_hf_bert_formula,
    'hf_gpt2': get_params_hf_gpt2_formula,
    'hf_gpt2_flex': get_params_hf_gpt2_flex_formula,
    'hf_transfo_xl': get_params_hf_transfo_xl_formula,
    'mem_transformer': get_params_mem_transformer_formula
}

ONNX_MODELS = {
    'hf_bert': HfBERTOnnxModel,
    'hf_gpt2': HfGPT2OnnxModel,
    'hf_gpt2_flex': HfGPT2OnnxModel,
    'hf_transfo_xl': HfTransfoXLOnnxModel,
    'mem_transformer': MemTransformerLMOnnxModel
}

ONNX_MODELS_CONFIGS = {
    'hf_bert': HfBERTOnnxConfig,
    'hf_gpt2': HfGPT2OnnxConfig,
    'hf_gpt2_flex': HfGPT2OnnxConfig,
    'hf_transfo_xl': HfTransfoXLOnnxConfig,
    'mem_transformer': MemTransformerLMOnnxConfig
}
