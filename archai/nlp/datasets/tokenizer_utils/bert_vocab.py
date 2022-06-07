from typing import Optional

from archai.nlp.datasets.tokenizer_utils.word_vocab import WordVocab


class BertVocab(WordVocab):
    def __init__(self, save_path:str, min_freq=0, vocab_size=None,
                 bos_token:Optional[str]=None, eos_token:Optional[str]='<eos>',
                 unk_token:Optional[str]='<unk>', pad_token:Optional[str]=None,
                 mask_token:Optional[str]='<mask>', cls_token:Optional[str]='<cls>', sep_token:Optional[str]='<sep>',
                 lower_case=False, delimiter=None, encode_special_tokens=True, decode_special_tokens=True):
        super().__init__(save_path, min_freq=min_freq, vocab_size=vocab_size,
                         bos_token=bos_token, eos_token=eos_token, unk_token=unk_token,
                         pad_token=pad_token, mask_token=mask_token, cls_token=cls_token,
                         sep_token=sep_token, lower_case=lower_case, delimiter=delimiter,
                         encode_special_tokens=encode_special_tokens, decode_special_tokens=decode_special_tokens)
