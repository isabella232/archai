import datasets
from bert_utils import BERTTokenizer

dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

tokenizer = BERTTokenizer()
tokenizer.train_from_iterator(dataset)
tokenizer.save("tokenizer.json")
