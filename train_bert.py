from transformers import BertConfig, BertForPreTraining, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from bert_utils import ArchaiPreTrainedTokenizer, load_and_prepare_dataset


if __name__ == '__main__':
    # Loads the pre-trained tokenizer
    tokenizer = ArchaiPreTrainedTokenizer(
        token_config_file="token_config.json",
        tokenizer_file="tokenizer.json",
        model_input_names=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
        ],
        model_max_length=512,
    )

    # Loads and prepares the dataset
    dataset = load_and_prepare_dataset(
        tokenizer,
        "wikitext",
        "wikitext-103-raw-v1",
        next_sentence_prediction=True,
    )

    # Creates the masked language modeling collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    # Initializes BERT with a custom configuration
    config = BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        pad_token_id=3,
        vocab_size=30522,
    )
    model = BertForPreTraining(config=config)

    # Defines training and trainer arguments
    training_args = TrainingArguments(
        "wikitext_bert_pretrain",
        evaluation_strategy="steps",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_steps=40000,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # Starts the training
    trainer.train()
