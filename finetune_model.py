import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset, Dataset # Import Dataset for consistency

def fine_tune_and_save_model(data_file='data/cleaned_fantasy_names.txt', model_name="distilgpt2", output_dir="./finetuned_model"):
    """
    Fine-tunes a pre-trained DistilGPT2 model on the custom dataset
    and saves the fine-tuned model.
    """
    print(f"--- Starting Fine-tuning Process for {model_name} ---")

    # 1. Load Tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    if "<|newline|>" not in tokenizer.vocab:
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|newline|>']})
        print("Added <|newline|> as a special token.")
    tokenizer.pad_token = tokenizer.eos_token 

    # 2. Load Dataset
    print(f"Loading dataset from {data_file}...")
    raw_dataset = load_dataset('text', data_files={'train': data_file})

    def process_examples(examples):
        concatenated_text = [text + "<|newline|>" for text in examples["text"]]
        return {"text": concatenated_text}

    processed_dataset = raw_dataset.map(process_examples, batched=True, remove_columns=["text"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = processed_dataset.map(
        tokenize_function, batched=True, num_proc=1, remove_columns=["text"]
    )

    block_size = 128

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=1,
    )

    # Split into training and validation sets using datasets library's own method
    print("Splitting dataset into train and validation sets...")
    split_dataset = lm_dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"] # The split is named 'test' by default
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

    # 3. Load Model
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model {model_name} loaded. Number of parameters: {model.num_parameters()}")

    # 4. Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2, # Adjust batch size if you face memory issues (e.g., to 4 or 2)
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir='./logs',
        logging_steps=500,
        
        # Corrected parameters for compatibility with recent Transformers versions
        eval_strategy="epoch", 
        save_strategy="epoch", 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
    )

    # 6. Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 7. Train
    print("Starting model training...")
    try:
        trainer.train()
        print("Model training completed.")

        # 8. Save Fine-tuned Model
        print(f"Saving fine-tuned model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Fine-tuned model and tokenizer saved.")
    except Exception as e:
        print(f"!!! ERROR DURING FINE-TUNING: {e}")
        print("Model was not saved due to an error. Please check the error message above.")

    print("--- Fine-tuning Process Finished ---")

if __name__ == '__main__':
    fine_tune_and_save_model()