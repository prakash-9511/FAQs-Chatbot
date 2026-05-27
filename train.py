import argparse
import inspect
import json
from pathlib import Path

from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


PROMPT_TEMPLATE = """You are the official-style RCOEM website chatbot. Answer only from verified RCOEM FAQ data.

Question: {question}
Answer: {answer}"""


def build_training_text(example):
    return PROMPT_TEMPLATE.format(question=example["question"], answer=example["answer"])


class FAQDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.items = []
        for row in rows:
            encoded = tokenizer(
                build_training_text(row),
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            self.items.append({key: value for key, value in encoded.items()})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face text-generation model on RCOEM FAQs.")
    parser.add_argument("--model_name", default="distilgpt2", help="Hugging Face causal LM to fine-tune.")
    parser.add_argument("--data_path", default="data/rcoem_faqs.jsonl", help="JSONL file with question/answer/source fields.")
    parser.add_argument("--output_dir", default="models/rcoem-chatbot", help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    with data_path.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    tokenized = FAQDataset(rows, tokenizer, args.max_length)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_arg_values = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "logging_steps": 5,
        "save_strategy": "no",
        "report_to": [],
        "do_train": True,
    }
    supported_args = inspect.signature(TrainingArguments).parameters
    training_args = TrainingArguments(
        **{key: value for key, value in training_arg_values.items() if key in supported_args}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
