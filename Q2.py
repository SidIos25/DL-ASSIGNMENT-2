!pip install torch transformers pandas scikit-learn

import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile

!curl -L -o /content/poetry.zip https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/poetry
!unzip /content/poetry.zip -d /content/poetry_data

def load_poetry_dataset(dataset_dir):
    poems = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(dataset_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                current_poem = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('###'):
                        current_poem.append(line)
                    elif current_poem:
                        poems.append('\n'.join(current_poem))
                        current_poem = []
                if current_poem:
                    poems.append('\n'.join(current_poem))
    return poems

def prepare_datasets(poems, output_dir='/content/poetry_processed', test_size=0.1):
    os.makedirs(output_dir, exist_ok=True)
    train_poems, test_poems = train_test_split(poems, test_size=test_size)
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_poems))
    with open(os.path.join(output_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(test_poems))
    return os.path.join(output_dir, 'train.txt'), os.path.join(output_dir, 'test.txt')

def load_model_and_tokenizer(model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size)

def train_model(train_dataset, test_dataset, model, tokenizer, output_dir='/content/results'):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=500,
        save_steps=500,
        warmup_steps=500,
        prediction_loss_only=True,
        logging_dir='/content/logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    trainer.save_model()
    return model

def generate_poetry(model, tokenizer, prompt="", max_length=100, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Loading and processing dataset...")
poems = load_poetry_dataset('/content/poetry_data')
train_path, test_path = prepare_datasets(poems)

print("Loading model...")
model, tokenizer = load_model_and_tokenizer()

print("Preparing datasets...")
train_dataset = load_dataset(train_path, tokenizer)
test_dataset = load_dataset(test_path, tokenizer)

print("Training model...")
model = train_model(train_dataset, test_dataset, model, tokenizer)
model.to('cuda')

print("\nSample generated poetry:")
print(generate_poetry(model, tokenizer, prompt="The stars whisper"))
print("\nAnother example:")
print(generate_poetry(model, tokenizer, prompt="In the quiet night"))