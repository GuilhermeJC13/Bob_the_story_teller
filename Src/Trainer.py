import json
import re
import pandas as pd
import wandb
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, pipeline


def add_token(dataset):
    sentences = eval(dataset["Text"])
    treated_sentences = []

    for text in sentences:
        treated_sentences.append("<start> " + text + " <end>")

    return " ".join(treated_sentences)


def add_special_token():
    pass


def encode(dataset):
    return tokenizer(dataset["Text"], max_length = 1024, truncation = True, add_special_token = True, 
    padding = False)



with open("Data\dialog_data.json") as file:
    data = json.load(file)

    conversations = []

for x in data:
    for y in range(len(x["dialog"])-1):
        #token = x["dialog"][y+1]["text"]
        token = "<start> " + x["dialog"][y+1]["text"] + " <end>"

        try:
            print(token)
            conversations.append(token)

        except:
            continue

d = pd.DataFrame(conversations, columns=["Text"])
print(d)

#d["Text"] = d.apply(add_token, axis=1)

train_set, val_set = train_test_split(d["Text"], test_size = 0.2, random_state = 98)

train_set.to_csv("Data\Dataset/train.csv")
val_set.to_csv("Data\Dataset/test.csv")

dataset = load_dataset("csv", data_files={'train': "Data\Dataset/train.csv", 'test' : "Data\Dataset/test.csv"})

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', bos_token = '<start>', eos_token = '<end>');
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

train_set = dataset['train'].map(encode, batched = True).remove_columns(column_names= ['Unnamed: 0', 'Text'])
eval_set = dataset['test'].map(encode, batched = True).remove_columns(column_names= ['Unnamed: 0', 'Text'])

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)

block_size = 128

model = GPT2LMHeadModel.from_pretrained('distilgpt2', eos_token_id = tokenizer.eos_token_id, 
                                        bos_token_id = tokenizer.bos_token_id)

model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(output_dir= 'Data\model\model1/', overwrite_output_dir= True, 
                                  num_train_epochs = 1,
                                  save_steps= -1, per_device_train_batch_size = 16, 
                                  prediction_loss_only=True,
                                  per_device_eval_batch_size = 64, evaluation_strategy = 'epoch',
                                   load_best_model_at_end =True, eval_accumulation_steps = 1)

trainer = Trainer(model = model, args = training_args, 
                train_dataset= train_set, eval_dataset=val_set, 
                 data_collator = data_collator)

wandb.login(key = "4600171b3cd1c3a58144143b970c1d1a1d2076cf")

trainer.train()
wandb.watch(model, log='all')

