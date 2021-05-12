# encoding: utf-8


import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import BertWordPieceTokenizer

from torch.utils.data import DataLoader
import torch

from datasets.mrc_ner_dataset import MRCNERDataset
from trainer import BertLabeling
from datasets.mrc_ner_dataset import collate_to_max_length

# Задать пути к модели, данным и словарю
CHECKPOINTS = "lightning_logs/version_37/checkpoints/epoch=2_v0.ckpt"
DATASET_PATH = "NEREL_01/test"
VOCAB_PATH = "rubert/vocab.txt"

set_random_seed(0)

checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINTS,
    save_top_k=-1, # не сохранять
    verbose=False,
    monitor="span_f1",
    period=1,
    mode="max",
)

trainer = Trainer(
    checkpoint_callback=checkpoint_callback
)

model = BertLabeling.load_from_checkpoint(
    checkpoint_path=CHECKPOINTS,
)

tokenizer = BertWordPieceTokenizer(VOCAB_PATH, lowercase = False)

dataset = MRCNERDataset(dataset_path=DATASET_PATH,
                        tokenizer=tokenizer,
                        max_length=100,
                        pad_to_maxlen=False
                        )

dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_to_max_length
    )

model.eval()

# batch_num = 0

example_num = 0 # для демонстрации выводится лишь первый пример из каждого батча. 

for batch in dataloader:

    tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels = batch
    attention_mask = (tokens != 0).long()

    with torch.no_grad():
        start_logits, end_logits, match_logits = model(tokens, attention_mask, token_type_ids)

    batch_size, seq_len = start_logits.size()

    start_preds = start_logits > 0 
    end_preds = end_logits > 0
    match_preds = match_logits > 0

    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)
    match_preds = match_label_mask & match_preds
    
    start_preds = start_preds[example_num]
    end_preds = end_preds[example_num]
    match_preds = match_preds[example_num]
    
    input_ids = tokens[example_num].tolist()
    match_labels = match_labels[example_num]

    start_positions, end_positions = torch.where(match_preds == True)
    start_label_positions, end_label_positions = torch.where(match_labels > 0)

    start_positions = start_positions.tolist()
    end_positions = end_positions.tolist()
    start_label_positions = start_label_positions.tolist()
    end_label_positions = end_label_positions.tolist()

    if start_positions:
        print("="*40)
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        print("-"*(40 - len("Предсказанные сущности")) / 2 + 
              "Предсказанные сущности" + 
              "-"*(40 - len("Предсказанные сущности")) / 2)
        for start, end in zip(start_positions, end_positions):
            print(tokenizer.decode(input_ids[start: end+1]))
        print("-"*40)
        print("-"*(40 - len("Настоящие сущности")) / 2 + 
              "Настоящие сущности" + 
              "-"*(40 - len("Настоящие сущности")) / 2)
        for start, end in zip(start_label_positions, end_label_positions):
            print(tokenizer.decode(input_ids[start: end+1]))
        print("-"*40)