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
from utils.random_seed import set_random_seed


def pretty_print(string, length, symbol, file):
     print (symbol *((length - len(string)) // 2) + string + symbol *((length - len(string)) // 2), file = file)

# Задать пути к модели, данным и словарю
CHECKPOINTS = "../../mainmodel.ckpt"
DATASET_PATH = "../../NEREL_01/test"
VOCAB_PATH = "for_mrc/rubert/vocab.txt"

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
        batch_size=1, # батчи по одному примеру для вывода
        shuffle=False,
        collate_fn=collate_to_max_length
    )

model.eval()

f = open("test_dataset.out", "w", encoding = "utf-8") # Вывод в файл

batch_num = 0

example_num = 0 # выбор номера батча => для батча размером 1 выводятся все примеры

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

    print("="*45, file = f)
    decoded_spec = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(decoded_spec, file = f)
    pretty_print("Предсказанные сущности", 45, '-', f)
    if start_positions:
        for pos_idx, (start, end) in enumerate(zip(start_positions, end_positions)):
            decoded = tokenizer.decode(input_ids[start: end+1])
            print(decoded, file = f)
    else:
        print("<Ни одной сущности не найдено.>", file = f)
    print("-"*45, file = f)
    pretty_print("Настоящие сущности", 45, '-', f)
    if start_label_positions:
        for start, end in zip(start_label_positions, end_label_positions):
            decoded = tokenizer.decode(input_ids[start: end+1])
            print(decoded, file = f)
    else:
        print("<Ни одной сущности не найдено.>", file = f)
    print("-"*45, file = f)

    print("Батч " + str(batch_num))
    batch_num += 1

print("="*45)

f.close()