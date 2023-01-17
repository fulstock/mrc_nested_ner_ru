# encoding: utf-8

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
import torch

import json

from trainer import *
from utils.get_parser import get_parser
from tqdm.auto import tqdm

from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.mrc_ner_dataset import collate_to_max_length
from utils.random_seed import set_random_seed

def pretty_print(string, length, symbol, file):
     print (symbol *((length - len(string)) // 2) + string + symbol *((length - len(string)) // 2), file = file)

def test_dataset():

    parser = get_parser()
    parser = BertLabeling.add_model_specific_args(parser) 
    parser = Trainer.add_argparse_args(parser) 
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    # Задать пути к модели, данным и словарю
    CHECKPOINTS = args.pretrained_checkpoint
    DATASET_PATH = os.path.join(args.data_dir, "test.json")
    VOCAB_PATH = os.path.join(args.bert_config_dir, "vocab.txt")

    model = BertLabeling.load_from_checkpoint(
        checkpoint_path=CHECKPOINTS,
    )

    tokenizer = BertWordPieceTokenizer(VOCAB_PATH, lowercase = False)

    dataset = MRCNERDataset(dataset_path=DATASET_PATH,
                            tokenizer=tokenizer,
                            max_length=args.max_length,
                            pad_to_maxlen=False
                            )

    dataloader = DataLoader(
            dataset=dataset,
            batch_size=1, # батчи по одному примеру для вывода
            shuffle=False,
            num_workers = args.workers,
            collate_fn=collate_to_max_length
        )

    model.eval()

    if args.default_root_dir is None:
        args.default_root_dir = "./"

    f = open(os.path.join(args.default_root_dir, "test_dataset.out"), "w", encoding = "utf-8") # Вывод в файл

    batch_num = 0
    example_num = 0 # выбор номера батча => для батча размером 1 выводятся все примеры

    print("Всего батчей: " + str(len(dataloader)))

    for batch in tqdm(dataloader):

        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_ids, tag_ids = batch
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

        sample_id = sample_ids[example_num]
        # print(sample_id)
        filename = [d["filename"] for d in dataset.all_data if int(d["id"].split('.')[0]) == int(sample_id)][0]

        print("="*45, file = f)
        pretty_print(filename, 45, '_', f)
        decoded_spec = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(decoded_spec, file = f)
        pretty_print("Предсказанные сущности", 45, '-', f)
        if start_positions:
            for pos_idx, (start, end) in enumerate(zip(start_positions, end_positions)):
                decoded = tokenizer.decode(input_ids[start: end+1])
                print(decoded, file = f)
        else:
            print("<Ни одной сущности не найдено>", file = f)
        print("-"*45, file = f)
        pretty_print("Настоящие сущности", 45, '-', f)
        if start_label_positions:
            for start, end in zip(start_label_positions, end_label_positions):
                decoded = tokenizer.decode(input_ids[start: end+1])
                print(decoded, file = f)
        else:
            print("<Ни одной сущности не найдено>", file = f)
        print("-"*45, file = f)

        print("Батч " + str(batch_num))
        batch_num += 1

        if batch_num == 20:
            break

    print("="*45, file = f)

    f.close()

if __name__ == '__main__':
    test_dataset()