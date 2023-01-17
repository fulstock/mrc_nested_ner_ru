# encoding: utf-8


import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset

import torch
from typing import List

import os
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from string import punctuation
from nltk.corpus import stopwords
import nltk
import re

import json


def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    output.append(torch.stack([x[-2] for x in batch]))
    output.append(torch.stack([x[-1] for x in batch]))

    return output


class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
    """
    def __init__(self, dataset_path, tokenizer: BertWordPieceTokenizer, max_length: int = 128, pad_to_maxlen=False, tag = None):

        dataset_file = open(dataset_path, encoding='UTF-8')
        self.all_data = json.load(dataset_file)
        
        if tag:
            self.all_data = [sample for sample in self.all_data if sample["tag"] == tag]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_maxlen = pad_to_maxlen

        dataset_file.close()

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id

        """
        data = self.all_data[item]
        tokenizer = self.tokenizer

        sample_id = data.get("id", "0.0")
        sample_idx, tag_idx = sample_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        tag_idx = torch.LongTensor([int(tag_idx)])

        query = data["query"]
        context = data["context"]
        start_positions = data["start_positions"]
        end_positions = data["end_positions"]

        # add space offsets
        # words = context.split()
        # start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
        # end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        # print(query, end = '\n\n')
        # print(context, end = '\n\n')

        # print(start_positions, end = '\n\n')
        # print(end_positions, end = '\n\n')

        query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True) #.encodings[0]
        tokens = query_context_tokens.ids
        type_ids = query_context_tokens.type_ids
        offsets = query_context_tokens.offsets

        # print(tokens)
        # print(offsets)

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            # skip query tokens
            if type_ids[token_idx] == 0:
                continue
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx

        # print(origin_offset2token_idx_start, end = '\n\n')
        # print(origin_offset2token_idx_end, end = '\n\n')
        
        try:
            new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
            new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]
        except KeyError:
            
            ''' print(query, end = '\n\n')
            print(context, end = '\n\n')
            print(start_positions, end = '\n\n')
            print(end_positions, end = '\n\n')
            print(tokens, end = '\n\n')
            print(offsets, end = '\n\n')
            print(origin_offset2token_idx_start, end = '\n\n')
            print(origin_offset2token_idx_end, end = '\n\n')'''
            
            # print(data) 
        
            # Пример некорректен из-за опечатки, обнуляем наличие сущности
        
            start_positions = []
            end_positions = []
            new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
            new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]
            # pass # Ошибка в исходной разметке
            

        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        # the start/end position must be whole word
        for token_idx in range(len(tokens)):    
            current_word_idx = query_context_tokens.word_ids[token_idx]
            next_word_idx = query_context_tokens.word_ids[token_idx+1] if token_idx+1 < len(tokens) else None
            prev_word_idx = query_context_tokens.word_ids[token_idx-1] if token_idx-1 > 0 else None
            if prev_word_idx is not None and current_word_idx == prev_word_idx:
                start_label_mask[token_idx] = 0
            if next_word_idx is not None and current_word_idx == next_word_idx:
                end_label_mask[token_idx] = 0

        try:
            assert all(start_label_mask[p] != 0 for p in new_start_positions)
            assert all(end_label_mask[p] != 0 for p in new_end_positions)

            assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
            assert len(label_mask) == len(tokens)
            
        except:
            
            pass
            #print(data)
            

        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # last token is [SEP]
        sep_token = tokenizer.token_to_id("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)
            start_labels = self.pad(start_labels)
            end_labels = self.pad(end_labels)
            start_label_mask = self.pad(start_label_mask)
            end_label_mask = self.pad(end_label_mask)

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
            sample_idx,
            tag_idx
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def run_dataset():
    """test dataset"""
    import os
    from torch.utils.data import DataLoader

    bert_path = "mrc/for_mrc/rubert"
    dataset_path = "data/RuNNE_empty/dev.json"

    vocab_file = os.path.join(bert_path, "vocab.txt")
    
    tokenizer = BertWordPieceTokenizer(vocab_file, lowercase = False)
    dataset = MRCNERDataset(dataset_path=dataset_path, 
                            tokenizer=tokenizer,
                            max_length=192,
                            pad_to_maxlen=False,
                            tag = None # для тестирования по конкретным классам сущностей
                            )

    # if limit is not None:
    #    dataset = TruncateDataset(dataset, limit) 

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=4,
        persistent_workers = True,
        shuffle=False,
        collate_fn=collate_to_max_length
    )

    entities = 0

    for idx, batch in enumerate(dataloader):
        for tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, tag_idx in zip(*batch):
            tokens = tokens.tolist()
            start_positions, end_positions = torch.where(match_labels > 0)
            start_positions = start_positions.tolist()
            end_positions = end_positions.tolist()
            if not start_positions:
                continue
            print("="*20)
            print(f"len: {len(tokens)}", tokenizer.decode(tokens, skip_special_tokens=False))
            for start, end in zip(start_positions, end_positions):
                entities += 1
                print(str(idx) + "\t" + tokenizer.decode(tokens[start: end+1]))
    print("Total num: ", str(entities))


if __name__ == '__main__':
    run_dataset()
