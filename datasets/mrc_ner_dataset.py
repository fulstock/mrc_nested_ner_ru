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

    return output


class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """
    def __init__(self, dataset_path, tokenizer: BertWordPieceTokenizer, max_length: int = 128, pad_to_maxlen=False, tag = None):
        self.all_data = self.load_dataset(dataset_path, tag_filter = tag)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_maxlen = pad_to_maxlen

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
        sample = self.all_data[item]
        tokenizer = self.tokenizer

        '''qas_id = data.get("qas_id", "0.0")
                                sample_idx, label_idx = qas_id.split(".")
                                sample_idx = torch.LongTensor([int(sample_idx)])
                                label_idx = torch.LongTensor([int(label_idx)])'''

        query, _, context = sample
        start_positions = [s for (s, e) in sample[1]]
        end_positions = [e for (s, e) in sample[1]]

        # add space offsets
        words = context.split()
        start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
        end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        # print(query, end = '\n\n')
        # print(context, end = '\n\n')

       #  print(start_positions, end = '\n\n')
        # print(end_positions, end = '\n\n')

        query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True) #.encodings[0]
        tokens = query_context_tokens.ids
        type_ids = query_context_tokens.type_ids
        offsets = query_context_tokens.offsets

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

        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]

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

        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
        assert len(label_mask) == len(tokens)
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
            # sample_idx,
            query
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst

    def load_dataset(self, dataset_path, tag_filter = None):

        nltk.download('stopwords')
        ru_stopwords = stopwords.words("russian")

        named_entities = []
        sentences = []
        tags = set ()

        for ad, dirs, files in os.walk(dataset_path):
            for f in files:
                if f[-4:] == '.ann':
                    try: 
                        annfile = open(dataset_path + '/' + f, "r", encoding='UTF-8')
                        txtfile = open(dataset_path + '/' + f[:-4] + ".txt", "r", encoding='UTF-8')

                        file_data = txtfile.read()

                        file_named_entities = []
                        for line in annfile:
                            line_tokens = line.split()
                            if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                                try:
                                    tags.add(line_tokens[1])
                                    start_char_index = int (line_tokens[2])
                                    end_char_index = int (line_tokens[3])
                                    named_entity = file_data[start_char_index : end_char_index]
                                    # print(named_entity, line_tokens[1], sep=' : ', end = '\n')
                                    file_named_entities.append((named_entity, start_char_index, end_char_index, line_tokens[1]))
                                except ValueError:
                                    pass # пока игнорирую эти (;) разделенные сущности (в статье их вообще не рассматривают)
                        
                        file_sents = sent_tokenize(file_data, language="russian")
                        file_sents = [(sent, file_data.index(sent), file_data.index(sent) + len(sent)) for sent in file_sents if len(sent.split()) > 2] 

                        annfile.close()
                        txtfile.close()

                        named_entities.append(file_named_entities)
                        sentences.append(file_sents)
                    except FileNotFoundError:
                        pass

        tags.add('OUT')

        tags_ord = OrderedDict()

        for idx, tag in enumerate(tags):
            tags_ord[tag] = idx

        tags_descripted = {
            ('AGE', 'Возраст - это период времени, когда кто-то был жив или что-то существовало.'),
            ('AWARD', 'Награда - это приз или денежная сумма, которую человек или организация получают за достижение.'),
            ('CHARGE', 'Обвинение - это официальное заявление полиции о том, что кто-то обвиняется в преступлении.'),
            ('CITY','Город - это место, где живет много людей, со множеством домов, магазинов, предприятий и т.д.'),
            ('COUNTRY','Страна - это территория земли, на которой есть собственное правительство, армия и т.д.'),
            ('CRIME', 'Преступление - это действие или деятельность, противоречащая закону, или незаконная деятельность в целом.'),
            ('DATE','Дата - это номер дня в месяце, часто указываемый в сочетании с названием дня, месяца и года.'),
            ('DISEASE','Болезнь - это заболевание людей, животных, растений и т.д., вызванное инфекцией или нарушением здоровья.'),
            ('DISTRICT','Район - это территория страны или города, имеющая фиксированные границы, которые используются для официальных целей, или имеющая особую особенность, которая отличает его от окружающих территорий.'),
            ('EVENT', 'Событие - это все, что происходит, особенно что-то важное или необычное.'),
            ('FACILITY', 'Объект - это место, включая здания, где происходит определенная деятельность.'),
            ('FAMILY', 'Семья - это группа людей, связанных друг с другом, таких как мать, отец и их дети.'),
            ('IDEOLOGY', 'Идеология - это набор убеждений или принципов, особенно тех, на которых основана политическая система, партия или организация.'),
            ('LANGUAGE', 'Язык - это система общения, используемая людьми, живущими в определенной стране.'),
            ('LAW', 'Закон - это правило, обычно устанавливаемое правительством, которое используется для упорядочивания поведения общества.'),
            ('LOCATION', 'Местоположение - это место или позиция.'),
            ('MONEY', 'Деньги - это монеты или банкноты с указанием их стоимости, которые используются для покупки вещей, или их общая сумма.'),
            ('NATIONALITY', 'Национальность - это группа людей одной расы, религии, традиций и т.д.'),
            ('NUMBER', 'Число - это знак или символ, представляющий единицу, которая является частью системы подсчета и расчета.'),
            ('ORDINAL', 'Порядковый номер - это число, которое показывает положение чего-либо в списке.'),
            ('ORGANIZATION', 'Организация - это компания или другая группа людей, которые работают вместе для определенной цели.'),
            ('PERCENT', 'Процент - это одна часть из каждых 100 или указанное количество чего-либо, деленное на 100.'),
            ('OUT', '[UNK]'),
            ('PERSON', 'Человек - мужчина, женщина или ребенок.'),
            ('PENALTY', 'Штраф - это вид наказания, часто включающий выплату денег, который назначается вам, если вы нарушите соглашение или не соблюдаете правила.'),
            ('PRODUCT', 'Продукт - это то, что предназначено для продажи, особенно что-то произведенное в результате промышленного процесса или что-то выращенное в сельском хозяйстве.'),
            ('PROFESSION', 'Профессия - это любой вид работы, требующий специальной подготовки или определенных навыков.'),
            ('RELIGION', 'Религия - это вера и поклонение богу или богам или любая подобная система веры и поклонения.'),
            ('STATE_OR_PROVINCE', 'Штат или провинция - одна из областей, на которые страна или империя делятся в рамках организации своего правительства, которое часто имеет некоторый контроль над своими собственными законами.'),
            ('TIME', 'Время - это часть существования, которая измеряется минутами, днями, годами и т.д., или его процесс, рассматриваемый как единое целое.'),
            ('WORK_OF_ART', 'Произведение искусства - это предмет, созданный творцом большого мастерства, особенно картина, рисунок или статуя.')
        }

        queries = {}
        for tag, desc in tags_descripted:
            queries[tag] = desc

        sent_triplets = []

        for file_idx, file_sents in enumerate(sentences):
            for sent, sent_start, sent_end in file_sents:
                sent = re.sub(r'[^a-zA-Zа-яА-ЯёЁйЙ0-9_-]', ' ', sent)
                sent_tokens = sent.strip().split()
                start_idx = sent_start
                end_idx = sent_start
                token_spans = []
                for idx, token in enumerate(sent_tokens):
                    try:
                        start_idx = sent.index(token, end_idx - sent_start) + sent_start
                        end_idx = start_idx + len(token)
                        token_spans.append((idx, start_idx, end_idx))
                    except ValueError:
                        pass # такая же ошибка как ниже
                triplets = []
                for entity, ent_start, ent_end, ent_label in named_entities[file_idx]:
                    if ent_start >= sent_start and ent_end <= sent_end:
                        entity_tokens = [token_id for token_id, start_idx, end_idx in token_spans 
                                             if start_idx >= ent_start and end_idx <= ent_end]
                        try:
                            sent = re.sub(r' +', ' ', sent).strip()
                            triplet = (queries[ent_label], (entity_tokens[0], entity_tokens[-1]), sent)
                            triplets.append(triplet)
                        except IndexError:
                            pass # очень особенные редкие случаи плохой токенизации, пока игнорирую
                sent_triplets.append(triplets)

        samples = []

        for sentence in sent_triplets:
            if len(sentence) >= 1:
                queries_sent = set([q for q, (s, e), c in sentence])
                context = sentence[0][-1]
                for query in queries_sent:
                    sample = [(s, e) for q, (s, e), c in sentence if q == query]
                    samples.append((query, sample, context))

        if tag_filter:
            query = queries[tag_filter]
            samples = [(query, sample, context) for query_i, sample, context in samples if query == query_i]

        return samples


def run_dataset():
    """test dataset"""
    import os
    from torch.utils.data import DataLoader

    bert_path = "../rubert"
    dataset_path = "../NEREL_01/dev"

    vocab_file = os.path.join(bert_path, "vocab.txt")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_path, do_lower_case = False)
    dataset = MRCNERDataset(dataset_path = dataset_path, tokenizer = tokenizer)

    dataloader = DataLoader(dataset, batch_size=4, 
                             collate_fn=collate_to_max_length)

    for idx, batch in enumerate(dataloader):
        for tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels in zip(*batch):
            tokens = tokens.tolist()
            start_positions, end_positions = torch.where(match_labels > 0)
            start_positions = start_positions.tolist()
            end_positions = end_positions.tolist()
            if not start_positions:
                continue
            print("="*20)
            print(f"len: {len(tokens)}", tokenizer.decode(tokens, skip_special_tokens=False))
            for start, end in zip(start_positions, end_positions):
                print(str(idx) + "\t" + tokenizer.decode(tokens[start: end+1]))


if __name__ == '__main__':
    run_dataset()
