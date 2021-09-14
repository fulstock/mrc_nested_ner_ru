


import os
import torch
from torch.utils.data import DataLoader
from models.bert_labeling import BertLabeling
from tokenizers import BertWordPieceTokenizer
from datasets.mrc_ner_dataset import MRCNERDataset
import numpy as np

import json
import tqdm
import time

DATA_PATH = "jsons"
BERT_PATH = "for_mrc/rubert"
CHECKPOINTS = "model_23_08-16_58.ckpt"
DATASET_PATH = "twits.json"

dataset_path = os.path.join(DATA_PATH, DATASET_PATH)
vocab_path = os.path.join(BERT_PATH, "vocab.txt")
data_tokenizer = BertWordPieceTokenizer(vocab_path, lowercase = False)

dataset = MRCNERDataset(dataset_path=dataset_path,
                        tokenizer=data_tokenizer,
                        max_length=128,
                        pad_to_maxlen=False,
                        tag = None)

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

model_args = {
    "data_dir" : DATA_PATH ,
    "bert_config_dir" : BERT_PATH ,
    "pretrained_checkpoint" : CHECKPOINTS ,
    "max_length" : 128 , 
    "batch_size" : 1 , 
    "lr" : 2e-5 , 
    "workers" : 8 , 
    "weight_decay" : 0.01 , 
    "warmup_steps" : 0 , 
    "adam_epsilon" : 1e-8 , # Epsilon для алгоритма ADAMW
    "mrc_dropout" : 0.1 , # Dropout вероятность в модели MRC
    "weight_start" : 1.0 , # Коэффициент для стартовых позиций меток (альфа)
    "weight_end" : 1.0 , # Коэффициент для конечных позиций меток (бета)
    "weight_span" : 1.0 , # Коэффициент для спанов меток (гамма)
    "loss_type" : "bce" , 
    "optimizer" : "adamw" ,
    "dice_smooth" : 1e-8 ,
    "final_div_factor" : 1e4 ,
    "span_loss_candidates" : "all" , 
    "accumulate_grad_batches" : 1
}

bert_args = {
    "bert_dropout" : 0.1 # Dropout самого берта
}

trainer_args = {
    "default_root_dir" : "logs" , # Куда сохранять модели, логи и т.д.
    "max_epochs" : 16 , # Число эпох для обучения
    "resume_from_checkpoint" : None , # Воспроизвести обучение с чекпойнта
    # "val_check_interval" : 1.0 , # Как часто валидировать модель
    "gpus" : 1 # , # число используемых видеокарт
    # Добавьте любые нужные аргументы для конфигурации модели. Их можно найти на 
    # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
}

trained_mrc_ner_model = BertLabeling.load_from_checkpoint(
        # model_args = model_args,
        bert_args = bert_args,
        trainer_args = trainer_args,
        checkpoint_path=CHECKPOINTS,
        map_location=None,
        # batch_size=1,
        # max_length=128,
        # bert_config_dir = "for_mrc/rubert",
        # data_dir = "jsons",
        # mrc_dropout = 0.1,
        # loss_type = 
        # workers=0,
        **model_args
    )

tags = [ 'AGE', 'AWARD', 'CITY', 'COUNTRY', 'CHARGE', 'CRIME', 'DATE', 'DISEASE', 'DISTRICT', 'EVENT', 'FACILITY', 
         'FAMILY', 'IDEOLOGY', 'LANGUAGE', 'LAW', 'LOCATION', 'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL', 
         'ORGANIZATION', 'PERCENT', 'PERSON', 'PENALTY', 'PRODUCT', 'PROFESSION', 'RELIGION', 'STATE_OR_PROVINCE', # 'OUT', 
         'TIME', 'WORK_OF_ART' ]

tag_to_queries = {
     'MONEY': 'Деньги - это монеты или банкноты с указанием их стоимости, которые используются для покупки вещей, или их общая сумма.', 
     'CRIME': 'Преступление - это действие или деятельность, противоречащая закону, или незаконная деятельность в целом.', 
     'PENALTY': 'Штраф - это вид наказания, часто включающий выплату денег, который назначается вам, если вы нарушите соглашение или не соблюдаете правила.', 
     'STATE_OR_PROVINCE': 'Штат или провинция - одна из областей, на которые страна или империя делятся в рамках организации своего правительства, которое часто имеет некоторый контроль над своими собственными законами.', 
     # 'OUT': '[UNK]', 
     'ORGANIZATION': 'Организация - это компания или другая группа людей, которые работают вместе для определенной цели.', 
     'PRODUCT': 'Продукт - это то, что предназначено для продажи, особенно что-то произведенное в результате промышленного процесса или что-то выращенное в сельском хозяйстве.',
     'NATIONALITY': 'Национальность - это группа людей одной расы, религии, традиций и т.д.',
     'RELIGION': 'Религия - это вера и поклонение богу или богам или любая подобная система веры и поклонения.',
     'DISTRICT': 'Район - это территория страны или города, имеющая фиксированные границы, которые используются для официальных целей, или имеющая особую особенность, которая отличает его от окружающих территорий.',
     'PROFESSION': 'Профессия - это любой вид работы, требующий специальной подготовки или определенных навыков.', 
     'LANGUAGE': 'Язык - это система общения, используемая людьми, живущими в определенной стране.',
     'PERSON': 'Человек - мужчина, женщина или ребенок.',
     'DATE': 'Дата - это номер дня в месяце, часто указываемый в сочетании с названием дня, месяца и года.',
     'WORK_OF_ART': 'Произведение искусства - это предмет, созданный творцом большого мастерства, особенно картина, рисунок или статуя.',
     'TIME': 'Время - это часть существования, которая измеряется минутами, днями, годами и т.д., или его процесс, рассматриваемый как единое целое.',
     'AWARD': 'Награда - это приз или денежная сумма, которую человек или организация получают за достижение.',
     'FACILITY': 'Объект - это место, включая здания, где происходит определенная деятельность.',
     'ORDINAL': 'Порядковый номер - это число, которое показывает положение чего-либо в списке.',
     'DISEASE': 'Болезнь - это заболевание людей, животных, растений и т.д., вызванное инфекцией или нарушением здоровья.',
     'IDEOLOGY': 'Идеология - это набор убеждений или принципов, особенно тех, на которых основана политическая система, партия или организация.',
     'NUMBER': 'Число - это знак или символ, представляющий единицу, которая является частью системы подсчета и расчета.',
     'EVENT': 'Событие - это все, что происходит, особенно что-то важное или необычное.',
     'CHARGE': 'Обвинение - это официальное заявление полиции о том, что кто-то обвиняется в преступлении.',
     'AGE': 'Возраст - это период времени, когда кто-то был жив или что-то существовало.',
     'LOCATION': 'Местоположение - это место или позиция.',
     'COUNTRY': 'Страна - это территория земли, на которой есть собственное правительство, армия и т.д.',
     'PERCENT': 'Процент - это одна часть из каждых 100 или указанное количество чего-либо, деленное на 100.',
     'FAMILY': 'Семья - это группа людей, связанных друг с другом, таких как мать, отец и их дети.',
     'CITY': 'Город - это место, где живет много людей, со множеством домов, магазинов, предприятий и т.д.',
     'LAW': 'Закон - это правило, обычно устанавливаемое правительством, которое используется для упорядочивания поведения общества.'
}

def pretty_print(string, length, symbol, file):
     print (symbol *((length - len(string)) // 2) + string + symbol *((length - len(string)) // 2), file = file)

example_num = 0

trained_mrc_ner_model.eval()

f = open("test_dataset.out", "w", encoding = "utf-8")
inf_file = open("inference/twits_0.json", "w", encoding="utf-8")
file_index = 0

entities = []

deltatime = time.time()

for bidx, batch in enumerate(tqdm.tqdm(dataloader)):

    tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, tag_idx = batch
    attention_mask = (tokens != 0).long()

    with torch.no_grad():
        start_logits, end_logits, match_logits = trained_mrc_ner_model(tokens, attention_mask, token_type_ids)

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
    # match_labels = match_labels[example_num]

    start_positions, end_positions = torch.where(match_preds == True)
    # start_label_positions, end_label_positions = torch.where(match_labels > 0)

    start_positions = start_positions.tolist()
    end_positions = end_positions.tolist()
    # start_label_positions = start_label_positions.tolist()
    # end_label_positions = end_label_positions.tolist()

    tag_idx = int(tag_idx)

    print("="*45, file = f)
    context = data_tokenizer.decode(input_ids, skip_special_tokens=False)
    print(context, file = f)
    pretty_print("Предсказанные сущности", 45, '-', f)
    if start_positions:
        for pos_idx, (start, end) in enumerate(zip(start_positions, end_positions)):
            decoded = data_tokenizer.decode(input_ids[start: end+1])
            print(decoded, file = f)

        entity = { "id" : "{}.{}".format(sample_idx, tag_idx), 
                   "context" : context,
                   "tag" : tags[tag_idx],
                   "query" : tag_to_queries[tags[tag_idx]],
                   "filename" : "",
                   "exists" : True,
                   "start_positions" : start_positions,
                   "end_positions" : end_positions, 
                   "span_positions" : ["{};{}".format(start, end) for start, end in zip(start_positions, end_positions)],
                   "spans" : [data_tokenizer.decode(input_ids[start: end+1]) for start, end in zip(start_positions, end_positions)]
        }


    else:
        print("<Ни одной сущности не найдено.>", file = f)

        entity = { "id" : "{}.{}".format(sample_idx, tag_idx), 
                   "context" : context,
                   "tag" : tags[tag_idx],
                   "query" : tag_to_queries[tags[tag_idx]],
                   "filename" : "",
                   "exists" : False,
                   "start_positions" : [],
                   "end_positions" : [], 
                   "span_positions" : [],
                   "spans" : []
        }

    print("-"*45, file = f)

    entities.append(entity)

    if bidx % 55 == 0:

        recent_time = time.time()
        if recent_time - deltatime > 120:

            json.dump(entities, inf_file, ensure_ascii = False, indent = 2)
            print("Saved total of {} entities to inference/{}_{}.json}".format(len(entities), DATASET_PATH[:-5], file_index))
            entities = []

            inf_file.close()
            f.close()

            file_index += 1

            f = open("test_dataset.out", "a", encoding = "utf-8")
            inf_file = open("inference/twits_{}.json".format(file_index), "w", encoding="utf-8")

            deltatime = recent_time

print("="*45, file = f)

json.dump(entities, inf_file, ensure_ascii = False, indent = 2)

inf_file.close()
f.close()