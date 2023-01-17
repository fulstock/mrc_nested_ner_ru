
# Преобразовать данные из формата BRAT в единый json-формат файл, подходящий под задачу MRC.

# Формат данных будет представлен следующим образом.
# Одна запись соответствует одной именованной сущности, она содержит следующие ключи:
# {
#   id - идентификатор сущности (Строка вида "<Номер сущности в тексте>.<Номер вложенной подсущности>")
#   context - контекст, обычно предложение, к которому относится сущность
#   tag - имя класса сущности
#   query - запрос или определение, соответствующее имени класса сущности
#   filename - имя файла, откуда была получена сущность и контекст
#   exists - существует ли вообще в исходных данных сущность в этом контексте с этим классом (true / false)
#   start_positions - началА сущностей, если их несколько, в предложении
#   end_positions - концы сущностей, если их несколько, в предложении
#   span_positions - спаны (диапазоны) сущностей - пары "start;end"
#   spans - сами сущности (строкой)
# }

import json
import os
import argparse

from nltk.data import load
from nltk.tokenize import word_tokenize

from collections import Counter

from tqdm.auto import tqdm

ru_tokenizer = load("tokenizers/punkt/russian.pickle") # Загрузка токенизатора для русского языка

brat2mrc_parser = argparse.ArgumentParser(description = "Brat to mrc-json formatter script.")
brat2mrc_parser.add_argument('--brat_dataset_path', type = str, required = True, help = "Path to brat dataset (with train, dev, test dirs).")
brat2mrc_parser.add_argument('--train_dataset_path', type = str, default = None, help = "Path to train dataset, which is used for prompt generation. By default, train subset from --brat_dataset_path would be used.")
brat2mrc_parser.add_argument('--mrcjson_output_path', type = str, default = None, help = "Path, where formatted dataset would be stored. By default, same path as in --brat_dataset_path would be used.")
brat2mrc_parser.add_argument('--tags_file', type = str, required = True, help = "Path to <>.tags file with entity tags that would be processed.")

args = brat2mrc_parser.parse_args()

with open(args.tags_file, "r") as f:
    tags = json.load(f)

brat_dataset_path = args.brat_dataset_path

train_dataset_path = args.train_dataset_path
if train_dataset_path is None:
    train_dataset_path = os.path.join(brat_dataset_path, "train")

mrcjson_output_path = args.mrcjson_output_path
if mrcjson_output_path is None:
    mrcjson_output_path = brat_dataset_path

print("Train dataset parse for prompt:")

all_entities = []

for ad, dirs, files in os.walk(train_dataset_path):
    for f in tqdm(files):

        if f[-4:] == '.ann':
            try:

                if os.stat(train_dataset_path + '/' + f).st_size == 0:
                    continue

                annfile = open(train_dataset_path + '/' + f, "r", encoding='UTF-8')
                txtfile = open(train_dataset_path + '/' + f[:-4] + ".txt", "r", encoding='UTF-8')

                txtdata = txtfile.read()
                # txtdata = txtdata.replace('\n', '.', 1) # Отделение заголовков

                # Шаг 1. Считать все именованные сущности из файла, закрыть файл.

                file_entities = []

                # Именованная сущность пока что будет представленна укороченной записью. Позже она будет приведена к выду выше.

                for line in annfile:
                    line_tokens = line.split()
                    if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                        if line_tokens[1] in tags:
                            try:
                                file_entities.append( { 
                                                    "txtdata" : txtdata,
                                                    "tag" : line_tokens[1], 
                                                    "start" : int(line_tokens[2]),
                                                    "end" : int(line_tokens[3]),
                                                    "span" : txtdata[int(line_tokens[2]) : int(line_tokens[3])]
                                                    } )
                            except ValueError:
                                pass # Все неподходящие сущности

                annfile.close()

                all_entities.extend(file_entities)

            except FileNotFoundError:
                pass

tag_to_spans = {tag : [] for tag in tags}
for entity in all_entities:
    tag_to_spans[entity["tag"]].append(entity)

for tag, entities in tag_to_spans.items():
    ent_cont = []
    for entity in entities:
        txtdata = entity["txtdata"]
        sentence_spans = ru_tokenizer.span_tokenize(txtdata)
        for span in sentence_spans:
            start, end = span
            context = txtdata[start : end]
            if entity["span"] in context and entity["start"] >= start and entity["end"] <= end:
                # context = context[ : entity["start"] - start] + tag + context[entity["end"] - end : ]
                ent_cont.append((context, entity, start, end))
    tag_to_spans[tag] = ent_cont
    all_contexts = list(set([v[0] for v in tag_to_spans[tag]]))
    new_contexts = {}
    for context in all_contexts:
        context_entities = [(v[1], v[2], v[3]) for v in tag_to_spans[tag] if v[0] == context]
        context_entities = [(v[0], v[0]["start"] - v[1], v[0]["end"] - v[1]) for v in context_entities]

        outermost_entities = [v for v in context_entities if \
            not any([(s[1], s[2]) for s in context_entities if v is not s and v[1] >= s[1] and v[2] <= s[2]])]
        outermost_entities = sorted(outermost_entities, key = lambda x: x[1])

        #if len(outermost_entities) != len(context_entities):
        #    print(context)
        #    print([(v[1], v[2]) for v in context_entities])
        #    print([(v[1], v[2]) for v in outermost_entities])

        lex_context = ""
        rcontext = context
        offset = 0
        # print("---------")
        # print(context)
        # print([(v[1], v[2]) for v in outermost_entities])
        for entity in outermost_entities:
            span = entity[0]["span"]
            s, e = entity[1], entity[2]
            s += offset
            e += offset
            lex_context += rcontext[ : s] + "[" + span + " | " + tag + "]"
            rcontext = context[e - offset : ]
            offset = len(rcontext) - len(context)
            # print(s, e)
            # print(rcontext)
            # print(offset)
            # print(lex_context)
        lex_context += rcontext
        # print(lex_context)
        
        new_contexts[context] = lex_context

    tag_to_spans[tag] = [(new_contexts[t[0]], t[1], t[2], t[3]) for t in tag_to_spans[tag]]

    span_count = [v[1]["span"] for v in ent_cont]
    span_count = Counter(span_count)
    tag_to_spans[tag] = [(v[0], v[1], span_count[v[1]["span"]], v[2], v[3]) for v in tag_to_spans[tag]]
    tag_to_spans[tag] = sorted(tag_to_spans[tag], key = lambda x : x[2], reverse = True)

    lex_context = tag_to_spans[tag][0][0]
    tag_to_spans[tag] = lex_context
    # print(f"{tag} : {tag_to_spans[tag]}")    

sets = ["train", "dev", "test"]

for ds in sets:

    print(ds + " set:")

    jsonpath = os.path.join(mrcjson_output_path, ds + ".json") 
    dataset_path = os.path.join(brat_dataset_path, ds)

    jsondir = os.path.dirname(jsonpath)

    if not os.path.exists(jsondir):
        os.makedirs(jsondir)

    jsonfile = open(jsonpath, "w", encoding='UTF-8')     

    span_id = 0
    entities = []

    for ad, dirs, files in os.walk(dataset_path):
        for f in tqdm(files):

            if f[-4:] == '.ann':
                try:

                    if os.stat(dataset_path + '/' + f).st_size == 0:
                        continue

                    annfile = open(dataset_path + '/' + f, "r", encoding='UTF-8')
                    txtfile = open(dataset_path + '/' + f[:-4] + ".txt", "r", encoding='UTF-8')

                    txtdata = txtfile.read()
                    # txtdata = txtdata.replace('\n', '.', 1) # Отделение заголовков

                    # Шаг 1. Считать все именованные сущности из файла, закрыть файл.

                    file_entities = []

                    # Именованная сущность пока что будет представленна укороченной записью. Позже она будет приведена к выду выше.

                    for line in annfile:
                        line_tokens = line.split()
                        if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                            try:
                                file_entities.append( { "tag" : line_tokens[1], 
                                                   "start" : int(line_tokens[2]),
                                                   "end" : int(line_tokens[3]),
                                                  } )
                            except ValueError:
                                pass # Все неподходящие сущности

                    annfile.close()

                    # Шаг 2. В каждом файле выделить контексты отдельно друг от друга.

                    sentence_spans = ru_tokenizer.span_tokenize(txtdata)
                    for span in sentence_spans:

                        start, end = span
                        context = txtdata[start : end]

                        sentence_entities = [e for e in file_entities if e["start"] >= start and e["end"] <= end]
                        sentence_tags = [e["tag"] for e in sentence_entities]

                        # Шаг 3. Для каждого существующего класса генерируем пример. Если такой есть, то добавляем, иначе оставляем пустым.

                        for tag_id, tag in enumerate(tags):

                            query = tag_to_spans[tag]

                            if tag in sentence_tags:

                                start_positions = [e["start"] - start for e in sentence_entities if e["tag"] == tag]
                                end_positions = [e["end"] - start for e in sentence_entities if e["tag"] == tag]

                                entity = { "id" : "{}.{}".format(span_id, tag_id), 
                                           "context" : context,
                                           "tag" : tag,
                                           "query" : query,
                                           "filename" : f[:-4],
                                           "exists" : True,
                                           "start_positions" : start_positions,
                                           "end_positions" : end_positions, 
                                           "span_positions" : ["{};{}".format(start, end) for start, end in zip(start_positions, end_positions)],
                                           "spans" : [context[start : end] for start, end in zip(start_positions, end_positions)]
                                }

                            else:

                                entity = { "id" : "{}.{}".format(span_id, tag_id), 
                                           "context" : context,
                                           "tag" : tag,
                                           "query" : query,
                                           "filename" : f[:-4],
                                           "exists" : False,
                                           "start_positions" : [],
                                           "end_positions" : [], 
                                           "span_positions" : [],
                                           "spans" : []
                                }
                            
                            entities.append(entity)

                        span_id += 1 # Перейти к следующему предложению

                    txtfile.close()

                except FileNotFoundError:
                    pass

    # Шаг 4. Сохранить все сущности в json формат.

    print(f"{len(entities)} entities from {dataset_path} jsoned to {jsonpath}.")

    json.dump(entities, jsonfile, ensure_ascii = False, indent = 2)

    jsonfile.close()