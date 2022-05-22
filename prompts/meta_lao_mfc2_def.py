
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

from nltk.data import load
from nltk.tokenize import word_tokenize
import pymorphy2

# import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

from collections import Counter

morph = pymorphy2.MorphAnalyzer()
train_dataset_path = "data/RuNNE/train"

########################################################

jsonpath = "data/RuNNE_meta_lao_mfc2_def/test.json" # Здесь выбираем, куда будет сохраняться датасет, и под каким именем
dataset_path = "data/RuNNE/test" # Здесь указываем путь к каталогу с файлами, подготовленными через BRAT

#########################################################

jsondir = "/".join(jsonpath.split('/')[:-1])

if not os.path.exists(jsondir):
    os.makedirs(jsondir)

jsonfile = open(jsonpath, "w", encoding='UTF-8') 
ru_tokenizer = load("tokenizers/punkt/russian.pickle") # Загрузка токенизатора для русского языка

tags = [ 'AGE', 'AWARD', 'CITY', 'COUNTRY', 'CRIME', 'DATE', 'DISEASE', 'DISTRICT', 'EVENT', 'FACILITY', 
         'FAMILY', 'IDEOLOGY', 'LANGUAGE', 'LAW', 'LOCATION', 'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL', 
         'ORGANIZATION', 'PERCENT', 'PERSON', 'PENALTY', 'PRODUCT', 'PROFESSION', 'RELIGION', 'STATE_OR_PROVINCE', # 'OUT', 
         'TIME', 'WORK_OF_ART']

# Лист с записями как выше. Его заполним в json

all_entities = []

span_id = 0

for ad, dirs, files in os.walk(train_dataset_path):
    for f in files:

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

### lexical_all_outer

tag_to_spans = {tag : [] for tag in tags}
for entity in all_entities:
    tag_to_spans[entity["tag"]].append(entity)

final_map = {tag : [] for tag in tags}

for tag, entities in tag_to_spans.items():
    ent_cont = []
    for entity in entities:
        txtdata = entity["txtdata"]
        sentence_spans = ru_tokenizer.span_tokenize(txtdata)
        for span in sentence_spans:
            start, end = span
            context = txtdata[start : end]
            if entity["span"] in context and entity["start"] >= start and entity["end"] <= end:
                context = context[ : entity["start"] - start] + tag + context[entity["end"] - end : ]
                ent_cont.append((context, entity, start, end))
    tag_to_spans[tag] = ent_cont

    span_count = [v[1]["span"] for v in ent_cont]
    span_count = Counter(span_count)
    tag_to_spans[tag] = [(v[0], v[1], span_count[v[1]["span"]], v[2], v[3]) for v in tag_to_spans[tag]]
    tag_to_spans[tag] = sorted(tag_to_spans[tag], key = lambda x : x[2], reverse = True)
    # context, entity, span_count, start, end

    lex_context = tag_to_spans[tag][0][0]
    final_map[tag].append(lex_context)
    # print(f"{tag} : {tag_to_spans[tag]}")

####

#### most frequent components (2)

tag_to_spans = {tag : [] for tag in tags}
for entity in all_entities:
    tag_to_spans[entity["tag"]].extend(word_tokenize(entity["span"]))

for tag, entities in tag_to_spans.items():
    tag_to_spans[tag] = [morph.parse(word)[0].normal_form for word in entities if word not in russian_stopwords]
    tag_to_spans[tag] = Counter(tag_to_spans[tag]).most_common(2)
    tag_to_spans[tag] = [w for w, n in sorted(tag_to_spans[tag], key = lambda x : x[1], reverse = True)]  
    final_map[tag].append(f"{tag} - это сущности, такие как " + ", ".join(tag_to_spans[tag]) + ".")

####

#### definitions

tag_to_queries = {
     'MONEY': 'Деньги - это монеты или банкноты с указанием их стоимости, которые используются для покупки вещей, или их общая сумма.', 
     'CRIME': 'Преступление - это действие или деятельность, противоречащая закону, или незаконная деятельность в целом.', 
     'PENALTY': 'Наказание - мера воздействия против совершившего преступление или проступок.', 
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
     'WORK_OF_ART': 'Произведение искусства - это предмет, созданный творцом большого мастерства.',
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

for tag in final_map.keys():
    final_map[tag].append(tag_to_queries[tag])

####

entities = [] 

for ad, dirs, files in os.walk(dataset_path):
    for f in files:

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

                        for query in final_map[tag]:

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