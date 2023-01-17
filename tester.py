import torch
from pytorch_lightning import Trainer

import json
import os

from torch.utils.data import DataLoader
from datasets.mrc_ner_dataset import MRCNERDataset, collate_to_max_length

from trainer import *
from utils.get_parser import get_parser
from utils.random_seed import set_random_seed
# from utils.progress import TGProgressBar

# set_random_seed(0)

def get_dataloader(model, dataset_path, tag = None):

    dataset = MRCNERDataset(dataset_path=dataset_path, 
                            tokenizer=model.tokenizer,
                            max_length=192,
                            pad_to_maxlen=False,
                            tag = tag # для тестирования по конкретным классам сущностей
                            )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=model.args.batch_size,
        num_workers=model.args.workers,
        persistent_workers = True,
        shuffle=False,
        collate_fn=collate_to_max_length
    )

    return dataloader

def test(tag_test = None):

    parser = get_parser() # Создание парсера командной строки в общем виде (см. get_parser)

    # Добавление аргументов командной строки, отвечающих самой модели
    parser = BertLabeling.add_model_specific_args(parser) 

    # Добавление всех возможных флагов Trainer (--gpus, --num_nodes и т.д.) из командной строки
    parser = Trainer.add_argparse_args(parser) 

    parser.add_argument('--result_dirpath', type = str, default = "./ckpt", help = "Directory path, where result would be saved. Default: \"ckpt\".")

    # Помощь по всем флагам командой строки - либо через -h / --help, либо (если указано pl.Trainer)
    # см. документацию по Trainer от Pytorch Lightning

    # Сохраняем все аргументы из командной строки
    args = parser.parse_args()

    result_dirpath = args.result_dirpath

    seed_everything(args.seed, workers=True)

    # bar = TGProgressBar()

    ckpt_model = BertLabeling.load_from_checkpoint(args.pretrained_checkpoint) # Инициализиуем модель на их основе

    trainer = Trainer.from_argparse_args(
        args,
        logger = False,
        deterministic = True #,
        # callbacks = [bar]
    )

    # trainer.test(ckpt_model, ckpt_model.get_dataloader("test", tag = tag_test))

    # tag_classes = ['AGE', 'AWARD', 'CITY', 'COUNTRY', 'CRIME', 'DATE', 'DISEASE', 'DISTRICT', 'EVENT', 'FACILITY', 
    #    'FAMILY', 'IDEOLOGY', 'LANGUAGE', 'LAW', 'LOCATION', 'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL', 
    #    'ORGANIZATION', 'PERCENT', 'PERSON', 'PENALTY', 'PRODUCT', 'PROFESSION', 'RELIGION', 'STATE_OR_PROVINCE', # 'OUT', 
    #    'TIME', 'WORK_OF_ART'] 

    tag_classes = sorted(["AGE", "CITY", "COUNTRY", "DATE", "DISEASE", "FACILITY", "LOCATION", "NUMBER", "ORDINAL", "ORGANIZATION", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "STATE_OR_PROVINCE", "TIME"])

    score_dict = dict()
    for tag in tag_classes:
        print(tag + ":")
        score_dict[tag] = trainer.test(ckpt_model, get_dataloader(model = ckpt_model, dataset_path = os.path.join(args.data_dir, "test.json"), tag = tag))
        print(tag + ":" + str(get_dataloader(model = ckpt_model, dataset_path = os.path.join(args.data_dir, "test.json"), tag = tag).__len__()))

    score_dict = {tag : {score : value for score, value in scores[0].items() if score in ["span_f1", "span_precision", "span_recall", 
    "span_tp", "span_fp", "span_fn"]} for tag, scores in score_dict.items()}

    with open(result_dirpath + "/" + args.data_dir.split('/')[-1] + "-score_" + 
            args.pretrained_checkpoint.split('/')[-1] + ".json", "w", encoding = "UTF-8") as output:
        json.dump(score_dict, output, ensure_ascii = False, indent = 2, sort_keys = True)

    fscores = [s["span_f1"] for s in score_dict.values()]
    prscores = [s["span_precision"] for s in score_dict.values()]
    rscores = [s["span_recall"] for s in score_dict.values()]

    tp_scores = [s["span_tp"] for s in score_dict.values()]
    fp_scores = [s["span_fp"] for s in score_dict.values()]
    fn_scores = [s["span_fn"] for s in score_dict.values()]
    micro_scores = list(zip(tp_scores, fp_scores, fn_scores))

    import sys

    print(f"Macro precision = {sum(prscores) / len(prscores)}", file = sys.stderr)
    print(f"Macro recall = {sum(rscores) / len(rscores)}", file = sys.stderr)
    print(f"Macro F1 = {sum(fscores) / len(fscores)}", file = sys.stderr)
    print("-----------")
    micro_prec = sum(tp_scores) / (sum(tp_scores) + sum(fn_scores) + 1e-10)
    micro_rec = sum(tp_scores) / (sum(tp_scores) + sum(fp_scores) + 1e-10)
    print(f"Micro precision = {micro_prec}", file = sys.stderr)
    print(f"Micro recall = {micro_rec}", file = sys.stderr)
    print(f"Micro F1 = {micro_prec * micro_rec * 2 / (micro_prec + micro_rec + 1e-10)}", file = sys.stderr)

    print('Расчет окончен!')

    # fscores = [s["span_f1"] for s in score_dict.values()]
    # prscores = [s["span_precision"] for s in score_dict.values()]
    # rscores = [s["span_recall"] for s in score_dict.values()]

    # fewshot_fscores = [s["span_f1"] for tag, s in score_dict.items() if tag in ['DISEASE', 'PENALTY', 'WORK_OF_ART']]
    # fewshot_prscores = [s["span_precision"] for tag, s in score_dict.items() if tag in ['DISEASE', 'PENALTY', 'WORK_OF_ART']]
    # fewshot_rscores = [s["span_recall"] for tag, s in score_dict.items() if tag in ['DISEASE', 'PENALTY', 'WORK_OF_ART']]

    # import sys

    # print(f"Macro precision = {sum(prscores) / len(prscores)}", file = sys.stderr)
    # print(f"Few-shot precision = {sum(fewshot_prscores) / len(fewshot_prscores)}", file = sys.stderr)
    # print(f"Macro recall = {sum(rscores) / len(rscores)}", file = sys.stderr)
    # print(f"Few-shot recall = {sum(fewshot_rscores) / len(fewshot_rscores)}", file = sys.stderr)
    # print(f"Macro F1 = {sum(fscores) / len(fscores)}", file = sys.stderr)
    # print(f"Few-shot F1 = {sum(fewshot_fscores) / len(fewshot_fscores)}", file = sys.stderr)

    # print('Расчет окончен!')

def test_bio(tag_test = None):

    parser = get_parser() # Создание парсера командной строки в общем виде (см. get_parser)

    # Добавление аргументов командной строки, отвечающих самой модели
    parser = BertLabeling.add_model_specific_args(parser) 

    # Добавление всех возможных флагов Trainer (--gpus, --num_nodes и т.д.) из командной строки
    parser = Trainer.add_argparse_args(parser) 

    parser.add_argument('--result_dirpath', type = str, default = "./ckpt", help = "Directory path, where result would be saved. Default: \"ckpt\".")

    # Помощь по всем флагам командой строки - либо через -h / --help, либо (если указано pl.Trainer)
    # см. документацию по Trainer от Pytorch Lightning

    # Сохраняем все аргументы из командной строки
    args = parser.parse_args()

    result_dirpath = args.result_dirpath

    seed_everything(args.seed, workers=True)

    # bar = TGProgressBar()

    ckpt_model = BertLabeling.load_from_checkpoint(args.pretrained_checkpoint) # Инициализиуем модель на их основе

    trainer = Trainer.from_argparse_args(
        args,
        logger = False,
        deterministic = True #,
        # callbacks = [bar]
    )

    # trainer.test(ckpt_model, ckpt_model.get_dataloader("test", tag = tag_test))

    tag_classes = ['ACTIVITY', 'ADMINISTRATION_ROUTE', 'AGE', 'ANATOMY', 'CHEM', 'CITY', 'COUNTRY', 'DATE', 'DEVICE', 
            'DISO', 'FACILITY', 'FINDING', 'FOOD', 'GENE', 'HEALTH_CARE_ACTIVITY', 'INJURY_POISONING', 'LABPROC', 
            'LIVB', 'LOCATION', 'MEDPROC', 'MENTALPROC', 'NUMBER', 'ORDINAL', 'ORGANIZATION', 'PERCENT', 'PERSON', 
            'PHYS', 'PRODUCT', 'PROFESSION', 'SCIPROC', 'STATE_OR_PROVINCE', 'TIME']

    # tag_classes = sorted(["AGE", "CITY", "COUNTRY", "DATE", "DISO", "FACILITY", "LOCATION", "NUMBER", "ORDINAL", "ORGANIZATION", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "STATE_OR_PROVINCE", "TIME"])

    score_dict = dict()
    for tag in tag_classes:
        print(tag + ":")
        score_dict[tag] = trainer.test(ckpt_model, get_dataloader(model = ckpt_model, dataset_path = os.path.join(args.data_dir, "test.json"), tag = tag))
        print(tag + ":" + str(get_dataloader(model = ckpt_model, dataset_path = os.path.join(args.data_dir, "test.json"), tag = tag).__len__()))

    score_dict = {tag : {score : value for score, value in scores[0].items() if score in ["span_f1", "span_precision", "span_recall", 
    "span_tp", "span_fp", "span_fn"]} for tag, scores in score_dict.items()}

    with open(result_dirpath + "/" + args.data_dir.split('/')[-1] + "-score_" + 
            args.pretrained_checkpoint.split('/')[-1] + ".json", "w", encoding = "UTF-8") as output:
        json.dump(score_dict, output, ensure_ascii = False, indent = 2, sort_keys = True)

    fscores = [s["span_f1"] for s in score_dict.values()]
    prscores = [s["span_precision"] for s in score_dict.values()]
    rscores = [s["span_recall"] for s in score_dict.values()]
    
    tp_scores = [s["span_tp"] for s in score_dict.values()]
    fp_scores = [s["span_fp"] for s in score_dict.values()]
    fn_scores = [s["span_fn"] for s in score_dict.values()]
    micro_scores = list(zip(tp_scores, fp_scores, fn_scores))

    import sys

    print(f"Macro precision = {sum(prscores) / len(prscores)}", file = sys.stderr)
    print(f"Macro recall = {sum(rscores) / len(rscores)}", file = sys.stderr)
    print(f"Macro F1 = {sum(fscores) / len(fscores)}", file = sys.stderr)
    print("-----------")
    micro_prec = sum(tp_scores) / (sum(tp_scores) + sum(fn_scores) + 1e-10)
    micro_rec = sum(tp_scores) / (sum(tp_scores) + sum(fp_scores) + 1e-10)
    print(f"Micro precision = {micro_prec}", file = sys.stderr)
    print(f"Micro recall = {micro_rec}", file = sys.stderr)
    print(f"Micro F1 = {micro_prec * micro_rec * 2 / (micro_prec + micro_rec + 1e-10)}", file = sys.stderr)

    print('Расчет окончен!')


def test_bio_fix(tag_test = None):

    parser = get_parser() # Создание парсера командной строки в общем виде (см. get_parser)

    # Добавление аргументов командной строки, отвечающих самой модели
    parser = BertLabeling.add_model_specific_args(parser) 

    # Добавление всех возможных флагов Trainer (--gpus, --num_nodes и т.д.) из командной строки
    parser = Trainer.add_argparse_args(parser) 

    parser.add_argument('--result_dirpath', type = str, default = "./ckpt", help = "Directory path, where result would be saved. Default: \"ckpt\".")

    # Помощь по всем флагам командой строки - либо через -h / --help, либо (если указано pl.Trainer)
    # см. документацию по Trainer от Pytorch Lightning

    # Сохраняем все аргументы из командной строки
    args = parser.parse_args()

    result_dirpath = args.result_dirpath

    seed_everything(args.seed, workers=True)

    # bar = TGProgressBar()

    ckpt_model = BertLabeling.load_from_checkpoint(args.pretrained_checkpoint) # Инициализиуем модель на их основе

    trainer = Trainer.from_argparse_args(
        args,
        logger = False,
        deterministic = True #,
        # callbacks = [bar]
    )

    # trainer.test(ckpt_model, ckpt_model.get_dataloader("test", tag = tag_test))

    tag_classes = ['ACTIVITY', 'ADMINISTRATION_ROUTE', 'AGE', 'ANATOMY', 'CHEM', 'CITY', 'COUNTRY', 'DATE', 'DEVICE', 'DISO', 'DISTRICT', 'EVENT', 'FACILITY', 'FAMILY', 'FINDING', 'FOOD', 'GENE', 'HEALTH_CARE_ACTIVITY', 'INJURY_POISONING', 'LABPROC', 'LIVB', 'LOCATION', 'MEDPROC', 'MENTALPROC', 'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL', 'ORGANIZATION', 'PERCENT', 'PERSON', 'PHYS', 'PRODUCT', 'PROFESSION', 'SCIPROC', 'STATE_OR_PROVINCE', 'TIME']

    # tag_classes = sorted(["AGE", "CITY", "COUNTRY", "DATE", "DISO", "FACILITY", "LOCATION", "NUMBER", "ORDINAL", "ORGANIZATION", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "STATE_OR_PROVINCE", "TIME"])

    score_dict = dict()
    for tag in tag_classes:
        print(tag + ":")
        score_dict[tag] = trainer.test(ckpt_model, get_dataloader(model = ckpt_model, dataset_path = os.path.join(args.data_dir, "test.json"), tag = tag))
        print(tag + ":" + str(get_dataloader(model = ckpt_model, dataset_path = os.path.join(args.data_dir, "test.json"), tag = tag).__len__()))

    score_dict = {tag : {score : value for score, value in scores[0].items() if score in ["span_f1", "span_precision", "span_recall", 
    "span_tp", "span_fp", "span_fn"]} for tag, scores in score_dict.items()}

    with open(result_dirpath + "/" + args.data_dir.split('/')[-1] + "-score_" + 
            args.pretrained_checkpoint.split('/')[-1] + ".json", "w", encoding = "UTF-8") as output:
        json.dump(score_dict, output, ensure_ascii = False, indent = 2, sort_keys = True)

    fscores = [s["span_f1"] for s in score_dict.values()]
    prscores = [s["span_precision"] for s in score_dict.values()]
    rscores = [s["span_recall"] for s in score_dict.values()]
    
    tp_scores = [s["span_tp"] for s in score_dict.values()]
    fp_scores = [s["span_fp"] for s in score_dict.values()]
    fn_scores = [s["span_fn"] for s in score_dict.values()]
    micro_scores = list(zip(tp_scores, fp_scores, fn_scores))

    import sys

    print(f"Macro precision = {sum(prscores) / len(prscores)}", file = sys.stderr)
    print(f"Macro recall = {sum(rscores) / len(rscores)}", file = sys.stderr)
    print(f"Macro F1 = {sum(fscores) / len(fscores)}", file = sys.stderr)
    print("-----------")
    micro_prec = sum(tp_scores) / (sum(tp_scores) + sum(fn_scores) + 1e-10)
    micro_rec = sum(tp_scores) / (sum(tp_scores) + sum(fp_scores) + 1e-10)
    print(f"Micro precision = {micro_prec}", file = sys.stderr)
    print(f"Micro recall = {micro_rec}", file = sys.stderr)
    print(f"Micro F1 = {micro_prec * micro_rec * 2 / (micro_prec + micro_rec + 1e-10)}", file = sys.stderr)

    print('Расчет окончен!')


def test_conll(tag_test = None):

    parser = get_parser()
    parser = BertLabeling.add_model_specific_args(parser) 
    parser = Trainer.add_argparse_args(parser) 

    parser.add_argument('--result_dirpath', type = str, default = "./ckpt", help = "Directory path, where result would be saved. Default: \"ckpt\".")

    args = parser.parse_args()
    result_dirpath = args.result_dirpath

    seed_everything(args.seed, workers=True)

    ckpt_model = BertLabeling.load_from_checkpoint(args.pretrained_checkpoint) # Инициализиуем модель на их основе

    trainer = Trainer.from_argparse_args(
        args,
        logger = False,
        deterministic = True
    )

    tag_classes = sorted(["LOC", "MISC", "PER", "ORG"])

    score_dict = dict()
    for tag in tag_classes:
        print(tag + ":")
        score_dict[tag] = trainer.test(ckpt_model, get_dataloader(model = ckpt_model, dataset_path = os.path.join(args.data_dir, "test.json"), tag = tag))
        print(tag + ":" + str(get_dataloader(model = ckpt_model, dataset_path = os.path.join(args.data_dir, "test.json"), tag = tag).__len__()))

    score_dict = {tag : {score : value for score, value in scores[0].items() if score in ["span_f1", "span_precision", "span_recall", 
    "span_tp", "span_fp", "span_fn"]} for tag, scores in score_dict.items()}

    with open(result_dirpath + "/" + args.data_dir.split('/')[-1] + "-score_" + 
            args.pretrained_checkpoint.split('/')[-1] + ".json", "w", encoding = "UTF-8") as output:
        json.dump(score_dict, output, ensure_ascii = False, indent = 2, sort_keys = True)

    fscores = [s["span_f1"] for s in score_dict.values()]
    prscores = [s["span_precision"] for s in score_dict.values()]
    rscores = [s["span_recall"] for s in score_dict.values()]
    
    tp_scores = [s["span_tp"] for s in score_dict.values()]
    fp_scores = [s["span_fp"] for s in score_dict.values()]
    fn_scores = [s["span_fn"] for s in score_dict.values()]
    micro_scores = list(zip(tp_scores, fp_scores, fn_scores))

    import sys

    print(f"Macro precision = {sum(prscores) / len(prscores)}", file = sys.stderr)
    print(f"Macro recall = {sum(rscores) / len(rscores)}", file = sys.stderr)
    print(f"Macro F1 = {sum(fscores) / len(fscores)}", file = sys.stderr)
    print("-----------")
    micro_prec = sum(tp_scores) / (sum(tp_scores) + sum(fn_scores) + 1e-10)
    micro_rec = sum(tp_scores) / (sum(tp_scores) + sum(fp_scores) + 1e-10)
    print(f"Micro precision = {micro_prec}", file = sys.stderr)
    print(f"Micro recall = {micro_rec}", file = sys.stderr)
    print(f"Micro F1 = {micro_prec * micro_rec * 2 / (micro_prec + micro_rec + 1e-10)}", file = sys.stderr)

    print('Расчет окончен!')


if __name__ == '__main__':
    test_bio_fix(tag_test = None) # None - тест по всем сущностям; иначе, выбрать сущность (например, tag_test='AGE')