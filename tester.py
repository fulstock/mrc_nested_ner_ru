import torch
from pytorch_lightning import Trainer

import json

from trainer import *
from utils.get_parser import get_parser
from utils.random_seed import set_random_seed
from utils.progress import TGProgressBar

set_random_seed(0)

def test(tag_test = None):

    parser = get_parser() # Создание парсера командной строки в общем виде (см. get_parser)

    # Добавление аргументов командной строки, отвечающих самой модели
    parser = BertLabeling.add_model_specific_args(parser) 

    # Добавление всех возможных флагов Trainer (--gpus, --num_nodes и т.д.) из командной строки
    parser = Trainer.add_argparse_args(parser) 

    # Помощь по всем флагам командой строки - либо через -h / --help, либо (если указано pl.Trainer)
    # см. документацию по Trainer от Pytorch Lightning

    # Сохраняем все аргументы из командной строки
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    bar = TGProgressBar()

    ckpt_model = BertLabeling.load_from_checkpoint(args.pretrained_checkpoint) # Инициализиуем модель на их основе

    trainer = Trainer.from_argparse_args(
        args,
        logger = False,
        deterministic = True #,
        # callbacks = [bar]
    )

    # trainer.test(ckpt_model, ckpt_model.get_dataloader("test", tag = tag_test))

    tag_classes = ['AGE', 'AWARD', 'CITY', 'COUNTRY', 'CRIME', 'DATE', 'DISEASE', 'DISTRICT', 'EVENT', 'FACILITY', 
       'FAMILY', 'IDEOLOGY', 'LANGUAGE', 'LAW', 'LOCATION', 'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL', 
       'ORGANIZATION', 'PERCENT', 'PERSON', 'PENALTY', 'PRODUCT', 'PROFESSION', 'RELIGION', 'STATE_OR_PROVINCE', # 'OUT', 
       'TIME', 'WORK_OF_ART'] 

    score_dict = dict()
    for tag in tag_classes:
       print(tag + ":")
       score_dict[tag] = trainer.test(ckpt_model, ckpt_model.get_dataloader("test", tag = tag))
    print(tag + ":" + str(ckpt_model.get_dataloader("test", tag = tag).__len__()))

    score_dict = {tag : {score : value for score, value in scores[0].items() if score in ["span_f1", "span_precision", "span_recall"]} \
        for tag, scores in score_dict.items()}

    with open("mrc/slurm/tests/" + args.data_dir.split('/')[-1] + "/score_" + args.seed + ".json", "w", encoding = "UTF-8") as output:
        json.dump(score_dict, output, ensure_ascii = False, indent = 2, sort_keys = True)

    fscores = [s["span_f1"] for s in score_dict.values()]
    prscores = [s["span_precision"] for s in score_dict.values()]
    rscores = [s["span_recall"] for s in score_dict.values()]

    fewshot_fscores = [s["span_f1"] for tag, s in score_dict.items() if tag in ['DISEASE', 'PENALTY', 'WORK_OF_ART']]
    fewshot_prscores = [s["span_precision"] for tag, s in score_dict.items() if tag in ['DISEASE', 'PENALTY', 'WORK_OF_ART']]
    fewshot_rscores = [s["span_recall"] for tag, s in score_dict.items() if tag in ['DISEASE', 'PENALTY', 'WORK_OF_ART']]

    import sys

    print(f"Macro precision = {sum(prscores) / len(prscores)}", file = sys.stderr)
    print(f"Few-shot precision = {sum(fewshot_prscores) / len(fewshot_prscores)}", file = sys.stderr)
    print(f"Macro recall = {sum(rscores) / len(rscores)}", file = sys.stderr)
    print(f"Few-shot recall = {sum(fewshot_rscores) / len(fewshot_rscores)}", file = sys.stderr)
    print(f"Macro F1 = {sum(fscores) / len(fscores)}", file = sys.stderr)
    print(f"Few-shot F1 = {sum(fewshot_fscores) / len(fewshot_fscores)}", file = sys.stderr)

    print('Расчет окончен!')

if __name__ == '__main__':
    test(tag_test = None) # None - тест по всем сущностям; иначе, выбрать сущность (например, tag_test='AGE')