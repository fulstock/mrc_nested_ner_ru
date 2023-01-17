# encoding: utf-8


import argparse
import os
from collections import namedtuple
from typing import Dict
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch.optim import SGD

from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.truncate_dataset import TruncateDataset
from datasets.mrc_ner_dataset import collate_to_max_length
from metrics.query_span_f1 import QuerySpanF1
from models.bert_query_ner import BertQueryNER
from models.query_ner_config import BertQueryNerConfig
from models.bert_labeling import BertLabeling
from loss import *
from utils.get_parser import get_parser
from utils.random_seed import set_random_seed
# from utils.progress import TGProgressBar
import logging

def main():
    """main"""

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

    model = BertLabeling(args) # Инициализиуем модель на их основе

    # Если грузим из чекпойнта
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint)["state_dict"]) # ? Убрать бы map_location

    serial_number = datetime.now().strftime('%y%m%d_%H%M%S')

    param = ""

    if args.default_root_dir is None:
        args.default_root_dir = "."

    checkpoint_callback = ModelCheckpoint(
        # Директория, куда будут сохраняться чекпойнты и логи (по умолчанию корневая папка проекта)
        dirpath=os.path.join(args.default_root_dir, "ckpt"), 
        filename = "model-" + param + args.data_dir.split('/')[-1] + f"-seed={args.seed}-" + serial_number + "-{epoch}",
        save_top_k=-1, # Сохранять топ 10 моделей по метрике monitor
        verbose=True, # Уведомлять о результатах валидации
        monitor="span_f1", # Метрика для подсчета качества модели, см. span_f1
        mode="max", # Сохраняем самые максимальные по метрике модели
        save_last = True
    )

    # bar = TGProgressBar()

    # Инициализация Trainer на основе аргументов командной строки 
    # Настройка сохранения моделей через callbacks
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback #,bar
        ],
        deterministic = True
    )

    trainer.fit(model) # Запуск процесса обучения и валидации, с мониторингом

    trainer.test(model, dataloaders=model.get_dataloader("test"))


if __name__ == '__main__':

    # import os
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'

    main()
