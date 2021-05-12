# encoding: utf-8


import argparse
import os
from collections import namedtuple
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim import SGD

from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.truncate_dataset import TruncateDataset
from datasets.mrc_ner_dataset import collate_to_max_length
from metrics.query_span_f1 import QuerySpanF1
from models.bert_query_ner import BertQueryNER
from models.query_ner_config import BertQueryNerConfig
from loss import *
from utils.get_parser import get_parser
from utils.random_seed import set_random_seed
import logging

set_random_seed(0)

class BertLabeling(pl.LightningModule):
    """MLM Trainer"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Инициализация модели, конфига и т.п."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
            self.eval_mode = False
        else: 
            # eval mode
            self.eval_mode = True
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.bert_dir = args.bert_config_dir
        self.data_dir = args.data_dir

        bert_config = BertQueryNerConfig.from_pretrained(args.bert_config_dir,
                                                         hidden_dropout_prob=args.bert_dropout,
                                                         attention_probs_dropout_prob=args.bert_dropout,
                                                         mrc_dropout=args.mrc_dropout)

        self.model = BertQueryNER.from_pretrained(args.bert_config_dir,
                                                  config=bert_config)
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args)) # ? Зачем это?
        self.loss_type = args.loss_type
        if self.loss_type == "bce":
            self.bce_loss = BCEWithLogitsLoss(reduction="none")
        else:
            self.dice_loss = DiceLoss(with_logits=True, smooth=args.dice_smooth)

        # Нормализуем
        weight_sum = args.weight_start + args.weight_end + args.weight_span
        self.weight_start = args.weight_start / weight_sum
        self.weight_end = args.weight_end / weight_sum
        self.weight_span = args.weight_span / weight_sum

        # метрика для подсчета качества
        self.span_f1 = QuerySpanF1()
        self.optimizer = args.optimizer
        self.span_loss_candidates = args.span_loss_candidates

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--mrc_dropout", type=float, default=0.1,
                            help="MRC dropout rate. Default: 0.1")
        parser.add_argument("--bert_dropout", type=float, default=0.1,
                            help="Bert dropout rate. Default: 0.1")
        parser.add_argument("--weight_start", type=float, default=1.0, help="Start logits weight for the loss. Default: 1.0")
        parser.add_argument("--weight_end", type=float, default=1.0, help="End logits weight for the loss. Default: 1.0")
        parser.add_argument("--weight_span", type=float, default=1.0, help="Span logits weight for the loss. Default: 1.0")
        parser.add_argument("--span_loss_candidates", choices=["all", "pred_and_gold", "gold"],
                            default="all", help="Candidates used to compute span loss. Default: \"all\"")
        parser.add_argument("--loss_type", choices=["bce", "dice"], default="bce",
                            help="Loss type, BCE or Dice. Default: \"bce\" ") # ?
        parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                            help="Optimizer used. Default: \"adamw\"")
        parser.add_argument("--dice_smooth", type=float, default=1e-8,
                            help="Smooth value of dice loss. Default: 1e-8")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="Final div factor of linear decay scheduler. Default: 1e4")
        return parser

    def configure_optimizers(self):

        """Подготовка оптимизаторов и расписания"""

        if self.eval_mode:
            return None # Просто тестируем, обучения нет, их можно выключить. 
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # (RoBERTa paper)
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)

        num_devices = len([x for x in str(self.args.gpus).split(",") if x.strip()]) # TODO: Исправить ошибку тут
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_devices) + 1) * self.args.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids): 
        """"""
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def compute_loss(self, 
                start_logits, 
                end_logits,  
                span_logits, 
                start_labels, 
                end_labels, 
                match_labels, 
                start_label_mask, 
                end_label_mask):

        batch_size, seq_len = start_logits.size()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start \leq end

        if self.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        if self.loss_type == "bce":
            start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
            start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
            end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
            end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
            match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
            match_loss = match_loss * float_match_label_mask
            match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
        else:
            start_loss = self.dice_loss(start_logits, start_labels.float(), start_float_label_mask)
            end_loss = self.dice_loss(end_logits, end_labels.float(), end_float_label_mask)
            match_loss = self.dice_loss(span_logits, match_labels.float(), float_match_label_mask)

        return start_loss, end_loss, match_loss

    def training_step(self, batch, batch_idx):
        """"""
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr'] 
        }
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels = batch

        # num_tasks * [bsz, length, num_labels]
        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)

        start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )

        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

        tf_board_logs[f"train_loss"] = total_loss
        tf_board_logs[f"start_loss"] = start_loss
        tf_board_logs[f"end_loss"] = end_loss
        tf_board_logs[f"match_loss"] = match_loss

        return {'loss': total_loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""

        output = {}

        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels = batch

        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)

        start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )

        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

        output[f"val_loss"] = total_loss
        output[f"start_loss"] = start_loss
        output[f"end_loss"] = end_loss
        output[f"match_loss"] = match_loss

        start_preds, end_preds = start_logits > 0, end_logits > 0
        span_f1_stats = self.span_f1(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                     start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                     match_labels=match_labels)
        output["span_f1_stats"] = span_f1_stats

        return output

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        all_counts = torch.stack([x[f'span_f1_stats'] for x in outputs]).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs[f"span_precision"] = span_precision
        tensorboard_logs[f"span_recall"] = span_recall
        tensorboard_logs[f"span_f1"] = span_f1

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self,
        outputs
    ) -> Dict[str, Dict[str, Tensor]]:
        """"""
        return self.validation_epoch_end(outputs)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def test_dataloader(self):
        return self.get_dataloader(prefix = "test")

    def get_dataloader(self, prefix="train", limit: int = None, tag = None) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        dataset_path = self.data_dir
        vocab_path = os.path.join(self.bert_dir, "vocab.txt") # важно знать, по какому словарю токенизировать
        dataset = MRCNERDataset(dataset_path=os.path.join(dataset_path, prefix), 
                                tokenizer=BertWordPieceTokenizer(vocab_path, lowercase = False),
                                max_length=self.args.max_length,
                                pad_to_maxlen=False,
                                tag = tag # для тестирования по конкретным классам сущностей
                                )

        if limit is not None:
            dataset = TruncateDataset(dataset, limit) 

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True if prefix == "train" else False,
            collate_fn=collate_to_max_length
        )

        return dataloader


def run_dataloader():
    """test dataloader"""
    
    parser = get_parser()
    parser = BertLabeling.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser) 

    args = parser.parse_args()
    args.workers = 0
    args.default_root_dir = "train_logs/debug"

    model = BertLabeling(args)
    from tokenizers import BertWordPieceTokenizer
    tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_config_dir, "vocab.txt"), lowercase = False)

    loader = model.get_dataloader("dev", limit=50)
    for d in loader:
        input_ids = d[0][0].tolist()
        match_labels = d[-1][0]
        start_positions, end_positions = torch.where(match_labels > 0)
        start_positions = start_positions.tolist()
        end_positions = end_positions.tolist()
        if not start_positions:
            continue
        print("="*20)
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        for start, end in zip(start_positions, end_positions):
            print(tokenizer.decode(input_ids[start: end+1]))


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

    model = BertLabeling(args) # Инициализиуем модель на их основе

    # Если грузим из чекпойнта
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint, 
                                         map_location=torch.device('cpu'))["state_dict"]) # ? Убрать бы map_location

    checkpoint_callback = ModelCheckpoint(
        # Директория, куда будут сохраняться чекпойнты и логи (по умолчанию корневая папка проекта)
        filepath=args.default_root_dir, 
        save_top_k=10, # Сохранять топ 10 моделей по метрике monitor
        verbose=True, # Уведомлять о результатах валидации
        monitor="span_f1", # Метрика для подсчета качества модели, см. span_f1
        period=-1, # Сохранять чекпойнты каждую эпоху
        mode="max", # Сохраняем самые максимальные по метрике модели
    )

    # Инициализация Trainer на основе аргументов командной строки 
    # Настройка сохранения моделей через callbacks
    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model) # Запуск процесса обучения и валидации, с мониторингом


if __name__ == '__main__':
    main()
