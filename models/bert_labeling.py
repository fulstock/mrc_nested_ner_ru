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
from typing import Dict
import os
import tqdm
from tqdm import tqdm
import argparse
import logging
from collections import namedtuple

from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.truncate_dataset import TruncateDataset, TruncateByTypeDataset
from datasets.mrc_ner_dataset import collate_to_max_length
from metrics.query_span_f1 import QuerySpanF1
from models.bert_query_ner import BertQueryNER
from models.query_ner_config import BertQueryNerConfig
from loss import *
from utils.get_parser import get_parser
from utils.random_seed import set_random_seed

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

        vocab_path = os.path.join(self.bert_dir, "vocab.txt") # важно знать, по какому словарю токенизировать
        self.tokenizer = BertWordPieceTokenizer(vocab_path, lowercase = False)

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

        ###

        self.limit_by_type = args.limit_by_type

        ###

        ### cycle

        # self.datacycle = args.datacycle

        # self.change_factor_epnum = args.change_factor_epnum
        # self.epochs_left_before_change = self.change_factor_epnum
        # self.last_value = 0.0
        # self.curr_datacycle_num = 0

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
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # (RoBERTa paper)
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)

        num_devices = torch.cuda.device_count()
        if num_devices == 0:
            num_devices = 1 # cpu only
        if self.args.accumulate_grad_batches is None:
            self.args.accumulate_grad_batches = 1
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_devices)) * self.args.max_epochs
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
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, tag_idx = batch

        # print(start_labels)
        # print(end_labels)
        # print(match_labels)

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

        tf_board_logs["train_total_loss"] = total_loss
        tf_board_logs["train_start_loss"] = start_loss
        tf_board_logs["train_end_loss"] = end_loss
        tf_board_logs["train_match_loss"] = match_loss

        self.log_dict(tf_board_logs)

        return {'loss': total_loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""

        output = {}

        tokens_batch, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels_batch, sample_ids, tag_ids = batch

        attention_mask = (tokens_batch != 0).long()
        start_logits, end_logits, span_logits = self(tokens_batch, attention_mask, token_type_ids)

        start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels_batch,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )

        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

        output["val_total_loss"] = total_loss
        output["val_start_loss"] = start_loss
        output["val_end_loss"] = end_loss
        output["val_match_loss"] = match_loss

        self.log_dict(output)

        start_preds, end_preds = start_logits > 0, end_logits > 0
        span_f1_stats = self.span_f1(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                     start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                     match_labels=match_labels_batch)
        output["span_f1_stats"] = span_f1_stats

        seq_len = tokens_batch.size(dim = 1)

        start_preds = start_logits > 0 
        end_preds = end_logits > 0
        match_preds = span_logits > 0

        match_preds = (match_preds
                       & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                       & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
        
        match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                            & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
        match_label_mask = torch.triu(match_label_mask, 0)
        match_preds_batch = match_label_mask & match_preds
        
        # import sys

        # for tokens, sample_idx, match_preds, match_labels in zip(tokens_batch, sample_ids, match_preds_batch, match_labels_batch):
        #     tokens = tokens.tolist()

        #     start_pos_pred, end_pos_pred = torch.where(match_preds > 0)
        #     start_pos_pred = start_pos_pred.tolist()
        #     end_pos_pred = end_pos_pred.tolist()

        #     start_pos_label, end_pos_label = torch.where(match_labels > 0)
        #     start_pos_label = start_pos_label.tolist()
        #     end_pos_label = end_pos_label.tolist()

        #     if not start_pos_pred and not start_pos_label:
        #         continue

        #     print("="*20, file = sys.stderr)
        #     print(f"Sample #{str(int(sample_idx))}, len = {len(tokens)}: ", self.tokenizer.decode(tokens, skip_special_tokens=False), file = sys.stderr)
        #     print("-"*20, file = sys.stderr)
        #     if start_pos_label:
        #         print("True labels:", file = sys.stderr)
        #         for start, end in zip(start_pos_label, end_pos_label):
        #             print(f"[{start}:{end}]\t" + self.tokenizer.decode(tokens[start: end+1]), file = sys.stderr)
        #     else:
        #         print("No true labels.", file = sys.stderr)
        #     if start_pos_pred:
        #         print("Predicted labels:", file = sys.stderr)
        #         for start, end in zip(start_pos_pred, end_pos_pred):
        #             print(f"[{start}:{end}]\t" + self.tokenizer.decode(tokens[start: end+1]), file = sys.stderr)
        #     else:
        #         print("No predicted labels.", file = sys.stderr)

        return output

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['val_total_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_avg_loss': avg_loss}

        all_counts = torch.stack([x[f'span_f1_stats'] for x in outputs]).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs["span_tp"] = span_tp.detach().float()
        tensorboard_logs["span_fp"] = span_fp.detach().float()
        tensorboard_logs["span_fn"] = span_fn.detach().float()
        tensorboard_logs["span_precision"] = span_precision.detach()
        tensorboard_logs["span_recall"] = span_recall.detach()
        tensorboard_logs["span_f1"] = span_f1.detach()

        self.log_dict(tensorboard_logs)
        self.log("span_f1", span_f1)

        # print(f"Span F1 for epoch {self.current_epoch} is {span_f1}.")

        # if self.change_factor_epnum >= 0:

        #     self.epochs_left_before_change -= 1
            
        #     if self.epochs_left_before_change < 0 and self.last_value > span_f1:
        #         self.curr_datacycle_num = min(15, self.curr_datacycle_num + 1)
        #         self.epochs_left_before_change = self.change_factor_epnum

        # self.last_value = span_f1

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
        return self.get_dataloader("train", limit_by_type = self.limit_by_type)

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def get_dataloader(self, prefix, limit: int = None, limit_by_type = None, tag = None) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """

        limit = 10 
        
        dataset_path = self.data_dir

        # ds = "" if self.datacycle < 0 or prefix != "train" else "_dc" + str(self.current_epoch % self.datacycle)

        ds = ""

        # ds = "" if prefix != "train" else "_dc" + str(self.curr_datacycle_num)

        dataset = MRCNERDataset(dataset_path=os.path.join(dataset_path + ds, prefix + ".json"), 
                                tokenizer=self.tokenizer,
                                max_length=self.args.max_length,
                                pad_to_maxlen=False,
                                tag = tag # для тестирования по конкретным классам сущностей
                                )

        # print(limit)
        # print(len(dataset))

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)
        elif limit_by_type is not None:
            dataset = TruncateByTypeDataset(dataset, int(limit_by_type))

        # print(len(dataset))

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            persistent_workers = self.args.workers > 0,
            shuffle=True if prefix == "train" else False,
            collate_fn=collate_to_max_length
        )

        return dataloader