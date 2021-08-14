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

from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.truncate_dataset import TruncateDataset
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
        model_args = None,
        bert_args = None,
        trainer_args = None
    ):
        """Инициализация модели, конфига и т.п."""
        super().__init__()

        self.eval_mode = False
        self.model_args = model_args
        self.bert_args = bert_args
        self.trainer_args = trainer_args
        
        self.bert_dir = model_args["bert_config_dir"]
        self.data_dir = model_args["data_dir"]
        
        vocab_path = os.path.join(self.bert_dir, "vocab.txt")
        self.tokenizer = BertWordPieceTokenizer(vocab_path, lowercase = False)
        self.output_test_file = open("test_dataset.out", "w", encoding = "utf-8")

        bert_config = BertQueryNerConfig.from_pretrained(
            self.bert_dir,
            hidden_dropout_prob = bert_args["bert_dropout"],
            attention_probs_dropout_prob = bert_args["bert_dropout"],
            mrc_dropout = model_args["mrc_dropout"]
        )

        self.model = BertQueryNER.from_pretrained(self.bert_dir,
                                                  config=bert_config)
        
        self.loss_type = model_args["loss_type"]
        if self.loss_type == "bce":
            self.bce_loss = BCEWithLogitsLoss(reduction="none")
        else:
            self.dice_loss = DiceLoss(with_logits=True, smooth=args.dice_smooth)

        # Нормализуем
        weight_sum = model_args["weight_start"] + model_args["weight_end"] + model_args["weight_span"]
        self.weight_start = model_args["weight_start"] / weight_sum
        self.weight_end = model_args["weight_end"] / weight_sum
        self.weight_span = model_args["weight_span"] / weight_sum

        # метрика для подсчета качества
        self.span_f1 = QuerySpanF1()
        self.optimizer = model_args["optimizer"]
        self.span_loss_candidates = model_args["span_loss_candidates"]
        
        # self.output_test_file.close()

    def configure_optimizers(self):

        """Подготовка оптимизаторов и расписания"""

        if self.eval_mode:
            return None # Просто тестируем, обучения нет, их можно выключить. 
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.model_args["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # (RoBERTa paper)
                              lr=self.model_args["lr"],
                              eps=self.model_args["adam_epsilon"])
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.model_args["lr"], momentum=0.9)

        num_devices = self.trainer_args["gpus"] # TODO: Исправить ошибку тут
        t_total = (len(self.train_dataloader()) // (self.model_args["accumulate_grad_batches"] * 
                                                    num_devices) + 1) * self.trainer_args["max_epochs"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.model_args["lr"], pct_start=float(self.model_args["warmup_steps"]/t_total),
            final_div_factor=self.model_args["final_div_factor"],
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

        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, tag_idx = batch

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

        return {'span_tp' : span_tp, 'span_fp' : span_fp, 'span_fn' : span_fn, 'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        
        output = {}

        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, tag_idx = batch

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
        
        batch_size, seq_len = start_logits.size()

        start_preds = start_logits > 0 
        end_preds = end_logits > 0
        match_preds = span_logits > 0

        match_preds = (match_preds
                       & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                       & end_preds.unsqueeze(1).expand(-1, seq_len, -1))

        match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                            & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
        match_label_mask = torch.triu(match_label_mask, 0)
        match_preds = match_label_mask & match_preds
        
        example_num = 0

        start_preds = start_preds[example_num]
        end_preds = end_preds[example_num]
        match_preds = match_preds[example_num]

        input_ids = tokens[example_num].tolist()
        match_labels = match_labels[example_num]

        start_positions, end_positions = torch.where(match_preds == True)
        start_label_positions, end_label_positions = torch.where(match_labels > 0)

        start_positions = start_positions.tolist()
        end_positions = end_positions.tolist()
        start_label_positions = start_label_positions.tolist()
        end_label_positions = end_label_positions.tolist()

        def pretty_print(string, length, symbol, file):
            print (symbol *((length - len(string)) // 2) + string + symbol *((length - len(string)) // 2), file = file)
        
        f = self.output_test_file
        
        print("="*45, file = f)
        decoded_spec = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        print(decoded_spec, file = f)
        pretty_print("Предсказанные сущности", 45, '-', f)
        if start_positions:
            for pos_idx, (start, end) in enumerate(zip(start_positions, end_positions)):
                decoded = self.tokenizer.decode(input_ids[start: end+1])
                print(decoded, file = f)
        else:
            print("<Ни одной сущности не найдено.>", file = f)
        print("-"*45, file = f)
        pretty_print("Настоящие сущности", 45, '-', f)
        if start_label_positions:
            for start, end in zip(start_label_positions, end_label_positions):
                decoded = self.tokenizer.decode(input_ids[start: end+1])
                print(decoded, file = f)
        else:
            print("<Ни одной сущности не найдено.>", file = f)
        print("-"*45, file = f)

        # print("Батч " + str(batch_idx))
        
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self,
        outputs
    ) -> Dict[str, Dict[str, Tensor]]:
        """""" 
        print("="*45, file = self.output_test_file)
        self.output_test_file.close()
        
        return self.validation_epoch_end(outputs)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def test_dataloader(self):
        return self.get_dataloader(prefix = "test")

    def get_dataloader(self, prefix="train", limit: int = None, tag = None) -> DataLoader:
        """get training dataloader"""
        dataset_path = os.path.join(self.data_dir, f"{prefix}.json")
        vocab_path = os.path.join(self.bert_dir, "vocab.txt") # важно знать, по какому словарю токенизировать
        dataset = MRCNERDataset(dataset_path=dataset_path, 
                                tokenizer=BertWordPieceTokenizer(vocab_path, lowercase = False),
                                max_length=self.model_args["max_length"],
                                pad_to_maxlen=False,
                                tag = tag # для тестирования по конкретным классам сущностей
                                )

        if limit is not None:
            dataset = TruncateDataset(dataset, limit) 

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.model_args["batch_size"],
            num_workers=self.model_args["workers"],
            shuffle=True if prefix == "train" else False,
            collate_fn=collate_to_max_length
        )

        return dataloader