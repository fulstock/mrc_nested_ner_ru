# encoding: utf-8

from torchmetrics import Metric
import torch

def query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels):
    """
    Compute span f1 according to query-based model output
    Args:
        start_preds: [bsz, seq_len]
        end_preds: [bsz, seq_len]
        match_logits: [bsz, seq_len, seq_len]
        start_label_mask: [bsz, seq_len]
        end_label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    tp = (match_labels & match_preds).long().sum()
    fp = (~match_labels & match_preds).long().sum()
    fn = (match_labels & ~match_preds).long().sum()
    # return torch.stack([tp, fp, fn])
    return tp, fp, fn

class QuerySpanF1(Metric):
    """
    Query Span F1
    """
    def __init__(self):
        super().__init__()
        
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx = "sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx = "sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx = "sum")

    def update(self, start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels):
        
        tp, fp, fn = query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels)
        self.tp += tp
        self.fp += fp
        self.fn += fn
    
    def compute(self):
        
        return torch.stack([self.tp, self.fp, self.fn])
    
    # def forward(self, start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels):
    #    return query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels)
