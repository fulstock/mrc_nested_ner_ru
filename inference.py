import torch
from pytorch_lightning import Trainer

from tokenizers import BertWordPieceTokenizer

import matplotlib.pyplot as plt

from trainer import *
from utils.get_parser import get_parser
from utils.random_seed import set_random_seed
from utils.progress import TGProgressBar

def process_data(context, query, tokenizer):

    query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True)
    tokens = query_context_tokens.ids
    type_ids = query_context_tokens.type_ids
    offsets = query_context_tokens.offsets

    return torch.LongTensor(tokens).unsqueeze(0), torch.LongTensor(type_ids).unsqueeze(0)

def process_attn(tokens, attention):

    attn_data = []

    n_heads = attention[0][0].size(0)
    layers = list(range(len(attention)))
    heads = list(range(n_heads))

    print("n_heads = " + str(n_heads))
    print("n_layers = " + str(len(attention)))

    attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        # print(layer_attention)
        layer_attention = layer_attention.squeeze(0)
        layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    attention = torch.stack(squeezed)
    attention = attention.tolist()

    return attention

def test(tag_test = None):

    parser = get_parser()
    parser = BertLabeling.add_model_specific_args(parser) 
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    ckpt_model = BertLabeling.load_from_checkpoint(args.pretrained_checkpoint) # Инициализиуем модель на их основе

    sequence = "В ноябре прошлого года независимая комиссия Всемирного антидопингового агентства обвинила Россию в нарушениях антидопинговых правил."
    query = "Дата - это номер дня в месяце, часто указываемый в сочетании с названием дня, месяца и года."

    vocab_path = "mrc/for_mrc/rubert/vocab.txt"
    tokenizer = BertWordPieceTokenizer(vocab_path, lowercase = False)

    tokens, type_ids = process_data(sequence, query, tokenizer)
    readable_tokens = tokenizer.decode(tokens.squeeze(0).tolist(), skip_special_tokens=False)
    print(tokens)
    print(tokens.size())
    attention_mask = (tokens != 0).long()

    ckpt_model.eval()
    start_logits, end_logits, span_logits, attention = ckpt_model.forward(tokens, attention_mask=attention_mask, token_type_ids=type_ids)

    print("Attentions:")
    attn_view = process_attn(tokens, attention)
    print(len(attn_view))
    layer = attn_view[2] # 12 layers, 12 heads, head = seq_len x seq_len (doubled list)
    
    import numpy as np
    import seaborn as sns; sns.set_theme()

    avg_heads = np.sum(np.array(layer), axis = 0)
    print(avg_heads.shape)

    def plot_matrix(cm, query_len, tokens, title):
        ax = sns.heatmap(cm, cmap="Blues", annot=False, square = True, robust = False,
                xticklabels=tokens[query_len + 1 : -1], yticklabels=tokens[1:query_len], cbar=True)
        ax.set(title=title, xlabel="", ylabel="")
        return ax

    cm = np.array(avg_heads)
    tokens = tokens.squeeze(0).tolist()
    query_len = tokens.index(102)
    cm = cm[1 : query_len, query_len + 1 : -1]
    ax = plot_matrix(cm, query_len, [tokenizer.decode([t], skip_special_tokens=False) for t in tokens], "")
    fig = ax.get_figure()
    fig.set(figheight = 11, figwidth = 9)
    fig.savefig("mrc/attn.png")

if __name__ == '__main__':
    test() 






# {
#     "id": "2075.3",
#     "context": "В начале 2016 г. Григорий Родченков и его заместитель Тимофей Соболевский эмигрировали в США, а министр спорта России Виталий Мутко сообщил, что лаборатория будет готовиться переаттестации.",
#     "tag": "COUNTRY",
#     "query": "Страна - это территория земли, на которой есть собственное правительство, армия и т.д.",
#     "filename": "100756_text",
#     "exists": true,
#     "start_positions": [
#       89,
#       111
#     ],
#     "end_positions": [
#       92,
#       117
#     ],
#     "span_positions": [
#       "89;92",
#       "111;117"
#     ],
#     "spans": [
#       "США",
#       "России"
#     ]
#   },