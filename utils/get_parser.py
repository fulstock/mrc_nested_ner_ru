# encoding: utf-8


import argparse


def get_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="Training", add_help = True)

    parser.add_argument("--data_dir", type=str, required=True, help="Data directory. Always required.")
    parser.add_argument("--bert_config_dir", type=str, required=True, help="Bert config (e.g. bert_config.json) directory. Always required.")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="Pretrained checkpoint (.ckpt) filepath. Default: \"\" ")
    parser.add_argument("--max_length", type=int, default=128, help="Max length of dataset. Default: 128")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default: 32")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate. Default: 2e-5")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader") # ?
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if applied. Default: 0.01")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Warmup steps used for scheduler. Default: 0")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer. Default: 1e-8")

    return parser
