import numpy as np
import pandas as pd
import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import wandb
import wandb
import lightning as pl
import gc
from src.config import *
from src.dataloader import LatNatDataset, NativeTokenizer
from src.model import RNN_light, RNN_light_attention
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import seaborn as sns

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--cell', type=str, default='LSTM', help='RNN cell type')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.35, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--project_name', type=str, default='DLA3_sweeps', help='Project name')
    parser.add_argument('--name', type=str, default='best_model', help='Model name')
    parser.add_argument("--model_type", type=str, default="w_attn", help="Model type (w_attn or wo_attn)")
    return parser.parse_args()

def main():
    args = argparse()
    batch_size = args.batch_size
    EMBEDDING_SIZE = args.embedding_size
    HIDDEN_SIZE = args.hidden_size
    cell = args.cell
    layers = args.layers
    dropout = args.dropout
    activation = args.activation
    optim = args.optim
    lr = args.lr


    os.environ['WANDB_API_KEY'] = "key"
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    train_path = "/kaggle/input/dakshina/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv"
    valid_path = "/kaggle/input/dakshina/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv"
    test_path = "/kaggle/input/dakshina/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.test.tsv"

    train_df = pd.read_csv(train_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')
    valid_df = pd.read_csv(valid_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')
    test_df = pd.read_csv(test_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')

    train_df = train_df[~train_df['latin'].isna()]
    valid_df = valid_df[~valid_df['latin'].isna()]
    test_df = test_df[~test_df['latin'].isna()]


    tokenizer = NativeTokenizer(train_path, valid_path, test_path)
    print(f"Latin vocab size: {tokenizer.latin_vocab_size}")
    print(f"Native vocab size: {tokenizer.nat_vocab_size}")

    train_dataset = LatNatDataset(train_df, tokenizer)
    valid_dataset = LatNatDataset(valid_df, tokenizer)
    test_dataset = LatNatDataset(test_df, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn , num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=2)


    special_tokens = {key: val for key, val in tokenizer.native_vocab.items() if key in ['<start>', '<end>', '<pad>']}
    if args.model_type == "w_attn":
        model = RNN_light_attention(
            input_sizes=(tokenizer.latin_vocab_size, tokenizer.nat_vocab_size),
            embedding_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            cell=cell,
            layers=layers,
            dropout=dropout,
            activation=activation,
            beam_size=3,
            optim=optim,
            special_tokens=special_tokens,
            lr=lr
        )
    else:
        model = RNN_light(
            input_sizes=(tokenizer.latin_vocab_size, tokenizer.nat_vocab_size),
            embedding_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            cell=cell,
            layers=layers,
            dropout=dropout,
            activation=activation,
            beam_size=3,
            optim=optim,
            special_tokens=special_tokens,
            lr=lr
        )
    logger= WandbLogger(project= 'DLA3_sweeps', name = "bestmodel") #,resume="never")
    trainer = pl.Trainer(max_epochs=1,  accelerator="auto",logger=logger, profiler='simple',  precision="16-mixed",)
    trainer.fit(model, train_dataloader,  valid_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
    # attention map
    if model_type == "w_attn":
        rand_ind = np.random.choice(len(model.test_preds), size=9, replace=False)
        attention_map = [model.attention_maps[ind] for ind in rand_ind]
        src = np.array(model.test_inputs)[rand_ind]
        pred = np.array(model.test_preds)[rand_ind]
        tgt = np.array(model.test_labels)[rand_ind]

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle("Attention map", fontsize=16)

        for i, ax in enumerate(axes.flat):
            attn_map = attention_map[i][0:len(pred[i]), 0:len(src[i])]  

            sns.heatmap(attn_map, ax=ax, xticklabels=src[i], yticklabels=pred[i],
                        cmap="Blues", cbar=True)

            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            fig.supxlabel("Latin script", fontsize=14)
            fig.supylabel("Tamil script", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  
            plt.show()

    else:
        rand_ind = np.random.choice(len(model.test_preds), size=9, replace=False)
        src = np.array(model.test_inputs)[rand_ind]
        tgt = np.array(model.test_labels)[rand_ind]
        preds = np.array(model.test_preds)[rand_ind]

        # table fo comparison
        fig, ax = plt.subplots(figsize=(10, len(src) * 1.5))
        ax.axis("off")

        table_data = [["Input", "Actual", "Prediction"]]
        for inp, true, pred in zip(src, tgt, preds):
            table_data.append([inp, true, pred])

        table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)



if __name__ == "__main__":
    main()