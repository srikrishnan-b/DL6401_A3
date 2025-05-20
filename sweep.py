from src.sweep_config import sweep_config
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
from src.model import RNN_light
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":

    train_path = "train.tsv"
    valid_path = "valid.tsv"
    test_path = "test.tsv"

    train_df = pd.read_csv(train_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')
    valid_df = pd.read_csv(valid_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')
    test_df = pd.read_csv(test_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')

    train_df = train_df[~train_df['latin'].isna()]
    valid_df = valid_df[~valid_df['latin'].isna()]
    test_df = test_df[~test_df['latin'].isna()]

    tokenizer = NativeTokenizer(train_path, valid_path, test_path)
    print(f"Latin vocab size: {tokenizer.latin_vocab_size}")
    print(f"Native vocab size: {tokenizer.nat_vocab_size}")


    def trainCNN(config=None):
        with wandb.init(config=config) as run:
            config = wandb.config

            run.name = f"cell_{config.cell}_emb_{config.embedding_size}_hidden_{config.hidden_size}_D_{config.dropout:.2f}_layers_{config.layers}"
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            try:
                train_dataset = LatNatDataset(train_df, tokenizer)
                valid_dataset = LatNatDataset(valid_df, tokenizer)
                test_dataset = LatNatDataset(test_df, tokenizer)
                
                train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=2)
                valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn , num_workers=2)
                test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=2)
                special_tokens = {key: val for key, val in tokenizer.native_vocab.items() if key in ['<start>', '<end>', '<pad>']}

                model = RNN_light(
                    input_sizes=(tokenizer.latin_vocab_size, tokenizer.nat_vocab_size),
                    embedding_size=config.embedding_size,
                    hidden_size=config.hidden_size,
                    cell=config.cell,
                    layers=config.layers,
                    dropout=config.dropout,
                    activation=config.activation,
                    beam_size=3,
                    optim=config.optim,
                    special_tokens=special_tokens,
                    lr=config.lr
                )
                logger = WandbLogger(
                    project=project_name, name=run.name, experiment=run, log_model=False
                )
                trainer = pl.Trainer(
                    devices=1,
                    accelerator="auto",
                    precision="16-mixed",
                    gradient_clip_val=1.0,
                    max_epochs=config.epochs,
                    logger=logger,
                    profiler=None,
                )

                trainer.fit(model, train_dataloader, valid_dataloader)
            finally:
                del trainer
                del model
                gc.collect()
                torch.cuda.empty_cache()


    project_name = "DLA3_sweeps"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=trainCNN, count=10)