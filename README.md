# DA6401 Assignment 2
Assignment 3 of DA6401 JanMay2025

Author: Srikrishnan B, BT23S013

### Description
This repository contains codes written as a part of Assignment 3 of DA6401 Jan-May 2025. The codes are written for the following tasks: training and hyperparameter tuning of RNN models for transliteration from Latin text to Tamil text with and without the attention mechanism. The dataset used is `dakshina_ dataset`, which has ~81000 entries of pairs of Latin and transliterated Tamil words segregated into train, validation and test sets.  

### Requirements
The following packages are required:
- numpy   
- matplotlib
- torch
- torchvision
- pytorch_lightning
- wandb
- regex

An environment can be created using `requirements.txt`.

### Usage

The repository has notebooks to run the following tasks:
    - train and evaluate a RNN (with and without attention) from scratch with a given configuration
        - plot attention map
    - perform hyperparameter tuning using wandb sweep functionality

The implementations are organized in script files. The notebooks utilize these functions and carry out the tasks.

- Use `notebooks/train_rnn.ipynb` for training. Hyperparameters and project names are set in `src/config.py`
- Use `notebooks/sweep.ipynb` for wand sweeps. Sweep configurations, project names are set in `src/sweep_config.py`
- Run `train.py` for training the model from command line. Use `src/config.py` for setting variables and hyperparameters or pass them as commandline arguments. 
- Run `sweep.py` for running a hyperparameter sweep using wandb

### Folder organization

The codes are organized in two folders: `src`,`notebooks` and `results`. `src` contains all the source codes (.py files) and config files, and `notebooks` contains ipython notebooks that demonstrate training RNN model from scratch and performing a wandb sweep. `*.py` files in the root directory are used for training and sweep. `results` directory contain predictions on the test made by the best model from hyperparameter sweep both for models with attention and without attention.

```

│
├── README.md               # Documentation
├── requirements.txt        # Dependencies
├── train.py                # Training
├── sweep.py                # Hyperparameter sweep
├── src/                   
│   ├── config.py            # Configuration for training
│   ├── dataloader.py        # Load dataset, build vocabulary, tokenizers
│   ├── models.py            # RNN implementations are defined here: with attention and without attention, along with pytorch implementation
|   ├── sweep_config.py      # Configuration for sweep
│   ├── utils.py             # Attention heatmap
├── notebooks/              
│   ├── sweep.ipynb          # Wandb sweep
│   └── train_rnn.ipynb      # Train CNN from scratch

```

- `models.py`: The following functions are defined: Encoder and Decoder (with and without attention) are defined separately and are integrated in the lightning implementation.



### Link to wandb report: 
### Link to Github repo: [Link](https://github.com/srikrishnan-b/DL6401_A3) 
 
