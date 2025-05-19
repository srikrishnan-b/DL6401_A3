import pandas as pd
import regex as re
import torch 
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

train_path = "../data/train.tsv"
valid_path = "../valid.tsv"
test_path = "../test.tsv"

# reading train valid and test data
train_df = pd.read_csv(train_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')
valid_df = pd.read_csv(valid_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')
test_df = pd.read_csv(test_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')

# removing rows with NaN
train_df = train_df[~train_df['latin'].isna()]
valid_df = valid_df[~valid_df['latin'].isna()]
test_df = test_df[~test_df['latin'].isna()]


# Tokenizer class that has vocabulary, tokenization, id_to_word and word_to_id mapping
class NativeTokenizer():
    def __init__(self, train_path, valid_path, test_path, special_tokens={'START': '<start>','END':'<end>', 'PAD':'<pad>'}):
        
        self.train_df = pd.read_csv(train_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')
        self.valid_df = pd.read_csv(valid_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')
        self.test_df = pd.read_csv(test_path, sep="\t", header=None, names=["native", "latin", 'n_annot'], encoding='utf-8')
        self.special_tokens = special_tokens
        # Build vocabulary
        self._build_vocab(add_special_tokens=True)
        
        # Id to token mapping
        self.id_to_latin = {i: char for i, char in enumerate(self.latin_vocab)}
        self.id_to_native = {i: char for i, char in enumerate(self.native_vocab)}

        self.latin_vocab_size = len(self.latin_vocab)
        self.nat_vocab_size = len(self.native_vocab)

    # Build vocabulary
    def _build_vocab(self, add_special_tokens=True):
        self.nat_set = set()
        self.latin_set = set()
        for lat, nat in zip(self.train_df['latin'], self.train_df['native']):
            nat_chars = re.findall(r'\X' , nat)
            try:
                lat_chars = list(lat)
            except:
                print(f"Invalid latin string: {lat}, skipping....")
            
            for char in nat_chars:
                self.nat_set.add(char)
            for char in lat_chars:
               self.latin_set.add(char.lower())
            
        self.nat_set = sorted(list(self.nat_set))
        self.latin_set = sorted(list(self.latin_set))
        
        if add_special_tokens:
            self.nat_set = list(self.special_tokens.values()) + self.nat_set
            self.latin_set = [self.special_tokens['PAD']] + self.latin_set   

        self.latin_vocab = {char: i for i, char in enumerate(self.latin_set)}
        self.native_vocab = {char: i for i, char in enumerate(self.nat_set)}

    def tokenize(self, text, lang='latin'):
        if type(text) != str:
            print("Invalid text:", text)
            print("Language must be a string, but got", type(text))
        if lang == 'latin':
            return [self.latin_vocab[char] for char in text]
        elif lang == 'native':
            return [self.native_vocab['<start>']] + [self.native_vocab[char] for char in re.findall('\X', text)] + [self.native_vocab['<end>']]
        else:
            raise ValueError("Language must be either 'latin' or 'native'.")
        


# Dataset class for building train, valid and test splits of the data 
class LatNatDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        latin_word = entry['latin']
        native_word = entry['native']
               
        latin_ids = self.tokenizer.tokenize(latin_word, lang='latin')
        native_ids = self.tokenizer.tokenize(native_word, lang='native')


        return (torch.tensor(latin_ids),
            torch.tensor(native_ids))

    def collate_fn(self, batch):
        x,y = zip(*batch)
        x_len = [len(seq) for seq in x]
        y_len = [len(seq) for seq in y]

        padded_x = pad_sequence(x, batch_first=True, padding_value=self.tokenizer.latin_vocab['<pad>'])
        padded_y = pad_sequence(y, batch_first=True, padding_value=self.tokenizer.native_vocab['<pad>'])
        
        x_len, perm_idx = torch.tensor(x_len).sort(0, descending=True)
        padded_x = padded_x[perm_idx]

        y_len = torch.tensor(y_len).sort(0, descending=True)
        padded_y = padded_y[perm_idx]

        return padded_x, x_len, padded_y, y_len




