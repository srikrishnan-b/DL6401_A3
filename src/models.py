import torch
import numpy as np
import gc
import wandb
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Encoder
class Encoder(torch.nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, cell, num_layers, dropout, activation=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size, )
        if cell =='rnn':
            self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, nonlinearity=activation)
        elif cell == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers,dropout=dropout)
        elif cell == 'GRU':
            self.rnn = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
    
    def forward(self, seq, seq_len):
        embedding = self.embedding(input=seq)
        packed = pack_padded_sequence(input=embedding, lengths=seq_len.cpu(), batch_first=True, enforce_sorted=True)
        output, hidden = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, hidden
    

# Decoder
class Decoder(torch.nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, cell, num_layers, dropout, activation=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size)
        if cell == 'rnn':
            self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, nonlinearity=activation, dropout=dropout)
        elif cell == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        elif cell == 'GRU':
            self.rnn = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)  

    def forward(self, input_step, hidden):
        # input_step: (batch_size, 1) [a single timestep]
        embedded = self.embedding(input_step)  # (batch_size, 1, hidden_size)

        rnn_output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, 1, hidden_size)
        output = self.out(rnn_output)  # (batch_size, 1, output_size)
        return output, hidden

# Torch lightning implementation of RNN     
class RNN_light(pl.LightningModule):
    def __init__(self, input_sizes, embedding_size, hidden_size, cell, layers, dropout, activation, beam_size, optim, special_tokens, lr):
        super().__init__()
        self.optim = optim
        self.save_hyperparameters()
        self.beam_size = beam_size
        if layers == 1:
            print("Dropout is not applied for 1 layer")
            dropout = 0 
        self.encoder = Encoder(input_sizes[0], embedding_size, hidden_size, cell=cell, num_layers=layers, dropout=dropout, activation=activation)
        self.decoder = Decoder(input_sizes[1], embedding_size, hidden_size, cell=cell, num_layers=layers, dropout=dropout, activation=activation)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_tokens['<pad>'], reduction='sum')
        self.special_tokens = special_tokens   
        self.beam_size = beam_size 
        self.cell = cell
    
    def forward(self, input_tensor=[], input_lengths=[], decoder_input=[], decoder_hidden= [], encoder=False):
        if encoder:
            _, decoder_hidden = self.encoder(input_tensor, input_lengths)

            if self.cell == 'LSTM':
                h, c = decoder_hidden
                decoder_hidden =  (h.contiguous(), c.contiguous())
            else:
                decoder_hidden = decoder_hidden.contiguous()
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
        else:
            if self.cell == 'LSTM':
                h, c = decoder_hidden
                decoder_hidden =  (h.contiguous(), c.contiguous())
            else:
                decoder_hidden = decoder_hidden.contiguous()

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        return decoder_output, decoder_hidden

    
    def training_step(self, batch, batch_idx):
        input_tensor, input_lengths, target_tensor, target_lengths = batch
        decoder_input = target_tensor[:, :-1].detach().clone()
        decoder_target = target_tensor[:, 1:].detach().clone()
        loss = 0

        for i in range(target_tensor.shape[1]-1):
            if i ==0:
                # first step
                decoder_output, decoder_hidden = self(input_tensor = input_tensor, input_lengths=input_lengths, decoder_input = decoder_input[:, i].unsqueeze(1), encoder=True)
   
                loss += self.loss_fn(decoder_output.squeeze(1), decoder_target[:, i])
                preds = decoder_output.argmax(dim=2).cpu().numpy()
            else:
                # rest of the steps
                decoder_output, decoder_hidden = self(decoder_input=decoder_input[:, i].unsqueeze(1), decoder_hidden=decoder_hidden)
 
                loss += self.loss_fn(decoder_output.squeeze(1), decoder_target[:, i])
                preds = np.hstack((preds, decoder_output.argmax(dim=2).cpu().numpy()))
        
        # masking pad tokens and end tokens for accuracy calculation
        non_pad = (decoder_target[:, :-1] != self.special_tokens['<pad>']).sum()
        mask = ~torch.isin(decoder_target[:,:-1], torch.tensor(list(self.special_tokens.values()), device = input_tensor.device))
        masked_preds = torch.tensor(preds[:, :-1], device = input_tensor.device).masked_fill(~mask, self.special_tokens['<pad>'])
        masked_targets = decoder_target[:, :-1].masked_fill(~mask, self.special_tokens['<pad>'])
        exact_matches = (masked_preds == masked_targets).all(dim=1)
        accuracy = exact_matches.float().mean()
        self.log("train loss", loss/non_pad, on_step = False, on_epoch = True)
        self.log("train accuracy", accuracy, on_step = False, on_epoch = True)

        return loss/non_pad

    def validation_step(self, batch, batch_idx):
        input_tensor, input_lengths, target_tensor, target_lengths = batch
        decoder_input = target_tensor[:, :-1].detach().clone()
        decoder_target = target_tensor[:, 1:].detach().clone()
        loss = 0
        for i in range(target_tensor.shape[1]-1):
            if i ==0:
                # first step                
                decoder_input = torch.tensor([[self.special_tokens['<start>']]* input_tensor.shape[0]], device=input_tensor.device).reshape(-1, 1)                
                decoder_output, decoder_hidden = self(input_tensor = input_tensor, input_lengths=input_lengths, decoder_input = decoder_input[:,], encoder=True)
                loss += self.loss_fn(decoder_output.squeeze(1), decoder_target[:, i])
                preds = decoder_output.argmax(dim=2).cpu().numpy()
                decoder_input =decoder_output.argmax(dim=2)
            else:
                # rest of the steps
                decoder_output, decoder_hidden = self(decoder_input=decoder_input[:, ], decoder_hidden=decoder_hidden)
                decoder_input =decoder_output.argmax(dim=2)

                loss += self.loss_fn(decoder_output.squeeze(1), decoder_target[:, i])
                preds = np.hstack((preds, decoder_output.argmax(dim=2).cpu().numpy()))
        
        # masking pad tokens and end tokens for accuracy calculation
        non_pad = (decoder_target[:, :-1] != self.special_tokens['<pad>']).sum()
        mask = ~torch.isin(decoder_target[:,:-1], torch.tensor(list(self.special_tokens.values()), device = input_tensor.device))
        masked_preds = torch.tensor(preds[:, :-1], device = input_tensor.device).masked_fill(~mask, self.special_tokens['<pad>'])
        masked_targets = decoder_target[:, :-1].masked_fill(~mask, self.special_tokens['<pad>'])
        exact_matches = (masked_preds == masked_targets).all(dim=1)
        accuracy = exact_matches.float().mean()
        self.log("val loss", loss/non_pad, on_step = False, on_epoch = True)
        self.log("val accuracy", accuracy, on_step = False, on_epoch = True)

        return loss/non_pad

    def test_step(self, batch, batch_idx):
        input_tensor, input_lengths, target_tensor, target_lengths = batch
        decoder_input = target_tensor[:, :-1].detach().clone()
        decoder_target = target_tensor[:, 1:].detach().clone()
        loss = 0
        for i in range(target_tensor.shape[1]-1):
            if i ==0:
                # first step               
                decoder_input = torch.tensor([[self.special_tokens['<start>']]* input_tensor.shape[0]], device=input_tensor.device).reshape(-1, 1)
                decoder_output, decoder_hidden = self(input_tensor = input_tensor, input_lengths=input_lengths, decoder_input = decoder_input[:,], encoder=True)
        
                loss += self.loss_fn(decoder_output.squeeze(1), decoder_target[:, i])
                preds = decoder_output.argmax(dim=2).cpu().numpy()
                decoder_input =decoder_output.argmax(dim=2)
            else:
                # rest of the steps
                decoder_output, decoder_hidden = self(decoder_input=decoder_input[:, ], decoder_hidden=decoder_hidden)
                decoder_input =decoder_output.argmax(dim=2)

                loss += self.loss_fn(decoder_output.squeeze(1), decoder_target[:, i])
                preds = np.hstack((preds, decoder_output.argmax(dim=2).cpu().numpy()))
        
        # masking pad tokens and end tokens for accuracy calculation
        non_pad = (decoder_target[:, :-1] != self.special_tokens['<pad>']).sum()
        mask = ~torch.isin(decoder_target[:,:-1], torch.tensor(list(self.special_tokens.values()), device = input_tensor.device))
        masked_preds = torch.tensor(preds[:, :-1], device = input_tensor.device).masked_fill(~mask, self.special_tokens['<pad>'])
        masked_targets = decoder_target[:, :-1].masked_fill(~mask, self.special_tokens['<pad>'])
        exact_matches = (masked_preds == masked_targets).all(dim=1)
        accuracy = exact_matches.float().mean()
        self.log("test loss", loss/non_pad, on_step = False, on_epoch = True)
        self.log("test accuracy", accuracy, on_step = False, on_epoch = True)

        return loss/non_pad

    def configure_optimizers(self):
        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
