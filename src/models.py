import torch
import torch.nn as nn
import numpy as np
import gc
import wandb
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from utils import save_as_html
from IPython.display import HTML

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
        self.test_preds = []
        self.test_labels = []
        self.test_inputs = []
        if layers == 1:
            print("Dropout is not applied for 1 layer")
            dropout = 0 
        self.encoder = Encoder(input_sizes[0], embedding_size, hidden_size, cell=cell, num_layers=layers, dropout=dropout, activation=activation)
        self.decoder = Decoder(input_sizes[1], embedding_size, hidden_size, cell=cell, num_layers=layers, dropout=dropout, activation=activation)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_tokens['<pad>'], reduction='sum')
        self.special_tokens = special_tokens   
        self.beam_size = beam_size 
        self.cell = cell
    
    def on_test_start(self):
        self.test_preds.clear()
        self.test_labels.clear()
        self.test_inputs.clear()


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
        predicted_words = ["".join([tokenizer.id_to_native[i] for i in pred if i not in self.special_tokens.values()])for pred in preds]
        true_words = ["".join([tokenizer.id_to_native[i] for i in target.tolist() if i not in self.special_tokens.values()]) for target in decoder_target]
        input_words = ["".join([tokenizer.id_to_latin[i] for i in input_word.tolist() if i != 0]) for input_word in input_tensor]
        self.test_preds.extend(predicted_words)
        self.test_labels.extend(true_words)
        self.test_inputs.extend(input_words)


        self.log("test loss", loss/non_pad, on_step = False, on_epoch = True)
        self.log("test accuracy", accuracy, on_step = False, on_epoch = True)

        return loss/non_pad

    def configure_optimizers(self):
        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def on_test_epoch_end(self):
        save_as_html(self.test_preds, self.test_labels, self.test_inputs, path="predictions.html")
    


class Attention(nn.Module):
 
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn_hid = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):

        txt_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        hidden = hidden.repeat(1, txt_len, 1)  
        attn_input = torch.cat((encoder_outputs, hidden), dim=2)   # input for attention layer - concat encoder output and hidden
        attn_hidden = torch.tanh(self.attn_hid(attn_input))     # apply tanh
        attn_scores = self.attn_score(attn_hidden).squeeze(2)   # attention scores for encoder output
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)
        
        context = torch.bmm(attn_weights, encoder_outputs)  # context vector as weighted sum
        
        return context, attn_weights

class AttentionDecoder(torch.nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, cell, num_layers, dropout, activation=None):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size)
        self.attention = Attention(hidden_size)
        self.cell = cell
        if cell == 'rnn':
            self.rnn = torch.nn.RNN(input_size=embedding_size + hidden_size, hidden_size=hidden_size, 
                                    batch_first=True, num_layers=num_layers, nonlinearity=activation, dropout=dropout)
        elif cell == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=embedding_size + hidden_size, hidden_size=hidden_size, 
                                     batch_first=True, num_layers=num_layers, dropout=dropout)
        elif cell == 'GRU':
            self.rnn = torch.nn.GRU(input_size=embedding_size + hidden_size, hidden_size=hidden_size, 
                                    batch_first=True, num_layers=num_layers, dropout=dropout)
        
        self.out = torch.nn.Linear(hidden_size * 2, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)  

    def forward(self, input_step, hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        
        if self.cell == 'LSTM': 
            attn_hidden = hidden[0][0:1]  
        else: 
            attn_hidden = hidden[0:1] 
        
        context, attn_weights = self.attention(attn_hidden.transpose(0, 1), encoder_outputs) # context vector and attention weights
        decoder_input = torch.cat((embedded, context), dim=2)  
        rnn_output, hidden = self.rnn(decoder_input, hidden) 
        output_context = torch.cat((rnn_output, context), dim=2)  
        output = self.out(output_context) 
        
        return output, hidden, attn_weights
    


class RNN_light_attention(pl.LightningModule):
    def __init__(self, input_sizes, embedding_size, hidden_size, cell, layers, dropout, activation, beam_size, optim, special_tokens, lr):
        super().__init__()
        self.optim = optim
        self.save_hyperparameters()
        self.beam_size = beam_size
        self.test_preds = []
        self.test_labels = []
        self.test_inputs = []
        self.attention_maps = []
        if layers == 1:
            print("Dropout is not applied for 1 layer")
            dropout = 0 
        self.encoder = Encoder(input_sizes[0], embedding_size, hidden_size, cell=cell, num_layers=layers, dropout=dropout, activation=activation)
        self.decoder = AttentionDecoder(input_sizes[1], embedding_size, hidden_size, cell=cell, num_layers=layers, dropout=dropout, activation=activation)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_tokens['<pad>'], reduction='sum')
        self.special_tokens = special_tokens   
        self.beam_size = beam_size 
        self.cell = cell

    # to track predictions from test set along with attention maps
    def on_test_start(self):
        self.test_preds.clear()
        self.test_labels.clear()
        self.test_inputs.clear()
        self.attention_maps.clear()

       
    def forward(self, input_tensor=[], input_lengths=[], decoder_input=[], decoder_hidden=[], encoder_outputs=None, encoder=False):
        if encoder:
            # Run encoder and get outputs and hidden state
            encoder_outputs, decoder_hidden = self.encoder(input_tensor, input_lengths)

            if self.cell == 'LSTM':
                h, c = decoder_hidden
                decoder_hidden = (h.contiguous(), c.contiguous())
            else:
                decoder_hidden = decoder_hidden.contiguous()
                
            # Run decoder with attention
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            
        else:
            # Just run decoder with attention using provided hidden state and encoder outputs
            if self.cell == 'LSTM':
                h, c = decoder_hidden
                decoder_hidden = (h.contiguous(), c.contiguous())
            else:
                decoder_hidden = decoder_hidden.contiguous()

            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            
        return decoder_output, decoder_hidden
    
    def training_step(self, batch, batch_idx):
        input_tensor, input_lengths, target_tensor, target_lengths = batch
        decoder_input = target_tensor[:, :-1].detach().clone()
        decoder_target = target_tensor[:, 1:].detach().clone()
        loss = 0
        # Get encoder outputs once for the whole sequence
        encoder_outputs, decoder_hidden = self.encoder(input_tensor, input_lengths)
        
        for i in range(target_tensor.shape[1]-1):
            if i == 0:
                # First step
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input[:, i].unsqueeze(1), 
                    decoder_hidden, 
                    encoder_outputs
                )
                loss += self.loss_fn(decoder_output.squeeze(1), decoder_target[:, i])
                preds = decoder_output.argmax(dim=2).cpu().numpy()
            else:
                # Rest of the steps
                decoder_output, decoder_hidden, attn = self.decoder(
                    decoder_input[:, i].unsqueeze(1), 
                    decoder_hidden, 
                    encoder_outputs
                )
                loss += self.loss_fn(decoder_output.squeeze(1), decoder_target[:, i])
                preds = np.hstack((preds, decoder_output.argmax(dim=2).cpu().numpy()))
        
        # Masking pad tokens and end tokens for accuracy calculation
        non_pad = (decoder_target[:, :-1] != self.special_tokens['<pad>']).sum()
        mask = ~torch.isin(decoder_target[:,:-1], torch.tensor(list(self.special_tokens.values()), device=input_tensor.device))

        masked_preds = torch.tensor(preds[:, :-1], device=input_tensor.device).masked_fill(~mask, self.special_tokens['<pad>'])
        masked_targets = decoder_target[:, :-1].masked_fill(~mask, self.special_tokens['<pad>'])
        exact_matches = (masked_preds == masked_targets).all(dim=1)
        accuracy = exact_matches.float().mean()
        
        self.log("train loss", loss/non_pad, on_step=False, on_epoch=True)
        self.log("train accuracy", accuracy, on_step=False, on_epoch=True)

        return loss/non_pad

    def validation_step(self, batch, batch_idx):
        input_tensor, input_lengths, target_tensor, target_lengths = batch
        decoder_target = target_tensor[:, 1:].detach().clone()
        loss = 0
        
        # Get encoder outputs once
        encoder_outputs, decoder_hidden = self.encoder(input_tensor, input_lengths)
        
        decoder_input = torch.tensor([[self.special_tokens['<start>']] * input_tensor.shape[0]], 
                                     device=input_tensor.device).reshape(-1, 1)
        
        for i in range(target_tensor.shape[1]-1):
            # Run decoder with attention
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, 
                decoder_hidden, 
                encoder_outputs
            )
            
            loss += self.loss_fn(decoder_output.squeeze(1), decoder_target[:, i])
            
            if i == 0:
                preds = decoder_output.argmax(dim=2).cpu().numpy()
            else:
                preds = np.hstack((preds, decoder_output.argmax(dim=2).cpu().numpy()))
                
            # Use predicted token as next input
            decoder_input = decoder_output.argmax(dim=2)
        
        # Masking pad tokens and end tokens for accuracy calculation
        non_pad = (decoder_target[:, :-1] != self.special_tokens['<pad>']).sum()
        mask = ~torch.isin(decoder_target[:,:-1], torch.tensor(list(self.special_tokens.values()), device=input_tensor.device))
        
        masked_preds = torch.tensor(preds[:, :-1], device=input_tensor.device).masked_fill(~mask, self.special_tokens['<pad>'])
        masked_targets = decoder_target[:, :-1].masked_fill(~mask, self.special_tokens['<pad>'])
        exact_matches = (masked_preds == masked_targets).all(dim=1)
        accuracy = exact_matches.float().mean()

        self.log("val loss", loss/non_pad, on_step=False, on_epoch=True)
        self.log("val accuracy", accuracy, on_step=False, on_epoch=True)

        return loss/non_pad

    def test_step(self, batch, batch_idx):
        input_tensor, input_lengths, target_tensor, target_lengths = batch
        decoder_target = target_tensor[:, 1:].detach().clone()
        loss = 0
        attention_map = []

        # Get encoder outputs once
        encoder_outputs, decoder_hidden = self.encoder(input_tensor, input_lengths)
        decoder_input = torch.tensor([[self.special_tokens['<start>']] * input_tensor.shape[0]], 
                                     device=input_tensor.device).reshape(-1, 1)
        
        for i in range(target_tensor.shape[1]-1):
            # Run decoder with attention
            decoder_output, decoder_hidden, attn = self.decoder(
                decoder_input, 
                decoder_hidden, 
                encoder_outputs
            )
            
            loss += self.loss_fn(decoder_output.squeeze(1), decoder_target[:, i])
            
            if i == 0:
                preds = decoder_output.argmax(dim=2).cpu().numpy()
            else:
                preds = np.hstack((preds, decoder_output.argmax(dim=2).cpu().numpy()))
                
            # Use predicted token as next input
            decoder_input = decoder_output.argmax(dim=2)
            attention_map.append(attn.squeeze(1).cpu()) ####

        
        # Masking pad tokens and end tokens for accuracy calculation
        non_pad = (decoder_target[:, :-1] != self.special_tokens['<pad>']).sum()
        mask = ~torch.isin(decoder_target[:,:-1], torch.tensor(list(self.special_tokens.values()), device=input_tensor.device))
        
        masked_preds = torch.tensor(preds[:, :-1], device=input_tensor.device).masked_fill(~mask, self.special_tokens['<pad>'])
        masked_targets = decoder_target[:, :-1].masked_fill(~mask, self.special_tokens['<pad>'])
        exact_matches = (masked_preds == masked_targets).all(dim=1)
        accuracy = exact_matches.float().mean()

        # converting tokens to words for predictions, input and targets
        predicted_words = ["".join([tokenizer.id_to_native[i] for i in pred if i not in self.special_tokens.values()])for pred in preds]
        true_words = ["".join([tokenizer.id_to_native[i] for i in target.tolist() if i not in self.special_tokens.values()]) for target in decoder_target]
        input_words = ["".join([tokenizer.id_to_latin[i] for i in input_word.tolist() if i != 0]) for input_word in input_tensor]
        self.test_preds.extend(predicted_words)
        self.test_labels.extend(true_words)
        self.test_inputs.extend(input_words)
        self.attention_maps.extend(torch.stack(attention_map, dim=1))
        self.log("test loss", loss/non_pad, on_step = False, on_epoch = True)
        self.log("test accuracy", accuracy, on_step = False, on_epoch = True)
        return loss/non_pad

    
    def on_test_epoch_end(self):
        save_as_html(self.test_preds, self.test_labels, self.test_inputs, path="predictions.html")


    def configure_optimizers(self):
        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
