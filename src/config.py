batch_size = 64
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256
cell='LSTM'
layers=3
dropout=0.35
activation='relu'
optim='adam'
lr=0.0002

model_type = "w_attn" # or "wo_attn"
epochs = 25
project_name = 'DLA3_sweeps'
name = 'best_model'