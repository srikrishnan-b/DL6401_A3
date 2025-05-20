sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize',
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3, "eta": 2},
    'parameters': {
        'lr': {
            'min': 1e-4,
            'max': 1e-3
        },
        'batch_size': {
            'values': [64]
        },
        'embedding_size': {
            'values': [64, 128, 256]
        },
        'hidden_size': {
            'values': [64, 128, 256]
        },
        'cell': {
            'values': ['LSTM']
            
        },
        'activation': {'values': ['relu', 'tanh']},
        'layers': {'values': [3,4,5]},
        'optim': {'values': ['adam']},
        'dropout': {
            'min': 0.2,
            'max': 0.4
        },
        'epochs': {'values': [5]}
    
    }
}
