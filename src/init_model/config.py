config = {
    # Training parameters
    "n_epochs": 200,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    
    # Model architecture
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 100.0,
    "input_size": 3000,
    "seq_length": 20,       
    "n_classes": 5,
    "l2_weight_decay": 1e-3,

    # Dataset parameters
    "dataset": "sleepedfx",
    "data_dir": "./data/eeg_fpz_cz",
    "n_folds": 10,
    "n_subjects": 78,
    "batch_size": 15,        

    # Data augmentation
    "augment_seq": True,
    "augment_signal_full": True,
    
    # Training utilities
    "patience": 50,        
    "checkpoint_freq": 50,   
}

predict_config = {
    "batch_size": 1,
    "seq_length": 1
}