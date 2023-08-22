import pytorch_lightning as pl
from models.train import train_model

def main():
    # Hyperparameters
    hyperparams = {
        'z_dim': 128,
        'lambda_gp': 10,
        'batch_size': 32,
        'epochs': 1,
        'npz_path': 'data/data.npz',
        'learning_rate': 0.0002
    }

    # Logger
    logger = pl.loggers.TensorBoardLogger('logs/')

    # Train the model
    train_model(hyperparams, logger)

if __name__ == '__main__':
    main()
