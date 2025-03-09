from models.autolog import AutoLogAutoencoder
from pathlib import Path

def train_autolog(datasets: dict, epochs: int = 50) -> AutoLogAutoencoder:
    model = AutoLogAutoencoder(67, datasets)
    model.train_batch(epochs=epochs)
    return model

if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
