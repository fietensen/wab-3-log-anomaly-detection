from models.autolog import AutoLogAutoencoder

def train_autolog(datasets: dict, epochs: int = 100, **kwargs) -> AutoLogAutoencoder:
    model = AutoLogAutoencoder(67, datasets)
    model.train_batch(epochs=epochs, **kwargs)
    return model

if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
