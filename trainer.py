from dataloader import get_training_dataloader, get_testing_dataloader

import math
import pandas as pd
import torch
from torch.optim import Adamax
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

class Trainer:

    def __init__(self, model, num_epochs=20, batch_size=16, init_lr=0.05, device='cpu'):
        self.model = model.to(device)

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.device = device

    def shuffle_csv(path):
        dataset = pd.read_csv(path)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        dataset.to_csv(path, index=False)

    def train(self, path='estimator_weights.pkl'):
        data_path = 'Data_i11_is/dataset_i11_is.csv'
        # Trainer.shuffle_csv(data_path)
        
        training_dataloader = get_training_dataloader(data_path)
        testing_dataloader  = get_testing_dataloader(data_path)

        loss_fn = torch.nn.MSELoss()
        optimizer = Adamax(self.model.parameters(), lr = self.init_lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        min_mse_loss = 999999
        for epoch in range(self.num_epochs):
            self.model.train()

            with tqdm(training_dataloader, unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {epoch + 1}/{self.num_epochs}')
                for data in tepoch:
                    images, moisture_labels = data
                    images, moisture_labels = images.to(self.device), moisture_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    mse_loss = loss_fn(outputs, moisture_labels)
                    mse_loss.backward()
                    optimizer.step()

                    tepoch.set_postfix(
                        loss=mse_loss
                    )
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                mse_loss = 0.0
                for data in testing_dataloader:
                    images, moisture_labels = data
                    images, moisture_labels = images.to(self.device), moisture_labels.to(self.device)

                    outputs = self.model(images)
                    mse_loss += loss_fn(outputs, moisture_labels)

                print(f'Epoch {epoch + 1}: Validation Loss: {mse_loss}')

                if mse_loss < min_mse_loss:
                    min_mse_loss = mse_loss
                    torch.save(self.model.state_dict(), path)
