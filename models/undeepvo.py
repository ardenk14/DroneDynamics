import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class UnDeepVOModel(nn.Module):
    """
    """

    def __init__(self, img_dim, action_dim, lr=1e-3):
        super().__init__()
        self.img_dim = img_dim
        self.action_dim = action_dim
        #self.loss_fcn = nn.MSELoss()

        self.tr_root =  nn.Sequential(
          nn.Conv2d(self.s)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, state):
        """
        """
        next_action = self.model(state)
        return next_action
    
    def train_step(self, train_loader) -> float:
        """
        Performs an epoch train step.
        :param model: Pytorch nn.Module
        :param train_loader: Pytorch DataLoader
        :param optimizer: Pytorch optimizer
        :return: train_loss <float> representing the average loss among the different mini-batches.
            Loss needs to be MSE loss.
        """
        train_loss = 0. 

        for batch_idx, data in enumerate(train_loader):
            self.optimizer.zero_grad()

            # TODO: extract data correctly
            state = data['state']
            action = data['true_action']

            pred_action = self.model(state)
            loss = F.mse_loss(pred_action, action) 
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        return train_loss/len(train_loader)
    
    def train_model(self, train_dataloader, num_epochs=100):
        """
        Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
        You may need to use `train_step` and `val_step`.
        :param train_dataloader: Pytorch DataLoader with the training data.
        :param num_epochs: <int> number of epochs to train the model.
        :param lr: <float> learning rate for the weight update.
        :return:
        """
        pbar = tqdm(range(num_epochs))
        train_losses = []
        for epoch_i in pbar:
            train_loss_i = self.train_step(train_dataloader)
            pbar.set_description(f'Train Loss: {train_loss_i:.4f}')
            train_losses.append(train_loss_i)
        return train_losses