import torch
import torch.nn as nn
from models import ResidualDynamicsModel
from custom_losses import MultiStepLoss
import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_step(model, train_loader, optimizer, loss_fcn) -> float:
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
        optimizer.zero_grad()
        
        state = data['state']
        action = data['action']
        next_state = data['next_state']

        loss = loss_fcn(model, state, action, next_state)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss/len(train_loader)


def val_step(model, val_loader, loss_fcn) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0.

    for batch_idx, data in enumerate(val_loader):

        state = data['state']
        action = data['action']
        next_state = data['next_state']

        loss = loss_fcn(model, state, action, next_state)
        val_loss += loss.item()
    return val_loss/len(val_loader)

def train_model(model, train_dataloader, val_dataloader, loss_fcn, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    optimizer = torch.optim.Adam(model.parameters(), lr)

    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        train_loss_i = train_step(model, train_dataloader, optimizer, loss_fcn)
        val_loss_i = val_step(model, val_dataloader, loss_fcn)

        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
    return train_losses, val_losses


# TODO: Take command arguments to give a file to save in or read from
if __name__ == '__main__':
    # TODO: Setup data file
    trainloader = load_data.get_dataloader_multi_step('data.npz') #get_dataloader('data.npz')
    print("train loader: ", trainloader)

    # Create model
    state_dim = 12 # TODO: have dataloader function return these dimensions
    action_dim = 4
    model = ResidualDynamicsModel(state_dim, action_dim)

    # Train forward model
    pose_loss = nn.MSELoss()
    pose_loss = MultiStepLoss(pose_loss, discount=0.9)
    train_losses, val_losses = train_model(model, train_loader, val_loader, pose_loss, num_epochs=10000, lr=0.0001)

    # Save the model
    print("Saving...")
    torch.save(model.state_dict(), 'multistep_residual_model.pt')
    print("Saved at multistep_residual_model.pt")

    # Plot forward only losses
    plt.plot([i for i in range(len(train_losses))], train_losses)
    plt.plot([i for i in range(len(val_losses))], val_losses)
    plt.show()
