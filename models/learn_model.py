import torch
import torch.nn as nn
from models import ResidualDynamicsModel, AbsoluteDynamicsModel
from custom_losses import MultiStepLoss
import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def train_step(model, train_loader, optimizer, loss_fcn) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    model.train()
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
    model.eval()
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
    print("DEVICE: ", device)    

    data_filepaths = []
    for i in ['0.15', '0.16', '0.17', '0.18', '0.19', '0.21', '0.23']:
        for j in range(19):
            data_filepaths.append("../data/Bag1_start" + i + "_" + str(j) + ".npz")
        for j in range(19):
            data_filepaths.append("../data/Bag2_start" + i + "_" + str(j) + ".npz")
    # TODO: Setup data file
    train_loader, val_loader = load_data.get_dataloader_multi_step(data_filepaths, num_steps=5) #get_dataloader('data.npz')
    #print("train loader: ", train_loader)

    # Create model
    state_dim = 12 # TODO: have dataloader function return these dimensions
    action_dim = 16
    model = ResidualDynamicsModel(state_dim, action_dim).to(device)
    #model.load_state_dict(torch.load("multistep_residual_model.pt")) # To train a pre-trained network
    #model = AbsoluteDynamicsModel(state_dim, action_dim).to(device)

    # Train forward model
    pose_lss = nn.MSELoss()
    pose_loss = MultiStepLoss(pose_lss, discount=0.95)
    train_losses, val_losses = train_model(model, train_loader, val_loader, pose_loss, num_epochs=5000, lr=0.001)

    # Save the model
    print("Saving...")
    torch.save(model.state_dict(), 'multistep_residual_model.pt')
    #torch.save(model.state_dict(), 'multistep_absolute_model.pt')
    print("Saved at multistep_residual_model.pt")

    # Plot forward only losses
    plt.plot([i for i in range(len(train_losses))], train_losses, label="Training Loss")
    plt.plot([i for i in range(len(val_losses))], val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs. Validation Loss")
    plt.show()
