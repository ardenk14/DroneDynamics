import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

def get_dataloader_single_step(data_fp, batch_size=500):
    """
    """
    d_set = SingleStepDynamicsDataset(data_fp)

    train, val = random_split(d_set, [0.8, 0.2])
    
    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)

    return train_loader, val_loader

def get_dataloader_multi_step(data_fp, batch_size=500):
    """
    """
    d_set = MultiStepDynamicsDataset(data_fp)

    train, val = random_split(d_set, [0.8, 0.2])
    
    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)

    return train_loader, val_loader

# TODO: Fill in this class to read from our data file and create the dataset
class SingleStepDynamicsDataset(Dataset):
    """
    """

    def __init__(self, data_fp):        
        self.data = None #np.load(data_fp)['data']
        self.trajectory_length = len(self.data)

    def __len__(self):
        return len(self.data) * len(self.data[0]) #* self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None,
        }

        # Get current item
        trial = item // self.trajectory_length
        index = item % self.trajectory_length

        sample = {
            'state': None,
            'action': None,
            'next_state': None            
        }

        return sample
    
class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None
        }
        # --- Your code here
        trial = item // self.trajectory_length
        index = item % self.trajectory_length
        sample = {
            'state': self.data[trial]['states'][index],
            'action': self.data[trial]['actions'][index:index+self.num_steps],
            'next_state': self.data[trial]['states'][index+1:index+1+self.num_steps]
        }


        # ---