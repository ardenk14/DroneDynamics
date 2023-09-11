import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pdb

def get_dataloader_drone_multi_step(data_filepath, batch_size=500, train_test=[0.8,0.2], chunk_size=100, num_steps=4):
    """
    """
    d_set = DroneMultiStepDynamicsDataset(data_filepath, chunk_size, num_steps=num_steps)

    train, val = random_split(d_set, train_test)
    
    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)

    return train_loader, val_loader


class DroneMultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data, modified to read from the drone dataset csv file

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, dataset_filenames, trajectory_chunk_length, num_steps=4):
        """
        :param dataset_filenames: <list> of <str> containing the path to the dataset files.
        :param num_steps: <int> number of steps to predict in the future.
        :param trajectory_chunk_length: <int> Each dataset file is chunked into trajectories of fixed length for easier loading
        """

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data = [] # List of dictionaries containing the state-action trajectories.
                        #     Each trajectory dictionary should have the following structure:
                        #         {'states': states,
                        #         'actions': actions}
                        #     where
                        #         * states is a numpy array of shape (trajectory_chunk_length, state_size) containing the states [x_0, ...., x_T]
                        #         * actions is a numpy array of shape (trajectory_chunk_length-1, actions_size) containing the actions [u_0, ...., u_{T-1}]
                        #           states and actions both contain np.float32.
        
        #pdb.set_trace()
        self.trajectory_length = trajectory_chunk_length - num_steps# Length of each trajectory that you can access
        #print("traject length: ", self.trajectory_length)
        self.num_steps = num_steps # Number of steps BEYOND the current that you want to predict
        self.n_trajectories = len(dataset_filenames)

        for filename in dataset_filenames:
            full_csv = np.loadtxt(filename, delimiter=',', dtype=np.float32, skiprows=1)
            i = 0
            if len(full_csv) < trajectory_chunk_length: 
                print(f"WARNING: length ({len(full_csv)}) of {filename} is less than trajectory chunk length ({trajectory_chunk_length})")
            # Split the data into trajectories of length trajectory_chunk_length
            while i <= len(full_csv) - trajectory_chunk_length:
                commands = torch.from_numpy(full_csv[i:i+trajectory_chunk_length-1, 1:5]) #(trajectory_chunk_length, 4)
                indices = [5, 6, 7, 14, 15, 16, 17, 18, 19, 21, 22, 23]
                states = torch.from_numpy(full_csv[i:i+trajectory_chunk_length, indices])#5:20])
                # 15-dimensional state space: position (x,y,z), angular position (first two columns of 3x3 rotation matrix), linear velocity (x,y,z), angular velocity (roll, pitch, yaw)                                      
                
                self.data.append({'states': states.to(self.device), #position (x,y,z)
                                  'actions': commands.to(self.device)})
                i += trajectory_chunk_length
            print(f"{filename} of length {len(full_csv)} split into {i//trajectory_chunk_length} chunks")

        print("Number of chunks in self.data: ", len(self.data))
        
        self.action_dim = self.data[0]['actions'].shape[1]
        self.state_dim = self.data[0]['states'].shape[1]


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

        trial = item // self.trajectory_length
        index = item % self.trajectory_length
        sample = {
            'state': self.data[trial]['states'][index],
            'action': self.data[trial]['actions'][index:index+self.num_steps],
            'next_state': self.data[trial]['states'][index+1:index+1+self.num_steps]
        }
        #print("NEXT STATE: ", sample['next_state'].shape)
        return sample
        # ---

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

    def __init__(self, file_paths, num_steps=4):
        #self.data = collected_data
        #self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        #self.num_steps = num_steps

        self.file_paths = file_paths
        self.num_steps = num_steps
        self.data = []
        for file_path in file_paths:
            try:
                npz_data = np.load(file_path)
            except:
                continue
            print("KEYS: ", list(npz_data.keys()))
            states = npz_data['data'][:, :12]
            actions = npz_data['data'][:, 12:]
            traj_len = states.shape[0]
            # Combine states and actions and append to the dataset
            for i in range(traj_len - num_steps):
                state_seq = states[i:i+num_steps+1]
                action_seq = actions[i:i+num_steps]
                self.data.append((state_seq[0], action_seq, state_seq[1:]))

    def __len__(self):
        return len(self.data)
        #return len(self.data) * (self.trajectory_length)

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
        """trial = item // self.trajectory_length
        index = item % self.trajectory_length
        sample = {
            'state': self.data[trial]['states'][index],
            'action': self.data[trial]['actions'][index:index+self.num_steps],
            'next_state': self.data[trial]['states'][index+1:index+1+self.num_steps]
        }"""
        state_seq, action_seq, next_state_seq = self.data[item]
        # Convert numpy arrays to PyTorch tensors
        state_seq = torch.tensor(state_seq, dtype=torch.float32)
        action_seq = torch.tensor(action_seq, dtype=torch.float32)
        next_state_seq = torch.tensor(next_state_seq, dtype=torch.float32)
        sample = {
            'state': state_seq,
            'action': action_seq,
            'next_state': next_state_seq
        }
        #return state_seq, action_seq, next_state_seq
        return sample
    
    
class CustomDataset(Dataset):
    def __init__(self, file_paths, num_steps=4):
        self.file_paths = file_paths
        self.num_steps = num_steps
        self.data = []
        for file_path in file_paths:
            npz_data = np.load(file_path)
            states = npz_data['states']
            actions = npz_data['actions']
            traj_len = states.shape[0]
            # Combine states and actions and append to the dataset
            for i in range(traj_len - num_steps):
                state_seq = states[i:i+num_steps+1]
                action_seq = actions[i:i+num_steps]
                self.data.append((state_seq[:-1], action_seq, state_seq[1:]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        state_seq, action_seq, next_state_seq = self.data[index]
        # Convert numpy arrays to PyTorch tensors
        state_seq = torch.tensor(state_seq, dtype=torch.float32)
        action_seq = torch.tensor(action_seq, dtype=torch.float32)
        next_state_seq = torch.tensor(next_state_seq, dtype=torch.float32)
        return state_seq, action_seq, next_state_seq

def get_dataloader_single_step(data_fp, batch_size=500):
    """
    """
    d_set = SingleStepDynamicsDataset(data_fp)

    train, val = random_split(d_set, [0.8, 0.2])
    
    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)

    return train_loader, val_loader

def get_dataloader_multi_step(data_fp, batch_size=2048, num_steps=4):
    """
    """
    d_set = MultiStepDynamicsDataset(data_fp, num_steps=num_steps)

    train, val = random_split(d_set, [0.95, 0.05])
    
    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)

    return train_loader, val_loader


if __name__ == "__main__":
    data_filepaths = [
        r"../data/Bag1_0.npz",
        r"../data/Bag1_1.npz",
        r"../data/Bag1_2.npz",
        r"../data/Bag1_3.npz",
        r"../data/Bag1_4.npz",
        r"../data/Bag1_5.npz",
        r"../data/Bag1_6.npz",
        r"../data/Bag1_7.npz",
        r"../data/Bag1_8.npz",
        r"../data/Bag1_9.npz",
        r"../data/Bag1_10.npz",
        r"../data/Bag1_11.npz",
        r"../data/Bag1_12.npz",
        r"../data/Bag1_13.npz",
        r"../data/Bag1_14.npz",
        r"../data/Bag1_15.npz",
        r"../data/Bag1_16.npz",
        r"../data/Bag1_17.npz",
        r"../data/Bag1_18.npz",
    ]
    data_filepaths = [
        #r"../data/Bag1_0.npz",
        #r"../data/Bag1_1.npz",
        #r"../data/Bag1_2.npz",
        #r"../data/Bag1_3.npz",
        #r"../data/Bag1_4.npz",
        #r"../data/Bag1_5.npz",
        #r"../data/Bag1_6.npz",
        #r"../data/Bag1_7.npz",
        #r"../data/Bag1_8.npz",
        #r"../data/Bag1_9.npz",
        #r"../data/Bag1_10.npz",
        #r"../data/Bag1_11.npz",
        #r"../data/Bag1_12.npz",
        #3r"../data/Bag1_13.npz",
        #r"../data/Bag1_14.npz",
        #r"../data/Bag1_15.npz",
        #r"../data/Bag1_16.npz",
        #r"../data/Bag1_17.npz",
        #r"../data/Bag1_18.npz",
        r"../data/Bag1_start0.15_0.npz",
        r"../data/Bag1_start0.15_1.npz",
        r"../data/Bag1_start0.15_2.npz",
        r"../data/Bag1_start0.15_3.npz",
        r"../data/Bag1_start0.15_4.npz",
        r"../data/Bag1_start0.15_5.npz",
        r"../data/Bag1_start0.15_6.npz",
        r"../data/Bag1_start0.15_7.npz",
        r"../data/Bag1_start0.15_8.npz",
        r"../data/Bag1_start0.15_9.npz",
        r"../data/Bag1_start0.15_10.npz",
        r"../data/Bag1_start0.15_11.npz",
        r"../data/Bag1_start0.15_12.npz",
        r"../data/Bag1_start0.15_13.npz",
        r"../data/Bag1_start0.15_14.npz",
        r"../data/Bag1_start0.15_15.npz",
        r"../data/Bag1_start0.15_16.npz",
        r"../data/Bag1_start0.15_17.npz",
        r"../data/Bag1_start0.15_18.npz",
        #r"../data/Bag1_start0.2_0.npz",
        #r"../data/Bag1_start0.2_1.npz",
        #r"../data/Bag1_start0.2_2.npz",
        #r"../data/Bag1_start0.2_3.npz",
        #r"../data/Bag1_start0.2_4.npz",
        #r"../data/Bag1_start0.2_5.npz",
        #r"../data/Bag1_start0.2_6.npz",
        #r"../data/Bag1_start0.2_7.npz",
        #r"../data/Bag1_start0.2_8.npz",
        #r"../data/Bag1_start0.2_9.npz",
        #r"../data/Bag1_start0.2_10.npz",
        #r"../data/Bag1_start0.2_11.npz",
        #r"../data/Bag1_start0.2_12.npz",
        #r"../data/Bag1_start0.2_13.npz",
        #r"../data/Bag1_start0.2_14.npz",
        #r"../data/Bag1_start0.2_15.npz",
        #r"../data/Bag1_start0.2_16.npz",
        #r"../data/Bag1_start0.2_17.npz",
        #r"../data/Bag1_start0.2_18.npz"
    ]
    dataset = MultiStepDynamicsDataset(data_filepaths, num_steps=4)
    print(f"dataset length: {len(dataset)}")

    # get first sample and unpack
    first_data = dataset[0]
    #print("dataset action_dim: ", dataset.action_dim)
    #print("dataset state_dim: ", dataset.state_dim)
    print("first state: ", first_data["state"])
    print("first action trajectory: ", first_data["action"])
    print("first next_state trajectory", first_data['next_state'])

    train, val = get_dataloader_multi_step(data_filepaths)
    for batch_idx, data in enumerate(train):
        print("BATCH ID: ", batch_idx)
        print(data['state'].shape)
        print(data['action'].shape)
        print(data['next_state'].shape)
    print("END")



    """data_filepaths = [
        r"../data/50hz_processed_tags2_right_wall1681856256.2356164_1681856260.4683304.csv",
        r"../data/50hz_processed_tags2_right_wall1681856259.673429_1681856345.057893.csv",
        r"../data/50hz_processed_tags2_right_wall1681856264.1351044_1681856271.0596986.csv",
        r"../data/50hz_processed_tags2_right_wall1681856362.371829_1681856418.62231.csv",
        r"../data/50hz_processed_tags2_right_wall1681856454.295917_1681856489.001135.csv",
        r"../data/50hz_processed_tags2_right_wall1681856531.591236_1681856538.807909.csv",
        r"../data/50hz_processed_tags3_right_wall1681856774.1_1681856865.920745.csv",
        r"../data/50hz_processed_tags3_right_wall1681856871.257578_1681856949.932647.csv",
        r"../data/50hz_processed_tags3_right_wall1681856954.754664_1681857018.73437.csv"
    ]
    chunk_size = 500
    dataset = DroneMultiStepDynamicsDataset(data_filepaths, chunk_size)
    print(f"dataset length: {len(dataset)}")
    
    # get first sample and unpack
    first_data = dataset[0]
    print("dataset action_dim: ", dataset.action_dim)
    print("dataset state_dim: ", dataset.state_dim)
    print("first state: ", first_data["state"])
    print("first action trajectory: ", first_data["action"])
    print("first next_state trajectory", first_data['next_state'])

    val_size = int(len(dataset)*0.2)
    train_size = len(dataset)- val_size

    train, val = get_dataloader_drone_multi_step(data_filepaths, batch_size=16, train_test=[train_size,val_size], chunk_size=chunk_size)
    for batch_idx, data in enumerate(train):
        print("BATCH ID: ", batch_idx)
        print(data['state'].shape)
        print(data['action'].shape)
        print(data['next_state'].shape)
    print("END")"""
