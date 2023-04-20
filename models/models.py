import torch
import torch.nn as nn

class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model =  nn.Sequential(
          nn.BatchNorm1d(self.state_dim + self.action_dim),
          nn.Linear(self.state_dim + self.action_dim, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, self.state_dim)
        )

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        inpt = torch.cat((state, action), dim=-1)
        next_state = self.model(inpt)
        
        return next_state


class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        """if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')"""

        self.model =  nn.Sequential(
          nn.BatchNorm1d(self.state_dim + self.action_dim),
          nn.Linear(self.state_dim + self.action_dim, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, self.state_dim)
        )#.to(self.device)

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        state = state#.to(self.device)
        action = action#.to(self.device)
        inpt = torch.cat((state, action), dim=-1)
        next_state = self.model(inpt) + state

        return next_state