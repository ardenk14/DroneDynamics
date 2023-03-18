import torch
from mppi import MPPI

class FlightController(object):
    """
    MPPI-based flight controller
    """

    def __init__(self, model, cost_function, num_samples=100, horizon=10):
        #self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = None #env.observation_space.shape[0]
        u_min = None #torch.from_numpy(env.action_space.low)
        u_max = None #torch.from_numpy(env.action_space.high)
        noise_sigma = None #0.5 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = self.model(state, action)
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        """
        # Convert numpy array to a tensor
        state_tensor = torch.from_numpy(state)
        # Get mppi command
        action_tensor = self.mppi.command(state_tensor)
        # Convert returned action from a tensor to a numpy array
        action = action_tensor.detach().numpy()
        return action